"""Kiểm T14 bằng packed tổng hợp và Wav2Vec2 thu nhỏ, không mạng/GPU."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from scripts import benchmark_xlsr_cuda as benchmark
from scripts.smoke_test_xlsr import build_tiny_classifier
from tests.test_b0_v2_runner import _write_artifact


class BenchmarkContractTests(unittest.TestCase):
    def test_grid_and_fixed_measurement(self) -> None:
        grid = benchmark.configuration_grid()
        self.assertEqual([(c.unfrozen_layers, c.batch_size) for c in grid],
                         [(n, b) for n in (0, 1, 2, 4, 8, 12) for b in (8, 16)])
        self.assertEqual([c.mode for c in grid[:3]], ['X0', 'X0', 'X1'])
        self.assertEqual(benchmark.Measurement(), benchmark.Measurement(5, 30, 512, 8))

    def test_projection_uses_ceiling_and_actual_dev_size(self) -> None:
        for batch, steps in ((8, 1986), (16, 993)):
            cost = benchmark.project_cost(batch, 2.0, 0.1)
            self.assertEqual(cost['train_hours_per_epoch'], steps * 2.0 / 3600)
            self.assertEqual(cost['dev_full_hours_per_pass'], 38777 * 0.1 / 3600)
        for values in ((0, 1., 1.), (8, float('nan'), 1.), (8, 1., -1.)):
            with self.assertRaises(ValueError):
                benchmark.project_cost(*values)

    def test_dev_selection_spans_full_manifest_without_duplicates(self) -> None:
        indices = benchmark.evenly_spaced_indices(38777, 512)
        self.assertEqual((len(indices), len(set(indices)), indices[0], indices[-1]), (512, 512, 0, 38776))
        with self.assertRaises(ValueError):
            benchmark.evenly_spaced_indices(3, 4)

    def test_oom_is_recorded_and_next_configuration_runs_after_cache_cleanup(self) -> None:
        configs = [benchmark.Configuration(0, 16), benchmark.Configuration(1, 8)]
        measure = Mock(side_effect=[torch.cuda.OutOfMemoryError('OOM giả lập'), {'status': 'ok'}])
        saved: list[int] = []
        with patch.object(benchmark, 'peak_memory', return_value={}), patch('torch.cuda.empty_cache') as clear:
            results, blocked = benchmark.run_grid(configs, measure, device=torch.device('cuda:0'),
                                                  on_result=lambda rows: saved.append(len(rows)))
        self.assertFalse(blocked)
        self.assertEqual([r['status'] for r in results], ['oom', 'ok'])
        self.assertEqual(measure.call_count, 2)
        self.assertEqual(clear.call_count, 2)
        self.assertEqual(saved, [1, 2])

    def test_x0_batch8_oom_stops_before_remaining_grid(self) -> None:
        measure = Mock(side_effect=torch.cuda.OutOfMemoryError('OOM giả lập'))
        results, blocked = benchmark.run_grid(benchmark.configuration_grid(), measure, device=torch.device('cpu'))
        self.assertTrue(blocked)
        self.assertEqual(len(results), 1)
        self.assertEqual(measure.call_count, 1)

    def test_non_oom_error_is_not_swallowed(self) -> None:
        measure = Mock(side_effect=RuntimeError('Loss train không hữu hạn.'))
        with self.assertRaisesRegex(RuntimeError, 'không hữu hạn'):
            benchmark.run_grid([benchmark.Configuration(1, 8)], measure, device=torch.device('cpu'))

    def test_weight_gate_and_checksum_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
                (root/name).write_bytes(name.encode())
            candidate = {'source': {'weights_downloaded': False, 'weights_sha256': None}, 'published_config': {}}
            with self.assertRaisesRegex(ValueError, 'G1'):
                benchmark.check_weights(root, candidate)
            candidate['source'].update(weights_downloaded=True, weights_sha256=benchmark.sha256(root/'model.safetensors'),
                                       model_id='fixture', revision='a'*40)
            candidate['published_config'] = {'config_sha256': benchmark.sha256(root/'config.json'),
                                             'preprocessor_config_sha256': benchmark.sha256(root/'preprocessor_config.json')}
            result = benchmark.check_weights(root, candidate)
            self.assertEqual(result['weights_sha256'], {'model.safetensors': benchmark.sha256(root/'model.safetensors')})
            (root/'config.json').write_bytes(b'khac')
            with self.assertRaisesRegex(ValueError, 'Checksum config'):
                benchmark.check_weights(root, candidate)
            (root/'config.json').write_bytes(b'config.json')
            (root/'model.safetensors').write_bytes(b'khac')
            with self.assertRaisesRegex(ValueError, 'Checksum trọng số'):
                benchmark.check_weights(root, candidate)

    def test_missing_transformers_writes_failure_without_loading_data(self) -> None:
        import builtins
        original_import = builtins.__import__
        def missing(name: str, *args: object, **kwargs: object) -> object:
            if name == 'transformers':
                raise ImportError('Giả lập thiếu transformers')
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'failed.json'
            with patch('builtins.__import__', side_effect=missing), patch.object(benchmark, 'open_datasets') as opened:
                rc = benchmark.main(['--report-json', str(output), '--report-md', str(output.with_suffix('.md'))])
            self.assertEqual(rc, 1)
            opened.assert_not_called()
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'failed')
            self.assertIsNone(report['environment']['transformers'])
            self.assertIn('Không import', report['error'])


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'Cần transformers cho backbone CPU thu nhỏ.')
class BenchmarkPackedCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _write_artifact(self.root/'train', split='closed_train', rows=8, manifest_sha256='a'*64, offset=1)
        _write_artifact(self.root/'dev', split='closed_dev', rows=6, manifest_sha256='b'*64, offset=2)
        self.config = json.loads((benchmark.ROOT/'configs/xlsr_smoke.json').read_text(encoding='utf-8'))

    def test_x0_x1_training_eval_normalization_and_frozen_checks_on_packed(self) -> None:
        train, dev = benchmark.open_datasets(self.root/'train', self.root/'dev', locked=False)
        self.addCleanup(train.close)
        self.addCleanup(dev.close)
        with patch.object(benchmark, 'apply_xlsr_waveform_normalization', wraps=benchmark.apply_xlsr_waveform_normalization) as normalize:
            for layers in (0, 1):
                with self.subTest(layers=layers):
                    result = benchmark.measure_configuration(benchmark.Configuration(layers, 2),
                        factory=lambda n: build_tiny_classifier(self.config, n), train=train, dev=dev,
                        device=torch.device('cpu'), measurement=benchmark.Measurement(1, 2, 4, 2))
                    self.assertEqual(result['status'], 'ok')
                    self.assertEqual(len(result['train_step_seconds']), 2)
                    self.assertEqual(len(result['technical_train_losses']), 3)
                    self.assertTrue(result['checks']['all_train_and_dev_losses_finite'])
                    self.assertTrue(result['checks']['frozen_parameters_unchanged'])
                    self.assertTrue(result['checks']['classifier_updated'])
                    self.assertTrue(all(result['checks']['opened_layers_updated'].values()))
                    self.assertEqual(result['selected_dev_samples'], 4)
                    self.assertEqual(result['dev_indices'], [0, 1, 3, 5])
                    self.assertEqual(result['peak_vram'], {'allocated_bytes': 0, 'reserved_bytes': 0})
            self.assertEqual(normalize.call_count, 10)
            self.assertTrue(all(call.args[1] == benchmark.NORMALIZATION for call in normalize.call_args_list))

    def test_freeze_check_detects_unintended_parameter_change(self) -> None:
        model = build_tiny_classifier(self.config, 0)
        before = benchmark.parameter_hashes(model)
        with torch.no_grad():
            next(model.backbone.parameters()).add_(1)
            next(model.classifier.parameters()).add_(1)
        with self.assertRaisesRegex(RuntimeError, 'đóng băng'):
            benchmark.verify_updates(model, before)

    def test_technical_cli_writes_separate_cpu_report(self) -> None:
        output = self.root/'cpu.json'
        rc = benchmark.main(['--technical-smoke', '--train-artifact', str(self.root/'train'),
            '--dev-artifact', str(self.root/'dev'), '--report-json', str(output), '--report-md', str(output.with_suffix('.md'))])
        self.assertEqual(rc, 0)
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertTrue(report['technical_smoke'])
        self.assertEqual(report['environment']['device'], 'cpu')
        self.assertEqual(report['environment']['cuda_device_count_used'], 0)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(len(report['results']), 2)

    def test_foreign_transformers_version_stops_before_loading_data(self) -> None:
        import transformers
        output = self.root/'version.json'
        with patch.object(transformers, '__version__', '0.0'), patch.object(benchmark, 'open_datasets') as opened:
            rc = benchmark.main(['--report-json', str(output), '--report-md', str(output.with_suffix('.md'))])
        self.assertEqual(rc, 1)
        opened.assert_not_called()
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(report['environment']['transformers'], '0.0')
        self.assertIn('4.57.6', report['error'])

    def test_wrong_split_and_real_artifact_checksum_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'split'):
            benchmark.open_datasets(self.root/'dev', self.root/'train', locked=False)
        with self.assertRaisesRegex(ValueError, 'đã khóa'):
            benchmark.open_datasets(self.root/'train', self.root/'dev', locked=True)


if __name__ == '__main__':
    unittest.main()
