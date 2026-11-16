"""T29/D035: bộ lọc T19b, identity, exact resume CPU và profile X1R; không mạng."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from scripts import kaggle_workflow as workflow
from scripts import train_xlsr as runner
from scripts.train_b0 import _set_seed
from src.data import lowpass
from src.data.amplitude import apply_amplitude_policy
from src.data.packed import PackedManifestDataset
from src.training.full_training import GenerationStore
from tests import test_xlsr_runner as runner_tests
from tests.test_b0_v2_runner import _write_artifact
from tests.test_kaggle_workflow import config_payload

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'
X1R = 'X1-L8-lr3e-5-lp7k'
# Chỉ hai khóa provenance này được loại; mọi khóa còn lại phải trùng byte.
IDENTITY_EXCLUDED_KEYS = ('source_files', 'source_tree_sha256')


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')


def bootstrap() -> Any:
    spec = importlib.util.spec_from_file_location('bootstrap_t29', ROOT / '.kaggle/xlsr-d028/bootstrap.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LowpassTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_coefficient_hash_and_load_guard(self) -> None:
        actual = hashlib.sha256(lowpass.fir_kernel().numpy().tobytes()).hexdigest()
        self.assertEqual(actual, lowpass.LOWPASS_7K_COEFFICIENT_SHA256)
        self.assertEqual(actual, '973b82a36db8dab4f802de2102c5feed2bfb8cf6bb5846472c07d9984ab4c977')
        with patch.object(lowpass, 'LOWPASS_7K_COEFFICIENT_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'SHA hệ số'):
                lowpass.verify_coefficients()

    def test_parameters_match_t19b_fixture(self) -> None:
        reference = json.loads((FIXTURES / 'lowpass_7k_t19b.json').read_text(encoding='utf-8'))
        self.assertEqual(lowpass.filter_parameters(), reference['filter_parameters'])

    def test_parameters_match_dev_report_when_present(self) -> None:
        reference = ROOT.parent / 'DoAnTTNT/reports/shortcut_probe_dev_x1_final.json'
        if not reference.is_file():
            self.skipTest('Không có report T22 của main để đối chiếu tham số T19b.')
        report = json.loads(reference.read_text(encoding='utf-8'))
        self.assertEqual(lowpass.filter_parameters(), report['filters']['lowpass_7k'])

    def test_three_synthetic_waveforms_are_bit_exact_t19b(self) -> None:
        metadata = json.loads((FIXTURES / 'lowpass_7k_t19b.json').read_text(encoding='utf-8'))
        path = FIXTURES / 'lowpass_7k_t19b.npz'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata['fixture_sha256'])
        with np.load(path, allow_pickle=False) as fixture:
            self.assertEqual(hashlib.sha256(fixture['inputs'].tobytes()).hexdigest(), metadata['inputs_sha256'])
            self.assertEqual(hashlib.sha256(fixture['outputs'].tobytes()).hexdigest(), metadata['outputs_sha256'])
            for waveform, expected in zip(fixture['inputs'], fixture['outputs'], strict=True):
                with self.subTest(input_sha256=hashlib.sha256(waveform.tobytes()).hexdigest()):
                    original = torch.from_numpy(waveform.copy())
                    value = lowpass.apply_condition(original)
                    self.assertTrue(torch.equal(value, torch.from_numpy(expected)))
                    self.assertEqual(value.shape, original.shape)
                    self.assertEqual(value.dtype, original.dtype)
                    self.assertTrue(torch.equal(original, torch.from_numpy(waveform)))
                    self.assertAlmostEqual(float(value.abs().max()), .95, places=6)

    def test_repeak_matches_amplitude_policy_including_silence(self) -> None:
        for waveform in (torch.zeros(64000), torch.full((64000,), 1e-6), torch.ones(64000)):
            expected = apply_amplitude_policy(lowpass.fir_filter(waveform), 'peak').waveform
            self.assertTrue(torch.equal(lowpass.apply_condition(waveform), expected))

    def test_unregistered_conditions_and_invalid_inputs_rejected(self) -> None:
        for condition in ('original', 'lowpass_4k', 'roundtrip_40k', 'unknown'):
            with self.subTest(condition=condition):
                with self.assertRaisesRegex(ValueError, 'Chỉ chấp nhận'):
                    lowpass.filter_parameters(condition)
                with self.assertRaises(ValueError):
                    lowpass.apply_condition(torch.zeros(64000), condition)
        with self.assertRaises(ValueError):
            lowpass.fir_kernel(4000)
        for waveform in (torch.zeros(16), torch.zeros(64000, dtype=torch.float64)):
            with self.assertRaisesRegex(ValueError, 'float32/64000'):
                lowpass.apply_condition(waveform)


class CaptureIdentity(Exception):
    pass


class X1RRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = json.loads((ROOT / 'configs/xlsr_training.json').read_text(encoding='utf-8'))
        self.train, self.dev = self.root / 'train', self.root / 'dev'
        _write_artifact(self.train, split='closed_train', rows=4, manifest_sha256='a' * 64, offset=3)
        _write_artifact(self.dev, split='closed_dev', rows=4, manifest_sha256='b' * 64, offset=6)
        self.fixture_plan = copy.deepcopy(self.plan)
        self.fixture_plan['implementation']['ready_for_training'] = True
        for prefix, path in (('train', self.train), ('development', self.dev)):
            metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
            self.fixture_plan['data'][prefix + '_samples'] = 4
            self.fixture_plan['data'][prefix + '_metadata_sha256'] = runner.sha256(path / 'metadata.json')
            self.fixture_plan['data'][prefix + '_manifest_sha256'] = metadata['source_manifest_sha256']
            self.fixture_plan['data'][prefix + '_artifact_waveforms_sha256'] = metadata['waveforms_sha256']
        self.plan_path = self.root / 'plan.json'
        self.plan_path.write_text(json.dumps(self.fixture_plan), encoding='utf-8')

    def tearDown(self) -> None:
        self.temp.cleanup()

    def backbone(self, *args: Any, **kwargs: Any) -> Wav2Vec2Model:
        _set_seed(2026)
        config = json.loads((ROOT / 'configs/xlsr_smoke.json').read_text(encoding='utf-8'))['backbone']
        config.update(num_hidden_layers=9, layerdrop=.1, apply_spec_augment=True)
        return Wav2Vec2Model(Wav2Vec2Config(**config))

    def invoke(self, run_id: str, name: str, callback: Any, extra: list[str] | None = None) -> int:
        configuration = runner.validate_plan(self.plan, run_id, 2026)
        command = ['--plan', str(self.plan_path), '--run-id', run_id, '--training-seed', '2026',
            '--device', 'cpu', '--pretrained-dir', str(self.root / 'no-real-weights'),
            '--train-artifact', str(self.train), '--dev-artifact', str(self.dev),
            '--run-root', str(self.root / name), '--report-json', str(self.root / (name + '.json'))]
        with patch.object(runner, 'validate_plan', return_value=configuration), \
             patch.object(runner, 'verify_pretrained', return_value={'fixture': 'random'}), \
             patch.object(runner, 'load_xlsr_backbone', side_effect=self.backbone), \
             patch.object(runner, 'verify_packed_artifact'), \
             patch.object(runner, '_sample_selection_sha256', return_value=self.plan['data']['sample_selection_sha256']), \
             patch.object(runner, 'run_training', side_effect=callback):
            return runner.main(command + (extra or []))

    def capture(self, run_id: str) -> dict[str, Any]:
        identities = []
        def observe(**kwargs: Any) -> dict[str, Any]:
            identities.append(copy.deepcopy(kwargs['identity']))
            raise CaptureIdentity
        with self.assertRaises(CaptureIdentity):
            self.invoke(run_id, run_id, observe)
        return identities[0]

    def test_validate_variant_uses_exact_base_and_does_not_change_grid_or_plan(self) -> None:
        before = canonical(self.plan)
        self.assertEqual(runner.validate_plan(self.plan, X1R, 2026), runner.validate_plan(self.plan, 'X1-L8-lr3e-5', 2026))
        self.assertEqual(canonical(self.plan), before)
        self.assertEqual(runner.GRID, self.plan['grid'])
        with self.assertRaises(ValueError):
            runner.validate_plan(self.plan, 'unknown', 2026)
        with self.assertRaises(ValueError):
            runner.validate_plan(self.plan, X1R, 2029)

    def test_old_identities_only_change_the_two_authorized_source_keys(self) -> None:
        self.assertEqual(IDENTITY_EXCLUDED_KEYS, ('source_files', 'source_tree_sha256'))
        for run_id in ('X0', 'X1-L8-lr3e-5'):
            with self.subTest(run_id=run_id):
                identity = self.capture(run_id)
                baseline = json.loads((FIXTURES / f't29_identity_{run_id}_seed2026.json').read_text(encoding='utf-8'))
                differences = sorted(key for key in set(identity) | set(baseline) if identity.get(key) != baseline.get(key))
                self.assertEqual(differences, sorted(IDENTITY_EXCLUDED_KEYS))
                stripped = lambda value: {key: item for key, item in value.items() if key not in IDENTITY_EXCLUDED_KEYS}
                self.assertEqual(canonical(stripped(identity)), canonical(stripped(baseline)))
                files = [{'path': path.relative_to(ROOT).as_posix(), 'sha256': runner.sha256(path)} for path in workflow.source_files()]
                self.assertEqual(identity['source_files'], files)
                self.assertIn('src/data/lowpass.py', [row['path'] for row in files])
                self.assertEqual(identity['source_tree_sha256'], hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
                self.assertNotIn('input_condition', identity)

    def test_variant_identity_registers_exact_filter_implementation_and_base(self) -> None:
        identity = self.capture(X1R)
        self.assertEqual(identity['run_id'], X1R)
        self.assertEqual(identity['run'], runner.validate_plan(self.plan, 'X1-L8-lr3e-5', 2026))
        self.assertEqual(identity['input_condition'], {'decision': 'D035', 'condition': 'lowpass_7k',
            'base_run_id': 'X1-L8-lr3e-5', 'filter': lowpass.filter_parameters(),
            'implementation_sha256': runner.sha256(ROOT / 'src/data/lowpass.py')})

    def test_dataset_wrapper_preserves_keys_crop_stream_and_normalization_order(self) -> None:
        for training, artifact in ((True, self.train), (False, self.dev)):
            dataset = PackedManifestDataset(artifact, training=training, seed=2026, amplitude_policy='peak')
            wrapped = runner.InputConditionDataset(dataset, 'lowpass_7k')
            try:
                wrapped.set_epoch(2)
                self.assertEqual(dataset.epoch, 2)
                before, after = dataset[0], wrapped[0]
                self.assertEqual(before.keys(), after.keys())
                for key in before.keys() - {'waveform'}:
                    self.assertEqual(before[key], after[key])
                expected = lowpass.apply_condition(before['waveform'])
                self.assertTrue(torch.equal(after['waveform'], expected))
                normalized = runner.apply_xlsr_waveform_normalization(after['waveform'].unsqueeze(0), runner.NORMALIZATION)
                self.assertTrue(torch.equal(normalized, runner.apply_xlsr_waveform_normalization(expected.unsqueeze(0), runner.NORMALIZATION)))
            finally:
                wrapped.close()

    def test_grid_training_and_evaluation_do_not_call_filter(self) -> None:
        real_run = runner.run_training
        def shortened(**kwargs: Any) -> dict[str, Any]:
            kwargs.update(max_epochs=1, checkpoint_interval=1)
            return real_run(**kwargs)
        with patch.object(lowpass, 'apply_condition', side_effect=AssertionError('Run GRID không được gọi bộ lọc.')) as filtered:
            self.assertEqual(self.invoke('X0', 'grid', shortened), 0)
        filtered.assert_not_called()

    def test_variant_cpu_train_eval_and_exact_resume(self) -> None:
        real_run, real_filter = runner.run_training, lowpass.apply_condition
        phase = ['outside']
        calls: list[str] = []
        def filtered(*args: Any, **kwargs: Any) -> torch.Tensor:
            calls.append(phase[0])
            return real_filter(*args, **kwargs)
        def shortened(**kwargs: Any) -> dict[str, Any]:
            kwargs.update(max_epochs=2, checkpoint_interval=1)
            step, evaluate = kwargs['train_step'], kwargs['evaluate']
            def measured_step(*args: Any) -> float:
                phase[0] = 'train'
                return step(*args)
            def measured_eval() -> dict[str, Any]:
                phase[0] = 'eval'
                return evaluate()
            kwargs.update(train_step=measured_step, evaluate=measured_eval)
            return real_run(**kwargs)
        with patch.object(lowpass, 'apply_condition', side_effect=filtered):
            self.assertEqual(self.invoke(X1R, 'full', shortened), 0)
            self.assertEqual(self.invoke(X1R, 'split', shortened, ['--session-steps', '1']), 0)
            self.assertEqual(self.invoke(X1R, 'split', shortened, ['--resume']), 0)
        self.assertGreater(calls.count('train'), 0)
        self.assertGreater(calls.count('eval'), 0)
        full = GenerationStore(self.root / 'full').load()
        resumed = GenerationStore(self.root / 'split').load()
        self.assertEqual(full['identity'], resumed['identity'])
        runner_tests.RunnerTests.compare(self, {'payload': full}, {'payload': resumed})
        report = json.loads((self.root / 'full.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'max_epochs')
        self.assertEqual(report['controller']['global_step'], 2)
        self.assertEqual(report['identity']['run_id'], X1R)


class X1RProfileTests(unittest.TestCase):
    def test_pair_list_preserves_x0_commands_byte_for_byte(self) -> None:
        base = ['python', '-c', 'pass']
        profiles = workflow.build_xlsr_pair_profiles(base, pairs=[('X0', 2027, 2028), (X1R, 2026, 2027)])
        old = workflow.build_xlsr_pair_profiles(base)
        for name, command in old.items():
            self.assertEqual(canonical(profiles[name]), canonical(command))
        name = f'{X1R}-seed2026-{X1R}-seed2027'
        self.assertEqual(profiles[name + '-dry-run'][0], profiles[name][0] + ['--dry-run'])
        with self.assertRaises(workflow.WorkflowError):
            workflow.build_xlsr_pair_profiles(base, pairs=[(X1R, 2026, 2026)])
        with self.assertRaises(workflow.WorkflowError):
            workflow.build_xlsr_pair_profiles(base, pairs=[('X0', 2027, 2028)] * 2)

    def test_x1r_single_commands_and_pair_gpu_isolation(self) -> None:
        module = bootstrap()
        plan = json.loads((ROOT / 'configs/xlsr_training.json').read_text(encoding='utf-8'))
        for seed in ('2026', '2027', '2028'):
            args = argparse.Namespace(run_id=X1R, training_seed=seed, data_root=Path('/data'), dry_run=False, resume_manifest_sha256=None)
            cmd = module.build_runner_command(args, plan=plan, python='python', weights=Path('/weights'), run_root=Path('/run'), report=Path('/report'), started_unix=1000)
            self.assertEqual(cmd[cmd.index('--run-id') + 1], X1R)
            self.assertEqual(cmd[cmd.index('--training-seed') + 1], seed)
        args.second_run_id, args.training_seed, args.second_training_seed = X1R, '2026', '2027'
        args.working_root = Path('/working')
        specs = module.build_pair_runner_specs(args, plan=plan, python='python', weights=Path('/weights'), base_env={}, started_unix=1000)
        self.assertEqual([item['env']['CUDA_VISIBLE_DEVICES'] for item in specs], ['0', '1'])
        self.assertEqual([item['env']['OMP_NUM_THREADS'] for item in specs], ['2', '2'])
        self.assertEqual([item['env']['MKL_NUM_THREADS'] for item in specs], ['2', '2'])
        for spec in specs:
            self.assertIn(X1R, str(spec['run_root']))

    def test_registered_x1r_pair_metadata_and_guard(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache/t29/tmp') as temporary:
            path = Path(temporary)
            payload = config_payload(path.relative_to(ROOT).as_posix())
            profiles = workflow.build_xlsr_pair_profiles(
                ['python', '-c', 'pass', '--data-root', '{data_root}'], pairs=[(X1R, 2026, 2027)])
            payload['profiles'].update(profiles)
            config_path = path / 'project.json'
            config_path.write_text(json.dumps(payload), encoding='utf-8')
            config = workflow.load_config(config_path)
            source = config.workspace_dir / 'source_dataset'
            source.mkdir(parents=True)
            (source / 'source_manifest.json').write_text(json.dumps({'bundle_sha256': 'b' * 64}), encoding='utf-8')
            for name in profiles:
                directory = workflow.prepare_xlsr_pair_kernel(config, name, pair_profiles=list(profiles))
                metadata = json.loads((directory / 'kernel-metadata.json').read_text(encoding='utf-8'))
                self.assertEqual(metadata['machine_shape'], 'NvidiaTeslaT4')
                self.assertTrue(metadata['is_private'] and metadata['enable_gpu'])
                self.assertFalse(metadata['enable_internet'])
                self.assertEqual(len(metadata['dataset_sources']), 4)
            with self.assertRaisesRegex(workflow.WorkflowError, 'danh sách cặp'):
                workflow.prepare_xlsr_pair_kernel(config, config.default_profile, pair_profiles=list(profiles))


if __name__ == '__main__':
    unittest.main()
