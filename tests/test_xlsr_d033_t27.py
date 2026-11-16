"""T27 (D033): kiểm thử chế độ cặp Kaggle 2 GPU, khối hardware/GPU lock, script parity và entrypoint/notebook Colab."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import shutil
import re
import subprocess
import sys
import threading
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from colab import xlsr_colab_entry as colab_entry
from scripts import check_xlsr_env_parity as parity
from scripts import kaggle_workflow as workflow
from scripts import train_xlsr as runner
from scripts.train_b0 import _set_seed
from src.training.full_training import GenerationStore, sha256_file, write_json
from tests.test_b0_v2_runner import _write_artifact

ROOT = Path(__file__).resolve().parents[1]
TMP_BASE = ROOT / '.cache' / 't27b' / 'tmp'


real_collect_hardware_info = runner.collect_hardware_info


def fake_hardware(device: torch.device, name: str) -> dict[str, Any]:
    return real_collect_hardware_info(device, gpu_name_fn=lambda index: name, device_count_fn=lambda: 1)


def load_bootstrap_module() -> Any:
    spec = importlib.util.spec_from_file_location('xlsr_bootstrap_t27', ROOT / '.kaggle/xlsr-d028/bootstrap.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('Không thể nạp .kaggle/xlsr-d028/bootstrap.py.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SingleRunBaselineParityTests(unittest.TestCase):
    def test_single_run_commands_and_env_match_baseline_bytes(self) -> None:
        module = load_bootstrap_module()
        plan = json.loads((ROOT / 'configs/xlsr_training.json').read_text(encoding='utf-8'))
        expected_env = {
            'PYTHONUTF8': '1',
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'PIP_NO_INDEX': '1',
            'PIP_DISABLE_PIP_VERSION_CHECK': '1',
            'USE_TF': '0',
            'USE_FLAX': '0',
            'CUDA_VISIBLE_DEVICES': '0',
        }
        actual_env = module.build_single_run_env({})
        self.assertEqual(
            json.dumps(actual_env, sort_keys=True).encode('utf-8'),
            json.dumps(expected_env, sort_keys=True).encode('utf-8'),
        )
        before_path = ROOT / '.cache/t27/before.json'
        baseline_cmds = (
            json.loads(before_path.read_text(encoding='utf-8-sig'))['bootstrap_single_run_baseline']
            if before_path.is_file()
            else None
        )
        for seed in ('2027', '2028'):
            for dry_run in (False, True):
                args = argparse.Namespace(
                    run_id='X0',
                    training_seed=seed,
                    data_root=Path('/kaggle/input/doanttnt-t14-packed'),
                    dry_run=dry_run,
                    resume_manifest_sha256=None,
                )
                cmd = module.build_runner_command(
                    args,
                    plan=plan,
                    python='/kaggle/working/.xlsr-offline-env/bin/python',
                    weights=Path('/kaggle/input/doanttnt-t14-weights'),
                    run_root=Path(f'/kaggle/working/checkpoints/xlsr_X0_seed{seed}'),
                    report=Path(f'/kaggle/working/reports/xlsr_X0_seed{seed}.json'),
                    started_unix=1000.0,
                )
                if baseline_cmds is not None:
                    key = f'X0-seed{seed}-dry-run' if dry_run else f'X0-seed{seed}'
                    self.assertEqual(
                        json.dumps(cmd, ensure_ascii=False).encode('utf-8'),
                        json.dumps(baseline_cmds[key]['command'], ensure_ascii=False).encode('utf-8'),
                    )
                    self.assertEqual(
                        json.dumps(actual_env, sort_keys=True, ensure_ascii=False).encode('utf-8'),
                        json.dumps(baseline_cmds[key]['env_overrides'], sort_keys=True, ensure_ascii=False).encode('utf-8'),
                    )


class PairBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP_BASE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TMP_BASE)
        self.root = Path(self.temp.name)
        self.module = load_bootstrap_module()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _make_pair_args(self, *, dry_run: bool = False, second_seed: str | None = '2028') -> argparse.Namespace:
        return argparse.Namespace(
            data_root=self.root / 'data',
            working_root=self.root / 'working',
            run_id='X0',
            training_seed='2027',
            dry_run=dry_run,
            resume_manifest_sha256=None,
            expected_step=None,
            second_run_id='X0' if second_seed is not None else None,
            second_training_seed=second_seed,
            second_resume_manifest_sha256=None,
            second_expected_step=None,
        )

    def test_pair_bootstrap_spawns_two_processes_with_isolated_gpus_shared_deadline_and_threads(self) -> None:
        manifest = self.root / 'offline_manifest.json'
        manifest.write_text('{"files":{}}', encoding='utf-8')
        args = self._make_pair_args(dry_run=False)
        spawned: list[dict[str, Any]] = []

        class FakeProc:
            def __init__(self, command: list[str], env: dict[str, str], **kwargs: Any) -> None:
                self.stdout = io.BytesIO()
                self.command = command
                self.env = env
                self.waited = False
                spawned.append({'command': command, 'env': env, 'proc': self})

            def wait(self) -> int:
                self.waited = True
                return 0

        stdout = io.StringIO()
        with redirect_stdout(stdout), \
             patch.object(self.module, 'time', SimpleNamespace(time=lambda: 1000.0), create=True), \
             patch.object(self.module.argparse.ArgumentParser, 'parse_args', return_value=args), \
             patch.object(self.module, 'detect_gpu_count', return_value=2), \
             patch.object(self.module.Path, 'rglob', side_effect=lambda pattern: [manifest] if pattern == 'offline_manifest.json' else [self.root / 'pytorch_model.bin']), \
             patch.object(self.module.venv, 'EnvBuilder'), \
             patch.object(self.module.subprocess, 'check_output', return_value=b'unchanged'), \
             patch.object(self.module, 'checked'), \
             patch.object(self.module.subprocess, 'Popen', side_effect=FakeProc):
            exit_code = self.module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(spawned), 2)
        cmd_a, env_a = spawned[0]['command'], spawned[0]['env']
        cmd_b, env_b = spawned[1]['command'], spawned[1]['env']
        self.assertEqual(env_a['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(env_b['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(env_a['OMP_NUM_THREADS'], '2')
        self.assertEqual(env_a['MKL_NUM_THREADS'], '2')
        self.assertEqual(env_b['OMP_NUM_THREADS'], '2')
        self.assertEqual(env_b['MKL_NUM_THREADS'], '2')
        self.assertEqual(cmd_a[cmd_a.index('--device') + 1], 'cuda:0')
        self.assertEqual(cmd_b[cmd_b.index('--device') + 1], 'cuda:0')
        deadline_a = float(cmd_a[cmd_a.index('--deadline-unix') + 1])
        deadline_b = float(cmd_b[cmd_b.index('--deadline-unix') + 1])
        self.assertEqual(deadline_a, 40600.0)
        self.assertEqual(deadline_b, 40600.0)
        self.assertNotEqual(cmd_a[cmd_a.index('--run-root') + 1], cmd_b[cmd_b.index('--run-root') + 1])
        self.assertNotEqual(cmd_a[cmd_a.index('--report-json') + 1], cmd_b[cmd_b.index('--report-json') + 1])
        events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        starts = [e for e in events if e['event'] == 'runner_start']
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0]['omp_num_threads'], '2')
        self.assertEqual(starts[0]['mkl_num_threads'], '2')

    def test_pair_bootstrap_waits_for_second_process_when_first_fails_and_returns_nonzero(self) -> None:
        manifest = self.root / 'offline_manifest.json'
        manifest.write_text('{"files":{}}', encoding='utf-8')
        args = self._make_pair_args(dry_run=False)
        wait_order: list[str] = []
        terminated: list[str] = []

        class FakeProc:
            def __init__(self, command: list[str], env: dict[str, str], **kwargs: Any) -> None:
                self.stdout = io.BytesIO()
                self.slot = 'A' if env['CUDA_VISIBLE_DEVICES'] == '0' else 'B'

            def terminate(self) -> None:
                terminated.append(self.slot)

            def kill(self) -> None:
                terminated.append(self.slot)

            def wait(self) -> int:
                wait_order.append(self.slot)
                return 5 if self.slot == 'A' else 0

        stdout = io.StringIO()
        with redirect_stdout(stdout), \
             patch.object(self.module, 'time', SimpleNamespace(time=lambda: 1000.0), create=True), \
             patch.object(self.module.argparse.ArgumentParser, 'parse_args', return_value=args), \
             patch.object(self.module, 'detect_gpu_count', return_value=2), \
             patch.object(self.module.Path, 'rglob', side_effect=lambda pattern: [manifest] if pattern == 'offline_manifest.json' else [self.root / 'pytorch_model.bin']), \
             patch.object(self.module.venv, 'EnvBuilder'), \
             patch.object(self.module.subprocess, 'check_output', return_value=b'unchanged'), \
             patch.object(self.module, 'checked'), \
             patch.object(self.module.subprocess, 'Popen', side_effect=FakeProc):
            exit_code = self.module.main()

        self.assertEqual(exit_code, 5)
        self.assertEqual(wait_order, ['A', 'B'])
        self.assertEqual(terminated, [])
        events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        ends = {e['slot']: e['exit_code'] for e in events if e['event'] == 'runner_end'}
        self.assertEqual(ends, {'A': 5, 'B': 0})

    def test_pair_bootstrap_rejects_when_gpu_count_is_not_two(self) -> None:
        args = self._make_pair_args(dry_run=False)
        for bad_count in (0, 1, 3):
            with self.subTest(gpu_count=bad_count), \
                 patch.object(self.module.argparse.ArgumentParser, 'parse_args', return_value=args), \
                 patch.object(self.module, 'detect_gpu_count', return_value=bad_count), \
                 patch.object(self.module.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(ValueError, 'đúng 2 GPU'):
                    self.module.main()
                popen.assert_not_called()

    def test_pair_bootstrap_rejects_duplicate_or_incomplete_second_run(self) -> None:
        dup_args = self._make_pair_args(second_seed='2027')
        with patch.object(self.module.argparse.ArgumentParser, 'parse_args', return_value=dup_args):
            with self.assertRaisesRegex(ValueError, 'không được trùng'):
                self.module.main()
        incomplete = self._make_pair_args(second_seed=None)
        incomplete.second_run_id = 'X0'
        with patch.object(self.module.argparse.ArgumentParser, 'parse_args', return_value=incomplete):
            with self.assertRaisesRegex(ValueError, 'Chế độ cặp cần đủ'):
                self.module.main()


    def test_pair_logs_match_download_filter_at_working_root(self) -> None:
        args = self._make_pair_args()
        specs = self.module.build_pair_runner_specs(
            args, plan=json.loads((ROOT / 'configs/xlsr_training.json').read_text(encoding='utf-8')),
            python='python', weights=self.root, base_env={}, started_unix=1000.0,
        )
        for spec in specs:
            relative = spec['log_path'].relative_to(args.working_root).as_posix()
            self.assertEqual(spec['log_path'].parent, args.working_root)
            self.assertRegex(relative, r'^reports/|^checkpoints/|^[^/]+\.log$')
            self.assertNotIn('logs/', relative)

    def test_tee_preserves_bytes_and_prefixes_when_one_process_fails(self) -> None:
        output = io.StringIO()
        lock = threading.Lock()
        readers: list[threading.Thread] = []
        processes: list[subprocess.Popen[bytes]] = []
        with redirect_stdout(output):
            for slot, seed, code in [('A', 2027, 5), ('B', 2028, 0)]:
                proc = subprocess.Popen([sys.executable, '-c',
                    "import sys; print('dòng 1', flush=True); print('dòng 2', file=sys.stderr, flush=True); print('dòng 3', flush=True); sys.exit(" + str(code) + ")"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                reader = threading.Thread(target=self.module.tee_runner_output,
                    args=(proc, (self.root / (slot + '.log')).open('wb'), f'[{slot} X0 seed{seed}] ', lock))
                reader.start()
                readers.append(reader)
                processes.append(proc)
            self.assertEqual([proc.wait() for proc in processes], [5, 0])
            for reader in readers:
                reader.join()
        for slot, seed in [('A', 2027), ('B', 2028)]:
            raw = (self.root / (slot + '.log')).read_bytes()
            self.assertEqual(raw.decode('utf-8').splitlines(), ['dòng 1', 'dòng 2', 'dòng 3'])
            for line in raw.decode('utf-8').splitlines():
                self.assertIn(f'[{slot} X0 seed{seed}] ' + line, output.getvalue())
        (ROOT / '.cache/t27b/tee_stdout.log').write_text(output.getvalue(), encoding='utf-8')

    def test_tee_reader_failure_does_not_stop_other_reader(self) -> None:
        class BrokenReader(io.BytesIO):
            def readline(self, size: int = -1) -> bytes:
                raise OSError('Lỗi đọc giả lập')
        output = io.StringIO()
        lock = threading.Lock()
        with redirect_stdout(output):
            threads = [threading.Thread(target=self.module.tee_runner_output,
                args=(SimpleNamespace(stdout=stream), (self.root / (slot + '.log')).open('wb'), f'[{slot} X0 seed2028] ', lock))
                for slot, stream in [('A', BrokenReader()), ('B', io.BytesIO(b'one\ntwo\nthree\n'))]]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual((self.root / 'B.log').read_bytes(), b'one\ntwo\nthree\n')
        self.assertIn('Lỗi đọc/ghi log runner', output.getvalue())
        self.assertIn('[B X0 seed2028] three', output.getvalue())

    def test_gpu_probe_accepts_injected_count(self) -> None:
        self.assertEqual(self.module.detect_gpu_count(count_fn=lambda: 2), 2)

    def test_production_sources_do_not_detect_mock_modules(self) -> None:
        for path in ['scripts/train_xlsr.py', '.kaggle/xlsr-d028/bootstrap.py']:
            self.assertNotIn('__module__', (ROOT / path).read_text(encoding='utf-8'))



class RunnerHardwareAndGpuLockTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        TMP_BASE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TMP_BASE)
        self.root = Path(self.temp.name)
        self.plan = json.loads((ROOT / 'configs/xlsr_training.json').read_text(encoding='utf-8'))
        self.artifact = self.root / 'packed-train'
        _write_artifact(self.artifact, split='closed_train', rows=4, manifest_sha256='a' * 64, offset=3)
        self.dev_artifact = self.root / 'packed-dev'
        _write_artifact(self.dev_artifact, split='closed_dev', rows=4, manifest_sha256='b' * 64, offset=6)
        self.plan['implementation']['ready_for_training'] = True
        for prefix, path in (('train', self.artifact), ('development', self.dev_artifact)):
            meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
            self.plan['data'][prefix + '_samples'] = 4
            self.plan['data'][prefix + '_metadata_sha256'] = runner.sha256(path / 'metadata.json')
            self.plan['data'][prefix + '_manifest_sha256'] = meta['source_manifest_sha256']
            self.plan['data'][prefix + '_artifact_waveforms_sha256'] = meta['waveforms_sha256']
        self.plan_path = self.root / 'plan.json'
        self.plan_path.write_text(json.dumps(self.plan), encoding='utf-8')

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _tiny_backbone(self, *args: Any, **kwargs: Any) -> Wav2Vec2Model:
        _set_seed(2026)
        cfg = json.loads((ROOT / 'configs/xlsr_smoke.json').read_text(encoding='utf-8'))['backbone']
        cfg.update(num_hidden_layers=3, layerdrop=0.1, apply_spec_augment=True)
        return Wav2Vec2Model(Wav2Vec2Config(**cfg))

    def _base_cli_args(self, run_root: Path, report_json: Path) -> list[str]:
        return [
            '--plan', str(self.plan_path),
            '--run-id', 'X1-L2-lr1e-5',
            '--training-seed', '2026',
            '--device', 'cpu',
            '--pretrained-dir', str(self.root / 'weights'),
            '--train-artifact', str(self.artifact),
            '--dev-artifact', str(self.dev_artifact),
            '--run-root', str(run_root),
            '--report-json', str(report_json),
        ]

    def test_cpu_hardware_explicitly_records_no_gpu(self) -> None:
        hw = runner.collect_hardware_info(torch.device('cpu'))
        self.assertFalse(hw['has_gpu'])
        self.assertIsNone(hw['gpu_name'])
        self.assertEqual(hw['gpu_status'], 'không có GPU')
        self.assertEqual(hw['visible_gpu_count'], 0)

    def test_hardware_recorded_in_report_and_state_json_outside_byte_identical_identity(self) -> None:
        run_root = self.root / 'run-t4'
        report_path = self.root / 'report-t4.json'
        with patch.object(runner, 'validate_plan', return_value={'run_id': 'X1-L2-lr1e-5', 'layers': 2, 'encoder_lr': 1e-5}), \
             patch.object(runner, 'verify_pretrained', return_value={'fixture': 'ok'}), \
             patch.object(runner, 'load_xlsr_backbone', side_effect=self._tiny_backbone), \
             patch.object(runner, 'verify_packed_artifact'), \
             patch.object(runner, '_sample_selection_sha256', return_value=self.plan['data']['sample_selection_sha256']), \
             patch.object(runner, 'collect_hardware_info', side_effect=lambda device: fake_hardware(device, 'Tesla T4')):
            code = runner.main(self._base_cli_args(run_root, report_path) + ['--session-steps', '1', '--expected-gpu', 'Tesla T4'])
        self.assertEqual(code, 0)
        report = json.loads(report_path.read_text(encoding='utf-8'))
        self.assertIn('hardware', report)
        self.assertEqual(report['hardware']['gpu_name'], 'Tesla T4')
        self.assertEqual(report['hardware']['visible_gpu_count'], 1)
        self.assertFalse(report['hardware']['tf32']['cuda_matmul_allow_tf32'])
        self.assertFalse(report['hardware']['tf32']['cudnn_allow_tf32'])
        self.assertNotIn('hardware', report['identity'])

        expected_identity_keys = {
            'plan', 'run', 'run_id', 'training_seed', 'data_seed', 'device',
            'backend', 'pretrained', 'source_files', 'modes', 'trainable_only',
            'technical_smoke', 'batch_size', 'train_indices', 'development_indices',
            'artifacts', 'source_tree_sha256',
        }
        self.assertEqual(set(report['identity'].keys()), expected_identity_keys)
        pointer = json.loads((run_root / 'LATEST.json').read_text(encoding='utf-8'))
        state_json = json.loads((run_root / pointer['latest']['generation'] / 'state.json').read_text(encoding='utf-8'))
        self.assertIn('hardware', state_json)
        self.assertEqual(state_json['hardware']['gpu_name'], 'Tesla T4')

    def test_expected_gpu_mismatch_stops_before_training(self) -> None:
        run_root = self.root / 'run-mismatch'
        report_path = self.root / 'report-mismatch.json'
        with patch.object(runner, 'validate_plan', return_value={'run_id': 'X1-L2-lr1e-5', 'layers': 2, 'encoder_lr': 1e-5}), \
             patch.object(runner, 'verify_pretrained') as verify_mock, \
             patch.object(runner, 'collect_hardware_info', side_effect=lambda device: fake_hardware(device, 'Tesla T4')):
            with self.assertRaisesRegex(ValueError, 'expected-gpu'):
                runner.main(self._base_cli_args(run_root, report_path) + ['--expected-gpu', 'NVIDIA L4'])
            verify_mock.assert_not_called()

    def test_resume_rejects_different_gpu_in_state_json(self) -> None:
        run_root = self.root / 'run-gpu-lock'
        report_path = self.root / 'report-gpu-lock.json'
        with patch.object(runner, 'validate_plan', return_value={'run_id': 'X1-L2-lr1e-5', 'layers': 2, 'encoder_lr': 1e-5}), \
             patch.object(runner, 'verify_pretrained', return_value={'fixture': 'ok'}), \
             patch.object(runner, 'load_xlsr_backbone', side_effect=self._tiny_backbone), \
             patch.object(runner, 'verify_packed_artifact'), \
             patch.object(runner, '_sample_selection_sha256', return_value=self.plan['data']['sample_selection_sha256']):
            with patch.object(runner, 'collect_hardware_info', side_effect=lambda device: fake_hardware(device, 'Tesla T4')):
                self.assertEqual(runner.main(self._base_cli_args(run_root, report_path) + ['--session-steps', '1']), 0)
            with patch.object(runner, 'collect_hardware_info', side_effect=lambda device: fake_hardware(device, 'NVIDIA L4')):
                with self.assertRaisesRegex(ValueError, 'Không được resume trên loại GPU khác'):
                    runner.main(self._base_cli_args(run_root, report_path) + ['--session-steps', '1', '--resume'])

    def test_resume_same_gpu_with_hardware_matches_uninterrupted(self) -> None:
        full_root, split_root = self.root / 'full', self.root / 'split'
        report_path = self.root / 'report.json'
        with patch.object(runner, 'validate_plan', return_value={'run_id': 'X1-L2-lr1e-5', 'layers': 2, 'encoder_lr': 1e-5}), \
             patch.object(runner, 'verify_pretrained', return_value={'fixture': 'ok'}), \
             patch.object(runner, 'load_xlsr_backbone', side_effect=self._tiny_backbone), \
             patch.object(runner, 'verify_packed_artifact'), \
             patch.object(runner, '_sample_selection_sha256', return_value=self.plan['data']['sample_selection_sha256']), \
             patch.object(runner, 'collect_hardware_info', side_effect=lambda device: fake_hardware(device, 'Tesla T4')):
            self.assertEqual(runner.main(self._base_cli_args(full_root, report_path) + ['--session-steps', '99']), 0)
            self.assertEqual(runner.main(self._base_cli_args(split_root, report_path) + ['--session-steps', '1']), 0)
            pointer = json.loads((split_root / 'LATEST.json').read_text(encoding='utf-8'))
            state = json.loads((split_root / pointer['latest']['generation'] / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['hardware']['gpu_name'], 'Tesla T4')
            self.assertEqual(state['global_step'], 1)
            self.assertEqual(runner.main(self._base_cli_args(split_root, report_path) + ['--session-steps', '99', '--resume']), 0)
        controllers = [GenerationStore(root).load()['controller'] for root in (full_root, split_root)]
        for controller in controllers:
            controller.pop('hardware')
            controller['best']['generation'] = None
            controller['best']['development'].pop('timing', None)
            for row in controller['history']:
                row.get('development', {}).pop('timing', None)
        self.assertEqual(controllers[0], controllers[1])

    def test_cpu_never_calls_cuda_name_or_count_without_injection(self) -> None:
        with patch.object(torch.cuda, 'get_device_name', side_effect=AssertionError('Không được gọi CUDA')), \
             patch.object(torch.cuda, 'device_count', side_effect=AssertionError('Không được gọi CUDA')):
            self.assertIsNone(runner.get_gpu_name(torch.device('cpu')))
            self.assertEqual(runner.collect_hardware_info(torch.device('cpu'))['visible_gpu_count'], 0)
            self.assertEqual(fake_hardware(torch.device('cpu'), 'Tesla T4')['gpu_name'], 'Tesla T4')

    def test_legacy_v4_checkpoint_without_hardware_resumes_cleanly(self) -> None:
        run_root = self.root / 'run-legacy'
        report_path = self.root / 'report-legacy.json'
        with patch.object(runner, 'validate_plan', return_value={'run_id': 'X1-L2-lr1e-5', 'layers': 2, 'encoder_lr': 1e-5}), \
             patch.object(runner, 'verify_pretrained', return_value={'fixture': 'ok'}), \
             patch.object(runner, 'load_xlsr_backbone', side_effect=self._tiny_backbone), \
             patch.object(runner, 'verify_packed_artifact'), \
             patch.object(runner, '_sample_selection_sha256', return_value=self.plan['data']['sample_selection_sha256']), \
             patch.object(runner, 'collect_hardware_info', side_effect=lambda device: fake_hardware(device, 'Tesla T4')):
            self.assertEqual(runner.main(self._base_cli_args(run_root, report_path) + ['--session-steps', '1']), 0)
            # Xóa khối hardware khỏi checkpoint để mô phỏng checkpoint v4 cũ trước T27.
            pointer_path = run_root / 'LATEST.json'
            pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
            gen = pointer['latest']['generation']
            ckpt_path = run_root / gen / 'checkpoint.pt'
            state_path = run_root / gen / 'state.json'
            payload = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            payload['controller'].pop('hardware', None)
            torch.save(payload, ckpt_path)
            write_json(state_path, payload['controller'])
            pointer['latest']['checkpoint_sha256'] = sha256_file(ckpt_path)
            pointer['latest']['state_sha256'] = sha256_file(state_path)
            write_json(pointer_path, pointer)

            self.assertEqual(runner.main(self._base_cli_args(run_root, report_path) + ['--session-steps', '1', '--resume']), 0)


class ParityScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP_BASE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TMP_BASE)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _build_synthetic_reference(self, count_per_class: int = 500) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        scores: list[float] = []
        for i in range(count_per_class):
            score_neg = 0.10 + 0.35 * (i / count_per_class)
            records.append({'file': f'bonafide_{i:04d}.wav', 'label': 0, 'score': score_neg})
            scores.append(score_neg)
        for i in range(count_per_class):
            score_pos = 0.55 + 0.35 * (i / count_per_class)
            records.append({'file': f'spoof_{i:04d}.wav', 'label': 1, 'score': score_pos})
            scores.append(score_pos)
        ref_dev = {
            'sample_count': len(records),
            'decision_threshold': 0.50,
            'records': records,
            'scores': scores,
        }
        return ref_dev, copy.deepcopy(records)

    def test_parity_passes_when_within_all_three_thresholds(self) -> None:
        ref_dev, eval_records = self._build_synthetic_reference()
        for idx in range(5):
            eval_records[idx]['score'] += 0.01
        report = parity.compare_parity_scores(ref_dev, eval_records, hardware={'gpu_status': 'NVIDIA L4'})
        self.assertTrue(report['passed'])
        self.assertEqual(report['checks'], {
            'abs_delta_eer_pct_points_ok': True,
            'decision_flips_ok': True,
            'max_abs_delta_score_ok': True,
        })
        out_json = self.root / 'parity.json'
        out_md = self.root / 'parity.md'
        parity.write_parity_reports(report, out_json, out_md)
        self.assertTrue(out_json.is_file())
        self.assertIn('**ĐẠT**', out_md.read_text(encoding='utf-8'))

    def test_parity_fails_when_only_eer_delta_exceeds_threshold(self) -> None:
        # 500 âm, 500 dương: tạo giao điểm EER quanh 0.50 và dịch nhỏ (<= 0.04) ở 1 mẫu để |ΔEER| = 0.10 điểm % > 0.02 điểm %,
        # nhưng số mẫu đổi quyết định tại decision_threshold (0.50) = 1 <= 10 và max |Δ score| = 0.04 <= 0.05.
        ref_dev, eval_records = self._build_synthetic_reference(500)
        ref_dev['records'][499]['score'] = 0.51
        ref_dev['scores'][499] = 0.51
        ref_dev['records'][500]['score'] = 0.49
        ref_dev['scores'][500] = 0.49
        eval_records = copy.deepcopy(ref_dev['records'])
        eval_records[499]['score'] = 0.47  # Δscore = 0.04 <= 0.05, chỉ 1 mẫu đổi quyết định qua 0.50, nhưng FAR tại ngưỡng giao điểm giảm 1/500 = 0.2%
        report = parity.compare_parity_scores(ref_dev, eval_records, hardware={'gpu_status': 'NVIDIA L4'})
        self.assertFalse(report['checks']['abs_delta_eer_pct_points_ok'])
        self.assertTrue(report['checks']['decision_flips_ok'])
        self.assertTrue(report['checks']['max_abs_delta_score_ok'])
        self.assertFalse(report['passed'])

    def test_parity_fails_when_only_decision_flips_exceed_threshold(self) -> None:
        # Đặt 12 mẫu âm ở 0.499 -> 0.501 (Δscore = 0.002 <= 0.05) và đặt decision_threshold = 0.50,
        # trong khi toàn bộ mẫu dương ở >= 0.70 nên EER của cả hai đều bằng 0.0 (|ΔEER| = 0.0 <= 0.02%).
        ref_dev, eval_records = self._build_synthetic_reference(500)
        for i in range(12):
            ref_dev['records'][i]['score'] = 0.499
            ref_dev['scores'][i] = 0.499
            eval_records[i]['score'] = 0.501
        report = parity.compare_parity_scores(ref_dev, eval_records, hardware={'gpu_status': 'NVIDIA L4'})
        self.assertTrue(report['checks']['abs_delta_eer_pct_points_ok'])
        self.assertFalse(report['checks']['decision_flips_ok'])
        self.assertEqual(report['decision_flips'], 12)
        self.assertTrue(report['checks']['max_abs_delta_score_ok'])
        self.assertFalse(report['passed'])

    def test_parity_fails_when_only_max_abs_delta_score_exceeds_threshold(self) -> None:
        # Dịch 1 mẫu âm từ 0.10 lên 0.16 (Δscore = 0.06 > 0.05), không đổi quyết định tại 0.50 và không đổi EER.
        ref_dev, eval_records = self._build_synthetic_reference(500)
        eval_records[0]['score'] = ref_dev['records'][0]['score'] + 0.06
        report = parity.compare_parity_scores(ref_dev, eval_records, hardware={'gpu_status': 'NVIDIA L4'})
        self.assertTrue(report['checks']['abs_delta_eer_pct_points_ok'])
        self.assertTrue(report['checks']['decision_flips_ok'])
        self.assertFalse(report['checks']['max_abs_delta_score_ok'])
        self.assertFalse(report['passed'])

    def test_parity_rejects_file_order_mismatch_and_cli_threshold_arguments(self) -> None:
        ref_dev, eval_records = self._build_synthetic_reference(10)
        eval_records[0], eval_records[1] = eval_records[1], eval_records[0]
        with self.assertRaisesRegex(ValueError, 'Thứ tự mẫu lệch'):
            parity.compare_parity_scores(ref_dev, eval_records, hardware={})
        parser = parity.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                '--checkpoint', 'ckpt.pt',
                '--reference-report', 'ref.json',
                '--pretrained-dir', 'weights',
                '--output-json', 'out.json',
                '--output-md', 'out.md',
                '--max-eer-delta', '0.1',
            ])


class ColabEntrypointAndNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP_BASE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TMP_BASE)
        self.root = Path(self.temp.name)
        self.drive_data = self.root / 'drive_data'
        self.drive_out = self.root / 'drive_out'
        self.work_dir = self.root / 'local_work'
        self.plan = json.loads((ROOT / 'configs/xlsr_training.json').read_text(encoding='utf-8'))

        _write_artifact(self.drive_data / 'train_development_20k_v2', split='closed_train', rows=4, manifest_sha256='a' * 64, offset=3)
        _write_artifact(self.drive_data / 'closed_dev_full_v1', split='closed_dev', rows=4, manifest_sha256='b' * 64, offset=6)
        weights_dir = self.drive_data / 'wav2vec2-xls-r-300m'
        weights_dir.mkdir(parents=True, exist_ok=True)
        for name, key in (
            ('pytorch_model.bin', 'weights_sha256'),
            ('config.json', 'config_sha256'),
            ('preprocessor_config.json', 'preprocessor_config_sha256'),
        ):
            p = weights_dir / name
            p.write_bytes(name.encode('utf-8'))
            self.plan['pretrained'][key] = runner.sha256(p)
        for prefix, folder in (('train', 'train_development_20k_v2'), ('development', 'closed_dev_full_v1')):
            art = self.drive_data / folder
            meta = json.loads((art / 'metadata.json').read_text(encoding='utf-8'))
            self.plan['data'][prefix + '_samples'] = 4
            self.plan['data'][prefix + '_metadata_sha256'] = runner.sha256(art / 'metadata.json')
            self.plan['data'][prefix + '_manifest_sha256'] = meta['source_manifest_sha256']
            self.plan['data'][prefix + '_artifact_waveforms_sha256'] = meta['waveforms_sha256']

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_colab_entrypoint_stages_to_local_and_rejects_corrupted_checksum(self) -> None:
        staged = colab_entry.stage_and_verify_artifacts(
            drive_data_dir=self.drive_data,
            work_dir=self.work_dir,
            plan=self.plan,
            training_seed=2026,
            include_train=True,
        )
        self.assertTrue(staged['train_artifact'].is_relative_to(self.work_dir))
        self.assertTrue(staged['dev_artifact'].is_relative_to(self.work_dir))

        (self.drive_data / 'wav2vec2-xls-r-300m' / 'pytorch_model.bin').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Checksum pretrained không khớp'):
            colab_entry.stage_and_verify_artifacts(
                drive_data_dir=self.drive_data,
                work_dir=self.work_dir,
                plan=self.plan,
                training_seed=2026,
                include_train=True,
            )

    def test_colab_entrypoint_rejects_corrupted_packed_waveform_checksum(self) -> None:
        (self.drive_data / 'closed_dev_full_v1' / 'waveforms.f32').write_bytes(b'\x00' * 16)
        with self.assertRaises(ValueError):
            colab_entry.stage_and_verify_artifacts(
                drive_data_dir=self.drive_data,
                work_dir=self.work_dir,
                plan=self.plan,
                training_seed=2026,
                include_train=False,
            )

    def test_colab_entrypoint_verifies_library_versions_and_logs_pip_freeze(self) -> None:
        log_path = self.work_dir / 'env.log'
        with patch.object(colab_entry, 'record_pip_freeze', return_value=['torch==2.11.0+cu124']):
            info = colab_entry.verify_colab_environment(
                torch_version='2.11.0+cu124',
                cuda_version='12.4',
                transformers_version='4.57.6',
                numpy_version='2.1.3',
                log_path=log_path,
            )
        self.assertFalse(info['exact_cuda_build'])
        self.assertIn('biến thể CUDA khác', info['cuda_note'])

        for bad_kwargs, pattern in (
            ({'torch_version': '2.10.0+cu121', 'transformers_version': '4.57.6', 'numpy_version': '2.1.3'}, 'PyTorch'),
            ({'torch_version': '2.11.0+cu130', 'transformers_version': '4.56.0', 'numpy_version': '2.1.3'}, 'transformers'),
            ({'torch_version': '2.11.0+cu130', 'transformers_version': '4.57.6', 'numpy_version': '2.0.0'}, 'numpy'),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                colab_entry.verify_colab_environment(**bad_kwargs)

    def test_colab_entrypoint_never_allows_test_artifacts_or_drive_memmap(self) -> None:
        for bad_path in ('/content/drive/MyDrive/closed_test_full_v1', '/content/work/open_test', 'data/test'):
            with self.subTest(bad_path=bad_path), self.assertRaisesRegex(ValueError, 'test'):
                colab_entry.reject_test_paths(bad_path)
        with self.assertRaisesRegex(ValueError, 'Google Drive'):
            colab_entry.verify_local_work_dir_not_on_drive(
                self.drive_data / 'subwork',
                self.drive_data,
                self.drive_out,
            )
        cmd = colab_entry.build_colab_runner_command(
            python_exe='python',
            plan_path=ROOT / 'configs/xlsr_training.json',
            run_id='X0',
            training_seed=2027,
            device='cuda:0',
            expected_gpu='L4',
            train_artifact=self.work_dir / 'train_development_20k_v2',
            dev_artifact=self.work_dir / 'closed_dev_full_v1',
            pretrained_dir=self.work_dir / 'wav2vec2-xls-r-300m',
            run_root=self.work_dir / 'checkpoints/xlsr_X0_seed2027',
            report_json=self.work_dir / 'reports/xlsr_X0_seed2027.json',
            session_steps=9930,
            deadline_unix=50000.0,
            resume=False,
        )
        self.assertTrue(all('test' not in part.lower() for part in cmd))

    def test_colab_notebook_contains_only_mount_install_and_entrypoint_cells(self) -> None:
        nb_path = ROOT / 'colab/run_xlsr.ipynb'
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
        self.assertEqual(nb.get('nbformat'), 4)
        cells = nb.get('cells', [])
        self.assertEqual(len(cells), 3)
        sources = [''.join(c.get('source', [])) for c in cells]
        self.assertIn('drive.mount', sources[0])
        self.assertIn('transformers==4.57.6', sources[1])
        self.assertIn('numpy==2.1.3', sources[1])
        self.assertIn('colab/xlsr_colab_entry.py', sources[2])
        for src in sources:
            self.assertNotIn('closed_test', src)
            self.assertNotIn('open_test', src)


if __name__ == '__main__':
    unittest.main()
