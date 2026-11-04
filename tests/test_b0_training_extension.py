"""D028: nối trần epoch mà không thay đổi quỹ đạo Adam/RNG/dữ liệu."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from typing import Any
from unittest.mock import Mock, patch
import zipfile

import torch

from scripts.train_b0 import _set_seed
from scripts.train_b0_full import RunData, main, validate_plan_v2, verify_extension_source
from src.training import ResumableBatchSampler
from src.training.full_training import GenerationStore, run_training

ROOT = Path(__file__).resolve().parents[1]


class B0ExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def identity(epochs: int, patience: int) -> dict[str, Any]:
        return {
            "source_tree_sha256": "a" * 64, "training_seed": 2026, "data_seed": 2026,
            "backend": "packed", "device": "cpu", "batch_size": 2,
            "plan": {"data": {"train_samples": 5}, "training": {
                "batch_size": 2, "max_epochs": epochs, "steps_per_epoch": 3,
                "max_optimizer_steps": epochs * 3, "max_train_sample_exposures": epochs * 5,
            }, "evaluation": {"patience": patience}, "model": {"lr": .001}},
        }

    def run_fixture(
        self, name: str, *, epochs: int = 3, patience: int = 1,
        resume: bool = False, extend: bool = False, limit: int = 99,
        curve: tuple[float, ...] = (.6, .5, .4, .3, .2, .1),
        identity: dict[str, Any] | None = None,
        fault_stage: str | None = None,
    ) -> dict[str, Any]:
        _set_seed(2026)
        model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.4), torch.nn.Linear(4, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        sampler = ResumableBatchSampler(dataset_size=5, batch_size=2, seed=2026)
        store = GenerationStore(self.root / name)
        if fault_stage is not None:
            def fault(stage: str) -> None:
                if stage == fault_stage:
                    raise RuntimeError('Lỗi commit nối dài mô phỏng')
            store.fault = fault

        def step(indices: list[int], epoch: int) -> float:
            model.train()
            x = torch.tensor([[i / 5, (i + epoch) / 7] for i in indices])
            y = torch.tensor([[float(i % 2)] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward(); optimizer.step()
            return float(loss.detach())

        def evaluate() -> dict[str, Any]:
            epoch = len(store.load()["controller"]["history"])
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor([[.1, .2], [.8, .9]])).flatten()
                scores = logits.sigmoid().tolist()
                dev_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])))
            # Đường EER giảm xác định để chạm trần; score/loss dev thật từ mô hình.
            return dict(eer=curve[epoch], loss=dev_loss,
                        eer_threshold=scores[0], decision_threshold=scores[0], roc_auc=.8,
                        macro_f1=.8, sample_count=2, labels=[0, 1], scores=scores)

        result = run_training(
            model=model, optimizer=optimizer, sampler=sampler,
            identity=identity or self.identity(epochs, patience), store=store,
            train_step=step, evaluate=evaluate, max_epochs=epochs, patience=patience,
            development_samples=2, session_steps=limit, checkpoint_interval=2,
            resume=resume, extend_max_epochs=extend,
        )
        development_by_epoch = {}
        for path in store.root.glob('g-*/state.json'):
            best = json.loads(path.read_text(encoding='utf-8'))['best']
            if best is not None and best['generation'] == path.parent.name:
                development_by_epoch[best['epoch']] = best['development']
        return {"result": result, "payload": store.load(), "development_by_epoch": development_by_epoch}

    def assert_bits_equal(self, left: Any, right: Any) -> None:
        self.assertIs(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertTrue(torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_bits_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_bits_equal(a, b)
        elif isinstance(left, float):
            self.assertEqual(struct.pack('!d', left), struct.pack('!d', right))
        else:
            self.assertEqual(left, right)

    def assert_same_run(self, expected: dict[str, Any], actual: dict[str, Any]) -> None:
        for key in ("model", "optimizer", "sampler", "rng"):
            self.assert_bits_equal(expected["payload"][key], actual["payload"][key])
        a, b = (copy.deepcopy(x["payload"]["controller"]) for x in (expected, actual))
        a['best'].pop('generation'); b['best'].pop('generation')
        self.assert_bits_equal(a, b)
        self.assert_bits_equal(expected['development_by_epoch'], actual['development_by_epoch'])

    def test_three_to_six_is_bit_exact_including_adam_rng_history_batches_and_dev(self) -> None:
        expected = self.run_fixture('six', epochs=6, patience=2)
        self.run_fixture('extended')
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.root/'extended').glob('g-*/*')}
        actual = self.run_fixture('extended', epochs=6, patience=2, resume=True, extend=True)
        self.assert_same_run(expected, actual)
        self.assertEqual(actual['payload']['identity']['extension_history'][0]['old_max_epochs'], 3)
        self.assertEqual(actual['payload']['identity']['extension_history'][0]['new_max_epochs'], 6)
        self.assertEqual(actual['payload']['identity']['extension_history'][0]['old_patience'], 1)
        self.assertEqual(actual['payload']['identity']['extension_history'][0]['new_patience'], 2)
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_extension_then_ordinary_mid_epoch_resume_preserves_audit_and_exactness(self) -> None:
        expected = self.run_fixture('six', epochs=6, patience=2)
        self.run_fixture('resumed')
        paused = self.run_fixture('resumed', epochs=6, patience=2, resume=True, extend=True, limit=1)
        self.assertEqual(paused['result']['status'], 'paused')
        actual = self.run_fixture('resumed', epochs=6, patience=2, resume=True)
        self.assert_same_run(expected, actual)
        self.assertEqual(len(actual['payload']['identity']['extension_history']), 1)

    def test_extension_preserves_patience_two_and_earliest_best_on_equal_eer(self) -> None:
        curve = (.6, .5, .4, .4, .4, .1)
        expected = self.run_fixture('six', epochs=6, patience=2, curve=curve)
        self.run_fixture('extended', curve=curve)
        actual = self.run_fixture('extended', epochs=6, patience=2, curve=curve, resume=True, extend=True)
        self.assert_same_run(expected, actual)
        self.assertEqual(actual['result']['status'], 'early_stopped')
        self.assertEqual(actual['payload']['controller']['best']['epoch'], 3)

    def test_atomic_extension_can_recover_before_and_after_pointer_commit(self) -> None:
        expected = self.run_fixture('six', epochs=6, patience=2)
        for stage in ('before_pointer', 'after_pointer'):
            self.run_fixture(stage)
            with self.assertRaisesRegex(RuntimeError, 'mô phỏng'):
                self.run_fixture(stage, epochs=6, patience=2, resume=True, extend=True, fault_stage=stage)
            actual = self.run_fixture(stage, epochs=6, patience=2, resume=True, extend=stage=='before_pointer')
            self.assert_same_run(expected, actual)

    def test_rejects_not_at_cap_nonzero_patience_and_early_stopped_without_writes(self) -> None:
        self.run_fixture('paused', limit=1)
        self.run_fixture('bad_count', patience=2, curve=(.6, .5, .5))
        self.run_fixture('stopped', curve=(.6, .6, .5))
        for name in ('paused', 'bad_count', 'stopped'):
            before = (self.root/name/'LATEST.json').read_bytes()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_fixture(name, epochs=6, patience=2, resume=True, extend=True)
            self.assertEqual((self.root/name/'LATEST.json').read_bytes(), before)

    def test_rejects_any_other_identity_change_including_source_without_proof(self) -> None:
        self.run_fixture('old')
        changes = [('training_seed', 2027), ('device', 'cuda'), ('source_tree_sha256', 'b'*64), ('data_seed', 7)]
        for key, value in changes:
            current = self.identity(6, 2); current[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Identity|source'):
                self.run_fixture('old', epochs=6, patience=2, resume=True, extend=True, identity=current)
        current = self.identity(6, 2); current['plan']['model']['lr'] = .002
        with self.assertRaisesRegex(ValueError, 'Identity'):
            self.run_fixture('old', epochs=6, patience=2, resume=True, extend=True, identity=current)

    def test_rejects_smaller_equal_cap_incorrect_budget_and_missing_resume(self) -> None:
        self.run_fixture('old')
        for epochs in (2, 3):
            with self.assertRaises(ValueError):
                self.run_fixture('old', epochs=epochs, resume=True, extend=True)
        current = self.identity(6, 2); current['plan']['training']['max_optimizer_steps'] += 1
        with self.assertRaises(ValueError):
            self.run_fixture('old', epochs=6, patience=2, resume=True, extend=True, identity=current)
        with self.assertRaises(ValueError):
            self.run_fixture('fresh', epochs=6, patience=2, extend=True)
        with self.assertRaises(ValueError):
            self.run_fixture('old', epochs=6, patience=2, resume=True)
        self.run_fixture('old_patience_two', patience=2)
        with self.assertRaises(ValueError):
            self.run_fixture('old_patience_two', epochs=6, patience=1, resume=True, extend=True)

    def test_config_d028_budget_and_patience_are_locked(self) -> None:
        plan = json.loads((ROOT/'configs/b0_v2_training.json').read_text(encoding='utf-8'))
        self.assertEqual(plan['training']['max_epochs'], 10)
        self.assertEqual(plan['evaluation']['patience'], 2)
        self.assertEqual(plan['training']['max_optimizer_steps'], 19860)
        self.assertEqual(plan['training']['max_train_sample_exposures'], 158850)
        validate_plan_v2(plan)

    def test_runner_passes_plan_limits_to_engine_and_requires_explicit_resume(self) -> None:
        captured: dict[str, Any] = {}
        data = RunData(train=Mock(), dev=Mock(), train_indices=[0,1,2,3],
                       dev_indices=[0,1], training_seed=2026)
        def fake_training(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            state = {'global_step': 0, 'best': None}
            kwargs['store'].commit({'checkpoint_version':4, 'identity':kwargs['identity'], 'controller':state})
            return dict(status='paused', steps_this_invocation=0, controller=state)
        args = ['--plan', str(ROOT/'configs/b0_v2_training.json'), '--run-root', str(self.root/'cli'),
                '--report-json', str(self.root/'cli.json'), '--report-md', str(self.root/'cli.md'),
                '--device', 'cpu', '--training-seed', '2026', '--technical-smoke']
        with patch('scripts.train_b0_full._open_d027', return_value=data), \
             patch('scripts.train_b0_full.common.build_lcnn_from_config', return_value=torch.nn.Linear(2,1)), \
             patch('scripts.train_b0_full.common.build_lfcc_transform', return_value=torch.nn.Identity()), \
             patch('scripts.train_b0_full.run_training', side_effect=fake_training):
            self.assertEqual(main(args), 0)
        self.assertEqual((captured['max_epochs'], captured['patience']), (10,2))
        self.assertFalse(captured['extend_max_epochs'])
        with self.assertRaisesRegex(ValueError, 'resume'):
            main(args+['--extend-max-epochs'])
        with self.assertRaisesRegex(ValueError, 'extension-source-bundle'):
            main(args+['--extension-source-bundle', 'unused.zip'])

    def test_source_transition_checks_whole_inventory_and_limits_paths(self) -> None:
        old = [{'path': 'src/other.py', 'sha256': '1'*64},
               {'path': 'scripts/train_b0_full.py', 'sha256': '2'*64}]
        old.sort(key=lambda x: x['path'])
        previous = {'source_tree_sha256': hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest(), 'source_files': old}
        current = copy.deepcopy(old); current[0]['sha256'] = '3'*64
        transition = verify_extension_source(previous, current, None)
        self.assertEqual(transition['changed_files'], ['scripts/train_b0_full.py'])
        current[1]['sha256'] = '4'*64
        with self.assertRaisesRegex(ValueError, 'source|nguồn'):
            verify_extension_source(previous, current, None)
        previous['source_files'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source|nguồn'):
            verify_extension_source(previous, old, None)

    def test_checked_source_transition_is_recorded_and_cannot_be_forged(self) -> None:
        from src.training.full_training import extended_identity
        old_files = [{'path': 'scripts/train_b0_full.py', 'sha256': '1'*64}]
        new_files = [{'path': 'scripts/train_b0_full.py', 'sha256': '2'*64}]
        previous = self.identity(3, 1)
        previous['source_files'] = old_files
        previous['source_tree_sha256'] = hashlib.sha256(json.dumps(old_files, sort_keys=True).encode()).hexdigest()
        current = self.identity(6, 2)
        current['source_files'] = new_files
        current['source_tree_sha256'] = hashlib.sha256(json.dumps(new_files, sort_keys=True).encode()).hexdigest()
        transition = verify_extension_source(previous, new_files, None)
        result = extended_identity(previous, current, max_epochs=6, patience=2, source_transition=transition)
        self.assertEqual(result['extension_history'][0]['source_transition'], transition)
        transition['changed_files'] = []
        with self.assertRaisesRegex(ValueError, 'source'):
            extended_identity(previous, current, max_epochs=6, patience=2, source_transition=transition)

    def test_legacy_bundle_reconstructs_exact_old_source_not_an_unchecked_override(self) -> None:
        contents = {'scripts/train_b0_full.py': b'old runner', 'src/other.py': b'unchanged',
                    'configs/xlsr_pretrained_candidate.json': json.dumps({'source': {'weights_downloaded': True, 'weights_sha256': 'd'*64}}).encode(),
                    'scripts/benchmark_xlsr_cuda.py': b't14'}
        bundle = self.root/'source.zip'
        with zipfile.ZipFile(bundle, 'w') as archive:
            for name, value in contents.items():
                archive.writestr(name, value)
        legacy = copy.deepcopy(contents)
        legacy.pop('scripts/benchmark_xlsr_cuda.py')
        legacy['configs/xlsr_pretrained_candidate.json'] = (json.dumps({'source': {'weights_downloaded': False, 'weights_sha256': None}}, ensure_ascii=False, indent=2)+'\n').encode()
        old = [dict(path=k, sha256=hashlib.sha256(v).hexdigest()) for k,v in sorted(legacy.items())]
        previous = {'source_tree_sha256': hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()}
        current = [dict(path=k, sha256=hashlib.sha256(v).hexdigest()) for k,v in sorted(contents.items())]
        transition = verify_extension_source(previous, current, bundle)
        self.assertIn('scripts/benchmark_xlsr_cuda.py', transition['changed_files'])
        current[-1]['sha256'] = '9'*64
        with self.assertRaises(ValueError):
            verify_extension_source(previous, current, bundle)
        with self.assertRaises(ValueError):
            verify_extension_source(previous, current, None)


if __name__ == '__main__':
    unittest.main()
