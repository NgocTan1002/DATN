"""Đối chứng vòng đời và lỗi commit/resume của runner D023."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from scripts.train_b0 import _set_seed
from scripts.train_b0_full import validate_plan
from scripts.check_b0_full_smoke import compare_runs
from src.training import ResumableBatchSampler
from src.training.full_training import GenerationStore, run_training


ROOT = Path(__file__).resolve().parents[1]


class FullTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_fixture(
        self, name: str, *, resume: bool = False, limit: int = 99,
        curve: tuple[float, ...] = (0.3, 0.2, 0.1),
        fault_stage: str | None = None, identity: str = "fixture",
    ) -> dict[str, Any]:
        _set_seed(2026)
        model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.4), torch.nn.Linear(4, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sampler = ResumableBatchSampler(dataset_size=5, batch_size=2, seed=2026)
        store = GenerationStore(self.root / name)

        def fault(stage: str) -> None:
            if stage == fault_stage:
                raise RuntimeError("Lỗi mô phỏng tại " + stage)

        if fault_stage in {"before_pointer", "after_pointer"}:
            def pointer_fault(stage: str) -> None:
                if not (store.root / "LATEST.json").exists():
                    return
                if stage == fault_stage and len(store.load()["controller"]["history"]) == 0:
                    # Chỉ lỗi transaction kết quả evaluation, không lỗi khởi tạo.
                    staged = sorted(store.root.glob("g-*/state.json"))
                    if any(json.loads(p.read_text())["completed_epochs"] > 0 for p in staged):
                        fault(stage)
            store.fault = pointer_fault

        def step(indices: list[int], epoch: int) -> float:
            model.train()
            x = torch.tensor([[i / 5, (i + epoch) / 7] for i in indices])
            y = torch.tensor([[float(i % 2)] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        def evaluate() -> dict[str, Any]:
            fault("during_evaluation")
            i = len(store.load()["controller"]["history"])
            return {"eer": curve[i], "loss": 0.5, "eer_threshold": 0.5, "decision_threshold": 0.5, "roc_auc": 0.8, "macro_f1": 0.8, "sample_count": 2, "labels": (0, 1), "scores": (0.1, 0.9)}

        run_identity = {"id": identity, "mode": "technical_smoke", "train_indices": list(range(5)), "development_indices": [0, 1], "source_tree_sha256": "fixture"}
        result = run_training(model=model, optimizer=optimizer, sampler=sampler, identity=run_identity, store=store, train_step=step, evaluate=evaluate, max_epochs=3, patience=1, development_samples=2, session_steps=limit, checkpoint_interval=2, resume=resume, fault=fault)
        return {"result": result, "payload": store.load()}

    def assert_same(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        for key, value in a["payload"]["model"].items():
            self.assertTrue(torch.equal(value, b["payload"]["model"][key]), key)
        left = copy.deepcopy(a["payload"]["controller"])
        right = copy.deepcopy(b["payload"]["controller"])
        left["best"].pop("generation")
        right["best"].pop("generation")
        self.assertEqual(left, right)

    def test_resume_mid_epoch_matches_loss_batch_trace_weights_and_best(self) -> None:
        expected = self.run_fixture("continuous")
        partial = self.run_fixture("resume", limit=1)
        self.assertEqual(partial["result"]["status"], "paused")
        self.assert_same(expected, self.run_fixture("resume", resume=True))
        self.assertEqual(expected["result"]["status"], "max_epochs")
        self.assertEqual(len(expected["payload"]["controller"]["history"]), 3)
        self.assertEqual(sum(map(len, expected["payload"]["controller"]["batch_trace"])), 15)

    def test_epoch_boundary_session_completes_evaluation_before_pause(self) -> None:
        paused = self.run_fixture("boundary", limit=3)
        state = paused["payload"]["controller"]
        self.assertEqual(state["completed_epochs"], 1)
        self.assertFalse(state["needs_evaluation"])
        self.assert_same(self.run_fixture("continuous"), self.run_fixture("boundary", resume=True))

    def test_equal_eer_keeps_earliest_best_and_stops_after_epoch_two(self) -> None:
        result = self.run_fixture("tie", curve=(0.2, 0.2, 0.1))
        state = result["payload"]["controller"]
        self.assertEqual(result["result"]["status"], "early_stopped")
        self.assertEqual(state["global_step"], 6)
        self.assertEqual(state["best"]["epoch"], 1)
        self.assertEqual(state["bad_evaluations"], 1)

    def test_faults_before_or_during_evaluation_replay_once(self) -> None:
        expected = self.run_fixture("continuous")
        for stage in ("after_step", "before_evaluation", "during_evaluation"):
            with self.subTest(stage=stage):
                with self.assertRaises(RuntimeError):
                    self.run_fixture(stage, fault_stage=stage)
                self.assert_same(expected, self.run_fixture(stage, resume=True))

    def test_fault_before_evaluation_pointer_keeps_prior_commit_and_ignores_orphan(self) -> None:
        with self.assertRaises(RuntimeError):
            self.run_fixture("before", fault_stage="before_pointer")
        state = GenerationStore(self.root / "before").load()["controller"]
        self.assertTrue(state["needs_evaluation"])
        self.assertEqual(state["history"], [])
        self.assert_same(self.run_fixture("continuous"), self.run_fixture("before", resume=True))

    def test_fault_after_evaluation_pointer_does_not_duplicate_history(self) -> None:
        # Inject after pointer only for the first completed evaluation.
        original = GenerationStore.commit

        def commit(store: GenerationStore, payload: dict[str, Any], *, new_best: bool = False) -> dict[str, Any]:
            result = original(store, payload, new_best=new_best)
            if store.root.name == "after" and payload["controller"]["completed_epochs"] == 1 and new_best:
                raise RuntimeError("Lỗi mô phỏng sau commit evaluation")
            return result

        from unittest.mock import patch
        with patch.object(GenerationStore, "commit", commit):
            with self.assertRaises(RuntimeError):
                self.run_fixture("after")
        self.assert_same(self.run_fixture("continuous"), self.run_fixture("after", resume=True))

    def test_changed_identity_and_tampered_best_are_rejected(self) -> None:
        self.run_fixture("changed", limit=3)
        with self.assertRaisesRegex(ValueError, "Identity"):
            self.run_fixture("changed", resume=True, identity="khác")
        store = GenerationStore(self.root / "changed")
        pointer = json.loads((store.root / "LATEST.json").read_text())
        path = store.root / pointer["best"]["generation"] / "checkpoint.pt"
        path.write_bytes("hỏng".encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            store.load()

    def test_plan_rejects_policy_seed_budget_and_test_changes(self) -> None:
        plan = json.loads((ROOT / "configs/b0_full_training.json").read_text())
        validate_plan(plan)
        for section, key, value in (("training", "seed", 1), ("training", "max_epochs", 4), ("model", "amplitude_policy", "none"), ("data", "use_test", True)):
            with self.subTest(key=key):
                changed = copy.deepcopy(plan)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    validate_plan(changed)

    def test_independent_comparator_and_nonfinite_evaluation_guard(self) -> None:
        self.run_fixture("a")
        self.run_fixture("b", limit=1)
        with self.assertRaisesRegex(ValueError, "chưa kết thúc"):
            compare_runs(self.root / "a", self.root / "b")
        self.run_fixture("b", resume=True)
        self.assertTrue(compare_runs(self.root / "a", self.root / "b")["passed"])
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.run_fixture("nan", curve=(float("nan"), 0.2, 0.1))
        state = GenerationStore(self.root / "nan").load()["controller"]
        self.assertTrue(state["needs_evaluation"])
        self.assertEqual(state["history"], [])

    def test_controller_inconsistency_is_rejected_even_with_valid_file_hash(self) -> None:
        self.run_fixture("bad", limit=1)
        store = GenerationStore(self.root / "bad")
        payload = store.load()
        payload["controller"]["global_step"] = 2
        store.commit(payload)
        with self.assertRaisesRegex(ValueError, "không nhất quán"):
            self.run_fixture("bad", resume=True)


if __name__ == "__main__":
    unittest.main()
