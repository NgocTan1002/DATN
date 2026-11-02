"""Kiểm tra chuyển checkpoint qua mount mà không làm đổi trạng thái huấn luyện."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from deploy.kaggle.resume_checkpoint import MANIFEST_NAME, build_payload, digest, restore_payload
from scripts.check_b0_full_smoke import equal_nested
from src.training.full_training import GenerationStore
from src.training.full_training import run_training
from src.training.sampler import ResumableBatchSampler


class ResumeTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.mount = self.root / "input" / "datasets" / "owner" / "resume"
        self.destination = self.root / "working" / "run"
        self.report = self.root / "report.json"
        store = GenerationStore(self.source)
        payload: dict[str, Any] = {
            "checkpoint_version": 4,
            "identity": {"source_tree_sha256": "fixture", "mode": "technical_smoke"},
            "model": {"weights": torch.tensor([1.0, 2.0])},
            "optimizer": {"step": torch.tensor(512)},
            "sampler": {"batch_cursor": 512},
            "rng": {"torch": torch.get_rng_state()},
            "controller": {"global_step": 512, "best": None},
        }
        store.commit(payload)
        store.commit(payload)
        build_payload(self.source, self.mount, "owner/private-resume")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def restore(self, expected_step: int = 512, source_sha: str = "fixture") -> dict[str, Any]:
        return restore_payload(self.root / "input", self.destination,
                               digest(self.mount / MANIFEST_NAME), expected_step,
                               source_sha, self.report)

    def test_nested_mount_preserves_entire_checkpoint_and_generations(self) -> None:
        result = self.restore()
        self.assertTrue(result["passed"])
        self.assertEqual(result["files_verified"], 5)
        self.assertTrue(equal_nested(GenerationStore(self.source).load(), GenerationStore(self.destination).load()))
        self.assertEqual(sorted(p.relative_to(self.source).as_posix() for p in self.source.rglob("*") if p.is_file()),
                         sorted(p.relative_to(self.destination).as_posix() for p in self.destination.rglob("*") if p.is_file()))

    def test_corrupt_checkpoint_rejected_before_publishing_run(self) -> None:
        next(self.mount.glob("*__checkpoint.pt")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.restore()
        self.assertFalse(self.destination.exists())

    def test_manifest_pin_wrong_step_and_source_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Checksum manifest"):
            restore_payload(self.root / "input", self.destination, "0" * 64, 512, "fixture", self.report)
        for step, sha in [(0, "fixture"), (512, "changed-source")]:
            with self.subTest(step=step, sha=sha), self.assertRaisesRegex(ValueError, "không khớp"):
                self.restore(step, sha)
        self.assertFalse(self.destination.exists())

    def test_traversal_duplicate_paths_and_bad_identity_rejected(self) -> None:
        path = self.mount / MANIFEST_NAME
        original = json.loads(path.read_text(encoding="utf-8"))
        import copy
        for mutation in ("traversal", "duplicate", "identity"):
            value = copy.deepcopy(original)
            if mutation == "traversal":
                value["files"][0]["path"] = "../outside"
            elif mutation == "duplicate":
                value["files"].append(value["files"][0])
            else:
                value["identity_sha256"] = "wrong"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.restore()
            self.assertFalse(self.destination.exists())

    def test_ambiguous_mount_and_existing_destination_rejected(self) -> None:
        other = self.root / "input" / MANIFEST_NAME
        other.write_bytes((self.mount / MANIFEST_NAME).read_bytes())
        with self.assertRaisesRegex(ValueError, "duy nhất"):
            self.restore()
        other.unlink()
        self.destination.mkdir(parents=True)
        sentinel = self.destination / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "từ chối ghi đè"):
            self.restore()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_best_generation_kept_when_latest_advances(self) -> None:
        store = GenerationStore(self.source)
        payload = store.load()
        payload["controller"]["best"] = {"eer": 0.2}
        payload = store.commit(payload, new_best=True)
        payload["controller"]["global_step"] = 513
        store.commit(payload)
        mount = self.root / "best-input"
        build_payload(self.source, mount, "owner/private-resume")
        restore_payload(mount, self.destination, digest(mount / MANIFEST_NAME), 513, "fixture", self.report)
        self.assertTrue(equal_nested(store.load(), GenerationStore(self.destination).load()))
        pointer = json.loads((self.destination / "LATEST.json").read_text(encoding="utf-8"))
        self.assertNotEqual(pointer["latest"]["generation"], pointer["best"]["generation"])

    def test_training_after_transfer_matches_uninterrupted_training(self) -> None:
        def train(root: Path, limit: int, resume: bool = False) -> dict[str, Any]:
            torch.manual_seed(2026)
            model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(0.2))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            sampler = ResumableBatchSampler(dataset_size=8, batch_size=2, seed=2026)

            def step(indices: list[int], epoch: int) -> float:
                optimizer.zero_grad(set_to_none=True)
                loss = model(torch.tensor([[i / 8, (i + epoch) / 8] for i in indices])).square().mean()
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            run_training(model=model, optimizer=optimizer, sampler=sampler,
                         identity={"source_tree_sha256": "fixture"}, store=GenerationStore(root),
                         train_step=step, evaluate=lambda: {"eer": 0.2, "sample_count": 2},
                         max_epochs=3, patience=1, development_samples=2,
                         session_steps=limit, checkpoint_interval=1, resume=resume)
            return GenerationStore(root).load()

        expected = train(self.root / "continuous", 3)
        source = self.root / "partial"
        train(source, 1)
        mount = self.root / "training-input"
        build_payload(source, mount, "owner/private-resume")
        restore_payload(mount, self.destination, digest(mount / MANIFEST_NAME), 1, "fixture", self.report)
        actual = train(self.destination, 2, resume=True)
        self.assertTrue(equal_nested(expected, actual))


if __name__ == "__main__":
    unittest.main()
