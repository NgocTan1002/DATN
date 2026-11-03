"""Runner B0 v2 (D027): packed, dev-full và seed huấn luyện tách khỏi seed dữ liệu."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.train_b0 import _sample_selection_sha256
from scripts.train_b0_full import main, validate_plan, validate_plan_v2
from src.data.packed import (
    PACKED_FORMAT_VERSION,
    PACKED_INDEX_COLUMNS,
    PACKED_PREPROCESSING_STAGE,
)
from src.training.full_training import GenerationStore


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROWS = 15_885
DEV_ROWS = 38_777


def _locked_plan() -> dict[str, Any]:
    plan = json.loads((ROOT / "configs" / "b0_v2_training.json").read_text(encoding="utf-8"))
    plan["data"]["train_artifact_waveforms_sha256"] = "a" * 64
    plan["data"]["development_artifact_waveforms_sha256"] = "b" * 64
    return plan


class _Row:
    def __init__(self, file: str) -> None:
        self.file = file


def _write_artifact(directory: Path, *, split: str, rows: int, manifest_sha256: str, offset: int) -> str:
    """Ghi artifact packed tổng hợp nhỏ (16 mẫu mỗi waveform) đúng định dạng v1."""

    directory.mkdir(parents=True)
    generator = np.random.default_rng(offset)
    waveforms = (generator.standard_normal((rows, 16)) * 0.1).astype("<f4")
    digest = hashlib.sha256()
    with (directory / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PACKED_INDEX_COLUMNS)
        writer.writeheader()
        for i in range(rows):
            block = waveforms[i].tobytes()
            digest.update(block)
            utt = "bonafide" if i % 2 == 0 else "voice_conversion"
            writer.writerow(
                {
                    "artifact_index": i, "source_index": i, "offset_samples": 16 * i,
                    "num_samples": 16, "waveform_sha256": hashlib.sha256(block).hexdigest(),
                    "file": f"{split}/spk{i % 40}/{i:05d}.wav", "speaker_id": f"{split}-spk{i % 40}",
                    "split": split, "binary_label": 0 if utt == "bonafide" else 1,
                    "utt_type": utt, "native_sample_rate": 16_000,
                    "shard": "train-00000-of-00432.parquet",
                }
            )
    (directory / "waveforms.f32").write_bytes(waveforms.tobytes())
    metadata = {
        "format_version": PACKED_FORMAT_VERSION, "preprocessing_stage": PACKED_PREPROCESSING_STAGE,
        "dtype": "float32-le", "target_sample_rate": 16_000, "crop_seed": 2026, "split": split,
        "artifact_count": rows, "total_samples": 16 * rows, "manifest_version": "fixture",
        "source_snapshot": "fixture", "source_manifest": "fixture.csv",
        "source_manifest_sha256": manifest_sha256, "index_file": "index.csv",
        "index_sha256": hashlib.sha256((directory / "index.csv").read_bytes()).hexdigest(),
        "waveforms_file": "waveforms.f32", "waveforms_sha256": digest.hexdigest(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return digest.hexdigest()


class B0V2PlanTests(unittest.TestCase):
    def test_locked_plan_is_valid_and_not_a_d023_plan(self) -> None:
        plan = _locked_plan()
        validate_plan_v2(plan)
        with self.assertRaises(ValueError):
            validate_plan(plan)
        d023 = json.loads((ROOT / "configs" / "b0_full_training.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            validate_plan_v2(d023)

    def test_committed_plan_requires_locked_artifact_checksums(self) -> None:
        plan = json.loads((ROOT / "configs" / "b0_v2_training.json").read_text(encoding="utf-8"))
        if plan["data"]["train_artifact_waveforms_sha256"] is None:
            with self.assertRaisesRegex(ValueError, "Checksum artifact"):
                validate_plan_v2(plan)
        else:
            validate_plan_v2(plan)

    def test_rejects_changes_to_seeds_data_and_protocol(self) -> None:
        changes = (
            ("training", "seed", 2026),
            ("training", "training_seeds", [2026, 2027]),
            ("training", "training_seed_scope", ["sampler"]),
            ("training", "max_epochs", 4),
            ("data", "backend", "parquet"),
            ("data", "use_test", True),
            ("data", "data_seed", 2027),
            ("data", "development_samples", 4115),
            ("data", "development_artifact_waveforms_sha256", None),
            ("model", "lfcc_features", "batch"),
            ("model", "amplitude_policy", "none"),
        )
        for section, key, value in changes:
            with self.subTest(key=key):
                plan = _locked_plan()
                plan[section][key] = value
                with self.assertRaises(ValueError):
                    validate_plan_v2(plan)


class B0V2RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        plan = _locked_plan()
        train_sha = _write_artifact(
            cls.root / "train", split="closed_train", rows=TRAIN_ROWS,
            manifest_sha256=plan["data"]["train_manifest_sha256"], offset=1,
        )
        dev_sha = _write_artifact(
            cls.root / "dev", split="closed_dev", rows=DEV_ROWS,
            manifest_sha256=plan["data"]["development_manifest_sha256"], offset=2,
        )
        plan["data"]["train_artifact_waveforms_sha256"] = train_sha
        plan["data"]["development_artifact_waveforms_sha256"] = dev_sha
        plan["data"]["sample_selection_sha256"] = _sample_selection_sha256(
            selection="head",
            train_records=[_Row(f"closed_train/spk{i % 40}/{i:05d}.wav") for i in range(TRAIN_ROWS)],
            train_indices=list(range(TRAIN_ROWS)),
            development_records=[_Row(f"closed_dev/spk{i % 40}/{i:05d}.wav") for i in range(DEV_ROWS)],
            development_indices=list(range(DEV_ROWS)),
        )
        cls.plan = cls.root / "plan.json"
        cls.plan.write_text(json.dumps(plan), encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def setUp(self) -> None:
        torch.set_num_threads(1)

    def _run(self, name: str, *extra: str) -> int:
        return main(
            [
                "--plan", str(self.plan),
                "--train-artifact", str(self.root / "train"),
                "--dev-artifact", str(self.root / "dev"),
                "--run-root", str(self.root / name),
                "--report-json", str(self.root / f"{name}.json"),
                "--report-md", str(self.root / f"{name}.md"),
                "--device", "cpu", "--technical-smoke", *extra,
            ]
        )

    def test_smoke_records_seeds_resumes_exactly_and_rejects_seed_change(self) -> None:
        self.assertEqual(self._run("continuous", "--training-seed", "2027"), 0)
        self.assertEqual(self._run("partial", "--training-seed", "2027", "--smoke-session-steps", "1"), 0)
        while json.loads((self.root / "partial.json").read_text(encoding="utf-8"))["status"] == "paused":
            self.assertEqual(
                self._run("partial", "--training-seed", "2027", "--smoke-session-steps", "1", "--resume"), 0
            )
        continuous = GenerationStore(self.root / "continuous").load()
        partial = GenerationStore(self.root / "partial").load()
        identity = continuous["identity"]
        self.assertEqual(
            (identity["training_seed"], identity["data_seed"], identity["backend"], identity["device"]),
            (2027, 2026, "packed", "cpu"),
        )
        self.assertEqual(set(identity["artifacts"]), {"train", "development"})
        for key, value in continuous["model"].items():
            self.assertTrue(torch.equal(value, partial["model"][key]), key)
        self.assertEqual(continuous["controller"]["losses"], partial["controller"]["losses"])
        self.assertEqual(continuous["controller"]["batch_trace"], partial["controller"]["batch_trace"])
        report = json.loads((self.root / "continuous.json").read_text(encoding="utf-8"))
        self.assertEqual((report["training_seed"], report["data_seed"]), (2027, 2026))
        self.assertEqual(report["development_samples"], 2)
        self.assertIn("2027/2026", (self.root / "continuous.md").read_text(encoding="utf-8"))

        with self.assertRaisesRegex(ValueError, "Identity"):
            self._run("partial", "--training-seed", "2028", "--resume")

        self.assertEqual(self._run("other_seed", "--training-seed", "2028"), 0)
        other = GenerationStore(self.root / "other_seed").load()
        self.assertNotEqual(other["controller"]["losses"], continuous["controller"]["losses"])
        initial = lambda name: torch.load(  # noqa: E731
            sorted((self.root / name).glob("g-*/checkpoint.pt"), key=lambda p: p.stat().st_mtime)[0],
            map_location="cpu", weights_only=True,
        )["model"]
        first, second = initial("continuous"), initial("other_seed")
        self.assertFalse(all(torch.equal(first[k], second[k]) for k in first))

    def test_rejects_missing_or_foreign_seed_and_parquet_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "training-seed"):
            self._run("no_seed")
        with self.assertRaisesRegex(ValueError, "training-seed"):
            self._run("bad_seed", "--training-seed", "7")
        with self.assertRaisesRegex(ValueError, "packed"):
            self._run("parquet", "--training-seed", "2026", "--manifest", "x.csv")

    def test_rejects_artifact_with_wrong_checksum(self) -> None:
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        changed = copy.deepcopy(plan)
        changed["data"]["development_artifact_waveforms_sha256"] = "c" * 64
        path = self.root / "wrong_plan.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dev-full"):
            main(
                [
                    "--plan", str(path), "--train-artifact", str(self.root / "train"),
                    "--dev-artifact", str(self.root / "dev"), "--run-root", str(self.root / "wrong"),
                    "--report-json", str(self.root / "wrong.json"), "--report-md", str(self.root / "wrong.md"),
                    "--device", "cpu", "--technical-smoke", "--training-seed", "2026",
                ]
            )


if __name__ == "__main__":
    unittest.main()
