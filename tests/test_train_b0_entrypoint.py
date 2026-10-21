from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import duckdb
import torch

from src.data import MANIFEST_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "scripts" / "train_b0.py"
AUDIO_CONFIG = PROJECT_ROOT / "configs" / "audio.json"


class TrainB0EntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.manifest = self.root / "manifest.csv"
        self.config = self.root / "lfcc_lcnn.json"
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_fixture(self) -> None:
        shard_name = "fixture.parquet"
        rows = [
            ("train/a.wav", [0.01, 0.02, -0.01], 16_000, "train-a", "bonafide"),
            (
                "train/b.wav",
                [0.2, -0.1, 0.05],
                16_000,
                "train-b",
                "voice_conversion",
            ),
            ("train/c.wav", [0.03, 0.01], 16_000, "train-c", "bonafide"),
            (
                "train/d.wav",
                [-0.2, 0.15],
                16_000,
                "train-d",
                "adversarial_attack",
            ),
            ("dev/a.wav", [0.02, 0.01], 16_000, "dev-a", "bonafide"),
            ("dev/b.wav", [0.2, -0.2], 16_000, "dev-b", "replay"),
        ]
        target = (self.parquet_dir / shard_name).as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples ('
                'file VARCHAR, audio STRUCT("array" DOUBLE[], sampling_rate BIGINT), '
                'label VARCHAR, utt_type VARCHAR)'
            )
            for logical_file, waveform, sample_rate, speaker_id, utt_type in rows:
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    [
                        logical_file,
                        {"array": waveform, "sampling_rate": sample_rate},
                        speaker_id,
                        utt_type,
                    ],
                )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for logical_file, _waveform, sample_rate, speaker_id, utt_type in rows:
                writer.writerow(
                    {
                        "manifest_version": "fixture-v1",
                        "shard": shard_name,
                        "file": logical_file,
                        "speaker_id": speaker_id,
                        "split": (
                            "closed_train"
                            if logical_file.startswith("train/")
                            else "closed_dev"
                        ),
                        "binary_label": 0 if utt_type == "bonafide" else 1,
                        "utt_type": utt_type,
                        "native_sample_rate": sample_rate,
                        "source_snapshot": "fixture-snapshot-v1",
                    }
                )

        config: dict[str, Any] = {
            "version": 1,
            "seed": 2026,
            "lfcc": {
                "sample_rate": 16_000,
                "n_filter": 8,
                "n_lfcc": 4,
                "f_min": 0.0,
                "f_max": 8_000.0,
                "dct_type": 2,
                "norm": "ortho",
                "log_lf": False,
                "spectrum": {
                    "n_fft": 512,
                    "win_length": 400,
                    "hop_length": 160,
                    "power": 2.0,
                    "center": True,
                    "pad_mode": "reflect",
                },
            },
            "lcnn_interface": {
                "input_shape_for_locked_input": ["batch", 1, 4, 401]
            },
            "lcnn": {
                "channels": [4],
                "kernels": [3],
                "pool_after_blocks": [],
                "hidden_features": 8,
                "dropout": 0.5,
            },
            "smoke_optimization": {
                "batch_size": 2,
                "learning_rate": 0.001,
            },
        }
        self.config.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )

    def _run(
        self,
        *extra: str,
        step_budget: int = 1,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ENTRYPOINT),
                "--run-mode",
                "technical_smoke",
                "--manifest",
                str(self.manifest),
                "--parquet-dir",
                str(self.parquet_dir),
                "--config",
                str(self.config),
                "--audio-config",
                str(AUDIO_CONFIG),
                "--amplitude-policy",
                "none",
                "--step-budget",
                str(step_budget),
                "--batch-size",
                "2",
                "--train-limit",
                "4",
                "--development-limit",
                "2",
                "--device",
                "cpu",
                *extra,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_writes_metrics_report_and_production_checkpoint(self) -> None:
        checkpoint = self.root / "checkpoint.pt"
        report_json = self.root / "report.json"
        report_md = self.root / "report.md"

        completed = self._run(
            "--checkpoint",
            str(checkpoint),
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        report = json.loads(report_json.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["run_mode"], "technical_smoke")
        self.assertEqual(report["training"]["global_step"], 1)
        self.assertEqual(report["development"]["sample_count"], 2)
        self.assertEqual(report["development"]["threshold_source"], "development_eer")
        self.assertEqual(report["development"]["labels"], [0, 1])
        self.assertEqual(len(report["development"]["records"]), 2)
        self.assertEqual(report["inputs"]["limit_selection"], "head")
        self.assertEqual(report["inputs"]["train_selection"]["labels"], {"0": 2, "1": 2})
        self.assertEqual(
            report["inputs"]["development_selection"]["labels"],
            {"0": 1, "1": 1},
        )
        self.assertEqual(len(report["identity"]["sample_selection_sha256"]), 64)
        self.assertIn("git_commit", report["environment"])
        for field in (
            "rss_start_mib",
            "rss_end_mib",
            "cuda_peak_allocated_mib",
            "training_loader_seconds",
            "training_model_seconds",
            "training_throughput_samples_per_second",
            "development_loader_seconds",
            "development_model_seconds",
            "development_throughput_samples_per_second",
        ):
            self.assertIn(field, report["resources"])
            self.assertTrue(math.isfinite(report["resources"][field]), msg=field)
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(report_md.is_file())

    def test_cli_resume_matches_continuous_run_exactly(self) -> None:
        continuous_checkpoint = self.root / "continuous.pt"
        continuous_report = self.root / "continuous.json"
        continuous = self._run(
            "--checkpoint",
            str(continuous_checkpoint),
            "--report-json",
            str(continuous_report),
            "--report-md",
            str(self.root / "continuous.md"),
            step_budget=2,
        )
        self.assertEqual(continuous.returncode, 0, msg=continuous.stderr)

        interrupted_checkpoint = self.root / "interrupted.pt"
        interrupted_report = self.root / "interrupted.json"
        interrupted = self._run(
            "--max-steps-this-run",
            "1",
            "--checkpoint",
            str(interrupted_checkpoint),
            "--report-json",
            str(interrupted_report),
            "--report-md",
            str(self.root / "interrupted.md"),
            step_budget=2,
        )
        self.assertEqual(interrupted.returncode, 0, msg=interrupted.stderr)
        self.assertEqual(
            json.loads(interrupted_report.read_text(encoding="utf-8"))["status"],
            "interrupted",
        )

        resumed_checkpoint = self.root / "resumed.pt"
        resumed_report = self.root / "resumed.json"
        resumed = self._run(
            "--resume-from",
            str(interrupted_checkpoint),
            "--checkpoint",
            str(resumed_checkpoint),
            "--report-json",
            str(resumed_report),
            "--report-md",
            str(self.root / "resumed.md"),
            step_budget=2,
        )
        self.assertEqual(resumed.returncode, 0, msg=resumed.stderr)

        continuous_state = torch.load(
            continuous_checkpoint, map_location="cpu", weights_only=True
        )
        resumed_state = torch.load(
            resumed_checkpoint, map_location="cpu", weights_only=True
        )
        for name, expected in continuous_state["model_state_dict"].items():
            self.assertTrue(
                torch.equal(expected, resumed_state["model_state_dict"][name]),
                msg=name,
            )
        report = json.loads(resumed_report.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["training"]["global_step"], 2)
        self.assertTrue(report["resume"]["used"])
        self.assertEqual(report["resume"]["initial_global_step"], 1)

    def test_cli_rejects_resume_with_changed_step_budget(self) -> None:
        checkpoint = self.root / "identity.pt"
        first = self._run(
            "--max-steps-this-run",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--report-json",
            str(self.root / "identity.json"),
            "--report-md",
            str(self.root / "identity.md"),
            step_budget=2,
        )
        self.assertEqual(first.returncode, 0, msg=first.stderr)

        incompatible = self._run(
            "--resume-from",
            str(checkpoint),
            "--checkpoint",
            str(self.root / "incompatible.pt"),
            "--report-json",
            str(self.root / "incompatible.json"),
            "--report-md",
            str(self.root / "incompatible.md"),
            step_budget=3,
        )
        self.assertNotEqual(incompatible.returncode, 0)
        self.assertIn("Identity checkpoint", incompatible.stderr)

    def test_cli_rejects_development_limit_without_two_classes(self) -> None:
        completed = self._run(
            "--development-limit",
            "1",
            "--checkpoint",
            str(self.root / "limit.pt"),
            "--report-json",
            str(self.root / "limit.json"),
            "--report-md",
            str(self.root / "limit.md"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("development", completed.stderr)
        self.assertIn("2", completed.stderr)

    def test_balanced_hash_selection_is_reported_and_locked_for_resume(self) -> None:
        checkpoint = self.root / "balanced.pt"
        first_report = self.root / "balanced.json"
        first = self._run(
            "--limit-selection",
            "balanced_hash",
            "--max-steps-this-run",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--report-json",
            str(first_report),
            "--report-md",
            str(self.root / "balanced.md"),
            step_budget=2,
        )
        self.assertEqual(first.returncode, 0, msg=first.stderr)
        report = json.loads(first_report.read_text(encoding="utf-8"))
        self.assertEqual(report["inputs"]["limit_selection"], "balanced_hash")
        self.assertEqual(report["inputs"]["train_selection"]["labels"], {"0": 2, "1": 2})

        incompatible = self._run(
            "--limit-selection",
            "head",
            "--resume-from",
            str(checkpoint),
            "--checkpoint",
            str(self.root / "head.pt"),
            "--report-json",
            str(self.root / "head.json"),
            "--report-md",
            str(self.root / "head.md"),
            step_budget=2,
        )
        self.assertNotEqual(incompatible.returncode, 0)
        self.assertIn("Identity checkpoint", incompatible.stderr)


if __name__ == "__main__":
    unittest.main()
