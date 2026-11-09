from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import IO, Any
from unittest.mock import patch

from scripts.summarize_b0_v2 import render_markdown, summarize_reports


def make_report(seed: int, eer: float) -> dict:
    generation = f"g-{seed}"
    return {
        "status": "early_stopped", "run_mode": "full_training", "training_seed": seed,
        "data_seed": 2026, "backend": "packed", "environment": {"device": "cpu"},
        "train_samples": 15885, "development_samples": 38777,
        "extension_history": [{"old_max_epochs": 3, "new_max_epochs": 10,
                               "old_patience": 1, "new_patience": 2}],
        "checkpoints": {"best": {"generation": generation}, "latest": {"generation": "latest"}},
        "controller": {
            "stop_reason": "early_stopped", "global_step": 10, "completed_epochs": 5,
            "bad_evaluations": 2,
            "history": [{"epoch": n, "global_step": 2 * n, "eer": value}
                        for n, value in enumerate((0.5, 0.4, eer, 0.4, 0.5), 1)],
            "best": {"generation": generation, "epoch": 3, "global_step": 6,
                     "development": {"sample_count": 38777, "eer": eer,
                                     "roc_auc": 1 - eer, "macro_f1": 0.6 + eer,
                                     "decision_threshold": eer / 10}},
        },
    }


class SummarizeB0V2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reports = {seed: make_report(seed, eer)
                        for seed, eer in zip((2026, 2027, 2028), (0.1, 0.2, 0.3))}
        self.write_reports()

    def write_reports(self) -> None:
        for seed, report in self.reports.items():
            (self.root / f"b0_v2_ext_seed{seed}.json").write_text(json.dumps(report), encoding="utf-8")

    def test_mean_and_sample_standard_deviation(self) -> None:
        summary = summarize_reports(self.root)
        for metric, mean in (("eer", 0.2), ("roc_auc", 0.8), ("macro_f1", 0.8)):
            with self.subTest(metric=metric):
                self.assertAlmostEqual(summary["aggregate"][metric]["mean"], mean)
                self.assertAlmostEqual(summary["aggregate"][metric]["sample_std"], 0.1)
                self.assertEqual(summary["aggregate"][metric]["ddof"], 1)

    def test_only_reads_three_reports_and_hashes_exact_input_bytes(self) -> None:
        original = Path.read_bytes
        allowed = {self.root / f"b0_v2_ext_seed{seed}.json" for seed in self.reports}
        opened: list[Path] = []
        original_open = Path.open

        def guarded_open(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
            self.assertIn(path, allowed, f"Đọc ngoài ba report: {path}")
            opened.append(path)
            return original_open(path, *args, **kwargs)

        # An unrelated file must never be discovered or loaded.
        (self.root / "unrelated.json").write_text("không đọc", encoding="utf-8")
        with patch.object(Path, "open", guarded_open):
            summary = summarize_reports(self.root)
        self.assertCountEqual(opened, allowed)
        for row in summary["seeds"]:
            path = self.root / Path(row["input_report"]).name
            self.assertEqual(row["input_sha256"], hashlib.sha256(original(path)).hexdigest())

    def test_rejects_unfinished_status(self) -> None:
        for status in ("paused", "complete", "running", None):
            with self.subTest(status=status):
                self.reports[2026]["status"] = status
                self.write_reports()
                with self.assertRaisesRegex(ValueError, "status"):
                    summarize_reports(self.root)

    def test_accepts_terminal_max_epochs(self) -> None:
        self.reports[2026]["status"] = "max_epochs"
        self.reports[2026]["controller"]["stop_reason"] = "max_epochs"
        self.write_reports()
        self.assertEqual(summarize_reports(self.root)["seeds"][0]["status"], "max_epochs")

    def test_rejects_test_fields_even_nested_or_false(self) -> None:
        for key in ("test", "test_samples", "allow_test", "use_test", "testFull", "testMetrics"):
            with self.subTest(key=key):
                self.reports[2028]["unexpected"] = [{key: False}]
                self.write_reports()
                with self.assertRaisesRegex(ValueError, "liên quan tới test"):
                    summarize_reports(self.root)

    def test_rejects_wrong_training_seed(self) -> None:
        self.reports[2028]["training_seed"] = 2026
        self.write_reports()
        with self.assertRaisesRegex(ValueError, "identity"):
            summarize_reports(self.root)

    def test_rejects_intermediate_three_epoch_report(self) -> None:
        self.reports[2027]["extension_history"] = []
        self.write_reports()
        with self.assertRaisesRegex(ValueError, "nối dài"):
            summarize_reports(self.root)

    def test_rejects_incomplete_development(self) -> None:
        self.reports[2027]["development_samples"] = 4115
        self.write_reports()
        with self.assertRaisesRegex(ValueError, "số mẫu"):
            summarize_reports(self.root)

    def test_rejects_nonfinite_metrics(self) -> None:
        self.reports[2026]["controller"]["best"]["development"]["roc_auc"] = float("nan")
        self.write_reports()
        with self.assertRaisesRegex(ValueError, "miền giá trị"):
            summarize_reports(self.root)

    def test_rejects_best_inconsistent_with_history(self) -> None:
        self.reports[2026]["controller"]["best"]["epoch"] = 2
        self.write_reports()
        with self.assertRaisesRegex(ValueError, "best không khớp"):
            summarize_reports(self.root)

    def test_markdown_keeps_scope_and_per_seed_thresholds(self) -> None:
        summary = summarize_reports(self.root)
        markdown = render_markdown(summary)
        for phrase in ("dev-full, không phải test", "không gộp threshold", "trần 3 epoch", "n−1"):
            self.assertIn(phrase, markdown)
        self.assertEqual([row["best"]["decision_threshold"] for row in summary["seeds"]],
                         [0.01, 0.02, 0.03])


if __name__ == "__main__":
    unittest.main()
