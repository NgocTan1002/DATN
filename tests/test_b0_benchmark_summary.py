from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "summarize_b0_benchmark.py"
SPEC = importlib.util.spec_from_file_location("summarize_b0_benchmark", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Không thể nạp scripts/summarize_b0_benchmark.py")
summary_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_module
SPEC.loader.exec_module(summary_module)


def benchmark_report(
    *, training_seconds: float, development_seconds: float
) -> dict[str, object]:
    return {
        "status": "complete",
        "run_mode": "technical_smoke",
        "identity": {
            "manifest_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "sample_selection_sha256": "c" * 64,
            "seed": 2026,
            "run_mode": "technical_smoke",
            "amplitude_policy": "none",
            "batch_size": 8,
            "optimizer_name": "Adam",
            "learning_rate": 0.001,
            "step_budget": 32,
        },
        "inputs": {
            "limit_selection": "balanced_hash",
            "train_sample_count": 256,
            "development_sample_count": 128,
            "sample_selection_sha256": "c" * 64,
        },
        "training": {
            "global_step": 32,
            "step_budget": 32,
            "steps_this_run": 32,
        },
        "development": {"sample_count": 128},
        "environment": {"device": "cuda", "gpu_name": "Tesla T4"},
        "resources": {
            "training_loader_seconds": training_seconds * 0.8,
            "training_model_seconds": training_seconds * 0.2,
            "development_loader_seconds": development_seconds * 0.9,
            "development_model_seconds": development_seconds * 0.1,
        },
    }


class B0BenchmarkSummaryTests(unittest.TestCase):
    def test_summary_uses_worst_repeat_and_rounds_budget_down(self) -> None:
        reports = [
            benchmark_report(training_seconds=32.0, development_seconds=64.0),
            benchmark_report(training_seconds=40.0, development_seconds=80.0),
            benchmark_report(training_seconds=36.0, development_seconds=72.0),
        ]
        summary = summary_module.build_summary(
            reports,
            source_names=["r1.json", "r2.json", "r3.json"],
            full_development_samples=4115,
            pilot_count=3,
            reserve_run_equivalents=1,
            total_gpu_hours=20.0,
            step_multiple=64,
        )

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["repeat_count"], 3)
        self.assertEqual(
            summary["measurement"]["conservative_training_step_seconds"], 1.25
        )
        self.assertEqual(
            summary["measurement"]["conservative_development_sample_seconds"],
            0.625,
        )
        self.assertEqual(summary["recommendation"]["step_budget"], 12288)
        self.assertLessEqual(
            summary["recommendation"]["projected_total_gpu_hours"], 20.0
        )

    def test_summary_rejects_identity_mismatch(self) -> None:
        first = benchmark_report(training_seconds=32.0, development_seconds=64.0)
        second = deepcopy(first)
        second["identity"]["batch_size"] = 16  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "identity"):
            summary_module.build_summary(
                [first, second, deepcopy(first)],
                source_names=["r1.json", "r2.json", "r3.json"],
                full_development_samples=4115,
                pilot_count=3,
                reserve_run_equivalents=1,
                total_gpu_hours=20.0,
                step_multiple=64,
            )

    def test_cli_prints_success_with_cp1252_stdout(self) -> None:
        class StrictCp1252Stream(io.StringIO):
            def write(self, value: str) -> int:
                value.encode("cp1252")
                return super().write(value)

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            report_paths: list[Path] = []
            for index in range(3):
                path = root / f"repeat_{index + 1}.json"
                path.write_text(
                    json.dumps(
                        benchmark_report(
                            training_seconds=32.0 + index,
                            development_seconds=64.0 + index,
                        )
                    ),
                    encoding="utf-8",
                )
                report_paths.append(path)
            output_json = root / "summary.json"
            output_md = root / "summary.md"
            arguments = [
                str(MODULE_PATH),
                "--reports",
                *(str(path) for path in report_paths),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                sys, "stdout", StrictCp1252Stream()
            ):
                self.assertEqual(summary_module.main(), 0)
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_md.is_file())


if __name__ == "__main__":
    unittest.main()
