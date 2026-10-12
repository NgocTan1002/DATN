from __future__ import annotations

import argparse
import unittest
from pathlib import Path

import torch

from scripts.benchmark_manifest_loaders import (
    align_common_records,
    calculate_improvement_percent,
    calculate_time_reduction_percent,
    select_benchmark_files,
    summarize_runs,
    validate_protocol,
    validate_sample_equivalence,
)
from src.data.vsasv import VSASVRecord


class ManifestLoaderBenchmarkTests(unittest.TestCase):
    def test_selection_is_deterministic_and_order_independent(self) -> None:
        rows = [
            {"file": "b.wav", "split": "closed_train"},
            {"file": "dev.wav", "split": "closed_dev"},
            {"file": "a.wav", "split": "closed_train"},
            {"file": "c.wav", "split": "closed_train"},
        ]

        first = select_benchmark_files(rows, samples=2, seed=2026)
        second = select_benchmark_files(reversed(rows), samples=2, seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertNotIn("dev.wav", first)

    def test_selection_rejects_duplicate_and_insufficient_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "file bị lặp"):
            select_benchmark_files(
                [
                    {"file": "a.wav", "split": "closed_train"},
                    {"file": "a.wav", "split": "closed_train"},
                ],
                samples=1,
                seed=2026,
            )

        with self.assertRaisesRegex(ValueError, "Không đủ file"):
            select_benchmark_files(
                [{"file": "a.wav", "split": "closed_train"}],
                samples=2,
                seed=2026,
            )

    def test_common_records_are_aligned_and_metadata_checked(self) -> None:
        legacy = [
            VSASVRecord("b.wav", "s2", "bonafide", Path("s2.parquet")),
            VSASVRecord("a.wav", "s1", "voice_conversion", Path("s1.parquet")),
        ]
        manifest = [
            VSASVRecord(
                "a.wav", "s1", "voice_conversion", Path("s1.parquet"), 16_000
            ),
            VSASVRecord("b.wav", "s2", "bonafide", Path("s2.parquet"), 16_000),
        ]

        aligned_legacy, aligned_manifest = align_common_records(
            legacy,
            manifest,
            ["a.wav", "b.wav"],
        )

        self.assertEqual([row.file for row in aligned_legacy], ["a.wav", "b.wav"])
        self.assertEqual(
            [row.file for row in aligned_manifest], ["a.wav", "b.wav"]
        )

        broken = [
            VSASVRecord("a.wav", "wrong", "voice_conversion", Path("s1.parquet")),
            manifest[1],
        ]
        with self.assertRaisesRegex(ValueError, "Metadata không tương đương"):
            align_common_records(legacy, broken, ["a.wav", "b.wav"])

    def test_sample_equivalence_checks_metadata_and_waveform(self) -> None:
        waveform = torch.tensor([0.1, -0.2], dtype=torch.float32)
        first = {
            "file": "a.wav",
            "speaker_id": "s1",
            "utt_type": "bonafide",
            "label": torch.tensor(0.0),
            "waveform": waveform,
        }
        second = {**first, "waveform": waveform.clone()}

        reference = validate_sample_equivalence(first, second)

        self.assertEqual(reference["file"], "a.wav")
        self.assertTrue(reference["waveform_sha256"])
        with self.assertRaisesRegex(RuntimeError, "Waveform không tương đương"):
            validate_sample_equivalence(
                first,
                {**second, "waveform": torch.tensor([0.1, -0.3])},
            )

    def test_run_summary_and_improvement_are_exact(self) -> None:
        summary = summarize_runs(
            [
                {
                    "seconds": 2.0,
                    "peak_rss_mib": 100.0,
                    "startup_seconds": 0.1,
                    "first_batch_seconds": 0.5,
                },
                {
                    "seconds": 4.0,
                    "peak_rss_mib": 120.0,
                    "startup_seconds": 0.2,
                    "first_batch_seconds": 0.7,
                },
            ],
            samples_per_run=8,
        )

        self.assertAlmostEqual(summary["throughput_samples_per_second"], 16 / 6)
        self.assertEqual(summary["peak_rss_mib"], 120.0)
        self.assertEqual(summary["run_seconds"]["median_ms"], 3000.0)
        self.assertAlmostEqual(calculate_improvement_percent(2.0, 2.5), 25.0)
        self.assertAlmostEqual(calculate_time_reduction_percent(4.0, 1.0), 75.0)
        with self.assertRaisesRegex(ValueError, "lớn hơn 0"):
            calculate_improvement_percent(0.0, 2.5)

    def test_protocol_rejects_invalid_repeats_batch_and_workers(self) -> None:
        args = argparse.Namespace(
            samples=16,
            repeats=2,
            batch_size=8,
            workers=[0, 2],
            minimum_improvement_percent=15.0,
        )
        with self.assertRaisesRegex(ValueError, "ít nhất bằng 3"):
            validate_protocol(args)

        args.repeats = 3
        args.batch_size = 0
        with self.assertRaisesRegex(ValueError, "batch-size phải lớn hơn 0"):
            validate_protocol(args)

        args.batch_size = 8
        args.workers = [0, -1]
        with self.assertRaisesRegex(ValueError, "không âm"):
            validate_protocol(args)


if __name__ == "__main__":
    unittest.main()
