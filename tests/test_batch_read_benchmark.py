from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.benchmark_batch_read import (
    decide_candidate,
    expected_query_count,
    improvement_percent,
    select_index_sets,
    summarize_variant,
    validate_protocol,
)
from src.data.vsasv import VSASVRecord


class BatchReadBenchmarkTests(unittest.TestCase):
    def _records(self) -> list[VSASVRecord]:
        records = [
            VSASVRecord(
                file=f"speaker/main-{index}.wav",
                speaker_id="speaker",
                utt_type="bonafide",
                shard=Path("main.parquet"),
                native_sample_rate=16_000,
            )
            for index in range(8)
        ]
        records.extend(
            VSASVRecord(
                file=f"speaker/extra-{index}.wav",
                speaker_id="speaker",
                utt_type="voice_conversion",
                shard=Path(f"extra-{index}.parquet"),
                native_sample_rate=16_000,
            )
            for index in range(8)
        )
        return records

    def test_index_sets_are_deterministic_and_cover_both_shard_cases(self) -> None:
        records = self._records()

        first = select_index_sets(records, samples=8, batch_size=4, seed=2026)
        second = select_index_sets(records, samples=8, batch_size=4, seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(
            len({records[index].shard for index in first["same_shard"]}),
            1,
        )
        self.assertEqual(
            len({records[index].shard for index in first["dispersed"]}),
            4,
        )
        self.assertEqual(len(first["end_to_end"]), 8)

    def test_expected_query_count_uses_distinct_shards_per_batch(self) -> None:
        records = self._records()
        indices = [0, 1, 8, 9, 2, 3, 10, 11]

        self.assertEqual(
            expected_query_count(
                records,
                indices,
                batch_size=4,
                variant="per_item",
            ),
            8,
        )
        self.assertEqual(
            expected_query_count(
                records,
                indices,
                batch_size=4,
                variant="batch_read",
            ),
            6,
        )
        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            expected_query_count(
                records,
                indices,
                batch_size=4,
                variant="wrong",
            )

    def test_summary_and_decision_apply_all_acceptance_gates(self) -> None:
        runs = [
            {
                "seconds": seconds,
                "throughput_samples_per_second": 8.0 / seconds,
                "first_batch_seconds": seconds / 2,
                "batch_seconds": [seconds / 2, seconds / 2],
                "loader_wait_seconds": [0.1, 0.1],
                "parquet_queries": 8,
                "peak_rss_mib": 200.0,
                "rss_growth_mib": 20.0,
            }
            for seconds in (2.0, 2.1, 1.9)
        ]
        baseline = summarize_variant(runs, samples=8)
        candidate = {
            **baseline,
            "throughput_samples_per_second": (
                baseline["throughput_samples_per_second"] * 1.20
            ),
            "throughput_cv_percent": 2.0,
            "peak_rss_mib": 210.0,
        }

        accepted = decide_candidate(
            baseline,
            candidate,
            minimum_improvement_percent=15.0,
            maximum_throughput_cv_percent=15.0,
            maximum_rss_increase_mib=256.0,
            query_contract_met=True,
            output_equivalent=True,
        )
        rejected = decide_candidate(
            baseline,
            {**candidate, "throughput_samples_per_second": 4.1},
            minimum_improvement_percent=15.0,
            maximum_throughput_cv_percent=15.0,
            maximum_rss_increase_mib=256.0,
            query_contract_met=True,
            output_equivalent=True,
        )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["status"], "GIỮ BATCH-READ")
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["status"], "LOẠI BATCH-READ")
        self.assertAlmostEqual(improvement_percent(4.0, 4.6), 15.0)

    def test_protocol_requires_three_repeats_and_full_batches(self) -> None:
        args = argparse.Namespace(
            samples=24,
            repeats=3,
            batch_size=8,
            minimum_improvement_percent=15.0,
            maximum_throughput_cv_percent=15.0,
            maximum_rss_increase_mib=256.0,
        )
        validate_protocol(args)

        args.repeats = 2
        with self.assertRaisesRegex(ValueError, "ít nhất bằng 3"):
            validate_protocol(args)
        args.repeats = 3
        args.samples = 23
        with self.assertRaisesRegex(ValueError, "bội số"):
            validate_protocol(args)


if __name__ == "__main__":
    unittest.main()
