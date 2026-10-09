from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock

from scripts.audio_smoke_test import (
    configure_duckdb,
    percentile_cont,
    render_markdown,
    summarize_waveform,
)


class WaveformSummaryTests(unittest.TestCase):
    def test_continuous_percentile_interpolates_between_values(self) -> None:
        self.assertEqual(percentile_cont([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile_cont([1.0, 2.0, 3.0], 0.95), 2.9)

    def test_duckdb_scan_is_configured_for_bounded_memory(self) -> None:
        connection = MagicMock()

        configure_duckdb(connection)

        self.assertEqual(
            [call.args[0] for call in connection.execute.call_args_list],
            [
                "SET threads = 1",
                "SET preserve_insertion_order = false",
                "SET memory_limit = '8GB'",
            ],
        )

    def test_regular_waveform(self) -> None:
        result = summarize_waveform([0.0, 0.5, -0.5, 1.0], sampling_rate=4)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["nonfinite_samples"], 0)
        self.assertAlmostEqual(result["duration_seconds"], 1.0)
        self.assertAlmostEqual(result["peak_absolute"], 1.0)
        self.assertAlmostEqual(result["rms"], math.sqrt(1.5 / 4.0))
        self.assertAlmostEqual(result["clipped_fraction"], 0.25)
        self.assertFalse(result["silent"])

    def test_nonfinite_values_are_counted(self) -> None:
        result = summarize_waveform([0.1, math.nan, math.inf], sampling_rate=16_000)
        self.assertEqual(result["finite_samples"], 1)
        self.assertEqual(result["nonfinite_samples"], 2)

    def test_empty_waveform_is_silent(self) -> None:
        result = summarize_waveform([], sampling_rate=16_000)
        self.assertTrue(result["silent"])
        self.assertIsNone(result["rms"])

    def test_markdown_uses_actual_shard_count(self) -> None:
        report = {
            "technical_passed": True,
            "safe_to_continue_pipeline": True,
            "local_shards": 1,
            "expected_shards": 432,
            "required_target_sample_rate": 16_000,
            "summary": {
                "utterances": 2,
                "unique_files": 2,
                "speakers": 1,
                "missing_file": 0,
                "missing_label": 0,
                "missing_utt_type": 0,
                "missing_waveform": 0,
                "invalid_sampling_rate": 0,
                "empty_waveform": 0,
            },
            "distribution": [],
            "duration_by_type": [],
            "sampled_waveforms": [],
            "warnings": [
                "1 shard chỉ là một phần nhỏ, không ngẫu nhiên của 432 shard."
            ],
        }

        markdown = render_markdown(report)

        self.assertIn("Speaker trong 1 shard", markdown)
        self.assertIn("Kiểm tra toàn bộ 1 shard", markdown)
        self.assertIn("kết quả trên 1 shard", markdown)
        self.assertNotIn("năm shard", markdown.lower())


if __name__ == "__main__":
    unittest.main()
