from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.verify_vsasv_snapshot import (
    configure_duckdb,
    quantiles,
    render_markdown,
    summarize_duplicate_fingerprints,
    summarize_manifest_duplicate_groups,
    waveform_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VerifyVSASVSnapshotReportTests(unittest.TestCase):
    def test_manifest_duplicate_summary_counts_cross_split_groups(self) -> None:
        duplicate_groups = [
            {
                "cryptographically_confirmed": True,
                "members": [{"file": "a.wav"}, {"file": "b.wav"}],
            },
            {
                "cryptographically_confirmed": True,
                "members": [{"file": "c.wav"}, {"file": "d.wav"}],
            },
        ]

        summary = summarize_manifest_duplicate_groups(
            duplicate_groups,
            {
                "a.wav": "closed_train",
                "b.wav": "closed_dev",
                "c.wav": "closed_train",
                "d.wav": "closed_train",
            },
        )

        self.assertEqual(summary["groups_with_multiple_members"], 2)
        self.assertEqual(summary["groups_crossing_splits"], 1)
        self.assertEqual(summary["files_in_duplicate_groups"], 4)

    def test_waveform_sha256_covers_samples_and_sample_rate(self) -> None:
        first = waveform_sha256([0.1, -0.2, 0.3], 16_000)
        second = waveform_sha256([0.1, -0.2, 0.3], 16_000)

        self.assertEqual(first, second)
        self.assertNotEqual(first, waveform_sha256([0.1, -0.2, 0.4], 16_000))
        self.assertNotEqual(first, waveform_sha256([0.1, -0.2, 0.3], 40_000))

    def test_duplicate_summary_detects_cross_split_and_mixed_label_group(self) -> None:
        groups = summarize_duplicate_fingerprints(
            {
                123: [
                    {
                        "file": "id00001/00001.wav",
                        "speaker_id": "id00001",
                        "utt_type": "bonafide",
                        "sampling_rate": "16000",
                        "shard": "train-00000.parquet",
                        "closed_split": "closed_train",
                    },
                    {
                        "file": "id00002/replay/id00002_replay_00001.wav",
                        "speaker_id": "id00002",
                        "utt_type": "replay",
                        "sampling_rate": "16000",
                        "shard": "train-00001.parquet",
                        "closed_split": "closed_dev",
                    },
                ]
            }
        )

        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["crosses_speakers"])
        self.assertTrue(groups[0]["crosses_closed_splits"])
        self.assertTrue(groups[0]["mixes_binary_labels"])

    def test_quantiles_support_waveform_durations(self) -> None:
        self.assertEqual(
            quantiles([1.0, 2.0, 3.0, 4.0]),
            {"minimum": 1.0, "median": 2.5, "maximum": 4.0},
        )

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

    def test_markdown_uses_current_local_parquet_counts(self) -> None:
        report_path = PROJECT_ROOT / "reports" / "vsasv_snapshot_verification.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["warnings"] = []
        report["technical_consistency_passed"] = False
        report["parquet"].update(
            {
                "local_shards": 68,
                "coverage_by_shard_count_percent": round(68 / 432 * 100, 4),
                "rows": 34_782,
                "metadata_exact_matches": 34_782,
                "hard_issues": ["Waveform trùng qua train/development."],
            }
        )

        markdown = render_markdown(report)

        self.assertIn("## Parquet cục bộ (68 shard)", markdown)
        self.assertIn("toàn bộ 34,782 file cục bộ", markdown)
        self.assertIn("hiện có 68/432 shard", markdown)
        self.assertIn("### Hard issue", markdown)
        self.assertIn("Waveform trùng qua train/development.", markdown)
        self.assertIn("Trạng thái development manifest được báo riêng", markdown)
        self.assertIn(
            "development manifest không giữ nhiều file trong cùng nhóm", markdown
        )
        self.assertIn(
            "chỉ khi manifest đó vượt audit content hash độc lập", markdown
        )
        self.assertNotIn("hoặc development manifest cho kết quả khoa học", markdown)
        self.assertNotIn("năm shard", markdown.casefold())
        self.assertNotIn("5/432", markdown)
        self.assertNotIn("2.558", markdown)


if __name__ == "__main__":
    unittest.main()
