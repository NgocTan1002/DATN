from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.audit_local_storage import build_inventory, render_markdown


class LocalStorageAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.metadata_path = self.root / "metadata.csv"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_shard(
        self, name: str, rows: list[tuple[str, str, str]]
    ) -> Path:
        path = self.parquet_dir / name
        escaped_path = path.as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                "CREATE TABLE fixture(file VARCHAR, label VARCHAR, utt_type VARCHAR)"
            )
            connection.executemany("INSERT INTO fixture VALUES (?, ?, ?)", rows)
            connection.execute(
                f"COPY fixture TO '{escaped_path}' (FORMAT PARQUET)"
            )
        finally:
            connection.close()
        return path

    def test_inventory_counts_storage_and_metadata_coverage(self) -> None:
        first = self._write_shard(
            "train-00000-of-00432.parquet",
            [("a.wav", "bonafide", "bonafide"), ("b.wav", "spoof", "replay")],
        )
        second = self._write_shard(
            "train-00001-of-00432.parquet",
            [("c.wav", "spoof", "voice_conversion")],
        )
        self.metadata_path.write_text(
            "file,label,utt_type\n"
            "a.wav,bonafide,bonafide\n"
            "b.wav,spoof,replay\n"
            "c.wav,spoof,voice_conversion\n",
            encoding="utf-8",
        )

        report = build_inventory(
            self.parquet_dir,
            self.metadata_path,
            expected_shards=432,
            target_samples=[3, 6],
        )

        self.assertEqual(report["status"], "ĐẠT")
        self.assertEqual(report["local_shards"], 2)
        self.assertEqual(report["totals"]["samples"], 3)
        self.assertEqual(report["totals"]["unique_files"], 3)
        self.assertEqual(report["totals"]["metadata_exact_matches"], 3)
        self.assertEqual(
            report["totals"]["size_bytes"], first.stat().st_size + second.stat().st_size
        )
        self.assertAlmostEqual(
            report["linear_sample_projections"][1]["estimated_bytes"],
            report["totals"]["bytes_per_sample"] * 6,
        )
        self.assertIn("2 shard hiện có", report["warnings"][0])
        self.assertNotIn("Năm shard", report["warnings"][0])
        self.assertIn("train-00000-of-00432.parquet", render_markdown(report))

    def test_metadata_mismatch_marks_report_as_failed(self) -> None:
        self._write_shard(
            "train-00000-of-00432.parquet",
            [("a.wav", "bonafide", "bonafide")],
        )
        self.metadata_path.write_text(
            "file,label,utt_type\na.wav,spoof,bonafide\n", encoding="utf-8"
        )

        report = build_inventory(self.parquet_dir, self.metadata_path)

        self.assertEqual(report["status"], "KHÔNG ĐẠT")
        self.assertEqual(report["totals"]["metadata_exact_matches"], 0)
        self.assertTrue(report["hard_issues"])

    def test_missing_parquet_is_rejected(self) -> None:
        self.metadata_path.write_text(
            "file,label,utt_type\na.wav,bonafide,bonafide\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(FileNotFoundError, "Không tìm thấy shard Parquet"):
            build_inventory(self.parquet_dir, self.metadata_path)


if __name__ == "__main__":
    unittest.main()
