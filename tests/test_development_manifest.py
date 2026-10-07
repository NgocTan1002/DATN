from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.make_development_manifest import (
    MANIFEST_COLUMNS,
    allocate_largest_remainder,
    generate_development_manifest,
)


class DevelopmentManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.split_dir = self.root / "splits"
        self.split_dir.mkdir()
        self.metadata_path = self.root / "metadata.csv"
        self.output_path = self.root / "manifest.csv"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_csv(self, path: Path, rows: list[tuple[str, str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("file", "label", "utt_type"))
            writer.writerows(rows)

    def _write_shard(self, rows: list[tuple[str, str, str]]) -> None:
        path = self.parquet_dir / "train-00000-of-00432.parquet"
        escaped_path = path.as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                """
                CREATE TABLE fixture(
                    file VARCHAR,
                    label VARCHAR,
                    utt_type VARCHAR,
                    audio STRUCT("array" DOUBLE[], sampling_rate BIGINT)
                )
                """
            )
            values = [
                (file_name, speaker, utt_type, {"array": [0.0], "sampling_rate": 16000})
                for file_name, speaker, utt_type in rows
            ]
            connection.executemany("INSERT INTO fixture VALUES (?, ?, ?, ?)", values)
            connection.execute(f"COPY fixture TO '{escaped_path}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _create_complete_fixture(self) -> None:
        train = [
            ("tr_b1.wav", "tr_s1", "bonafide"),
            ("tr_b2.wav", "tr_s2", "bonafide"),
            ("tr_v1.wav", "tr_s3", "voice_conversion"),
            ("tr_v2.wav", "tr_s4", "voice_conversion"),
            ("tr_a1.wav", "tr_s5", "adversarial_attack"),
            ("tr_r1.wav", "tr_s6", "replay"),
        ]
        dev = [
            ("dv_b1.wav", "dv_s1", "bonafide"),
            ("dv_v1.wav", "dv_s2", "voice_conversion"),
            ("dv_a1.wav", "dv_s3", "adversarial_attack"),
            ("dv_r1.wav", "dv_s4", "replay"),
        ]
        all_rows = train + dev
        self._write_csv(self.metadata_path, all_rows)
        self._write_csv(self.split_dir / "closed_train.csv", train)
        self._write_csv(self.split_dir / "closed_dev.csv", dev)
        self._write_shard(all_rows)

    def test_largest_remainder_is_exact_and_deterministic(self) -> None:
        allocation = allocate_largest_remainder(
            7, {"a": 2, "b": 1, "c": 1}, ("a", "b", "c")
        )
        self.assertEqual(allocation, {"a": 3, "b": 2, "c": 2})
        self.assertEqual(sum(allocation.values()), 7)

    def test_manifest_is_valid_and_reproducible(self) -> None:
        self._create_complete_fixture()
        first = generate_development_manifest(
            self.parquet_dir,
            self.metadata_path,
            self.split_dir,
            self.output_path,
            target_total=10,
            seed=2026,
            manifest_version="test-v1",
        )
        first_bytes = self.output_path.read_bytes()
        second = generate_development_manifest(
            self.parquet_dir,
            self.metadata_path,
            self.split_dir,
            self.output_path,
            target_total=10,
            seed=2026,
            manifest_version="test-v1",
        )

        self.assertTrue(first["technical_passed"])
        self.assertEqual(first["manifest"]["samples"], 10)
        self.assertEqual(first["manifest"]["sha256"], second["manifest"]["sha256"])
        self.assertEqual(first_bytes, self.output_path.read_bytes())
        self.assertEqual(first["checks"]["speaker_overlap_train_dev"], 0)
        with self.output_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(tuple(rows[0]), MANIFEST_COLUMNS)
        self.assertEqual(len({row["file"] for row in rows}), 10)
        self.assertEqual({row["binary_label"] for row in rows}, {"0", "1"})
        self.assertNotIn("closed_test", {row["split"] for row in rows})

    def test_shortfall_is_reported_without_writing_manifest(self) -> None:
        train = [
            ("tr_b1.wav", "tr_s1", "bonafide"),
            ("tr_v1.wav", "tr_s2", "voice_conversion"),
        ]
        dev = [
            ("dv_b1.wav", "dv_s1", "bonafide"),
            ("dv_v1.wav", "dv_s2", "voice_conversion"),
        ]
        self._write_csv(self.metadata_path, train + dev)
        self._write_csv(self.split_dir / "closed_train.csv", train)
        self._write_csv(self.split_dir / "closed_dev.csv", dev)
        self._write_shard(train)

        report = generate_development_manifest(
            self.parquet_dir,
            self.metadata_path,
            self.split_dir,
            self.output_path,
            target_total=4,
        )

        self.assertFalse(report["technical_passed"])
        self.assertEqual(report["status"], "KHÔNG ĐỦ AUDIO")
        self.assertTrue(report["shortfalls"])
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
