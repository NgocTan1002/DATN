from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.audit_development_manifest_content import (
    audit_development_manifest_content,
)
from scripts.make_development_manifest import MANIFEST_COLUMNS


class DevelopmentManifestContentAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.manifest_path = self.root / "manifest.csv"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_shard(
        self,
        rows: list[tuple[str, list[float], str, str]],
    ) -> str:
        shard_name = "train-00000-of-00432.parquet"
        shard_path = self.parquet_dir / shard_name
        escaped_path = shard_path.as_posix().replace("'", "''")
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
                (
                    file_name,
                    speaker,
                    utt_type,
                    {"array": waveform, "sampling_rate": 16000},
                )
                for file_name, waveform, speaker, utt_type in rows
            ]
            connection.executemany("INSERT INTO fixture VALUES (?, ?, ?, ?)", values)
            connection.execute(f"COPY fixture TO '{escaped_path}' (FORMAT PARQUET)")
        finally:
            connection.close()
        return shard_name

    def _write_manifest(
        self,
        shard_name: str,
        rows: list[tuple[str, str, str, str]],
    ) -> None:
        with self.manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for file_name, speaker, split, utt_type in rows:
                writer.writerow(
                    {
                        "manifest_version": "test-v2",
                        "shard": shard_name,
                        "file": file_name,
                        "speaker_id": speaker,
                        "split": split,
                        "binary_label": 0 if utt_type == "bonafide" else 1,
                        "utt_type": utt_type,
                        "native_sample_rate": 16000,
                        "source_snapshot": "VSASV-HF-public-snapshot-v1",
                    }
                )

    def test_audit_detects_duplicates_within_and_across_splits(self) -> None:
        parquet_rows = [
            ("train/a.wav", [1.0], "tr1", "bonafide"),
            ("train/b.wav", [1.0], "tr2", "bonafide"),
            ("train/c.wav", [2.0], "tr3", "bonafide"),
            ("dev/d.wav", [2.0], "dv1", "bonafide"),
            ("dev/e.wav", [3.0], "dv2", "bonafide"),
        ]
        shard = self._write_shard(parquet_rows)
        self._write_manifest(
            shard,
            [
                (file_name, speaker, "closed_train" if file_name.startswith("train/") else "closed_dev", utt_type)
                for file_name, _, speaker, utt_type in parquet_rows
            ],
        )

        report = audit_development_manifest_content(
            self.manifest_path,
            self.parquet_dir,
        )

        self.assertFalse(report["technical_passed"])
        self.assertEqual(report["checks"]["duplicate_groups_total"], 2)
        self.assertEqual(report["checks"]["duplicate_groups_within_split"], 1)
        self.assertEqual(report["checks"]["duplicate_groups_crossing_splits"], 1)
        self.assertEqual(len(report["row_hashes"]), 5)

    def test_audit_passes_and_records_every_row_hash(self) -> None:
        parquet_rows = [
            ("train/a.wav", [1.0], "tr1", "bonafide"),
            ("dev/b.wav", [2.0], "dv1", "voice_conversion"),
        ]
        shard = self._write_shard(parquet_rows)
        self._write_manifest(
            shard,
            [
                ("train/a.wav", "tr1", "closed_train", "bonafide"),
                ("dev/b.wav", "dv1", "closed_dev", "voice_conversion"),
            ],
        )

        report = audit_development_manifest_content(
            self.manifest_path,
            self.parquet_dir,
        )

        self.assertTrue(report["technical_passed"])
        self.assertEqual(report["checks"]["duplicate_groups_total"], 0)
        self.assertEqual(report["checks"]["audited_files"], 2)
        self.assertEqual(len(report["row_hashes"]), 2)
        self.assertTrue(all(row["waveform_sha256"] for row in report["row_hashes"]))


if __name__ == "__main__":
    unittest.main()
