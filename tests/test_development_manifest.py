from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.make_development_manifest import (
    MANIFEST_COLUMNS,
    allocate_largest_remainder,
    choose_content_representatives,
    generate_development_manifest,
    stable_key,
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

    def _write_shard(
        self,
        rows: list[tuple[str, str, str]],
        waveforms: dict[str, list[float]] | None = None,
    ) -> None:
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
            values = []
            for index, (file_name, speaker, utt_type) in enumerate(rows, start=1):
                waveform = (
                    waveforms[file_name]
                    if waveforms is not None and file_name in waveforms
                    else [float(index)]
                )
                values.append(
                    (
                        file_name,
                        speaker,
                        utt_type,
                        {"array": waveform, "sampling_rate": 16000},
                    )
                )
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

    def test_content_representative_is_chosen_globally_across_splits(self) -> None:
        train_file = "train/duplicate.wav"
        dev_file = next(
            f"dev/duplicate-{index}.wav"
            for index in range(100)
            if stable_key(f"dev/duplicate-{index}.wav", 2026)
            < stable_key(train_file, 2026)
        )
        rows = [
            {
                "file": train_file,
                "split": "closed_train",
                "speaker_id": "train-speaker",
                "utt_type": "bonafide",
                "content_sha256": "same-content",
            },
            {
                "file": dev_file,
                "split": "closed_dev",
                "speaker_id": "dev-speaker",
                "utt_type": "bonafide",
                "content_sha256": "same-content",
            },
        ]

        eligible_files, groups = choose_content_representatives(rows, seed=2026)

        self.assertEqual(eligible_files, {dev_file})
        self.assertEqual(groups[0]["representative_file"], dev_file)
        self.assertEqual(groups[0]["removed_files"], [train_file])

    def test_manifest_refills_quota_without_reselecting_duplicate_content(self) -> None:
        train = [
            ("tr_dup.wav", "tr_s1", "bonafide"),
            ("tr_unique.wav", "tr_s2", "bonafide"),
            ("tr_v.wav", "tr_s3", "voice_conversion"),
            ("tr_a.wav", "tr_s4", "adversarial_attack"),
            ("tr_r.wav", "tr_s5", "replay"),
        ]
        dev = [
            ("dv_dup.wav", "dv_s1", "bonafide"),
            ("dv_unique.wav", "dv_s2", "bonafide"),
            ("dv_v.wav", "dv_s3", "voice_conversion"),
            ("dv_a.wav", "dv_s4", "adversarial_attack"),
            ("dv_r.wav", "dv_s5", "replay"),
        ]
        all_rows = train + dev
        waveforms = {
            file_name: [float(index)]
            for index, (file_name, _, _) in enumerate(all_rows, start=1)
        }
        waveforms["tr_dup.wav"] = [999.0]
        waveforms["dv_dup.wav"] = [999.0]
        self._write_csv(self.metadata_path, all_rows)
        self._write_csv(self.split_dir / "closed_train.csv", train)
        self._write_csv(self.split_dir / "closed_dev.csv", dev)
        self._write_shard(all_rows, waveforms)

        report = generate_development_manifest(
            self.parquet_dir,
            self.metadata_path,
            self.split_dir,
            self.output_path,
            target_total=8,
            seed=2026,
            manifest_version="test-v2",
        )

        self.assertTrue(report["technical_passed"])
        self.assertEqual(report["checks"]["duplicate_content_hashes"], 0)
        self.assertEqual(report["manifest"]["samples"], 8)
        with self.output_path.open("r", encoding="utf-8", newline="") as handle:
            selected = list(csv.DictReader(handle))
        selected_files = {row["file"] for row in selected}
        self.assertLessEqual(
            len({"tr_dup.wav", "dv_dup.wav"} & selected_files),
            1,
        )
        self.assertEqual(
            report["deduplication"]["selected_content_hashes"],
            8,
        )

    def test_shortfall_after_content_deduplication_is_reported(self) -> None:
        train = [
            ("tr_dup.wav", "tr_s1", "bonafide"),
            ("tr_unique.wav", "tr_s2", "bonafide"),
        ]
        dev = [
            ("dv_dup.wav", "dv_s1", "bonafide"),
            ("dv_unique.wav", "dv_s2", "bonafide"),
        ]
        all_rows = train + dev
        self._write_csv(self.metadata_path, all_rows)
        self._write_csv(self.split_dir / "closed_train.csv", train)
        self._write_csv(self.split_dir / "closed_dev.csv", dev)
        self._write_shard(
            all_rows,
            {
                "tr_dup.wav": [1.0],
                "dv_dup.wav": [1.0],
                "tr_unique.wav": [2.0],
                "dv_unique.wav": [3.0],
            },
        )

        report = generate_development_manifest(
            self.parquet_dir,
            self.metadata_path,
            self.split_dir,
            self.output_path,
            target_total=4,
            seed=2026,
            manifest_version="test-v2",
        )

        self.assertFalse(report["technical_passed"])
        self.assertEqual(report["status"], "KHÔNG ĐỦ AUDIO SAU KHỬ TRÙNG")
        self.assertTrue(report["shortfalls"])
        self.assertFalse(self.output_path.exists())

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
