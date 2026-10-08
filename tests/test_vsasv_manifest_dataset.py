from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import torch

from src.data import (
    MANIFEST_COLUMNS,
    VSASVManifestDataset,
    VSASVParquetDataset,
)


class VSASVManifestDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.manifest_csv = self.root / "development_20k_v1.csv"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_parquet(
        self,
        shard_name: str,
        rows: list[tuple[str, list[float], int, str, str]],
    ) -> None:
        target = (self.parquet_dir / shard_name).as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples ('
                'file VARCHAR, audio STRUCT("array" DOUBLE[], sampling_rate BIGINT), '
                'label VARCHAR, utt_type VARCHAR)'
            )
            for logical_file, waveform, sample_rate, speaker_id, utt_type in rows:
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    [
                        logical_file,
                        {"array": waveform, "sampling_rate": sample_rate},
                        speaker_id,
                        utt_type,
                    ],
                )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _manifest_row(
        self,
        *,
        shard: str,
        logical_file: str,
        speaker_id: str,
        split: str,
        utt_type: str,
        native_sample_rate: int = 16_000,
        binary_label: int | None = None,
    ) -> dict[str, str | int]:
        return {
            "manifest_version": "development-20k-v1",
            "shard": shard,
            "file": logical_file,
            "speaker_id": speaker_id,
            "split": split,
            "binary_label": (
                binary_label
                if binary_label is not None
                else 0 if utt_type == "bonafide" else 1
            ),
            "utt_type": utt_type,
            "native_sample_rate": native_sample_rate,
            "source_snapshot": "VSASV-HF-public-snapshot-v1",
        }

    def _write_manifest(self, rows: list[dict[str, str | int]]) -> None:
        with self.manifest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_reads_selected_split_without_building_global_index(self) -> None:
        train_shard = "train-00000-of-00432.parquet"
        dev_shard = "train-00001-of-00432.parquet"
        self._write_parquet(
            train_shard,
            [("train/real.wav", [0.1, -0.1], 16_000, "train", "bonafide")],
        )
        self._write_parquet(
            dev_shard,
            [
                (
                    "dev/fake.wav",
                    [0.2, -0.2],
                    16_000,
                    "dev",
                    "voice_conversion",
                )
            ],
        )
        self._write_manifest(
            [
                self._manifest_row(
                    shard=train_shard,
                    logical_file="train/real.wav",
                    speaker_id="train",
                    split="closed_train",
                    utt_type="bonafide",
                ),
                self._manifest_row(
                    shard=dev_shard,
                    logical_file="dev/fake.wav",
                    speaker_id="dev",
                    split="closed_dev",
                    utt_type="voice_conversion",
                ),
            ]
        )

        with patch.object(
            VSASVParquetDataset,
            "_build_local_index",
            side_effect=AssertionError("Không được quét chỉ mục Parquet toàn cục."),
        ):
            dataset = VSASVManifestDataset(
                self.manifest_csv,
                self.parquet_dir,
                split="closed_train",
                training=False,
                target_samples=4,
            )

        try:
            sample = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.total_split_rows, 1)
            self.assertEqual(dataset.local_coverage, 1.0)
            self.assertEqual(dataset.manifest_version, "development-20k-v1")
            self.assertEqual(
                dataset.source_snapshot, "VSASV-HF-public-snapshot-v1"
            )
            self.assertEqual(sample["file"], "train/real.wav")
            self.assertEqual(sample["speaker_id"], "train")
            self.assertEqual(sample["label"].item(), 0.0)
            self.assertEqual(sample["waveform"].shape, (4,))
        finally:
            dataset.close()

    def test_train_crop_is_deterministic_by_seed_and_epoch(self) -> None:
        shard = "train-00000-of-00432.parquet"
        waveform = torch.arange(100, dtype=torch.float32).tolist()
        self._write_parquet(
            shard,
            [("train/real.wav", waveform, 16_000, "train", "bonafide")],
        )
        self._write_manifest(
            [
                self._manifest_row(
                    shard=shard,
                    logical_file="train/real.wav",
                    speaker_id="train",
                    split="closed_train",
                    utt_type="bonafide",
                )
            ]
        )
        dataset = VSASVManifestDataset(
            self.manifest_csv,
            self.parquet_dir,
            split="closed_train",
            training=True,
            target_samples=16,
            seed=2026,
        )

        try:
            first = dataset[0]["waveform"]
            repeated = dataset[0]["waveform"]
            dataset.set_epoch(1)
            next_epoch = dataset[0]["waveform"]

            self.assertTrue(torch.equal(first, repeated))
            self.assertFalse(torch.equal(first, next_epoch))
        finally:
            dataset.close()

    def test_rejects_binary_label_that_disagrees_with_utt_type(self) -> None:
        shard = "train-00000-of-00432.parquet"
        self._write_parquet(
            shard,
            [("train/real.wav", [0.1], 16_000, "train", "bonafide")],
        )
        row = self._manifest_row(
            shard=shard,
            logical_file="train/real.wav",
            speaker_id="train",
            split="closed_train",
            utt_type="bonafide",
        )
        row["binary_label"] = 1
        self._write_manifest([row])

        with self.assertRaisesRegex(ValueError, "binary_label không khớp"):
            VSASVManifestDataset(
                self.manifest_csv,
                self.parquet_dir,
                split="closed_train",
                training=False,
            )

    def test_rejects_missing_shard(self) -> None:
        existing_shard = "train-00000-of-00432.parquet"
        self._write_parquet(
            existing_shard,
            [("train/real.wav", [0.1], 16_000, "train", "bonafide")],
        )
        self._write_manifest(
            [
                self._manifest_row(
                    shard="train-00431-of-00432.parquet",
                    logical_file="train/real.wav",
                    speaker_id="train",
                    split="closed_train",
                    utt_type="bonafide",
                )
            ]
        )

        with self.assertRaisesRegex(FileNotFoundError, "Không tìm thấy shard"):
            VSASVManifestDataset(
                self.manifest_csv,
                self.parquet_dir,
                split="closed_train",
                training=False,
            )

    def test_rejects_speaker_overlap_between_manifest_splits(self) -> None:
        shard = "train-00000-of-00432.parquet"
        self._write_parquet(
            shard,
            [
                ("shared/train.wav", [0.1], 16_000, "shared", "bonafide"),
                (
                    "shared/dev.wav",
                    [0.2],
                    16_000,
                    "shared",
                    "voice_conversion",
                ),
            ],
        )
        self._write_manifest(
            [
                self._manifest_row(
                    shard=shard,
                    logical_file="shared/train.wav",
                    speaker_id="shared",
                    split="closed_train",
                    utt_type="bonafide",
                ),
                self._manifest_row(
                    shard=shard,
                    logical_file="shared/dev.wav",
                    speaker_id="shared",
                    split="closed_dev",
                    utt_type="voice_conversion",
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "xuất hiện ở nhiều split"):
            VSASVManifestDataset(
                self.manifest_csv,
                self.parquet_dir,
                split="closed_train",
                training=False,
            )

    def test_rejects_native_sample_rate_mismatch_when_loading(self) -> None:
        shard = "train-00000-of-00432.parquet"
        self._write_parquet(
            shard,
            [("train/real.wav", [0.1], 16_000, "train", "bonafide")],
        )
        self._write_manifest(
            [
                self._manifest_row(
                    shard=shard,
                    logical_file="train/real.wav",
                    speaker_id="train",
                    split="closed_train",
                    utt_type="bonafide",
                    native_sample_rate=40_000,
                )
            ]
        )
        dataset = VSASVManifestDataset(
            self.manifest_csv,
            self.parquet_dir,
            split="closed_train",
            training=False,
        )

        try:
            with self.assertRaisesRegex(ValueError, "Sample rate manifest"):
                _ = dataset[0]
        finally:
            dataset.close()


if __name__ == "__main__":
    unittest.main()
