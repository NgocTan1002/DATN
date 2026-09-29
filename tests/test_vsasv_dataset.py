from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import duckdb
import torch

from src.data import VSASVParquetDataset


class VSASVParquetDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.split_csv = self.root / "closed_train.csv"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_parquet(self, rows: list[tuple[str, list[float], int, str, str]]) -> None:
        target = (self.parquet_dir / "fixture.parquet").as_posix().replace("'", "''")
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

    def _write_split(self, rows: list[tuple[str, str, str]]) -> None:
        with self.split_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "label", "utt_type"])
            writer.writerows(rows)

    def test_intersects_split_and_derives_binary_labels(self) -> None:
        self._write_parquet(
            [
                ("speaker1/real.wav", [0.1, -0.1, 0.2, -0.2], 16_000, "speaker1", "bonafide"),
                ("speaker2/fake.wav", [0.3, -0.3], 16_000, "speaker2", "voice_conversion"),
            ]
        )
        self._write_split(
            [
                ("speaker1/real.wav", "speaker1", "bonafide"),
                ("speaker2/fake.wav", "speaker2", "voice_conversion"),
                ("speaker3/missing.wav", "speaker3", "bonafide"),
            ]
        )

        dataset = VSASVParquetDataset(
            self.split_csv,
            self.parquet_dir,
            training=False,
            target_samples=8,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.total_split_rows, 3)
        self.assertAlmostEqual(dataset.local_coverage, 2 / 3)
        self.assertEqual(dataset[0]["waveform"].shape, (8,))
        self.assertEqual(dataset[0]["label"].item(), 0.0)
        self.assertEqual(dataset[1]["label"].item(), 1.0)
        self.assertEqual(dataset[1]["speaker_id"], "speaker2")

    def test_resamples_and_returns_finite_float32_waveform(self) -> None:
        source = torch.linspace(-0.5, 0.5, 400).tolist()
        self._write_parquet(
            [("speaker1/fake.wav", source, 40_000, "speaker1", "adversarial_attack")]
        )
        self._write_split(
            [("speaker1/fake.wav", "speaker1", "adversarial_attack")]
        )

        dataset = VSASVParquetDataset(
            self.split_csv,
            self.parquet_dir,
            training=False,
            target_sample_rate=16_000,
            target_samples=160,
        )
        sample = dataset[0]

        self.assertEqual(sample["waveform"].dtype, torch.float32)
        self.assertEqual(sample["waveform"].shape, (160,))
        self.assertTrue(torch.isfinite(sample["waveform"]).all())
        self.assertEqual(sample["native_sample_rate"], 40_000)
        self.assertEqual(sample["sample_rate"], 16_000)

    def test_evaluation_center_crop_is_deterministic(self) -> None:
        waveform = [float(value) for value in range(10)]
        self._write_parquet(
            [("speaker1/real.wav", waveform, 16_000, "speaker1", "bonafide")]
        )
        self._write_split([("speaker1/real.wav", "speaker1", "bonafide")])
        dataset = VSASVParquetDataset(
            self.split_csv,
            self.parquet_dir,
            training=False,
            target_samples=4,
        )

        self.assertTrue(torch.equal(dataset[0]["waveform"], torch.tensor([3, 4, 5, 6])))
        self.assertTrue(torch.equal(dataset[0]["waveform"], dataset[0]["waveform"]))

    def test_training_crop_is_reproducible_and_changes_by_epoch(self) -> None:
        waveform = [float(value) for value in range(100)]
        self._write_parquet(
            [("speaker1/real.wav", waveform, 16_000, "speaker1", "bonafide")]
        )
        self._write_split([("speaker1/real.wav", "speaker1", "bonafide")])
        dataset = VSASVParquetDataset(
            self.split_csv,
            self.parquet_dir,
            training=True,
            target_samples=10,
            seed=2026,
        )

        epoch_zero = dataset[0]["waveform"]
        self.assertTrue(torch.equal(epoch_zero, dataset[0]["waveform"]))
        dataset.set_epoch(1)
        epoch_one = dataset[0]["waveform"]
        self.assertTrue(torch.equal(epoch_one, dataset[0]["waveform"]))
        self.assertFalse(torch.equal(epoch_zero, epoch_one))

    def test_metadata_mismatch_is_rejected(self) -> None:
        self._write_parquet(
            [("speaker1/real.wav", [0.1], 16_000, "speaker1", "bonafide")]
        )
        self._write_split([("speaker1/real.wav", "wrong", "bonafide")])

        with self.assertRaisesRegex(ValueError, "không khớp Parquet"):
            VSASVParquetDataset(
                self.split_csv,
                self.parquet_dir,
                training=False,
                target_samples=4,
            )


if __name__ == "__main__":
    unittest.main()
