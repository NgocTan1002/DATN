from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb
import torch
from torch.utils.data import DataLoader, Subset

from src.data import (
    MANIFEST_COLUMNS,
    VSASVManifestDataset,
    VSASVParquetDataset,
)


class CountingDuckDBConnection:
    """Đếm số truy vấn Parquet nhưng vẫn dùng kết nối DuckDB thật."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self.parquet_query_count = 0

    def execute(
        self, query: str, parameters: list[Any] | None = None
    ) -> Any:
        if "from read_parquet" in query.lower():
            self.parquet_query_count += 1
        if parameters is None:
            return self.connection.execute(query)
        return self.connection.execute(query, parameters)

    def close(self) -> None:
        self.connection.close()


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

    def _make_batch_dataset(
        self, *, training: bool = True
    ) -> VSASVManifestDataset:
        first_shard = "train-00000-of-00432.parquet"
        second_shard = "train-00001-of-00432.parquet"
        self._write_parquet(
            first_shard,
            [
                (
                    "speaker1/first.wav",
                    torch.arange(40, dtype=torch.float32).tolist(),
                    16_000,
                    "speaker1",
                    "bonafide",
                ),
                (
                    "speaker2/second.wav",
                    torch.arange(40, dtype=torch.float32).add(100).tolist(),
                    16_000,
                    "speaker2",
                    "voice_conversion",
                ),
            ],
        )
        self._write_parquet(
            second_shard,
            [
                (
                    "speaker3/third.wav",
                    torch.arange(40, dtype=torch.float32).add(200).tolist(),
                    16_000,
                    "speaker3",
                    "replay",
                )
            ],
        )
        self._write_manifest(
            [
                self._manifest_row(
                    shard=first_shard,
                    logical_file="speaker1/first.wav",
                    speaker_id="speaker1",
                    split="closed_train",
                    utt_type="bonafide",
                ),
                self._manifest_row(
                    shard=first_shard,
                    logical_file="speaker2/second.wav",
                    speaker_id="speaker2",
                    split="closed_train",
                    utt_type="voice_conversion",
                ),
                self._manifest_row(
                    shard=second_shard,
                    logical_file="speaker3/third.wav",
                    speaker_id="speaker3",
                    split="closed_train",
                    utt_type="replay",
                ),
            ]
        )
        return VSASVManifestDataset(
            self.manifest_csv,
            self.parquet_dir,
            split="closed_train",
            training=training,
            target_samples=8,
            seed=2026,
        )

    def _assert_samples_equal(
        self, actual: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        self.assertEqual(set(actual), set(expected))
        for field, expected_value in expected.items():
            actual_value = actual[field]
            if isinstance(expected_value, torch.Tensor):
                self.assertTrue(
                    torch.equal(actual_value, expected_value),
                    msg=f"Tensor {field} không tương đương.",
                )
            else:
                self.assertEqual(
                    actual_value,
                    expected_value,
                    msg=f"Trường {field} sai.",
                )

    def test_batch_read_queries_one_time_for_one_shard(self) -> None:
        dataset = self._make_batch_dataset()
        real_connection = dataset._get_connection()
        counting_connection = CountingDuckDBConnection(real_connection)
        dataset._connection = counting_connection  # type: ignore[assignment]

        try:
            samples = dataset.__getitems__([1, 0])

            self.assertEqual(counting_connection.parquet_query_count, 1)
            self.assertEqual(
                [sample["file"] for sample in samples],
                ["speaker2/second.wav", "speaker1/first.wav"],
            )
        finally:
            dataset.close()

    def test_batch_read_queries_once_per_distinct_shard(self) -> None:
        dataset = self._make_batch_dataset()
        real_connection = dataset._get_connection()
        counting_connection = CountingDuckDBConnection(real_connection)
        dataset._connection = counting_connection  # type: ignore[assignment]

        try:
            samples = dataset.__getitems__([2, 0, 1])

            self.assertEqual(counting_connection.parquet_query_count, 2)
            self.assertEqual(
                [sample["file"] for sample in samples],
                [
                    "speaker3/third.wav",
                    "speaker1/first.wav",
                    "speaker2/second.wav",
                ],
            )
        finally:
            dataset.close()

    def test_batch_read_matches_single_item_path_across_epochs(self) -> None:
        dataset = self._make_batch_dataset()
        indices = [2, 0, 1]

        try:
            epoch_zero_single = [dataset[index] for index in indices]
            epoch_zero_batch = dataset.__getitems__(indices)
            for actual, expected in zip(epoch_zero_batch, epoch_zero_single):
                self._assert_samples_equal(actual, expected)

            dataset.set_epoch(1)
            epoch_one_single = [dataset[index] for index in indices]
            epoch_one_batch = dataset.__getitems__(indices)
            for actual, expected in zip(epoch_one_batch, epoch_one_single):
                self._assert_samples_equal(actual, expected)

            self.assertTrue(
                any(
                    not torch.equal(before["waveform"], after["waveform"])
                    for before, after in zip(epoch_zero_batch, epoch_one_batch)
                )
            )
        finally:
            dataset.close()

    def test_empty_batch_does_not_open_connection(self) -> None:
        dataset = self._make_batch_dataset()

        try:
            with patch.object(
                dataset,
                "_get_connection",
                side_effect=AssertionError("Batch rỗng không được mở kết nối."),
            ):
                self.assertEqual(dataset.__getitems__([]), [])
        finally:
            dataset.close()

    def test_dataloader_uses_batch_getitems_through_subset(self) -> None:
        dataset = self._make_batch_dataset(training=False)
        original_getitems = dataset.__getitems__
        batch_calls: list[list[int]] = []
        subset_indices = [2, 0, 1, 2, 0, 1, 2, 0]

        def tracking_getitems(indices: list[int]) -> list[dict[str, Any]]:
            batch_calls.append(list(indices))
            return original_getitems(indices)

        dataset.__getitems__ = tracking_getitems  # type: ignore[method-assign]

        try:
            loader = DataLoader(
                Subset(dataset, subset_indices),
                batch_size=8,
                shuffle=False,
                num_workers=0,
            )
            batch = next(iter(loader))

            self.assertEqual(batch_calls, [subset_indices])
            self.assertEqual(
                batch["file"],
                [
                    "speaker3/third.wav",
                    "speaker1/first.wav",
                    "speaker2/second.wav",
                    "speaker3/third.wav",
                    "speaker1/first.wav",
                    "speaker2/second.wav",
                    "speaker3/third.wav",
                    "speaker1/first.wav",
                ],
            )
        finally:
            dataset.close()

    def test_batch_read_rejects_invalid_source_rows(self) -> None:
        cases: list[
            tuple[
                str,
                list[tuple[str, list[float], int, str, str]],
                str,
                str,
                int,
                type[Exception],
                str,
            ]
        ] = [
            (
                "missing",
                [("other.wav", [0.1], 16_000, "speaker1", "bonafide")],
                "missing.wav",
                "speaker1",
                16_000,
                RuntimeError,
                "đúng một waveform",
            ),
            (
                "duplicate",
                [
                    ("duplicate.wav", [0.1], 16_000, "speaker1", "bonafide"),
                    ("duplicate.wav", [0.2], 16_000, "speaker1", "bonafide"),
                ],
                "duplicate.wav",
                "speaker1",
                16_000,
                RuntimeError,
                "đúng một waveform",
            ),
            (
                "metadata",
                [("metadata.wav", [0.1], 16_000, "wrong", "bonafide")],
                "metadata.wav",
                "speaker1",
                16_000,
                ValueError,
                "không khớp Parquet",
            ),
            (
                "sample-rate",
                [("rate.wav", [0.1], 16_000, "speaker1", "bonafide")],
                "rate.wav",
                "speaker1",
                40_000,
                ValueError,
                "Sample rate manifest",
            ),
        ]

        for case_index, (
            case_name,
            parquet_rows,
            logical_file,
            speaker_id,
            native_sample_rate,
            error_type,
            message,
        ) in enumerate(cases):
            with self.subTest(case=case_name):
                shard = f"train-{case_index:05d}-of-00432.parquet"
                self._write_parquet(shard, parquet_rows)
                self._write_manifest(
                    [
                        self._manifest_row(
                            shard=shard,
                            logical_file=logical_file,
                            speaker_id=speaker_id,
                            split="closed_train",
                            utt_type="bonafide",
                            native_sample_rate=native_sample_rate,
                        )
                    ]
                )
                dataset = VSASVManifestDataset(
                    self.manifest_csv,
                    self.parquet_dir,
                    split="closed_train",
                    training=False,
                    target_samples=4,
                )

                try:
                    with self.assertRaisesRegex(error_type, message):
                        dataset.__getitems__([0])
                finally:
                    dataset.close()

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
