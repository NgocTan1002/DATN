from __future__ import annotations

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any

import duckdb
import torch

from src.data import (
    MANIFEST_COLUMNS,
    PackedManifestDataset,
    VSASVManifestDataset,
    build_packed_artifact,
    verify_packed_artifact,
)


class PackedManifestDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.manifest_csv = self.root / "development_20k_v2.csv"
        self.artifact_dir = self.root / "packed"

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

    def _write_fixture(self) -> None:
        shard = "train-00000-of-00432.parquet"
        self._write_parquet(
            shard,
            [
                (
                    "speaker1/first.wav",
                    torch.linspace(-0.4, 0.4, 24).tolist(),
                    16_000,
                    "speaker1",
                    "bonafide",
                ),
                (
                    "speaker2/second.wav",
                    torch.linspace(0.3, -0.3, 36).tolist(),
                    16_000,
                    "speaker2",
                    "voice_conversion",
                ),
            ],
        )
        rows = [
            {
                "manifest_version": "development-20k-v2",
                "shard": shard,
                "file": "speaker1/first.wav",
                "speaker_id": "speaker1",
                "split": "closed_train",
                "binary_label": 0,
                "utt_type": "bonafide",
                "native_sample_rate": 16_000,
                "source_snapshot": "VSASV-HF-public-snapshot-v1",
            },
            {
                "manifest_version": "development-20k-v2",
                "shard": shard,
                "file": "speaker2/second.wav",
                "speaker_id": "speaker2",
                "split": "closed_train",
                "binary_label": 1,
                "utt_type": "voice_conversion",
                "native_sample_rate": 16_000,
                "source_snapshot": "VSASV-HF-public-snapshot-v1",
            },
        ]
        with self.manifest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

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
                self.assertEqual(actual_value, expected_value, msg=f"Trường {field} sai.")

    def test_build_and_read_preserve_public_sample_contract(self) -> None:
        self._write_fixture()
        summary = build_packed_artifact(
            self.manifest_csv,
            self.parquet_dir,
            self.artifact_dir,
            split="closed_train",
            indices=[1, 0],
        )
        source = VSASVManifestDataset(
            self.manifest_csv,
            self.parquet_dir,
            split="closed_train",
            training=False,
            target_samples=16,
            amplitude_policy="peak",
        )
        packed = PackedManifestDataset(
            self.artifact_dir,
            training=False,
            target_samples=16,
            amplitude_policy="peak",
        )

        try:
            self.assertEqual(summary.artifact_count, 2)
            self.assertEqual(len(packed), 2)
            self._assert_samples_equal(packed[0], source[1])
            self._assert_samples_equal(packed[1], source[0])

            metadata = json.loads(
                (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["preprocessing_stage"],
                "post_resample_pre_amplitude_pre_crop",
            )
            self.assertEqual(metadata["target_sample_rate"], 16_000)
            self.assertEqual(metadata["dtype"], "float32-le")
            self.assertEqual(metadata["artifact_count"], 2)
            self.assertEqual(len(metadata["source_manifest_sha256"]), 64)
            verification = verify_packed_artifact(self.artifact_dir)
            self.assertEqual(verification.artifact_count, 2)
            self.assertEqual(
                verification.waveforms_sha256,
                metadata["waveforms_sha256"],
            )
        finally:
            packed.close()
            source.close()

    def test_training_crop_uses_original_split_index(self) -> None:
        self._write_fixture()
        build_packed_artifact(
            self.manifest_csv,
            self.parquet_dir,
            self.artifact_dir,
            split="closed_train",
            indices=[1],
        )
        source = VSASVManifestDataset(
            self.manifest_csv,
            self.parquet_dir,
            split="closed_train",
            training=True,
            target_samples=12,
            seed=2026,
            amplitude_policy="none",
        )
        packed = PackedManifestDataset(
            self.artifact_dir,
            training=True,
            target_samples=12,
            seed=2026,
            amplitude_policy="none",
        )

        try:
            self._assert_samples_equal(packed[0], source[1])
            source.set_epoch(1)
            packed.set_epoch(1)
            self._assert_samples_equal(packed[0], source[1])
        finally:
            packed.close()
            source.close()

    def test_rejects_truncated_waveform_storage(self) -> None:
        self._write_fixture()
        build_packed_artifact(
            self.manifest_csv,
            self.parquet_dir,
            self.artifact_dir,
            split="closed_train",
            indices=[0],
        )
        data_path = self.artifact_dir / "waveforms.f32"
        data_path.write_bytes(data_path.read_bytes()[:-4])

        with self.assertRaisesRegex(ValueError, "kích thước waveform"):
            PackedManifestDataset(
                self.artifact_dir,
                training=False,
                target_samples=16,
            )

    def test_pickled_loader_reopens_memory_map(self) -> None:
        self._write_fixture()
        build_packed_artifact(
            self.manifest_csv,
            self.parquet_dir,
            self.artifact_dir,
            split="closed_train",
            indices=[0],
        )
        original = PackedManifestDataset(
            self.artifact_dir,
            training=False,
            target_samples=16,
        )
        restored = pickle.loads(pickle.dumps(original))

        try:
            self._assert_samples_equal(restored[0], original[0])
        finally:
            restored.close()
            original.close()

    def test_integrity_check_rejects_same_size_waveform_change(self) -> None:
        self._write_fixture()
        build_packed_artifact(
            self.manifest_csv,
            self.parquet_dir,
            self.artifact_dir,
            split="closed_train",
            indices=[0],
        )
        data_path = self.artifact_dir / "waveforms.f32"
        waveform_bytes = bytearray(data_path.read_bytes())
        waveform_bytes[0] ^= 1
        data_path.write_bytes(waveform_bytes)

        with self.assertRaisesRegex(ValueError, "Checksum waveform không khớp"):
            verify_packed_artifact(self.artifact_dir)

    def test_loader_rejects_wrong_preprocessing_stage(self) -> None:
        self._write_fixture()
        build_packed_artifact(
            self.manifest_csv,
            self.parquet_dir,
            self.artifact_dir,
            split="closed_train",
            indices=[0],
        )
        metadata_path = self.artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["preprocessing_stage"] = "post_amplitude"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "bước tiền xử lý"):
            PackedManifestDataset(
                self.artifact_dir,
                training=False,
                target_samples=16,
            )


if __name__ == "__main__":
    unittest.main()
