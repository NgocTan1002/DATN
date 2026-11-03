from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

from src.data import (
    MANIFEST_COLUMNS,
    PackedManifestDataset,
    VSASVManifestDataset,
    assemble_packed_from_chunks,
    build_packed_artifact,
    verify_packed_artifact,
)
from src.data.eval_ingest import WantedRecord, chunk_paths, ingest_shard


SNAPSHOT = "VSASV-HF-public-snapshot-v1"


def _shard(index: int) -> str:
    return f"train-{index:05d}-of-00432.parquet"


class PackedAssemblyTests(unittest.TestCase):
    """Lắp packed artifact từ chunk từng shard và đối chiếu với đường Parquet."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.chunk_dir = self.root / "chunks"
        # (file, speaker, utt_type, sample_rate, split, length)
        self.rows = {
            0: [
                ("spk1/a.wav", "spk1", "bonafide", 16_000, "closed_train", 70_000),
                ("spk2/b.wav", "spk2", "voice_conversion", 40_000, "closed_train", 90_000),
                ("spk3/c.wav", "spk3", "bonafide", 16_000, "closed_dev", 30_000),
                ("spk4/d.wav", "spk4", "bonafide", 16_000, "closed_test", 20_000),
            ],
            1: [
                ("spk1/e.wav", "spk1", "replay", 16_000, "closed_train", 50_000),
                ("spk3/f.wav", "spk3", "adversarial_attack", 16_000, "closed_dev", 66_000),
                ("spk5/g.wav", "spk5", "voice_conversion", 40_000, "closed_dev", 45_000),
                ("spk4/h.wav", "spk4", "replay", 16_000, "closed_test", 64_000),
            ],
        }
        generator = np.random.default_rng(7)
        for shard_index, rows in self.rows.items():
            self._write_parquet(
                _shard(shard_index),
                [
                    (
                        file_name,
                        (generator.standard_normal(length) * 0.1).tolist(),
                        rate,
                        speaker,
                        utt_type,
                    )
                    for file_name, speaker, utt_type, rate, _split, length in rows
                ],
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_parquet(self, name: str, rows: list[tuple[str, list[float], int, str, str]]) -> None:
        target = (self.parquet_dir / name).as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples (file VARCHAR, audio STRUCT("array" DOUBLE[], '
                "sampling_rate BIGINT), label VARCHAR, utt_type VARCHAR)"
            )
            for file_name, waveform, rate, speaker, utt_type in rows:
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    [file_name, {"array": waveform, "sampling_rate": rate}, speaker, utt_type],
                )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _ingest(self, splits: set[str]) -> None:
        for shard_index, rows in self.rows.items():
            wanted = {
                file_name: WantedRecord(file=file_name, split=split, speaker_id=speaker, utt_type=utt)
                for file_name, speaker, utt, _rate, split, _length in rows
                if split in splits
            }
            ingest_shard(
                shard_path=self.parquet_dir / _shard(shard_index),
                shard_index=shard_index,
                wanted=wanted,
                chunk_dir=self.chunk_dir,
            )

    def _write_manifest(
        self, path: Path, version: str, selected: list[tuple[int, str]]
    ) -> Path:
        lookup = {
            row[0]: (shard_index, row)
            for shard_index, rows in self.rows.items()
            for row in rows
        }
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for _order, file_name in selected:
                shard_index, (name, speaker, utt, rate, split, _length) = lookup[file_name]
                writer.writerow(
                    {
                        "manifest_version": version,
                        "shard": _shard(shard_index),
                        "file": name,
                        "speaker_id": speaker,
                        "split": split,
                        "binary_label": 0 if utt == "bonafide" else 1,
                        "utt_type": utt,
                        "native_sample_rate": rate,
                        "source_snapshot": SNAPSHOT,
                    }
                )
        return path

    def _train_manifest(self) -> Path:
        # Thứ tự cố ý không theo shard để source_index phải lấy theo manifest.
        return self._write_manifest(
            self.root / "train.csv",
            "development-20k-v2",
            [(0, "spk1/e.wav"), (1, "spk1/a.wav"), (2, "spk2/b.wav"), (3, "spk3/c.wav")],
        )

    def _assert_items_equal(self, left: dict[str, Any], right: dict[str, Any]) -> None:
        self.assertEqual(set(left), set(right))
        for key, value in right.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(left[key], value), key)
            else:
                self.assertEqual(left[key], value, key)

    def test_train_artifact_from_chunks_matches_parquet_builder_and_loader(self) -> None:
        self._ingest({"closed_train"})
        manifest = self._train_manifest()
        summary = assemble_packed_from_chunks(
            manifest, self.chunk_dir, self.root / "train_packed", split="closed_train"
        )
        reference = build_packed_artifact(
            manifest, self.parquet_dir, self.root / "reference", split="closed_train", indices=[0, 1, 2]
        )
        self.assertEqual(summary.artifact_count, 3)
        self.assertEqual(summary.waveforms_sha256, reference.waveforms_sha256)
        self.assertEqual(summary.source_manifest_sha256, reference.source_manifest_sha256)
        verify_packed_artifact(summary.artifact_dir)

        for seed, epoch in ((2026, 0), (2027, 2)):
            packed = PackedManifestDataset(
                summary.artifact_dir, training=True, seed=seed, amplitude_policy="peak"
            )
            parquet = VSASVManifestDataset(
                manifest, self.parquet_dir, split="closed_train", training=True,
                seed=seed, amplitude_policy="peak",
            )
            try:
                packed.set_epoch(epoch)
                parquet.set_epoch(epoch)
                self.assertEqual([r.source_index for r in packed.records], [0, 1, 2])
                for index in range(3):
                    self._assert_items_equal(packed[index], parquet[index])
            finally:
                packed.close()
                parquet.close()

    def test_dev_artifact_keeps_manifest_order_and_eval_items_match_parquet(self) -> None:
        self._ingest({"closed_dev", "closed_test"})
        manifest = self._write_manifest(
            self.root / "dev_full.csv",
            "closed-dev-full-v1",
            [(0, "spk5/g.wav"), (1, "spk3/c.wav"), (2, "spk3/f.wav")],
        )
        summary = assemble_packed_from_chunks(
            manifest, self.chunk_dir, self.root / "dev_packed", split="closed_dev"
        )
        metadata = json.loads((summary.artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["manifest_version"], "closed-dev-full-v1")
        self.assertEqual(metadata["split"], "closed_dev")
        self.assertEqual(metadata["assembled_from"], "shard_chunks")
        packed = PackedManifestDataset(summary.artifact_dir, training=False, amplitude_policy="peak")
        parquet = VSASVManifestDataset(
            manifest, self.parquet_dir, split="closed_dev", training=False, amplitude_policy="peak"
        )
        try:
            self.assertEqual([r.file for r in packed.records], ["spk5/g.wav", "spk3/c.wav", "spk3/f.wav"])
            for index in range(3):
                self._assert_items_equal(packed[index], parquet[index])
        finally:
            packed.close()
            parquet.close()

    def test_test_artifact_requires_explicit_allow_test(self) -> None:
        self._ingest({"closed_dev", "closed_test"})
        manifest = self._write_manifest(
            self.root / "test_full.csv",
            "closed-test-full-v1",
            [(0, "spk4/d.wav"), (1, "spk4/h.wav")],
        )
        summary = assemble_packed_from_chunks(
            manifest, self.chunk_dir, self.root / "test_packed", split="closed_test"
        )
        with self.assertRaisesRegex(ValueError, "closed_test"):
            PackedManifestDataset(summary.artifact_dir, training=False)
        with self.assertRaisesRegex(ValueError, "closed_test"):
            verify_packed_artifact(summary.artifact_dir)
        verification = verify_packed_artifact(summary.artifact_dir, allow_test=True)
        self.assertEqual(verification.artifact_count, 2)
        dataset = PackedManifestDataset(summary.artifact_dir, training=False, allow_test=True)
        try:
            self.assertEqual(len(dataset), 2)
        finally:
            dataset.close()

    def test_loader_rejects_row_split_different_from_metadata(self) -> None:
        self._ingest({"closed_train"})
        summary = assemble_packed_from_chunks(
            self._train_manifest(), self.chunk_dir, self.root / "packed", split="closed_train"
        )
        metadata_path = summary.artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["split"] = "closed_dev"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "split"):
            PackedManifestDataset(summary.artifact_dir, training=False)

    def test_rejects_chunk_metadata_mismatch(self) -> None:
        self._ingest({"closed_train"})
        manifest = self._train_manifest()
        text = manifest.read_text(encoding="utf-8").replace(",spk2,closed_train", ",spk9,closed_train")
        manifest.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "spk2/b.wav"):
            assemble_packed_from_chunks(manifest, self.chunk_dir, self.root / "packed", split="closed_train")
        self.assertFalse((self.root / "packed").exists())

    def test_rejects_missing_chunk_row_and_tampered_waveform(self) -> None:
        self._ingest({"closed_train"})
        manifest = self._train_manifest()
        with self.assertRaisesRegex(ValueError, "spk3/c.wav"):
            # Dòng dev không có trong chunk train.
            assemble_packed_from_chunks(manifest, self.chunk_dir, self.root / "p1", split="closed_dev")
        paths = chunk_paths(self.chunk_dir, 0)
        data = bytearray(paths.waveforms.read_bytes())
        data[8] ^= 0xFF
        paths.waveforms.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "checksum"):
            assemble_packed_from_chunks(manifest, self.chunk_dir, self.root / "p2", split="closed_train")
        self.assertFalse((self.root / "p2").exists())

    def test_rejects_index_not_matching_done_marker(self) -> None:
        self._ingest({"closed_train"})
        paths = chunk_paths(self.chunk_dir, 1)
        paths.index.write_text(paths.index.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "done marker"):
            assemble_packed_from_chunks(
                self._train_manifest(), self.chunk_dir, self.root / "packed", split="closed_train"
            )

    def test_refuses_to_overwrite_existing_artifact(self) -> None:
        self._ingest({"closed_train"})
        (self.root / "packed").mkdir()
        with self.assertRaises(FileExistsError):
            assemble_packed_from_chunks(
                self._train_manifest(), self.chunk_dir, self.root / "packed", split="closed_train"
            )


if __name__ == "__main__":
    unittest.main()
