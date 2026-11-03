from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.build_packed_artifacts import (
    build_train_chunks,
    render_markdown,
    shard_index,
    train_wanted_by_shard,
)
from scripts.ingest_evaluation_shards import TreeEntry
from src.data import MANIFEST_COLUMNS
from src.data.eval_ingest import chunk_paths


def _name(index: int) -> str:
    return f"train-{index:05d}-of-00432.parquet"


class BuildPackedArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.chunk_dir = self.root / "chunks"
        self.rows = [
            (0, "spk1/a.wav", "spk1", "bonafide", "closed_train"),
            (0, "spk2/b.wav", "spk2", "voice_conversion", "closed_train"),
            (1, "spk1/c.wav", "spk1", "replay", "closed_train"),
            (1, "spk3/d.wav", "spk3", "bonafide", "closed_dev"),
        ]
        self.tree: dict[int, TreeEntry] = {}
        for index in (0, 1):
            path = self.parquet_dir / _name(index)
            self._write_parquet(path, [row for row in self.rows if row[0] == index])
            self.tree[index] = TreeEntry(
                size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for index, file_name, speaker, utt, split in self.rows:
                writer.writerow(
                    {
                        "manifest_version": "development-20k-v2",
                        "shard": _name(index),
                        "file": file_name,
                        "speaker_id": speaker,
                        "split": split,
                        "binary_label": 0 if utt == "bonafide" else 1,
                        "utt_type": utt,
                        "native_sample_rate": 16_000,
                        "source_snapshot": "VSASV-HF-public-snapshot-v1",
                    }
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_parquet(self, path: Path, rows: list[tuple[int, str, str, str, str]]) -> None:
        target = path.as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples (file VARCHAR, audio STRUCT("array" DOUBLE[], '
                "sampling_rate BIGINT), label VARCHAR, utt_type VARCHAR)"
            )
            for position, (_index, file_name, speaker, utt, _split) in enumerate(rows):
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    [file_name, {"array": [0.01 * (position + 1)] * 32, "sampling_rate": 16_000}, speaker, utt],
                )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _build(self, tree: dict[int, TreeEntry] | None = None) -> dict[str, int]:
        return build_train_chunks(
            wanted_by_shard=train_wanted_by_shard(self.manifest),
            parquet_dir=self.parquet_dir,
            chunk_dir=self.chunk_dir,
            tree=tree or self.tree,
            shard_name=_name,
        )

    def test_shard_index_parses_official_names(self) -> None:
        self.assertEqual(shard_index("train-00431-of-00432.parquet"), 431)
        with self.assertRaises(ValueError):
            shard_index("data/train-00001-of-00432.parquet")

    def test_train_wanted_only_contains_train_rows_grouped_by_shard(self) -> None:
        wanted = train_wanted_by_shard(self.manifest)
        self.assertEqual(sorted(wanted), [0, 1])
        self.assertEqual(sorted(wanted[0]), ["spk1/a.wav", "spk2/b.wav"])
        self.assertEqual(sorted(wanted[1]), ["spk1/c.wav"])

    def test_build_train_chunks_keeps_parquet_and_skips_done(self) -> None:
        self.assertEqual(self._build(), {"ingested": 2, "already_done": 0})
        for index in (0, 1):
            self.assertTrue(chunk_paths(self.chunk_dir, index).done.is_file())
            self.assertTrue((self.parquet_dir / _name(index)).is_file())
        self.assertEqual(self._build(), {"ingested": 0, "already_done": 2})

    def test_build_train_chunks_rejects_wrong_sha256(self) -> None:
        tree = dict(self.tree)
        tree[1] = TreeEntry(size=tree[1].size, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._build(tree)
        self.assertFalse(chunk_paths(self.chunk_dir, 1).done.exists())
        self.assertTrue((self.parquet_dir / _name(1)).is_file())

    def test_markdown_uses_vietnamese_number_format(self) -> None:
        text = render_markdown(
            {
                "generated_at": "2026-11-02",
                "train_chunks": {"ingested": 1, "already_done": 0},
                "artifacts": [
                    {
                        "name": "dev-full", "split": "closed_dev", "manifest": "m.csv",
                        "artifact_count": 38777, "chunk_shard_count": 186,
                        "waveform_bytes": 3 * 1024**3, "build_seconds": 12.25,
                        "verify_seconds": 3.5, "waveforms_sha256": "a", "index_sha256": "b",
                        "metadata_sha256": "c", "source_manifest_sha256": "d",
                    }
                ],
            }
        )
        self.assertIn("| 38.777 | 186 | 3,00 |", text)


if __name__ == "__main__":
    unittest.main()
