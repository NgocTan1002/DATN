from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np

from scripts.check_packed_parity import check_artifact_parity, render_markdown
from src.data import MANIFEST_COLUMNS, assemble_packed_from_chunks
from src.data.eval_ingest import WantedRecord, ingest_shard


ROOT = Path(__file__).resolve().parents[1]
AUDIO = json.loads((ROOT / "configs" / "audio.json").read_text(encoding="utf-8"))


def _shard(index: int) -> str:
    return f"train-{index:05d}-of-00432.parquet"


class PackedParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.chunk_dir = self.root / "chunks"
        self.generator = np.random.default_rng(11)
        self.rows = {
            0: [
                ("spk1/a.wav", "spk1", "bonafide", 16_000, "closed_train", 70_000),
                ("spk2/b.wav", "spk2", "voice_conversion", 40_000, "closed_train", 90_000),
                ("spk4/t.wav", "spk4", "bonafide", 16_000, "closed_test", 30_000),
            ],
            1: [
                ("spk1/c.wav", "spk1", "replay", 16_000, "closed_train", 50_000),
                ("spk4/u.wav", "spk4", "replay", 16_000, "closed_test", 70_000),
            ],
        }
        self.audio = {
            row[0]: (self.generator.standard_normal(row[5]) * 0.1).tolist()
            for rows in self.rows.values()
            for row in rows
        }
        for index in self.rows:
            self._write_parquet(index)
        for index, rows in self.rows.items():
            ingest_shard(
                shard_path=self.parquet_dir / _shard(index),
                shard_index=index,
                wanted={
                    f: WantedRecord(file=f, split=split, speaker_id=s, utt_type=u)
                    for f, s, u, _r, split, _n in rows
                },
                chunk_dir=self.chunk_dir,
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_parquet(self, index: int) -> None:
        target = (self.parquet_dir / _shard(index)).as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples (file VARCHAR, audio STRUCT("array" DOUBLE[], '
                "sampling_rate BIGINT), label VARCHAR, utt_type VARCHAR)"
            )
            for file_name, speaker, utt, rate, _split, _n in self.rows[index]:
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    [file_name, {"array": self.audio[file_name], "sampling_rate": rate}, speaker, utt],
                )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _artifact(self, split: str, version: str) -> tuple[Path, Path]:
        manifest = self.root / f"{split}.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for index, rows in self.rows.items():
                for f, s, u, rate, row_split, _n in rows:
                    if row_split == split:
                        writer.writerow(
                            {
                                "manifest_version": version, "shard": _shard(index), "file": f,
                                "speaker_id": s, "split": row_split,
                                "binary_label": 0 if u == "bonafide" else 1, "utt_type": u,
                                "native_sample_rate": rate,
                                "source_snapshot": "VSASV-HF-public-snapshot-v1",
                            }
                        )
        artifact = self.root / f"packed_{split}"
        assemble_packed_from_chunks(manifest, self.chunk_dir, artifact, split=split)  # type: ignore[arg-type]
        return artifact, manifest

    def _check(self, artifact: Path, manifest: Path, split: str) -> dict:
        return check_artifact_parity(
            artifact_dir=artifact, manifest_csv=manifest, split=split,
            parquet_dir=self.parquet_dir, audio_config=AUDIO,
            train_checks=((2026, 0), (2028, 2)),
        )

    def test_train_parity_passes_with_full_coverage_and_crop_checks(self) -> None:
        artifact, manifest = self._artifact("closed_train", "development-20k-v2")
        result = self._check(artifact, manifest, "closed_train")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["local_rows"], 3)
        self.assertEqual(result["local_coverage"], 1.0)
        self.assertEqual(set(result["train_item_mismatches"]), {"seed2026_epoch0", "seed2028_epoch2"})

    def test_detects_waveform_differing_from_parquet(self) -> None:
        artifact, manifest = self._artifact("closed_train", "development-20k-v2")
        self.audio["spk1/c.wav"] = (np.asarray(self.audio["spk1/c.wav"]) * 0.5).tolist()
        self._write_parquet(1)
        result = self._check(artifact, manifest, "closed_train")
        self.assertFalse(result["passed"])
        self.assertEqual(result["raw_mismatches"], 1)
        self.assertGreater(result["raw_max_abs_diff"], 0.0)
        self.assertEqual(result["examples"][0]["file"], "spk1/c.wav")

    def test_train_requires_full_local_coverage(self) -> None:
        artifact, manifest = self._artifact("closed_train", "development-20k-v2")
        (self.parquet_dir / _shard(1)).unlink()
        result = self._check(artifact, manifest, "closed_train")
        self.assertEqual(result["local_rows"], 2)
        self.assertEqual(result["missing_local_shards"], 1)
        self.assertFalse(result["passed"])

    def test_test_split_compares_local_rows_only_without_training_crops(self) -> None:
        artifact, manifest = self._artifact("closed_test", "closed-test-full-v1")
        (self.parquet_dir / _shard(1)).unlink()
        result = self._check(artifact, manifest, "closed_test")
        self.assertTrue(result["passed"], result)
        self.assertEqual((result["artifact_rows"], result["local_rows"]), (2, 1))
        self.assertEqual(result["train_item_mismatches"], {})
        text = render_markdown(
            {"generated_at": "x", "status": "ĐẠT", "train_checks": [[2026, 0]],
             "artifacts": {"test-full": result}}
        )
        self.assertIn("| test-full | 2 | 1 | 50,00% | 1 |", text)


if __name__ == "__main__":
    unittest.main()
