from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np

from src.data.content_hash import waveform_sha256
from src.data.eval_ingest import (
    WantedRecord,
    chunk_paths,
    ingest_shard,
    process_shard,
    verify_chunk,
)


SHARD_NAME = "train-00003-of-00432.parquet"


def _wave(length: int, offset: float) -> list[float]:
    return [offset + 0.01 * ((i * 7) % 11) for i in range(length)]


class EvalIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.shard = self.root / SHARD_NAME
        self.chunks = self.root / "chunks"
        self.rows = {
            "id00001/a.wav": ("id00001", "bonafide", _wave(32, 0.0), 16_000),
            "id00001/b.wav": ("id00001", "voice_conversion", _wave(40, 0.1), 40_000),
            "id00002/c.wav": ("id00002", "bonafide", _wave(32, 0.2), 16_000),
        }
        self._write_shard(self.rows)
        self.wanted = {
            "id00001/a.wav": WantedRecord("id00001/a.wav", "closed_dev", "id00001", "bonafide"),
            "id00001/b.wav": WantedRecord("id00001/b.wav", "closed_dev", "id00001", "voice_conversion"),
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_shard(self, rows: dict) -> None:
        target = self.shard.as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples (file VARCHAR, '
                'audio STRUCT("array" DOUBLE[], sampling_rate BIGINT), '
                "label VARCHAR, utt_type VARCHAR)"
            )
            for name, (speaker, utt_type, wave, rate) in rows.items():
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    [name, {"array": wave, "sampling_rate": rate}, speaker, utt_type],
                )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _ingest(self, **overrides):
        arguments = dict(
            shard_path=self.shard,
            shard_index=3,
            wanted=self.wanted,
            chunk_dir=self.chunks,
        )
        arguments.update(overrides)
        return ingest_shard(**arguments)

    def _index_rows(self) -> list[dict[str, str]]:
        with chunk_paths(self.chunks, 3).index.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_ingest_records_content_hash_and_resampled_waveform(self) -> None:
        summary = self._ingest()
        self.assertEqual(summary["records"], 2)
        by_file = {row["file"]: row for row in self._index_rows()}
        a = by_file["id00001/a.wav"]
        self.assertEqual(a["content_sha256"], waveform_sha256(self.rows["id00001/a.wav"][2], 16_000))
        self.assertEqual(a["native_sample_rate"], "16000")
        self.assertEqual(a["num_samples"], "32")
        self.assertEqual(a["binary_label"], "0")
        b = by_file["id00001/b.wav"]
        self.assertEqual(b["content_sha256"], waveform_sha256(self.rows["id00001/b.wav"][2], 40_000))
        self.assertEqual(b["native_sample_rate"], "40000")
        self.assertEqual(b["num_samples"], "16")
        self.assertEqual(b["binary_label"], "1")
        self.assertEqual(b["split"], "closed_dev")
        data = np.fromfile(chunk_paths(self.chunks, 3).waveforms, dtype="<f4")
        self.assertEqual(data.size, 32 + 16)
        start, size = int(b["offset_samples"]), int(b["num_samples"])
        self.assertEqual(
            hashlib.sha256(data[start : start + size].tobytes()).hexdigest(),
            b["waveform_sha256"],
        )

    def test_unwanted_rows_are_not_stored(self) -> None:
        self._ingest()
        self.assertNotIn("id00002/c.wav", {row["file"] for row in self._index_rows()})

    def test_wrong_shard_checksum_is_rejected_before_writing(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._ingest(expected_sha256="0" * 64)
        self.assertFalse(self.chunks.exists() and any(self.chunks.iterdir()))

    def test_wrong_shard_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "kích thước"):
            self._ingest(expected_size=self.shard.stat().st_size + 1)

    def test_missing_wanted_file_leaves_no_partial_chunk(self) -> None:
        self.wanted["id00009/zzz.wav"] = WantedRecord(
            "id00009/zzz.wav", "closed_test", "id00009", "bonafide"
        )
        with self.assertRaisesRegex(ValueError, "thiếu"):
            self._ingest()
        self.assertFalse(chunk_paths(self.chunks, 3).done.exists())
        self.assertFalse(chunk_paths(self.chunks, 3).index.exists())

    def test_metadata_mismatch_is_rejected(self) -> None:
        self.wanted["id00001/a.wav"] = WantedRecord(
            "id00001/a.wav", "closed_dev", "id99999", "bonafide"
        )
        with self.assertRaisesRegex(ValueError, "không khớp"):
            self._ingest()

    def test_verify_chunk_detects_corruption(self) -> None:
        self._ingest()
        self.assertTrue(verify_chunk(self.chunks, 3)["passed"])
        with chunk_paths(self.chunks, 3).waveforms.open("r+b") as handle:
            handle.write(b"\x00\x00\x80\x3f")
        self.assertFalse(verify_chunk(self.chunks, 3)["passed"])

    def test_process_shard_deletes_parquet_only_after_success(self) -> None:
        summary = process_shard(
            shard_path=self.shard,
            shard_index=3,
            wanted=self.wanted,
            chunk_dir=self.chunks,
            delete_after=True,
        )
        self.assertEqual(summary["status"], "ingested")
        self.assertFalse(self.shard.exists())
        marker = json.loads(chunk_paths(self.chunks, 3).done.read_text(encoding="utf-8"))
        self.assertEqual(marker["records"], 2)

    def test_process_shard_keeps_parquet_on_failure(self) -> None:
        self.wanted["id00009/zzz.wav"] = WantedRecord(
            "id00009/zzz.wav", "closed_test", "id00009", "bonafide"
        )
        with self.assertRaises(ValueError):
            process_shard(
                shard_path=self.shard,
                shard_index=3,
                wanted=self.wanted,
                chunk_dir=self.chunks,
                delete_after=True,
            )
        self.assertTrue(self.shard.exists())

    def test_process_shard_resumes_from_done_marker(self) -> None:
        process_shard(
            shard_path=self.shard, shard_index=3, wanted=self.wanted,
            chunk_dir=self.chunks, delete_after=True,
        )
        again = process_shard(
            shard_path=self.shard, shard_index=3, wanted=self.wanted,
            chunk_dir=self.chunks, delete_after=True,
        )
        self.assertEqual(again["status"], "already_done")

    def test_done_marker_with_changed_wanted_set_is_rejected(self) -> None:
        process_shard(
            shard_path=self.shard, shard_index=3, wanted=self.wanted,
            chunk_dir=self.chunks, delete_after=False,
        )
        self.wanted.pop("id00001/b.wav")
        with self.assertRaisesRegex(ValueError, "tập mẫu cần"):
            process_shard(
                shard_path=self.shard, shard_index=3, wanted=self.wanted,
                chunk_dir=self.chunks, delete_after=False,
            )


if __name__ == "__main__":
    unittest.main()
