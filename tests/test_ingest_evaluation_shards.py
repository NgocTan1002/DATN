from __future__ import annotations

import csv
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.ingest_evaluation_shards import (
    TreeEntry,
    build_wanted_by_shard,
    load_tree,
    run_ingest,
)
from scripts.plan_evaluation_shards import ShardGeometry, shard_filename
from src.data.eval_ingest import chunk_paths


GEOMETRY = ShardGeometry(first_count=2, first_rows=3, rest_count=2, rest_rows=2)


def _wave(length: int, offset: float) -> list[float]:
    return [offset + 0.01 * ((i * 5) % 7) for i in range(length)]


class IngestDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.files = [f"id{i:05d}/{i:05d}.wav" for i in range(10)]
        self.utt = ["bonafide"] * 6 + ["voice_conversion"] * 4
        self.metadata = self.root / "metadata.csv"
        self._write_csv(self.metadata, range(10))
        self.splits = self.root / "splits"
        self.splits.mkdir()
        self._write_csv(self.splits / "closed_dev.csv", [1, 4, 6])
        self._write_csv(self.splits / "closed_test.csv", [2, 8, 9])
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.local = self.root / "local"
        self.local.mkdir()
        self.download = self.root / "download"
        self.chunks = self.root / "chunks"
        self.tree: dict[int, TreeEntry] = {}
        layout = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7], 3: [8, 9]}
        for index, rows in layout.items():
            path = self.remote / shard_filename(GEOMETRY, index)
            self._write_shard(path, rows)
            self.tree[index] = TreeEntry(
                size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        # Shard 0 đã có sẵn cục bộ; các shard khác phải tải.
        shutil.copy(self.remote / shard_filename(GEOMETRY, 0), self.local)
        self.downloads: list[int] = []
        self.wanted = build_wanted_by_shard(
            geometry=GEOMETRY, metadata_path=self.metadata, splits_dir=self.splits
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_csv(self, path: Path, rows) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "label", "utt_type"])
            for row in rows:
                writer.writerow([self.files[row], self.files[row].split("/")[0], self.utt[row]])

    def _write_shard(self, path: Path, rows: list[int]) -> None:
        target = path.as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples (file VARCHAR, '
                'audio STRUCT("array" DOUBLE[], sampling_rate BIGINT), '
                "label VARCHAR, utt_type VARCHAR)"
            )
            for row in rows:
                rate = 40_000 if self.utt[row] == "voice_conversion" else 16_000
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    [
                        self.files[row],
                        {"array": _wave(40, 0.01 * row), "sampling_rate": rate},
                        self.files[row].split("/")[0],
                        self.utt[row],
                    ],
                )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _downloader(self, filename: str, destination: Path) -> Path:
        index = int(filename.split("-")[1])
        self.downloads.append(index)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / filename
        shutil.copy(self.remote / filename, target)
        return target

    def _run(self, **overrides):
        arguments = dict(
            geometry=GEOMETRY,
            wanted_by_shard=self.wanted,
            shard_indices=[0, 1, 2, 3],
            local_dir=self.local,
            download_dir=self.download,
            chunk_dir=self.chunks,
            tree=self.tree,
            downloader=self._downloader,
            execute=True,
            reserve_bytes=0,
            bytes_per_sample=10,
            free_bytes_fn=lambda _path: 10**12,
        )
        arguments.update(overrides)
        return run_ingest(**arguments)

    def test_build_wanted_groups_rows_by_shard_with_speaker(self) -> None:
        self.assertEqual(sorted(self.wanted), [0, 1, 2, 3])
        self.assertEqual(sorted(self.wanted[0]), [self.files[1], self.files[2]])
        record = self.wanted[1][self.files[4]]
        self.assertEqual((record.split, record.speaker_id, record.utt_type), ("closed_dev", "id00004", "bonafide"))
        self.assertEqual(self.wanted[3][self.files[8]].split, "closed_test")

    def test_load_tree_reads_size_and_sha256(self) -> None:
        cache = self.root / "tree.json"
        cache.write_text(
            '[{"path": "data/train-00001-of-00004.parquet", "size": 5, "sha256": "ab"},'
            ' {"path": "README.md", "size": 1, "sha256": null}]',
            encoding="utf-8",
        )
        tree = load_tree(cache, GEOMETRY)
        self.assertEqual(tree, {1: TreeEntry(size=5, sha256="ab")})

    def test_dry_run_changes_nothing(self) -> None:
        summary = self._run(execute=False)
        self.assertEqual(self.downloads, [])
        self.assertFalse(self.chunks.exists())
        actions = {item["shard"]: item["action"] for item in summary["shards"]}
        self.assertEqual(actions, {0: "ingest_local", 1: "download_ingest_delete", 2: "download_ingest_delete", 3: "download_ingest_delete"})

    def test_execute_downloads_ingests_and_deletes_only_downloads(self) -> None:
        summary = self._run()
        self.assertEqual(self.downloads, [1, 2, 3])
        self.assertEqual(summary["failed"], 0)
        for index in range(4):
            self.assertTrue(chunk_paths(self.chunks, index).done.is_file())
        self.assertTrue((self.local / shard_filename(GEOMETRY, 0)).exists())
        self.assertFalse(list(self.download.rglob("*.parquet")))

    def test_done_shards_are_skipped_without_download(self) -> None:
        self._run()
        self.downloads.clear()
        summary = self._run()
        self.assertEqual(self.downloads, [])
        self.assertEqual({item["action"] for item in summary["shards"]}, {"skip_done"})

    def test_corrupt_download_is_retried_once_then_fails(self) -> None:
        calls: list[int] = []

        def corrupt(filename: str, destination: Path) -> Path:
            calls.append(int(filename.split("-")[1]))
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / filename
            target.write_bytes(b"khong phai parquet")
            return target

        summary = self._run(
            shard_indices=[1], downloader=corrupt, max_consecutive_failures=5
        )
        self.assertEqual(calls, [1, 1])
        self.assertEqual(summary["failed"], 1)
        self.assertFalse(chunk_paths(self.chunks, 1).done.exists())
        self.assertFalse(list(self.download.rglob("*.parquet")))

    def _wanted_with_missing_file(self) -> dict:
        wanted = {index: dict(records) for index, records in self.wanted.items()}
        template = next(iter(wanted[1].values()))
        wanted[1]["id99999/99999.wav"] = type(template)(
            file="id99999/99999.wav",
            split=template.split,
            speaker_id="id99999",
            utt_type=template.utt_type,
        )
        return wanted

    def test_failed_ingest_keeps_verified_download(self) -> None:
        summary = self._run(
            shard_indices=[1], wanted_by_shard=self._wanted_with_missing_file()
        )
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(self.downloads, [1])
        self.assertFalse(chunk_paths(self.chunks, 1).done.exists())
        kept = self.download / shard_filename(GEOMETRY, 1)
        self.assertTrue(kept.is_file())
        self.assertEqual(
            hashlib.sha256(kept.read_bytes()).hexdigest(), self.tree[1].sha256
        )

    def test_rerun_reuses_intact_download_then_deletes_it(self) -> None:
        self._run(shard_indices=[1], wanted_by_shard=self._wanted_with_missing_file())
        self.downloads.clear()
        summary = self._run(shard_indices=[1])
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(self.downloads, [])
        self.assertTrue(chunk_paths(self.chunks, 1).done.is_file())
        self.assertFalse(list(self.download.rglob("*.parquet")))

    def test_corrupt_leftover_download_is_replaced(self) -> None:
        self.download.mkdir()
        (self.download / shard_filename(GEOMETRY, 1)).write_bytes(b"hong")
        summary = self._run(shard_indices=[1])
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(self.downloads, [1])
        self.assertFalse(list(self.download.rglob("*.parquet")))

    def test_low_disk_stops_before_download(self) -> None:
        summary = self._run(free_bytes_fn=lambda _path: 1)
        self.assertEqual(self.downloads, [])
        self.assertEqual(summary["stopped_reason"], "low_disk")
        self.assertFalse(chunk_paths(self.chunks, 1).done.exists())

    def test_consecutive_failures_abort_run(self) -> None:
        def broken(filename: str, destination: Path) -> Path:
            raise OSError("mất kết nối")

        summary = self._run(
            shard_indices=[1, 2, 3], downloader=broken, max_consecutive_failures=2
        )
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["stopped_reason"], "too_many_failures")

    def test_missing_tree_entry_is_rejected_up_front(self) -> None:
        tree = dict(self.tree)
        tree.pop(2)
        with self.assertRaisesRegex(ValueError, "tree"):
            self._run(tree=tree)
        self.assertFalse(self.chunks.exists())


if __name__ == "__main__":
    unittest.main()
