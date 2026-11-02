from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.plan_evaluation_shards import (
    OFFICIAL_GEOMETRY,
    ShardGeometry,
    build_plan,
    parse_tree_sizes,
    render_markdown,
    shard_filename,
    shard_index_for_row,
    verify_geometry_against_local,
)


SMALL = ShardGeometry(first_count=2, first_rows=3, rest_count=2, rest_rows=2)


class ShardGeometryTests(unittest.TestCase):
    def test_official_geometry_covers_all_metadata_rows(self) -> None:
        self.assertEqual(OFFICIAL_GEOMETRY.total_shards, 432)
        self.assertEqual(OFFICIAL_GEOMETRY.total_rows, 220_963)

    def test_official_geometry_boundaries(self) -> None:
        expected = {
            0: 0,
            511: 0,
            512: 1,
            108_031: 210,
            108_032: 211,
            108_542: 211,
            108_543: 212,
            220_962: 431,
        }
        for row, shard in expected.items():
            with self.subTest(row=row):
                self.assertEqual(shard_index_for_row(OFFICIAL_GEOMETRY, row), shard)

    def test_small_geometry_boundaries(self) -> None:
        mapping = [shard_index_for_row(SMALL, row) for row in range(SMALL.total_rows)]
        self.assertEqual(mapping, [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])

    def test_row_out_of_range_is_rejected(self) -> None:
        for row in (-1, SMALL.total_rows):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "ngoài phạm vi"):
                    shard_index_for_row(SMALL, row)

    def test_shard_filename_uses_official_pattern(self) -> None:
        self.assertEqual(
            shard_filename(OFFICIAL_GEOMETRY, 2), "train-00002-of-00432.parquet"
        )
        self.assertEqual(
            shard_filename(OFFICIAL_GEOMETRY, 431), "train-00431-of-00432.parquet"
        )
        with self.assertRaisesRegex(ValueError, "ngoài phạm vi"):
            shard_filename(OFFICIAL_GEOMETRY, 432)


class TreeParsingTests(unittest.TestCase):
    def test_parse_tree_sizes_keeps_only_data_shards(self) -> None:
        entries = [
            {"path": "data/train-00001-of-00432.parquet", "size": 10, "type": "file"},
            {"path": "README.md", "size": 99, "type": "file"},
            {"path": "data", "size": 0, "type": "directory"},
        ]
        self.assertEqual(
            parse_tree_sizes(entries, OFFICIAL_GEOMETRY),
            {1: 10},
        )

    def test_parse_tree_sizes_rejects_wrong_shard_count(self) -> None:
        entries = [
            {"path": "data/train-00001-of-00999.parquet", "size": 10, "type": "file"}
        ]
        with self.assertRaisesRegex(ValueError, "tổng số shard"):
            parse_tree_sizes(entries, OFFICIAL_GEOMETRY)


class FixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.files = [f"id{index:05d}/{index:05d}.wav" for index in range(10)]
        self.metadata = self.root / "metadata.csv"
        with self.metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "label", "utt_type"])
            for file_name in self.files:
                writer.writerow([file_name, file_name.split("/")[0], "bonafide"])
        self.splits = self.root / "splits"
        self.splits.mkdir()
        self._write_split("closed_dev", [1, 4, 6])
        self._write_split("closed_test", [2, 8, 9])
        self.parquet = self.root / "parquet"
        self.parquet.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_split(self, name: str, rows: list[int]) -> None:
        with (self.splits / f"{name}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "label", "utt_type"])
            for row in rows:
                writer.writerow([self.files[row], self.files[row].split("/")[0], "bonafide"])

    def _write_shard(self, index: int, rows: list[int]) -> None:
        path = self.parquet / shard_filename(SMALL, index)
        escaped = path.as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute("CREATE TABLE fixture(file VARCHAR)")
            connection.executemany(
                "INSERT INTO fixture VALUES (?)", [(self.files[row],) for row in rows]
            )
            connection.execute(f"COPY fixture TO '{escaped}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def test_geometry_matches_local_shards(self) -> None:
        self._write_shard(0, [0, 1, 2])
        self._write_shard(2, [6, 7])
        result = verify_geometry_against_local(
            SMALL, self.parquet, self.metadata
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["checked_shards"], 2)

    def test_geometry_mismatch_is_detected(self) -> None:
        self._write_shard(0, [0, 1, 3])
        result = verify_geometry_against_local(
            SMALL, self.parquet, self.metadata
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatched_shards"], [0])

    def test_plan_lists_missing_shards_per_split(self) -> None:
        self._write_shard(0, [0, 1, 2])
        sizes = {0: 100, 1: 200, 2: 300, 3: 400}
        plan = build_plan(
            geometry=SMALL,
            metadata_path=self.metadata,
            splits_dir=self.splits,
            parquet_dir=self.parquet,
            tree_sizes=sizes,
            free_bytes=10_000,
            packed_bytes_per_sample=10,
        )
        # dev: dòng 1,4,6 -> shard 0,1,2; test: dòng 2,8,9 -> shard 0,3,3.
        self.assertEqual(plan["splits"]["closed_dev"]["shards"], [0, 1, 2])
        self.assertEqual(plan["splits"]["closed_test"]["shards"], [0, 3])
        self.assertEqual(plan["required_union"]["shards"], [0, 1, 2, 3])
        self.assertEqual(plan["missing"]["shards"], [1, 2, 3])
        self.assertEqual(plan["missing"]["size_bytes"], 900)
        self.assertEqual(plan["missing"]["size_source"], "huggingface_tree")
        self.assertEqual(plan["missing"]["dev_only_shards"], [1, 2])
        self.assertEqual(plan["missing"]["test_only_shards"], [3])
        self.assertEqual(plan["missing"]["both_shards"], [])
        self.assertEqual(plan["splits"]["closed_dev"]["rows"], 3)
        self.assertEqual(plan["splits"]["closed_test"]["rows"], 3)

    def test_plan_estimates_sizes_when_tree_is_missing(self) -> None:
        self._write_shard(0, [0, 1, 2])
        path = self.parquet / shard_filename(SMALL, 0)
        local_size = path.stat().st_size
        plan = build_plan(
            geometry=SMALL,
            metadata_path=self.metadata,
            splits_dir=self.splits,
            parquet_dir=self.parquet,
            tree_sizes=None,
            free_bytes=10_000,
            packed_bytes_per_sample=10,
        )
        self.assertEqual(plan["missing"]["size_source"], "local_mean_estimate")
        self.assertEqual(plan["missing"]["size_bytes"], 3 * local_size)

    def test_plan_budget_flags_insufficient_disk(self) -> None:
        self._write_shard(0, [0, 1, 2])
        plan = build_plan(
            geometry=SMALL,
            metadata_path=self.metadata,
            splits_dir=self.splits,
            parquet_dir=self.parquet,
            tree_sizes={0: 100, 1: 200, 2: 300, 3: 400},
            free_bytes=1_000,
            packed_bytes_per_sample=1_000,
        )
        budget = plan["disk_budget"]
        self.assertEqual(budget["packed_eval_bytes"], 6 * 1_000)
        self.assertEqual(budget["keep_parquet_total_bytes"], 900 + 6_000)
        self.assertFalse(budget["keep_parquet_fits"])
        self.assertEqual(budget["streaming_peak_bytes"], 6_000 + 400)
        self.assertFalse(budget["streaming_fits"])

    def test_plan_budget_requires_headroom(self) -> None:
        self._write_shard(0, [0, 1, 2])
        sizes = {0: 100, 1: 200, 2: 300, 3: 400}
        common = dict(
            geometry=SMALL,
            metadata_path=self.metadata,
            splits_dir=self.splits,
            parquet_dir=self.parquet,
            tree_sizes=sizes,
            packed_bytes_per_sample=1_000,
        )
        # Cần 900 + 6.000 = 6.900 byte; 10% dự phòng của 7.000 chỉ còn 6.300.
        tight = build_plan(free_bytes=7_000, **common)["disk_budget"]
        self.assertFalse(tight["keep_parquet_fits"])
        self.assertEqual(tight["headroom_fraction"], 0.10)
        roomy = build_plan(free_bytes=8_000, **common)["disk_budget"]
        self.assertTrue(roomy["keep_parquet_fits"])

    def test_render_markdown_reports_core_numbers(self) -> None:
        self._write_shard(0, [0, 1, 2])
        plan = build_plan(
            geometry=SMALL,
            metadata_path=self.metadata,
            splits_dir=self.splits,
            parquet_dir=self.parquet,
            tree_sizes={0: 100, 1: 200, 2: 300, 3: 400},
            free_bytes=10_000,
            packed_bytes_per_sample=10,
        )
        text = render_markdown(plan)
        self.assertIn("closed_dev", text)
        self.assertIn("closed_test", text)
        self.assertIn("1, 2, 3", text)
        self.assertIn("huggingface_tree", text)

    def test_unknown_file_in_split_is_rejected(self) -> None:
        with (self.splits / "closed_dev.csv").open(
            "a", encoding="utf-8", newline=""
        ) as handle:
            handle.write("idXXXXX/99999.wav,idXXXXX,bonafide\n")
        with self.assertRaisesRegex(ValueError, "không có trong metadata"):
            build_plan(
                geometry=SMALL,
                metadata_path=self.metadata,
                splits_dir=self.splits,
                parquet_dir=self.parquet,
                tree_sizes=None,
                free_bytes=1,
                packed_bytes_per_sample=1,
            )


if __name__ == "__main__":
    unittest.main()
