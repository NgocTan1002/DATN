#!/usr/bin/env python3
"""Lập kế hoạch shard và dung lượng cho dev-full/test-full theo D025.

Script chỉ đọc: tính shard Hugging Face chứa từng dòng của `closed_dev` và
`closed_test`, so với shard cục bộ và ước tính dung lượng cần tải. Nó không tải
shard, không đọc waveform và không đọc metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_PACKED_REPORT = PROJECT_ROOT / "reports" / "packed_artifact_benchmark.json"
DEFAULT_TREE_CACHE = PROJECT_ROOT / "reports" / "vsasv_shard_tree.json"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "evaluation_shard_plan.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "evaluation_shard_plan.md"
REPOSITORY = "hustep-lab/VSASV-Dataset"
REVISION = "92de668616780ed4f3d7f58a82262f82b56b4557"
EVALUATION_SPLITS = ("closed_dev", "closed_test")
TRAIN_MANIFEST_SAMPLES = 15_885
DISK_HEADROOM_FRACTION = 0.10
SHARD_PATTERN = re.compile(r"^data/train-(\d{5})-of-(\d{5})\.parquet$")


@dataclass(frozen=True)
class ShardGeometry:
    """Các shard đầu có `first_rows` dòng, các shard sau có `rest_rows` dòng."""

    first_count: int
    first_rows: int
    rest_count: int
    rest_rows: int

    @property
    def total_shards(self) -> int:
        return self.first_count + self.rest_count

    @property
    def total_rows(self) -> int:
        return (
            self.first_count * self.first_rows + self.rest_count * self.rest_rows
        )


OFFICIAL_GEOMETRY = ShardGeometry(
    first_count=211, first_rows=512, rest_count=221, rest_rows=511
)


def shard_index_for_row(geometry: ShardGeometry, row_index: int) -> int:
    """Trả chỉ số shard chứa dòng metadata thứ `row_index` (đếm từ 0)."""

    if not 0 <= row_index < geometry.total_rows:
        raise ValueError(
            f"Dòng {row_index} ngoài phạm vi metadata 0..{geometry.total_rows - 1}."
        )
    first_total = geometry.first_count * geometry.first_rows
    if row_index < first_total:
        return row_index // geometry.first_rows
    return geometry.first_count + (row_index - first_total) // geometry.rest_rows


def shard_filename(geometry: ShardGeometry, index: int) -> str:
    if not 0 <= index < geometry.total_shards:
        raise ValueError(
            f"Shard {index} ngoài phạm vi 0..{geometry.total_shards - 1}."
        )
    return f"train-{index:05d}-of-{geometry.total_shards:05d}.parquet"


def parse_tree_sizes(
    entries: Iterable[Mapping[str, Any]], geometry: ShardGeometry
) -> dict[int, int]:
    """Lấy kích thước byte của các shard `data/train-*.parquet` từ tree Hugging Face."""

    sizes: dict[int, int] = {}
    for entry in entries:
        match = SHARD_PATTERN.match(str(entry.get("path", "")))
        if match is None:
            continue
        if int(match.group(2)) != geometry.total_shards:
            raise ValueError(
                "Tree Hugging Face có tổng số shard "
                f"{int(match.group(2))}, kỳ vọng {geometry.total_shards}."
            )
        sizes[int(match.group(1))] = int(entry["size"])
    return sizes


def _read_metadata_rows(metadata_path: Path) -> dict[str, int]:
    positions: dict[str, int] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            file_name = row["file"]
            if file_name in positions:
                raise ValueError(f"Metadata lặp file: {file_name}")
            positions[file_name] = row_index
    return positions


def _split_rows(
    splits_dir: Path, split: str, positions: Mapping[str, int]
) -> list[int]:
    rows: list[int] = []
    with (splits_dir / f"{split}.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            file_name = row["file"]
            if file_name not in positions:
                raise ValueError(
                    f"File {file_name} của {split} không có trong metadata."
                )
            rows.append(positions[file_name])
    return rows


def _local_shard_indices(parquet_dir: Path, geometry: ShardGeometry) -> dict[int, Path]:
    pattern = re.compile(
        rf"^train-(\d{{5}})-of-{geometry.total_shards:05d}\.parquet$"
    )
    result: dict[int, Path] = {}
    for path in sorted(parquet_dir.glob("*.parquet")):
        match = pattern.match(path.name)
        if match is not None:
            result[int(match.group(1))] = path
    return result


def verify_geometry_against_local(
    geometry: ShardGeometry, parquet_dir: Path, metadata_path: Path
) -> dict[str, Any]:
    """Kiểm tra mỗi shard cục bộ chứa đúng các file mà công thức gán cho nó."""

    positions = _read_metadata_rows(metadata_path)
    expected: dict[int, set[str]] = {}
    for file_name, row_index in positions.items():
        expected.setdefault(shard_index_for_row(geometry, row_index), set()).add(
            file_name
        )
    local = _local_shard_indices(parquet_dir, geometry)
    mismatched: list[int] = []
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads = 1")
        for index, path in local.items():
            actual = {
                str(value)
                for (value,) in connection.execute(
                    "SELECT file FROM read_parquet(?)", [path.as_posix()]
                ).fetchall()
            }
            if actual != expected.get(index, set()):
                mismatched.append(index)
    finally:
        connection.close()
    return {
        "checked_shards": len(local),
        "mismatched_shards": sorted(mismatched),
        "passed": not mismatched and bool(local),
    }


def build_plan(
    *,
    geometry: ShardGeometry,
    metadata_path: Path,
    splits_dir: Path,
    parquet_dir: Path,
    tree_sizes: Mapping[int, int] | None,
    free_bytes: int,
    packed_bytes_per_sample: float,
    train_samples: int = 0,
) -> dict[str, Any]:
    """Tính shard cần, shard thiếu và ngân sách đĩa cho dev/test đầy đủ."""

    positions = _read_metadata_rows(metadata_path)
    if len(positions) != geometry.total_rows:
        raise ValueError(
            f"Metadata có {len(positions)} dòng, kỳ vọng {geometry.total_rows}."
        )
    local = _local_shard_indices(parquet_dir, geometry)
    split_info: dict[str, dict[str, Any]] = {}
    for split in EVALUATION_SPLITS:
        rows = _split_rows(splits_dir, split, positions)
        shards = sorted({shard_index_for_row(geometry, row) for row in rows})
        split_info[split] = {
            "rows": len(rows),
            "shards": shards,
            "local_shards": [index for index in shards if index in local],
            "missing_shards": [index for index in shards if index not in local],
        }
    dev_set = set(split_info["closed_dev"]["shards"])
    test_set = set(split_info["closed_test"]["shards"])
    union = sorted(dev_set | test_set)
    missing = [index for index in union if index not in local]
    missing_dev_only = [i for i in missing if i in dev_set and i not in test_set]
    missing_test_only = [i for i in missing if i in test_set and i not in dev_set]
    missing_both = [i for i in missing if i in dev_set and i in test_set]

    local_sizes = {index: path.stat().st_size for index, path in local.items()}
    if tree_sizes is not None and all(index in tree_sizes for index in missing):
        sizes = {index: int(tree_sizes[index]) for index in missing}
        size_source = "huggingface_tree"
    else:
        if not local_sizes:
            raise ValueError("Không có shard cục bộ để ước tính kích thước trung bình.")
        mean = sum(local_sizes.values()) / len(local_sizes)
        sizes = {index: round(mean) for index in missing}
        size_source = "local_mean_estimate"
    missing_bytes = sum(sizes.values())

    eval_rows = sum(info["rows"] for info in split_info.values())
    packed_eval = round(eval_rows * packed_bytes_per_sample)
    packed_train = round(train_samples * packed_bytes_per_sample)
    packed_total = packed_eval + packed_train
    largest_missing = max(sizes.values(), default=0)
    keep_total = missing_bytes + packed_total
    streaming_peak = packed_total + largest_missing
    usable_bytes = free_bytes * (1.0 - DISK_HEADROOM_FRACTION)
    return {
        "scope": "evaluation_shard_planning_not_scientific_result",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "D025",
        "repository": REPOSITORY,
        "revision": REVISION,
        "geometry": {
            "total_shards": geometry.total_shards,
            "total_rows": geometry.total_rows,
            "first_count": geometry.first_count,
            "first_rows": geometry.first_rows,
            "rest_count": geometry.rest_count,
            "rest_rows": geometry.rest_rows,
        },
        "local": {
            "shards": len(local),
            "size_bytes": sum(local_sizes.values()),
        },
        "splits": split_info,
        "required_union": {"shards": union, "count": len(union)},
        "missing": {
            "shards": missing,
            "count": len(missing),
            "dev_only_shards": missing_dev_only,
            "test_only_shards": missing_test_only,
            "both_shards": missing_both,
            "size_bytes": missing_bytes,
            "size_source": size_source,
            "test_only_and_both_size_bytes": sum(
                sizes[i] for i in missing_test_only + missing_both
            ),
            "largest_shard_bytes": largest_missing,
        },
        "disk_budget": {
            "free_bytes": free_bytes,
            "headroom_fraction": DISK_HEADROOM_FRACTION,
            "packed_bytes_per_sample": packed_bytes_per_sample,
            "packed_eval_bytes": packed_eval,
            "packed_train_bytes": packed_train,
            "keep_parquet_total_bytes": keep_total,
            "keep_parquet_fits": keep_total <= usable_bytes,
            "streaming_peak_bytes": streaming_peak,
            "streaming_fits": streaming_peak <= usable_bytes,
        },
    }


def _gib(value: float) -> str:
    return f"{value / 1024**3:,.2f} GiB".replace(",", "_").replace(".", ",").replace("_", ".")


def _join(values: Sequence[int]) -> str:
    return ", ".join(str(value) for value in values) if values else "(không)"


def render_markdown(plan: dict[str, Any]) -> str:
    missing = plan["missing"]
    budget = plan["disk_budget"]
    lines = [
        "# Kế hoạch shard cho dev-full/test-full (D025)",
        "",
        "- Phạm vi: lập kế hoạch tải và dung lượng; không tải shard, không đọc waveform, không có metric.",
        f"- Nguồn: `{plan['repository']}` revision `{plan['revision']}`.",
        f"- Shard cục bộ: **{plan['local']['shards']}** ({_gib(plan['local']['size_bytes'])}).",
        "",
        "## Shard theo split",
        "",
        "| Split | Dòng gốc | Shard cần | Đã có | Còn thiếu |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in EVALUATION_SPLITS:
        info = plan["splits"][split]
        lines.append(
            f"| `{split}` | {info['rows']:,} | {len(info['shards'])} | "
            f"{len(info['local_shards'])} | {len(info['missing_shards'])} |".replace(",", ".")
        )
    lines += [
        "",
        f"- Hợp dev + test cần **{plan['required_union']['count']}** shard; còn thiếu **{missing['count']}**.",
        f"- Chỉ dev: {len(missing['dev_only_shards'])} shard; chỉ test: {len(missing['test_only_shards'])}; cả hai: {len(missing['both_shards'])}.",
        f"- Dung lượng cần tải: **{_gib(missing['size_bytes'])}** (nguồn: `{missing['size_source']}`); riêng phần chứa test: {_gib(missing['test_only_and_both_size_bytes'])}.",
        f"- Danh sách shard thiếu: {_join(missing['shards'])}.",
        "",
        "## Ngân sách đĩa",
        "",
        f"- Dung lượng trống hiện tại: **{_gib(budget['free_bytes'])}**; chỉ coi là vừa nếu nhu cầu không vượt {100 * (1 - budget['headroom_fraction']):.0f}% số này (dự phòng {100 * budget['headroom_fraction']:.0f}%).",
        f"- Packed ước tính (train manifest + dev + test, trước khử trùng): {_gib(budget['packed_eval_bytes'] + budget['packed_train_bytes'])} với {budget['packed_bytes_per_sample']:.0f} byte/mẫu; số này rút từ PoC 24 mẫu nên chỉ là tham chiếu thô.",
        f"- Giữ Parquet đã tải cùng packed: **{_gib(budget['keep_parquet_total_bytes'])}** — {'vừa' if budget['keep_parquet_fits'] else 'KHÔNG vừa'} chỗ trống (có dự phòng).",
        f"- Đóng gói từng shard rồi xóa Parquet (đỉnh = packed + shard lớn nhất): **{_gib(budget['streaming_peak_bytes'])}** — {'vừa' if budget['streaming_fits'] else 'KHÔNG vừa'} chỗ trống. Phương án này cần quy trình build tăng dần chưa được triển khai.",
        "",
        "## Giới hạn",
        "",
        "- Công thức file→shard dựa trên thứ tự dòng của `vsasv_metadata.csv`; phải được kiểm lại trên shard mới tải trước khi dùng cho manifest.",
        "- Số mẫu là số dòng split gốc; khử trùng nội dung D025 sẽ loại thêm mẫu.",
        "- Quyết định nơi lưu, giữ hay xóa Parquet và việc tải cần xác nhận riêng của người dùng.",
        "",
    ]
    return "\n".join(lines)


def fetch_tree_entries() -> list[dict[str, Any]]:
    """Liệt kê tree `data/` của dataset ở revision ghim; chỉ lấy metadata."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Thiếu huggingface_hub để đọc tree Hugging Face.") from exc
    entries: list[dict[str, Any]] = []
    for item in HfApi().list_repo_tree(
        REPOSITORY, path_in_repo="data", repo_type="dataset", revision=REVISION
    ):
        lfs = getattr(item, "lfs", None)
        sha256 = getattr(lfs, "sha256", None)
        entries.append(
            {
                "path": item.path,
                "size": getattr(item, "size", 0),
                "type": "file" if hasattr(item, "size") else "directory",
                "sha256": sha256,
            }
        )
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--packed-report", type=Path, default=DEFAULT_PACKED_REPORT)
    parser.add_argument("--tree-cache", type=Path, default=DEFAULT_TREE_CACHE)
    parser.add_argument(
        "--fetch-tree",
        action="store_true",
        help="Đọc metadata tree từ Hugging Face (không tải shard) và lưu vào cache.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args()


def _packed_bytes_per_sample(report_path: Path) -> float:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return float(report["artifact"]["size_bytes"]) / int(report["protocol"]["samples"])


def main() -> int:
    args = parse_args()
    tree_sizes: dict[int, int] | None = None
    if args.fetch_tree:
        entries = fetch_tree_entries()
        args.tree_cache.parent.mkdir(parents=True, exist_ok=True)
        args.tree_cache.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.tree_cache.is_file():
        tree_sizes = parse_tree_sizes(
            json.loads(args.tree_cache.read_text(encoding="utf-8")), OFFICIAL_GEOMETRY
        )
        if len(tree_sizes) != OFFICIAL_GEOMETRY.total_shards:
            raise ValueError(
                f"Cache tree có {len(tree_sizes)} shard, kỳ vọng "
                f"{OFFICIAL_GEOMETRY.total_shards}."
            )
    verification = verify_geometry_against_local(
        OFFICIAL_GEOMETRY, args.parquet_dir, args.metadata
    )
    if not verification["passed"]:
        raise ValueError(
            "Công thức file→shard không khớp shard cục bộ: "
            f"{verification['mismatched_shards']}."
        )
    plan = build_plan(
        geometry=OFFICIAL_GEOMETRY,
        metadata_path=args.metadata,
        splits_dir=args.splits_dir,
        parquet_dir=args.parquet_dir,
        tree_sizes=tree_sizes,
        free_bytes=shutil.disk_usage(args.parquet_dir).free,
        packed_bytes_per_sample=_packed_bytes_per_sample(args.packed_report),
        train_samples=TRAIN_MANIFEST_SAMPLES,
    )
    plan["geometry_verification"] = verification
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(plan), encoding="utf-8")
    print(
        f"ĐẠT: cần {plan['required_union']['count']} shard, thiếu "
        f"{plan['missing']['count']} ({_gib(plan['missing']['size_bytes'])}, "
        f"{plan['missing']['size_source']})."
    )
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
