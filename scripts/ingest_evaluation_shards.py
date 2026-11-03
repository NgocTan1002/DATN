#!/usr/bin/env python3
"""Điều phối tải → xác minh → nạp → xóa Parquet cho dev-full/test-full (D025).

Mặc định chỉ in kế hoạch (dry-run). Phải truyền `--execute` thì script mới tải shard.
Mỗi shard tải về được kiểm kích thước/SHA-256 theo tree Hugging Face, nạp thành chunk
bằng `src.data.eval_ingest` và chỉ xóa sau khi chunk qua kiểm tra; nếu nạp lỗi, bản tải
đúng SHA-256 được giữ lại để chẩn đoán và dùng lại ở lần chạy sau. Shard đã có sẵn trong
thư mục cục bộ chỉ được nạp, không bao giờ bị xóa. Có thể ngắt và chạy lại: shard đã có
done marker được bỏ qua.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plan_evaluation_shards import (  # noqa: E402
    EVALUATION_SPLITS,
    OFFICIAL_GEOMETRY,
    REPOSITORY,
    REVISION,
    SHARD_PATTERN,
    ShardGeometry,
    _read_metadata_rows,
    shard_filename,
    shard_index_for_row,
)
from src.data.eval_ingest import (  # noqa: E402
    WantedRecord,
    chunk_paths,
    process_shard,
)


DEFAULT_LOCAL_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "incoming_eval"
DEFAULT_CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "eval_chunks"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_TREE_CACHE = PROJECT_ROOT / "reports" / "vsasv_shard_tree.json"
DEFAULT_PLAN = PROJECT_ROOT / "reports" / "evaluation_shard_plan.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "evaluation_ingest_summary.json"
DOWNLOAD_ATTEMPTS = 2

Downloader = Callable[[str, Path], Path]


@dataclass(frozen=True)
class TreeEntry:
    size: int
    sha256: str


def load_tree(cache_path: Path, geometry: ShardGeometry) -> dict[int, TreeEntry]:
    """Đọc kích thước và SHA-256 từng shard từ cache tree Hugging Face."""

    tree: dict[int, TreeEntry] = {}
    for entry in json.loads(cache_path.read_text(encoding="utf-8")):
        match = SHARD_PATTERN.match(str(entry.get("path", "")))
        if match is None:
            continue
        if int(match.group(2)) != geometry.total_shards:
            raise ValueError("Tree có tổng số shard không khớp cấu hình.")
        if not entry.get("sha256"):
            raise ValueError(f"Tree thiếu SHA-256 cho {entry['path']}.")
        tree[int(match.group(1))] = TreeEntry(
            size=int(entry["size"]), sha256=str(entry["sha256"])
        )
    return tree


def build_wanted_by_shard(
    *, geometry: ShardGeometry, metadata_path: Path, splits_dir: Path
) -> dict[int, dict[str, WantedRecord]]:
    """Nhóm các dòng closed_dev/closed_test theo shard Hugging Face chứa chúng."""

    positions = _read_metadata_rows(metadata_path)
    wanted: dict[int, dict[str, WantedRecord]] = {}
    for split in EVALUATION_SPLITS:
        with (splits_dir / f"{split}.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                file_name = row["file"]
                if file_name not in positions:
                    raise ValueError(f"File {file_name} của {split} không có trong metadata.")
                shard = shard_index_for_row(geometry, positions[file_name])
                bucket = wanted.setdefault(shard, {})
                if file_name in bucket:
                    raise ValueError(f"File {file_name} xuất hiện ở nhiều split đánh giá.")
                bucket[file_name] = WantedRecord(
                    file=file_name,
                    split=split,
                    speaker_id=row["label"],
                    utt_type=row["utt_type"],
                )
    return wanted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _intact(path: Path, entry: TreeEntry) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry.size
        and _sha256_file(path) == entry.sha256
    )


def _download_candidates(download_dir: Path, filename: str) -> tuple[Path, ...]:
    # hf_hub_download với local_dir giữ tiền tố "data/" của repo.
    return (download_dir / "data" / filename, download_dir / filename)


def _cleanup_download(path: Path, download_dir: Path) -> None:
    path.unlink(missing_ok=True)
    metadata_dir = download_dir / ".cache" / "huggingface" / "download" / "data"
    if metadata_dir.is_dir():
        for leftover in metadata_dir.glob(f"{path.name}*"):
            leftover.unlink(missing_ok=True)


def hf_downloader(filename: str, destination: Path) -> Path:
    """Tải đúng một shard ở revision ghim bằng huggingface_hub."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Thiếu huggingface_hub để tải shard.") from exc
    destination.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            REPOSITORY,
            f"data/{filename}",
            repo_type="dataset",
            revision=REVISION,
            local_dir=str(destination),
        )
    )


def run_ingest(
    *,
    geometry: ShardGeometry,
    wanted_by_shard: Mapping[int, Mapping[str, WantedRecord]],
    shard_indices: Sequence[int],
    local_dir: Path,
    download_dir: Path,
    chunk_dir: Path,
    tree: Mapping[int, TreeEntry],
    downloader: Downloader,
    execute: bool,
    reserve_bytes: int,
    bytes_per_sample: float,
    free_bytes_fn: Callable[[Path], int],
    max_consecutive_failures: int = 3,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Xử lý lần lượt các shard; dry-run chỉ lập danh sách hành động."""

    say = progress or (lambda _message: None)
    for index in shard_indices:
        if index not in wanted_by_shard or not wanted_by_shard[index]:
            raise ValueError(f"Shard {index} không chứa mẫu dev/test cần nạp.")
        if index not in tree:
            raise ValueError(f"Thiếu thông tin tree Hugging Face cho shard {index}.")

    plan: list[tuple[int, str]] = []
    for index in shard_indices:
        if chunk_paths(chunk_dir, index).done.is_file():
            action = "skip_done"
        elif (local_dir / shard_filename(geometry, index)).is_file():
            action = "ingest_local"
        else:
            action = "download_ingest_delete"
        plan.append((index, action))
    results = [{"shard": i, "action": a, "status": "not_run"} for i, a in plan]
    summary: dict[str, Any] = {
        "executed": execute,
        "shards": results,
        "ingested": 0,
        "skipped": 0,
        "failed": 0,
        "stopped_reason": None,
    }
    if not execute:
        return summary

    consecutive_failures = 0
    for position, (index, action) in enumerate(plan):
        result = results[position]
        wanted = wanted_by_shard[index]
        entry = tree[index]
        name = shard_filename(geometry, index)
        if action == "skip_done":
            process_shard(
                shard_path=local_dir / name,
                shard_index=index,
                wanted=wanted,
                chunk_dir=chunk_dir,
                delete_after=False,
            )
            result["status"] = "already_done"
            summary["skipped"] += 1
            continue

        needed = (
            round(len(wanted) * bytes_per_sample)
            + reserve_bytes
            + (entry.size if action == "download_ingest_delete" else 0)
        )
        probe = download_dir if download_dir.exists() else download_dir.parent
        if free_bytes_fn(probe) < needed:
            summary["stopped_reason"] = "low_disk"
            say(f"Dừng ở shard {index}: không đủ dung lượng trống (cần {needed} byte).")
            break

        started = time.perf_counter()
        error: str | None = None
        try:
            if action == "ingest_local":
                process_shard(
                    shard_path=local_dir / name,
                    shard_index=index,
                    wanted=wanted,
                    chunk_dir=chunk_dir,
                    delete_after=False,
                    expected_size=entry.size,
                    expected_sha256=entry.sha256,
                )
            else:
                # Bản tải còn nguyên từ lần chạy lỗi trước được dùng lại, không tải lại.
                downloaded: Path | None = next(
                    (
                        path
                        for path in _download_candidates(download_dir, name)
                        if _intact(path, entry)
                    ),
                    None,
                )
                for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                    if downloaded is not None:
                        break
                    try:
                        candidate = downloader(name, download_dir)
                    except OSError as exc:
                        error = f"tải lỗi: {exc}"
                        continue
                    if _intact(candidate, entry):
                        downloaded = candidate
                        error = None
                        break
                    error = "shard tải về sai kích thước hoặc SHA-256"
                    _cleanup_download(candidate, download_dir)
                if downloaded is None:
                    raise ValueError(error or "tải shard thất bại")
                # Chỉ dọn sau khi chunk đã qua verify_chunk; nếu nạp lỗi thì giữ bản tải
                # đã đúng SHA-256 để chẩn đoán và để lần chạy sau dùng lại.
                process_shard(
                    shard_path=downloaded,
                    shard_index=index,
                    wanted=wanted,
                    chunk_dir=chunk_dir,
                    delete_after=True,
                    expected_size=entry.size,
                    expected_sha256=entry.sha256,
                )
                _cleanup_download(downloaded, download_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            error = str(exc)
        elapsed = round(time.perf_counter() - started, 2)
        if error is None:
            result.update(status="ingested", seconds=elapsed)
            summary["ingested"] += 1
            consecutive_failures = 0
            say(f"Shard {index}: {action} xong trong {elapsed} giây.")
        else:
            result.update(status="failed", error=error, seconds=elapsed)
            summary["failed"] += 1
            consecutive_failures += 1
            say(f"Shard {index}: LỖI — {error}")
            if consecutive_failures >= max_consecutive_failures:
                summary["stopped_reason"] = "too_many_failures"
                break
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--tree-cache", type=Path, default=DEFAULT_TREE_CACHE)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--shards", type=int, nargs="+", help="Chỉ xử lý các shard này.")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N shard đầu của danh sách.")
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Thực sự tải và nạp shard. Không có cờ này script chỉ in kế hoạch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    indices = list(args.shards) if args.shards else list(plan["required_union"]["shards"])
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit phải lớn hơn 0.")
        indices = indices[: args.limit]
    tree = load_tree(args.tree_cache, OFFICIAL_GEOMETRY)
    wanted = build_wanted_by_shard(
        geometry=OFFICIAL_GEOMETRY,
        metadata_path=args.metadata,
        splits_dir=args.splits_dir,
    )
    summary = run_ingest(
        geometry=OFFICIAL_GEOMETRY,
        wanted_by_shard=wanted,
        shard_indices=indices,
        local_dir=args.local_dir,
        download_dir=args.download_dir,
        chunk_dir=args.chunk_dir,
        tree=tree,
        downloader=hf_downloader,
        execute=args.execute,
        reserve_bytes=round(args.reserve_gib * 1024**3),
        bytes_per_sample=float(plan["disk_budget"]["packed_bytes_per_sample"]),
        free_bytes_fn=lambda path: shutil.disk_usage(path).free,
        max_consecutive_failures=args.max_consecutive_failures,
        progress=lambda message: print(message, flush=True),
    )
    counts: dict[str, int] = {}
    for item in summary["shards"]:
        counts[item["action"]] = counts.get(item["action"], 0) + 1
    summary["action_counts"] = counts
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    mode = "THỰC THI" if args.execute else "DRY-RUN"
    print(
        f"{mode}: {len(indices)} shard, hành động {counts}; nạp {summary['ingested']}, "
        f"bỏ qua {summary['skipped']}, lỗi {summary['failed']}, "
        f"dừng: {summary['stopped_reason']}."
    )
    return 0 if summary["failed"] == 0 and summary["stopped_reason"] is None else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
