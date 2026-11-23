#!/usr/bin/env python3
"""SCALE2-02: Tải shard theo kế hoạch SCALE2-01, khử trùng D036 và lập manifest ×2.

Script thực hiện tuần tự:
1. Kiểm dung lượng ổ đĩa tối thiểu 30 GB.
2. Tải lần lượt 35 shard kế hoạch (và tải tiếp nếu sau khử trùng còn thiếu, trong trần
   120 shard / 10 GB) từ Hugging Face ở revision ghim, gọi `hf_hub_download` không
   truyền tham số `token` (dùng token đã lưu cục bộ theo mặc định thư viện như
   `ingest_evaluation_shards.py`), kiểm kích thước, SHA-256, đọc được bằng DuckDB,
   đúng số dòng geometry và khớp `vsasv_metadata.csv`.
3. Trích ứng viên `closed_train` mới từ shard tải về, băm D017, gộp với ứng viên cục
   bộ ở SCALE2-01 và khử trùng theo train hiện tại, dev-full, test-full, nội bộ.
4. Lập `data/manifests/development_scale2_v1.csv` (31.770 dòng = 15.885 train ×1 giữ nguyên
   thứ tự đầu + 15.885 mẫu mới chọn theo `SHA256("2026|file")` tăng dần).
5. Băm lại độc lập toàn bộ 31.770 waveform từ Parquet để kiểm chứng không trùng nội
   bộ và không trùng dev-full/test-full.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_development_manifest_content import (  # noqa: E402
    _hash_manifest_waveforms,
    read_manifest as read_v2_manifest,
)
from scripts.ingest_evaluation_shards import TreeEntry, load_tree  # noqa: E402
from scripts.make_development_manifest import (  # noqa: E402
    MANIFEST_COLUMNS,
    SOURCE_SNAPSHOT,
    UTT_TYPES,
    allocate_largest_remainder,
    read_source_csv,
    sha256_file,
    stable_key,
    write_manifest,
)
from scripts.plan_evaluation_shards import (  # noqa: E402
    OFFICIAL_GEOMETRY,
    REPOSITORY,
    REVISION,
    ShardGeometry,
    shard_filename,
    shard_index_for_row,
)
from scripts.plan_scale2_shards import (  # noqa: E402
    CURRENT_TRAIN_TOTAL,
    DEFAULT_AUDIT_JSON,
    DEFAULT_CLOSED_TRAIN_SPLIT,
    DEFAULT_CURRENT_MANIFEST,
    DEFAULT_EVAL_REPORT_JSON,
    DEFAULT_LEDGER_CSV,
    DEFAULT_METADATA,
    DEFAULT_PARQUET_DIR,
    DEFAULT_TREE_JSON,
    EXPECTED_D036_QUOTA_TOTAL,
    MAX_BYTES_CEILING_GB,
    MAX_BYTES_CEILING_GIB,
    MAX_SHARDS_CEILING,
    SCALE2_TRAIN_TOTAL,
    SEED,
    deduplicate_and_select_candidates,
    load_and_verify_hash_sources,
    order_digest,
    read_metadata_index,
    require,
    write_candidate_cache,
)
from src.data.content_hash import waveform_sha256  # noqa: E402

SCALE2_MANIFEST_VERSION = "development-scale2-v1"
MIN_FREE_BYTES = 30_000_000_000
MAX_DOWNLOAD_ATTEMPTS = 3
SHARD_HASH_CACHE_VERSION = 1

DEFAULT_SCALE2_PLAN_JSON = PROJECT_ROOT / "reports" / "scale2_shard_plan.json"
DEFAULT_BASELINE_CANDIDATES_CSV = PROJECT_ROOT / ".cache" / "scale2_01" / "local_candidates_scale2.csv"
DEFAULT_BASELINE_HASHED_JSON = PROJECT_ROOT / ".cache" / "scale2_01" / "local_candidates_hashed.json"
DEFAULT_DEV_FULL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "closed_dev_full_v1.csv"
DEFAULT_INCOMING_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "incoming_scale2"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "scale2_02"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "development_scale2_v1.csv"
DEFAULT_DOWNLOAD_JSON = PROJECT_ROOT / "reports" / "scale2_download.json"
DEFAULT_DOWNLOAD_MD = PROJECT_ROOT / "reports" / "scale2_download.md"
DEFAULT_MANIFEST_JSON = PROJECT_ROOT / "reports" / "development_manifest_scale2_v1.json"
DEFAULT_MANIFEST_MD = PROJECT_ROOT / "reports" / "development_manifest_scale2_v1.md"
DEFAULT_AUDIT_OUTPUT_JSON = (
    PROJECT_ROOT / "reports" / "development_manifest_scale2_v1_content_audit.json"
)

AUTH_ERROR_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "gated",
    "authentication",
    "invalid token",
    "missing token",
    "login",
    "forbidden",
)

DownloaderFn = Callable[[str, Path], Path]


def _display_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def expected_rows_for_shard(geometry: ShardGeometry, shard_index: int) -> int:
    """Số dòng kỳ vọng trong một shard dựa trên hình học chính thức."""
    require(0 <= shard_index < geometry.total_shards, f"Shard {shard_index} ngoài phạm vi.")
    if shard_index < geometry.first_count:
        return geometry.first_rows
    return geometry.rest_rows


def public_hf_downloader(shard_name: str, destination: Path) -> Path:
    """Tải đúng một shard ở revision ghim từ Hugging Face (không truyền tham số token, dùng cấu hình mặc định của huggingface_hub)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Thiếu thư viện huggingface_hub để tải shard.") from exc
    destination.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            REPOSITORY,
            f"data/{shard_name}",
            repo_type="dataset",
            revision=REVISION,
            local_dir=str(destination),
        )
    )


def _is_auth_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "gated" in name or "auth" in name:
        return True
    return any(marker in msg for marker in AUTH_ERROR_MARKERS)


def _cleanup_incoming(shard_name: str, incoming_dir: Path, candidate: Path | None = None) -> None:
    if candidate is not None:
        candidate.unlink(missing_ok=True)
    for rel in (incoming_dir / shard_name, incoming_dir / "data" / shard_name):
        rel.unlink(missing_ok=True)
    meta_dir = incoming_dir / ".cache" / "huggingface" / "download" / "data"
    if meta_dir.is_dir():
        for leftover in meta_dir.glob(f"{shard_name}*"):
            leftover.unlink(missing_ok=True)


def verify_parquet_shard(
    *,
    shard_path: Path,
    expected_size: int,
    expected_sha256: str,
    expected_row_count: int,
    expected_metadata_rows: Mapping[str, tuple[str, str]],
) -> str:
    """Kiểm kích thước, SHA-256, khả năng đọc DuckDB, số dòng geometry và metadata shard."""
    require(shard_path.is_file(), f"Không tìm thấy shard Parquet: {shard_path}")
    actual_size = shard_path.stat().st_size
    require(
        actual_size == expected_size,
        f"Sai kích thước shard {shard_path.name}: nhận {actual_size} B, kỳ vọng {expected_size} B.",
    )
    actual_sha = sha256_file(shard_path)
    require(
        actual_sha == expected_sha256,
        f"Sai SHA-256 shard {shard_path.name}: nhận {actual_sha}, kỳ vọng {expected_sha256}.",
    )
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads = 1")
        rows = connection.execute(
            "SELECT file, label, utt_type FROM read_parquet(?)",
            [shard_path.resolve().as_posix()],
        ).fetchall()
    finally:
        connection.close()

    require(
        len(rows) == expected_row_count,
        f"Shard {shard_path.name} có {len(rows)} dòng, kỳ vọng {expected_row_count} dòng.",
    )
    seen: dict[str, tuple[str, str]] = {}
    for file_name, speaker_id, utt_type in rows:
        key = str(file_name)
        require(key not in seen, f"Shard {shard_path.name} bị lặp file: {key}")
        seen[key] = (str(speaker_id), str(utt_type))

    require(
        len(expected_metadata_rows) == expected_row_count,
        f"Số dòng metadata kỳ vọng cho {shard_path.name} là {len(expected_metadata_rows)}, "
        f"khác {expected_row_count}.",
    )
    require(
        seen == dict(expected_metadata_rows),
        f"Nội dung (file, label, utt_type) trong shard {shard_path.name} không khớp vsasv_metadata.csv.",
    )
    return actual_sha


def download_and_verify_shard(
    *,
    order_position: int,
    shard_index: int,
    shard_name: str,
    entry: TreeEntry,
    expected_row_count: int,
    expected_metadata_rows: Mapping[str, tuple[str, str]],
    data_dir: Path,
    incoming_dir: Path,
    downloader: DownloaderFn,
    download_log_path: Path,
    adaptive_extra: bool = False,
    max_attempts: int = MAX_DOWNLOAD_ATTEMPTS,
) -> dict[str, Any]:
    """Tải (nếu chưa có), kiểm đủ điều kiện và chuyển shard vào `data_dir`."""
    data_dir.mkdir(parents=True, exist_ok=True)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    download_log_path.parent.mkdir(parents=True, exist_ok=True)

    target_path = data_dir / shard_name
    started = time.perf_counter()

    def _append_log(record: dict[str, Any]) -> None:
        with download_log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if target_path.is_file():
        try:
            verify_parquet_shard(
                shard_path=target_path,
                expected_size=entry.size,
                expected_sha256=entry.sha256,
                expected_row_count=expected_row_count,
                expected_metadata_rows=expected_metadata_rows,
            )
            elapsed = round(time.perf_counter() - started, 3)
            rec = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "order_position": order_position,
                "shard_index": shard_index,
                "shard": shard_name,
                "size_bytes": entry.size,
                "sha256": entry.sha256,
                "expected_rows": expected_row_count,
                "adaptive_extra": adaptive_extra,
                "status": "reused_verified",
                "attempts": 0,
                "seconds": elapsed,
            }
            _append_log(rec)
            return rec
        except (OSError, ValueError, duckdb.Error) as exc:
            raise ValueError(
                f"Shard {shard_name} đã tồn tại ở thư mục data/ nhưng không hợp lệ: {exc}"
            ) from exc

    last_error: str = ""
    attempts = 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        candidate: Path | None = None
        try:
            candidate = downloader(shard_name, incoming_dir)
            verify_parquet_shard(
                shard_path=candidate,
                expected_size=entry.size,
                expected_sha256=entry.sha256,
                expected_row_count=expected_row_count,
                expected_metadata_rows=expected_metadata_rows,
            )
            os.replace(candidate, target_path)
            _cleanup_incoming(shard_name, incoming_dir)
            elapsed = round(time.perf_counter() - started, 3)
            rec = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "order_position": order_position,
                "shard_index": shard_index,
                "shard": shard_name,
                "size_bytes": entry.size,
                "sha256": entry.sha256,
                "expected_rows": expected_row_count,
                "adaptive_extra": adaptive_extra,
                "status": "downloaded_verified",
                "attempts": attempts,
                "seconds": elapsed,
            }
            _append_log(rec)
            return rec
        except Exception as exc:  # noqa: BLE001
            _cleanup_incoming(shard_name, incoming_dir, candidate=candidate)
            last_error = str(exc)
            if _is_auth_error(exc):
                _append_log(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "order_position": order_position,
                        "shard_index": shard_index,
                        "shard": shard_name,
                        "status": "failed_auth",
                        "attempts": attempts,
                        "error": last_error,
                    }
                )
                raise PermissionError(
                    f"Hugging Face yêu cầu xác thực/từ chối truy cập cho {shard_name}: {last_error}"
                ) from exc

    elapsed = round(time.perf_counter() - started, 3)
    _append_log(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "order_position": order_position,
            "shard_index": shard_index,
            "shard": shard_name,
            "status": "failed_after_retries",
            "attempts": attempts,
            "seconds": elapsed,
            "error": last_error,
        }
    )
    raise RuntimeError(
        f"Tải hoặc kiểm shard {shard_name} thất bại sau {attempts} lần thử: {last_error}"
    )


def hash_shard_train_candidates(
    *,
    shard_path: Path,
    shard_index: int,
    shard_sha256: str,
    candidate_rows: Sequence[Mapping[str, str]],
    cache_path: Path | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Băm content hash D017 cho các ứng viên `closed_train` mới trong một shard vừa tải."""
    if not candidate_rows:
        return [], True

    expected_by_file = {str(row["file"]): row for row in candidate_rows}
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("cache_version") == SHARD_HASH_CACHE_VERSION
            and cached.get("shard_index") == shard_index
            and cached.get("shard_sha256") == shard_sha256
            and cached.get("candidate_count") == len(expected_by_file)
            and isinstance(cached.get("candidates"), list)
        ):
            return list(cached["candidates"]), True

    results: list[dict[str, Any]] = []
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '8GB'")
    try:
        file_list = sorted(expected_by_file)
        seen: set[str] = set()
        cursor = connection.execute(
            """
            SELECT file, label, utt_type, audio.sampling_rate, audio.array
            FROM read_parquet(?)
            WHERE list_contains(?, file)
            ORDER BY file
            """,
            [shard_path.resolve().as_posix(), file_list],
        )
        while True:
            batch = cursor.fetchmany(32)
            if not batch:
                break
            for file_name, speaker_id, utt_type, sample_rate, waveform in batch:
                logical_file = str(file_name)
                expected = expected_by_file.get(logical_file)
                require(expected is not None, f"Đọc ngoài danh sách ứng viên: {logical_file}")
                require(logical_file not in seen, f"Shard {shard_path.name} lặp file {logical_file}")
                seen.add(logical_file)
                require(
                    str(speaker_id) == expected["speaker_id"]
                    and str(utt_type) == expected["utt_type"],
                    f"Metadata Parquet không khớp split/metadata cho {logical_file}",
                )
                rate = int(sample_rate or 0)
                require(rate > 0 and waveform is not None, f"Audio rỗng/lỗi: {logical_file}")
                digest = waveform_sha256(waveform, rate)
                results.append(
                    {
                        "file": logical_file,
                        "speaker_id": str(speaker_id),
                        "utt_type": str(utt_type),
                        "binary_label": 0 if str(utt_type) == "bonafide" else 1,
                        "shard": shard_path.name,
                        "shard_index": shard_index,
                        "native_sample_rate": rate,
                        "content_sha256": digest,
                    }
                )
        missing = sorted(set(expected_by_file) - seen)
        require(not missing, f"Thiếu {len(missing)} ứng viên trong {shard_path.name}: {missing[:3]}")
    finally:
        connection.close()

    results.sort(key=lambda item: item["file"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "cache_version": SHARD_HASH_CACHE_VERSION,
                    "shard_index": shard_index,
                    "shard": shard_path.name,
                    "shard_sha256": shard_sha256,
                    "candidate_count": len(results),
                    "candidates": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return results, False


def load_baseline_local_candidates(
    *,
    csv_path: Path,
    expected_files: set[str],
    expected_shards: set[str],
) -> list[dict[str, Any]]:
    """Nạp 6.882 ứng viên cục bộ đã băm từ SCALE2-01 và đối chiếu khớp hoàn toàn."""
    require(csv_path.is_file(), f"Không tìm thấy cache ứng viên SCALE2-01: {csv_path}")
    candidates: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            file_name = str(row["file"])
            require(file_name not in seen_files, f"Cache SCALE2-01 lặp file: {file_name}")
            seen_files.add(file_name)
            shard_name = str(row["shard"])
            require(shard_name in expected_shards, f"Shard trong cache không thuộc 68 shard gốc: {shard_name}")
            utt_type = str(row["utt_type"])
            require(utt_type in UTT_TYPES, f"utt_type không hợp lệ trong cache: {file_name}")
            digest = str(row["content_sha256"])
            require(len(digest) == 64, f"Content hash không hợp lệ trong cache: {file_name}")
            candidates.append(
                {
                    "file": file_name,
                    "speaker_id": str(row["speaker_id"]),
                    "utt_type": utt_type,
                    "binary_label": 0 if utt_type == "bonafide" else 1,
                    "shard": shard_name,
                    "native_sample_rate": int(row["native_sample_rate"]),
                    "content_sha256": digest,
                }
            )
    require(
        seen_files == expected_files,
        f"Tập file ứng viên trong {csv_path} ({len(seen_files)}) không khớp danh sách "
        f"ứng viên closed_train trên 68 shard ban đầu ({len(expected_files)}).",
    )
    return candidates


def read_baseline_local_shard_indices(
    *,
    hashed_cache_json: Path | None,
    parquet_dir: Path,
    geometry: ShardGeometry,
    expected_count: int | None = 68,
) -> set[int]:
    """Xác định đúng tập shard cục bộ ban đầu (trước khi tải mới để hỗ trợ chạy lại)."""
    if hashed_cache_json is not None and hashed_cache_json.is_file():
        payload = json.loads(hashed_cache_json.read_text(encoding="utf-8"))
        indices = {int(item["shard_index"]) for item in payload.get("shard_fingerprint", [])}
        if expected_count is None or len(indices) == expected_count:
            return indices
    indices = set()
    for path in sorted(parquet_dir.glob(f"train-*-of-{geometry.total_shards:05d}.parquet")):
        indices.add(int(path.name.split("-")[1]))
    if expected_count is not None:
        require(
            len(indices) == expected_count,
            f"Số shard cục bộ ban đầu không khớp {expected_count}: nhận {len(indices)}.",
        )
    return indices


def read_closed_train_prefix_rows(manifest_path: Path) -> list[dict[str, Any]]:
    """Đọc toàn bộ 15.885 dòng `closed_train` từ manifest ×1 theo đúng thứ tự gốc."""
    rows = read_v2_manifest(manifest_path)
    prefix = [row for row in rows if row["split"] == "closed_train"]
    require(
        len(prefix) == CURRENT_TRAIN_TOTAL,
        f"Manifest hiện tại có {len(prefix)} dòng closed_train, kỳ vọng {CURRENT_TRAIN_TOTAL}.",
    )
    return prefix


def build_scale2_manifest_rows(
    *,
    prefix_rows: Sequence[Mapping[str, str]],
    selected_new_files_by_utt: Mapping[str, Sequence[str]],
    candidate_by_file: Mapping[str, Mapping[str, Any]],
    additional_needed: Mapping[str, int],
    manifest_version: str = SCALE2_MANIFEST_VERSION,
    expected_total: int = SCALE2_TRAIN_TOTAL,
) -> list[dict[str, Any]]:
    """Ghép dòng train hiện tại (đầu) + dòng mới xếp theo utt_type & SHA256('2026|file')."""
    combined: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for row in prefix_rows:
        file_name = str(row["file"])
        require(file_name not in seen_files, f"Prefix lặp file: {file_name}")
        seen_files.add(file_name)
        combined.append(
            {
                "manifest_version": manifest_version,
                "shard": str(row["shard"]),
                "file": file_name,
                "speaker_id": str(row["speaker_id"]),
                "split": "closed_train",
                "binary_label": int(row["binary_label"]),
                "utt_type": str(row["utt_type"]),
                "native_sample_rate": int(row["native_sample_rate"]),
                "source_snapshot": SOURCE_SNAPSHOT,
            }
        )

    for utt in UTT_TYPES:
        chosen = list(selected_new_files_by_utt[utt])
        require(
            len(chosen) == int(additional_needed[utt]),
            f"Số mẫu mới chọn cho {utt} là {len(chosen)}, cần {additional_needed[utt]}.",
        )
        for file_name in chosen:
            require(file_name not in seen_files, f"Mẫu mới trùng file train ×1: {file_name}")
            seen_files.add(file_name)
            cand = candidate_by_file[file_name]
            require(cand["utt_type"] == utt, f"utt_type lệch cho mẫu mới {file_name}")
            combined.append(
                {
                    "manifest_version": manifest_version,
                    "shard": str(cand["shard"]),
                    "file": file_name,
                    "speaker_id": str(cand["speaker_id"]),
                    "split": "closed_train",
                    "binary_label": 0 if utt == "bonafide" else 1,
                    "utt_type": utt,
                    "native_sample_rate": int(cand["native_sample_rate"]),
                    "source_snapshot": SOURCE_SNAPSHOT,
                }
            )

    require(
        len(combined) == expected_total,
        f"Tổng số dòng manifest scale2 ({len(combined)}) khác {expected_total}.",
    )
    return combined


def audit_scale2_manifest_independently(
    *,
    manifest_path: Path,
    parquet_dir: Path,
    expected_prefix_rows: Sequence[Mapping[str, str]],
    expected_train_hashes_by_file: Mapping[str, str],
    dev_hashes: set[str],
    test_hashes: set[str],
    all_ledger_dev_hashes: set[str] | None = None,
    all_ledger_test_hashes: set[str] | None = None,
    closed_train_split_rows: Mapping[str, tuple[str, str]],
    metadata_by_file: Mapping[str, tuple[int, str, str]],
    dev_speakers: set[str],
    expected_quota: Mapping[str, int] = EXPECTED_D036_QUOTA_TOTAL,
    manifest_version: str = SCALE2_MANIFEST_VERSION,
) -> dict[str, Any]:
    """Đọc lại manifest từ đĩa và băm lại độc lập mọi waveform từ Parquet (D017)."""
    rows = read_v2_manifest(manifest_path)
    expected_total = sum(expected_quota.values())
    require(
        len(rows) == expected_total,
        f"Manifest có {len(rows)} dòng, kỳ vọng {expected_total}.",
    )
    require(
        rows[0]["manifest_version"] == manifest_version,
        f"manifest_version không đúng: {rows[0]['manifest_version']}",
    )
    require(
        rows[0]["source_snapshot"] == SOURCE_SNAPSHOT,
        f"source_snapshot không đúng: {rows[0]['source_snapshot']}",
    )

    for idx, row in enumerate(rows):
        file_name = row["file"]
        require(row["split"] == "closed_train", f"Dòng không phải closed_train: {file_name}")
        split_meta = closed_train_split_rows.get(file_name)
        require(split_meta is not None, f"File {file_name} không nằm trong split closed_train.")
        require(
            split_meta == (row["speaker_id"], row["utt_type"]),
            f"Split closed_train không khớp manifest cho {file_name}",
        )
        meta_entry = metadata_by_file.get(file_name)
        require(meta_entry is not None, f"File {file_name} không có trong metadata.")
        require(
            (meta_entry[1], meta_entry[2]) == (row["speaker_id"], row["utt_type"]),
            f"Metadata không khớp manifest cho {file_name}",
        )
        expected_binary = 0 if row["utt_type"] == "bonafide" else 1
        require(
            int(row["binary_label"]) == expected_binary,
            f"binary_label không khớp utt_type cho {file_name}",
        )
        require(
            row["speaker_id"] not in dev_speakers,
            f"Phát hiện speaker trùng với closed_dev: {row['speaker_id']}",
        )
        if idx < len(expected_prefix_rows):
            prefix_expected = expected_prefix_rows[idx]
            for col in ("shard", "file", "speaker_id", "split", "binary_label", "utt_type", "native_sample_rate"):
                require(
                    str(row[col]) == str(prefix_expected[col]),
                    f"Phần đầu train ×1 bị thay đổi tại dòng {idx + 2}, cột {col}: {file_name}",
                )

    utt_counts = Counter(row["utt_type"] for row in rows)
    require(
        {utt: utt_counts[utt] for utt in UTT_TYPES} == dict(expected_quota),
        f"Quota utt_type không khớp: {dict(utt_counts)}, kỳ vọng {dict(expected_quota)}.",
    )

    # Băm lại độc lập mọi waveform từ file Parquet (không dùng hash bước 3)
    hashes_by_file = _hash_manifest_waveforms(rows, parquet_dir)
    row_hashes: list[dict[str, Any]] = []
    seen_hashes: dict[str, list[str]] = defaultdict(list)
    dev_overlaps: list[str] = []
    test_overlaps: list[str] = []
    new_rows_all_dev_overlaps: list[str] = []
    new_rows_all_test_overlaps: list[str] = []
    prefix_hash_mismatches: list[str] = []

    for index, row in enumerate(rows, start=2):
        file_name = row["file"]
        digest = hashes_by_file[file_name]
        is_new_row = (index - 2) >= len(expected_prefix_rows)
        seen_hashes[digest].append(file_name)
        if digest in dev_hashes:
            dev_overlaps.append(file_name)
        if digest in test_hashes:
            test_overlaps.append(file_name)
        if is_new_row and all_ledger_dev_hashes is not None and digest in all_ledger_dev_hashes:
            new_rows_all_dev_overlaps.append(file_name)
        if is_new_row and all_ledger_test_hashes is not None and digest in all_ledger_test_hashes:
            new_rows_all_test_overlaps.append(file_name)
        expected_v2_hash = expected_train_hashes_by_file.get(file_name)
        if expected_v2_hash is not None and expected_v2_hash != digest:
            prefix_hash_mismatches.append(file_name)
        row_hashes.append(
            {
                "manifest_row": index,
                "manifest_version": row["manifest_version"],
                "shard": row["shard"],
                "file": file_name,
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "binary_label": int(row["binary_label"]),
                "utt_type": row["utt_type"],
                "native_sample_rate": int(row["native_sample_rate"]),
                "waveform_sha256": digest,
            }
        )

    duplicate_groups = [
        {"waveform_sha256": digest, "files": files}
        for digest, files in sorted(seen_hashes.items())
        if len(files) > 1
    ]
    require(
        not duplicate_groups,
        f"Audit độc lập phát hiện {len(duplicate_groups)} nhóm waveform_sha256 lặp trong manifest: "
        f"{duplicate_groups[0]['files'] if duplicate_groups else ''}",
    )
    require(
        not dev_overlaps,
        f"Audit độc lập phát hiện {len(dev_overlaps)} mẫu trùng content_sha256 với dev-full: {dev_overlaps[:3]}",
    )
    require(
        not test_overlaps,
        f"Audit độc lập phát hiện {len(test_overlaps)} mẫu trùng content_sha256 với test-full: {test_overlaps[:3]}",
    )
    require(
        not new_rows_all_dev_overlaps,
        f"Audit độc lập phát hiện {len(new_rows_all_dev_overlaps)} mẫu mới trùng content_sha256 với ledger closed_dev: "
        f"{new_rows_all_dev_overlaps[:3]}",
    )
    require(
        not new_rows_all_test_overlaps,
        f"Audit độc lập phát hiện {len(new_rows_all_test_overlaps)} mẫu mới trùng content_sha256 với ledger closed_test: "
        f"{new_rows_all_test_overlaps[:3]}",
    )
    require(
        not prefix_hash_mismatches,
        f"Hash của {len(prefix_hash_mismatches)} mẫu train ×1 không khớp audit D017 cũ: {prefix_hash_mismatches[:3]}",
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ĐẠT",
        "technical_passed": True,
        "scope": "Băm lại độc lập toàn bộ waveform của development_scale2_v1 từ Parquet (D017/D036).",
        "manifest": {
            "path": _display_path(manifest_path),
            "sha256": sha256_file(manifest_path),
            "version": manifest_version,
            "source_snapshot": SOURCE_SNAPSHOT,
            "samples": len(rows),
            "counts_by_split": {"closed_train": len(rows)},
            "counts_by_utt_type": {utt: utt_counts[utt] for utt in UTT_TYPES},
            "speakers": len({row["speaker_id"] for row in rows}),
            "shards": len({row["shard"] for row in rows}),
        },
        "hash_method": "sha256(sample_rate_signed_int64_le || waveform_float64_le)",
        "checks": {
            "expected_files": len(rows),
            "audited_files": len(row_hashes),
            "unique_content_hashes": len(seen_hashes),
            "duplicate_groups_total": len(duplicate_groups),
            "dev_full_hash_overlaps": len(dev_overlaps),
            "test_full_hash_overlaps": len(test_overlaps),
            "new_rows_all_ledger_dev_overlaps": len(new_rows_all_dev_overlaps),
            "new_rows_all_ledger_test_overlaps": len(new_rows_all_test_overlaps),
            "contains_exact_current_train_prefix": True,
            "current_train_hash_mismatches": len(prefix_hash_mismatches),
            "speaker_overlap_with_closed_dev": 0,
            "metadata_and_sample_rate_match": True,
        },
        "duplicate_groups": duplicate_groups,
        "row_hashes": row_hashes,
    }


def _build_download_report(
    *,
    status: str,
    disk_info: Mapping[str, Any],
    initial_target: int,
    acquired_records: Sequence[Mapping[str, Any]],
    acquired_bytes: int,
    max_shards_ceiling: int,
    max_bytes_ceiling: int,
    ceiling_error: str | None,
    stop_reason: str | None,
    hashing_seconds: float,
    failed_events: Sequence[Mapping[str, Any]] = (),
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Dựng dictionary báo cáo tải shard `scale2_download.json`."""
    adaptive_shards = [r for r in acquired_records if r.get("adaptive_extra")]
    return {
        "task": "SCALE2-02",
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": status,
        "stop_reason": stop_reason,
        "repository": REPOSITORY,
        "revision": REVISION,
        "disk_check": dict(disk_info),
        "summary": {
            "initial_plan_shards": initial_target,
            "total_shards_acquired": len(acquired_records),
            "downloaded_from_hf_count": sum(1 for r in acquired_records if r["status"] == "downloaded_verified"),
            "reused_verified_count": sum(1 for r in acquired_records if r["status"] == "reused_verified"),
            "adaptive_extra_shards_count": len(adaptive_shards),
            "downloaded_beyond_35": bool(adaptive_shards),
            "total_bytes_acquired": acquired_bytes,
            "total_gb": round(acquired_bytes / 1e9, 3),
            "total_gib": round(acquired_bytes / (1024**3), 3),
            "max_shards_ceiling": max_shards_ceiling,
            "max_bytes_ceiling_gb": max_bytes_ceiling,
            "max_bytes_ceiling_gib": MAX_BYTES_CEILING_GIB,
            "ceiling_exceeded": bool(ceiling_error),
            "ceiling_status": ceiling_error or "Đạt (không vượt trần 120 shard / 10 GB)",
            "total_download_and_verify_seconds": round(sum(float(r["seconds"]) for r in acquired_records), 3),
            "new_candidate_hashing_seconds": round(hashing_seconds, 3),
        },
        "shards": [dict(r) for r in acquired_records],
        "failed_events": [dict(ev) for ev in failed_events],
        "notes": [
            "Môi trường khóa không cài `pyarrow`, nên việc đọc Parquet và đối chiếu số dòng/cột với `vsasv_metadata.csv` thực hiện qua `duckdb` (`read_parquet`).",
            "Lần đầu dừng ở bước 2 vì truyền `token=False` vào gated repo; lần này gọi `hf_hub_download` không truyền tham số `token` để dùng token đã lưu cục bộ theo mặc định thư viện (giống `scripts/ingest_evaluation_shards.py`), không đọc hay ghi giá trị token.",
        ],
    }


def _render_download_markdown(report: Mapping[str, Any]) -> str:
    disk = report["disk_check"]
    summary = report["summary"]
    lines = [
        "# SCALE2-02 — Báo cáo tải shard mở rộng cho `development-scale2-v1` (D036)",
        "",
        f"- Thời điểm tạo (UTC): `{report['generated_at_utc']}`",
        f"- Trạng thái: **{report['status']}**",
    ]
    if report.get("stop_reason"):
        reason_flat = " ".join(str(report["stop_reason"]).splitlines()).strip()
        lines.append(f"- Lý do dừng: `{reason_flat}`")
    lines.extend(
        [
            f"- Nguồn: `{report['repository']}` revision `{report['revision']}`.",
            "- Xác thực tải: lần đầu dừng vì `token=False`; lần này dùng token đã lưu cục bộ theo mặc định thư viện (không truyền tham số `token`, không ghi giá trị token).",
            f"- Kiểm đĩa trước tải: trống `{disk['free_bytes']:,}` B (`{disk['free_gb']:.3f} GB` / `{disk['free_gib']:.3f} GiB`, ngưỡng tối thiểu `30 GB` → `{disk['meets_min_30gb']}`).",
            f"- Tổng shard đã tải/xác minh trong lần này: `{summary['total_shards_acquired']}` (kế hoạch ban đầu `{summary['initial_plan_shards']}`, tải bổ sung ngoài 35 shard: `{summary['adaptive_extra_shards_count']}` = **{summary['downloaded_beyond_35']}**).",
            f"- Tổng dung lượng shard tải mới: `{summary['total_bytes_acquired']:,}` B (`{summary['total_gb']:.3f} GB` / `{summary['total_gib']:.3f} GiB`).",
            f"- So với trần D036 (`120` shard / `10 GB`): **{summary['ceiling_status']}**.",
            "",
        ]
    )
    if report.get("failed_events"):
        lines.extend(
            [
                "## Sự kiện lỗi/dừng tải shard",
                "",
                "| Thời điểm (UTC) | Vị trí thứ tự | Shard | Trạng thái | Lần thử | Chi tiết |",
                "|---|---:|---|---|---:|---|",
            ]
        )
        for ev in report["failed_events"]:
            err = str(ev.get("error", "")).replace("\r", " ").replace("\n", " | ").replace("`", "'")
            lines.append(
                f"| `{ev.get('timestamp_utc', '')}` | {ev.get('order_position', '')} | "
                f"`{ev.get('shard', '')}` | `{ev.get('status', '')}` | {ev.get('attempts', '')} | `{err}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Danh sách shard đã tải/xác minh theo thứ tự `SHA256(\"2026|shard\")`",
            "",
            "| # | Shard | Kích thước (B) | SHA-256 | Trạng thái | Lần thử | Thời gian (s) | Bổ sung ngoài 35? |",
            "|---:|---|---:|---|---|---:|---:|---|",
        ]
    )
    if report["shards"]:
        for item in report["shards"]:
            lines.append(
                f"| {item['order_position']} | `{item['shard']}` | {item['size_bytes']:,} | "
                f"`{item['sha256']}` | `{item['status']}` | {item['attempts']} | "
                f"{item['seconds']:.3f} | {'Có' if item['adaptive_extra'] else 'Không'} |"
            )
    else:
        lines.append("| - | *(Chưa có shard mới nào tải thành công)* | 0 | - | - | 0 | 0.000 | Không |")
    if report.get("notes"):
        lines.extend(["", "## Ghi chú kỹ thuật", ""])
        for note in report["notes"]:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _render_manifest_markdown(report: Mapping[str, Any]) -> str:
    quota = report["quota"]
    dedup = report["deduplication"]
    man = report["manifest"]
    audit = report["independent_audit_summary"]
    lines = [
        "# Báo cáo development manifest scale2 (`development-scale2-v1`, D036)",
        "",
        f"- Thời điểm tạo (UTC): `{report['generated_at_utc']}`",
        f"- Trạng thái: **{report['status']}**",
        f"- Đường dẫn manifest: `{man['path']}`",
        f"- SHA-256 manifest: `{man['sha256']}`",
        f"- Số mẫu: `{man['samples']:,}` (`{quota['current_train_total']:,}` train ×1 giữ nguyên phần đầu + `{quota['additional_total']:,}` mới).",
        f"- Số speaker: `{man['speakers']:,}` (tăng từ `{quota['current_train_speakers']:,}` của train ×1); số shard được dùng: `{man['shards']}`.",
        f"- Quy tắc thứ tự dòng: {man['row_order_rule']}",
        "",
        "## 1. Thống kê ứng viên, khử trùng và số chọn theo `utt_type`",
        "",
        "| `utt_type` | Đã có (×1) | Phần thêm cần | Tổng ứng viên thô (cục bộ+tải) | Loại `train` | Loại `dev` | Loại `test` | Loại nội bộ | Ứng viên sạch | Số mới chọn | Tổng trong `scale2` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    drops = dedup["dropped_by_primary_reason"]
    for utt in UTT_TYPES:
        lines.append(
            f"| `{utt}` | {quota['current_train'][utt]:,} | {quota['additional_needed'][utt]:,} | "
            f"{dedup['raw_counts_by_utt_type'][utt]:,} | {drops['train_overlap'][utt]:,} | "
            f"{drops['dev_overlap'][utt]:,} | {drops['test_overlap'][utt]:,} | "
            f"{drops['internal_duplicate'][utt]:,} | {dedup['clean_counts_by_utt_type'][utt]:,} | "
            f"{quota['additional_selected'][utt]:,} | {man['counts_by_utt_type'][utt]:,} |"
        )
    lines.append(
        f"| **Tổng** | **{sum(quota['current_train'].values()):,}** | **{sum(quota['additional_needed'].values()):,}** | "
        f"**{dedup['total_raw']:,}** | **{sum(drops['train_overlap'].values()):,}** | "
        f"**{sum(drops['dev_overlap'].values()):,}** | **{sum(drops['test_overlap'].values()):,}** | "
        f"**{sum(drops['internal_duplicate'].values()):,}** | **{dedup['total_clean']:,}** | "
        f"**{sum(quota['additional_selected'].values()):,}** | **{man['samples']:,}** |"
    )

    lines.extend(
        [
            "",
            "## 2. Kết quả kiểm chứng độc lập từ Parquet (Bước 6)",
            "",
            f"- Audit file: `{audit['report_path']}` (thời gian băm lại `31.770` waveform: `{audit['audit_seconds']:.2f} s`).",
            f"- Đã băm lại: `{audit['checks']['audited_files']:,}/{audit['checks']['expected_files']:,}`; content hash duy nhất: `{audit['checks']['unique_content_hashes']:,}`.",
            f"- Trùng nội bộ: `{audit['checks']['duplicate_groups_total']}`; trùng `dev-full` (`kept`): `{audit['checks']['dev_full_hash_overlaps']}`; trùng `test-full` (`kept`): `{audit['checks']['test_full_hash_overlaps']}`.",
            f"- Nhóm mẫu mới (`15.885` dòng mới) trùng toàn bộ ledger `closed_dev` / `closed_test`: `{audit['checks'].get('new_rows_all_ledger_dev_overlaps', 0)}` / `{audit['checks'].get('new_rows_all_ledger_test_overlaps', 0)}`.",
            f"- Giữ nguyên trọn `15.885` mẫu train ×1 và khớp audit cũ: `{audit['checks']['contains_exact_current_train_prefix']}` (lệch hash: `{audit['checks']['current_train_hash_mismatches']}`).",
            f"- Speaker overlap với `closed_dev`: `{audit['checks']['speaker_overlap_with_closed_dev']}`; metadata & sample rate khớp: `{audit['checks']['metadata_and_sample_rate_match']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_dev_speakers(dev_manifest_path: Path) -> set[str]:
    if not dev_manifest_path.is_file():
        return set()
    speakers: set[str] = set()
    with dev_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            speakers.add(str(row["speaker_id"]))
    return speakers


def _load_kept_eval_hashes(ledger_csv_path: Path) -> tuple[set[str], set[str]]:
    """Lấy content_sha256 của đúng các dòng `decision == 'kept'` tạo nên dev-full và test-full."""
    kept_dev: set[str] = set()
    kept_test: set[str] = set()
    with ledger_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["decision"]) != "kept":
                continue
            digest = str(row["content_sha256"])
            if row["split"] == "closed_dev":
                kept_dev.add(digest)
            elif row["split"] == "closed_test":
                kept_test.add(digest)
    return kept_dev, kept_test


def _load_train_hashes_by_file(audit_json_path: Path) -> dict[str, str]:
    payload = json.loads(audit_json_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for row in payload.get("row_hashes", []):
        if row.get("split") == "closed_train":
            result[str(row["file"])] = str(row["waveform_sha256"])
    return result


def run_scale2_build(
    *,
    parquet_dir: Path = DEFAULT_PARQUET_DIR,
    incoming_dir: Path = DEFAULT_INCOMING_DIR,
    metadata_path: Path = DEFAULT_METADATA,
    closed_train_split_path: Path = DEFAULT_CLOSED_TRAIN_SPLIT,
    current_manifest_path: Path = DEFAULT_CURRENT_MANIFEST,
    audit_json_path: Path = DEFAULT_AUDIT_JSON,
    eval_report_json_path: Path = DEFAULT_EVAL_REPORT_JSON,
    ledger_csv_path: Path = DEFAULT_LEDGER_CSV,
    tree_json_path: Path = DEFAULT_TREE_JSON,
    baseline_candidates_csv: Path | None = DEFAULT_BASELINE_CANDIDATES_CSV,
    baseline_hashed_json: Path | None = DEFAULT_BASELINE_HASHED_JSON,
    dev_manifest_path: Path = DEFAULT_DEV_FULL_MANIFEST,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_manifest_path: Path = DEFAULT_OUTPUT_MANIFEST,
    download_json_path: Path = DEFAULT_DOWNLOAD_JSON,
    download_md_path: Path = DEFAULT_DOWNLOAD_MD,
    manifest_json_path: Path = DEFAULT_MANIFEST_JSON,
    manifest_md_path: Path = DEFAULT_MANIFEST_MD,
    audit_output_json_path: Path = DEFAULT_AUDIT_OUTPUT_JSON,
    geometry: ShardGeometry = OFFICIAL_GEOMETRY,
    expected_baseline_shards: int | None = 68,
    min_free_bytes: int = MIN_FREE_BYTES,
    verify_ledger_sha: bool = True,
    max_shards_ceiling: int = MAX_SHARDS_CEILING,
    max_bytes_ceiling: int = MAX_BYTES_CEILING_GB,
    initial_plan_shards: int = 35,
    downloader: DownloaderFn = public_hf_downloader,
    seed: int = SEED,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline SCALE2-02: kiểm đĩa, tải/khử trùng thích ứng, lập & audit manifest."""
    overall_started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    disk_usage = shutil.disk_usage(parquet_dir if parquet_dir.exists() else PROJECT_ROOT)
    disk_info = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "free_bytes": disk_usage.free,
        "free_gb": round(disk_usage.free / 1e9, 3),
        "free_gib": round(disk_usage.free / (1024**3), 3),
        "min_required_bytes": min_free_bytes,
        "meets_min_30gb": disk_usage.free >= min_free_bytes,
    }
    if not disk_info["meets_min_30gb"]:
        raise RuntimeError(
            f"Dung lượng trống ({disk_usage.free:,} B) thấp hơn ngưỡng tối thiểu {min_free_bytes:,} B."
        )

    hash_sources = load_and_verify_hash_sources(
        audit_json_path=audit_json_path,
        eval_report_json_path=eval_report_json_path,
        ledger_csv_path=ledger_csv_path,
        current_manifest_csv_path=current_manifest_path,
        verify_ledger_sha=verify_ledger_sha,
    )
    current_train_files: set[str] = hash_sources["current_train"]["files"]
    current_train_counts: dict[str, int] = hash_sources["current_train"]["counts_by_utt_type"]
    train_hashes: dict[str, int] = hash_sources["train_hashes"]
    dev_hashes: set[str] = hash_sources["dev_hashes"]
    test_hashes: set[str] = hash_sources["test_hashes"]

    metadata_index = read_metadata_index(metadata_path)
    require(
        len(metadata_index) == geometry.total_rows,
        f"Metadata có {len(metadata_index)} dòng, kỳ vọng {geometry.total_rows}.",
    )
    tree = load_tree(tree_json_path, geometry)

    metadata_by_shard: dict[int, dict[str, tuple[str, str]]] = defaultdict(dict)
    for file_name, (row_idx, spk, utt) in metadata_index.items():
        s_idx = shard_index_for_row(geometry, row_idx)
        metadata_by_shard[s_idx][file_name] = (spk, utt)

    split_rows = read_source_csv(closed_train_split_path)
    closed_train_split_map: dict[str, tuple[str, str]] = {}
    closed_train_counts = Counter({utt: 0 for utt in UTT_TYPES})
    candidates_by_shard: dict[int, list[dict[str, str]]] = defaultdict(list)

    for row in split_rows:
        file_name = row["file"]
        require(file_name not in closed_train_split_map, f"Split closed_train lặp file: {file_name}")
        closed_train_split_map[file_name] = (row["label"], row["utt_type"])
        meta = metadata_index.get(file_name)
        require(meta is not None, f"File trong closed_train không có ở metadata: {file_name}")
        row_idx, spk, utt = meta
        require((spk, utt) == (row["label"], row["utt_type"]), f"Metadata không khớp split cho {file_name}")
        closed_train_counts[utt] += 1
        if file_name in current_train_files:
            continue
        s_idx = shard_index_for_row(geometry, row_idx)
        candidates_by_shard[s_idx].append(
            {"file": file_name, "speaker_id": spk, "utt_type": utt}
        )

    scale2_total_target = 2 * len(current_train_files)
    scale2_quota = allocate_largest_remainder(scale2_total_target, dict(closed_train_counts), UTT_TYPES)
    if expected_baseline_shards == 68 and scale2_total_target == SCALE2_TRAIN_TOTAL:
        require(
            scale2_quota == EXPECTED_D036_QUOTA_TOTAL,
            f"Quota {scale2_quota} khác D036 {EXPECTED_D036_QUOTA_TOTAL}.",
        )
    additional_needed = {utt: scale2_quota[utt] - current_train_counts[utt] for utt in UTT_TYPES}
    require(all(val >= 0 for val in additional_needed.values()), "Phần thêm cần bị âm.")

    baseline_indices = read_baseline_local_shard_indices(
        hashed_cache_json=baseline_hashed_json,
        parquet_dir=parquet_dir,
        geometry=geometry,
        expected_count=expected_baseline_shards,
    )
    baseline_shard_names = {shard_filename(geometry, i) for i in baseline_indices}
    baseline_expected_files = {
        row["file"]
        for s_idx in baseline_indices
        for row in candidates_by_shard.get(s_idx, ())
    }
    if baseline_candidates_csv is not None and baseline_candidates_csv.is_file():
        baseline_candidates = load_baseline_local_candidates(
            csv_path=baseline_candidates_csv,
            expected_files=baseline_expected_files,
            expected_shards=baseline_shard_names,
        )
    else:
        require(
            expected_baseline_shards != 68,
            f"Thiếu cache ứng viên SCALE2-01 cho 68 shard ban đầu: {baseline_candidates_csv}",
        )
        baseline_candidates = []
        for s_idx in sorted(baseline_indices):
            s_name = shard_filename(geometry, s_idx)
            cands, _hit = hash_shard_train_candidates(
                shard_path=parquet_dir / s_name,
                shard_index=s_idx,
                shard_sha256=tree[s_idx].sha256,
                candidate_rows=candidates_by_shard.get(s_idx, ()),
                cache_path=None,
            )
            baseline_candidates.extend(cands)

    remote_indices = sorted(
        (idx for idx in range(geometry.total_shards) if idx not in baseline_indices),
        key=lambda idx: (order_digest(shard_filename(geometry, idx), seed), shard_filename(geometry, idx)),
    )

    download_log_path = cache_dir / "download_log.jsonl"
    shard_cache_dir = cache_dir / "shard_candidates"
    acquired_records: list[dict[str, Any]] = []
    acquired_bytes = 0
    new_candidates: list[dict[str, Any]] = []
    hashing_seconds = 0.0

    def _acquire_at(pos_zero_based: int, adaptive_extra: bool) -> None:
        nonlocal acquired_bytes, hashing_seconds
        idx = remote_indices[pos_zero_based]
        name = shard_filename(geometry, idx)
        entry = tree[idx]
        if len(acquired_records) + 1 > max_shards_ceiling:
            raise OverflowError(
                f"Chạm trần số shard ({max_shards_ceiling}) khi chuẩn bị tải shard thứ "
                f"{len(acquired_records) + 1} ({name})."
            )
        if acquired_bytes + entry.size > max_bytes_ceiling:
            raise OverflowError(
                f"Chạm trần dung lượng ({max_bytes_ceiling:,} B) khi chuẩn bị tải {name} "
                f"(đang có {acquired_bytes:,} B + {entry.size:,} B)."
            )
        rec = download_and_verify_shard(
            order_position=pos_zero_based + 1,
            shard_index=idx,
            shard_name=name,
            entry=entry,
            expected_row_count=expected_rows_for_shard(geometry, idx),
            expected_metadata_rows=metadata_by_shard[idx],
            data_dir=parquet_dir,
            incoming_dir=incoming_dir,
            downloader=downloader,
            download_log_path=download_log_path,
            adaptive_extra=adaptive_extra,
        )
        acquired_records.append(rec)
        acquired_bytes += entry.size

        t0 = time.perf_counter()
        cands, _hit = hash_shard_train_candidates(
            shard_path=parquet_dir / name,
            shard_index=idx,
            shard_sha256=entry.sha256,
            candidate_rows=candidates_by_shard.get(idx, ()),
            cache_path=shard_cache_dir / f"shard-{idx:05d}.json",
        )
        hashing_seconds += time.perf_counter() - t0
        new_candidates.extend(cands)

    ceiling_error: str | None = None
    stop_error: Exception | None = None
    stop_status = "ĐẠT"
    stop_reason: str | None = None
    initial_target = min(initial_plan_shards, len(remote_indices))
    try:
        for pos in range(initial_target):
            _acquire_at(pos, adaptive_extra=False)

        dedup_result = deduplicate_and_select_candidates(
            baseline_candidates + new_candidates,
            train_hashes=train_hashes,
            dev_hashes=dev_hashes,
            test_hashes=test_hashes,
            additional_needed=additional_needed,
            seed=seed,
        )

        next_pos = initial_target
        while dedup_result["total_shortfall"] > 0:
            if next_pos >= len(remote_indices):
                raise OverflowError("Đã hết toàn bộ shard từ xa mà vẫn thiếu quota sau khử trùng.")
            _acquire_at(next_pos, adaptive_extra=True)
            next_pos += 1
            dedup_result = deduplicate_and_select_candidates(
                baseline_candidates + new_candidates,
                train_hashes=train_hashes,
                dev_hashes=dev_hashes,
                test_hashes=test_hashes,
                additional_needed=additional_needed,
                seed=seed,
            )
    except OverflowError as exc:
        ceiling_error = str(exc)
        stop_error = exc
        stop_status = "STOPPED_CEILING_EXCEEDED"
        stop_reason = str(exc)
    except PermissionError as exc:
        stop_error = exc
        stop_status = "STOPPED_HF_AUTH_REQUIRED"
        stop_reason = str(exc)
    except RuntimeError as exc:
        stop_error = exc
        stop_status = "STOPPED_DOWNLOAD_ERROR"
        stop_reason = str(exc)

    failed_events: list[dict[str, Any]] = []
    if stop_error is not None and download_log_path.is_file():
        for line in download_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("status", "")).startswith("failed"):
                failed_events.append(item)

    initial_download_timing: dict[str, float] | None = None
    if (
        download_json_path.is_file()
        and acquired_records
        and all(r["status"] == "reused_verified" for r in acquired_records)
    ):
        prev_download = json.loads(download_json_path.read_text(encoding="utf-8"))
        prev_summary = prev_download.get("summary", {})
        if (
            int(prev_summary.get("downloaded_from_hf_count", 0)) == len(acquired_records)
            and int(prev_summary.get("total_shards_acquired", 0)) == len(acquired_records)
        ):
            initial_download_timing = {
                "initial_download_and_verify_seconds": float(
                    prev_summary.get("total_download_and_verify_seconds", 0.0)
                ),
                "initial_new_candidate_hashing_seconds": float(
                    prev_summary.get("new_candidate_hashing_seconds", 0.0)
                ),
            }

    download_report = _build_download_report(
        status=stop_status,
        disk_info=disk_info,
        initial_target=initial_target,
        acquired_records=acquired_records,
        acquired_bytes=acquired_bytes,
        max_shards_ceiling=max_shards_ceiling,
        max_bytes_ceiling=max_bytes_ceiling,
        ceiling_error=ceiling_error,
        stop_reason=stop_reason,
        hashing_seconds=hashing_seconds,
        failed_events=failed_events,
    )
    download_json_path.parent.mkdir(parents=True, exist_ok=True)
    download_json_path.write_text(
        json.dumps(download_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    download_md_path.parent.mkdir(parents=True, exist_ok=True)
    download_md_path.write_text(_render_download_markdown(download_report), encoding="utf-8")

    if isinstance(stop_error, OverflowError):
        raise RuntimeError(ceiling_error) from stop_error
    if stop_error is not None:
        raise stop_error

    # Bước 5: lập manifest development_scale2_v1.csv và lưu cache ứng viên hợp nhất
    write_candidate_cache(cache_dir, dedup_result)
    prefix_rows = read_closed_train_prefix_rows(current_manifest_path) if expected_baseline_shards == 68 else [
        row for row in read_v2_manifest(current_manifest_path) if row["split"] == "closed_train"
    ]
    all_candidates = baseline_candidates + new_candidates
    candidate_by_file = {str(cand["file"]): cand for cand in all_candidates}
    combined_rows = build_scale2_manifest_rows(
        prefix_rows=prefix_rows,
        selected_new_files_by_utt=dedup_result["selected_local_files_by_utt_type"],
        candidate_by_file=candidate_by_file,
        additional_needed=additional_needed,
        manifest_version=SCALE2_MANIFEST_VERSION,
        expected_total=scale2_total_target,
    )
    write_manifest(output_manifest_path, combined_rows)

    # Bước 6: kiểm chứng độc lập từ Parquet
    audit_start = time.perf_counter()
    kept_dev_hashes, kept_test_hashes = _load_kept_eval_hashes(ledger_csv_path)
    audit_report = audit_scale2_manifest_independently(
        manifest_path=output_manifest_path,
        parquet_dir=parquet_dir,
        expected_prefix_rows=prefix_rows,
        expected_train_hashes_by_file=_load_train_hashes_by_file(audit_json_path),
        dev_hashes=kept_dev_hashes,
        test_hashes=kept_test_hashes,
        all_ledger_dev_hashes=dev_hashes,
        all_ledger_test_hashes=test_hashes,
        closed_train_split_rows=closed_train_split_map,
        metadata_by_file=metadata_index,
        dev_speakers=_load_dev_speakers(dev_manifest_path),
        expected_quota=scale2_quota,
        manifest_version=SCALE2_MANIFEST_VERSION,
    )
    audit_seconds = round(time.perf_counter() - audit_start, 3)
    audit_output_json_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output_json_path.write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Lưu danh sách ứng viên hợp nhất vào cache SCALE2-02
    selected_new_counts = {
        utt: len(dedup_result["selected_local_files_by_utt_type"][utt]) for utt in UTT_TYPES
    }
    manifest_summary = audit_report["manifest"]
    timing_seconds: dict[str, Any] = {
        "download_and_verify_shards": round(sum(r["seconds"] for r in acquired_records), 3),
        "hash_new_candidates": round(hashing_seconds, 3),
        "independent_audit": audit_seconds,
        "total_elapsed": round(time.perf_counter() - overall_started, 3),
    }
    if initial_download_timing is not None:
        timing_seconds.update(initial_download_timing)
    manifest_report = {
        "task": "SCALE2-02",
        "decision": "D036",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ĐẠT",
        "technical_passed": True,
        "manifest_version": SCALE2_MANIFEST_VERSION,
        "source_snapshot": SOURCE_SNAPSHOT,
        "seed": seed,
        "quota": {
            "current_train_total": len(prefix_rows),
            "current_train_speakers": len({str(r["speaker_id"]) for r in prefix_rows}),
            "additional_total": sum(additional_needed.values()),
            "scale2_total": manifest_summary["samples"],
            "current_train": current_train_counts,
            "scale2_quota": scale2_quota,
            "additional_needed": additional_needed,
            "additional_selected": selected_new_counts,
        },
        "deduplication": {
            key: value
            for key, value in dedup_result.items()
            if key not in {"candidate_rows", "selected_local_files_by_utt_type"}
        },
        "manifest": {
            **manifest_summary,
            "row_order_rule": (
                "Dòng 1..15.885 giữ nguyên thứ tự gốc của closed_train trong development_20k_v2.csv "
                "(để bảo toàn source_index và _crop_seed); dòng 15.886..31.770 là 15.885 mẫu mới xếp "
                "theo utt_type (bonafide, voice_conversion, adversarial_attack, replay) và SHA256('2026|file') tăng dần."
            ),
        },
        "independent_audit_summary": {
            "report_path": _display_path(audit_output_json_path),
            "audit_seconds": audit_seconds,
            "checks": audit_report["checks"],
        },
        "timing_seconds": timing_seconds,
    }
    manifest_json_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json_path.write_text(
        json.dumps(manifest_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_md_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_md_path.write_text(_render_manifest_markdown(manifest_report), encoding="utf-8")

    return {
        "download_report": download_report,
        "manifest_report": manifest_report,
        "audit_report": audit_report,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--incoming-dir", type=Path, default=DEFAULT_INCOMING_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        result = run_scale2_build(
            parquet_dir=args.parquet_dir,
            incoming_dir=args.incoming_dir,
            cache_dir=args.cache_dir,
            output_manifest_path=args.output_manifest,
        )
        man = result["manifest_report"]["manifest"]
        dl = result["download_report"]["summary"]
        print(
            json.dumps(
                {
                    "status": "ok",
                    "total_shards_acquired": dl["total_shards_acquired"],
                    "downloaded_beyond_35": dl["downloaded_beyond_35"],
                    "total_bytes_acquired": dl["total_bytes_acquired"],
                    "manifest_path": man["path"],
                    "manifest_samples": man["samples"],
                    "manifest_sha256": man["sha256"],
                    "manifest_speakers": man["speakers"],
                    "manifest_shards": man["shards"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"LỖI SCALE2-02: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
