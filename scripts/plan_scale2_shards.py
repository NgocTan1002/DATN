#!/usr/bin/env python3
"""SCALE2-01: Khử trùng ứng viên cục bộ và lập kế hoạch shard cho train ×2 (D036 mục 2–3).

Script chỉ đọc dữ liệu huấn luyện cục bộ và ledger hash D017 đã xác minh; không tải
mạng, không đọc hay giải mã audio `closed_dev`/`closed_test`, không dùng `utt_type`
hay `speaker_id` của `closed_test` và không sửa dữ liệu hay manifest hiện có.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_evaluation_shards import TreeEntry, load_tree  # noqa: E402
from scripts.make_development_manifest import (  # noqa: E402
    UTT_TYPES,
    allocate_largest_remainder,
    read_source_csv,
    sha256_file,
    stable_key,
)
from scripts.plan_evaluation_shards import (  # noqa: E402
    OFFICIAL_GEOMETRY,
    REPOSITORY,
    REVISION,
    ShardGeometry,
    shard_filename,
    shard_index_for_row,
)
from src.data.content_hash import waveform_sha256  # noqa: E402
from src.data.vsasv import MANIFEST_COLUMNS  # noqa: E402

SEED = 2026
CURRENT_TRAIN_TOTAL = 15_885
SCALE2_TRAIN_TOTAL = 31_770
EXPECTED_D036_QUOTA_TOTAL = {
    "bonafide": 14_267,
    "voice_conversion": 8_697,
    "adversarial_attack": 8_697,
    "replay": 109,
}
EXPECTED_LOCAL_SHARDS = 68
MAX_SHARDS_CEILING = 120
MAX_BYTES_CEILING_GB = 10_000_000_000  # 10 GB thập phân (chặt hơn 10 GiB = 10_737_418_240 B)
MAX_BYTES_CEILING_GIB = 10 * 1024**3
LEDGER_COLUMNS = ("file", "split", "speaker_id", "utt_type", "content_sha256", "decision")
LOCAL_CANDIDATE_CACHE_VERSION = 1

DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_CLOSED_TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "closed_train.csv"
DEFAULT_CURRENT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "development_20k_v2.csv"
DEFAULT_AUDIT_JSON = PROJECT_ROOT / "reports" / "development_manifest_20k_v2_content_audit.json"
DEFAULT_EVAL_REPORT_JSON = PROJECT_ROOT / "reports" / "closed_eval_full_v1.json"
DEFAULT_LEDGER_CSV = PROJECT_ROOT / "data" / "processed" / "eval_chunks" / "content_hash_ledger.csv"
DEFAULT_TREE_JSON = PROJECT_ROOT / "reports" / "vsasv_shard_tree.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "scale2_01"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "scale2_shard_plan.json"
DEFAULT_MD_REPORT = PROJECT_ROOT / "reports" / "scale2_shard_plan.md"

CANDIDATE_CSV_COLUMNS = (
    "file",
    "speaker_id",
    "utt_type",
    "shard",
    "native_sample_rate",
    "content_sha256",
    "order_sha256",
    "selection_status",
    "drop_reason",
    "in_train_current",
    "in_dev_all",
    "in_test_all",
    "kept_representative_file",
)


def require(condition: bool, message: str) -> None:
    """Kiểm tra điều kiện và ném ValueError tiếng Việt nếu không đạt."""
    if not condition:
        raise ValueError(message)


def order_digest(name: str, seed: int = SEED) -> str:
    """Tính SHA-256('{seed}|{name}') để sắp thứ tự file/shard xác định theo D036."""
    return hashlib.sha256(f"{seed}|{name}".encode("utf-8")).hexdigest()


def _display_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _format_bytes_gb(num_bytes: int) -> float:
    return round(num_bytes / 1_000_000_000, 3)


def _format_bytes_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 3)


def read_metadata_index(metadata_path: Path) -> dict[str, tuple[int, str, str]]:
    """Đọc toàn bộ metadata theo thứ tự dòng: file -> (row_index, speaker_id, utt_type)."""
    rows = read_source_csv(metadata_path)
    index: dict[str, tuple[int, str, str]] = {}
    for row_index, row in enumerate(rows):
        file_name = row["file"]
        require(file_name not in index, f"Metadata lặp file: {file_name}")
        require(row["utt_type"] in UTT_TYPES, f"utt_type không hợp lệ trong metadata: {file_name}")
        index[file_name] = (row_index, row["label"], row["utt_type"])
    return index


def read_current_train_manifest(manifest_path: Path) -> dict[str, Any]:
    """Đọc phân vùng `closed_train` của manifest train hiện tại (`development_20k_v2.csv`)."""
    require(manifest_path.is_file(), f"Không tìm thấy manifest hiện tại: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            tuple(reader.fieldnames or ()) == MANIFEST_COLUMNS,
            f"Schema manifest không hợp lệ: {manifest_path}",
        )
        train_files: set[str] = set()
        counts = Counter({utt: 0 for utt in UTT_TYPES})
        by_file: dict[str, dict[str, str]] = {}
        for row in reader:
            if row["split"] != "closed_train":
                continue
            file_name = row["file"]
            require(file_name not in train_files, f"Manifest lặp file closed_train: {file_name}")
            train_files.add(file_name)
            utt_type = row["utt_type"]
            require(utt_type in UTT_TYPES, f"utt_type không hợp lệ trong manifest: {file_name}")
            counts[utt_type] += 1
            by_file[file_name] = dict(row)
    return {
        "path": _display_path(manifest_path),
        "sha256": sha256_file(manifest_path),
        "files": train_files,
        "by_file": by_file,
        "counts_by_utt_type": {utt: counts[utt] for utt in UTT_TYPES},
        "total": len(train_files),
    }


def load_and_verify_hash_sources(
    *,
    audit_json_path: Path,
    eval_report_json_path: Path,
    ledger_csv_path: Path,
    current_manifest_csv_path: Path,
    verify_ledger_sha: bool = True,
) -> dict[str, Any]:
    """Nạp và kiểm chéo content hash D017 của train hiện tại, dev-full và test-full.

    Lưu ý bảo mật dữ liệu test: từ ledger `closed_test` chỉ đọc cột `split`,
    `content_sha256`, `decision` để loại trùng theo D036; không đọc hay dùng
    `utt_type` hoặc `speaker_id` của `closed_test`.
    """
    current_train = read_current_train_manifest(current_manifest_csv_path)
    require(audit_json_path.is_file(), f"Không tìm thấy audit content hash train: {audit_json_path}")
    audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    row_hashes = audit.get("row_hashes")
    require(isinstance(row_hashes, list) and len(row_hashes) > 0, "Audit thiếu row_hashes.")

    train_hashes: dict[str, int] = {}
    train_files_audited: set[str] = set()
    for row in row_hashes:
        if row.get("split") != "closed_train":
            continue
        file_name = str(row["file"])
        digest = str(row["waveform_sha256"])
        binary_label = int(row["binary_label"])
        require(len(digest) == 64, f"Content hash train không hợp lệ cho {file_name}.")
        expected_row = current_train["by_file"].get(file_name)
        require(expected_row is not None, f"Audit chứa file không thuộc manifest: {file_name}")
        require(
            int(expected_row["binary_label"]) == binary_label,
            f"Nhãn nhị phân audit không khớp manifest cho {file_name}",
        )
        if digest in train_hashes:
            require(
                train_hashes[digest] == binary_label,
                f"Content hash train trộn hai nhãn nhị phân: {digest}",
            )
        train_hashes[digest] = binary_label
        train_files_audited.add(file_name)

    require(
        train_files_audited == current_train["files"],
        "Tập file closed_train trong audit không khớp manifest hiện tại.",
    )

    require(eval_report_json_path.is_file(), f"Không tìm thấy report eval full: {eval_report_json_path}")
    eval_report = json.loads(eval_report_json_path.read_text(encoding="utf-8"))
    expected_ledger_sha = str(eval_report.get("ledger_sha256", ""))
    expected_ledger_rows = int(eval_report.get("ledger_rows", 0))
    require(ledger_csv_path.is_file(), f"Không tìm thấy ledger hash dev/test: {ledger_csv_path}")

    actual_ledger_sha = sha256_file(ledger_csv_path)
    if verify_ledger_sha:
        require(
            bool(expected_ledger_sha) and actual_ledger_sha == expected_ledger_sha,
            f"SHA-256 ledger lệch với closed_eval_full_v1.json: thực tế {actual_ledger_sha}, "
            f"kỳ vọng {expected_ledger_sha}.",
        )

    dev_hashes: set[str] = set()
    dev_kept_hashes: set[str] = set()
    test_hashes: set[str] = set()
    test_kept_hashes: set[str] = set()
    split_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()

    with ledger_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            tuple(reader.fieldnames or ()) == LEDGER_COLUMNS,
            f"Schema ledger không hợp lệ: {ledger_csv_path}",
        )
        total_rows = 0
        for row in reader:
            total_rows += 1
            split_name = row["split"]
            digest = row["content_sha256"]
            decision = row["decision"]
            require(len(digest) == 64, f"Content hash trong ledger không hợp lệ ở dòng {total_rows}.")
            split_counts[split_name] += 1
            decision_counts[decision] += 1
            if split_name == "closed_dev":
                dev_hashes.add(digest)
                if decision == "kept":
                    dev_kept_hashes.add(digest)
            elif split_name == "closed_test":
                # Chỉ lấy content_sha256 để loại trùng theo D036; không đọc utt_type/speaker_id.
                test_hashes.add(digest)
                if decision == "kept":
                    test_kept_hashes.add(digest)
            else:
                raise ValueError(f"Split không hợp lệ trong ledger: {split_name}")

    if verify_ledger_sha and expected_ledger_rows > 0:
        require(
            total_rows == expected_ledger_rows,
            f"Số dòng ledger ({total_rows}) không khớp closed_eval_full_v1.json ({expected_ledger_rows}).",
        )

    return {
        "current_train": current_train,
        "train_hashes": train_hashes,
        "dev_hashes": dev_hashes,
        "dev_kept_hashes": dev_kept_hashes,
        "test_hashes": test_hashes,
        "test_kept_hashes": test_kept_hashes,
        "ledger_summary": {
            "path": _display_path(ledger_csv_path),
            "sha256": actual_ledger_sha,
            "expected_sha256": expected_ledger_sha,
            "rows": total_rows,
            "split_counts": dict(sorted(split_counts.items())),
            "decision_counts": dict(sorted(decision_counts.items())),
            "unique_dev_hashes_all": len(dev_hashes),
            "unique_dev_hashes_kept": len(dev_kept_hashes),
            "unique_test_hashes_all": len(test_hashes),
            "unique_test_hashes_kept": len(test_kept_hashes),
            "dedup_policy_note": (
                "Loại trùng theo content_sha256 của MỌI dòng closed_dev và closed_test trong ledger "
                "(gồm cả kept lẫn internal_duplicate/overlap, vì mọi dòng đều là nội dung thuộc dev/test gốc); "
                "không đọc hay dùng cột utt_type/speaker_id của closed_test."
            ),
        },
        "audit_summary": {
            "path": _display_path(audit_json_path),
            "sha256": sha256_file(audit_json_path),
            "audited_current_train_rows": len(train_files_audited),
            "unique_current_train_hashes": len(train_hashes),
        },
    }


def verify_local_shards(
    *,
    parquet_dir: Path,
    tree: Mapping[int, TreeEntry],
    metadata_index: Mapping[str, tuple[int, str, str]],
    geometry: ShardGeometry = OFFICIAL_GEOMETRY,
    expected_local_shards: int | None = EXPECTED_LOCAL_SHARDS,
) -> dict[str, Any]:
    """Kiểm kích thước, SHA-256 và danh sách file của mọi shard cục bộ so với tree/metadata."""
    require(parquet_dir.is_dir(), f"Không tìm thấy thư mục shard cục bộ: {parquet_dir}")
    pattern = re.compile(rf"^train-(\d{{5}})-of-{geometry.total_shards:05d}\.parquet$")
    local_shards: dict[int, Path] = {}
    for path in sorted(parquet_dir.glob("*.parquet")):
        match = pattern.match(path.name)
        require(match is not None, f"File Parquet có tên không hợp lệ: {path.name}")
        idx = int(match.group(1))
        require(0 <= idx < geometry.total_shards, f"Chỉ số shard ngoài phạm vi: {idx}")
        local_shards[idx] = path

    if expected_local_shards is not None:
        require(
            len(local_shards) == expected_local_shards,
            f"Số shard cục bộ thực tế ({len(local_shards)}) khác kỳ vọng ({expected_local_shards}).",
        )
    else:
        require(len(local_shards) > 0, f"Không có shard Parquet nào trong {parquet_dir}")

    expected_files_by_shard: dict[int, set[str]] = defaultdict(set)
    for file_name, (row_idx, _spk, _utt) in metadata_index.items():
        expected_files_by_shard[shard_index_for_row(geometry, row_idx)].add(file_name)

    verified_details: dict[int, dict[str, Any]] = {}
    total_bytes = 0
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads = 1")
        for idx in sorted(local_shards):
            path = local_shards[idx]
            entry = tree.get(idx)
            require(entry is not None, f"Thiếu shard {idx} trong vsasv_shard_tree.json.")
            actual_size = path.stat().st_size
            require(
                actual_size == entry.size,
                f"Shard {path.name} lệch kích thước: cục bộ {actual_size} B, tree {entry.size} B.",
            )
            actual_sha = sha256_file(path)
            require(
                actual_sha == entry.sha256,
                f"Shard {path.name} lệch SHA-256: cục bộ {actual_sha}, tree {entry.sha256}.",
            )
            actual_files = {
                str(value)
                for (value,) in connection.execute(
                    "SELECT file FROM read_parquet(?)",
                    [path.resolve().as_posix()],
                ).fetchall()
            }
            require(
                actual_files == expected_files_by_shard.get(idx, set()),
                f"Danh sách file trong shard {path.name} không khớp vsasv_metadata.csv.",
            )
            total_bytes += actual_size
            verified_details[idx] = {
                "shard": path.name,
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "row_count": len(actual_files),
            }
    finally:
        connection.close()

    return {
        "local_shards": local_shards,
        "count": len(local_shards),
        "total_bytes": total_bytes,
        "all_size_and_sha_matched": True,
        "all_geometry_files_matched": True,
        "details_by_index": verified_details,
    }


def hash_local_train_candidates(
    *,
    local_shards: Mapping[int, Path],
    shard_details: Mapping[int, Mapping[str, Any]],
    candidate_rows_by_shard: Mapping[int, Sequence[Mapping[str, str]]],
    cache_path: Path | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Băm content hash D017 cho các ứng viên `closed_train` mới trong pool cục bộ."""
    fingerprint = [
        {
            "shard_index": idx,
            "shard": shard_details[idx]["shard"],
            "sha256": shard_details[idx]["sha256"],
            "candidate_count": len(candidate_rows_by_shard.get(idx, ())),
        }
        for idx in sorted(local_shards)
    ]

    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("cache_version") == LOCAL_CANDIDATE_CACHE_VERSION
            and cached.get("shard_fingerprint") == fingerprint
            and isinstance(cached.get("candidates"), list)
        ):
            return list(cached["candidates"]), True

    candidates: list[dict[str, Any]] = []
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '8GB'")
    try:
        for idx in sorted(candidate_rows_by_shard):
            expected_rows = candidate_rows_by_shard[idx]
            if not expected_rows:
                continue
            parquet_path = local_shards[idx]
            expected_by_file = {row["file"]: row for row in expected_rows}
            file_list = sorted(expected_by_file)
            seen: set[str] = set()
            cursor = connection.execute(
                """
                SELECT file, label, utt_type, audio.sampling_rate, audio.array
                FROM read_parquet(?)
                WHERE list_contains(?, file)
                ORDER BY file
                """,
                [parquet_path.resolve().as_posix(), file_list],
            )
            while True:
                batch = cursor.fetchmany(32)
                if not batch:
                    break
                for file_name, speaker_id, utt_type, sample_rate, waveform in batch:
                    logical_file = str(file_name)
                    expected = expected_by_file.get(logical_file)
                    require(expected is not None, f"Đọc ngoài danh sách ứng viên: {logical_file}")
                    require(logical_file not in seen, f"Shard lặp file {logical_file}")
                    seen.add(logical_file)
                    require(
                        str(speaker_id) == expected["speaker_id"]
                        and str(utt_type) == expected["utt_type"],
                        f"Metadata Parquet không khớp split/metadata cho {logical_file}",
                    )
                    rate = int(sample_rate or 0)
                    require(rate > 0 and waveform is not None, f"Audio rỗng/lỗi: {logical_file}")
                    digest = waveform_sha256(waveform, rate)
                    candidates.append(
                        {
                            "file": logical_file,
                            "speaker_id": str(speaker_id),
                            "utt_type": str(utt_type),
                            "binary_label": 0 if str(utt_type) == "bonafide" else 1,
                            "shard": parquet_path.name,
                            "shard_index": idx,
                            "native_sample_rate": rate,
                            "content_sha256": digest,
                        }
                    )
            missing = sorted(set(expected_by_file) - seen)
            require(not missing, f"Thiếu {len(missing)} ứng viên trong {parquet_path.name}: {missing[:3]}")
    finally:
        connection.close()

    candidates.sort(key=lambda item: item["file"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "cache_version": LOCAL_CANDIDATE_CACHE_VERSION,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "shard_fingerprint": fingerprint,
                    "candidates": candidates,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return candidates, False


def deduplicate_and_select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    train_hashes: Mapping[str, int],
    dev_hashes: set[str],
    test_hashes: set[str],
    additional_needed: Mapping[str, int],
    seed: int = SEED,
) -> dict[str, Any]:
    """Khử trùng ứng viên theo D036 mục 2, đếm theo utt_type/nguồn trùng và tính thiếu hụt."""
    labels_by_hash: dict[str, set[int]] = defaultdict(set)
    for digest, label in train_hashes.items():
        labels_by_hash[digest].add(int(label))
    for cand in candidates:
        labels_by_hash[str(cand["content_sha256"])].add(int(cand["binary_label"]))
    mixed = sorted(digest for digest, labels in labels_by_hash.items() if len(labels) > 1)
    if mixed:
        raise ValueError(f"Phát hiện {len(mixed)} content hash trộn nhãn nhị phân: {mixed[0]}")

    raw_counts = Counter({utt: 0 for utt in UTT_TYPES})
    drop_counts_by_reason: dict[str, Counter[str]] = defaultdict(lambda: Counter({utt: 0 for utt in UTT_TYPES}))
    overlap_presence_counts: dict[str, Counter[str]] = defaultdict(lambda: Counter({utt: 0 for utt in UTT_TYPES}))

    enriched: dict[str, dict[str, Any]] = {}
    survivors_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cand in candidates:
        file_name = str(cand["file"])
        utt_type = str(cand["utt_type"])
        digest = str(cand["content_sha256"])
        raw_counts[utt_type] += 1

        in_train = digest in train_hashes
        in_dev = digest in dev_hashes
        in_test = digest in test_hashes
        if in_train:
            overlap_presence_counts["in_train_current"][utt_type] += 1
        if in_dev:
            overlap_presence_counts["in_dev_all"][utt_type] += 1
        if in_test:
            overlap_presence_counts["in_test_all"][utt_type] += 1

        row = {
            "file": file_name,
            "speaker_id": str(cand["speaker_id"]),
            "utt_type": utt_type,
            "binary_label": int(cand["binary_label"]),
            "shard": str(cand["shard"]),
            "native_sample_rate": int(cand["native_sample_rate"]),
            "content_sha256": digest,
            "order_sha256": order_digest(file_name, seed),
            "in_train_current": in_train,
            "in_dev_all": in_dev,
            "in_test_all": in_test,
            "selection_status": "",
            "drop_reason": "",
            "kept_representative_file": "",
        }
        enriched[file_name] = row

        if in_train:
            row["selection_status"] = "dropped"
            row["drop_reason"] = "train_overlap"
            drop_counts_by_reason["train_overlap"][utt_type] += 1
        elif in_dev:
            row["selection_status"] = "dropped"
            row["drop_reason"] = "dev_overlap"
            drop_counts_by_reason["dev_overlap"][utt_type] += 1
        elif in_test:
            row["selection_status"] = "dropped"
            row["drop_reason"] = "test_overlap"
            drop_counts_by_reason["test_overlap"][utt_type] += 1
        else:
            survivors_by_hash[digest].append(row)

    clean_by_utt: dict[str, list[dict[str, Any]]] = {utt: [] for utt in UTT_TYPES}
    internal_duplicate_groups: list[dict[str, Any]] = []
    for digest, members in sorted(survivors_by_hash.items()):
        ordered = sorted(members, key=lambda r: stable_key(r["file"], seed))
        rep = ordered[0]
        rep["kept_representative_file"] = rep["file"]
        clean_by_utt[rep["utt_type"]].append(rep)
        if len(ordered) > 1:
            for extra in ordered[1:]:
                extra["selection_status"] = "dropped"
                extra["drop_reason"] = "internal_duplicate"
                extra["kept_representative_file"] = rep["file"]
                drop_counts_by_reason["internal_duplicate"][extra["utt_type"]] += 1
            internal_duplicate_groups.append(
                {
                    "content_sha256": digest,
                    "representative_file": rep["file"],
                    "representative_order_sha256": rep["order_sha256"],
                    "removed_files": [m["file"] for m in ordered[1:]],
                }
            )

    clean_counts: dict[str, int] = {}
    selected_local: dict[str, list[str]] = {}
    shortfalls: dict[str, int] = {}
    surplus_local: dict[str, int] = {}
    dropped_totals_by_utt: dict[str, int] = {}
    observed_drop_rate_by_utt: dict[str, float] = {}

    for utt in UTT_TYPES:
        ordered_clean = sorted(clean_by_utt[utt], key=lambda r: stable_key(r["file"], seed))
        clean_counts[utt] = len(ordered_clean)
        need = int(additional_needed[utt])
        take = min(need, len(ordered_clean))
        for item in ordered_clean[:take]:
            item["selection_status"] = "selected_local"
        for item in ordered_clean[take:]:
            item["selection_status"] = "clean_surplus_local"
        selected_local[utt] = [item["file"] for item in ordered_clean[:take]]
        shortfalls[utt] = max(0, need - len(ordered_clean))
        surplus_local[utt] = max(0, len(ordered_clean) - need)
        dropped_u = raw_counts[utt] - clean_counts[utt]
        dropped_totals_by_utt[utt] = dropped_u
        observed_drop_rate_by_utt[utt] = (dropped_u / raw_counts[utt]) if raw_counts[utt] > 0 else 0.0

    total_raw = sum(raw_counts.values())
    total_clean = sum(clean_counts.values())
    total_dropped = total_raw - total_clean
    observed_drop_rate_overall = (total_dropped / total_raw) if total_raw > 0 else 0.0

    ordered_rows = [enriched[k] for k in sorted(enriched)]
    return {
        "raw_counts_by_utt_type": {utt: raw_counts[utt] for utt in UTT_TYPES},
        "clean_counts_by_utt_type": clean_counts,
        "dropped_totals_by_utt_type": dropped_totals_by_utt,
        "dropped_by_primary_reason": {
            reason: {utt: drop_counts_by_reason[reason][utt] for utt in UTT_TYPES}
            for reason in ("train_overlap", "dev_overlap", "test_overlap", "internal_duplicate")
        },
        "overlap_presence_by_source": {
            source: {utt: overlap_presence_counts[source][utt] for utt in UTT_TYPES}
            for source in ("in_train_current", "in_dev_all", "in_test_all")
        },
        "observed_drop_rate_by_utt_type": observed_drop_rate_by_utt,
        "observed_drop_rate_overall": observed_drop_rate_overall,
        "internal_duplicate_group_count": len(internal_duplicate_groups),
        "internal_duplicate_groups_sample": internal_duplicate_groups[:20],
        "selected_local_counts_by_utt_type": {utt: len(selected_local[utt]) for utt in UTT_TYPES},
        "selected_local_files_by_utt_type": selected_local,
        "shortfalls_by_utt_type": shortfalls,
        "surplus_local_by_utt_type": surplus_local,
        "total_raw": total_raw,
        "total_clean": total_clean,
        "total_dropped": total_dropped,
        "total_shortfall": sum(shortfalls.values()),
        "candidate_rows": ordered_rows,
    }


def _accumulate_until_targets(
    ordered_remote: Sequence[Mapping[str, Any]],
    targets: Mapping[str, int],
    *,
    max_shards: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Cộng dồn shard theo thứ tự đã xếp cho tới khi đủ mọi `targets` theo `utt_type`."""
    cumulative = {utt: 0 for utt in UTT_TYPES}
    total_bytes = 0
    selected_shards: list[dict[str, Any]] = []
    satisfied = all(cumulative[utt] >= int(targets[utt]) for utt in UTT_TYPES)

    if not satisfied:
        for order_pos, shard in enumerate(ordered_remote, start=1):
            counts = shard["closed_train_unselected_by_utt_type"]
            for utt in UTT_TYPES:
                cumulative[utt] += int(counts[utt])
            total_bytes += int(shard["size_bytes"])
            selected_shards.append(
                {
                    "order_position": order_pos,
                    "shard": shard["shard"],
                    "shard_index": shard["shard_index"],
                    "order_sha256": shard["order_sha256"],
                    "size_bytes": shard["size_bytes"],
                    "sha256": shard["sha256"],
                    "closed_train_unselected_by_utt_type": dict(counts),
                    "cumulative_by_utt_type": dict(cumulative),
                    "cumulative_bytes": total_bytes,
                }
            )
            if all(cumulative[utt] >= int(targets[utt]) for utt in UTT_TYPES):
                satisfied = True
                break

    shard_count = len(selected_shards)
    exceeds_shard_ceiling = shard_count > max_shards
    exceeds_bytes_ceiling = total_bytes > max_bytes
    return {
        "targets_by_utt_type": {utt: int(targets[utt]) for utt in UTT_TYPES},
        "satisfied": satisfied,
        "shard_count": shard_count,
        "total_bytes": total_bytes,
        "total_gb": _format_bytes_gb(total_bytes),
        "total_gib": _format_bytes_gib(total_bytes),
        "cumulative_by_utt_type": cumulative,
        "exceeds_shard_ceiling": exceeds_shard_ceiling,
        "exceeds_bytes_ceiling_gb": exceeds_bytes_ceiling,
        "exceeds_bytes_ceiling_gib": total_bytes > MAX_BYTES_CEILING_GIB,
        "exceeds_ceiling": (not satisfied) or exceeds_shard_ceiling or exceeds_bytes_ceiling,
        "stopping_shard": selected_shards[-1]["shard"] if selected_shards else None,
        "shards": selected_shards,
    }


def plan_remote_shards(
    *,
    geometry: ShardGeometry,
    tree: Mapping[int, TreeEntry],
    local_shard_indices: set[int],
    closed_train_unselected_by_shard: Mapping[int, Mapping[str, int]],
    shortfalls: Mapping[str, int],
    observed_drop_rate_by_utt: Mapping[str, float],
    observed_drop_rate_overall: float,
    seed: int = SEED,
    max_shards: int = MAX_SHARDS_CEILING,
    max_bytes: int = MAX_BYTES_CEILING_GB,
) -> dict[str, Any]:
    """Lập thứ tự tải shard từ xa theo SHA256('2026|shard') và kiểm trần 120 shard / 10 GB."""
    missing_indices = [idx for idx in range(geometry.total_shards) if idx not in local_shard_indices]
    remote_entries: list[dict[str, Any]] = []
    for idx in missing_indices:
        entry = tree.get(idx)
        require(entry is not None, f"Thiếu thông tin tree cho shard từ xa {idx}.")
        name = shard_filename(geometry, idx)
        counts = closed_train_unselected_by_shard.get(idx, {})
        remote_entries.append(
            {
                "shard_index": idx,
                "shard": name,
                "order_sha256": order_digest(name, seed),
                "size_bytes": entry.size,
                "sha256": entry.sha256,
                "closed_train_unselected_by_utt_type": {
                    utt: int(counts.get(utt, 0)) for utt in UTT_TYPES
                },
            }
        )

    ordered_remote = sorted(remote_entries, key=lambda r: (r["order_sha256"], r["shard"]))

    base_plan = _accumulate_until_targets(
        ordered_remote,
        shortfalls,
        max_shards=max_shards,
        max_bytes=max_bytes,
    )

    # Mức dự phòng 1 (ước tính theo tỷ lệ trùng riêng từng utt_type quan sát ở bước 4)
    reserve_targets_by_utt: dict[str, int] = {}
    for utt in UTT_TYPES:
        need = int(shortfalls[utt])
        rate = float(observed_drop_rate_by_utt.get(utt, 0.0))
        require(rate < 1.0, f"Tỷ lệ loại trùng cho {utt} bằng 100%, không thể ước tính dự phòng.")
        reserve_targets_by_utt[utt] = math.ceil(need / (1.0 - rate)) if need > 0 else 0
    reserve_plan_by_utt_rate = _accumulate_until_targets(
        ordered_remote,
        reserve_targets_by_utt,
        max_shards=max_shards,
        max_bytes=max_bytes,
    )

    # Mức dự phòng 2 (ước tính áp dụng tỷ lệ trùng chung toàn pool cục bộ cho mọi utt_type thiếu)
    require(
        observed_drop_rate_overall < 1.0,
        "Tỷ lệ loại trùng chung bằng 100%, không thể ước tính dự phòng.",
    )
    reserve_targets_overall = {
        utt: (math.ceil(int(shortfalls[utt]) / (1.0 - observed_drop_rate_overall)) if int(shortfalls[utt]) > 0 else 0)
        for utt in UTT_TYPES
    }
    reserve_plan_overall_rate = _accumulate_until_targets(
        ordered_remote,
        reserve_targets_overall,
        max_shards=max_shards,
        max_bytes=max_bytes,
    )

    any_exceeds = (
        base_plan["exceeds_ceiling"]
        or reserve_plan_by_utt_rate["exceeds_ceiling"]
        or reserve_plan_overall_rate["exceeds_ceiling"]
    )
    return {
        "selection_basis_note": (
            "Thứ tự shard chỉ phụ thuộc SHA256('2026|<tên_shard>') và số dòng closed_train trong "
            "data/splits/closed_train.csv + data/metadata/vsasv_metadata.csv; tuyệt đối không đọc hay "
            "dùng closed_test để chọn shard."
        ),
        "ceiling": {
            "max_shards": max_shards,
            "max_bytes_gb": max_bytes,
            "max_bytes_gib": MAX_BYTES_CEILING_GIB,
        },
        "total_remote_shards_available": len(ordered_remote),
        "base_plan_pre_dedup": base_plan,
        "reserve_estimate_by_utt_type_rate": {
            "is_estimate": True,
            "note": (
                "Ước tính chia thiếu hụt mỗi utt_type cho (1 - tỷ_lệ_loại_trùng_utt_type_ở_bước_4); "
                "chỉ mang tính lập kế hoạch trước khử trùng shard mới."
            ),
            "rates_used": {utt: float(observed_drop_rate_by_utt.get(utt, 0.0)) for utt in UTT_TYPES},
            "plan": reserve_plan_by_utt_rate,
        },
        "reserve_estimate_overall_rate": {
            "is_estimate": True,
            "note": (
                "Ước tính chia thiếu hụt mọi utt_type cho (1 - tỷ_lệ_loại_trùng_chung_toàn_pool_cục_bộ_bước_4); "
                "đề phòng trường hợp shard mới có tỷ lệ trùng VC/AP khác 0% dù cục bộ VC/AP ít trùng."
            ),
            "rate_used": float(observed_drop_rate_overall),
            "plan": reserve_plan_overall_rate,
        },
        "any_exceeds_ceiling": any_exceeds,
    }


def write_candidate_cache(
    cache_dir: Path,
    dedup_result: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Lưu danh sách ứng viên cục bộ kèm hash và quyết định vào `.cache/scale2_01/`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "local_candidates_scale2.csv"
    json_path = cache_dir / "local_candidates_scale2.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in dedup_result["candidate_rows"]:
            writer.writerow({col: row[col] for col in CANDIDATE_CSV_COLUMNS})

    json_payload = {
        key: value
        for key, value in dedup_result.items()
        if key != "selected_local_files_by_utt_type"
    }
    json_payload["selected_local_counts_by_utt_type"] = dedup_result[
        "selected_local_counts_by_utt_type"
    ]
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Kết xuất báo cáo Markdown đọc nhanh cho `SCALE2-01`."""
    quota = report["quota"]
    local = report["local_pool"]
    dedup = report["local_dedup"]
    remote = report["remote_shard_plan"]
    base = remote["base_plan_pre_dedup"]
    res_utt = remote["reserve_estimate_by_utt_type_rate"]
    res_all = remote["reserve_estimate_overall_rate"]
    res_utt_plan = res_utt["plan"]
    res_all_plan = res_all["plan"]

    lines = [
        "# SCALE2-01 — Kế hoạch shard và khử trùng cho train ×2 (`development-scale2-v1`, D036)",
        "",
        f"- Thời điểm tạo (UTC): `{report['generated_at_utc']}`",
        f"- Nguồn snapshot: `{report['repository']}` revision `{report['revision']}` (seed `{report['seed']}`).",
        f"- Chính sách chống rò rỉ test: **{report['no_test_selection_guarantee']}**",
        f"- Kiểm 68 shard cục bộ vs `vsasv_shard_tree.json` & `vsasv_metadata.csv`: **{local['verified_shards']}/{local['expected_shards']} shard khớp 100%** ({local['total_bytes']:,} B).",
        "",
        "## 1. Quota D036, ứng viên cục bộ sạch và thiếu hụt còn lại",
        "",
        "| `utt_type` | Quota tổng (×2) | Đã có (×1) | Phần thêm cần | Ứng viên cục bộ thô | Trùng `train` | Trùng `dev` | Trùng `test` | Trùng nội bộ | Sạch cục bộ | Còn thiếu sau cục bộ | Tỷ lệ loại cục bộ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    drops = dedup["dropped_by_primary_reason"]
    for utt in UTT_TYPES:
        lines.append(
            f"| `{utt}` | {quota['scale2_total'][utt]:,} | {quota['current_train'][utt]:,} | "
            f"{quota['additional_needed'][utt]:,} | {dedup['raw_counts_by_utt_type'][utt]:,} | "
            f"{drops['train_overlap'][utt]:,} | {drops['dev_overlap'][utt]:,} | "
            f"{drops['test_overlap'][utt]:,} | {drops['internal_duplicate'][utt]:,} | "
            f"{dedup['clean_counts_by_utt_type'][utt]:,} | {dedup['shortfalls_by_utt_type'][utt]:,} | "
            f"{100.0 * dedup['observed_drop_rate_by_utt_type'][utt]:.2f}% |"
        )
    lines.append(
        f"| **Tổng** | **{sum(quota['scale2_total'].values()):,}** | **{sum(quota['current_train'].values()):,}** | "
        f"**{sum(quota['additional_needed'].values()):,}** | **{dedup['total_raw']:,}** | "
        f"**{sum(drops['train_overlap'].values()):,}** | **{sum(drops['dev_overlap'].values()):,}** | "
        f"**{sum(drops['test_overlap'].values()):,}** | **{sum(drops['internal_duplicate'].values()):,}** | "
        f"**{dedup['total_clean']:,}** | **{dedup['total_shortfall']:,}** | "
        f"**{100.0 * dedup['observed_drop_rate_overall']:.2f}%** |"
    )

    lines.extend(
        [
            "",
            "- Ghi chú `replay`: pool cục bộ có "
            f"`{dedup['clean_counts_by_utt_type']['replay']}` mẫu sạch, dư `{dedup['surplus_local_by_utt_type']['replay']}` mẫu so với phần thêm cần `{quota['additional_needed']['replay']}` (đã đủ, thiếu hụt = `0`).",
            f"- Lưu ý tập hash loại trùng: {report['hash_sources']['ledger_summary']['dedup_policy_note']}",
            f"- Số nhóm content hash trùng hoàn toàn trong `closed_train` cục bộ (nội bộ ứng viên): `{dedup['internal_duplicate_group_count']}` nhóm.",
            "",
            "## 2. Tóm tắt kế hoạch tải shard từ xa vs trần D036 (120 shard / 10 GB)",
            "",
            "| Mức kế hoạch | Mục tiêu `bonafide` / `VC` / `AP` / `replay` | Số shard cần tải | Tổng dung lượng (byte) | GB (10⁹ B) | GiB (2³⁰ B) | Vượt trần 120 shard / 10 GB? |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )

    def _plan_row(label: str, plan: Mapping[str, Any]) -> str:
        t = plan["targets_by_utt_type"]
        flag = "VƯỢT TRẦN" if plan["exceeds_ceiling"] else "Đạt (không vượt)"
        return (
            f"| {label} | {t['bonafide']:,} / {t['voice_conversion']:,} / {t['adversarial_attack']:,} / {t['replay']:,} | "
            f"{plan['shard_count']} | {plan['total_bytes']:,} | {plan['total_gb']:.3f} GB | {plan['total_gib']:.3f} GiB | **{flag}** |"
        )

    lines.append(_plan_row("Cơ sở (chưa khử trùng shard mới)", base))
    lines.append(
        _plan_row(
            f"Dự phòng 1 (*ước tính theo tỷ lệ trùng từng `utt_type` bước 4*)",
            res_utt_plan,
        )
    )
    lines.append(
        _plan_row(
            f"Dự phòng 2 (*ước tính áp dụng tỷ lệ trùng chung {100.0 * res_all['rate_used']:.2f}% bước 4*)",
            res_all_plan,
        )
    )

    lines.extend(
        [
            "",
            "## 3. Danh sách shard cần tải theo thứ tự `SHA256(\"2026|shard\")` tăng dần (kế hoạch cơ sở)",
            "",
            "| # | Tên shard | Kích thước (byte) | SHA-256 (`vsasv_shard_tree.json`) | Dòng `closed_train` mới (`bonafide` / `VC` / `AP` / `replay`) | Cộng dồn (`bonafide` / `VC` / `AP` / `replay`) |",
            "|---:|---|---:|---|---|---|",
        ]
    )
    for item in base["shards"]:
        cnt = item["closed_train_unselected_by_utt_type"]
        cum = item["cumulative_by_utt_type"]
        lines.append(
            f"| {item['order_position']} | `{item['shard']}` | {item['size_bytes']:,} | `{item['sha256']}` | "
            f"{cnt['bonafide']} / {cnt['voice_conversion']} / {cnt['adversarial_attack']} / {cnt['replay']} | "
            f"{cum['bonafide']:,} / {cum['voice_conversion']:,} / {cum['adversarial_attack']:,} / {cum['replay']:,} |"
        )

    extra_utt = [
        item for item in res_utt_plan["shards"] if item["order_position"] > base["shard_count"]
    ]
    extra_all = [
        item for item in res_all_plan["shards"] if item["order_position"] > res_utt_plan["shard_count"]
    ]
    if extra_utt or extra_all:
        lines.extend(
            [
                "",
                "### Shard bổ sung nếu dùng mức dự phòng (*ước tính*)",
                "",
            ]
        )
        all_extra = [
            item for item in res_all_plan["shards"] if item["order_position"] > base["shard_count"]
        ]
        for item in all_extra:
            cnt = item["closed_train_unselected_by_utt_type"]
            cum = item["cumulative_by_utt_type"]
            lines.append(
                f"- Mức #{item['order_position']}: `{item['shard']}` ({item['size_bytes']:,} B, `{item['sha256']}`) — "
                f"+({cnt['bonafide']}/{cnt['voice_conversion']}/{cnt['adversarial_attack']}/{cnt['replay']}) → "
                f"cộng dồn ({cum['bonafide']:,}/{cum['voice_conversion']:,}/{cum['adversarial_attack']:,}/{cum['replay']:,}), "
                f"{_format_bytes_gb(item['cumulative_bytes']):.3f} GB."
            )
    else:
        lines.extend(
            [
                "",
                "- Với cả hai mức dự phòng (*ước tính theo tỷ lệ trùng ở bước 4*), danh sách shard dừng tại đúng cùng shard thứ "
                f"`{base['shard_count']}` (`{base['stopping_shard']}`) mà không cần thêm shard nào ngoài bảng trên.",
            ]
        )

    lines.extend(
        [
            "",
            "## 4. Nguồn hash và phạm vi chỉ đọc",
            "",
            f"- Train hiện tại: `{report['hash_sources']['audit_summary']['path']}` (`{report['hash_sources']['audit_summary']['audited_current_train_rows']:,}` dòng `closed_train`, SHA-256 file `{report['hash_sources']['audit_summary']['sha256']}`).",
            f"- Dev-full & Test-full: `{report['hash_sources']['ledger_summary']['path']}` (`{report['hash_sources']['ledger_summary']['rows']:,}` dòng, SHA-256 `{report['hash_sources']['ledger_summary']['sha256']}`, khớp `ledger_sha256` trong `reports/closed_eval_full_v1.json`).",
            "- Không gọi mạng, không tải shard, không đọc audio `closed_dev` hay `closed_test`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_scale2_plan(
    *,
    parquet_dir: Path = DEFAULT_PARQUET_DIR,
    metadata_path: Path = DEFAULT_METADATA,
    closed_train_split_path: Path = DEFAULT_CLOSED_TRAIN_SPLIT,
    current_manifest_path: Path = DEFAULT_CURRENT_MANIFEST,
    audit_json_path: Path = DEFAULT_AUDIT_JSON,
    eval_report_json_path: Path = DEFAULT_EVAL_REPORT_JSON,
    ledger_csv_path: Path = DEFAULT_LEDGER_CSV,
    tree_json_path: Path = DEFAULT_TREE_JSON,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    json_output: Path = DEFAULT_JSON_REPORT,
    md_output: Path = DEFAULT_MD_REPORT,
    geometry: ShardGeometry = OFFICIAL_GEOMETRY,
    expected_local_shards: int | None = EXPECTED_LOCAL_SHARDS,
    verify_ledger_sha: bool = True,
    seed: int = SEED,
) -> dict[str, Any]:
    """Thực thi toàn bộ bước 2–9 của SCALE2-01 (chỉ đọc, chưa tải shard)."""
    # Bước 2: kiểm nguồn content hash của train hiện tại, dev-full, test-full
    hash_sources = load_and_verify_hash_sources(
        audit_json_path=audit_json_path,
        eval_report_json_path=eval_report_json_path,
        ledger_csv_path=ledger_csv_path,
        current_manifest_csv_path=current_manifest_path,
        verify_ledger_sha=verify_ledger_sha,
    )
    current_train_files: set[str] = hash_sources["current_train"]["files"]
    current_train_counts: dict[str, int] = hash_sources["current_train"]["counts_by_utt_type"]

    # Bước 3: đọc metadata, tree và xác minh toàn bộ 68 shard cục bộ
    metadata_index = read_metadata_index(metadata_path)
    require(
        len(metadata_index) == geometry.total_rows,
        f"Metadata có {len(metadata_index)} dòng, kỳ vọng {geometry.total_rows}.",
    )
    tree = load_tree(tree_json_path, geometry)
    require(len(tree) == geometry.total_shards, f"Tree thiếu shard: có {len(tree)}/{geometry.total_shards}.")

    local_verification = verify_local_shards(
        parquet_dir=parquet_dir,
        tree=tree,
        metadata_index=metadata_index,
        geometry=geometry,
        expected_local_shards=expected_local_shards,
    )
    local_shards: dict[int, Path] = local_verification["local_shards"]

    # Đọc ONLY `data/splits/closed_train.csv`, đối chiếu metadata và tính quota D036
    split_rows = read_source_csv(closed_train_split_path)
    closed_train_counts = Counter({utt: 0 for utt in UTT_TYPES})
    local_candidate_rows_by_shard: dict[int, list[dict[str, str]]] = defaultdict(list)
    closed_train_unselected_by_shard: dict[int, Counter[str]] = defaultdict(Counter)

    seen_split_files: set[str] = set()
    for row in split_rows:
        file_name = row["file"]
        require(file_name not in seen_split_files, f"Split closed_train lặp file: {file_name}")
        seen_split_files.add(file_name)
        meta_entry = metadata_index.get(file_name)
        require(meta_entry is not None, f"File trong closed_train không có ở metadata: {file_name}")
        row_index, speaker_id, utt_type = meta_entry
        require(
            (speaker_id, utt_type) == (row["label"], row["utt_type"]),
            f"Metadata không khớp closed_train cho {file_name}",
        )
        closed_train_counts[utt_type] += 1
        if file_name in current_train_files:
            continue
        shard_idx = shard_index_for_row(geometry, row_index)
        closed_train_unselected_by_shard[shard_idx][utt_type] += 1
        if shard_idx in local_shards:
            local_candidate_rows_by_shard[shard_idx].append(
                {
                    "file": file_name,
                    "speaker_id": speaker_id,
                    "utt_type": utt_type,
                }
            )

    missing_current = current_train_files - seen_split_files
    require(not missing_current, f"Train hiện tại có {len(missing_current)} file không thuộc closed_train.")

    scale2_quota = allocate_largest_remainder(SCALE2_TRAIN_TOTAL, dict(closed_train_counts), UTT_TYPES)
    require(
        scale2_quota == EXPECTED_D036_QUOTA_TOTAL,
        f"Quota phân bổ ({scale2_quota}) khác D036 ({EXPECTED_D036_QUOTA_TOTAL}).",
    )
    additional_needed = {
        utt: scale2_quota[utt] - current_train_counts[utt] for utt in UTT_TYPES
    }
    require(all(val >= 0 for val in additional_needed.values()), "Phần thêm cần bị âm.")

    # Bước 4: băm content hash D017 các ứng viên closed_train cục bộ và khử trùng
    hashed_candidates, cache_hit = hash_local_train_candidates(
        local_shards=local_shards,
        shard_details=local_verification["details_by_index"],
        candidate_rows_by_shard=local_candidate_rows_by_shard,
        cache_path=cache_dir / "local_candidates_hashed.json",
    )
    dedup_result = deduplicate_and_select_candidates(
        hashed_candidates,
        train_hashes=hash_sources["train_hashes"],
        dev_hashes=hash_sources["dev_hashes"],
        test_hashes=hash_sources["test_hashes"],
        additional_needed=additional_needed,
        seed=seed,
    )
    cand_csv_path, cand_json_path = write_candidate_cache(cache_dir, dedup_result)

    # Bước 5–7: thiếu hụt và thứ tự shard tải
    remote_plan = plan_remote_shards(
        geometry=geometry,
        tree=tree,
        local_shard_indices=set(local_shards),
        closed_train_unselected_by_shard=closed_train_unselected_by_shard,
        shortfalls=dedup_result["shortfalls_by_utt_type"],
        observed_drop_rate_by_utt=dedup_result["observed_drop_rate_by_utt_type"],
        observed_drop_rate_overall=dedup_result["observed_drop_rate_overall"],
        seed=seed,
        max_shards=MAX_SHARDS_CEILING,
        max_bytes=MAX_BYTES_CEILING_GB,
    )

    report: dict[str, Any] = {
        "task": "SCALE2-01",
        "decision": "D036",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "revision": REVISION,
        "seed": seed,
        "no_test_selection_guarantee": (
            "Không đọc audio dev/test; không dùng bảng split của closed_test hay utt_type/speaker_id "
            "của closed_test; thứ tự chọn shard hoàn toàn chỉ phụ thuộc SHA256('2026|shard')."
        ),
        "hash_sources": {
            "audit_summary": hash_sources["audit_summary"],
            "ledger_summary": hash_sources["ledger_summary"],
        },
        "local_pool": {
            "parquet_dir": _display_path(parquet_dir),
            "expected_shards": expected_local_shards,
            "verified_shards": local_verification["count"],
            "total_bytes": local_verification["total_bytes"],
            "all_size_and_sha_matched": local_verification["all_size_and_sha_matched"],
            "all_geometry_files_matched": local_verification["all_geometry_files_matched"],
            "candidate_hash_cache_hit": cache_hit,
            "candidate_cache_csv": _display_path(cand_csv_path),
            "candidate_cache_json": _display_path(cand_json_path),
        },
        "quota": {
            "closed_train_full_counts": {utt: closed_train_counts[utt] for utt in UTT_TYPES},
            "current_train": current_train_counts,
            "scale2_total": scale2_quota,
            "additional_needed": additional_needed,
        },
        "local_dedup": {
            key: value
            for key, value in dedup_result.items()
            if key not in {"candidate_rows", "selected_local_files_by_utt_type"}
        },
        "remote_shard_plan": remote_plan,
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(render_markdown_report(report), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--closed-train-split", type=Path, default=DEFAULT_CLOSED_TRAIN_SPLIT)
    parser.add_argument("--current-manifest", type=Path, default=DEFAULT_CURRENT_MANIFEST)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--eval-report-json", type=Path, default=DEFAULT_EVAL_REPORT_JSON)
    parser.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    parser.add_argument("--tree-json", type=Path, default=DEFAULT_TREE_JSON)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        report = run_scale2_plan(
            parquet_dir=args.parquet_dir,
            metadata_path=args.metadata,
            closed_train_split_path=args.closed_train_split,
            current_manifest_path=args.current_manifest,
            audit_json_path=args.audit_json,
            eval_report_json_path=args.eval_report_json,
            ledger_csv_path=args.ledger_csv,
            tree_json_path=args.tree_json,
            cache_dir=args.cache_dir,
            json_output=args.json_output,
            md_output=args.md_output,
        )
        remote = report["remote_shard_plan"]
        base = remote["base_plan_pre_dedup"]
        summary = {
            "status": "exceeds_ceiling" if remote["any_exceeds_ceiling"] else "ok",
            "verified_local_shards": report["local_pool"]["verified_shards"],
            "clean_local_by_utt_type": report["local_dedup"]["clean_counts_by_utt_type"],
            "shortfalls_by_utt_type": report["local_dedup"]["shortfalls_by_utt_type"],
            "base_shards": base["shard_count"],
            "base_bytes": base["total_bytes"],
            "reserve_utt_shards": remote["reserve_estimate_by_utt_type_rate"]["plan"]["shard_count"],
            "reserve_utt_bytes": remote["reserve_estimate_by_utt_type_rate"]["plan"]["total_bytes"],
            "reserve_overall_shards": remote["reserve_estimate_overall_rate"]["plan"]["shard_count"],
            "reserve_overall_bytes": remote["reserve_estimate_overall_rate"]["plan"]["total_bytes"],
            "any_exceeds_ceiling": remote["any_exceeds_ceiling"],
            "json_output": str(args.json_output),
            "md_output": str(args.md_output),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if remote["any_exceeds_ceiling"] else 0
    except Exception as exc:
        print(f"LỖI lập kế hoạch shard scale2: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
