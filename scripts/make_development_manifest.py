#!/usr/bin/env python3
"""Tạo manifest development xác định từ closed train/dev và audio cục bộ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.content_hash import waveform_sha256


DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "development_20k_v2.csv"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "development_manifest_20k_v2.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "development_manifest_20k_v2.md"
DEFAULT_TARGET_TOTAL = 20_000
DEFAULT_SEED = 2026
DEFAULT_MANIFEST_VERSION = "development-20k-v2"
SOURCE_SNAPSHOT = "VSASV-HF-public-snapshot-v1"
SPLIT_NAMES = ("closed_train", "closed_dev")
UTT_TYPES = ("bonafide", "voice_conversion", "adversarial_attack", "replay")
SOURCE_COLUMNS = ("file", "label", "utt_type")
MANIFEST_COLUMNS = (
    "manifest_version",
    "shard",
    "file",
    "speaker_id",
    "split",
    "binary_label",
    "utt_type",
    "native_sample_rate",
    "source_snapshot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--manifest-version", type=str, default=DEFAULT_MANIFEST_VERSION
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(file_name: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}|{file_name}".encode("utf-8")).hexdigest()
    return digest, file_name


def choose_content_representatives(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Keep one deterministic file for every waveform content hash."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        content_hash = str(row.get("content_sha256", ""))
        if not content_hash:
            raise ValueError(f"Thiếu content hash cho file: {row.get('file', '')}")
        groups[content_hash].append(row)

    eligible_files: set[str] = set()
    duplicate_groups: list[dict[str, Any]] = []
    for content_hash, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda row: stable_key(str(row["file"]), seed))
        representative = ordered[0]
        eligible_files.add(str(representative["file"]))
        if len(ordered) <= 1:
            continue
        duplicate_groups.append(
            {
                "waveform_sha256": content_hash,
                "representative_file": str(representative["file"]),
                "representative_split": str(representative["split"]),
                "representative_speaker_id": str(representative["speaker_id"]),
                "files": [str(row["file"]) for row in ordered],
                "splits": sorted({str(row["split"]) for row in ordered}),
                "speaker_ids": sorted(
                    {str(row["speaker_id"]) for row in ordered}
                ),
                "removed_files": [str(row["file"]) for row in ordered[1:]],
            }
        )
    return eligible_files, duplicate_groups


def allocate_largest_remainder(
    total: int,
    weights: dict[str, int],
    order: Iterable[str],
) -> dict[str, int]:
    """Phân bổ số nguyên theo tỷ lệ, phá hòa bằng thứ tự đã khai báo."""
    ordered = tuple(order)
    if total < 0:
        raise ValueError("Tổng quota không được âm.")
    if set(weights) != set(ordered):
        raise ValueError("Trọng số và thứ tự phân bổ không cùng tập khóa.")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Tổng trọng số phải lớn hơn 0.")
    if any(value < 0 for value in weights.values()):
        raise ValueError("Trọng số không được âm.")

    allocation = {
        name: (total * weights[name]) // weight_sum for name in ordered
    }
    remaining = total - sum(allocation.values())
    ranked = sorted(
        ordered,
        key=lambda name: (
            -((total * weights[name]) % weight_sum),
            ordered.index(name),
        ),
    )
    for name in ranked[:remaining]:
        allocation[name] += 1
    return allocation


def read_source_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError(f"Schema CSV không hợp lệ: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV không có dữ liệu: {path}")
    return rows


def read_metadata(path: Path) -> dict[str, tuple[str, str]]:
    rows = read_source_csv(path)
    result: dict[str, tuple[str, str]] = {}
    for row in rows:
        file_name = row["file"]
        if file_name in result:
            raise ValueError(f"Metadata có file lặp: {file_name}")
        if row["utt_type"] not in UTT_TYPES:
            raise ValueError(f"utt_type không hợp lệ trong metadata: {file_name}")
        result[file_name] = (row["label"], row["utt_type"])
    return result


def build_local_index(parquet_dir: Path) -> dict[str, dict[str, Any]]:
    """Index and hash every local waveform one shard at a time."""

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy shard Parquet trong {parquet_dir}")
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '8GB'")
    local_index: dict[str, dict[str, Any]] = {}
    try:
        for parquet_file in parquet_files:
            cursor = connection.execute(
                """
                SELECT file, label, utt_type, audio.sampling_rate, audio.array
                FROM read_parquet(?)
                ORDER BY file
                """,
                [parquet_file.resolve().as_posix()],
            )
            while True:
                rows = cursor.fetchmany(32)
                if not rows:
                    break
                for file_name, speaker_id, utt_type, sample_rate, waveform in rows:
                    logical_file = str(file_name)
                    if logical_file in local_index:
                        raise ValueError(
                            f"Audio cục bộ có file logic bị lặp: {logical_file}"
                        )
                    rate = int(sample_rate or 0)
                    if rate <= 0:
                        raise ValueError(
                            f"Audio có sample rate không hợp lệ: {logical_file}"
                        )
                    local_index[logical_file] = {
                        "shard": parquet_file.name,
                        "speaker_id": str(speaker_id),
                        "utt_type": str(utt_type),
                        "native_sample_rate": rate,
                        "content_sha256": waveform_sha256(waveform or [], rate),
                    }
    finally:
        connection.close()
    return local_index


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _count_nested(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        counts = Counter(str(row[field]) for row in rows if row["split"] == split_name)
        result[split_name] = dict(sorted(counts.items()))
    return result


def generate_development_manifest(
    parquet_dir: Path,
    metadata_path: Path,
    split_dir: Path,
    output_path: Path,
    *,
    target_total: int = DEFAULT_TARGET_TOTAL,
    seed: int = DEFAULT_SEED,
    manifest_version: str = DEFAULT_MANIFEST_VERSION,
) -> dict[str, Any]:
    if target_total < 1:
        raise ValueError("target-total phải lớn hơn hoặc bằng 1.")
    if not manifest_version.strip():
        raise ValueError("manifest-version không được để trống.")

    metadata = read_metadata(metadata_path)
    local_index = build_local_index(parquet_dir)
    for file_name, local in local_index.items():
        expected = metadata.get(file_name)
        if expected is None:
            raise ValueError(f"Audio không có trong metadata: {file_name}")
        if expected != (local["speaker_id"], local["utt_type"]):
            raise ValueError(f"Audio không khớp metadata: {file_name}")

    split_rows: dict[str, list[dict[str, str]]] = {}
    split_counts: dict[str, int] = {}
    type_counts: dict[str, dict[str, int]] = {}
    source_reports: dict[str, dict[str, Any]] = {}
    seen_files: set[str] = set()
    speakers_by_split: dict[str, set[str]] = {}
    for split_name in SPLIT_NAMES:
        path = split_dir / f"{split_name}.csv"
        rows = read_source_csv(path)
        speakers: set[str] = set()
        counts = Counter({name: 0 for name in UTT_TYPES})
        for row in rows:
            file_name = row["file"]
            if file_name in seen_files:
                raise ValueError(f"File xuất hiện ở nhiều split: {file_name}")
            seen_files.add(file_name)
            expected = metadata.get(file_name)
            if expected != (row["label"], row["utt_type"]):
                raise ValueError(f"Split không khớp metadata: {file_name}")
            if row["utt_type"] not in UTT_TYPES:
                raise ValueError(f"utt_type không hợp lệ trong split: {file_name}")
            speakers.add(row["label"])
            counts[row["utt_type"]] += 1
        split_rows[split_name] = rows
        split_counts[split_name] = len(rows)
        type_counts[split_name] = {name: counts[name] for name in UTT_TYPES}
        speakers_by_split[split_name] = speakers
        source_reports[split_name] = {
            "path": display_path(path),
            "sha256": sha256_file(path),
            "samples": len(rows),
            "speakers": len(speakers),
            "utt_type_counts": type_counts[split_name],
        }

    overlap = speakers_by_split["closed_train"] & speakers_by_split["closed_dev"]
    if overlap:
        raise ValueError(
            f"Phát hiện {len(overlap)} speaker trùng giữa closed_train và closed_dev."
        )
    if target_total > sum(split_counts.values()):
        raise ValueError("target-total lớn hơn tổng số mẫu closed train/dev.")

    split_quotas = allocate_largest_remainder(
        target_total, split_counts, SPLIT_NAMES
    )
    quotas: dict[str, dict[str, int]] = {
        split_name: allocate_largest_remainder(
            split_quotas[split_name], type_counts[split_name], UTT_TYPES
        )
        for split_name in SPLIT_NAMES
    }

    raw_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        split_name: {utt_type: [] for utt_type in UTT_TYPES}
        for split_name in SPLIT_NAMES
    }
    for split_name in SPLIT_NAMES:
        for row in split_rows[split_name]:
            if row["file"] in local_index:
                local = local_index[row["file"]]
                raw_candidates[split_name][row["utt_type"]].append(
                    {
                        **row,
                        "speaker_id": row["label"],
                        "split": split_name,
                        "content_sha256": local["content_sha256"],
                    }
                )

    raw_candidate_counts = {
        split_name: {
            utt_type: len(raw_candidates[split_name][utt_type])
            for utt_type in UTT_TYPES
        }
        for split_name in SPLIT_NAMES
    }
    raw_shortfalls: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        for utt_type in UTT_TYPES:
            available = raw_candidate_counts[split_name][utt_type]
            required = quotas[split_name][utt_type]
            if available < required:
                raw_shortfalls.append(
                    {
                        "split": split_name,
                        "utt_type": utt_type,
                        "required": required,
                        "available": available,
                        "missing": required - available,
                    }
                )
    all_candidate_rows = [
        row
        for split_name in SPLIT_NAMES
        for utt_type in UTT_TYPES
        for row in raw_candidates[split_name][utt_type]
    ]
    eligible_files, duplicate_groups = choose_content_representatives(
        all_candidate_rows,
        seed=seed,
    )
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        split_name: {
            utt_type: [
                row
                for row in raw_candidates[split_name][utt_type]
                if row["file"] in eligible_files
            ]
            for utt_type in UTT_TYPES
        }
        for split_name in SPLIT_NAMES
    }
    candidate_counts = {
        split_name: {
            utt_type: len(candidates[split_name][utt_type])
            for utt_type in UTT_TYPES
        }
        for split_name in SPLIT_NAMES
    }
    row_by_file = {str(row["file"]): row for row in all_candidate_rows}
    removed_by_split = Counter()
    removed_by_split_and_utt_type: dict[str, Counter[str]] = {
        split_name: Counter() for split_name in SPLIT_NAMES
    }
    for group in duplicate_groups:
        for file_name in group["removed_files"]:
            removed = row_by_file[file_name]
            split_name = str(removed["split"])
            removed_by_split[split_name] += 1
            removed_by_split_and_utt_type[split_name][str(removed["utt_type"])] += 1

    shortfalls: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        for utt_type in UTT_TYPES:
            available = candidate_counts[split_name][utt_type]
            required = quotas[split_name][utt_type]
            if available < required:
                shortfalls.append(
                    {
                        "split": split_name,
                        "utt_type": utt_type,
                        "required": required,
                        "available": available,
                        "missing": required - available,
                    }
                )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "KHÔNG ĐỦ AUDIO"
            if raw_shortfalls
            else "KHÔNG ĐỦ AUDIO SAU KHỬ TRÙNG"
            if shortfalls
            else "ĐẠT"
        ),
        "technical_passed": not shortfalls,
        "manifest_version": manifest_version,
        "source_snapshot": SOURCE_SNAPSHOT,
        "seed": seed,
        "target_total": target_total,
        "selection": {
            "partitions": list(SPLIT_NAMES),
            "stable_order": "sha256(f'{seed}|{file}')",
            "quota_basis": "full_closed_train_dev_distribution",
            "allocation": "largest_remainder",
            "content_uniqueness": "global_across_closed_train_and_closed_dev",
            "split_quotas": split_quotas,
            "utt_type_quotas": quotas,
        },
        "source": {
            "metadata": {
                "path": display_path(metadata_path),
                "sha256": sha256_file(metadata_path),
                "samples": len(metadata),
            },
            "closed_splits": source_reports,
            "parquet_directory": display_path(parquet_dir),
            "local_shards": len({item["shard"] for item in local_index.values()}),
            "local_samples": len(local_index),
            "local_candidate_counts": raw_candidate_counts,
            "deduplicated_candidate_counts": candidate_counts,
        },
        "deduplication": {
            "method": (
                "sha256(sample_rate_signed_int64_le || waveform_float64_le); "
                "representative=min(sha256(f'{seed}|{file}'), file)"
            ),
            "indexed_local_files": len(local_index),
            "candidate_files_before": len(all_candidate_rows),
            "unique_candidate_content_hashes": len(
                {str(row["content_sha256"]) for row in all_candidate_rows}
            ),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_files": sum(len(group["files"]) for group in duplicate_groups),
            "removed_files": sum(
                len(group["removed_files"]) for group in duplicate_groups
            ),
            "removed_by_split": {
                split_name: removed_by_split[split_name]
                for split_name in SPLIT_NAMES
            },
            "removed_by_split_and_utt_type": {
                split_name: {
                    utt_type: removed_by_split_and_utt_type[split_name][utt_type]
                    for utt_type in UTT_TYPES
                }
                for split_name in SPLIT_NAMES
            },
            "groups": duplicate_groups,
            "baseline_files_removed_by_split": None,
            "replacement_files_by_split": None,
            "selected_content_hashes": None,
        },
        "shortfalls": shortfalls,
        "raw_shortfalls": raw_shortfalls,
        "manifest": None,
        "checks": {
            "unique_files": None,
            "duplicate_content_hashes": None,
            "speaker_overlap_train_dev": len(overlap),
            "metadata_exact": True,
            "binary_labels_valid": None,
            "shards_exist": None,
        },
        "warnings": [
            "Manifest chỉ dùng closed_train và closed_dev; closed_test không tham gia chọn quy mô hoặc phân bố.",
            "Khử trùng nội dung không chứng minh hoặc sửa quan hệ giữa các speaker ID có cùng waveform.",
            "Duplicate đi qua closed_test phải được quyết định riêng trước đánh giá cuối.",
            "Quy mô và quota phản ánh bản phát hành công khai hiện có, không được diễn giải là tái lập dữ liệu trong bài báo gốc.",
        ],
    }
    if shortfalls:
        return report

    baseline_files_by_split: dict[str, set[str]] = {
        split_name: set() for split_name in SPLIT_NAMES
    }
    for split_name in SPLIT_NAMES:
        for utt_type in UTT_TYPES:
            ordered = sorted(
                raw_candidates[split_name][utt_type],
                key=lambda row: stable_key(str(row["file"]), seed),
            )
            baseline_files_by_split[split_name].update(
                str(row["file"])
                for row in ordered[: quotas[split_name][utt_type]]
            )

    selected: list[dict[str, Any]] = []
    selected_content_hashes: set[str] = set()
    for split_name in SPLIT_NAMES:
        for utt_type in UTT_TYPES:
            ordered = sorted(
                candidates[split_name][utt_type],
                key=lambda row: stable_key(str(row["file"]), seed),
            )
            selected_in_stratum = 0
            for row in ordered:
                content_hash = str(row["content_sha256"])
                if content_hash in selected_content_hashes:
                    continue
                local = local_index[row["file"]]
                selected.append(
                    {
                        "manifest_version": manifest_version,
                        "shard": local["shard"],
                        "file": row["file"],
                        "speaker_id": row["label"],
                        "split": split_name,
                        "binary_label": 0 if utt_type == "bonafide" else 1,
                        "utt_type": utt_type,
                        "native_sample_rate": local["native_sample_rate"],
                        "source_snapshot": SOURCE_SNAPSHOT,
                    }
                )
                selected_content_hashes.add(content_hash)
                selected_in_stratum += 1
                if selected_in_stratum == quotas[split_name][utt_type]:
                    break
            if selected_in_stratum != quotas[split_name][utt_type]:
                raise RuntimeError(
                    "Không thể bù đủ quota sau khi loại content hash đã chọn: "
                    f"{split_name}/{utt_type}."
                )

    if len(selected) != target_total:
        raise RuntimeError("Số hàng đã chọn không khớp target-total.")
    if len({row["file"] for row in selected}) != len(selected):
        raise RuntimeError("Manifest có file bị lặp.")
    if len(selected_content_hashes) != len(selected):
        raise RuntimeError("Manifest có content hash bị lặp.")
    if any(row["binary_label"] not in (0, 1) for row in selected):
        raise RuntimeError("Manifest có binary_label không hợp lệ.")
    parquet_names = {path.name for path in parquet_dir.glob("*.parquet")}
    if any(row["shard"] not in parquet_names for row in selected):
        raise RuntimeError("Manifest trỏ tới shard không tồn tại.")

    write_manifest(output_path, selected)
    per_split = Counter(str(row["split"]) for row in selected)
    per_binary_label = _count_nested(selected, "binary_label")
    per_utt_type = _count_nested(selected, "utt_type")
    per_shard = Counter(str(row["shard"]) for row in selected)
    per_split_speakers = {
        split_name: len(
            {row["speaker_id"] for row in selected if row["split"] == split_name}
        )
        for split_name in SPLIT_NAMES
    }
    selected_speakers = defaultdict(set)
    for row in selected:
        selected_speakers[str(row["split"])].add(str(row["speaker_id"]))
    selected_overlap = selected_speakers["closed_train"] & selected_speakers["closed_dev"]
    if selected_overlap:
        raise RuntimeError("Manifest có speaker trùng giữa train và dev.")

    selected_files_by_split = {
        split_name: {
            str(row["file"])
            for row in selected
            if row["split"] == split_name
        }
        for split_name in SPLIT_NAMES
    }
    report["deduplication"]["baseline_files_removed_by_split"] = {
        split_name: len(
            baseline_files_by_split[split_name] - selected_files_by_split[split_name]
        )
        for split_name in SPLIT_NAMES
    }
    report["deduplication"]["replacement_files_by_split"] = {
        split_name: len(
            selected_files_by_split[split_name] - baseline_files_by_split[split_name]
        )
        for split_name in SPLIT_NAMES
    }
    report["deduplication"]["selected_content_hashes"] = len(
        selected_content_hashes
    )

    report["manifest"] = {
        "path": display_path(output_path),
        "sha256": sha256_file(output_path),
        "samples": len(selected),
        "speakers": len({row["speaker_id"] for row in selected}),
        "shards": len(per_shard),
        "counts_by_split": dict(sorted(per_split.items())),
        "counts_by_binary_label": per_binary_label,
        "counts_by_utt_type": per_utt_type,
        "speakers_by_split": per_split_speakers,
        "counts_by_shard": dict(sorted(per_shard.items())),
        "columns": list(MANIFEST_COLUMNS),
    }
    report["checks"] = {
        "unique_files": True,
        "duplicate_content_hashes": 0,
        "speaker_overlap_train_dev": len(selected_overlap),
        "metadata_exact": True,
        "binary_labels_valid": True,
        "shards_exist": True,
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Báo cáo development manifest 20k",
        "",
        f"- **Trạng thái:** {report['status']}",
        f"- **Phiên bản:** `{report['manifest_version']}`",
        f"- **Seed:** `{report['seed']}`",
        f"- **Mục tiêu:** {report['target_total']:,} mẫu",
        f"- **Shard cục bộ:** {report['source']['local_shards']:,}",
        f"- **Audio cục bộ:** {report['source']['local_samples']:,} mẫu",
        "",
        "## Quota đã khóa",
        "",
        "| Split | Tổng | Bonafide | Voice conversion | Adversarial | Replay |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split_name in SPLIT_NAMES:
        quotas = report["selection"]["utt_type_quotas"][split_name]
        lines.append(
            f"| `{split_name}` | {report['selection']['split_quotas'][split_name]:,} | "
            f"{quotas['bonafide']:,} | {quotas['voice_conversion']:,} | "
            f"{quotas['adversarial_attack']:,} | {quotas['replay']:,} |"
        )

    lines.extend(
        [
            "",
            "## Ứng viên audio cục bộ",
            "",
            "| Split | Bonafide | Voice conversion | Adversarial | Replay |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split_name in SPLIT_NAMES:
        counts = report["source"]["local_candidate_counts"][split_name]
        lines.append(
            f"| `{split_name}` | {counts['bonafide']:,} | "
            f"{counts['voice_conversion']:,} | {counts['adversarial_attack']:,} | "
            f"{counts['replay']:,} |"
        )

    deduplication = report["deduplication"]
    lines.extend(
        [
            "",
            "## Khử trùng nội dung trước chọn quota",
            "",
            f"- File cục bộ đã băm: {deduplication['indexed_local_files']:,}",
            f"- Nhóm content hash trùng: {deduplication['duplicate_groups']:,}",
            f"- File thuộc nhóm trùng: {deduplication['duplicate_files']:,}",
            f"- File bị loại khỏi pool ứng viên: {deduplication['removed_files']:,}",
            "- Quy tắc đại diện: file có `SHA256(2026|file)` nhỏ nhất trong mỗi nhóm.",
            "",
            "| Split | Loại khỏi pool | Mất so với lựa chọn v1 | File bù |",
            "|---|---:|---:|---:|",
        ]
    )
    for split_name in SPLIT_NAMES:
        baseline_removed = deduplication["baseline_files_removed_by_split"]
        replacements = deduplication["replacement_files_by_split"]
        lines.append(
            f"| `{split_name}` | {deduplication['removed_by_split'][split_name]:,} | "
            f"{0 if baseline_removed is None else baseline_removed[split_name]:,} | "
            f"{0 if replacements is None else replacements[split_name]:,} |"
        )

    if report["manifest"] is not None:
        manifest = report["manifest"]
        lines.extend(
            [
                "",
                "## Manifest đã tạo",
                "",
                f"- Đường dẫn: `{manifest['path']}`",
                f"- SHA-256: `{manifest['sha256']}`",
                f"- Số mẫu: {manifest['samples']:,}",
                f"- Số speaker: {manifest['speakers']:,}",
                f"- Số shard được dùng: {manifest['shards']:,}",
                "- Speaker overlap train/dev: 0",
                "- File lặp: 0",
                "- Content hash lặp: 0",
                "- Nhãn nhị phân, metadata và đường dẫn shard: ĐẠT",
            ]
        )
    if report["shortfalls"]:
        lines.extend(["", "## Thiếu audio", ""])
        for item in report["shortfalls"]:
            lines.append(
                f"- `{item['split']}` / `{item['utt_type']}`: cần "
                f"{item['required']:,}, có {item['available']:,}, "
                f"thiếu {item['missing']:,}."
            )

    lines.extend(["", "## Giới hạn diễn giải", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    try:
        report = generate_development_manifest(
            args.parquet_dir.resolve(),
            args.metadata.resolve(),
            args.split_dir.resolve(),
            args.output.resolve(),
            target_total=args.target_total,
            seed=args.seed,
            manifest_version=args.manifest_version,
        )
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args.markdown_report.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, RuntimeError, ValueError, duckdb.Error) as error:
        print(f"Lỗi tạo development manifest: {error}", file=sys.stderr)
        return 2

    if not report["technical_passed"]:
        print(
            f"KHÔNG ĐỦ AUDIO: {len(report['shortfalls'])} stratum chưa đạt quota.",
            file=sys.stderr,
        )
        return 3
    manifest = report["manifest"]
    print(
        f"ĐẠT: {manifest['samples']:,} mẫu, {manifest['speakers']:,} speaker, "
        f"{manifest['shards']:,} shard, sha256={manifest['sha256']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
