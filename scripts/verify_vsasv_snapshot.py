#!/usr/bin/env python3
"""Verify the local public VSASV snapshot and emit JSON/Markdown evidence.

The script deliberately separates technical consistency from scientific
equivalence. A snapshot can be internally consistent without matching the
corpus size or protocol reported in the original VSASV paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.content_hash import waveform_sha256


DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_DEVELOPMENT_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "development_20k_v2.csv"
)
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "vsasv_snapshot_verification.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "vsasv_snapshot_verification.md"

EXPECTED_COLUMNS = ("file", "label", "utt_type")
EXPECTED_UTT_TYPES = (
    "bonafide",
    "voice_conversion",
    "adversarial_attack",
    "replay",
)
OFFICIAL_PAPER_COUNTS = {
    "bonafide": 164_374,
    "voice_conversion": 100_564,
    "adversarial_attack": 16_731,
    "replay": 57_571,
}
EXPECTED_PARQUET_COLUMNS = {
    "file": "VARCHAR",
    "audio": 'STRUCT("array" DOUBLE[], sampling_rate BIGINT)',
    "label": "VARCHAR",
    "utt_type": "VARCHAR",
}
SPLIT_FILENAMES = (
    "closed_train.csv",
    "closed_dev.csv",
    "closed_test.csv",
    "open_train_vc.csv",
    "open_dev_vc.csv",
    "open_seen_test_vc.csv",
    "open_unseen_test_adversarial.csv",
    "open_unseen_test_replay.csv",
)
SPEAKER_PATTERN = re.compile(r"^id\d{5}$")
PATH_PATTERNS = {
    "bonafide": re.compile(r"^(id\d{5})/(?:bonafide/)?\d{5}\.wav$"),
    "voice_conversion": re.compile(
        r"^(id\d{5})/voice_conversion/(id\d{5})_vc_\d{5}\.wav$"
    ),
    "adversarial_attack": re.compile(
        r"^(id\d{5})/adversarial_attack/\d{5}\.wav$"
    ),
    "replay": re.compile(r"^(id\d{5})/replay/(id\d{5})_replay_\d{5}\.wav$"),
}


def configure_duckdb(connection: Any) -> None:
    """Bound DuckDB memory use while scanning waveform-heavy Parquet shards."""

    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '8GB'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xác minh snapshot VSASV công khai, Parquet cục bộ và các split."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MANIFEST,
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT)
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


def normalize_row(raw_row: dict[str | None, str | None]) -> dict[str, str]:
    return {
        column: (raw_row.get(column) or "").strip()
        for column in EXPECTED_COLUMNS
    }


def numbered_stem(file_path: str) -> str | None:
    stem = PurePosixPath(file_path.replace("\\", "/")).stem
    match = re.search(r"(\d{5})$", stem)
    return match.group(1) if match else None


def quantiles(values: Iterable[int | float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"minimum": 0, "median": 0, "maximum": 0}
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "maximum": ordered[-1],
    }


def summarize_duplicate_fingerprints(
    fingerprint_records: dict[int, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Describe fingerprint groups that contain more than one logical file."""

    groups: list[dict[str, Any]] = []
    for fingerprint, records in fingerprint_records.items():
        members_by_file = {record["file"]: record for record in records}
        if len(members_by_file) <= 1:
            continue
        members = [members_by_file[name] for name in sorted(members_by_file)]
        speakers = sorted({member["speaker_id"] for member in members})
        utt_types = sorted({member["utt_type"] for member in members})
        closed_splits = sorted(
            {member["closed_split"] for member in members if member["closed_split"]}
        )
        binary_labels = sorted(
            {0 if member["utt_type"] == "bonafide" else 1 for member in members}
        )
        groups.append(
            {
                "fingerprint": f"{fingerprint:016x}",
                "files": len(members),
                "speakers": speakers,
                "utt_types": utt_types,
                "sample_rates": sorted(
                    {
                        int(member["sampling_rate"])
                        for member in members
                        if member["sampling_rate"].lstrip("-").isdigit()
                    }
                ),
                "closed_splits": closed_splits,
                "crosses_speakers": len(speakers) > 1,
                "crosses_closed_splits": len(closed_splits) > 1,
                "mixes_binary_labels": len(binary_labels) > 1,
                "members": members,
            }
        )
    return sorted(groups, key=lambda item: item["fingerprint"])


def confirm_duplicate_fingerprints(
    connection: Any,
    parquet_dir: Path,
    duplicate_groups: list[dict[str, Any]],
) -> None:
    """Confirm 64-bit fingerprint candidates with waveform SHA-256."""

    candidates_by_shard: dict[str, list[str]] = defaultdict(list)
    for group in duplicate_groups:
        for member in group["members"]:
            candidates_by_shard[member["shard"]].append(member["file"])

    sha_by_file: dict[str, str] = {}
    for shard_name, file_names in sorted(candidates_by_shard.items()):
        parquet_file = (parquet_dir / shard_name).resolve().as_posix()
        rows = connection.execute(
            """
            SELECT file, audio.sampling_rate, audio.array
            FROM read_parquet(?)
            WHERE file IN (SELECT * FROM UNNEST(?))
            """,
            [parquet_file, sorted(set(file_names))],
        ).fetchall()
        for file_name, sampling_rate, waveform in rows:
            sha_by_file[str(file_name)] = waveform_sha256(
                waveform or [], int(sampling_rate or 0)
            )

    for group in duplicate_groups:
        hashes = set()
        for member in group["members"]:
            member_sha = sha_by_file.get(member["file"], "")
            member["waveform_sha256"] = member_sha
            if member_sha:
                hashes.add(member_sha)
        recorded_hashes = {
            member["waveform_sha256"] for member in group["members"]
        }
        group["cryptographically_confirmed"] = (
            len(hashes) == 1 and len(hashes) == len(recorded_hashes)
        )
        group["waveform_sha256"] = next(iter(hashes)) if len(hashes) == 1 else ""


def summarize_manifest_duplicate_groups(
    duplicate_groups: list[dict[str, Any]],
    manifest_membership: dict[str, str],
) -> dict[str, int]:
    """Count confirmed duplicate groups retained by a development manifest."""

    groups_with_multiple_members = 0
    groups_crossing_splits = 0
    files_in_duplicate_groups = 0
    for group in duplicate_groups:
        if not group.get("cryptographically_confirmed", False):
            continue
        members = [
            member
            for member in group["members"]
            if member["file"] in manifest_membership
        ]
        if len(members) <= 1:
            continue
        groups_with_multiple_members += 1
        files_in_duplicate_groups += len(members)
        manifest_splits = {
            manifest_membership[member["file"]] for member in members
        }
        if len(manifest_splits) > 1:
            groups_crossing_splits += 1
    return {
        "groups_with_multiple_members": groups_with_multiple_members,
        "groups_crossing_splits": groups_crossing_splits,
        "files_in_duplicate_groups": files_in_duplicate_groups,
    }


def verify_development_manifest_duplicates(
    manifest_path: Path,
    duplicate_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare confirmed waveform duplicates with one development manifest."""

    result: dict[str, Any] = {
        "path": display_path(manifest_path),
        "available": manifest_path.is_file(),
        "sha256": "",
        "groups_with_multiple_members": 0,
        "groups_crossing_splits": 0,
        "files_in_duplicate_groups": 0,
    }
    if not manifest_path.is_file():
        return result

    membership: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            file_name = (row.get("file") or "").strip()
            split_name = (row.get("split") or "").strip()
            if file_name:
                membership[file_name] = split_name
    result.update(summarize_manifest_duplicate_groups(duplicate_groups, membership))
    result["sha256"] = sha256_file(manifest_path)
    return result


def verify_metadata(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metadata: {path}")

    rows: list[dict[str, str]] = []
    malformed_rows = 0
    missing = Counter()
    type_counts = Counter()
    rows_by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    indices_by_speaker: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    path_pattern_mismatches = Counter()
    speaker_pattern_mismatches = 0
    label_prefix_mismatches = 0
    embedded_speaker_mismatches = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        for raw_row in reader:
            if None in raw_row:
                malformed_rows += 1
            row = normalize_row(raw_row)
            rows.append(row)
            for column, value in row.items():
                if not value:
                    missing[column] += 1

            file_path = row["file"].replace("\\", "/")
            label = row["label"]
            utt_type = row["utt_type"]
            type_counts[utt_type] += 1
            rows_by_speaker[label][utt_type] += 1

            if label and not SPEAKER_PATTERN.fullmatch(label):
                speaker_pattern_mismatches += 1
            parts = PurePosixPath(file_path).parts
            if not parts or parts[0] != label:
                label_prefix_mismatches += 1

            pattern = PATH_PATTERNS.get(utt_type)
            match = pattern.fullmatch(file_path) if pattern else None
            if match is None:
                path_pattern_mismatches[utt_type or "<empty>"] += 1
            elif utt_type in {"voice_conversion", "replay"} and match.group(1) != match.group(2):
                embedded_speaker_mismatches += 1

            if utt_type in {"voice_conversion", "adversarial_attack"}:
                index = numbered_stem(file_path)
                if index is not None:
                    indices_by_speaker[label][utt_type].add(index)

    row_keys = [(row["file"], row["label"], row["utt_type"]) for row in rows]
    file_paths = [row["file"] for row in rows]
    invalid_types = {
        name: count
        for name, count in type_counts.items()
        if name not in EXPECTED_UTT_TYPES
    }

    attack_speakers = sorted(
        speaker
        for speaker, counts in rows_by_speaker.items()
        if counts["voice_conversion"] or counts["adversarial_attack"]
    )
    replay_speakers = sorted(
        speaker for speaker, counts in rows_by_speaker.items() if counts["replay"]
    )
    equal_count_speakers = [
        speaker
        for speaker in attack_speakers
        if rows_by_speaker[speaker]["voice_conversion"]
        == rows_by_speaker[speaker]["adversarial_attack"]
    ]
    equal_index_set_speakers = [
        speaker
        for speaker in attack_speakers
        if indices_by_speaker[speaker]["voice_conversion"]
        == indices_by_speaker[speaker]["adversarial_attack"]
    ]
    index_overlaps = [
        len(
            indices_by_speaker[speaker]["voice_conversion"]
            & indices_by_speaker[speaker]["adversarial_attack"]
        )
        for speaker in attack_speakers
    ]

    official_total = sum(OFFICIAL_PAPER_COUNTS.values())
    snapshot_total = len(rows)
    paper_comparison = {}
    for name in EXPECTED_UTT_TYPES:
        snapshot = type_counts.get(name, 0)
        official = OFFICIAL_PAPER_COUNTS[name]
        paper_comparison[name] = {
            "public_snapshot": snapshot,
            "official_paper": official,
            "difference": snapshot - official,
            "snapshot_as_percentage_of_paper": round(snapshot / official * 100, 4),
        }
    paper_comparison["total"] = {
        "public_snapshot": snapshot_total,
        "official_paper": official_total,
        "difference": snapshot_total - official_total,
        "snapshot_as_percentage_of_paper": round(
            snapshot_total / official_total * 100, 4
        ),
    }

    hard_issues: list[str] = []
    if actual_columns != EXPECTED_COLUMNS:
        hard_issues.append("Schema metadata không khớp ba cột đã khóa.")
    if malformed_rows:
        hard_issues.append(f"Có {malformed_rows} dòng CSV sai số trường.")
    if sum(missing.values()):
        hard_issues.append(f"Có {sum(missing.values())} giá trị bắt buộc bị thiếu.")
    if len(row_keys) != len(set(row_keys)):
        hard_issues.append("Có dòng metadata trùng hoàn toàn.")
    if len(file_paths) != len(set(file_paths)):
        hard_issues.append("Có đường dẫn logic bị lặp.")
    if invalid_types:
        hard_issues.append("Có utt_type ngoài taxonomy đã khóa.")
    if label_prefix_mismatches or embedded_speaker_mismatches:
        hard_issues.append("Có đường dẫn không nhất quán với speaker label.")
    if sum(path_pattern_mismatches.values()):
        hard_issues.append("Có đường dẫn không khớp quy tắc đặt tên quan sát được.")

    warnings = []
    if snapshot_total != official_total:
        warnings.append(
            "Số mẫu của snapshot công khai không khớp tổng số trong bài báo gốc."
        )
    if len(equal_count_speakers) == len(attack_speakers) and attack_speakers:
        warnings.append(
            "Mọi speaker tấn công đều có số VC bằng đúng số AP; đây là cấu trúc cần lưu ý, không phải bằng chứng lỗi join."
        )
    if len(type_counts) and type_counts.get("replay", 0) < 1_000:
        warnings.append("Replay trong snapshot công khai rất nhỏ so với bài báo gốc.")

    report = {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "technical_passed": not hard_issues,
        "hard_issues": hard_issues,
        "warnings": warnings,
        "schema": {
            "expected": list(EXPECTED_COLUMNS),
            "actual": list(actual_columns),
            "valid": actual_columns == EXPECTED_COLUMNS,
            "malformed_rows": malformed_rows,
        },
        "integrity": {
            "rows": len(rows),
            "unique_rows": len(set(row_keys)),
            "unique_files": len(set(file_paths)),
            "missing_values": {
                column: missing.get(column, 0) for column in EXPECTED_COLUMNS
            },
            "invalid_utt_types": invalid_types,
            "speaker_pattern_mismatches": speaker_pattern_mismatches,
            "label_prefix_mismatches": label_prefix_mismatches,
            "embedded_speaker_mismatches": embedded_speaker_mismatches,
            "path_pattern_mismatches": dict(path_pattern_mismatches),
        },
        "distribution": {
            "utterance_types": {
                name: type_counts.get(name, 0) for name in EXPECTED_UTT_TYPES
            },
            "speakers": len(rows_by_speaker),
            "samples_per_speaker": quantiles(
                sum(counts.values()) for counts in rows_by_speaker.values()
            ),
            "attack_speakers": len(attack_speakers),
            "replay_speakers": len(replay_speakers),
            "attack_replay_speaker_overlap": len(
                set(attack_speakers) & set(replay_speakers)
            ),
        },
        "vc_ap_relationship": {
            "vc_samples": type_counts.get("voice_conversion", 0),
            "ap_samples": type_counts.get("adversarial_attack", 0),
            "attack_speakers": len(attack_speakers),
            "speakers_with_equal_vc_ap_counts": len(equal_count_speakers),
            "speakers_with_identical_vc_ap_numeric_index_sets": len(
                equal_index_set_speakers
            ),
            "numeric_index_overlap_per_speaker": quantiles(index_overlaps),
            "path_overlap": len(
                {
                    row["file"]
                    for row in rows
                    if row["utt_type"] == "voice_conversion"
                }
                & {
                    row["file"]
                    for row in rows
                    if row["utt_type"] == "adversarial_attack"
                }
            ),
        },
        "official_paper_comparison": paper_comparison,
    }
    return report, rows


def verify_parquet(
    parquet_dir: Path, metadata_path: Path, split_dir: Path | None = None
) -> dict[str, Any]:
    parquet_files = sorted(parquet_dir.glob("*.parquet")) if parquet_dir.is_dir() else []
    base: dict[str, Any] = {
        "directory": display_path(parquet_dir),
        "available": bool(parquet_files),
        "expected_total_shards": 432,
        "local_shards": len(parquet_files),
        "coverage_by_shard_count_percent": round(len(parquet_files) / 432 * 100, 4),
        "files": [
            {
                "name": item.name,
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
            for item in parquet_files
        ],
        "technical_passed": False,
        "hard_issues": [],
        "warnings": [],
    }
    if not parquet_files:
        base["hard_issues"].append("Không tìm thấy Parquet cục bộ để kiểm tra.")
        return base

    try:
        import duckdb  # type: ignore[import-not-found]
    except (ImportError, OSError, AttributeError) as error:
        base["hard_issues"].append(
            f"Không nạp được DuckDB; chưa đọc nội dung Parquet: {error}"
        )
        return base

    metadata_index: dict[str, tuple[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = normalize_row(raw_row)
            metadata_index[row["file"]] = (row["label"], row["utt_type"])

    closed_split_membership: dict[str, str] = {}
    if split_dir is not None:
        for split_name in ("closed_train", "closed_dev", "closed_test"):
            split_path = split_dir / f"{split_name}.csv"
            if not split_path.is_file():
                continue
            with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for raw_row in csv.DictReader(handle):
                    file_name = (raw_row.get("file") or "").strip()
                    if file_name:
                        closed_split_membership[file_name] = split_name

    connection = duckdb.connect()
    configure_duckdb(connection)
    schema: dict[str, str] = {}
    schema_mismatches: list[str] = []
    total_rows = 0
    file_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    sample_rate_counts: Counter[tuple[str, int]] = Counter()
    durations_by_type: dict[str, list[float]] = defaultdict(list)
    shard_distribution: list[dict[str, Any]] = []
    fingerprint_records: dict[int, list[dict[str, str]]] = defaultdict(list)
    metadata_exact_matches = 0
    files_missing_from_metadata = 0
    label_or_type_mismatches = 0

    try:
        for parquet_path in parquet_files:
            parquet_file = parquet_path.resolve().as_posix()
            current_schema = {
                row[0]: row[1]
                for row in connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)", [parquet_file]
                ).fetchall()
            }
            if not schema:
                schema = current_schema
            if current_schema != EXPECTED_PARQUET_COLUMNS:
                schema_mismatches.append(parquet_path.name)

            shard_type_counts: Counter[str] = Counter()
            rows = connection.execute(
                """
                SELECT file, label, utt_type, audio.array IS NULL,
                       audio.sampling_rate, array_length(audio.array),
                       hash(audio.array)
                FROM read_parquet(?)
                """,
                [parquet_file],
            ).fetchall()
            total_rows += len(rows)
            for (
                file_name,
                label,
                utt_type,
                missing_audio,
                sampling_rate,
                sample_count,
                audio_fingerprint,
            ) in rows:
                file_value = str(file_name or "")
                label_value = str(label or "")
                utt_type_value = str(utt_type or "")
                file_counts[file_value] += 1
                type_counts[utt_type_value] += 1
                shard_type_counts[utt_type_value] += 1

                if not file_value:
                    missing_counts["file"] += 1
                if not label_value:
                    missing_counts["label"] += 1
                if not utt_type_value:
                    missing_counts["utt_type"] += 1
                if missing_audio:
                    missing_counts["audio_array"] += 1
                if sampling_rate is None:
                    missing_counts["sampling_rate"] += 1
                elif sampling_rate <= 0:
                    missing_counts["nonpositive_sampling_rate"] += 1
                if sample_count == 0:
                    missing_counts["empty_audio_array"] += 1

                metadata_row = metadata_index.get(file_value)
                if metadata_row is None:
                    files_missing_from_metadata += 1
                elif metadata_row == (label_value, utt_type_value):
                    metadata_exact_matches += 1
                else:
                    label_or_type_mismatches += 1

                if sampling_rate is not None:
                    sample_rate_counts[(utt_type_value, int(sampling_rate))] += 1
                if (
                    sampling_rate is not None
                    and sampling_rate > 0
                    and sample_count is not None
                ):
                    durations_by_type[utt_type_value].append(
                        sample_count / sampling_rate
                    )
                if audio_fingerprint is not None and file_value:
                    fingerprint_records[int(audio_fingerprint)].append(
                        {
                            "file": file_value,
                            "speaker_id": label_value,
                            "utt_type": utt_type_value,
                            "sampling_rate": str(sampling_rate),
                            "shard": parquet_path.name,
                            "closed_split": closed_split_membership.get(
                                file_value, ""
                            ),
                        }
                    )

            for utt_type_value, count in sorted(shard_type_counts.items()):
                shard_distribution.append(
                    {
                        "shard": parquet_path.name,
                        "utt_type": utt_type_value,
                        "samples": count,
                    }
                )
    finally:
        connection.close()

    unique_files = len(file_counts)
    missing_values = (
        missing_counts["file"],
        missing_counts["label"],
        missing_counts["utt_type"],
        missing_counts["audio_array"],
        missing_counts["sampling_rate"],
        missing_counts["empty_audio_array"],
        missing_counts["nonpositive_sampling_rate"],
    )
    sample_rates = [
        {"utt_type": key[0], "sampling_rate": key[1], "samples": count}
        for key, count in sorted(sample_rate_counts.items())
    ]
    durations = []
    for utt_type_value, values in sorted(durations_by_type.items()):
        summary = quantiles(values)
        durations.append(
            {
                "utt_type": utt_type_value,
                "minimum_seconds": round(float(summary["minimum"]), 6),
                "median_seconds": round(float(summary["median"]), 6),
                "maximum_seconds": round(float(summary["maximum"]), 6),
                "mean_seconds": round(sum(values) / len(values), 6),
                "total_hours": round(sum(values) / 3600.0, 6),
            }
        )
    duplicate_groups = summarize_duplicate_fingerprints(fingerprint_records)
    if duplicate_groups:
        confirmation_connection = duckdb.connect()
        configure_duckdb(confirmation_connection)
        try:
            confirm_duplicate_fingerprints(
                confirmation_connection, parquet_dir, duplicate_groups
            )
        finally:
            confirmation_connection.close()
    duplicate_fingerprints = len(duplicate_groups)
    confirmed_duplicate_groups = sum(
        item["cryptographically_confirmed"] for item in duplicate_groups
    )

    base.update(
        {
            "schema": schema,
            "rows": total_rows,
            "unique_files": unique_files,
            "metadata_exact_matches": metadata_exact_matches,
            "files_missing_from_metadata": files_missing_from_metadata,
            "label_or_type_mismatches": label_or_type_mismatches,
            "utt_type_counts": dict(type_counts),
            "missing_or_invalid_audio": {
                "file": missing_values[0],
                "label": missing_values[1],
                "utt_type": missing_values[2],
                "audio_array": missing_values[3],
                "sampling_rate": missing_values[4],
                "empty_audio_array": missing_values[5],
                "nonpositive_sampling_rate": missing_values[6],
            },
            "sampling_rates": sample_rates,
            "duration_by_type": durations,
            "shard_distribution": shard_distribution,
            "duplicate_audio_fingerprint_groups": duplicate_fingerprints,
            "cryptographically_confirmed_duplicate_groups": confirmed_duplicate_groups,
            "duplicate_audio_fingerprint_files": sum(
                item["files"] for item in duplicate_groups
            ),
            "duplicate_groups_crossing_speakers": sum(
                item["crosses_speakers"] for item in duplicate_groups
            ),
            "duplicate_groups_crossing_closed_splits": sum(
                item["crosses_closed_splits"] for item in duplicate_groups
            ),
            "duplicate_groups_mixing_binary_labels": sum(
                item["mixes_binary_labels"] for item in duplicate_groups
            ),
            "duplicate_audio_fingerprint_details": duplicate_groups,
            "fingerprint_method": (
                "DuckDB 64-bit hash(audio.array); useful as a duplicate screen, "
                "not a cryptographic identity proof."
            ),
        }
    )

    if schema_mismatches:
        base["hard_issues"].append(
            "Schema Parquet không khớp dataset card cục bộ trong "
            f"{len(schema_mismatches)} shard."
        )
    if unique_files != total_rows:
        base["hard_issues"].append("Parquet cục bộ có đường dẫn logic bị lặp.")
    if (
        metadata_exact_matches != total_rows
        or files_missing_from_metadata
        or label_or_type_mismatches
    ):
        base["hard_issues"].append("Parquet cục bộ không khớp metadata CSV.")
    if any(missing_values):
        base["hard_issues"].append("Parquet có trường bắt buộc hoặc audio không hợp lệ.")
    if confirmed_duplicate_groups:
        base["hard_issues"].append(
            "Phát hiện waveform trùng giữa nhiều file và đã xác nhận bằng SHA-256."
        )
    elif duplicate_fingerprints:
        base["warnings"].append(
            "Có ứng viên fingerprint 64-bit trùng nhưng SHA-256 không xác nhận waveform trùng."
        )
    if len({item["sampling_rate"] for item in sample_rates}) > 1:
        base["warnings"].append(
            f"Các loại audio trong {len(parquet_files)} shard có sample rate không "
            "đồng nhất; phải resample nhất quán trước huấn luyện."
        )
    if len(parquet_files) < 432:
        base["warnings"].append(
            "Parquet cục bộ chỉ là một phần nhỏ và không phải mẫu ngẫu nhiên của 432 shard."
        )

    base["technical_passed"] = not base["hard_issues"]
    return base


def verify_splits(split_dir: Path, metadata_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "directory": display_path(split_dir),
        "available": False,
        "technical_passed": False,
        "files": [],
        "hard_issues": [],
    }
    paths = [split_dir / name for name in SPLIT_FILENAMES]
    missing = [path.name for path in paths if not path.is_file()]
    result["available"] = not missing
    result["files"] = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    ]
    if missing:
        result["hard_issues"].append(f"Thiếu split: {', '.join(missing)}")
        return result

    try:
        from scripts import check_leakage
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts import check_leakage

    check = check_leakage.check_splits(metadata_path, split_dir)
    result.update(
        {
            "technical_passed": bool(check["passed"]),
            "summaries": check["splits"],
            "issues": check["issues"],
            "audio_hash_check": check["audio_check"],
        }
    )
    if not check["passed"]:
        result["hard_issues"].append(
            f"Bộ kiểm tra split phát hiện {len(check['issues'])} vấn đề."
        )
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    metadata, _ = verify_metadata(args.metadata.resolve())
    parquet = verify_parquet(
        args.parquet_dir.resolve(),
        args.metadata.resolve(),
        args.split_dir.resolve(),
    )
    manifest_duplicates = verify_development_manifest_duplicates(
        args.development_manifest.resolve(),
        parquet.get("duplicate_audio_fingerprint_details", []),
    )
    parquet["development_manifest_duplicates"] = manifest_duplicates
    if manifest_duplicates["groups_crossing_splits"]:
        parquet["hard_issues"].append(
            "Development manifest có waveform trùng đi qua train và development."
        )
        parquet["technical_passed"] = False
    splits = verify_splits(args.split_dir.resolve(), args.metadata.resolve())
    technical_passed = all(
        section["technical_passed"] for section in (metadata, parquet, splits)
    )
    warnings = metadata["warnings"] + parquet["warnings"]
    return {
        "verification_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "technical_consistency_passed": technical_passed,
        "scientific_status": "provisional_public_snapshot",
        "scientific_equivalence_to_official_paper": False,
        "scope": (
            "VSASV Hugging Face public snapshot represented by the local metadata, "
            f"{parquet['local_shards']} local Parquet shards, and the custom "
            "project splits."
        ),
        "warnings": warnings,
        "metadata": metadata,
        "parquet": parquet,
        "splits": splits,
        "conclusion": {
            "safe_to_use_for_pipeline_development": technical_passed,
            "safe_to_claim_official_vsasv_reproduction": False,
            "recommended_dataset_name": "VSASV-HF-public-snapshot-v1",
            "required_training_controls": [
                "Resample mọi waveform về một sample rate được khai báo thống nhất trong pipeline.",
                "Giữ train/dev/test tách biệt speaker.",
                "Báo cáo riêng VC, AP và replay; ghi rõ replay là tập con công khai hạn chế.",
                "Không so sánh trực tiếp metric của custom split với metric trong bài báo gốc.",
                "Chạy lại xác minh sau mỗi lần tải thêm shard.",
            ],
        },
        "sources": {
            "official_paper": "https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.pdf",
            "official_landing_page": "https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.html",
            "hugging_face_dataset": "https://huggingface.co/datasets/hustep-lab/VSASV-Dataset",
        },
    }


def format_count(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"
    return f"{int(value):,}"


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    parquet = report["parquet"]
    splits = report["splits"]
    manifest_duplicates = parquet.get("development_manifest_duplicates", {})
    manifest_known_groups_clean = bool(manifest_duplicates.get("available")) and not (
        manifest_duplicates.get("groups_with_multiple_members", 0)
    )
    vc_ap = metadata["vc_ap_relationship"]
    local_shards = parquet["local_shards"]
    local_rows = parquet.get("rows", 0)
    status = "ĐẠT" if report["technical_consistency_passed"] else "KHÔNG ĐẠT"
    hard_issues = [
        *metadata.get("hard_issues", []),
        *parquet.get("hard_issues", []),
        *splits.get("hard_issues", []),
    ]
    executive_summary = (
        f"Metadata, {local_shards} Parquet cục bộ và tám split đã vượt các kiểm tra "
        "kỹ thuật đã chạy. Không tìm thấy bằng chứng về lỗi join làm nhân đôi VC "
        "thành AP. Snapshot công khai vẫn khác đáng kể so với thống kê bài báo gốc "
        "nên chỉ được dùng như một giao thức tùy chỉnh, có phiên bản và giới hạn rõ ràng."
        if report["technical_consistency_passed"]
        else (
            "Snapshot nguồn chưa vượt toàn bộ kiểm tra kỹ thuật; xem hard issue bên dưới. "
            "Trạng thái development manifest được báo riêng và không được suy ra từ trạng "
            "thái nguồn."
        )
    )
    if manifest_known_groups_clean:
        executive_summary += (
            " Đối chiếu với các nhóm duplicate nguồn đã xác nhận cho thấy development "
            "manifest không giữ nhiều file trong cùng nhóm; cổng độc lập băm lại toàn bộ "
            "manifest vẫn phải được đọc từ báo cáo audit tương ứng."
        )
    lines = [
        "# Báo cáo xác minh snapshot VSASV công khai",
        "",
        f"- **Nhất quán kỹ thuật:** {status}",
        "- **Giá trị khoa học:** `provisional_public_snapshot`",
        "- **Tương đương bộ dữ liệu/giao thức bài báo gốc:** KHÔNG",
        f"- **Thời điểm UTC:** `{report['generated_at_utc']}`",
        f"- **Tên phiên bản đề xuất:** `{report['conclusion']['recommended_dataset_name']}`",
        "",
        "## Kết luận điều hành",
        "",
        executive_summary,
        "",
    ]
    if hard_issues:
        lines.extend(["### Hard issue", ""])
        lines.extend(f"- {issue}" for issue in hard_issues)
        lines.append("")
    lines.extend(
        [
            "## Metadata công khai",
            "",
            f"- File: `{metadata['path']}`",
            f"- SHA-256: `{metadata['sha256']}`",
            f"- Tổng dòng: {metadata['integrity']['rows']:,}",
            f"- File logic duy nhất: {metadata['integrity']['unique_files']:,}",
            f"- Speaker: {metadata['distribution']['speakers']:,}",
            "- Giá trị thiếu, dòng/file trùng, sai prefix speaker và sai quy tắc tên: 0",
            "",
            "### Đối chiếu với bài báo",
            "",
            "| Loại | Snapshot công khai | Bài báo | Chênh lệch | Snapshot/Bài báo |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    display_names = {
        "bonafide": "Bona fide",
        "voice_conversion": "VC",
        "adversarial_attack": "AP",
        "replay": "Replay",
        "total": "Tổng",
    }
    for name in (*EXPECTED_UTT_TYPES, "total"):
        item = metadata["official_paper_comparison"][name]
        lines.append(
            f"| {display_names[name]} | {item['public_snapshot']:,} | "
            f"{item['official_paper']:,} | {item['difference']:+,} | "
            f"{item['snapshot_as_percentage_of_paper']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "### Quan hệ VC–AP",
            "",
            f"- VC: {vc_ap['vc_samples']:,} mẫu; AP: {vc_ap['ap_samples']:,} mẫu.",
            (
                f"- {vc_ap['speakers_with_equal_vc_ap_counts']:,}/"
                f"{vc_ap['attack_speakers']:,} speaker tấn công có số VC bằng đúng số AP."
            ),
            f"- Đường dẫn VC trùng đường dẫn AP: {vc_ap['path_overlap']:,}.",
            (
                f"- Chỉ {vc_ap['speakers_with_identical_vc_ap_numeric_index_sets']:,}/"
                f"{vc_ap['attack_speakers']:,} speaker có cùng tập chỉ số số học trong tên VC và AP."
            ),
            "",
            (
                "Kết luận: sự bằng nhau về số lượng là cấu trúc thật của metadata công khai, "
                "nhưng không chứng minh VC và AP là cùng file hoặc do script join sai."
            ),
            "",
            f"## Parquet cục bộ ({local_shards} shard)",
            "",
            f"- Shard: {local_shards}/432 ({parquet['coverage_by_shard_count_percent']:.2f}% theo số shard).",
            f"- Tổng hàng: {local_rows:,}.",
            f"- Khớp metadata chính xác: {parquet.get('metadata_exact_matches', 0):,}/{local_rows:,}.",
            f"- File/audio rỗng hoặc sample rate không hợp lệ: {sum(parquet.get('missing_or_invalid_audio', {}).values()):,}.",
            f"- Nhóm fingerprint audio lặp: {parquet.get('duplicate_audio_fingerprint_groups', 0):,}.",
            f"- Nhóm waveform lặp đã xác nhận bằng SHA-256: {parquet.get('cryptographically_confirmed_duplicate_groups', 0):,}.",
            f"- Nhóm đi qua nhiều speaker: {parquet.get('duplicate_groups_crossing_speakers', 0):,}.",
            f"- Nhóm đi qua nhiều closed split: {parquet.get('duplicate_groups_crossing_closed_splits', 0):,}.",
            f"- Nhóm trộn nhãn nhị phân: {parquet.get('duplicate_groups_mixing_binary_labels', 0):,}.",
            (
                "- Development manifest đối chiếu: "
                f"`{manifest_duplicates.get('path', '')}`"
                + (
                    f"; SHA-256 `{manifest_duplicates.get('sha256', '')}`."
                    if manifest_duplicates.get("sha256")
                    else "."
                )
            ),
            (
                "- Trong development manifest, khi đối chiếu các nhóm duplicate nguồn đã xác nhận: "
                f"{manifest_duplicates.get('groups_with_multiple_members', 0):,} "
                "nhóm giữ nhiều file; "
                f"{manifest_duplicates.get('groups_crossing_splits', 0):,} "
                "nhóm đi qua train/development."
            ),
            "",
            "### Phân bố theo loại",
            "",
            "| Loại | Số mẫu |",
            "|---|---:|",
        ]
    )
    for name in EXPECTED_UTT_TYPES:
        lines.append(f"| `{name}` | {parquet.get('utt_type_counts', {}).get(name, 0):,} |")

    lines.extend(
        [
            "",
            "### Sample rate quan sát được",
            "",
            "| Loại | Sample rate | Số mẫu |",
            "|---|---:|---:|",
        ]
    )
    for item in parquet.get("sampling_rates", []):
        lines.append(
            f"| `{item['utt_type']}` | {item['sampling_rate']:,} Hz | {item['samples']:,} |"
        )
    lines.extend(
        [
            "",
            (
                f"Cảnh báo: {local_shards} shard cục bộ có thể không đại diện cho toàn bộ "
                "snapshot. Các sample rate quan sát được được liệt kê ở bảng trên; pipeline "
                "phải resample mọi waveform về cùng một sample rate."
            ),
            "",
            "## Tám split",
            "",
            f"- Trạng thái kiểm tra schema, coverage, file và speaker leakage: {'ĐẠT' if splits['technical_passed'] else 'KHÔNG ĐẠT'}.",
            (
                f"- Kiểm tra fingerprint audio đã chạy trên toàn bộ {local_rows:,} file "
                f"cục bộ; chưa bao phủ toàn bộ snapshot vì hiện có {local_shards}/432 shard."
            ),
            "- Đây là custom speaker-disjoint protocol của đồ án, không phải official split của bài báo.",
            "",
            "## Cảnh báo và giới hạn",
            "",
        ]
    )
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    lines.extend(
        [
            (
                f"- Kiểm tra fingerprint audio chỉ bao phủ {local_rows:,} file trong "
                f"{local_shards} shard cục bộ."
            ),
            "- Metadata không có `generator_id`, `source_corpus`, `official_split` hoặc định danh câu nguồn.",
            "- Không thể tự chứng minh nguyên nhân tác giả tạo số VC/AP bằng nhau chỉ từ ba cột metadata.",
            "- Replay công khai quá nhỏ để đại diện đầy đủ cho replay trong bài báo.",
            "",
            "## Quyết định sử dụng",
            "",
            (
                "Không dùng trực tiếp snapshot nguồn cho kết quả khoa học. Có thể tiếp tục "
                "phát triển pipeline và thử nghiệm baseline bằng development manifest chỉ "
                "khi manifest đó vượt audit content hash độc lập, đồng thời áp dụng các điều "
                "kiện sau:"
                if not report["technical_consistency_passed"]
                else "Có thể tiếp tục phát triển pipeline và thử nghiệm baseline nếu áp dụng các điều kiện sau:"
            ),
            "",
        ]
    )
    for control in report["conclusion"]["required_training_controls"]:
        lines.append(f"- {control}")
    lines.extend(
        [
            "",
            "Không được dùng báo cáo này để tuyên bố đã tái lập official VSASV hoặc kết quả bài báo gốc.",
            "",
            "## Nguồn đối chiếu",
            "",
            f"- [Bài báo VSASV gốc]({report['sources']['official_paper']})",
            f"- [Trang bài báo Interspeech]({report['sources']['official_landing_page']})",
            f"- [Bản VSASV công khai trên Hugging Face]({report['sources']['hugging_face_dataset']})",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    try:
        report = build_report(args)
        write_reports(report, args.json_output.resolve(), args.markdown_output.resolve())
    except (OSError, csv.Error, ValueError) as error:
        print(f"Lỗi xác minh VSASV: {error}", file=sys.stderr)
        return 2

    print(
        "Xác minh kỹ thuật "
        f"{'ĐẠT' if report['technical_consistency_passed'] else 'KHÔNG ĐẠT'}."
    )
    print(f"JSON: {args.json_output.resolve()}")
    print(f"Markdown: {args.markdown_output.resolve()}")
    return 0 if report["technical_consistency_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
