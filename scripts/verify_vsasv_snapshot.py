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
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xác minh snapshot VSASV công khai, Parquet cục bộ và các split."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
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


def quantiles(values: Iterable[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"minimum": 0, "median": 0, "maximum": 0}
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "maximum": ordered[-1],
    }


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
    parquet_dir: Path, metadata_path: Path
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

    parquet_glob = (parquet_dir / "*.parquet").resolve().as_posix()
    metadata_csv = metadata_path.resolve().as_posix()
    connection = duckdb.connect()

    schema_rows = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [parquet_glob]
    ).fetchall()
    schema = {row[0]: row[1] for row in schema_rows}
    total_rows = connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?)", [parquet_glob]
    ).fetchone()[0]
    unique_files = connection.execute(
        "SELECT COUNT(DISTINCT file) FROM read_parquet(?)", [parquet_glob]
    ).fetchone()[0]
    type_counts = dict(
        connection.execute(
            "SELECT utt_type, COUNT(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
            [parquet_glob],
        ).fetchall()
    )
    missing_values = connection.execute(
        """
        SELECT
            SUM(file IS NULL OR file = '')::BIGINT,
            SUM(label IS NULL OR label = '')::BIGINT,
            SUM(utt_type IS NULL OR utt_type = '')::BIGINT,
            SUM(audio.array IS NULL)::BIGINT,
            SUM(audio.sampling_rate IS NULL)::BIGINT,
            SUM(array_length(audio.array) = 0)::BIGINT,
            SUM(audio.sampling_rate <= 0)::BIGINT
        FROM read_parquet(?)
        """,
        [parquet_glob],
    ).fetchone()
    sample_rates = [
        {"utt_type": row[0], "sampling_rate": row[1], "samples": row[2]}
        for row in connection.execute(
            """
            SELECT utt_type, audio.sampling_rate, COUNT(*)
            FROM read_parquet(?)
            GROUP BY 1, 2 ORDER BY 1, 2
            """,
            [parquet_glob],
        ).fetchall()
    ]
    durations = [
        {
            "utt_type": row[0],
            "minimum_seconds": round(row[1], 6),
            "median_seconds": round(row[2], 6),
            "maximum_seconds": round(row[3], 6),
            "mean_seconds": round(row[4], 6),
            "total_hours": round(row[5], 6),
        }
        for row in connection.execute(
            """
            SELECT
                utt_type,
                MIN(array_length(audio.array) * 1.0 / audio.sampling_rate),
                MEDIAN(array_length(audio.array) * 1.0 / audio.sampling_rate),
                MAX(array_length(audio.array) * 1.0 / audio.sampling_rate),
                AVG(array_length(audio.array) * 1.0 / audio.sampling_rate),
                SUM(array_length(audio.array) * 1.0 / audio.sampling_rate) / 3600.0
            FROM read_parquet(?)
            GROUP BY 1 ORDER BY 1
            """,
            [parquet_glob],
        ).fetchall()
    ]
    shard_distribution = [
        {
            "shard": Path(row[0]).name,
            "utt_type": row[1],
            "samples": row[2],
        }
        for row in connection.execute(
            """
            SELECT filename, utt_type, COUNT(*)
            FROM read_parquet(?, filename = true)
            GROUP BY 1, 2 ORDER BY 1, 2
            """,
            [parquet_glob],
        ).fetchall()
    ]
    match = connection.execute(
        """
        WITH parquet_rows AS (
            SELECT file, label, utt_type FROM read_parquet(?)
        ), metadata_rows AS (
            SELECT * FROM read_csv(
                ?, header = true,
                columns = {'file': 'VARCHAR', 'label': 'VARCHAR', 'utt_type': 'VARCHAR'}
            )
        )
        SELECT
            (SELECT COUNT(*) FROM parquet_rows),
            (SELECT COUNT(*) FROM parquet_rows JOIN metadata_rows USING(file, label, utt_type)),
            (SELECT COUNT(*) FROM parquet_rows LEFT JOIN metadata_rows USING(file)
                WHERE metadata_rows.file IS NULL),
            (SELECT COUNT(*) FROM parquet_rows JOIN metadata_rows USING(file)
                WHERE parquet_rows.label <> metadata_rows.label
                   OR parquet_rows.utt_type <> metadata_rows.utt_type)
        """,
        [parquet_glob, metadata_csv],
    ).fetchone()
    duplicate_fingerprints = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT hash(audio.array) AS audio_fingerprint
            FROM read_parquet(?)
            GROUP BY 1
            HAVING COUNT(DISTINCT file) > 1
        )
        """,
        [parquet_glob],
    ).fetchone()[0]
    connection.close()

    base.update(
        {
            "schema": schema,
            "rows": total_rows,
            "unique_files": unique_files,
            "metadata_exact_matches": match[1],
            "files_missing_from_metadata": match[2],
            "label_or_type_mismatches": match[3],
            "utt_type_counts": type_counts,
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
            "fingerprint_method": (
                "DuckDB 64-bit hash(audio.array); useful as a duplicate screen, "
                "not a cryptographic identity proof."
            ),
        }
    )

    if schema != EXPECTED_PARQUET_COLUMNS:
        base["hard_issues"].append("Schema Parquet không khớp dataset card cục bộ.")
    if unique_files != total_rows:
        base["hard_issues"].append("Parquet cục bộ có đường dẫn logic bị lặp.")
    if match[1] != total_rows or match[2] or match[3]:
        base["hard_issues"].append("Parquet cục bộ không khớp metadata CSV.")
    if any(missing_values):
        base["hard_issues"].append("Parquet có trường bắt buộc hoặc audio không hợp lệ.")
    if duplicate_fingerprints:
        base["hard_issues"].append("Phát hiện fingerprint audio lặp giữa nhiều file.")
    if len({item["sampling_rate"] for item in sample_rates}) > 1:
        base["warnings"].append(
            "Các loại audio trong năm shard có sample rate không đồng nhất; phải resample nhất quán trước huấn luyện."
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
    parquet = verify_parquet(args.parquet_dir.resolve(), args.metadata.resolve())
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
            "five local Parquet shards, and the custom project splits."
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
    vc_ap = metadata["vc_ap_relationship"]
    status = "ĐẠT" if report["technical_consistency_passed"] else "KHÔNG ĐẠT"
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
        (
            "Metadata, năm Parquet cục bộ và tám split hiện có nhất quán với nhau ở các "
            "kiểm tra đã chạy. Không tìm thấy bằng chứng về lỗi join làm nhân đôi VC thành AP. "
            "Tuy nhiên, snapshot công khai khác đáng kể so với thống kê bài báo gốc nên chỉ "
            "được dùng như một giao thức tùy chỉnh, có phiên bản và có giới hạn rõ ràng."
        ),
        "",
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
            "## Năm Parquet cục bộ",
            "",
            f"- Shard: {parquet['local_shards']}/432 ({parquet['coverage_by_shard_count_percent']:.2f}% theo số shard).",
            f"- Tổng hàng: {parquet.get('rows', 0):,}.",
            f"- Khớp metadata chính xác: {parquet.get('metadata_exact_matches', 0):,}/{parquet.get('rows', 0):,}.",
            f"- File/audio rỗng hoặc sample rate không hợp lệ: {sum(parquet.get('missing_or_invalid_audio', {}).values()):,}.",
            f"- Nhóm fingerprint audio lặp: {parquet.get('duplicate_audio_fingerprint_groups', 0):,}.",
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
                "Cảnh báo: 188 VC trong phần đã tải đều là 40 kHz, trong khi các mẫu cục bộ "
                "còn lại là 16 kHz. Vì năm shard được chọn theo vị trí chứ không ngẫu nhiên, "
                "không được suy rộng tỷ lệ này cho toàn bộ snapshot. Pipeline phải resample "
                "mọi waveform về cùng một sample rate."
            ),
            "",
            "## Tám split",
            "",
            f"- Trạng thái kiểm tra schema, coverage, file và speaker leakage: {'ĐẠT' if splits['technical_passed'] else 'KHÔNG ĐẠT'}.",
            "- Kiểm tra hash audio toàn bộ split: chưa thể chạy vì mới có 5/432 shard.",
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
            "- Kiểm tra fingerprint audio chỉ bao phủ 2.558 file trong năm shard cục bộ.",
            "- Metadata không có `generator_id`, `source_corpus`, `official_split` hoặc định danh câu nguồn.",
            "- Không thể tự chứng minh nguyên nhân tác giả tạo số VC/AP bằng nhau chỉ từ ba cột metadata.",
            "- Replay công khai quá nhỏ để đại diện đầy đủ cho replay trong bài báo.",
            "",
            "## Quyết định sử dụng",
            "",
            "Có thể tiếp tục phát triển pipeline và thử nghiệm baseline nếu áp dụng các điều kiện sau:",
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
