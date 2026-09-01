#!/usr/bin/env python3
"""Audit VSASV metadata and emit reproducible JSON/Markdown reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "metadata_audit.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "metadata_audit.md"

EXPECTED_COLUMNS = ("file", "label", "utt_type")
EXPECTED_UTT_TYPES = (
    "bonafide",
    "voice_conversion",
    "adversarial_attack",
    "replay",
)
SPOOF_TYPES = set(EXPECTED_UTT_TYPES) - {"bonafide"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra schema, phân bố và tính toàn vẹn của metadata VSASV."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Đường dẫn CSV (mặc định: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_REPORT,
        help=f"Đầu ra JSON (mặc định: {DEFAULT_JSON_REPORT})",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT,
        help=f"Đầu ra Markdown (mặc định: {DEFAULT_MARKDOWN_REPORT})",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def audit_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metadata: {path}")

    total_rows = 0
    seen_rows: set[tuple[str, str, str]] = set()
    seen_files: set[str] = set()
    duplicate_row_count = 0
    duplicate_file_count = 0
    missing_counts: Counter[str] = Counter()
    utt_type_counts: Counter[str] = Counter()
    speakers_by_type: dict[str, set[str]] = defaultdict(set)
    samples_per_speaker: Counter[str] = Counter()
    label_prefix_mismatches = 0
    label_prefix_comparable = 0
    invalid_utt_type_count = 0
    malformed_row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        schema_valid = actual_columns == EXPECTED_COLUMNS

        for raw_row in reader:
            total_rows += 1
            if None in raw_row:
                malformed_row_count += 1

            row = {
                column: (raw_row.get(column) or "").strip()
                for column in EXPECTED_COLUMNS
            }
            file_path = row["file"]
            label = row["label"]
            utt_type = row["utt_type"]

            for column, value in row.items():
                if not value:
                    missing_counts[column] += 1

            row_key = (file_path, label, utt_type)
            if row_key in seen_rows:
                duplicate_row_count += 1
            else:
                seen_rows.add(row_key)

            if file_path:
                if file_path in seen_files:
                    duplicate_file_count += 1
                else:
                    seen_files.add(file_path)

            if utt_type:
                utt_type_counts[utt_type] += 1
                if utt_type not in EXPECTED_UTT_TYPES:
                    invalid_utt_type_count += 1

            if label:
                samples_per_speaker[label] += 1
                if utt_type:
                    speakers_by_type[utt_type].add(label)

            if file_path and label:
                label_prefix_comparable += 1
                normalized_parts = PurePosixPath(file_path.replace("\\", "/")).parts
                first_directory = normalized_parts[0] if normalized_parts else ""
                if first_directory != label:
                    label_prefix_mismatches += 1

    sample_counts = list(samples_per_speaker.values())
    binary_counts = {
        "bonafide": utt_type_counts.get("bonafide", 0),
        "spoof": sum(utt_type_counts.get(name, 0) for name in SPOOF_TYPES),
    }
    binary_counts["unclassified"] = total_rows - sum(binary_counts.values())

    issues: list[str] = []
    if not schema_valid:
        issues.append(
            f"Schema không khớp: cần {list(EXPECTED_COLUMNS)}, nhận {list(actual_columns)}."
        )
    if total_rows == 0:
        issues.append("Metadata không có dòng dữ liệu.")
    if sum(missing_counts.values()):
        issues.append(f"Có {sum(missing_counts.values())} giá trị bắt buộc bị thiếu.")
    if duplicate_row_count:
        issues.append(f"Có {duplicate_row_count} dòng trùng hoàn toàn.")
    if duplicate_file_count:
        issues.append(f"Có {duplicate_file_count} lần lặp đường dẫn file.")
    if invalid_utt_type_count:
        issues.append(f"Có {invalid_utt_type_count} dòng có utt_type không hợp lệ.")
    if malformed_row_count:
        issues.append(f"Có {malformed_row_count} dòng có số trường không khớp header.")
    if label_prefix_mismatches:
        issues.append(
            f"Có {label_prefix_mismatches} dòng mà label không khớp thư mục đầu của file."
        )

    utterance_types = {}
    for name in sorted(set(EXPECTED_UTT_TYPES) | set(utt_type_counts)):
        count = utt_type_counts.get(name, 0)
        utterance_types[name] = {
            "samples": count,
            "percentage": round((count / total_rows * 100) if total_rows else 0.0, 4),
            "speakers": len(speakers_by_type.get(name, set())),
        }

    return {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "path": display_path(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
        "passed": not issues,
        "issues": issues,
        "schema": {
            "expected_columns": list(EXPECTED_COLUMNS),
            "actual_columns": list(actual_columns),
            "valid": schema_valid,
            "malformed_rows": malformed_row_count,
        },
        "integrity": {
            "total_rows": total_rows,
            "unique_files": len(seen_files),
            "exact_duplicate_rows": duplicate_row_count,
            "duplicate_file_occurrences": duplicate_file_count,
            "missing_values": {
                column: missing_counts.get(column, 0) for column in EXPECTED_COLUMNS
            },
            "label_prefix_match": {
                "comparable_rows": label_prefix_comparable,
                "matching_rows": label_prefix_comparable - label_prefix_mismatches,
                "mismatching_rows": label_prefix_mismatches,
                "percentage": round(
                    (
                        (label_prefix_comparable - label_prefix_mismatches)
                        / label_prefix_comparable
                        * 100
                    )
                    if label_prefix_comparable
                    else 0.0,
                    4,
                ),
            },
        },
        "speakers": {
            "unique": len(samples_per_speaker),
            "samples_per_speaker": {
                "minimum": min(sample_counts) if sample_counts else 0,
                "median": statistics.median(sample_counts) if sample_counts else 0,
                "maximum": max(sample_counts) if sample_counts else 0,
            },
        },
        "utterance_types": utterance_types,
        "binary_distribution": {
            name: {
                "samples": count,
                "percentage": round((count / total_rows * 100) if total_rows else 0.0, 4),
            }
            for name, count in binary_counts.items()
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "ĐẠT" if report["passed"] else "KHÔNG ĐẠT"
    integrity = report["integrity"]
    speakers = report["speakers"]
    missing_total = sum(integrity["missing_values"].values())

    lines = [
        "# Báo cáo audit metadata VSASV",
        "",
        f"- **Trạng thái:** {status}",
        f"- **Metadata:** `{report['metadata']['path']}`",
        f"- **SHA-256:** `{report['metadata']['sha256']}`",
        f"- **Thời điểm UTC:** `{report['generated_at_utc']}`",
        "",
        "## Tính toàn vẹn",
        "",
        "| Kiểm tra | Kết quả |",
        "|---|---:|",
        f"| Tổng số dòng | {integrity['total_rows']:,} |",
        f"| Đường dẫn duy nhất | {integrity['unique_files']:,} |",
        f"| Dòng trùng hoàn toàn | {integrity['exact_duplicate_rows']:,} |",
        f"| Đường dẫn lặp | {integrity['duplicate_file_occurrences']:,} |",
        f"| Giá trị thiếu | {missing_total:,} |",
        f"| Speaker duy nhất | {speakers['unique']:,} |",
        f"| Mẫu/speaker nhỏ nhất | {speakers['samples_per_speaker']['minimum']:,} |",
        f"| Mẫu/speaker trung vị | {speakers['samples_per_speaker']['median']:,} |",
        f"| Mẫu/speaker lớn nhất | {speakers['samples_per_speaker']['maximum']:,} |",
        f"| Label khớp prefix file | {integrity['label_prefix_match']['percentage']:.2f}% |",
        "",
        "## Phân bố `utt_type`",
        "",
        "| utt_type | Số mẫu | Tỷ lệ | Số speaker |",
        "|---|---:|---:|---:|",
    ]
    for name, values in report["utterance_types"].items():
        lines.append(
            f"| `{name}` | {values['samples']:,} | {values['percentage']:.2f}% | "
            f"{values['speakers']:,} |"
        )

    lines.extend(
        [
            "",
            "## Phân bố nhị phân",
            "",
            "| Nhóm | Số mẫu | Tỷ lệ |",
            "|---|---:|---:|",
        ]
    )
    for name, values in report["binary_distribution"].items():
        lines.append(
            f"| `{name}` | {values['samples']:,} | {values['percentage']:.2f}% |"
        )

    lines.extend(["", "## Vấn đề phát hiện", ""])
    if report["issues"]:
        lines.extend(f"- {issue}" for issue in report["issues"])
    else:
        lines.append("Không phát hiện lỗi schema hoặc tính toàn vẹn trong các kiểm tra đã chạy.")

    lines.extend(
        [
            "",
            "## Giới hạn metadata",
            "",
            "- Không có `generator_id` hoặc `source_id`.",
            "- Không có nhóm TTS trong snapshot hiện tại.",
            "- Không thể dùng metadata này để tuyên bố tổng quát hóa trên unseen TTS engine.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any], json_output: Path, markdown_output: Path
) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_output.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    # Windows PowerShell may default to a legacy code page that cannot print
    # Vietnamese status messages. Keep the CLI deterministic across terminals.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    try:
        report = audit_metadata(args.metadata.resolve())
        write_reports(report, args.json_output.resolve(), args.markdown_output.resolve())
    except (OSError, csv.Error) as exc:
        print(f"Lỗi audit metadata: {exc}", file=sys.stderr)
        return 2

    print(
        f"Audit {'ĐẠT' if report['passed'] else 'KHÔNG ĐẠT'}: "
        f"{report['integrity']['total_rows']:,} dòng, "
        f"{report['speakers']['unique']:,} speaker."
    )
    print(f"JSON: {args.json_output.resolve()}")
    print(f"Markdown: {args.markdown_output.resolve()}")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"- {issue}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
