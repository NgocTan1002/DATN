#!/usr/bin/env python3
"""Validate VSASV split integrity, protocol constraints, and leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "leakage_check.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "leakage_check.md"

EXPECTED_COLUMNS = ("file", "label", "utt_type")
EXPECTED_UTT_TYPES = {
    "bonafide",
    "voice_conversion",
    "adversarial_attack",
    "replay",
}
SPLIT_FILENAMES = {
    "closed_train": "closed_train.csv",
    "closed_dev": "closed_dev.csv",
    "closed_test": "closed_test.csv",
    "open_train_vc": "open_train_vc.csv",
    "open_dev_vc": "open_dev_vc.csv",
    "open_seen_test_vc": "open_seen_test_vc.csv",
    "open_unseen_test_adversarial": "open_unseen_test_adversarial.csv",
    "open_unseen_test_replay": "open_unseen_test_replay.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra rò rỉ speaker, file, protocol và tùy chọn audio hash."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Nếu cung cấp, kiểm tra file tồn tại và SHA-256 nội dung audio.",
    )
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument(
        "--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT
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


def add_issue(
    issues: list[dict[str, object]], code: str, message: str, **details: object
) -> None:
    issue: dict[str, object] = {"code": code, "message": message}
    if details:
        issue["details"] = details
    issues.append(issue)


def read_csv_rows(
    path: Path, issues: list[dict[str, object]], source_name: str
) -> list[dict[str, str]]:
    if not path.is_file():
        add_issue(issues, "missing_file", f"Không tìm thấy {source_name}: {path}")
        return []

    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            actual_columns = tuple(reader.fieldnames or ())
            if actual_columns != EXPECTED_COLUMNS:
                add_issue(
                    issues,
                    "schema_mismatch",
                    f"Schema {source_name} không hợp lệ.",
                    expected=list(EXPECTED_COLUMNS),
                    actual=list(actual_columns),
                )
            for line_number, raw_row in enumerate(reader, start=2):
                if None in raw_row:
                    add_issue(
                        issues,
                        "malformed_row",
                        f"{source_name} dòng {line_number} có số trường không hợp lệ.",
                    )
                row = {
                    column: (raw_row.get(column) or "").strip()
                    for column in EXPECTED_COLUMNS
                }
                if not all(row.values()):
                    add_issue(
                        issues,
                        "missing_value",
                        f"{source_name} dòng {line_number} thiếu giá trị bắt buộc.",
                    )
                rows.append(row)
    except OSError as error:
        add_issue(issues, "read_error", f"Không đọc được {source_name}: {error}")
    return rows


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["file"], row["label"], row["utt_type"]


def file_set(rows: Iterable[dict[str, str]]) -> set[str]:
    return {row["file"] for row in rows}


def speaker_set(rows: Iterable[dict[str, str]]) -> set[str]:
    return {row["label"] for row in rows}


def row_set(rows: Iterable[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {row_key(row) for row in rows}


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    types = Counter(row["utt_type"] for row in rows)
    files = [row["file"] for row in rows]
    keys = [row_key(row) for row in rows]
    return {
        "samples": len(rows),
        "speakers": len(speaker_set(rows)),
        "unique_files": len(set(files)),
        "duplicate_files": len(files) - len(set(files)),
        "duplicate_rows": len(keys) - len(set(keys)),
        "utt_type_counts": {
            name: types.get(name, 0) for name in sorted(EXPECTED_UTT_TYPES)
        },
    }


def check_internal_split(
    name: str,
    rows: list[dict[str, str]],
    metadata_rows: set[tuple[str, str, str]],
    issues: list[dict[str, object]],
) -> None:
    summary = summarize_rows(rows)
    if summary["duplicate_files"]:
        add_issue(
            issues,
            "duplicate_file",
            f"{name} có {summary['duplicate_files']} đường dẫn file lặp.",
        )
    if summary["duplicate_rows"]:
        add_issue(
            issues,
            "duplicate_row",
            f"{name} có {summary['duplicate_rows']} dòng lặp.",
        )

    invalid_types = sorted(
        {row["utt_type"] for row in rows if row["utt_type"] not in EXPECTED_UTT_TYPES}
    )
    if invalid_types:
        add_issue(
            issues,
            "invalid_utt_type",
            f"{name} có utt_type không hợp lệ.",
            values=invalid_types,
        )

    unknown_rows = [row_key(row) for row in rows if row_key(row) not in metadata_rows]
    if unknown_rows:
        add_issue(
            issues,
            "unknown_row",
            f"{name} có {len(unknown_rows)} dòng không khớp metadata nguồn.",
            examples=unknown_rows[:5],
        )

    prefix_mismatches = []
    for row in rows:
        parts = PurePosixPath(row["file"].replace("\\", "/")).parts
        if not parts or parts[0] != row["label"]:
            prefix_mismatches.append(row["file"])
    if prefix_mismatches:
        add_issue(
            issues,
            "label_path_mismatch",
            f"{name} có {len(prefix_mismatches)} label không khớp đường dẫn.",
            examples=prefix_mismatches[:5],
        )


def check_pairwise_disjoint(
    split_names: list[str],
    splits: dict[str, list[dict[str, str]]],
    issues: list[dict[str, object]],
    check_speakers: bool = True,
    check_files: bool = True,
) -> None:
    for left, right in combinations(split_names, 2):
        if check_speakers:
            overlap = speaker_set(splits[left]) & speaker_set(splits[right])
            if overlap:
                add_issue(
                    issues,
                    "speaker_overlap",
                    f"{left} và {right} trùng {len(overlap)} speaker.",
                    examples=sorted(overlap)[:10],
                )
        if check_files:
            overlap = file_set(splits[left]) & file_set(splits[right])
            if overlap:
                add_issue(
                    issues,
                    "file_overlap",
                    f"{left} và {right} trùng {len(overlap)} file.",
                    examples=sorted(overlap)[:10],
                )


def check_exact_rows(
    actual: set[tuple[str, str, str]],
    expected: set[tuple[str, str, str]],
    label: str,
    issues: list[dict[str, object]],
) -> None:
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        add_issue(
            issues,
            "coverage_mismatch",
            f"{label} không khớp tập dòng kỳ vọng.",
            missing=len(missing),
            extra=len(extra),
            missing_examples=sorted(missing)[:5],
            extra_examples=sorted(extra)[:5],
        )


def check_attack_types(
    name: str,
    rows: list[dict[str, str]],
    allowed: set[str],
    issues: list[dict[str, object]],
) -> None:
    actual = {row["utt_type"] for row in rows}
    invalid = actual - allowed
    if invalid:
        add_issue(
            issues,
            "attack_constraint",
            f"{name} chứa utt_type ngoài giao thức.",
            allowed=sorted(allowed),
            invalid=sorted(invalid),
        )


def check_protocols(
    metadata: list[dict[str, str]],
    splits: dict[str, list[dict[str, str]]],
    issues: list[dict[str, object]],
) -> None:
    metadata_set = row_set(metadata)
    closed = ["closed_train", "closed_dev", "closed_test"]
    check_pairwise_disjoint(closed, splits, issues)
    check_exact_rows(
        set().union(*(row_set(splits[name]) for name in closed)),
        metadata_set,
        "Closed protocol",
        issues,
    )

    open_base = ["open_train_vc", "open_dev_vc", "open_seen_test_vc"]
    check_pairwise_disjoint(open_base, splits, issues)
    check_pairwise_disjoint(
        [
            "open_train_vc",
            "open_dev_vc",
            "open_unseen_test_adversarial",
            "open_unseen_test_replay",
        ],
        splits,
        issues,
    )
    check_pairwise_disjoint(
        ["open_seen_test_vc", "open_unseen_test_replay"], splits, issues
    )

    constraints = {
        "open_train_vc": {"bonafide", "voice_conversion"},
        "open_dev_vc": {"bonafide", "voice_conversion"},
        "open_seen_test_vc": {"bonafide", "voice_conversion"},
        "open_unseen_test_adversarial": {"bonafide", "adversarial_attack"},
        "open_unseen_test_replay": {"bonafide", "replay"},
    }
    for name, allowed in constraints.items():
        check_attack_types(name, splits[name], allowed, issues)

    seen_speakers = speaker_set(splits["open_seen_test_vc"])
    adversarial_speakers = speaker_set(splits["open_unseen_test_adversarial"])
    if seen_speakers != adversarial_speakers:
        add_issue(
            issues,
            "test_cohort_mismatch",
            "Seen VC test và unseen adversarial test không có cùng speaker cohort.",
            only_seen=len(seen_speakers - adversarial_speakers),
            only_adversarial=len(adversarial_speakers - seen_speakers),
        )

    seen_bonafide = {
        row_key(row)
        for row in splits["open_seen_test_vc"]
        if row["utt_type"] == "bonafide"
    }
    adversarial_bonafide = {
        row_key(row)
        for row in splits["open_unseen_test_adversarial"]
        if row["utt_type"] == "bonafide"
    }
    if seen_bonafide != adversarial_bonafide:
        add_issue(
            issues,
            "bonafide_pool_mismatch",
            "Seen VC test và unseen adversarial test không dùng cùng bonafide pool.",
            only_seen=len(seen_bonafide - adversarial_bonafide),
            only_adversarial=len(adversarial_bonafide - seen_bonafide),
        )

    types_by_speaker: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        types_by_speaker[row["label"]].add(row["utt_type"])
    replay_speakers = {
        speaker for speaker, types in types_by_speaker.items() if "replay" in types
    }
    non_replay_speakers = set(types_by_speaker) - replay_speakers

    expected_open_base = {
        row_key(row)
        for row in metadata
        if row["label"] in non_replay_speakers
        and row["utt_type"] in {"bonafide", "voice_conversion"}
    }
    actual_open_base = set().union(
        *(row_set(splits[name]) for name in open_base)
    )
    check_exact_rows(actual_open_base, expected_open_base, "Open train/dev/seen", issues)

    expected_replay = {
        row_key(row) for row in metadata if row["label"] in replay_speakers
    }
    check_exact_rows(
        row_set(splits["open_unseen_test_replay"]),
        expected_replay,
        "Open unseen replay",
        issues,
    )

    expected_adversarial = {
        row_key(row)
        for row in metadata
        if row["label"] in seen_speakers
        and row["utt_type"] in {"bonafide", "adversarial_attack"}
    }
    check_exact_rows(
        row_set(splits["open_unseen_test_adversarial"]),
        expected_adversarial,
        "Open unseen adversarial",
        issues,
    )


def check_audio_hashes(
    splits: dict[str, list[dict[str, str]]],
    audio_root: Path,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    root = audio_root.resolve()
    unique_files = sorted(set().union(*(file_set(rows) for rows in splits.values())))
    hashes: dict[str, str] = {}
    missing = 0
    unsafe = 0

    for logical_file in unique_files:
        # Build from path parts to avoid treating a logical POSIX path as absolute on Windows.
        parts = PurePosixPath(logical_file.replace("\\", "/")).parts
        candidate = root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            unsafe += 1
            add_issue(
                issues,
                "unsafe_audio_path",
                f"Đường dẫn audio thoát khỏi audio root: {logical_file}",
            )
            continue
        if not candidate.is_file():
            missing += 1
            if missing <= 10:
                add_issue(
                    issues,
                    "missing_audio",
                    f"Không tìm thấy audio: {logical_file}",
                )
            continue
        hashes[logical_file] = sha256_file(candidate)

    role_groups = {
        "closed_train": "closed_train",
        "closed_dev": "closed_dev",
        "closed_test": "closed_test",
        "open_train_vc": "open_train",
        "open_dev_vc": "open_dev",
        "open_seen_test_vc": "open_test_main",
        "open_unseen_test_adversarial": "open_test_main",
        "open_unseen_test_replay": "open_test_replay",
    }
    hash_occurrences: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split_name, rows in splits.items():
        group = role_groups[split_name]
        for logical_file in file_set(rows):
            if logical_file in hashes:
                hash_occurrences[(group, hashes[logical_file])].add(logical_file)

    hash_groups: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for (group, digest), logical_files in hash_occurrences.items():
        hash_groups[digest][group] = logical_files

    overlap_count = 0
    for digest, groups in hash_groups.items():
        closed_groups = {name for name in groups if name.startswith("closed_")}
        open_groups = {name for name in groups if name.startswith("open_")}
        leaking = len(closed_groups) > 1 or len(open_groups) > 1
        distinct_files = set().union(*groups.values())
        if leaking and len(distinct_files) > 1:
            overlap_count += 1
            if overlap_count <= 20:
                add_issue(
                    issues,
                    "audio_hash_overlap",
                    "Nội dung audio giống nhau xuất hiện ở nhiều partition.",
                    sha256=digest,
                    groups={name: sorted(files)[:5] for name, files in groups.items()},
                )

    return {
        "enabled": True,
        "audio_root": str(root),
        "logical_files": len(unique_files),
        "hashed_files": len(hashes),
        "missing_files": missing,
        "unsafe_paths": unsafe,
        "cross_partition_hashes": overlap_count,
    }


def check_splits(
    metadata_path: Path, split_dir: Path, audio_root: Path | None = None
) -> dict[str, object]:
    issues: list[dict[str, object]] = []
    metadata = read_csv_rows(metadata_path, issues, "metadata")
    metadata_set = row_set(metadata)

    metadata_files = [row["file"] for row in metadata]
    if len(metadata_files) != len(set(metadata_files)):
        add_issue(issues, "metadata_duplicate_file", "Metadata nguồn có file lặp.")

    splits: dict[str, list[dict[str, str]]] = {}
    for name, filename in SPLIT_FILENAMES.items():
        rows = read_csv_rows(split_dir / filename, issues, name)
        splits[name] = rows
        check_internal_split(name, rows, metadata_set, issues)

    if metadata and all((split_dir / filename).is_file() for filename in SPLIT_FILENAMES.values()):
        check_protocols(metadata, splits, issues)

    if audio_root is None:
        audio_check: dict[str, object] = {
            "enabled": False,
            "reason": "Không cung cấp --audio-root; chưa kiểm tra tồn tại và hash audio.",
        }
    else:
        audio_check = check_audio_hashes(splits, audio_root, issues)

    return {
        "check_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not issues,
        "metadata": {
            "path": display_path(metadata_path),
            "samples": len(metadata),
            "speakers": len(speaker_set(metadata)),
        },
        "split_dir": display_path(split_dir),
        "splits": {name: summarize_rows(rows) for name, rows in splits.items()},
        "audio_check": audio_check,
        "issues": issues,
    }


def render_markdown(report: dict[str, object]) -> str:
    status = "ĐẠT" if report["passed"] else "KHÔNG ĐẠT"
    lines = [
        "# Báo cáo kiểm tra rò rỉ dữ liệu",
        "",
        f"- **Trạng thái:** {status}",
        f"- **Metadata:** `{report['metadata']['path']}`",
        f"- **Thư mục split:** `{report['split_dir']}`",
        f"- **Thời điểm UTC:** `{report['generated_at_utc']}`",
        "",
        "## Tóm tắt split",
        "",
        "| Split | Mẫu | Speaker | File duy nhất | File lặp |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in report["splits"].items():
        lines.append(
            f"| `{name}` | {summary['samples']:,} | {summary['speakers']:,} | "
            f"{summary['unique_files']:,} | {summary['duplicate_files']:,} |"
        )

    lines.extend(["", "## Kiểm tra audio", ""])
    audio = report["audio_check"]
    if audio["enabled"]:
        lines.extend(
            [
                f"- File logic: {audio['logical_files']:,}",
                f"- File đã hash: {audio['hashed_files']:,}",
                f"- File thiếu: {audio['missing_files']:,}",
                f"- Hash trùng qua partition: {audio['cross_partition_hashes']:,}",
            ]
        )
    else:
        lines.append(f"- {audio['reason']}")

    lines.extend(["", "## Vấn đề phát hiện", ""])
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- `{issue['code']}`: {issue['message']}")
    else:
        lines.append("Không phát hiện rò rỉ speaker, file hoặc vi phạm giao thức.")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    report: dict[str, object], json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    report = check_splits(args.metadata, args.split_dir, args.audio_root)
    try:
        write_reports(report, args.json_report, args.markdown_report)
    except OSError as error:
        print(f"LỖI khi ghi báo cáo: {error}", file=sys.stderr)
        return 1

    if report["passed"]:
        print("ĐẠT: Không phát hiện rò rỉ speaker, file hoặc vi phạm giao thức.")
        return 0
    print(
        f"KHÔNG ĐẠT: Phát hiện {len(report['issues'])} vấn đề. "
        f"Xem {display_path(args.markdown_report)}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
