#!/usr/bin/env python3
"""Create deterministic, speaker-disjoint VSASV split protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "split_summary.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "split_summary.md"

EXPECTED_COLUMNS = ("file", "label", "utt_type")
EXPECTED_UTT_TYPES = {
    "bonafide",
    "voice_conversion",
    "adversarial_attack",
    "replay",
}
SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}
DEFAULT_SEED = 2026

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
        description="Tạo split VSASV cố định, tách biệt người nói."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument(
        "--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metadata: {path}")

    rows: list[dict[str, str]] = []
    seen_files: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != EXPECTED_COLUMNS:
            raise ValueError(
                f"Schema metadata phải là {list(EXPECTED_COLUMNS)}, "
                f"nhận {list(actual_columns)}."
            )

        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(f"Dòng {line_number} có số trường không hợp lệ.")
            row = {column: (raw_row.get(column) or "").strip() for column in EXPECTED_COLUMNS}
            if not all(row.values()):
                raise ValueError(f"Dòng {line_number} có giá trị bắt buộc bị thiếu.")
            if row["utt_type"] not in EXPECTED_UTT_TYPES:
                raise ValueError(
                    f"Dòng {line_number} có utt_type không hợp lệ: {row['utt_type']}"
                )
            if row["file"] in seen_files:
                raise ValueError(f"Đường dẫn file bị lặp: {row['file']}")
            normalized_parts = PurePosixPath(row["file"].replace("\\", "/")).parts
            if not normalized_parts or normalized_parts[0] != row["label"]:
                raise ValueError(
                    f"Dòng {line_number}: label không khớp thư mục đầu của file."
                )
            seen_files.add(row["file"])
            rows.append(row)

    if not rows:
        raise ValueError("Metadata không có dòng dữ liệu.")
    return rows


def classify_speakers(rows: Iterable[dict[str, str]]) -> dict[str, list[str]]:
    types_by_speaker: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        types_by_speaker[row["label"]].add(row["utt_type"])

    strata: dict[str, list[str]] = {
        "real_only": [],
        "vc_adversarial": [],
        "replay": [],
    }
    expected_patterns = {
        frozenset({"bonafide"}): "real_only",
        frozenset({"bonafide", "voice_conversion", "adversarial_attack"}): "vc_adversarial",
        frozenset({"bonafide", "replay"}): "replay",
    }
    unexpected: list[str] = []
    for speaker, types in sorted(types_by_speaker.items()):
        stratum = expected_patterns.get(frozenset(types))
        if stratum is None:
            unexpected.append(f"{speaker}: {sorted(types)}")
        else:
            strata[stratum].append(speaker)

    if unexpected:
        examples = "; ".join(unexpected[:5])
        raise ValueError(
            "Phát hiện mẫu utt_type ngoài ba strata đã khóa. "
            f"Ví dụ: {examples}"
        )
    return strata


def allocate_counts(total: int) -> dict[str, int]:
    """Allocate counts by largest remainder with a documented stable tie order."""
    names = tuple(SPLIT_RATIOS)
    exact = {name: total * SPLIT_RATIOS[name] for name in names}
    counts = {name: int(exact[name]) for name in names}
    remaining = total - sum(counts.values())
    tie_order = {"train": 0, "dev": 1, "test": 2}
    ranked = sorted(
        names,
        key=lambda name: (-(exact[name] - counts[name]), tie_order[name]),
    )
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def stable_speaker_order(speakers: Iterable[str], stratum: str, seed: int) -> list[str]:
    def rank(speaker: str) -> tuple[bytes, str]:
        value = f"{seed}\0{stratum}\0{speaker}".encode("utf-8")
        return hashlib.sha256(value).digest(), speaker

    return sorted(speakers, key=rank)


def assign_speakers(
    strata: dict[str, list[str]], seed: int
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    assignment: dict[str, str] = {}
    allocation: dict[str, dict[str, int]] = {}
    for stratum, speakers in strata.items():
        ordered = stable_speaker_order(speakers, stratum, seed)
        counts = allocate_counts(len(ordered))
        allocation[stratum] = counts
        start = 0
        for split_name in SPLIT_RATIOS:
            end = start + counts[split_name]
            for speaker in ordered[start:end]:
                assignment[speaker] = split_name
            start = end
    return assignment, allocation


def sort_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (row["label"], row["file"], row["utt_type"]))


def build_protocols(
    rows: list[dict[str, str]], assignment: dict[str, str], strata: dict[str, list[str]]
) -> dict[str, list[dict[str, str]]]:
    protocols: dict[str, list[dict[str, str]]] = {
        name: [] for name in SPLIT_FILENAMES
    }

    replay_speakers = set(strata["replay"])
    for row in rows:
        speaker = row["label"]
        utt_type = row["utt_type"]
        split_name = assignment[speaker]
        protocols[f"closed_{split_name}"].append(row)

        if speaker in replay_speakers:
            if utt_type in {"bonafide", "replay"}:
                protocols["open_unseen_test_replay"].append(row)
            continue

        if split_name == "train" and utt_type in {"bonafide", "voice_conversion"}:
            protocols["open_train_vc"].append(row)
        elif split_name == "dev" and utt_type in {"bonafide", "voice_conversion"}:
            protocols["open_dev_vc"].append(row)
        elif split_name == "test":
            if utt_type in {"bonafide", "voice_conversion"}:
                protocols["open_seen_test_vc"].append(row)
            if utt_type in {"bonafide", "adversarial_attack"}:
                protocols["open_unseen_test_adversarial"].append(row)

    return {name: sort_rows(split_rows) for name, split_rows in protocols.items()}


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    type_counts = Counter(row["utt_type"] for row in rows)
    return {
        "samples": len(rows),
        "speakers": len({row["label"] for row in rows}),
        "bonafide": type_counts.get("bonafide", 0),
        "spoof": sum(
            type_counts.get(name, 0)
            for name in ("voice_conversion", "adversarial_attack", "replay")
        ),
        "utt_type_counts": {
            name: type_counts.get(name, 0) for name in sorted(EXPECTED_UTT_TYPES)
        },
    }


def generate_splits(
    metadata_path: Path, output_dir: Path, seed: int = DEFAULT_SEED
) -> dict[str, object]:
    rows = read_metadata(metadata_path)
    strata = classify_speakers(rows)
    assignment, allocation = assign_speakers(strata, seed)
    protocols = build_protocols(rows, assignment, strata)

    split_reports: dict[str, object] = {}
    for logical_name, filename in SPLIT_FILENAMES.items():
        path = output_dir / filename
        write_csv_atomic(path, protocols[logical_name])
        split_reports[logical_name] = {
            "path": display_path(path),
            "sha256": sha256_file(path),
            **summarize_rows(protocols[logical_name]),
        }

    return {
        "split_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "metadata": {
            "path": display_path(metadata_path),
            "sha256": sha256_file(metadata_path),
            "samples": len(rows),
            "speakers": len({row["label"] for row in rows}),
        },
        "speaker_strata": {
            name: {
                "speakers": len(speakers),
                "allocation": allocation[name],
            }
            for name, speakers in strata.items()
        },
        "splits": split_reports,
        "notes": [
            "Replay speakers are excluded from open train/dev/seen/adversarial and reserved for open unseen replay.",
            "Open seen VC and unseen adversarial reuse the same bonafide test rows for a comparable negative pool.",
        ],
    }


def render_markdown(report: dict[str, object]) -> str:
    metadata = report["metadata"]
    lines = [
        "# Báo cáo tạo split VSASV",
        "",
        f"- **Seed:** `{report['seed']}`",
        f"- **Metadata:** `{metadata['path']}`",
        f"- **SHA-256 metadata:** `{metadata['sha256']}`",
        f"- **Số mẫu:** {metadata['samples']:,}",
        f"- **Số speaker:** {metadata['speakers']:,}",
        f"- **Thời điểm UTC:** `{report['generated_at_utc']}`",
        "",
        "## Phân bổ speaker theo strata",
        "",
        "| Stratum | Tổng speaker | Train | Dev | Test |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, details in report["speaker_strata"].items():
        allocation = details["allocation"]
        lines.append(
            f"| `{name}` | {details['speakers']:,} | {allocation['train']:,} | "
            f"{allocation['dev']:,} | {allocation['test']:,} |"
        )

    lines.extend(
        [
            "",
            "## Kết quả từng split",
            "",
            "| Split | Mẫu | Speaker | Bonafide | Spoof | SHA-256 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, details in report["splits"].items():
        lines.append(
            f"| `{name}` | {details['samples']:,} | {details['speakers']:,} | "
            f"{details['bonafide']:,} | {details['spoof']:,} | "
            f"`{details['sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Quy ước quan trọng",
            "",
            "- Toàn bộ split được xếp theo speaker bằng SHA-256 của seed, stratum và speaker ID.",
            "- Speaker replay được giữ ngoài toàn bộ open train/dev/seen/adversarial.",
            "- Seen VC test và unseen adversarial test dùng chung tập bonafide test. Đây là overlap có chủ đích giữa hai phép đánh giá, không phải rò rỉ train/test.",
            "- CSV giữ nguyên ba cột nguồn: `file,label,utt_type`.",
            "",
        ]
    )
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
    try:
        report = generate_splits(args.metadata, args.output_dir, args.seed)
        write_reports(report, args.json_report, args.markdown_report)
    except (OSError, ValueError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        return 1

    print(
        f"Đã tạo {len(SPLIT_FILENAMES)} split trong {display_path(args.output_dir)} "
        f"với seed {args.seed}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
