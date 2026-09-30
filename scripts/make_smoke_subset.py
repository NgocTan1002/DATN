#!/usr/bin/env python3
"""Create deterministic, balanced smoke manifests from local closed data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    SmokeRow,
    select_smoke_rows,
    summarize_smoke_rows,
    validate_smoke_partitions,
)


DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "smoke_subset_summary.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "smoke_subset_summary.md"
DEFAULT_SEED = 2026
DEFAULT_PER_CLASS = {"train": 128, "dev": 64, "test": 32}
CSV_COLUMNS = ("file", "label", "utt_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo smoke subset cân bằng, cố định từ các shard cục bộ."
    )
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--dev-per-class", type=int, default=64)
    parser.add_argument("--test-per-class", type=int, default=32)
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


def build_local_index(parquet_dir: Path) -> dict[str, dict[str, Any]]:
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy Parquet trong {parquet_dir}")
    parquet_glob = (parquet_dir / "*.parquet").resolve().as_posix()
    connection = duckdb.connect(database=":memory:")
    try:
        rows = connection.execute(
            """
            SELECT file, label, utt_type, audio.sampling_rate,
                   array_length(audio.array)
            FROM read_parquet(?) ORDER BY file
            """,
            [parquet_glob],
        ).fetchall()
    finally:
        connection.close()

    local_index: dict[str, dict[str, Any]] = {}
    for logical_file, speaker_id, utt_type, sample_rate, sample_count in rows:
        if logical_file in local_index:
            raise ValueError(f"File cục bộ bị lặp: {logical_file}")
        local_index[str(logical_file)] = {
            "speaker_id": str(speaker_id),
            "utt_type": str(utt_type),
            "sample_rate": int(sample_rate),
            "sample_count": int(sample_count),
        }
    return local_index


def read_local_candidates(
    split_path: Path,
    local_index: dict[str, dict[str, Any]],
) -> tuple[list[SmokeRow], int]:
    candidates: list[SmokeRow] = []
    total_rows = 0
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"Schema split không hợp lệ: {split_path}")
        for row in reader:
            total_rows += 1
            local = local_index.get(row["file"])
            if local is None:
                continue
            if (row["label"], row["utt_type"]) != (
                local["speaker_id"],
                local["utt_type"],
            ):
                raise ValueError(f"Metadata split/Parquet không khớp: {row['file']}")
            candidates.append(
                SmokeRow(
                    file=row["file"],
                    speaker_id=row["label"],
                    utt_type=row["utt_type"],
                )
            )
    return candidates, total_rows


def write_manifest(path: Path, rows: list[SmokeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.as_csv_row() for row in rows)
    os.replace(temporary, path)


def generate_smoke_subset(
    parquet_dir: Path,
    split_dir: Path,
    output_dir: Path,
    seed: int,
    per_class: dict[str, int],
) -> dict[str, Any]:
    if any(value <= 0 for value in per_class.values()):
        raise ValueError("Mọi quota per-class phải lớn hơn 0.")
    local_index = build_local_index(parquet_dir)
    partitions: dict[str, list[SmokeRow]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "dev", "test"):
        source_path = split_dir / f"closed_{split_name}.csv"
        candidates, total_rows = read_local_candidates(source_path, local_index)
        partitions[split_name] = select_smoke_rows(
            candidates,
            split_name=split_name,
            per_class=per_class[split_name],
            seed=seed,
        )
        sources[split_name] = {
            "path": display_path(source_path),
            "sha256": sha256_file(source_path),
            "total_rows": total_rows,
            "local_candidates": len(candidates),
        }

    checks = validate_smoke_partitions(partitions)
    manifest_reports: dict[str, dict[str, Any]] = {}
    for split_name, rows in partitions.items():
        output_path = output_dir / f"smoke_{split_name}.csv"
        write_manifest(output_path, rows)
        rates = Counter(local_index[row.file]["sample_rate"] for row in rows)
        durations = [
            local_index[row.file]["sample_count"]
            / local_index[row.file]["sample_rate"]
            for row in rows
        ]
        manifest_reports[split_name] = {
            "path": display_path(output_path),
            "sha256": sha256_file(output_path),
            **summarize_smoke_rows(rows),
            "sample_rate_counts": {
                str(rate): count for rate, count in sorted(rates.items())
            },
            "duration_seconds": {
                "minimum": min(durations),
                "mean": sum(durations) / len(durations),
                "maximum": max(durations),
            },
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "technical_passed": True,
        "purpose": "code_smoke_test_only_not_scientific_evaluation",
        "seed": seed,
        "selection": {
            "stable_order": "sha256(seed, split, purpose, file)",
            "balanced_binary_labels": True,
            "spoof_type_strategy": "deterministic_round_robin_across_available_types",
            "per_class": per_class,
        },
        "source": {
            "parquet_directory": display_path(parquet_dir),
            "local_files": len(local_index),
            "closed_splits": sources,
        },
        "checks": checks,
        "manifests": manifest_reports,
        "warnings": [
            "Subset chỉ dùng để kiểm tra code và tốc độ, không dùng để báo cáo EER khoa học.",
            "Phân bố kiểu spoof bị giới hạn bởi năm shard hiện có; test cục bộ chỉ có replay.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Báo cáo smoke subset VSASV",
        "",
        f"- **Trạng thái kỹ thuật:** {'ĐẠT' if report['technical_passed'] else 'KHÔNG ĐẠT'}",
        f"- **Seed:** `{report['seed']}`",
        f"- **Tổng mẫu:** {report['checks']['total_samples']:,}",
        "- **Mục đích:** chỉ kiểm tra code, không dùng làm kết quả khoa học",
        "",
        "## Manifest",
        "",
        "| Split | Mẫu | Speaker | Bonafide | Spoof | VC | AP | Replay | SHA-256 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split_name in ("train", "dev", "test"):
        item = report["manifests"][split_name]
        types = item["utt_type_counts"]
        lines.append(
            f"| `{split_name}` | {item['samples']:,} | {item['speakers']:,} | "
            f"{item['bonafide']:,} | {item['spoof']:,} | "
            f"{types['voice_conversion']:,} | {types['adversarial_attack']:,} | "
            f"{types['replay']:,} | `{item['sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Kiểm tra bắt buộc",
            "",
            "- Cân bằng 50/50 trong từng partition: ĐẠT.",
            "- File trùng: 0.",
            "- Mọi file đều tồn tại trong năm shard cục bộ: ĐẠT.",
            "- Speaker overlap train/dev/test: 0.",
            "- Thứ tự chọn xác định bằng SHA-256 và seed `2026`.",
            "",
            "## Nguồn cục bộ",
            "",
            "| Split | Dòng closed split | Ứng viên cục bộ | Tỷ lệ |",
            "|---|---:|---:|---:|",
        ]
    )
    for split_name in ("train", "dev", "test"):
        item = report["source"]["closed_splits"][split_name]
        lines.append(
            f"| `{split_name}` | {item['total_rows']:,} | "
            f"{item['local_candidates']:,} | "
            f"{item['local_candidates'] / item['total_rows']:.2%} |"
        )
    lines.extend(["", "## Cảnh báo", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    per_class = {
        "train": args.train_per_class,
        "dev": args.dev_per_class,
        "test": args.test_per_class,
    }
    try:
        report = generate_smoke_subset(
            args.parquet_dir.resolve(),
            args.split_dir.resolve(),
            args.output_dir.resolve(),
            args.seed,
            per_class,
        )
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.markdown_report.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lỗi tạo smoke subset: {error}", file=sys.stderr)
        return 2

    print(
        f"Smoke subset ĐẠT: {report['checks']['total_samples']:,} mẫu, "
        f"seed {report['seed']}."
    )
    for split_name, item in report["manifests"].items():
        print(f"{split_name}: {item['samples']} mẫu, sha256={item['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
