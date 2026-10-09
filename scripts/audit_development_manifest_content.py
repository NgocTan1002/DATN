#!/usr/bin/env python3
"""Băm lại độc lập toàn bộ waveform trong development manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_development_manifest import (
    MANIFEST_COLUMNS,
    display_path,
    sha256_file,
)
from src.data.content_hash import waveform_sha256


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "development_20k_v2.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_JSON_REPORT = (
    PROJECT_ROOT / "reports" / "development_manifest_20k_v2_content_audit.json"
)
DEFAULT_MARKDOWN_REPORT = (
    PROJECT_ROOT / "reports" / "development_manifest_20k_v2_content_audit.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"Schema manifest không hợp lệ: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Manifest không có dữ liệu: {path}")
    files = [row["file"] for row in rows]
    if len(set(files)) != len(files):
        raise ValueError("Manifest có file path bị lặp.")
    if len({row["manifest_version"] for row in rows}) != 1:
        raise ValueError("Manifest phải có đúng một manifest_version.")
    if len({row["source_snapshot"] for row in rows}) != 1:
        raise ValueError("Manifest phải có đúng một source_snapshot.")
    return rows


def _hash_manifest_waveforms(
    rows: list[dict[str, str]],
    parquet_dir: Path,
) -> dict[str, str]:
    rows_by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    expected_by_file: dict[str, dict[str, str]] = {}
    for row in rows:
        shard_name = row["shard"]
        if Path(shard_name).name != shard_name:
            raise ValueError(f"Tên shard không hợp lệ: {shard_name}")
        rows_by_shard[shard_name].append(row)
        expected_by_file[row["file"]] = row

    hashes_by_file: dict[str, str] = {}
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '8GB'")
    try:
        for shard_name in sorted(rows_by_shard):
            parquet_path = parquet_dir / shard_name
            if not parquet_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy shard: {parquet_path}")
            expected_files = sorted(row["file"] for row in rows_by_shard[shard_name])
            cursor = connection.execute(
                """
                SELECT file, label, utt_type, audio.sampling_rate, audio.array
                FROM read_parquet(?)
                WHERE file IN ?
                ORDER BY file
                """,
                [parquet_path.resolve().as_posix(), expected_files],
            )
            while True:
                batch = cursor.fetchmany(32)
                if not batch:
                    break
                for file_name, speaker_id, utt_type, sample_rate, waveform in batch:
                    logical_file = str(file_name)
                    if logical_file in hashes_by_file:
                        raise ValueError(
                            f"Waveform xuất hiện nhiều lần trong Parquet: {logical_file}"
                        )
                    expected = expected_by_file.get(logical_file)
                    if expected is None:
                        raise ValueError(f"Đọc ngoài phạm vi manifest: {logical_file}")
                    rate = int(sample_rate or 0)
                    if str(speaker_id) != expected["speaker_id"]:
                        raise ValueError(f"Speaker không khớp manifest: {logical_file}")
                    if str(utt_type) != expected["utt_type"]:
                        raise ValueError(f"utt_type không khớp manifest: {logical_file}")
                    if rate != int(expected["native_sample_rate"]):
                        raise ValueError(f"Sample rate không khớp manifest: {logical_file}")
                    hashes_by_file[logical_file] = waveform_sha256(
                        waveform or [], rate
                    )
    finally:
        connection.close()

    missing = sorted(set(expected_by_file) - set(hashes_by_file))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"Thiếu {len(missing)} waveform của manifest trong Parquet: {preview}"
        )
    return hashes_by_file


def audit_development_manifest_content(
    manifest_path: Path,
    parquet_dir: Path,
) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    hashes_by_file = _hash_manifest_waveforms(rows, parquet_dir)
    row_hashes = [
        {
            "manifest_row": index,
            "manifest_version": row["manifest_version"],
            "shard": row["shard"],
            "file": row["file"],
            "speaker_id": row["speaker_id"],
            "split": row["split"],
            "binary_label": int(row["binary_label"]),
            "utt_type": row["utt_type"],
            "native_sample_rate": int(row["native_sample_rate"]),
            "waveform_sha256": hashes_by_file[row["file"]],
        }
        for index, row in enumerate(rows, start=2)
    ]

    rows_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in row_hashes:
        rows_by_hash[row["waveform_sha256"]].append(row)

    duplicate_groups: list[dict[str, Any]] = []
    for content_hash, members in sorted(rows_by_hash.items()):
        if len(members) <= 1:
            continue
        split_counts = Counter(str(member["split"]) for member in members)
        duplicate_groups.append(
            {
                "waveform_sha256": content_hash,
                "files": [str(member["file"]) for member in members],
                "splits": sorted(split_counts),
                "speaker_ids": sorted(
                    {str(member["speaker_id"]) for member in members}
                ),
                "within_split": any(count > 1 for count in split_counts.values()),
                "crossing_splits": len(split_counts) > 1,
            }
        )

    within_split = sum(group["within_split"] for group in duplicate_groups)
    crossing_splits = sum(group["crossing_splits"] for group in duplicate_groups)
    technical_passed = not duplicate_groups
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ĐẠT" if technical_passed else "KHÔNG ĐẠT",
        "technical_passed": technical_passed,
        "scope": "Chỉ băm lại các waveform được liệt kê trong development manifest.",
        "source_snapshot_status": (
            "Không đánh giá lại duplicate toàn snapshot; xem "
            "reports/vsasv_snapshot_verification.json."
        ),
        "manifest": {
            "path": display_path(manifest_path),
            "sha256": sha256_file(manifest_path),
            "version": rows[0]["manifest_version"],
            "source_snapshot": rows[0]["source_snapshot"],
            "samples": len(rows),
            "counts_by_split": dict(
                sorted(Counter(row["split"] for row in rows).items())
            ),
            "shards": len({row["shard"] for row in rows}),
        },
        "hash_method": (
            "sha256(sample_rate_signed_int64_le || waveform_float64_le)"
        ),
        "checks": {
            "expected_files": len(rows),
            "audited_files": len(row_hashes),
            "unique_content_hashes": len(rows_by_hash),
            "duplicate_groups_total": len(duplicate_groups),
            "duplicate_groups_within_split": within_split,
            "duplicate_groups_crossing_splits": crossing_splits,
            "metadata_and_sample_rate_match": True,
        },
        "duplicate_groups": duplicate_groups,
        "row_hashes": row_hashes,
    }


def render_markdown(report: dict[str, Any]) -> str:
    manifest = report["manifest"]
    checks = report["checks"]
    lines = [
        "# Audit nội dung development manifest 20k v2",
        "",
        f"- **Trạng thái manifest:** {report['status']}",
        f"- Manifest: `{manifest['path']}`",
        f"- Phiên bản: `{manifest['version']}`",
        f"- SHA-256 manifest: `{manifest['sha256']}`",
        f"- Waveform đã băm lại: {checks['audited_files']:,}/{checks['expected_files']:,}",
        f"- Content hash duy nhất: {checks['unique_content_hashes']:,}",
        f"- Nhóm trùng toàn manifest: {checks['duplicate_groups_total']:,}",
        f"- Nhóm trùng trong cùng split: {checks['duplicate_groups_within_split']:,}",
        f"- Nhóm trùng đi qua train/development: {checks['duplicate_groups_crossing_splits']:,}",
        "",
        "## Phạm vi",
        "",
        f"- {report['scope']}",
        f"- {report['source_snapshot_status']}",
        "- Hash từng dòng được lưu trong báo cáo JSON đi kèm để tái kiểm.",
    ]
    if report["duplicate_groups"]:
        lines.extend(["", "## Nhóm trùng", ""])
        for group in report["duplicate_groups"]:
            lines.append(
                f"- `{group['waveform_sha256']}`: "
                + ", ".join(f"`{file_name}`" for file_name in group["files"])
            )
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    try:
        report = audit_development_manifest_content(
            args.manifest.resolve(),
            args.parquet_dir.resolve(),
        )
        write_reports(
            report,
            args.json_output.resolve(),
            args.markdown_output.resolve(),
        )
    except (OSError, csv.Error, ValueError, duckdb.Error) as error:
        print(f"Lỗi audit nội dung development manifest: {error}", file=sys.stderr)
        return 2

    print(
        f"{report['status']}: đã băm {report['checks']['audited_files']:,} waveform, "
        f"nhóm trùng={report['checks']['duplicate_groups_total']:,}."
    )
    print(f"JSON: {args.json_output.resolve()}")
    print(f"Markdown: {args.markdown_output.resolve()}")
    return 0 if report["technical_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
