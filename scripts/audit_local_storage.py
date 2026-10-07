#!/usr/bin/env python3
"""Kiểm kê dung lượng và coverage của các shard VSASV Parquet cục bộ."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "local_storage_audit.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "local_storage_audit.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--expected-shards", type=int, default=432)
    parser.add_argument(
        "--target-samples", type=int, nargs="+", default=[20_000, 40_000, 220_963]
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


def _validate_inputs(
    parquet_dir: Path,
    metadata_path: Path,
    expected_shards: int,
    target_samples: list[int],
) -> list[Path]:
    if expected_shards < 1:
        raise ValueError("expected-shards phải lớn hơn hoặc bằng 1.")
    if not target_samples or any(value < 1 for value in target_samples):
        raise ValueError("Mỗi target-samples phải lớn hơn hoặc bằng 1.")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metadata: {metadata_path}")
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy shard Parquet trong {parquet_dir}")
    return parquet_files


def build_inventory(
    parquet_dir: Path,
    metadata_path: Path,
    *,
    expected_shards: int = 432,
    target_samples: list[int] | None = None,
) -> dict[str, Any]:
    targets = (
        [20_000, 40_000, 220_963]
        if target_samples is None
        else target_samples
    )
    parquet_files = _validate_inputs(
        parquet_dir, metadata_path, expected_shards, targets
    )
    parquet_glob = (parquet_dir.resolve() / "*.parquet").as_posix()
    metadata_csv = metadata_path.resolve().as_posix()

    connection = duckdb.connect(database=":memory:")
    try:
        shard_rows = connection.execute(
            """
            WITH parquet_rows AS (
                SELECT filename, file, label, utt_type
                FROM read_parquet(?, filename = true)
            ), metadata_rows AS (
                SELECT * FROM read_csv(
                    ?, header = true,
                    columns = {
                        'file': 'VARCHAR',
                        'label': 'VARCHAR',
                        'utt_type': 'VARCHAR'
                    }
                )
            )
            SELECT
                filename,
                COUNT(*)::BIGINT AS rows,
                COUNT(DISTINCT parquet_rows.file)::BIGINT AS unique_files,
                SUM(
                    metadata_rows.file IS NOT NULL
                    AND parquet_rows.label = metadata_rows.label
                    AND parquet_rows.utt_type = metadata_rows.utt_type
                )::BIGINT AS metadata_exact_matches
            FROM parquet_rows
            LEFT JOIN metadata_rows USING (file)
            GROUP BY filename
            ORDER BY filename
            """,
            [parquet_glob, metadata_csv],
        ).fetchall()
        type_rows = connection.execute(
            """
            SELECT filename, utt_type, COUNT(*)::BIGINT
            FROM read_parquet(?, filename = true)
            GROUP BY filename, utt_type
            ORDER BY filename, utt_type
            """,
            [parquet_glob],
        ).fetchall()
    finally:
        connection.close()

    type_counts_by_shard: dict[str, dict[str, int]] = {}
    for filename, utt_type, count in type_rows:
        shard_name = Path(filename).name
        type_counts_by_shard.setdefault(shard_name, {})[str(utt_type)] = int(count)

    file_by_name = {path.name: path for path in parquet_files}
    shards = []
    for filename, rows, unique_files, exact_matches in shard_rows:
        shard_name = Path(filename).name
        shard_path = file_by_name.get(shard_name)
        if shard_path is None:
            raise RuntimeError(f"Không ánh xạ được thống kê cho shard {shard_name}.")
        size_bytes = shard_path.stat().st_size
        shards.append(
            {
                "shard": shard_name,
                "size_bytes": size_bytes,
                "size_mib": size_bytes / (1024**2),
                "samples": int(rows),
                "unique_files": int(unique_files),
                "bytes_per_sample": size_bytes / int(rows),
                "metadata_exact_matches": int(exact_matches),
                "metadata_coverage_percent": int(exact_matches) / int(rows) * 100.0,
                "utt_type_counts": type_counts_by_shard.get(shard_name, {}),
            }
        )

    total_bytes = sum(item["size_bytes"] for item in shards)
    total_samples = sum(item["samples"] for item in shards)
    total_unique_files = sum(item["unique_files"] for item in shards)
    total_exact_matches = sum(item["metadata_exact_matches"] for item in shards)
    bytes_per_sample = total_bytes / total_samples
    disk = shutil.disk_usage(parquet_dir.resolve())

    hard_issues = []
    if total_samples != total_unique_files:
        hard_issues.append("Có file logic bị lặp trong các shard cục bộ.")
    if total_exact_matches != total_samples:
        hard_issues.append("Có hàng Parquet không khớp chính xác metadata.")

    projections = [
        {
            "target_samples": count,
            "estimated_bytes": bytes_per_sample * count,
            "estimated_gib": bytes_per_sample * count / (1024**3),
        }
        for count in targets
    ]

    warnings = [
        f"{len(shards)} shard hiện có không phải mẫu ngẫu nhiên của 432 shard; "
        "dự báo dung lượng theo byte/mẫu chỉ dùng để lập kế hoạch.",
        "Dung lượng dự báo không bao gồm cache, file trung gian, checkpoint hoặc khoảng trống an toàn của hệ điều hành.",
    ]
    if len(shards) < expected_shards:
        warnings.append(
            f"Chỉ có {len(shards)}/{expected_shards} shard, chưa đủ để mô tả toàn bộ snapshot."
        )

    return {
        "status": "ĐẠT" if not hard_issues else "KHÔNG ĐẠT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local_storage_planning_not_scientific_result",
        "dataset_name": "VSASV-HF-public-snapshot-v1",
        "parquet_directory": display_path(parquet_dir),
        "metadata_path": display_path(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "expected_shards": expected_shards,
        "local_shards": len(shards),
        "shard_coverage_percent": len(shards) / expected_shards * 100.0,
        "totals": {
            "size_bytes": total_bytes,
            "size_mib": total_bytes / (1024**2),
            "size_gib": total_bytes / (1024**3),
            "samples": total_samples,
            "unique_files": total_unique_files,
            "bytes_per_sample": bytes_per_sample,
            "metadata_exact_matches": total_exact_matches,
            "metadata_coverage_percent": total_exact_matches / total_samples * 100.0,
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_gib": disk.free / (1024**3),
        },
        "shards": shards,
        "linear_sample_projections": projections,
        "hard_issues": hard_issues,
        "warnings": warnings,
    }


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    disk = report["disk"]
    lines = [
        "# Kiểm kê dung lượng các shard VSASV cục bộ",
        "",
        f"- Trạng thái kỹ thuật: **{report['status']}**",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Shard cục bộ: **{report['local_shards']}/{report['expected_shards']}** ({report['shard_coverage_percent']:.2f}%)",
        f"- Tổng mẫu: **{totals['samples']:,}**",
        f"- Tổng dung lượng: **{totals['size_bytes']:,} byte** ({totals['size_mib']:.2f} MiB)",
        f"- Trung bình: **{totals['bytes_per_sample']:,.2f} byte/mẫu**",
        f"- Khớp metadata: **{totals['metadata_exact_matches']:,}/{totals['samples']:,}** ({totals['metadata_coverage_percent']:.2f}%)",
        f"- Dung lượng đĩa còn trống lúc đo: **{disk['free_gib']:.2f} GiB**",
        "",
        "## Theo shard",
        "",
        "| Shard | Mẫu | Dung lượng | Byte/mẫu | Khớp metadata | Phân bố `utt_type` |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for shard in report["shards"]:
        distribution = ", ".join(
            f"{name}={count}" for name, count in shard["utt_type_counts"].items()
        )
        lines.append(
            f"| `{shard['shard']}` | {shard['samples']:,} | "
            f"{shard['size_mib']:.2f} MiB | {shard['bytes_per_sample']:,.2f} | "
            f"{shard['metadata_exact_matches']:,}/{shard['samples']:,} | {distribution} |"
        )

    lines.extend(
        [
            "",
            "## Dự báo tuyến tính theo số mẫu",
            "",
            "| Mục tiêu | Dung lượng dự báo |",
            "|---:|---:|",
        ]
    )
    for projection in report["linear_sample_projections"]:
        lines.append(
            f"| {projection['target_samples']:,} mẫu | "
            f"{projection['estimated_gib']:.2f} GiB "
            f"({projection['estimated_bytes'] / 1_000_000_000:.2f} GB) |"
        )

    lines.extend(["", "## Giới hạn", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    if report["hard_issues"]:
        lines.extend(["", "## Lỗi", ""])
        lines.extend(f"- {issue}" for issue in report["hard_issues"])
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    report = build_inventory(
        args.parquet_dir.resolve(),
        args.metadata.resolve(),
        expected_shards=args.expected_shards,
        target_samples=args.target_samples,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"{report['status']}: {report['local_shards']} shard, "
        f"{report['totals']['samples']} mẫu, "
        f"{report['totals']['size_mib']:.2f} MiB."
    )
    if report["hard_issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
