#!/usr/bin/env python3
"""Run a reproducible smoke test over local VSASV Parquet audio shards."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "audio_smoke_test.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "audio_smoke_test.md"
EXPECTED_COLUMNS = {
    "file": "VARCHAR",
    "audio": 'STRUCT("array" DOUBLE[], sampling_rate BIGINT)',
    "label": "VARCHAR",
    "utt_type": "VARCHAR",
}
TARGET_SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra audio trong các shard Parquet VSASV cục bộ."
    )
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--samples-per-shard", type=int, default=3)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def summarize_waveform(values: Iterable[float], sampling_rate: int) -> dict[str, Any]:
    count = 0
    finite_count = 0
    nonfinite_count = 0
    squared_sum = 0.0
    value_sum = 0.0
    peak = 0.0
    clipped = 0
    minimum = math.inf
    maximum = -math.inf
    for raw_value in values:
        count += 1
        value = float(raw_value)
        if not math.isfinite(value):
            nonfinite_count += 1
            continue
        finite_count += 1
        value_sum += value
        squared_sum += value * value
        absolute = abs(value)
        peak = max(peak, absolute)
        if absolute >= 0.999:
            clipped += 1
        minimum = min(minimum, value)
        maximum = max(maximum, value)

    rms = math.sqrt(squared_sum / finite_count) if finite_count else None
    dc_offset = value_sum / finite_count if finite_count else None
    return {
        "samples": count,
        "duration_seconds": count / sampling_rate if sampling_rate > 0 else None,
        "finite_samples": finite_count,
        "nonfinite_samples": nonfinite_count,
        "minimum": minimum if finite_count else None,
        "maximum": maximum if finite_count else None,
        "peak_absolute": peak if finite_count else None,
        "rms": rms,
        "dc_offset": dc_offset,
        "clipped_fraction": clipped / finite_count if finite_count else None,
        "silent": rms is None or rms < 1e-4,
    }


def run_smoke_test(parquet_dir: Path, samples_per_shard: int) -> dict[str, Any]:
    if samples_per_shard < 1:
        raise ValueError("samples-per-shard phải lớn hơn hoặc bằng 1.")
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy Parquet trong {parquet_dir}")

    try:
        import duckdb  # type: ignore[import-not-found]
    except (ImportError, OSError, AttributeError) as error:
        raise RuntimeError(
            "Thiếu DuckDB. Chạy: python -m pip install duckdb"
        ) from error

    parquet_glob = (parquet_dir / "*.parquet").resolve().as_posix()
    connection = duckdb.connect()
    schema_rows = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [parquet_glob]
    ).fetchall()
    schema = {row[0]: row[1] for row in schema_rows}

    overall = connection.execute(
        """
        SELECT
            COUNT(*)::BIGINT,
            COUNT(DISTINCT file)::BIGINT,
            COUNT(DISTINCT label)::BIGINT,
            SUM(file IS NULL OR file = '')::BIGINT,
            SUM(label IS NULL OR label = '')::BIGINT,
            SUM(utt_type IS NULL OR utt_type = '')::BIGINT,
            SUM(audio.array IS NULL)::BIGINT,
            SUM(audio.sampling_rate IS NULL OR audio.sampling_rate <= 0)::BIGINT,
            SUM(array_length(audio.array) = 0)::BIGINT
        FROM read_parquet(?)
        """,
        [parquet_glob],
    ).fetchone()

    distribution = [
        {
            "utt_type": row[0],
            "sampling_rate": row[1],
            "utterances": row[2],
        }
        for row in connection.execute(
            """
            SELECT utt_type, audio.sampling_rate, COUNT(*)::BIGINT
            FROM read_parquet(?)
            GROUP BY 1, 2 ORDER BY 1, 2
            """,
            [parquet_glob],
        ).fetchall()
    ]

    duration_by_type = [
        {
            "utt_type": row[0],
            "minimum_seconds": round(row[1], 6),
            "median_seconds": round(row[2], 6),
            "mean_seconds": round(row[3], 6),
            "p95_seconds": round(row[4], 6),
            "maximum_seconds": round(row[5], 6),
            "total_hours": round(row[6], 6),
        }
        for row in connection.execute(
            """
            WITH durations AS (
                SELECT utt_type,
                       array_length(audio.array) * 1.0 / audio.sampling_rate AS seconds
                FROM read_parquet(?)
            )
            SELECT utt_type, MIN(seconds), MEDIAN(seconds), AVG(seconds),
                   QUANTILE_CONT(seconds, 0.95), MAX(seconds), SUM(seconds) / 3600.0
            FROM durations GROUP BY 1 ORDER BY 1
            """,
            [parquet_glob],
        ).fetchall()
    ]

    shard_rows = [
        {
            "shard": Path(row[0]).name,
            "utterances": row[1],
            "total_hours": round(row[2], 6),
        }
        for row in connection.execute(
            """
            SELECT filename, COUNT(*)::BIGINT,
                   SUM(array_length(audio.array) * 1.0 / audio.sampling_rate) / 3600.0
            FROM read_parquet(?, filename = true)
            GROUP BY 1 ORDER BY 1
            """,
            [parquet_glob],
        ).fetchall()
    ]

    sampled_rows = connection.execute(
        """
        WITH ranked AS (
            SELECT filename, file, label, utt_type,
                   audio.sampling_rate AS sampling_rate,
                   audio.array AS waveform,
                   ROW_NUMBER() OVER (
                       PARTITION BY filename, utt_type ORDER BY hash(file)
                   ) AS rank
            FROM read_parquet(?, filename = true)
        )
        SELECT filename, file, label, utt_type, sampling_rate, waveform
        FROM ranked WHERE rank <= ? ORDER BY filename, rank
        """,
        [parquet_glob, samples_per_shard],
    ).fetchall()
    connection.close()

    sample_checks = []
    for filename, file_name, label, utt_type, sampling_rate, waveform in sampled_rows:
        statistics = summarize_waveform(waveform or [], int(sampling_rate or 0))
        sample_checks.append(
            {
                "shard": Path(filename).name,
                "file": file_name,
                "label": label,
                "utt_type": utt_type,
                "sampling_rate": sampling_rate,
                **statistics,
            }
        )

    hard_issues: list[str] = []
    warnings: list[str] = []
    if schema != EXPECTED_COLUMNS:
        hard_issues.append("Schema Parquet không khớp schema đã khóa.")
    if overall[0] != overall[1]:
        hard_issues.append("Có đường dẫn logic bị lặp trong năm shard.")
    invalid_counts = overall[3:9]
    if any(invalid_counts):
        hard_issues.append("Có trường bắt buộc, waveform hoặc sample rate không hợp lệ.")
    if any(item["nonfinite_samples"] for item in sample_checks):
        hard_issues.append("Waveform mẫu có NaN hoặc vô cực.")
    if any(item["silent"] for item in sample_checks):
        hard_issues.append("Waveform mẫu im lặng hoặc có RMS quá nhỏ.")

    rates_by_type: dict[str, set[int]] = defaultdict(set)
    for item in distribution:
        rates_by_type[item["utt_type"]].add(item["sampling_rate"])
    all_rates = sorted({rate for rates in rates_by_type.values() for rate in rates})
    shortcut_risk = len(all_rates) > 1
    if shortcut_risk:
        warnings.append(
            "Sample rate không đồng nhất giữa các loại; đây là shortcut risk. "
            f"Phải resample toàn bộ waveform về {TARGET_SAMPLE_RATE} Hz."
        )
    if len(parquet_files) < 432:
        warnings.append(
            "Năm shard chỉ là một phần nhỏ, không ngẫu nhiên của 432 shard; "
            "không suy rộng phân bố cục bộ cho toàn bộ snapshot."
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parquet_directory": display_path(parquet_dir),
        "technical_passed": not hard_issues,
        "safe_to_continue_pipeline": not hard_issues,
        "required_target_sample_rate": TARGET_SAMPLE_RATE,
        "hard_issues": hard_issues,
        "warnings": warnings,
        "schema": schema,
        "local_shards": len(parquet_files),
        "expected_shards": 432,
        "summary": {
            "utterances": overall[0],
            "unique_files": overall[1],
            "speakers": overall[2],
            "missing_file": overall[3],
            "missing_label": overall[4],
            "missing_utt_type": overall[5],
            "missing_waveform": overall[6],
            "invalid_sampling_rate": overall[7],
            "empty_waveform": overall[8],
        },
        "distribution": distribution,
        "duration_by_type": duration_by_type,
        "shards": shard_rows,
        "sampled_waveforms": sample_checks,
        "sample_rate_shortcut_risk": shortcut_risk,
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "ĐẠT" if report["technical_passed"] else "KHÔNG ĐẠT"
    summary = report["summary"]
    lines = [
        "# Báo cáo audio smoke test VSASV",
        "",
        f"- **Trạng thái kỹ thuật:** {status}",
        f"- **Shard cục bộ:** {report['local_shards']}/{report['expected_shards']}",
        f"- **Utterance:** {summary['utterances']:,}",
        f"- **Speaker trong năm shard:** {summary['speakers']:,}",
        f"- **Sample rate đích bắt buộc:** {report['required_target_sample_rate']:,} Hz",
        "",
        "## Kiểm tra toàn bộ năm shard",
        "",
        f"- Đường dẫn duy nhất: {summary['unique_files']:,}/{summary['utterances']:,}",
        f"- Thiếu file/label/utt_type: {summary['missing_file'] + summary['missing_label'] + summary['missing_utt_type']:,}",
        f"- Waveform thiếu hoặc rỗng: {summary['missing_waveform'] + summary['empty_waveform']:,}",
        f"- Sample rate không hợp lệ: {summary['invalid_sampling_rate']:,}",
        "",
        "## Phân bố sample rate",
        "",
        "| Loại | Sample rate | Số mẫu |",
        "|---|---:|---:|",
    ]
    for item in report["distribution"]:
        lines.append(
            f"| `{item['utt_type']}` | {item['sampling_rate']:,} Hz | {item['utterances']:,} |"
        )
    lines.extend(
        [
            "",
            "## Thời lượng",
            "",
            "| Loại | Min (s) | Median (s) | Mean (s) | P95 (s) | Max (s) | Tổng giờ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["duration_by_type"]:
        lines.append(
            f"| `{item['utt_type']}` | {item['minimum_seconds']:.2f} | "
            f"{item['median_seconds']:.2f} | {item['mean_seconds']:.2f} | "
            f"{item['p95_seconds']:.2f} | {item['maximum_seconds']:.2f} | "
            f"{item['total_hours']:.2f} |"
        )
    lines.extend(["", "## Kiểm tra waveform mẫu", ""])
    lines.append(
        f"Đã đọc và kiểm tra số học {len(report['sampled_waveforms'])} waveform "
        "được chọn xác định theo từng cặp shard/loại bằng hash(file)."
    )
    lines.extend(
        [
            "",
            "| File | Loại | Hz | Giây | Peak | RMS | Non-finite | Silent |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["sampled_waveforms"]:
        lines.append(
            f"| `{item['file']}` | `{item['utt_type']}` | {item['sampling_rate']:,} | "
            f"{item['duration_seconds']:.2f} | {item['peak_absolute']:.4f} | "
            f"{item['rms']:.4f} | {item['nonfinite_samples']:,} | "
            f"{'Có' if item['silent'] else 'Không'} |"
        )
    lines.extend(["", "## Cảnh báo", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- Không có.")
    lines.extend(
        [
            "",
            "## Quyết định",
            "",
            (
                "Có thể tiếp tục xây dựng pipeline và baseline. Mọi waveform phải được "
                "resample về mono 16 kHz trong pipeline; kết quả trên năm shard chỉ dùng "
                "cho smoke test, không đại diện toàn bộ snapshot."
                if report["safe_to_continue_pipeline"]
                else "Chưa được tiếp tục baseline cho đến khi xử lý hết hard issue."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    try:
        report = run_smoke_test(args.parquet_dir.resolve(), args.samples_per_shard)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lỗi audio smoke test: {error}", file=sys.stderr)
        return 2

    print(f"Audio smoke test {('ĐẠT' if report['technical_passed'] else 'KHÔNG ĐẠT')}.")
    print(f"JSON: {args.json_output.resolve()}")
    print(f"Markdown: {args.markdown_output.resolve()}")
    return 0 if report["technical_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
