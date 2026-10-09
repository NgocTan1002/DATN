#!/usr/bin/env python3
"""Run a reproducible smoke test over local VSASV Parquet audio shards."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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


def configure_duckdb(connection: Any) -> None:
    """Bound DuckDB memory use while scanning waveform-heavy Parquet shards."""

    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '8GB'")


def percentile_cont(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated continuous percentile."""

    if not values:
        raise ValueError("Không thể tính phân vị từ danh sách rỗng.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile phải thuộc khoảng [0, 1].")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


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

    connection = duckdb.connect()
    configure_duckdb(connection)
    schema: dict[str, str] = {}
    schema_mismatches: list[str] = []
    total_rows = 0
    unique_files: set[str] = set()
    speakers: set[str] = set()
    missing_counts = Counter[str]()
    distribution_counts: Counter[tuple[str, int]] = Counter()
    durations_by_type: dict[str, list[float]] = defaultdict(list)
    shard_rows: list[dict[str, Any]] = []
    sample_checks: list[dict[str, Any]] = []

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
            if current_schema != EXPECTED_COLUMNS:
                schema_mismatches.append(parquet_path.name)

            scalar_rows = connection.execute(
                """
                SELECT file, label, utt_type, audio.sampling_rate,
                       audio.array IS NULL, array_length(audio.array)
                FROM read_parquet(?)
                """,
                [parquet_file],
            ).fetchall()
            total_rows += len(scalar_rows)
            shard_seconds = 0.0
            for (
                file_name,
                label,
                utt_type,
                sampling_rate,
                missing_waveform,
                sample_count,
            ) in scalar_rows:
                if file_name is None or file_name == "":
                    missing_counts["file"] += 1
                else:
                    unique_files.add(str(file_name))
                if label is None or label == "":
                    missing_counts["label"] += 1
                else:
                    speakers.add(str(label))
                if utt_type is None or utt_type == "":
                    missing_counts["utt_type"] += 1
                if missing_waveform:
                    missing_counts["waveform"] += 1
                if sample_count == 0:
                    missing_counts["empty_waveform"] += 1
                if sampling_rate is None or sampling_rate <= 0:
                    missing_counts["sampling_rate"] += 1
                    continue
                distribution_counts[(str(utt_type), int(sampling_rate))] += 1
                if sample_count is not None:
                    seconds = sample_count / sampling_rate
                    durations_by_type[str(utt_type)].append(seconds)
                    shard_seconds += seconds

            shard_rows.append(
                {
                    "shard": parquet_path.name,
                    "utterances": len(scalar_rows),
                    "total_hours": round(shard_seconds / 3600.0, 6),
                }
            )

            sampled_rows = connection.execute(
                """
                WITH ranked AS (
                    SELECT file, label, utt_type,
                           audio.sampling_rate AS sampling_rate,
                           audio.array AS waveform,
                           ROW_NUMBER() OVER (
                               PARTITION BY utt_type ORDER BY hash(file)
                           ) AS rank
                    FROM read_parquet(?)
                )
                SELECT file, label, utt_type, sampling_rate, waveform
                FROM ranked WHERE rank <= ? ORDER BY rank, utt_type
                """,
                [parquet_file, samples_per_shard],
            ).fetchall()
            for file_name, label, utt_type, sampling_rate, waveform in sampled_rows:
                waveform_statistics = summarize_waveform(
                    waveform or [], int(sampling_rate or 0)
                )
                sample_checks.append(
                    {
                        "shard": parquet_path.name,
                        "file": file_name,
                        "label": label,
                        "utt_type": utt_type,
                        "sampling_rate": sampling_rate,
                        **waveform_statistics,
                    }
                )
    finally:
        connection.close()

    overall = (
        total_rows,
        len(unique_files),
        len(speakers),
        missing_counts["file"],
        missing_counts["label"],
        missing_counts["utt_type"],
        missing_counts["waveform"],
        missing_counts["sampling_rate"],
        missing_counts["empty_waveform"],
    )
    distribution = [
        {
            "utt_type": utt_type,
            "sampling_rate": sampling_rate,
            "utterances": count,
        }
        for (utt_type, sampling_rate), count in sorted(distribution_counts.items())
    ]
    duration_by_type = []
    for utt_type, values in sorted(durations_by_type.items()):
        duration_by_type.append(
            {
                "utt_type": utt_type,
                "minimum_seconds": round(min(values), 6),
                "median_seconds": round(percentile_cont(values, 0.5), 6),
                "mean_seconds": round(sum(values) / len(values), 6),
                "p95_seconds": round(percentile_cont(values, 0.95), 6),
                "maximum_seconds": round(max(values), 6),
                "total_hours": round(sum(values) / 3600.0, 6),
            }
        )

    hard_issues: list[str] = []
    warnings: list[str] = []
    if schema_mismatches:
        hard_issues.append(
            "Schema Parquet không khớp schema đã khóa trong "
            f"{len(schema_mismatches)} shard."
        )
    shard_count = len(parquet_files)
    if overall[0] != overall[1]:
        hard_issues.append(
            f"Có đường dẫn logic bị lặp trong {shard_count} shard cục bộ."
        )
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
            f"{shard_count} shard chỉ là một phần nhỏ, không ngẫu nhiên của 432 shard; "
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
    local_shards = report["local_shards"]
    lines = [
        "# Báo cáo audio smoke test VSASV",
        "",
        f"- **Trạng thái kỹ thuật:** {status}",
        f"- **Shard cục bộ:** {report['local_shards']}/{report['expected_shards']}",
        f"- **Utterance:** {summary['utterances']:,}",
        f"- **Speaker trong {local_shards} shard:** {summary['speakers']:,}",
        f"- **Sample rate đích bắt buộc:** {report['required_target_sample_rate']:,} Hz",
        "",
        f"## Kiểm tra toàn bộ {local_shards} shard",
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
                f"resample về mono 16 kHz trong pipeline; kết quả trên {local_shards} "
                "shard chỉ dùng cho smoke test, không đại diện toàn bộ snapshot."
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
