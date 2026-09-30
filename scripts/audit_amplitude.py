#!/usr/bin/env python3
"""Audit locked amplitude policies over every locally available closed sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import apply_amplitude_policy, build_amplitude_summaries  # noqa: E402


DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "audio.json"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "amplitude_audit.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "amplitude_audit.md"
SPLIT_NAMES = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit peak/RMS/gain cho toàn bộ audio VSASV cục bộ."
    )
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--expected-samples", type=int, default=2_558)
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


def read_split_membership(
    split_dir: Path,
) -> tuple[dict[str, tuple[str, str, str]], dict[str, dict[str, Any]]]:
    membership: dict[str, tuple[str, str, str]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        path = split_dir / f"closed_{split_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy split: {path}")
        count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ("file", "label", "utt_type"):
                raise ValueError(f"Schema split không hợp lệ: {path}")
            for row in reader:
                logical_file = row["file"]
                if logical_file in membership:
                    raise ValueError(f"File xuất hiện ở nhiều closed split: {logical_file}")
                membership[logical_file] = (
                    split_name,
                    row["label"],
                    row["utt_type"],
                )
                count += 1
        sources[split_name] = {
            "path": display_path(path),
            "rows": count,
            "sha256": sha256_file(path),
        }
    return membership, sources


def load_audio_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    waveform = config.get("waveform", {})
    policy = waveform.get("amplitude_policy", {})
    required = {
        "candidates",
        "peak_target",
        "rms_target_dbfs",
        "minimum_input_rms_dbfs",
        "numeric_epsilon",
    }
    if not required.issubset(policy):
        raise ValueError(f"Cấu hình amplitude thiếu trường: {sorted(required - set(policy))}")
    return config


def run_audit(
    parquet_dir: Path,
    split_dir: Path,
    config_path: Path,
    expected_samples: int,
) -> dict[str, Any]:
    if expected_samples <= 0:
        raise ValueError("expected_samples phải lớn hơn 0.")
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy Parquet trong {parquet_dir}")

    membership, split_sources = read_split_membership(split_dir)
    config = load_audio_config(config_path)
    waveform_config = config["waveform"]
    policy_config = waveform_config["amplitude_policy"]
    target_sample_rate = int(waveform_config["target_sample_rate_hz"])
    policies = tuple(policy_config["candidates"])
    if policies != ("none", "peak", "rms_dbfs"):
        raise ValueError(f"Danh sách policy không khớp giao thức v3: {policies}")

    observations: list[dict[str, Any]] = []
    processed_files: set[str] = set()
    ignored_local_files = 0
    parquet_glob = (parquet_dir / "*.parquet").resolve().as_posix()
    connection = duckdb.connect(database=":memory:")
    try:
        cursor = connection.execute(
            """
            SELECT file, label, utt_type, audio.sampling_rate, audio.array
            FROM read_parquet(?)
            ORDER BY file
            """,
            [parquet_glob],
        )
        while True:
            rows = cursor.fetchmany(32)
            if not rows:
                break
            for logical_file, speaker_id, utt_type, native_rate, values in rows:
                split_metadata = membership.get(str(logical_file))
                if split_metadata is None:
                    ignored_local_files += 1
                    continue
                split_name, split_speaker, split_type = split_metadata
                if (str(speaker_id), str(utt_type)) != (split_speaker, split_type):
                    raise ValueError(
                        f"Metadata split/Parquet không khớp cho {logical_file}."
                    )
                if logical_file in processed_files:
                    raise ValueError(f"File Parquet bị lặp: {logical_file}")

                sample_rate = int(native_rate)
                waveform = torch.as_tensor(values, dtype=torch.float32)
                if waveform.ndim != 1 or waveform.numel() == 0:
                    raise ValueError(f"Waveform rỗng hoặc không mono: {logical_file}")
                if sample_rate <= 0:
                    raise ValueError(f"Sample rate không hợp lệ: {logical_file}")
                if not torch.isfinite(waveform).all():
                    raise ValueError(f"Waveform có NaN/Inf: {logical_file}")
                if sample_rate != target_sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sample_rate, target_sample_rate
                    )

                for policy in policies:
                    result = apply_amplitude_policy(
                        waveform,
                        policy,
                        peak_target=float(policy_config["peak_target"]),
                        rms_target_dbfs=float(policy_config["rms_target_dbfs"]),
                        minimum_input_rms_dbfs=float(
                            policy_config["minimum_input_rms_dbfs"]
                        ),
                        epsilon=float(policy_config["numeric_epsilon"]),
                    )
                    observations.append(
                        {
                            "split": split_name,
                            "binary_label": (
                                "bonafide" if split_type == "bonafide" else "spoof"
                            ),
                            "utt_type": split_type,
                            "native_sample_rate": sample_rate,
                            "policy": policy,
                            "input_peak": result.input_peak,
                            "output_peak": result.output_peak,
                            "input_rms_dbfs": result.input_rms_dbfs,
                            "output_rms_dbfs": result.output_rms_dbfs,
                            "applied_gain": result.applied_gain,
                            "near_silence": result.near_silence,
                            "peak_limited": result.peak_limited,
                        }
                    )
                processed_files.add(str(logical_file))
    finally:
        connection.close()

    hard_issues: list[str] = []
    if len(processed_files) != expected_samples:
        hard_issues.append(
            f"Số mẫu xử lý {len(processed_files):,} khác kỳ vọng {expected_samples:,}."
        )
    expected_observations = len(processed_files) * len(policies)
    if len(observations) != expected_observations:
        hard_issues.append(
            f"Số observation {len(observations):,} khác kỳ vọng {expected_observations:,}."
        )

    sample_distribution = Counter(
        (row["split"], row["binary_label"], row["utt_type"])
        for row in observations
        if row["policy"] == "none"
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "technical_passed": not hard_issues,
        "hard_issues": hard_issues,
        "warnings": [
            "Năm shard cục bộ không phải mẫu ngẫu nhiên của toàn bộ 432 shard.",
            "Audit chỉ mô tả shortcut risk; không được dùng closed test để chọn policy.",
        ],
        "protocol": {
            "audio_config": display_path(config_path),
            "audio_config_sha256": sha256_file(config_path),
            "version": config["version"],
            "target_sample_rate": target_sample_rate,
            "policies": list(policies),
            "measurement_scope": policy_config["measurement_scope"],
            "peak_target": policy_config["peak_target"],
            "rms_target_dbfs": policy_config["rms_target_dbfs"],
            "minimum_input_rms_dbfs": policy_config["minimum_input_rms_dbfs"],
        },
        "source": {
            "parquet_directory": display_path(parquet_dir),
            "local_shards": [path.name for path in parquet_files],
            "closed_splits": split_sources,
        },
        "summary": {
            "expected_samples": expected_samples,
            "processed_samples": len(processed_files),
            "policy_observations": len(observations),
            "ignored_local_files": ignored_local_files,
            "nonfinite_waveforms": 0,
            "metadata_mismatches": 0,
        },
        "sample_distribution": [
            {
                "split": key[0],
                "binary_label": key[1],
                "utt_type": key[2],
                "samples": count,
            }
            for key, count in sorted(sample_distribution.items())
        ],
        "aggregates": build_amplitude_summaries(observations),
    }


def _stat(row: dict[str, Any], field: str, statistic: str) -> float:
    return float(row["statistics"][field][statistic])


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    protocol = report["protocol"]
    lines = [
        "# Báo cáo amplitude audit VSASV",
        "",
        f"- **Trạng thái kỹ thuật:** {'ĐẠT' if report['technical_passed'] else 'KHÔNG ĐẠT'}",
        f"- **Mẫu đã xử lý:** {summary['processed_samples']:,}/{summary['expected_samples']:,}",
        f"- **Policy observation:** {summary['policy_observations']:,}",
        f"- **Giao thức âm thanh:** phiên bản {protocol['version']}",
        f"- **Phạm vi đo:** `{protocol['measurement_scope']}`",
        f"- **Sample rate sau resample:** {protocol['target_sample_rate']:,} Hz",
        "",
        "## Phân bố mẫu cục bộ",
        "",
        "| Split | Nhãn | Loại | Mẫu |",
        "|---|---|---|---:|",
    ]
    for row in report["sample_distribution"]:
        lines.append(
            f"| `{row['split']}` | `{row['binary_label']}` | "
            f"`{row['utt_type']}` | {row['samples']:,} |"
        )

    lines.extend(
        [
            "",
            "## Tổng hợp theo policy",
            "",
            "| Policy | Mẫu | Input peak median/P95 | Input RMS median/P95 (dBFS) | "
            "Gain median/P95 | Near-silence | Peak-limited |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["aggregates"]["by_policy"]:
        lines.append(
            f"| `{row['policy']}` | {row['samples']:,} | "
            f"{_stat(row, 'input_peak', 'median'):.4f} / {_stat(row, 'input_peak', 'p95'):.4f} | "
            f"{_stat(row, 'input_rms_dbfs', 'median'):.2f} / {_stat(row, 'input_rms_dbfs', 'p95'):.2f} | "
            f"{_stat(row, 'applied_gain', 'median'):.4f} / {_stat(row, 'applied_gain', 'p95'):.4f} | "
            f"{row['near_silence']:,} | {row['peak_limited']:,} |"
        )

    for title, key, dimensions in (
        ("Theo split", "by_split_and_policy", ("split", "policy")),
        ("Theo nhãn nhị phân", "by_label_and_policy", ("binary_label", "policy")),
        ("Theo utt_type", "by_utt_type_and_policy", ("utt_type", "policy")),
        (
            "Theo sample rate gốc",
            "by_native_sample_rate_and_policy",
            ("native_sample_rate", "policy"),
        ),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"| {' | '.join(dimensions)} | Mẫu | Input RMS median (dBFS) | "
                "Output RMS median (dBFS) | Gain median | Near-silence | Peak-limited |",
                f"|{'---|' * len(dimensions)}---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["aggregates"][key]:
            dimension_values = " | ".join(f"`{row[name]}`" for name in dimensions)
            lines.append(
                f"| {dimension_values} | {row['samples']:,} | "
                f"{_stat(row, 'input_rms_dbfs', 'median'):.2f} | "
                f"{_stat(row, 'output_rms_dbfs', 'median'):.2f} | "
                f"{_stat(row, 'applied_gain', 'median'):.4f} | "
                f"{row['near_silence']:,} | {row['peak_limited']:,} |"
            )

    lines.extend(["", "## Cảnh báo", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    if report["hard_issues"]:
        lines.extend(["", "## Lỗi chặn", ""])
        lines.extend(f"- {issue}" for issue in report["hard_issues"])
    lines.extend(
        [
            "",
            "## Kết luận sử dụng",
            "",
            "Báo cáo này dùng để phát hiện chênh lệch biên độ và kiểm tra việc cài đặt. "
            "Policy chiến thắng vẫn phải được chọn bằng development EER với B0; không "
            "được chọn từ closed test hoặc từ thống kê mô tả này.",
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
        report = run_audit(
            args.parquet_dir.resolve(),
            args.split_dir.resolve(),
            args.config.resolve(),
            args.expected_samples,
        )
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lỗi amplitude audit: {error}", file=sys.stderr)
        return 2

    print(
        f"Amplitude audit {'ĐẠT' if report['technical_passed'] else 'KHÔNG ĐẠT'}: "
        f"{report['summary']['processed_samples']:,} mẫu."
    )
    print(f"JSON: {args.json_output.resolve()}")
    print(f"Markdown: {args.markdown_output.resolve()}")
    return 0 if report["technical_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
