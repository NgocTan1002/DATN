#!/usr/bin/env python3
"""Project B0 train/evaluation time from measured local throughput."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimization-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_pipeline_optimization.json",
    )
    parser.add_argument(
        "--smoke-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "b0_smoke_training.json",
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "split_summary.json",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "runtime_projection.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "runtime_projection.md",
    )
    parser.add_argument("--planning-buffer-percent", type=float, default=20.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def duration(samples: int, throughput: float, buffer_percent: float) -> dict[str, Any]:
    seconds = samples / throughput
    buffered_seconds = seconds * (1.0 + buffer_percent / 100.0)
    return {
        "samples": samples,
        "seconds": seconds,
        "hours": seconds / 3600.0,
        "days_24h": seconds / 86400.0,
        "buffered_seconds": buffered_seconds,
        "buffered_hours": buffered_seconds / 3600.0,
    }


def find_variant(report: dict[str, Any], name: str) -> dict[str, Any]:
    for variant in report["variants"]:
        if variant["name"] == name:
            return variant
    raise ValueError(f"Không tìm thấy cấu hình {name!r} trong báo cáo tối ưu.")


def build_projection(args: argparse.Namespace) -> dict[str, Any]:
    optimization = load_json(args.optimization_report)
    smoke = load_json(args.smoke_report)
    split_report = load_json(args.split_report)

    baseline_name = optimization["decision"]["baseline"]
    baseline = find_variant(optimization, baseline_name)
    train_throughput = float(baseline["throughput_samples_per_second"])

    dev_examples = int(smoke["development"]["examples"])
    dev_seconds = float(smoke["resources"]["development_seconds"])
    evaluation_throughput = dev_examples / dev_seconds

    closed_train = int(split_report["splits"]["closed_train"]["samples"])
    closed_dev = int(split_report["splits"]["closed_dev"]["samples"])
    closed_test = int(split_report["splits"]["closed_test"]["samples"])
    train_ratio = closed_train / (closed_train + closed_dev)

    standalone = {}
    development_subsets = {}
    for total_samples in (20_000, 40_000):
        standalone[str(total_samples)] = {
            "train_only": duration(
                total_samples, train_throughput, args.planning_buffer_percent
            ),
            "evaluation_only": duration(
                total_samples, evaluation_throughput, args.planning_buffer_percent
            ),
        }
        subset_train = round(total_samples * train_ratio)
        subset_dev = total_samples - subset_train
        train_part = duration(
            subset_train, train_throughput, args.planning_buffer_percent
        )
        dev_part = duration(
            subset_dev, evaluation_throughput, args.planning_buffer_percent
        )
        total_seconds = train_part["seconds"] + dev_part["seconds"]
        development_subsets[str(total_samples)] = {
            "train_samples": subset_train,
            "dev_samples": subset_dev,
            "train": train_part,
            "dev": dev_part,
            "epoch_plus_dev_seconds": total_seconds,
            "epoch_plus_dev_hours": total_seconds / 3600.0,
            "epoch_plus_dev_buffered_hours": (
                total_seconds * (1.0 + args.planning_buffer_percent / 100.0) / 3600.0
            ),
        }

    full_train = duration(closed_train, train_throughput, args.planning_buffer_percent)
    full_dev = duration(closed_dev, evaluation_throughput, args.planning_buffer_percent)
    full_test = duration(closed_test, evaluation_throughput, args.planning_buffer_percent)
    full_seconds = full_train["seconds"] + full_dev["seconds"] + full_test["seconds"]

    return {
        "status": "ĐẠT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "planning_projection_not_scientific_result",
        "sources": {
            "optimization_report": str(args.optimization_report.resolve()),
            "smoke_report": str(args.smoke_report.resolve()),
            "split_report": str(args.split_report.resolve()),
        },
        "measured_throughput": {
            "training_samples_per_second": train_throughput,
            "training_source_variant": baseline_name,
            "evaluation_samples_per_second": evaluation_throughput,
            "evaluation_source_examples": dev_examples,
            "evaluation_source_seconds": dev_seconds,
        },
        "planning_buffer_percent": args.planning_buffer_percent,
        "standalone_projection": standalone,
        "development_subset_projection": {
            "split_basis": "closed_train:closed_dev",
            "train_ratio": train_ratio,
            "dev_ratio": 1.0 - train_ratio,
            "targets": development_subsets,
        },
        "full_closed_protocol_projection": {
            "train": full_train,
            "dev": full_dev,
            "test": full_test,
            "one_epoch_dev_test_seconds": full_seconds,
            "one_epoch_dev_test_hours": full_seconds / 3600.0,
            "one_epoch_dev_test_days_24h": full_seconds / 86400.0,
            "one_epoch_dev_test_buffered_hours": (
                full_seconds * (1.0 + args.planning_buffer_percent / 100.0) / 3600.0
            ),
        },
        "recommended_dataloader": {
            "batch_size": int(baseline["batch_size"]),
            "num_workers": int(baseline["num_workers"]),
            "status": optimization["decision"]["status"],
            "reason": (
                "Cấu hình 2 worker nhanh hơn 14,42% nhưng chưa vượt ngưỡng 15% "
                "và peak RSS cây tiến trình tăng từ 530,64 MiB lên 1.831,23 MiB."
            ),
            "revisit_when": (
                "Đo lại sau khi thay cách lưu/đọc Parquet hoặc bổ sung cache; "
                "không suy rộng quyết định này sang máy khác."
            ),
        },
        "assumptions": [
            "Thời gian tăng tuyến tính theo số mẫu và dùng cùng máy, đường đọc Parquet, LFCC và LCNN như benchmark ngày 02/10/2026.",
            "Train dùng throughput end-to-end gồm DataLoader, LFCC, forward, loss, backward và optimizer.",
            "Dev/test dùng throughput development của pilot B0; số đo này chỉ có một lần chạy 128 mẫu nên kém chắc chắn hơn benchmark train.",
            "Ước lượng chưa gồm tải dữ liệu, tạo manifest, kiểm tra leakage, lưu checkpoint hoặc thời gian gián đoạn.",
            "Biên dự phòng là ngân sách lập kế hoạch, không phải khoảng tin cậy thống kê.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    measured = report["measured_throughput"]
    targets = report["development_subset_projection"]["targets"]
    full = report["full_closed_protocol_projection"]
    standalone = report["standalone_projection"]
    loader = report["recommended_dataloader"]
    buffer_percent = report["planning_buffer_percent"]

    lines = [
        "# Dự báo thời gian chạy B0",
        "",
        "## Số đo đầu vào",
        "",
        f"- Train end-to-end: **{measured['training_samples_per_second']:.3f} mẫu/giây** từ `{measured['training_source_variant']}`.",
        f"- Đánh giá: **{measured['evaluation_samples_per_second']:.3f} mẫu/giây** từ {measured['evaluation_source_examples']} mẫu trong {measured['evaluation_source_seconds']:.3f} giây.",
        f"- Biên dự phòng kế hoạch: **+{buffer_percent:.0f}%**.",
        "",
        "## Nếu toàn bộ 20.000 hoặc 40.000 mẫu chạy cùng một chế độ",
        "",
        "| Số mẫu | Chỉ train | Train + dự phòng | Chỉ đánh giá | Đánh giá + dự phòng |",
        "|---:|---:|---:|---:|---:|",
    ]
    for sample_count in (20_000, 40_000):
        row = standalone[str(sample_count)]
        lines.append(
            f"| {sample_count:,} | {row['train_only']['hours']:.2f} giờ | "
            f"{row['train_only']['buffered_hours']:.2f} giờ | "
            f"{row['evaluation_only']['hours']:.2f} giờ | "
            f"{row['evaluation_only']['buffered_hours']:.2f} giờ |"
        )

    train_ratio = report["development_subset_projection"]["train_ratio"] * 100.0
    dev_ratio = report["development_subset_projection"]["dev_ratio"] * 100.0
    lines.extend(
        [
            "",
            "## Ngân sách development subset đề xuất",
            "",
            f"Phân bổ theo tỷ lệ closed train/dev hiện tại: **{train_ratio:.2f}% train / {dev_ratio:.2f}% dev**.",
            "",
            "| Quy mô | Train | Dev | 1 epoch train | 1 lượt dev | Tổng | Tổng + dự phòng |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample_count in (20_000, 40_000):
        row = targets[str(sample_count)]
        lines.append(
            f"| {sample_count:,} | {row['train_samples']:,} | {row['dev_samples']:,} | "
            f"{row['train']['hours']:.2f} giờ | {row['dev']['hours']:.2f} giờ | "
            f"{row['epoch_plus_dev_hours']:.2f} giờ | "
            f"{row['epoch_plus_dev_buffered_hours']:.2f} giờ |"
        )

    lines.extend(
        [
            "",
            "## Toàn bộ closed protocol",
            "",
            "| Giai đoạn | Số mẫu | Ước lượng | Có dự phòng |",
            "|---|---:|---:|---:|",
            f"| 1 epoch closed train | {full['train']['samples']:,} | {full['train']['hours']:.2f} giờ | {full['train']['buffered_hours']:.2f} giờ |",
            f"| 1 lượt closed dev | {full['dev']['samples']:,} | {full['dev']['hours']:.2f} giờ | {full['dev']['buffered_hours']:.2f} giờ |",
            f"| 1 lượt closed test | {full['test']['samples']:,} | {full['test']['hours']:.2f} giờ | {full['test']['buffered_hours']:.2f} giờ |",
            f"| Tổng | {full['train']['samples'] + full['dev']['samples'] + full['test']['samples']:,} | {full['one_epoch_dev_test_hours']:.2f} giờ | {full['one_epoch_dev_test_buffered_hours']:.2f} giờ |",
            "",
            "## Cấu hình DataLoader mặc định",
            "",
            f"Giữ `batch_size={loader['batch_size']}`, `num_workers={loader['num_workers']}` cho chặng B0 kế tiếp. {loader['reason']} {loader['revisit_when']}",
            "",
            "## Giả định và giới hạn",
            "",
        ]
    )
    lines.extend(f"- {assumption}" for assumption in report["assumptions"])
    lines.extend(
        [
            "",
            "Các con số trên dùng để lập lịch tài nguyên. Chúng không phải kết quả khoa học và không dự báo chắc chắn tốc độ trên đủ 432 shard.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_projection(args)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.report_md)
    full_hours = report["full_closed_protocol_projection"][
        "one_epoch_dev_test_hours"
    ]
    print(f"Runtime projection created: full epoch + dev + test = {full_hours:.2f} hours.")


if __name__ == "__main__":
    main()
