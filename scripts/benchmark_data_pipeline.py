#!/usr/bin/env python3
"""Benchmark each stage of the reproducible VSASV LFCC + LCNN pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import VSASVParquetDataset, apply_amplitude_policy  # noqa: E402
from src.models import (  # noqa: E402
    build_lcnn_from_config,
    build_lfcc_transform,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lfcc_lcnn.json",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "smoke_train.csv",
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_pipeline_benchmark.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_pipeline_benchmark.md",
    )
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Không thể tính percentile từ danh sách rỗng.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_seconds(values: list[float]) -> dict[str, Any]:
    if not values or any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("Danh sách thời gian phải gồm các số hữu hạn không âm.")
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_ms": statistics.fmean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "p95_ms": percentile(values, 0.95) * 1000.0,
        "minimum_ms": min(values) * 1000.0,
        "maximum_ms": max(values) * 1000.0,
        "values_seconds": values,
    }


def waveform_sha256(waveform: torch.Tensor) -> str:
    array = waveform.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def rss_mib(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024**2)


def record_rss(process: psutil.Process, observed: list[float]) -> None:
    observed.append(rss_mib(process))


def write_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    direct = report["direct_stage_timing"]
    pipeline = report["batch_pipeline_timing"]
    bottleneck = report["bottleneck"]
    resources = report["resources"]
    initialization = report["dataset_initialization"]
    markdown = f"""# Benchmark pipeline dữ liệu B0

## Kết quả

- Trạng thái: **{report['status']}**
- Thời điểm: `{report['created_at']}`
- Thiết bị: `{report['environment']['device']}`
- Seed: `{report['protocol']['seed']}`
- Amplitude policy: `{report['protocol']['amplitude_policy']}`
- Số mẫu cố định: {report['protocol']['samples']}
- Số lần lặp: {report['protocol']['repeats']}
- Batch size: {report['protocol']['batch_size']}
- DataLoader workers: {report['protocol']['num_workers']}
- Kiểm tra waveform/file/label tái lập: **{report['reproducibility']['status']}**

## Khởi tạo Dataset

| Chỉ số | Giá trị |
|---|---:|
| Lần đầu | {initialization['cold_start_ms']:.3f} ms |
| Trung vị các lần sau | {initialization['warm_median_ms']:.3f} ms |
| P95 toàn bộ | {initialization['summary']['p95_ms']:.3f} ms |
| Mẫu local/split | {report['dataset']['local_examples']}/{report['dataset']['split_examples']} |

## Đo trực tiếp từng giai đoạn trên một mẫu

| Giai đoạn | Trung vị | P95 | Tỷ lệ trong tiền xử lý |
|---|---:|---:|---:|
| Đọc/decode Parquet | {direct['read_decode']['median_ms']:.3f} ms | {direct['read_decode']['p95_ms']:.3f} ms | {direct['shares_percent']['read_decode']:.2f}% |
| Resample | {direct['resample']['median_ms']:.3f} ms | {direct['resample']['p95_ms']:.3f} ms | {direct['shares_percent']['resample']:.2f}% |
| Amplitude policy | {direct['amplitude']['median_ms']:.3f} ms | {direct['amplitude']['p95_ms']:.3f} ms | {direct['shares_percent']['amplitude']:.2f}% |
| Cắt/lặp 64.000 mẫu | {direct['segmentation']['median_ms']:.3f} ms | {direct['segmentation']['p95_ms']:.3f} ms | {direct['shares_percent']['segmentation']:.2f}% |

## Đường train theo batch

| Giai đoạn | Trung vị/batch | P95/batch | Tỷ lệ đường train |
|---|---:|---:|---:|
| DataLoader | {pipeline['data_loader_batch']['median_ms']:.3f} ms | {pipeline['data_loader_batch']['p95_ms']:.3f} ms | {pipeline['shares_percent']['data_loader']:.2f}% |
| LFCC | {pipeline['lfcc']['median_ms']:.3f} ms | {pipeline['lfcc']['p95_ms']:.3f} ms | {pipeline['shares_percent']['lfcc']:.2f}% |
| LCNN forward | {pipeline['lcnn_forward_train']['median_ms']:.3f} ms | {pipeline['lcnn_forward_train']['p95_ms']:.3f} ms | {pipeline['shares_percent']['lcnn_forward']:.2f}% |
| Loss | {pipeline['loss']['median_ms']:.3f} ms | {pipeline['loss']['p95_ms']:.3f} ms | {pipeline['shares_percent']['loss']:.2f}% |
| Backward | {pipeline['backward']['median_ms']:.3f} ms | {pipeline['backward']['p95_ms']:.3f} ms | {pipeline['shares_percent']['backward']:.2f}% |
| Optimizer | {pipeline['optimizer']['median_ms']:.3f} ms | {pipeline['optimizer']['p95_ms']:.3f} ms | {pipeline['shares_percent']['optimizer']:.2f}% |

DataLoader warm throughput: **{pipeline['data_loader_throughput_samples_per_second']:.3f} mẫu/giây**.

## Điểm nghẽn

Giai đoạn lớn nhất theo trung vị đường train là **{bottleneck['stage']}**, chiếm khoảng **{bottleneck['share_percent']:.2f}%** tổng thời gian các giai đoạn được đo. Trong phần đọc và tiền xử lý một mẫu, giai đoạn lớn nhất là **{bottleneck['direct_stage']}**.

Kết luận này chỉ áp dụng cho cấu hình CPU, {report['protocol']['samples']} mẫu smoke, batch size {report['protocol']['batch_size']} và `num_workers={report['protocol']['num_workers']}`. Đây là benchmark kỹ thuật, không phải kết quả mô hình.

## Tài nguyên và môi trường

- Python: `{report['environment']['python']}`
- PyTorch: `{report['environment']['pytorch']}`
- CPU: `{report['environment']['processor']}`
- Logical CPU: {report['environment']['logical_cpu_count']}
- RSS đầu: {resources['rss_start_mib']:.2f} MiB
- RSS lớn nhất quan sát: {resources['rss_observed_peak_mib']:.2f} MiB
- RSS cuối: {resources['rss_end_mib']:.2f} MiB
- Tổng thời gian benchmark: {resources['total_seconds']:.3f} giây

## Phạm vi sử dụng

Benchmark dùng cùng smoke manifest, seed và policy với pilot ngày 01/10. Số liệu dùng để chọn bước tối ưu tiếp theo; không ngoại suy thành tốc độ chắc chắn trên toàn bộ 432 shard nếu chưa kiểm tra phân bố shard và cache hệ điều hành.
"""
    md_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples phải lớn hơn 0.")
    if args.repeats < 3:
        raise ValueError("repeats phải ít nhất bằng 3.")
    if args.num_workers < 0:
        raise ValueError("num_workers không được âm.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(config["smoke_optimization"]["batch_size"])
    )
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")

    torch.manual_seed(seed)
    process = psutil.Process()
    rss_observations = [rss_mib(process)]
    total_start = time.perf_counter()

    initialization_times: list[float] = []
    dataset: VSASVParquetDataset | None = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        candidate = VSASVParquetDataset(
            args.split,
            args.parquet_dir,
            training=True,
            seed=seed,
            amplitude_policy="none",
        )
        initialization_times.append(time.perf_counter() - start)
        if dataset is not None:
            dataset.close()
        dataset = candidate
        record_rss(process, rss_observations)
    if dataset is None:
        raise RuntimeError("Không khởi tạo được Dataset.")

    effective_samples = min(args.samples, len(dataset))
    effective_samples = (effective_samples // batch_size) * batch_size
    if effective_samples < batch_size:
        raise ValueError("Số mẫu hiệu lực phải chứa ít nhất một batch đầy đủ.")
    generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(len(dataset), generator=generator)[
        :effective_samples
    ].tolist()

    stage_values: dict[str, list[float]] = {
        "read_decode": [],
        "resample": [],
        "amplitude": [],
        "segmentation": [],
    }
    reference_rows: list[dict[str, Any]] = []
    native_rate_counts: dict[str, int] = {}
    for repeat in range(args.repeats):
        current_rows: list[dict[str, Any]] = []
        for index in selected_indices:
            record = dataset.records[index]

            start = time.perf_counter()
            waveform, native_sample_rate = dataset._load_audio(record)
            stage_values["read_decode"].append(time.perf_counter() - start)

            start = time.perf_counter()
            if native_sample_rate != dataset.target_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, native_sample_rate, dataset.target_sample_rate
                )
            stage_values["resample"].append(time.perf_counter() - start)

            start = time.perf_counter()
            amplitude_result = apply_amplitude_policy(
                waveform,
                "none",
                peak_target=dataset.peak_target,
                rms_target_dbfs=dataset.rms_target_dbfs,
                minimum_input_rms_dbfs=dataset.minimum_input_rms_dbfs,
            )
            waveform = amplitude_result.waveform
            stage_values["amplitude"].append(time.perf_counter() - start)

            start = time.perf_counter()
            waveform = dataset._fixed_length(waveform, index).contiguous()
            stage_values["segmentation"].append(time.perf_counter() - start)

            if waveform.shape != (dataset.target_samples,):
                raise RuntimeError(f"Waveform benchmark sai shape: {waveform.shape}")
            if not torch.isfinite(waveform).all():
                raise RuntimeError("Waveform benchmark chứa NaN hoặc Inf.")
            current_rows.append(
                {
                    "index": index,
                    "file": record.file,
                    "label": float(record.binary_label),
                    "native_sample_rate": native_sample_rate,
                    "waveform_sha256": waveform_sha256(waveform),
                    "waveform": waveform if repeat == 0 else None,
                }
            )
            native_rate_key = str(native_sample_rate)
            if repeat == 0:
                native_rate_counts[native_rate_key] = (
                    native_rate_counts.get(native_rate_key, 0) + 1
                )
            record_rss(process, rss_observations)

        if repeat == 0:
            reference_rows = current_rows
        else:
            for expected, actual in zip(reference_rows, current_rows, strict=True):
                for field in ("index", "file", "label", "waveform_sha256"):
                    if expected[field] != actual[field]:
                        raise RuntimeError(
                            f"Kết quả stage không tái lập tại {field}: "
                            f"{expected[field]} != {actual[field]}"
                        )

    subset = Subset(dataset, selected_indices)
    loader_batch_times: list[float] = []
    loader_run_times: list[float] = []
    representative_batch: dict[str, Any] | None = None
    reference_by_file = {row["file"]: row for row in reference_rows}
    for repeat in range(args.repeats):
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        iterator = iter(loader)
        run_start = time.perf_counter()
        batch_number = 0
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            loader_batch_times.append(time.perf_counter() - start)
            if representative_batch is None:
                representative_batch = batch
            for position, file_name in enumerate(batch["file"]):
                expected = reference_by_file[file_name]
                checksum = waveform_sha256(batch["waveform"][position])
                if float(batch["label"][position].item()) != expected["label"]:
                    raise RuntimeError(f"DataLoader trả sai label cho {file_name}.")
                if checksum != expected["waveform_sha256"]:
                    raise RuntimeError(f"DataLoader trả sai waveform cho {file_name}.")
            batch_number += 1
            record_rss(process, rss_observations)
        loader_run_times.append(time.perf_counter() - run_start)
        expected_batches = effective_samples // batch_size
        if batch_number != expected_batches:
            raise RuntimeError(
                f"DataLoader phải trả {expected_batches} batch, nhận {batch_number}."
            )
    if representative_batch is None:
        raise RuntimeError("DataLoader không trả batch đại diện.")

    waveforms = representative_batch["waveform"]
    labels = representative_batch["label"]
    lfcc = build_lfcc_transform(config["lfcc"])
    model = build_lcnn_from_config(config)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["smoke_optimization"]["learning_rate"]),
    )

    with torch.no_grad():
        warm_features = lfcc(waveforms).unsqueeze(1)
        warm_logits = model(warm_features)
    if not torch.isfinite(warm_features).all() or not torch.isfinite(warm_logits).all():
        raise RuntimeError("Warm-up compute chứa NaN hoặc Inf.")

    compute_values: dict[str, list[float]] = {
        "lfcc": [],
        "lcnn_forward_eval": [],
        "lcnn_forward_train": [],
        "loss": [],
        "backward": [],
        "optimizer": [],
    }
    for _ in range(args.repeats):
        start = time.perf_counter()
        with torch.no_grad():
            features = lfcc(waveforms).unsqueeze(1)
        compute_values["lfcc"].append(time.perf_counter() - start)
        if not torch.isfinite(features).all():
            raise RuntimeError("LFCC benchmark chứa NaN hoặc Inf.")

        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            logits = model(features)
        compute_values["lcnn_forward_eval"].append(time.perf_counter() - start)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Logit eval benchmark chứa NaN hoặc Inf.")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        logits = model(features)
        compute_values["lcnn_forward_train"].append(time.perf_counter() - start)

        start = time.perf_counter()
        loss = loss_function(logits, labels)
        compute_values["loss"].append(time.perf_counter() - start)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss benchmark chứa NaN hoặc Inf.")

        start = time.perf_counter()
        loss.backward()
        compute_values["backward"].append(time.perf_counter() - start)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(
            torch.isfinite(gradient).all() for gradient in gradients
        ):
            raise RuntimeError("Gradient benchmark rỗng hoặc chứa NaN/Inf.")

        start = time.perf_counter()
        optimizer.step()
        compute_values["optimizer"].append(time.perf_counter() - start)
        record_rss(process, rss_observations)

    direct_summaries = {
        name: summarize_seconds(values) for name, values in stage_values.items()
    }
    direct_median_total = sum(
        summary["median_ms"] for summary in direct_summaries.values()
    )
    direct_shares = {
        name: summary["median_ms"] / direct_median_total * 100.0
        for name, summary in direct_summaries.items()
    }

    pipeline_summaries = {
        "data_loader_batch": summarize_seconds(loader_batch_times),
        **{
            name: summarize_seconds(values)
            for name, values in compute_values.items()
        },
    }
    training_stage_medians = {
        "data_loader": pipeline_summaries["data_loader_batch"]["median_ms"],
        "lfcc": pipeline_summaries["lfcc"]["median_ms"],
        "lcnn_forward": pipeline_summaries["lcnn_forward_train"]["median_ms"],
        "loss": pipeline_summaries["loss"]["median_ms"],
        "backward": pipeline_summaries["backward"]["median_ms"],
        "optimizer": pipeline_summaries["optimizer"]["median_ms"],
    }
    training_median_total = sum(training_stage_medians.values())
    training_shares = {
        name: value / training_median_total * 100.0
        for name, value in training_stage_medians.items()
    }
    bottleneck_stage = max(training_stage_medians, key=training_stage_medians.get)
    direct_bottleneck = max(
        direct_summaries,
        key=lambda name: direct_summaries[name]["median_ms"],
    )

    total_seconds = time.perf_counter() - total_start
    record_rss(process, rss_observations)
    initialization_summary = summarize_seconds(initialization_times)
    warm_initializations = initialization_times[1:] or initialization_times
    report = {
        "status": "ĐẠT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "technical_pipeline_benchmark_not_scientific_result",
        "protocol": {
            "seed": seed,
            "amplitude_policy": "none",
            "requested_samples": args.samples,
            "samples": effective_samples,
            "repeats": args.repeats,
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            "warmup_compute_batches": 1,
            "selected_indices": selected_indices,
        },
        "environment": {
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "duckdb": __import__("duckdb").__version__,
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "dataset": {
            "split": str(args.split.resolve()),
            "parquet_dir": str(args.parquet_dir.resolve()),
            "local_examples": len(dataset),
            "split_examples": dataset.total_split_rows,
            "local_coverage": dataset.local_coverage,
            "native_sample_rate_counts": native_rate_counts,
        },
        "dataset_initialization": {
            "cold_start_ms": initialization_times[0] * 1000.0,
            "warm_median_ms": statistics.median(warm_initializations) * 1000.0,
            "summary": initialization_summary,
        },
        "direct_stage_timing": {
            **direct_summaries,
            "shares_percent": direct_shares,
        },
        "batch_pipeline_timing": {
            **pipeline_summaries,
            "loader_run": summarize_seconds(loader_run_times),
            "data_loader_throughput_samples_per_second": (
                effective_samples * args.repeats / sum(loader_run_times)
            ),
            "shares_percent": training_shares,
        },
        "bottleneck": {
            "stage": bottleneck_stage,
            "median_ms_per_batch": training_stage_medians[bottleneck_stage],
            "share_percent": training_shares[bottleneck_stage],
            "direct_stage": direct_bottleneck,
            "direct_median_ms_per_sample": direct_summaries[direct_bottleneck][
                "median_ms"
            ],
            "direct_share_percent": direct_shares[direct_bottleneck],
        },
        "model": {
            "trainable_parameters": count_trainable_parameters(model),
            "waveform_shape": list(waveforms.shape),
            "lfcc_shape": list(warm_features.shape),
            "logit_shape": list(warm_logits.shape),
        },
        "reproducibility": {
            "status": "ĐẠT",
            "checked_fields": ["index", "file", "label", "waveform_sha256"],
            "reference": [
                {key: value for key, value in row.items() if key != "waveform"}
                for row in reference_rows
            ],
        },
        "resources": {
            "rss_start_mib": rss_observations[0],
            "rss_observed_peak_mib": max(rss_observations),
            "rss_end_mib": rss_observations[-1],
            "total_seconds": total_seconds,
        },
    }
    write_reports(report, args.report_json, args.report_md)
    dataset.close()

    print(
        "Benchmark pipeline: ĐẠT | "
        f"bottleneck={bottleneck_stage} "
        f"({training_shares[bottleneck_stage]:.2f}%) | "
        f"loader={report['batch_pipeline_timing']['data_loader_throughput_samples_per_second']:.3f} mẫu/s | "
        f"reproducibility=ĐẠT"
    )
    print(f"Báo cáo: {args.report_md.resolve()}")


if __name__ == "__main__":
    main()
