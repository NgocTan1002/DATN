#!/usr/bin/env python3
"""Compare batch sizes and DataLoader workers on a fixed B0 sample set."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_data_pipeline import summarize_seconds, waveform_sha256  # noqa: E402
from src.data import VSASVParquetDataset  # noqa: E402
from src.models import build_lcnn_from_config, build_lfcc_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lfcc_lcnn.json",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_pipeline_benchmark.json",
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
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--minimum-improvement-percent", type=float, default=15.0)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_pipeline_optimization.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_pipeline_optimization.md",
    )
    return parser.parse_args()


def process_tree_rss_mib(process: psutil.Process) -> float:
    total = 0
    candidates = [process]
    try:
        candidates.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for candidate in candidates:
        try:
            total += candidate.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024**2)


def optional_summary(values: list[float]) -> dict[str, Any] | None:
    return summarize_seconds(values) if values else None


def verify_batch(
    batch: dict[str, Any],
    reference_by_file: dict[str, dict[str, Any]],
) -> None:
    for position, file_name in enumerate(batch["file"]):
        expected = reference_by_file[file_name]
        label = float(batch["label"][position].item())
        checksum = waveform_sha256(batch["waveform"][position])
        if label != expected["label"]:
            raise RuntimeError(f"Sai label cho {file_name}: {label}")
        if checksum != expected["waveform_sha256"]:
            raise RuntimeError(f"Sai waveform checksum cho {file_name}.")


def benchmark_variant(
    *,
    name: str,
    batch_size: int,
    num_workers: int,
    repeats: int,
    seed: int,
    dataset: VSASVParquetDataset,
    selected_indices: list[int],
    reference_by_file: dict[str, dict[str, Any]],
    config: dict[str, Any],
    process: psutil.Process,
) -> dict[str, Any]:
    startup_times: list[float] = []
    first_batch_waits: list[float] = []
    steady_batch_waits: list[float] = []
    all_batch_waits: list[float] = []
    lfcc_times: list[float] = []
    forward_times: list[float] = []
    loss_times: list[float] = []
    backward_times: list[float] = []
    optimizer_times: list[float] = []
    run_times: list[float] = []
    rss_observations: list[float] = [process_tree_rss_mib(process)]
    observed_files: list[str] | None = None

    for repeat in range(repeats):
        torch.manual_seed(seed)
        dataset.set_epoch(0)
        subset = Subset(dataset, selected_indices)
        model = build_lcnn_from_config(config)
        lfcc = build_lfcc_transform(config["lfcc"])
        loss_function = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["smoke_optimization"]["learning_rate"]),
        )
        loader_generator = torch.Generator().manual_seed(seed + repeat)

        run_start = time.perf_counter()
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=loader_generator,
        )
        start = time.perf_counter()
        iterator = iter(loader)
        startup_times.append(time.perf_counter() - start)

        current_files: list[str] = []
        batch_index = 0
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            wait = time.perf_counter() - start
            all_batch_waits.append(wait)
            if batch_index == 0:
                first_batch_waits.append(wait)
            else:
                steady_batch_waits.append(wait)

            verify_batch(batch, reference_by_file)
            current_files.extend(batch["file"])
            waveforms = batch["waveform"]
            labels = batch["label"]

            start = time.perf_counter()
            features = lfcc(waveforms).unsqueeze(1)
            lfcc_times.append(time.perf_counter() - start)
            if not torch.isfinite(features).all():
                raise RuntimeError(f"{name}: LFCC chứa NaN hoặc Inf.")

            model.train()
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            logits = model(features)
            forward_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            loss = loss_function(logits, labels)
            loss_times.append(time.perf_counter() - start)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: loss chứa NaN hoặc Inf.")

            start = time.perf_counter()
            loss.backward()
            backward_times.append(time.perf_counter() - start)
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise RuntimeError(f"{name}: gradient rỗng hoặc không hữu hạn.")

            start = time.perf_counter()
            optimizer.step()
            optimizer_times.append(time.perf_counter() - start)
            rss_observations.append(process_tree_rss_mib(process))
            batch_index += 1

        run_times.append(time.perf_counter() - run_start)
        if current_files != [dataset.records[index].file for index in selected_indices]:
            raise RuntimeError(f"{name}: thứ tự file không tái lập.")
        if observed_files is None:
            observed_files = current_files
        elif observed_files != current_files:
            raise RuntimeError(f"{name}: thứ tự file thay đổi giữa các lần lặp.")

        del iterator, loader, model, lfcc, optimizer
        gc.collect()
        rss_observations.append(process_tree_rss_mib(process))

    samples_per_run = len(selected_indices)
    total_examples = samples_per_run * repeats
    total_run_seconds = sum(run_times)
    total_wait_seconds = sum(startup_times) + sum(all_batch_waits)
    return {
        "name": name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "samples_per_run": samples_per_run,
        "repeats": repeats,
        "batches_per_run": math.ceil(samples_per_run / batch_size),
        "throughput_samples_per_second": total_examples / total_run_seconds,
        "data_wait_share_percent": total_wait_seconds / total_run_seconds * 100.0,
        "timing": {
            "startup": summarize_seconds(startup_times),
            "first_batch_wait": summarize_seconds(first_batch_waits),
            "steady_batch_wait": optional_summary(steady_batch_waits),
            "all_batch_wait": summarize_seconds(all_batch_waits),
            "lfcc": summarize_seconds(lfcc_times),
            "lcnn_forward": summarize_seconds(forward_times),
            "loss": summarize_seconds(loss_times),
            "backward": summarize_seconds(backward_times),
            "optimizer": summarize_seconds(optimizer_times),
            "run": summarize_seconds(run_times),
        },
        "memory": {
            "rss_tree_start_mib": rss_observations[0],
            "rss_tree_peak_mib": max(rss_observations),
            "rss_tree_end_mib": rss_observations[-1],
        },
        "reproducibility": "ĐẠT",
    }


def write_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[str] = []
    for variant in report["variants"]:
        rows.append(
            "| {name} | {batch} | {workers} | {throughput:.3f} | "
            "{improvement:+.2f}% | {run_ms:.2f} ms | {wait:.2f}% | "
            "{rss:.2f} MiB | {repro} |".format(
                name=variant["name"],
                batch=variant["batch_size"],
                workers=variant["num_workers"],
                throughput=variant["throughput_samples_per_second"],
                improvement=variant["improvement_vs_baseline_percent"],
                run_ms=variant["timing"]["run"]["median_ms"],
                wait=variant["data_wait_share_percent"],
                rss=variant["memory"]["rss_tree_peak_mib"],
                repro=variant["reproducibility"],
            )
        )
    decision = report["decision"]
    markdown = f"""# So sánh cấu hình DataLoader B0

## Giao thức

- Seed: `{report['protocol']['seed']}`
- Mẫu cố định: {report['protocol']['samples']}
- Số lần lặp: {report['protocol']['repeats']}
- Amplitude policy: `{report['protocol']['amplitude_policy']}`
- Mỗi cấu hình chạy đủ DataLoader, LFCC, LCNN, loss, backward và optimizer.
- File, label và SHA-256 waveform được đối chiếu với benchmark P0.

## Kết quả

| Cấu hình | Batch | Worker | Mẫu/giây | So với chuẩn | Trung vị/lần | Chờ dữ liệu | Peak RSS cây tiến trình | Tái lập |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Quyết định

- Cấu hình chuẩn: `{decision['baseline']}`.
- Cấu hình nhanh nhất: `{decision['winner']}`.
- Mức cải thiện: **{decision['winner_improvement_percent']:.2f}%**.
- Ngưỡng chấp nhận: {decision['minimum_improvement_percent']:.2f}%.
- Kết quả: **{decision['status']}**.
- Khuyến nghị: {decision['recommendation']}

## Diễn giải

So sánh này chỉ thay đổi batch size hoặc số worker trên cùng danh sách mẫu. Cấu hình chỉ được chấp nhận khi checksum waveform, nhãn, thứ tự file và tính hữu hạn của loss/gradient đều đạt. Số liệu là benchmark kỹ thuật trên smoke subset, không phải kết quả khoa học của mô hình.
"""
    md_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("repeats phải ít nhất bằng 3.")
    if args.samples <= 0 or args.samples % 16 != 0:
        raise ValueError("samples phải là bội số dương của 16.")
    if args.worker_count <= 0:
        raise ValueError("worker_count phải lớn hơn 0.")
    if args.minimum_improvement_percent <= 0.0:
        raise ValueError("minimum_improvement_percent phải lớn hơn 0.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    selected_indices = baseline_report["protocol"]["selected_indices"][: args.samples]
    if len(selected_indices) != args.samples:
        raise ValueError("Benchmark P0 không có đủ chỉ số mẫu đã khóa.")
    reference_rows = baseline_report["reproducibility"]["reference"]
    reference_by_file = {
        row["file"]: row
        for row in reference_rows
        if row["index"] in selected_indices
    }
    if len(reference_by_file) != args.samples:
        raise ValueError("Benchmark P0 không có đủ checksum tham chiếu.")

    dataset = VSASVParquetDataset(
        args.split,
        args.parquet_dir,
        training=True,
        seed=seed,
        amplitude_policy="none",
    )
    process = psutil.Process()
    variants_spec = [
        ("batch4_workers0", 4, 0),
        ("batch8_workers0", 8, 0),
        ("batch16_workers0", 16, 0),
        (f"batch8_workers{args.worker_count}", 8, args.worker_count),
    ]
    variants = [
        benchmark_variant(
            name=name,
            batch_size=batch_size,
            num_workers=num_workers,
            repeats=args.repeats,
            seed=seed,
            dataset=dataset,
            selected_indices=selected_indices,
            reference_by_file=reference_by_file,
            config=config,
            process=process,
        )
        for name, batch_size, num_workers in variants_spec
    ]
    dataset.close()

    baseline_name = "batch8_workers0"
    baseline = next(variant for variant in variants if variant["name"] == baseline_name)
    baseline_throughput = baseline["throughput_samples_per_second"]
    for variant in variants:
        variant["improvement_vs_baseline_percent"] = (
            variant["throughput_samples_per_second"] / baseline_throughput - 1.0
        ) * 100.0
    winner = max(variants, key=lambda variant: variant["throughput_samples_per_second"])
    winner_improvement = winner["improvement_vs_baseline_percent"]
    accepted = (
        winner["name"] != baseline_name
        and winner_improvement >= args.minimum_improvement_percent
        and winner["reproducibility"] == "ĐẠT"
    )
    if accepted:
        recommendation = (
            f"Chấp nhận {winner['name']} cho smoke/B0 tiếp theo; "
            "giữ cấu hình cũ làm đối chứng và tiếp tục theo dõi RSS."
        )
        status = "CHẤP NHẬN TỐI ƯU"
    else:
        recommendation = (
            "Giữ batch8_workers0 vì chưa có cấu hình vượt ngưỡng cải thiện "
            "mà vẫn đạt mọi cổng chất lượng."
        )
        status = "GIỮ CẤU HÌNH CHUẨN"

    report = {
        "status": "ĐẠT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "technical_dataloader_optimization_not_scientific_result",
        "protocol": {
            "seed": seed,
            "samples": args.samples,
            "repeats": args.repeats,
            "amplitude_policy": "none",
            "selected_indices": selected_indices,
            "reference_report": str(args.baseline_report.resolve()),
        },
        "variants": variants,
        "decision": {
            "baseline": baseline_name,
            "winner": winner["name"],
            "winner_improvement_percent": winner_improvement,
            "minimum_improvement_percent": args.minimum_improvement_percent,
            "accepted": accepted,
            "status": status,
            "recommendation": recommendation,
        },
    }
    write_reports(report, args.report_json, args.report_md)
    print(
        "So sánh DataLoader: ĐẠT | "
        f"winner={winner['name']} | improvement={winner_improvement:.2f}% | "
        f"decision={status}"
    )
    print(f"Báo cáo: {args.report_md.resolve()}")


if __name__ == "__main__":
    main()
