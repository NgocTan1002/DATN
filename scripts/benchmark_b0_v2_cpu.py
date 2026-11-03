#!/usr/bin/env python3
"""Đo tốc độ B0 v2 (D027 mục 3, 7) trên dữ liệu packed thật, không huấn luyện chính thức.

Dùng đúng loader packed, `_model_input` (LFCC từng mẫu), LCNN, BCE và Adam của
`scripts/train_b0_full.py`, nhưng khởi tạo mô hình mới, không ghi checkpoint và không
đọc test. Chiếu thời gian một seed theo số epoch tối đa (không giả định dừng sớm).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Subset, default_collate

from scripts import train_b0 as common
from scripts.train_b0_full import _packed
from src.training import ResumableBatchSampler


def measure(
    *,
    train: Any,
    dev: Any,
    config: dict[str, Any],
    train_steps: int,
    dev_samples: int,
    warmup_steps: int = 3,
    batch_size: int = 8,
    seed: int = 2026,
) -> dict[str, Any]:
    """Đo giây/bước train và giây/mẫu development trên dữ liệu đã mở."""

    if train_steps <= 0 or dev_samples <= 0 or warmup_steps < 0:
        raise ValueError("Số bước và số mẫu đo phải lớn hơn 0.")
    device = torch.device("cpu")
    common._set_seed(seed)
    model = common.build_lcnn_from_config(config)
    lfcc = common.build_lfcc_transform(config["lfcc"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sampler = ResumableBatchSampler(dataset_size=len(train), batch_size=batch_size, seed=seed)
    iterator = iter(sampler)
    load_times: list[float] = []
    model_times: list[float] = []
    for step in range(warmup_steps + train_steps):
        indices = next(iterator)
        stamp = time.perf_counter()
        batch = default_collate([train[i] for i in indices])
        loaded = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(common._model_input(lfcc, batch["waveform"].to(device)))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["label"].to(device)
        )
        if not torch.isfinite(loss):
            raise ValueError("Training loss không hữu hạn khi đo.")
        loss.backward()
        optimizer.step()
        done = time.perf_counter()
        if step >= warmup_steps:
            load_times.append(loaded - stamp)
            model_times.append(done - loaded)

    count = min(dev_samples, len(dev))
    # Lấy mẫu rải đều toàn dev-full để không chỉ đo một vài shard/speaker đầu.
    stride = max(len(dev) // count, 1)
    indices = [min(i * stride, len(dev) - 1) for i in range(count)]
    _result, _records, timing = common._evaluate(
        model=model,
        lfcc=lfcc,
        dataset=Subset(dev, indices),
        batch_size=batch_size,
        device=device,
        process=__import__("psutil").Process(),
        seed=seed,
    )
    step_seconds = [a + b for a, b in zip(load_times, model_times)]
    return {
        "threads": torch.get_num_threads(),
        "train_steps_measured": train_steps,
        "seconds_per_step_mean": statistics.fmean(step_seconds),
        "seconds_per_step_median": statistics.median(step_seconds),
        "seconds_per_step_cv": statistics.pstdev(step_seconds) / statistics.fmean(step_seconds),
        "loader_seconds_per_step_mean": statistics.fmean(load_times),
        "model_seconds_per_step_mean": statistics.fmean(model_times),
        "dev_samples_measured": count,
        "seconds_per_dev_sample": (timing["loader_seconds"] + timing["model_seconds"]) / count,
        "dev_loader_seconds_per_sample": timing["loader_seconds"] / count,
        "dev_model_seconds_per_sample": timing["model_seconds"] / count,
        "rss_observed_peak_mib": timing["rss_observed_peak_mib"],
    }


def project(
    measured: dict[str, Any],
    *,
    steps_per_epoch: int = 1986,
    max_epochs: int = 3,
    dev_total: int = 38777,
) -> dict[str, float]:
    """Chiếu một seed theo số epoch tối đa, đánh giá dev-full sau mỗi epoch."""

    train_hours = steps_per_epoch * max_epochs * measured["seconds_per_step_mean"] / 3600
    dev_hours = dev_total * max_epochs * measured["seconds_per_dev_sample"] / 3600
    return {
        "train_hours_max_epochs": train_hours,
        "dev_full_hours_max_epochs": dev_hours,
        "hours_per_seed_max": train_hours + dev_hours,
        "hours_per_seed_one_epoch": (train_hours + dev_hours) / max_epochs,
        "hours_three_seeds_max": 3 * (train_hours + dev_hours),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/b0_v2_training.json")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--dev-samples", type=int, default=512)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        raise ValueError("--threads phải lớn hơn 0.")
    plan = common._load_json(args.plan, "đặc tả B0 v2")
    config = common._load_json(ROOT / plan["model"]["config"], "mô hình")
    audio = common._load_json(ROOT / plan["model"]["audio_config"], "audio")
    torch.set_num_threads(args.threads)
    data = plan["data"]
    train = _packed(ROOT / data["train_artifact"], training=True, seed=2026, audio=audio)
    dev = _packed(ROOT / data["development_artifact"], training=False, seed=2026, audio=audio)
    try:
        measured = measure(
            train=train, dev=dev, config=config,
            train_steps=args.train_steps, dev_samples=args.dev_samples,
        )
    finally:
        train.close()
        dev.close()
    report = {
        "report_version": 1,
        "task": "D027-benchmark",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": "cpu",
        "environment": common._device_details(torch.device("cpu")),
        "measured": measured,
        "projection": project(measured),
        "note": "Phép chiếu theo số epoch tối đa; không phải thời gian thực chạy.",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"ĐẠT: {measured['seconds_per_step_mean']:.3f} giây/bước, "
        f"{measured['seconds_per_dev_sample']:.4f} giây/mẫu dev, "
        f"{report['projection']['hours_per_seed_max']:.2f} giờ/seed (tối đa)."
    )
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
