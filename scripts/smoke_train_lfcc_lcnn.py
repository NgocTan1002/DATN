#!/usr/bin/env python3
"""Run and document the reproducible LFCC + LCNN technical smoke pilot."""

from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import VALID_AMPLITUDE_POLICIES, VSASVParquetDataset  # noqa: E402
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
        "--train-split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "smoke_train.csv",
    )
    parser.add_argument(
        "--dev-split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "smoke_dev.csv",
    )
    parser.add_argument(
        "--amplitude-policy",
        choices=sorted(VALID_AMPLITUDE_POLICIES),
        default="none",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1,
        help="Số bước train tối đa; dùng 0 để chạy hết mọi batch trong số epoch.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "b0_lfcc_lcnn_smoke.pt",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "b0_smoke_training.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "b0_smoke_training.md",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng môi trường không có CUDA khả dụng.")
    return torch.device(requested)


def rss_mib(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024**2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} chứa NaN hoặc Inf.")


def evaluate_development(
    *,
    model: torch.nn.Module,
    lfcc: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_function: torch.nn.Module,
    device: torch.device,
    process: psutil.Process,
) -> tuple[float, int, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    peak_rss = rss_mib(process)
    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            features = lfcc(waveforms).unsqueeze(1)
            logits = model(features)
            loss = loss_function(logits, labels)
            require_finite("Development loss", loss)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            peak_rss = max(peak_rss, rss_mib(process))
    if total_examples == 0:
        raise RuntimeError("Development loader không có mẫu.")
    return total_loss / total_examples, total_examples, peak_rss


def write_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train = report["training"]
    development = report["development"]
    checkpoint = report["checkpoint"]
    resources = report["resources"]
    markdown = f"""# Báo cáo smoke pilot B0 LFCC + LCNN

## Kết quả

- Trạng thái: **{report['status']}**
- Thời điểm: `{report['created_at']}`
- Thiết bị: `{report['device']}`
- Seed: `{report['seed']}`
- Amplitude policy: `{report['amplitude_policy']}`
- Số tham số trainable: {report['model']['trainable_parameters']:,}
- Shape waveform: `{tuple(report['tensor_shapes']['waveform'])}`
- Shape LFCC: `{tuple(report['tensor_shapes']['lfcc'])}`
- Shape LCNN input: `{tuple(report['tensor_shapes']['lcnn_input'])}`
- Shape logit: `{tuple(report['tensor_shapes']['logit'])}`

## Huấn luyện kỹ thuật

| Thuộc tính | Giá trị |
|---|---:|
| Epoch đã chạy | {train['epochs_completed']} |
| Bước optimizer | {train['steps']} |
| Mẫu đã xử lý | {train['examples']} |
| Loss đầu | {train['first_loss']:.6f} |
| Loss cuối | {train['last_loss']:.6f} |
| Loss trung bình | {train['mean_loss']:.6f} |
| Loss nhỏ nhất | {train['minimum_loss']:.6f} |
| Loss lớn nhất | {train['maximum_loss']:.6f} |
| Gradient hữu hạn | {train['finite_gradients']} |
| Tham số đã thay đổi | {train['parameters_changed']} |

## Kiểm tra development

- Số mẫu: {development['examples']}
- BCE loss trung bình: {development['mean_loss']:.6f}
- Mọi giá trị hữu hạn: {development['finite']}

## Checkpoint và tài nguyên

- Checkpoint: `{checkpoint['path']}`
- Dung lượng: {checkpoint['size_bytes']:,} byte
- SHA-256: `{checkpoint['sha256']}`
- Sai khác logit lớn nhất sau khôi phục: {checkpoint['restore_max_abs_difference']:.10f}
- Thời gian train: {resources['training_seconds']:.3f} giây
- Thời gian development: {resources['development_seconds']:.3f} giây
- Tổng thời gian: {resources['total_seconds']:.3f} giây
- RSS đầu: {resources['rss_start_mib']:.2f} MiB
- RSS lớn nhất quan sát: {resources['rss_observed_peak_mib']:.2f} MiB
- RSS cuối: {resources['rss_end_mib']:.2f} MiB
- Peak VRAM: {resources['cuda_peak_allocated_mib']:.2f} MiB

## Phạm vi diễn giải

Đây là smoke pilot kỹ thuật trên subset 448 mẫu cục bộ. Kết quả chỉ chứng minh pipeline, gradient và checkpoint hoạt động; không phải kết quả khoa học, không dùng để chọn amplitude policy và không đại diện cho toàn bộ VSASV.
"""
    md_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs phải lớn hơn 0.")
    if args.max_steps < 0:
        raise ValueError("max_steps không được âm.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    optimization = config["smoke_optimization"]
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(optimization["batch_size"])
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(optimization["learning_rate"])
    )
    if batch_size <= 0 or learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("batch_size và learning_rate phải là số dương hữu hạn.")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    process = psutil.Process()
    rss_start = rss_mib(process)
    rss_peak = rss_start
    total_start = time.perf_counter()

    train_dataset = VSASVParquetDataset(
        args.train_split,
        args.parquet_dir,
        training=True,
        seed=seed,
        amplitude_policy=args.amplitude_policy,
    )
    dev_dataset = VSASVParquetDataset(
        args.dev_split,
        args.parquet_dir,
        training=False,
        seed=seed,
        amplitude_policy=args.amplitude_policy,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    lfcc = build_lfcc_transform(config["lfcc"]).to(device)
    model = build_lcnn_from_config(config).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses: list[float] = []
    examples = 0
    steps = 0
    epochs_completed = 0
    parameters_changed = False
    last_features: torch.Tensor | None = None
    tensor_shapes: dict[str, list[int]] = {}
    train_start = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        completed_epoch = True
        for batch in train_loader:
            waveforms = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            features = lfcc(waveforms)
            require_finite("LFCC", features)
            model_inputs = features.unsqueeze(1)
            logits = model(model_inputs)
            require_finite("Logit", logits)
            loss = loss_function(logits, labels)
            require_finite("Loss", loss)

            before = None
            if steps == 0:
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                }
                tensor_shapes = {
                    "waveform": list(waveforms.shape),
                    "lfcc": list(features.shape),
                    "lcnn_input": list(model_inputs.shape),
                    "logit": list(logits.shape),
                }

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise RuntimeError("Gradient rỗng hoặc chứa NaN/Inf.")
            optimizer.step()

            if before is not None:
                parameters_changed = any(
                    not torch.equal(before[name], parameter.detach())
                    for name, parameter in model.named_parameters()
                )
                if not parameters_changed:
                    raise RuntimeError("Optimizer step không làm thay đổi tham số.")

            losses.append(float(loss.item()))
            examples += labels.shape[0]
            steps += 1
            last_features = model_inputs.detach()
            rss_peak = max(rss_peak, rss_mib(process))
            if args.max_steps and steps >= args.max_steps:
                completed_epoch = False
                break
        if completed_epoch:
            epochs_completed += 1
        if args.max_steps and steps >= args.max_steps:
            break
    training_seconds = time.perf_counter() - train_start
    if not losses or last_features is None:
        raise RuntimeError("Không thực hiện được bước train nào.")

    dev_start = time.perf_counter()
    dev_loss, dev_examples, dev_peak_rss = evaluate_development(
        model=model,
        lfcc=lfcc,
        loader=dev_loader,
        loss_function=loss_function,
        device=device,
        process=process,
    )
    development_seconds = time.perf_counter() - dev_start
    rss_peak = max(rss_peak, dev_peak_rss)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "seed": seed,
        "steps": steps,
        "epochs_completed": epochs_completed,
        "amplitude_policy": args.amplitude_policy,
    }
    torch.save(checkpoint_data, args.checkpoint)

    restored_model = build_lcnn_from_config(config).to(device)
    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(), lr=learning_rate
    )
    restored = torch.load(args.checkpoint, map_location=device, weights_only=True)
    restored_model.load_state_dict(restored["model_state_dict"])
    restored_optimizer.load_state_dict(restored["optimizer_state_dict"])
    model.eval()
    restored_model.eval()
    with torch.no_grad():
        expected_logits = model(last_features)
        restored_logits = restored_model(last_features)
    restore_difference = float(
        (expected_logits - restored_logits).abs().max().item()
    )
    if restore_difference > 1e-7:
        raise RuntimeError(
            "Logit sau khôi phục checkpoint sai khác quá ngưỡng: "
            f"{restore_difference}"
        )

    total_seconds = time.perf_counter() - total_start
    rss_end = rss_mib(process)
    rss_peak = max(rss_peak, rss_end)
    cuda_peak = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    checkpoint_path = args.checkpoint.resolve()
    report = {
        "status": "ĐẠT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": str(device),
        "seed": seed,
        "amplitude_policy": args.amplitude_policy,
        "scope": "technical_smoke_pilot_not_scientific_result",
        "model": {
            "name": "B0_LFCC_LCNN_smoke",
            "trainable_parameters": count_trainable_parameters(model),
        },
        "data": {
            "train_split": str(args.train_split.resolve()),
            "train_available_examples": len(train_dataset),
            "development_split": str(args.dev_split.resolve()),
            "development_available_examples": len(dev_dataset),
        },
        "tensor_shapes": tensor_shapes,
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": epochs_completed,
            "maximum_steps": args.max_steps,
            "steps": steps,
            "examples": examples,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "mean_loss": statistics.fmean(losses),
            "minimum_loss": min(losses),
            "maximum_loss": max(losses),
            "finite_gradients": True,
            "parameters_changed": parameters_changed,
        },
        "development": {
            "examples": dev_examples,
            "mean_loss": dev_loss,
            "finite": math.isfinite(dev_loss),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "restore_max_abs_difference": restore_difference,
        },
        "resources": {
            "training_seconds": training_seconds,
            "development_seconds": development_seconds,
            "total_seconds": total_seconds,
            "rss_start_mib": rss_start,
            "rss_observed_peak_mib": rss_peak,
            "rss_end_mib": rss_end,
            "cuda_peak_allocated_mib": cuda_peak,
        },
    }
    write_reports(report, args.report_json, args.report_md)
    train_dataset.close()
    dev_dataset.close()

    print(
        "B0 smoke pilot: ĐẠT | "
        f"steps={steps} | train_loss={losses[0]:.6f}->{losses[-1]:.6f} | "
        f"dev_loss={dev_loss:.6f} | checkpoint_restore_diff={restore_difference:.2e}"
    )
    print(f"Báo cáo: {args.report_md.resolve()}")


if __name__ == "__main__":
    main()
