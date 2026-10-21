#!/usr/bin/env python3
"""Huấn luyện và đánh giá baseline B0 LFCC + LCNN có thể resume chính xác."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    VALID_AMPLITUDE_POLICIES,
    SmokeRow,
    VSASVManifestDataset,
    VSASVRecord,
    select_smoke_rows,
)
from src.metrics import evaluate_development_scores  # noqa: E402
from src.models import (  # noqa: E402
    build_lcnn_from_config,
    build_lfcc_transform,
    count_trainable_parameters,
)
from src.training import (  # noqa: E402
    ResumableBatchSampler,
    TrainingProgress,
    TrainingRunIdentity,
    load_training_checkpoint,
    save_training_checkpoint,
)


RUN_MODES = ("technical_smoke", "pilot", "full_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-mode", choices=RUN_MODES, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lfcc_lcnn.json",
    )
    parser.add_argument(
        "--audio-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "audio.json",
    )
    parser.add_argument(
        "--amplitude-policy",
        choices=sorted(VALID_AMPLITUDE_POLICIES),
        required=True,
    )
    parser.add_argument("--step-budget", type=int, required=True)
    parser.add_argument(
        "--max-steps-this-run",
        type=int,
        help="Dừng có chủ đích sau số bước này để kiểm tra hoặc chia phiên chạy.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--development-limit", type=int)
    parser.add_argument(
        "--limit-selection",
        choices=("head", "balanced_hash"),
        default="head",
        help="Cách chọn mẫu khi technical_smoke có giới hạn.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _composite_config_sha256(model_sha256: str, audio_sha256: str) -> str:
    return hashlib.sha256(
        f"model={model_sha256}\naudio={audio_sha256}\n".encode("ascii")
    ).hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {description}: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{description} phải là một JSON object.")
    return loaded


def _report_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng môi trường không có CUDA khả dụng.")
    return torch.device(requested)


def _current_git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_positive(name: str, value: int | float) -> None:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} phải là số hữu hạn lớn hơn 0.")


def _limited_indices(
    records: Sequence[VSASVRecord],
    limit: int | None,
    *,
    require_both_labels: bool,
    selection: str,
    seed: int,
    split_name: str,
) -> list[int]:
    if limit is None:
        selected = list(range(len(records)))
    else:
        if limit <= 0:
            raise ValueError("Giới hạn số mẫu phải lớn hơn 0.")
        if require_both_labels and limit < 2:
            raise ValueError(
                "Giới hạn development phải từ 2 mẫu để có đủ hai lớp."
            )
        if selection == "balanced_hash":
            if limit % 2 != 0:
                raise ValueError(
                    "Giới hạn balanced_hash phải là số chẵn để cân bằng hai lớp."
                )
            smoke_rows = [
                SmokeRow(
                    file=record.file,
                    speaker_id=record.speaker_id,
                    utt_type=record.utt_type,
                )
                for record in records
            ]
            chosen = select_smoke_rows(
                smoke_rows,
                split_name=split_name,
                per_class=limit // 2,
                seed=seed,
            )
            index_by_file = {
                record.file: index for index, record in enumerate(records)
            }
            selected = [index_by_file[row.file] for row in chosen]
        elif selection == "head":
            selected = list(range(min(limit, len(records))))
        else:
            raise ValueError(f"Cách chọn technical subset không hợp lệ: {selection}")
    if not selected:
        raise ValueError("Tập dữ liệu được chọn không có mẫu nào.")
    if require_both_labels:
        labels = {records[index].binary_label for index in selected}
        for missing_label in {0, 1}.difference(labels):
            replacement = next(
                (
                    index
                    for index, record in enumerate(records)
                    if record.binary_label == missing_label
                ),
                None,
            )
            if replacement is None:
                raise ValueError(
                    "Tập development phải có cả bonafide và spoof để tính metric."
                )
            if limit is not None and len(selected) >= limit:
                selected[-1] = replacement
            else:
                selected.append(replacement)
    return selected


def _selection_summary(
    records: Sequence[VSASVRecord], indices: Sequence[int]
) -> dict[str, Any]:
    selected = [records[index] for index in indices]
    return {
        "samples": len(selected),
        "labels": dict(
            sorted(Counter(record.binary_label for record in selected).items())
        ),
        "utt_type_counts": dict(
            sorted(Counter(record.utt_type for record in selected).items())
        ),
        "speakers": len({record.speaker_id for record in selected}),
        "shards": len({record.shard for record in selected}),
    }


def _sample_selection_sha256(
    *,
    selection: str,
    train_records: Sequence[VSASVRecord],
    train_indices: Sequence[int],
    development_records: Sequence[VSASVRecord],
    development_indices: Sequence[int],
) -> str:
    payload = {
        "selection": selection,
        "train_files": [train_records[index].file for index in train_indices],
        "development_files": [
            development_records[index].file for index in development_indices
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _build_dataset(
    *,
    manifest: Path,
    parquet_dir: Path,
    split: str,
    training: bool,
    audio_config: dict[str, Any],
    seed: int,
    amplitude_policy: str,
) -> VSASVManifestDataset:
    waveform = audio_config["waveform"]
    amplitude = waveform["amplitude_policy"]
    return VSASVManifestDataset(
        manifest,
        parquet_dir,
        split=split,  # type: ignore[arg-type]
        training=training,
        target_sample_rate=int(waveform["target_sample_rate_hz"]),
        target_samples=int(audio_config["segmentation"]["samples"]),
        seed=seed,
        amplitude_policy=amplitude_policy,  # type: ignore[arg-type]
        peak_target=float(amplitude["peak_target"]),
        rms_target_dbfs=float(amplitude["rms_target_dbfs"]),
        minimum_input_rms_dbfs=float(amplitude["minimum_input_rms_dbfs"]),
    )


def _model_input(lfcc: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    features = lfcc(waveform)
    if features.ndim != 3:
        raise RuntimeError(f"LFCC trả tensor sai số chiều: {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise RuntimeError("Đặc trưng LFCC chứa NaN hoặc Inf.")
    return features.unsqueeze(1)


def _evaluate(
    *,
    model: torch.nn.Module,
    lfcc: torch.nn.Module,
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    process: psutil.Process,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="sum")
    labels: list[int] = []
    scores: list[float] = []
    utt_types: list[str] = []
    records: list[dict[str, Any]] = []
    loss_sum = 0.0
    loader_seconds = 0.0
    model_seconds = 0.0
    rss_peak_mib = process.memory_info().rss / (1024**2)
    model.eval()
    with torch.no_grad():
        iterator = iter(loader)
        while True:
            load_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            loader_seconds += time.perf_counter() - load_started
            model_started = time.perf_counter()
            waveform = batch["waveform"].to(device)
            targets = batch["label"].to(device)
            logits = model(_model_input(lfcc, waveform))
            loss_sum += float(loss_function(logits, targets).item())
            probabilities = torch.sigmoid(logits).detach().cpu().tolist()
            batch_labels = targets.detach().cpu().to(torch.int64).tolist()
            for file_name, speaker_id, utt_type, label, score in zip(
                batch["file"],
                batch["speaker_id"],
                batch["utt_type"],
                batch_labels,
                probabilities,
                strict=True,
            ):
                labels.append(int(label))
                scores.append(float(score))
                utt_types.append(str(utt_type))
                records.append(
                    {
                        "file": str(file_name),
                        "speaker_id": str(speaker_id),
                        "utt_type": str(utt_type),
                        "label": int(label),
                        "score": float(score),
                    }
                )
            model_seconds += time.perf_counter() - model_started
            rss_peak_mib = max(
                rss_peak_mib,
                process.memory_info().rss / (1024**2),
            )
    result = evaluate_development_scores(
        labels,
        scores,
        loss=loss_sum / len(labels),
        utt_types=utt_types,
    )
    total_seconds = loader_seconds + model_seconds
    timing = {
        "loader_seconds": loader_seconds,
        "model_seconds": model_seconds,
        "throughput_samples_per_second": len(labels) / max(total_seconds, 1e-12),
        "rss_observed_peak_mib": rss_peak_mib,
    }
    return asdict(result), records, timing


def _device_details(device: torch.device) -> dict[str, Any]:
    details: dict[str, Any] = {
        "device": str(device),
        "git_commit": _current_git_commit(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": None,
        "gpu_total_memory_mib": 0.0,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        details["gpu_name"] = properties.name
        details["gpu_total_memory_mib"] = properties.total_memory / (1024**2)
    return details


def _write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    development = report["development"]
    training = report["training"]
    resume = report["resume"]
    lines = [
        "# Báo cáo B0 LFCC + LCNN",
        "",
        f"- Trạng thái: `{report['status']}`",
        f"- Chế độ: `{report['run_mode']}`",
        f"- Thiết bị: `{report['environment']['device']}`",
        f"- Bước huấn luyện: `{training['global_step']}/{training['step_budget']}`",
        f"- Resume: `{'có' if resume['used'] else 'không'}`",
        f"- Số mẫu development: `{development['sample_count']}`",
        f"- Development loss: `{development['loss']:.8f}`",
        f"- Development EER: `{development['eer']:.8f}`",
        f"- ROC-AUC: `{development['roc_auc']:.8f}`",
        f"- Macro-F1: `{development['macro_f1']:.8f}`",
        f"- Threshold (development EER): `{development['decision_threshold']:.8f}`",
        "",
        "> Kết quả technical smoke chỉ xác nhận pipeline hoạt động, không phải kết quả khoa học cuối cùng.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    process = psutil.Process()
    rss_start_mib = process.memory_info().rss / (1024**2)
    if args.run_mode != "technical_smoke" and (
        args.train_limit is not None
        or args.development_limit is not None
        or args.limit_selection != "head"
    ):
        raise ValueError("Chỉ technical_smoke được phép giới hạn số mẫu.")
    _validate_positive("step-budget", args.step_budget)
    if args.max_steps_this_run is not None:
        _validate_positive("max-steps-this-run", args.max_steps_this_run)

    manifest = args.manifest.resolve()
    parquet_dir = args.parquet_dir.resolve()
    config_path = args.config.resolve()
    audio_config_path = args.audio_config.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {manifest}")
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục Parquet: {parquet_dir}")

    config = _load_json(config_path, "cấu hình LFCC + LCNN")
    audio_config = _load_json(audio_config_path, "cấu hình audio")
    seed = int(config["seed"])
    if seed != 2026:
        raise ValueError("Baseline B0 hiện khóa seed 2026.")
    optimization = config.get("smoke_optimization", {})
    batch_size = int(
        optimization.get("batch_size", 0)
        if args.batch_size is None
        else args.batch_size
    )
    learning_rate = float(
        optimization.get("learning_rate", 0.0)
        if args.learning_rate is None
        else args.learning_rate
    )
    _validate_positive("batch-size", batch_size)
    _validate_positive("learning-rate", learning_rate)
    device = _select_device(args.device)
    _set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_base = _build_dataset(
        manifest=manifest,
        parquet_dir=parquet_dir,
        split="closed_train",
        training=True,
        audio_config=audio_config,
        seed=seed,
        amplitude_policy=args.amplitude_policy,
    )
    development_base = _build_dataset(
        manifest=manifest,
        parquet_dir=parquet_dir,
        split="closed_dev",
        training=False,
        audio_config=audio_config,
        seed=seed,
        amplitude_policy=args.amplitude_policy,
    )
    try:
        train_indices = _limited_indices(
            train_base.records,
            args.train_limit,
            require_both_labels=False,
            selection=args.limit_selection,
            seed=seed,
            split_name="closed_train",
        )
        development_indices = _limited_indices(
            development_base.records,
            args.development_limit,
            require_both_labels=True,
            selection=args.limit_selection,
            seed=seed,
            split_name="closed_dev",
        )
        train_dataset = Subset(train_base, train_indices)
        development_dataset = Subset(development_base, development_indices)

        lfcc = build_lfcc_transform(config["lfcc"]).to(device)
        model = build_lcnn_from_config(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.BCEWithLogitsLoss()
        sampler = ResumableBatchSampler(
            dataset_size=len(train_dataset),
            batch_size=batch_size,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )

        model_sha256 = _sha256_file(config_path)
        audio_sha256 = _sha256_file(audio_config_path)
        sample_selection_sha256 = _sample_selection_sha256(
            selection=args.limit_selection,
            train_records=train_base.records,
            train_indices=train_indices,
            development_records=development_base.records,
            development_indices=development_indices,
        )
        identity = TrainingRunIdentity(
            manifest_sha256=_sha256_file(manifest),
            config_sha256=_composite_config_sha256(model_sha256, audio_sha256),
            sample_selection_sha256=sample_selection_sha256,
            seed=seed,
            run_mode=args.run_mode,
            amplitude_policy=args.amplitude_policy,
            batch_size=batch_size,
            optimizer_name=optimizer.__class__.__name__,
            learning_rate=learning_rate,
            step_budget=args.step_budget,
        )
        progress = TrainingProgress(global_step=0, epoch=0, batch_cursor=0)
        resume_source: str | None = None
        if args.resume_from is not None:
            resumed = load_training_checkpoint(
                args.resume_from,
                model=model,
                optimizer=optimizer,
                expected_identity=identity,
                sampler=sampler,
                device=device,
            )
            progress = resumed.progress
            resume_source = _report_path(resumed.source_path)
        initial_step = progress.global_step
        if initial_step > args.step_budget:
            raise ValueError("Checkpoint đã vượt quá step-budget của lần chạy hiện tại.")

        steps_this_run = 0
        training_examples = 0
        training_loader_seconds = 0.0
        training_model_seconds = 0.0
        rss_observed_peak_mib = rss_start_mib
        last_train_loss: float | None = None
        while progress.global_step < args.step_budget:
            if (
                args.max_steps_this_run is not None
                and steps_this_run >= args.max_steps_this_run
            ):
                break
            if len(sampler) == 0:
                sampler.set_epoch(sampler.epoch + 1)
            train_base.set_epoch(sampler.epoch)
            model.train()
            completed_batch = False
            iterator = iter(train_loader)
            while True:
                load_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                training_loader_seconds += time.perf_counter() - load_started
                model_started = time.perf_counter()
                waveform = batch["waveform"].to(device)
                targets = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(_model_input(lfcc, waveform))
                loss = loss_function(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss chứa NaN hoặc Inf.")
                loss.backward()
                optimizer.step()
                training_model_seconds += time.perf_counter() - model_started
                last_train_loss = float(loss.detach().cpu().item())
                training_examples += int(targets.numel())
                rss_observed_peak_mib = max(
                    rss_observed_peak_mib,
                    process.memory_info().rss / (1024**2),
                )
                steps_this_run += 1
                progress = TrainingProgress(
                    global_step=progress.global_step + 1,
                    epoch=sampler.epoch,
                    batch_cursor=sampler.batch_cursor,
                )
                completed_batch = True
                if progress.global_step >= args.step_budget or (
                    args.max_steps_this_run is not None
                    and steps_this_run >= args.max_steps_this_run
                ):
                    break
            if not completed_batch:
                raise RuntimeError("Không đọc được batch training nào.")

        checkpoint_path = save_training_checkpoint(
            args.checkpoint,
            model=model,
            optimizer=optimizer,
            progress=progress,
            identity=identity,
            sampler=sampler,
        )
        development, records, development_timing = _evaluate(
            model=model,
            lfcc=lfcc,
            dataset=development_dataset,
            batch_size=batch_size,
            device=device,
            process=process,
            seed=seed,
        )
        rss_observed_peak_mib = max(
            rss_observed_peak_mib,
            development_timing["rss_observed_peak_mib"],
        )
        development["records"] = records
        status = "complete" if progress.global_step == args.step_budget else "interrupted"
        report: dict[str, Any] = {
            "report_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": status,
            "run_mode": args.run_mode,
            "identity": asdict(identity),
            "inputs": {
                "manifest": _report_path(manifest),
                "manifest_sha256": identity.manifest_sha256,
                "parquet_dir": _report_path(parquet_dir),
                "model_config": _report_path(config_path),
                "model_config_sha256": model_sha256,
                "audio_config": _report_path(audio_config_path),
                "audio_config_sha256": audio_sha256,
                "train_sample_count": len(train_dataset),
                "development_sample_count": len(development_dataset),
                "train_limit": args.train_limit,
                "development_limit": args.development_limit,
                "limit_selection": args.limit_selection,
                "sample_selection_sha256": sample_selection_sha256,
                "train_selection": _selection_summary(
                    train_base.records, train_indices
                ),
                "development_selection": _selection_summary(
                    development_base.records, development_indices
                ),
            },
            "training": {
                "global_step": progress.global_step,
                "step_budget": args.step_budget,
                "steps_this_run": steps_this_run,
                "epoch": progress.epoch,
                "batch_cursor": progress.batch_cursor,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer.__class__.__name__,
                "last_train_loss": last_train_loss,
                "trainable_parameters": count_trainable_parameters(model),
                "num_workers": 0,
            },
            "resume": {
                "used": resume_source is not None,
                "source": resume_source,
                "initial_global_step": initial_step,
            },
            "development": development,
            "checkpoint": {
                "path": _report_path(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
            },
            "environment": _device_details(device),
            "resources": {
                "elapsed_seconds": time.perf_counter() - started,
                "process_id": os.getpid(),
                "rss_start_mib": rss_start_mib,
                "rss_observed_peak_mib": rss_observed_peak_mib,
                "rss_end_mib": process.memory_info().rss / (1024**2),
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0.0
                ),
                "training_loader_seconds": training_loader_seconds,
                "training_model_seconds": training_model_seconds,
                "training_throughput_samples_per_second": training_examples
                / max(training_loader_seconds + training_model_seconds, 1e-12),
                "development_loader_seconds": development_timing[
                    "loader_seconds"
                ],
                "development_model_seconds": development_timing["model_seconds"],
                "development_throughput_samples_per_second": development_timing[
                    "throughput_samples_per_second"
                ],
            },
        }
        args.report_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report_json.resolve().write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_report_markdown(args.report_md.resolve(), report)
        print(
            "B0 "
            f"{status}: global_step={progress.global_step}/{args.step_budget}, "
            f"development={development['sample_count']}, "
            f"report={_report_path(args.report_json)}"
        )
        return 0
    finally:
        train_base.close()
        development_base.close()


if __name__ == "__main__":
    raise SystemExit(main())
