"""Checkpoint production cho training có thể tiếp tục chính xác."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .sampler import ResumableBatchSampler


CHECKPOINT_VERSION = 3


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} phải là SHA-256 gồm 64 ký tự hex.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} phải là SHA-256 gồm 64 ký tự hex.") from error


@dataclass(frozen=True)
class TrainingRunIdentity:
    """Các thuộc tính không được phép thay đổi khi resume."""

    manifest_sha256: str
    config_sha256: str
    sample_selection_sha256: str
    seed: int
    run_mode: str
    amplitude_policy: str
    batch_size: int
    optimizer_name: str
    learning_rate: float
    step_budget: int

    def __post_init__(self) -> None:
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_sha256("config_sha256", self.config_sha256)
        _require_sha256("sample_selection_sha256", self.sample_selection_sha256)
        if self.run_mode not in {"technical_smoke", "pilot", "full_training"}:
            raise ValueError("run_mode không hợp lệ cho training B0.")
        if not self.amplitude_policy:
            raise ValueError("amplitude_policy không được rỗng.")
        if self.batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0.")
        if not self.optimizer_name:
            raise ValueError("optimizer_name không được rỗng.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate phải là số hữu hạn lớn hơn 0.")
        if self.step_budget <= 0:
            raise ValueError("step_budget phải lớn hơn 0.")


@dataclass(frozen=True)
class TrainingProgress:
    """Vị trí chính xác của batch kế tiếp trong training."""

    global_step: int
    epoch: int
    batch_cursor: int

    def __post_init__(self) -> None:
        if self.global_step < 0 or self.epoch < 0 or self.batch_cursor < 0:
            raise ValueError("Bộ đếm training không được âm.")


@dataclass(frozen=True)
class TrainingResumeState:
    """Kết quả sau khi nạp một checkpoint production hợp lệ."""

    source_path: Path
    progress: TrainingProgress


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Trạng thái RNG thiếu trường bắt buộc: {missing}")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("Trạng thái NumPy RNG phải là một mapping.")
    random.setstate(state["python"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint có CUDA RNG nhưng môi trường hiện tại không có CUDA."
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Số CUDA RNG state không khớp số GPU hiện tại.")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _validate_runtime_identity(
    identity: TrainingRunIdentity,
    optimizer: torch.optim.Optimizer,
    sampler: ResumableBatchSampler,
) -> None:
    if optimizer.__class__.__name__ != identity.optimizer_name:
        raise ValueError("Optimizer hiện tại không khớp identity của run.")
    learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    if learning_rates != {identity.learning_rate}:
        raise ValueError("Learning rate hiện tại không khớp identity của run.")
    if sampler.seed != identity.seed or sampler.batch_size != identity.batch_size:
        raise ValueError("Sampler hiện tại không khớp seed hoặc batch size của run.")


def save_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: TrainingProgress,
    identity: TrainingRunIdentity,
    sampler: ResumableBatchSampler,
) -> Path:
    """Lưu checkpoint production bằng thao tác thay thế file nguyên tử."""

    _validate_runtime_identity(identity, optimizer, sampler)
    if progress.epoch != sampler.epoch or progress.batch_cursor != sampler.batch_cursor:
        raise ValueError("Progress không khớp epoch hoặc batch cursor của sampler.")
    destination = Path(checkpoint_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.tmp")
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "identity": asdict(identity),
        "progress": asdict(progress),
        "sampler_state_dict": sampler.state_dict(),
        "rng_state": _capture_rng_state(),
    }
    try:
        torch.save(checkpoint, staging)
        staging.replace(destination)
    finally:
        if staging.exists():
            staging.unlink()
    return destination


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: TrainingRunIdentity,
    sampler: ResumableBatchSampler,
    device: torch.device,
) -> TrainingResumeState:
    """Kiểm tra identity rồi khôi phục đầy đủ trạng thái training."""

    source = Path(checkpoint_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint training: {source}")
    restored = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(restored, Mapping):
        raise ValueError("Checkpoint training phải là một mapping.")
    required = {
        "checkpoint_version",
        "model_state_dict",
        "optimizer_state_dict",
        "identity",
        "progress",
        "sampler_state_dict",
        "rng_state",
    }
    missing = sorted(required.difference(restored))
    if missing:
        raise ValueError(f"Checkpoint training thiếu trường bắt buộc: {missing}")
    if int(restored["checkpoint_version"]) != CHECKPOINT_VERSION:
        raise ValueError("Phiên bản checkpoint training không được hỗ trợ.")
    restored_identity = TrainingRunIdentity(**dict(restored["identity"]))
    if restored_identity != expected_identity:
        mismatched = sorted(
            name
            for name, value in asdict(expected_identity).items()
            if getattr(restored_identity, name) != value
        )
        raise ValueError(f"Identity checkpoint không khớp tại: {mismatched}")
    _validate_runtime_identity(expected_identity, optimizer, sampler)
    progress = TrainingProgress(**dict(restored["progress"]))
    sampler_state = restored["sampler_state_dict"]
    if (
        int(sampler_state["epoch"]) != progress.epoch
        or int(sampler_state["batch_cursor"]) != progress.batch_cursor
    ):
        raise ValueError("Progress checkpoint không khớp trạng thái sampler.")

    sampler.load_state_dict(sampler_state)
    model.load_state_dict(restored["model_state_dict"])
    optimizer.load_state_dict(restored["optimizer_state_dict"])
    _restore_rng_state(restored["rng_state"])
    return TrainingResumeState(source_path=source, progress=progress)
