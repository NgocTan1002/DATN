"""Checkpoint exact-resume cho XLS-R, chỉ lưu các tham số được phép học."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    TrainingProgress,
    TrainingResumeState,
    _capture_rng_state,
    _require_sha256,
    _restore_rng_state,
)
from .sampler import ResumableBatchSampler


XLSR_CHECKPOINT_VERSION = 1
WAVEFORM_NORMALIZATION_MODES = {
    "pipeline_only",
    "processor_zero_mean_unit_variance",
}


def _require_positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} phải là số hữu hạn lớn hơn 0.")


@dataclass(frozen=True)
class XLSRTrainingRunIdentity:
    """Các thuộc tính khoa học và trọng số không được đổi khi resume XLS-R."""

    manifest_sha256: str
    audio_config_sha256: str
    sample_selection_sha256: str
    pretrained_model_id: str
    pretrained_revision: str
    pretrained_weights_sha256: str
    model_config_sha256: str
    preprocessor_config_sha256: str
    amplitude_policy: str
    waveform_normalization: str
    seed: int
    mode: str
    num_unfrozen_layers: int
    classifier_hidden_features: int
    classifier_dropout: float
    batch_size: int
    optimizer_name: str
    encoder_learning_rate: float | None
    classifier_learning_rate: float
    step_budget: int

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "audio_config_sha256",
            "sample_selection_sha256",
            "pretrained_weights_sha256",
            "model_config_sha256",
            "preprocessor_config_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not self.pretrained_model_id:
            raise ValueError("pretrained_model_id không được rỗng.")
        if len(self.pretrained_revision) != 40:
            raise ValueError("pretrained_revision phải là commit SHA gồm 40 ký tự hex.")
        try:
            int(self.pretrained_revision, 16)
        except ValueError as error:
            raise ValueError(
                "pretrained_revision phải là commit SHA gồm 40 ký tự hex."
            ) from error
        if not self.amplitude_policy:
            raise ValueError("amplitude_policy không được rỗng.")
        if self.waveform_normalization not in WAVEFORM_NORMALIZATION_MODES:
            raise ValueError("Chế độ normalization waveform XLS-R không hợp lệ.")
        if self.seed != 2026:
            raise ValueError("XLS-R phải giữ seed nghiên cứu đã khóa là 2026.")
        if self.mode not in {"X0", "X1"}:
            raise ValueError("mode XLS-R phải là X0 hoặc X1.")
        if isinstance(self.num_unfrozen_layers, bool) or not isinstance(
            self.num_unfrozen_layers, int
        ):
            raise ValueError("Số block mở XLS-R phải là số nguyên.")
        if self.mode == "X0" and self.num_unfrozen_layers != 0:
            raise ValueError("X0 không được mở block Transformer.")
        if self.mode == "X1" and self.num_unfrozen_layers <= 0:
            raise ValueError("X1 phải mở ít nhất một block Transformer cuối.")
        if self.classifier_hidden_features <= 0:
            raise ValueError("classifier_hidden_features phải lớn hơn 0.")
        if not math.isfinite(self.classifier_dropout) or not 0.0 <= self.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout phải thuộc khoảng [0, 1).")
        if self.batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0.")
        if not self.optimizer_name:
            raise ValueError("optimizer_name không được rỗng.")
        _require_positive_float(
            "classifier_learning_rate", self.classifier_learning_rate
        )
        if self.mode == "X0":
            if self.encoder_learning_rate is not None:
                raise ValueError("X0 không được có learning rate cho encoder.")
        elif self.encoder_learning_rate is None:
            raise ValueError("X1 phải có learning rate cho encoder.")
        else:
            _require_positive_float(
                "encoder_learning_rate", self.encoder_learning_rate
            )
        if self.step_budget <= 0:
            raise ValueError("step_budget phải lớn hơn 0.")


def _trainable_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not parameters:
        raise ValueError("Mô hình XLS-R không có tham số trainable.")
    return parameters


def _validate_runtime(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: XLSRTrainingRunIdentity,
    sampler: ResumableBatchSampler,
) -> dict[str, torch.nn.Parameter]:
    if getattr(model, "num_unfrozen_layers", None) != identity.num_unfrozen_layers:
        raise ValueError("Số block mở của mô hình không khớp identity XLS-R.")
    if optimizer.__class__.__name__ != identity.optimizer_name:
        raise ValueError("Optimizer hiện tại không khớp identity XLS-R.")
    expected_rates = {"classifier": identity.classifier_learning_rate}
    if identity.mode == "X1":
        expected_rates["encoder"] = float(identity.encoder_learning_rate)
    actual_rates: dict[str, float] = {}
    optimizer_parameter_ids: list[int] = []
    for group in optimizer.param_groups:
        name = group.get("name")
        if not isinstance(name, str) or name in actual_rates:
            raise ValueError("Optimizer XLS-R cần tên group duy nhất.")
        actual_rates[name] = float(group["lr"])
        optimizer_parameter_ids.extend(id(parameter) for parameter in group["params"])
    if actual_rates != expected_rates:
        raise ValueError("Learning rate/group optimizer không khớp identity XLS-R.")
    trainable = _trainable_parameters(model)
    if len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids)) or set(
        optimizer_parameter_ids
    ) != {id(parameter) for parameter in trainable.values()}:
        raise ValueError("Optimizer XLS-R không chứa đúng một lần mọi tham số trainable.")
    if sampler.seed != identity.seed or sampler.batch_size != identity.batch_size:
        raise ValueError("Sampler không khớp seed hoặc batch size của identity XLS-R.")
    return trainable


def save_xlsr_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: TrainingProgress,
    identity: XLSRTrainingRunIdentity,
    sampler: ResumableBatchSampler,
) -> Path:
    """Lưu atomic state đủ để dựng lại run từ backbone pretrained đã ghim."""

    trainable = _validate_runtime(
        model=model, optimizer=optimizer, identity=identity, sampler=sampler
    )
    if progress.epoch != sampler.epoch or progress.batch_cursor != sampler.batch_cursor:
        raise ValueError("Progress không khớp epoch hoặc batch cursor của sampler XLS-R.")
    destination = Path(checkpoint_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.tmp")
    checkpoint: dict[str, Any] = {
        "checkpoint_version": XLSR_CHECKPOINT_VERSION,
        "trainable_state_dict": {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainable.items()
        },
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


def load_xlsr_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: XLSRTrainingRunIdentity,
    sampler: ResumableBatchSampler,
    device: torch.device,
) -> TrainingResumeState:
    """Xác minh identity rồi khôi phục trainable state, optimizer, sampler và RNG."""

    source = Path(checkpoint_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint XLS-R: {source}")
    restored = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(restored, Mapping):
        raise ValueError("Checkpoint XLS-R phải là một mapping.")
    required = {
        "checkpoint_version",
        "trainable_state_dict",
        "optimizer_state_dict",
        "identity",
        "progress",
        "sampler_state_dict",
        "rng_state",
    }
    missing = sorted(required.difference(restored))
    if missing:
        raise ValueError(f"Checkpoint XLS-R thiếu trường bắt buộc: {missing}")
    if int(restored["checkpoint_version"]) != XLSR_CHECKPOINT_VERSION:
        raise ValueError("Phiên bản checkpoint XLS-R không được hỗ trợ.")
    restored_identity = XLSRTrainingRunIdentity(**dict(restored["identity"]))
    if restored_identity != expected_identity:
        mismatched = sorted(
            name
            for name, value in asdict(expected_identity).items()
            if getattr(restored_identity, name) != value
        )
        raise ValueError(f"Identity checkpoint XLS-R không khớp tại: {mismatched}")
    trainable = _validate_runtime(
        model=model,
        optimizer=optimizer,
        identity=expected_identity,
        sampler=sampler,
    )
    saved_trainable = restored["trainable_state_dict"]
    if not isinstance(saved_trainable, Mapping) or set(saved_trainable) != set(trainable):
        raise ValueError("Danh sách tham số trainable trong checkpoint XLS-R không khớp.")
    for name, parameter in trainable.items():
        value = saved_trainable[name]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise ValueError(f"Tensor trainable XLS-R không tương thích tại: {name}")

    progress = TrainingProgress(**dict(restored["progress"]))
    sampler_state = restored["sampler_state_dict"]
    if (
        int(sampler_state["epoch"]) != progress.epoch
        or int(sampler_state["batch_cursor"]) != progress.batch_cursor
    ):
        raise ValueError("Progress checkpoint không khớp trạng thái sampler XLS-R.")

    sampler.load_state_dict(sampler_state)
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(saved_trainable[name].to(device=parameter.device))
    optimizer.load_state_dict(restored["optimizer_state_dict"])
    _restore_rng_state(restored["rng_state"])
    return TrainingResumeState(source_path=source, progress=progress)
