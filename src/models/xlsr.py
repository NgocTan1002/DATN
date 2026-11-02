"""Technical XLS-R binary classifier; no implicit waveform preprocessing."""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn


class MeanStdPooling(nn.Module):
    """Concatenate temporal mean and population std, stabilized at zero variance."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or any(size == 0 for size in features.shape):
            raise ValueError("Pooling cần tensor (batch, frame, feature) khác rỗng.")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("Đặc trưng pooling phải là số thực, không chứa NaN hoặc Inf.")
        variance, mean = torch.var_mean(features, dim=1, correction=0)
        return torch.cat((mean, torch.sqrt(variance + 1e-8)), dim=-1)


class XLSRBinaryClassifier(nn.Module):
    """Wrap a Wav2Vec2Model: fixed 16 kHz/4 s waveform -> one raw spoof logit.

    Zero unfrozen layers is X0. X1 opens only the last N Transformer blocks;
    convolution, projection, positional convolution and outer norms stay frozen.
    Frozen modules remain in eval mode even after train(). LayerDrop and waveform
    masking stay off; dropout is active only inside opened blocks and the head.
    These are technical defaults, not a locked scientific X0/X1 configuration.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_unfrozen_layers: int = 0,
        hidden_features: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        encoder = getattr(backbone, "encoder", None)
        layers = getattr(encoder, "layers", None)
        feature_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if not isinstance(layers, nn.ModuleList) or not layers or not isinstance(feature_size, int):
            raise ValueError("Backbone cần giao diện Wav2Vec2Model với encoder.layers và hidden_size.")
        if getattr(backbone.config, "add_adapter", False):
            raise ValueError("Khung XLS-R hiện chưa hỗ trợ backbone có adapter.")
        if (
            isinstance(num_unfrozen_layers, bool)
            or not isinstance(num_unfrozen_layers, int)
            or not 0 <= num_unfrozen_layers < len(layers)
        ):
            raise ValueError("Số tầng mở phải là số nguyên từ 0 đến nhỏ hơn tổng số tầng Transformer.")
        if isinstance(hidden_features, bool) or not isinstance(hidden_features, int) or hidden_features <= 0:
            raise ValueError("hidden_features phải là số nguyên dương.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải thuộc khoảng [0, 1).")

        self.backbone = backbone
        self.num_unfrozen_layers = num_unfrozen_layers
        self.pooling = MeanStdPooling()
        self.classifier = nn.Sequential(
            nn.Linear(feature_size * 2, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )
        self.backbone.requires_grad_(False)
        if num_unfrozen_layers:
            for layer in layers[-num_unfrozen_layers:]:
                layer.requires_grad_(True)
        self.train()

    def train(self, mode: bool = True) -> XLSRBinaryClassifier:
        super().train(mode)
        self.backbone.eval()
        if self.num_unfrozen_layers:
            for layer in self.backbone.encoder.layers[-self.num_unfrozen_layers:]:
                layer.train(mode)
        return self

    def extract_embeddings(
        self, waveform: torch.Tensor, *, sample_rate: int = 16_000
    ) -> torch.Tensor:
        """Return pooled embeddings; preserve autograd for X1 only."""
        if sample_rate != 16_000:
            raise ValueError("XLS-R cần waveform đã resample về 16.000 Hz.")
        if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[1] != 64_000:
            raise ValueError("XLS-R cần waveform (batch, 64000) mono dài 4 giây.")
        if waveform.dtype != torch.float32 or not torch.isfinite(waveform).all():
            raise ValueError("Waveform XLS-R phải là float32, không chứa NaN hoặc Inf.")
        # Fixed-length segments have no padding. Forward the existing amplitudes
        # unchanged: Wav2Vec2FeatureExtractor.do_normalize is intentionally absent.
        context = torch.no_grad() if self.num_unfrozen_layers == 0 else nullcontext()
        with context:
            output = self.backbone(waveform, return_dict=True)
            features = output.last_hidden_state
            if features.shape[0] != waveform.shape[0]:
                raise RuntimeError("Backbone XLS-R trả sai số mẫu trong batch.")
            return self.pooling(features)

    def forward(self, waveform: torch.Tensor, *, sample_rate: int = 16_000) -> torch.Tensor:
        embeddings = self.extract_embeddings(waveform, sample_rate=sample_rate)
        return self.classifier(embeddings).squeeze(-1)

    def optimizer_groups(self, *, encoder_lr: float, classifier_lr: float) -> list[dict[str, Any]]:
        """Build disjoint LR groups containing all and only trainable parameters."""
        if any(not math.isfinite(lr) or lr <= 0 for lr in (encoder_lr, classifier_lr)):
            raise ValueError("Learning rate phải hữu hạn và lớn hơn 0.")
        groups: list[dict[str, Any]] = [
            {
                "name": "classifier",
                "params": list(self.classifier.parameters()),
                "lr": classifier_lr,
            }
        ]
        parameters = [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
        if parameters:
            groups.append({"name": "encoder", "params": parameters, "lr": encoder_lr})
        return groups


def load_xlsr_backbone(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = True,
) -> nn.Module:
    """Load explicit local weights or an immutable HF revision; never random fallback.

    Transformers is optional and imported here only, so B0 has no new dependency.
    Network access requires local_files_only=False explicitly.
    """
    source = str(model_name_or_path)
    if not Path(source).is_dir() and (
        revision is None or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        raise ValueError("Model trên Hub cần revision là commit SHA đầy đủ 40 ký tự.")
    try:
        from transformers import Wav2Vec2Model
    except ImportError as exc:
        raise RuntimeError("Thiếu dependency XLS-R; hãy cài requirements-xlsr.txt.") from exc
    try:
        return Wav2Vec2Model.from_pretrained(
            source, revision=revision, local_files_only=local_files_only
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Không tải được backbone XLS-R từ {source}; kiểm tra trọng số và revision.") from exc
