"""Minimal LFCC + Light CNN baseline used by the B0 smoke pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torchaudio


class MaxFeatureMap(nn.Module):
    """Reduce paired features by taking their element-wise maximum."""

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_count = inputs.shape[self.dimension]
        if feature_count % 2 != 0:
            raise ValueError(
                "MaxFeatureMap cần số đặc trưng chẵn tại chiều "
                f"{self.dimension}, nhận {feature_count}."
            )
        first, second = torch.chunk(inputs, 2, dim=self.dimension)
        return torch.maximum(first, second)


class LightCNNBaseline(nn.Module):
    """Compact Light CNN that returns one raw spoof logit per utterance."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        expected_lfcc: int = 60,
        expected_frames: int = 401,
        channels: Sequence[int] = (32, 64, 128, 128),
        kernels: Sequence[int] = (5, 3, 3, 3),
        pool_after_blocks: Sequence[int] = (1, 2, 3),
        hidden_features: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels phải lớn hơn 0.")
        if expected_lfcc <= 0 or expected_frames <= 0:
            raise ValueError("Kích thước LFCC kỳ vọng phải lớn hơn 0.")
        if not channels or len(channels) != len(kernels):
            raise ValueError("channels và kernels phải có cùng độ dài khác 0.")
        if any(value <= 0 for value in channels):
            raise ValueError("Mọi số kênh LCNN phải lớn hơn 0.")
        if any(value <= 0 or value % 2 == 0 for value in kernels):
            raise ValueError("Mỗi kernel LCNN phải là số lẻ dương.")
        if hidden_features <= 0:
            raise ValueError("hidden_features phải lớn hơn 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải thuộc khoảng [0, 1).")

        pool_after = set(pool_after_blocks)
        valid_blocks = set(range(1, len(channels) + 1))
        if not pool_after.issubset(valid_blocks):
            raise ValueError("pool_after_blocks chứa chỉ số block không hợp lệ.")

        self.input_channels = input_channels
        self.expected_lfcc = expected_lfcc
        self.expected_frames = expected_frames

        layers: list[nn.Module] = []
        current_channels = input_channels
        for block_index, (output_channels, kernel_size) in enumerate(
            zip(channels, kernels, strict=True), start=1
        ):
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        output_channels * 2,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    MaxFeatureMap(dimension=1),
                ]
            )
            if block_index in pool_after:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = output_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, hidden_features * 2),
            MaxFeatureMap(dimension=1),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected_tail = (
            self.input_channels,
            self.expected_lfcc,
            self.expected_frames,
        )
        if features.ndim != 4 or tuple(features.shape[1:]) != expected_tail:
            raise ValueError(
                "LCNN cần input (batch, channel, lfcc, frame) với phần cuối "
                f"{expected_tail}, nhận {tuple(features.shape)}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("Input LCNN chứa NaN hoặc Inf.")

        activations = self.feature_extractor(features)
        pooled = self.spatial_pool(activations).flatten(start_dim=1)
        logits = self.classifier(pooled).squeeze(1)
        if logits.shape != (features.shape[0],):
            raise RuntimeError(f"LCNN trả shape logit không hợp lệ: {logits.shape}")
        return logits


def build_lfcc_transform(lfcc_config: Mapping[str, Any]) -> torchaudio.transforms.LFCC:
    """Build the locked LFCC transform from its JSON configuration."""

    spectrum = lfcc_config["spectrum"]
    return torchaudio.transforms.LFCC(
        sample_rate=int(lfcc_config["sample_rate"]),
        n_filter=int(lfcc_config["n_filter"]),
        n_lfcc=int(lfcc_config["n_lfcc"]),
        f_min=float(lfcc_config["f_min"]),
        f_max=float(lfcc_config["f_max"]),
        dct_type=int(lfcc_config["dct_type"]),
        norm=str(lfcc_config["norm"]),
        log_lf=bool(lfcc_config["log_lf"]),
        speckwargs={
            "n_fft": int(spectrum["n_fft"]),
            "win_length": int(spectrum["win_length"]),
            "hop_length": int(spectrum["hop_length"]),
            "window_fn": torch.hann_window,
            "power": float(spectrum["power"]),
            "center": bool(spectrum["center"]),
            "pad_mode": str(spectrum["pad_mode"]),
        },
    )


def build_lcnn_from_config(config: Mapping[str, Any]) -> LightCNNBaseline:
    """Build the B0 LCNN while enforcing the locked tensor interface."""

    interface_shape = config["lcnn_interface"]["input_shape_for_locked_input"]
    architecture = config["lcnn"]
    return LightCNNBaseline(
        input_channels=int(interface_shape[1]),
        expected_lfcc=int(interface_shape[2]),
        expected_frames=int(interface_shape[3]),
        channels=tuple(int(value) for value in architecture["channels"]),
        kernels=tuple(int(value) for value in architecture["kernels"]),
        pool_after_blocks=tuple(
            int(value) for value in architecture["pool_after_blocks"]
        ),
        hidden_features=int(architecture["hidden_features"]),
        dropout=float(architecture["dropout"]),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by the optimizer."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
