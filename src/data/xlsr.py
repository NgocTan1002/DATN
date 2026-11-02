"""Tiền xử lý waveform riêng cho các thí nghiệm XLS-R đã khai báo rõ."""

from __future__ import annotations

from typing import Literal

import torch


XLSRWaveformNormalization = Literal[
    "pipeline_only",
    "processor_zero_mean_unit_variance",
]
VALID_XLSR_WAVEFORM_NORMALIZATIONS: frozenset[str] = frozenset(
    {
        "pipeline_only",
        "processor_zero_mean_unit_variance",
    }
)
XLSR_PROCESSOR_NORMALIZATION_EPSILON = 1e-7


def apply_xlsr_waveform_normalization(
    waveform: torch.Tensor,
    mode: XLSRWaveformNormalization,
) -> torch.Tensor:
    """Áp dụng normalization tường minh sau khi đã tạo đoạn 4 giây.

    ``pipeline_only`` giữ nguyên tensor từ pipeline. Chế độ processor dùng
    trung bình và phương sai tổng thể riêng cho từng đoạn, tương ứng hành vi
    Wav2Vec2FeatureExtractor đã ghim cho pilot D024.
    """

    if mode not in VALID_XLSR_WAVEFORM_NORMALIZATIONS:
        raise ValueError("Chế độ normalization waveform XLS-R không hợp lệ.")
    if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[1] != 64_000:
        raise ValueError("Normalization XLS-R cần waveform (batch, 64000).")
    if waveform.dtype != torch.float32 or not torch.isfinite(waveform).all():
        raise ValueError(
            "Waveform normalization XLS-R phải là float32, không chứa NaN hoặc Inf."
        )
    if mode == "pipeline_only":
        return waveform

    variance, mean = torch.var_mean(waveform, dim=-1, correction=0, keepdim=True)
    return (waveform - mean) * torch.rsqrt(
        variance + XLSR_PROCESSOR_NORMALIZATION_EPSILON
    )
