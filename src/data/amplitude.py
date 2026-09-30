"""Deterministic amplitude normalization for audio waveforms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


AmplitudePolicy = Literal["none", "peak", "rms_dbfs"]
VALID_AMPLITUDE_POLICIES = frozenset({"none", "peak", "rms_dbfs"})


@dataclass(frozen=True)
class AmplitudeResult:
    waveform: torch.Tensor
    input_peak: float
    output_peak: float
    input_rms: float
    output_rms: float
    input_rms_dbfs: float
    output_rms_dbfs: float
    applied_gain: float
    near_silence: bool
    peak_limited: bool


def _measure(
    waveform: torch.Tensor,
    epsilon: float,
) -> tuple[float, float, float]:
    """Return peak, RMS and RMS dBFS."""

    precise = waveform.to(torch.float64)
    peak = float(precise.abs().max().item())
    rms = float(torch.sqrt(torch.mean(precise.square())).item())
    rms_dbfs = 20.0 * math.log10(max(rms, epsilon))
    return peak, rms, rms_dbfs


def apply_amplitude_policy(
    waveform: torch.Tensor,
    policy: AmplitudePolicy,
    *,
    peak_target: float = 0.95,
    rms_target_dbfs: float = -25.0,
    minimum_input_rms_dbfs: float = -50.0,
    epsilon: float = 1e-8,
) -> AmplitudeResult:
    """Apply one locked amplitude policy to a mono waveform."""

    if policy not in VALID_AMPLITUDE_POLICIES:
        raise ValueError(f"Chính sách biên độ không hợp lệ: {policy}")
    if waveform.ndim != 1:
        raise ValueError("Waveform phải là tensor mono một chiều.")
    if waveform.numel() == 0:
        raise ValueError("Waveform không được rỗng.")
    if not torch.isfinite(waveform).all():
        raise ValueError("Waveform chứa NaN hoặc Inf.")
    if not 0.0 < peak_target <= 1.0:
        raise ValueError("peak_target phải thuộc khoảng (0, 1].")
    if epsilon <= 0.0:
        raise ValueError("epsilon phải lớn hơn 0.")

    output = waveform.to(torch.float32).clone()
    input_peak, input_rms, input_rms_dbfs = _measure(output, epsilon)

    near_silence = input_rms_dbfs < minimum_input_rms_dbfs
    applied_gain = 1.0
    peak_limited = False

    if policy == "peak" and not near_silence:
        if input_peak > epsilon:
            applied_gain = peak_target / input_peak

    elif policy == "rms_dbfs" and not near_silence:
        desired_gain = 10.0 ** (
            (rms_target_dbfs - input_rms_dbfs) / 20.0
        )

        if input_peak > epsilon:
            peak_safe_gain = peak_target / input_peak
            if desired_gain > peak_safe_gain:
                applied_gain = peak_safe_gain
                peak_limited = True
            else:
                applied_gain = desired_gain

    output = output * applied_gain

    if not torch.isfinite(output).all():
        raise ValueError("Waveform sau chuẩn hóa chứa NaN hoặc Inf.")

    output_peak, output_rms, output_rms_dbfs = _measure(
        output,
        epsilon,
    )

    return AmplitudeResult(
        waveform=output,
        input_peak=input_peak,
        output_peak=output_peak,
        input_rms=input_rms,
        output_rms=output_rms,
        input_rms_dbfs=input_rms_dbfs,
        output_rms_dbfs=output_rms_dbfs,
        applied_gain=applied_gain,
        near_silence=near_silence,
        peak_limited=peak_limited,
    )