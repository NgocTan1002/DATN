"""D035: bộ lọc lowpass_7k port từ T19b, áp trên đoạn float32 16 kHz/4 giây."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from typing import Any

import numpy as np
import torch

from src.data.amplitude import apply_amplitude_policy

LOWPASS_7K_COEFFICIENT_SHA256 = '973b82a36db8dab4f802de2102c5feed2bfb8cf6bb5846472c07d9984ab4c977'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


@lru_cache(maxsize=1)
def fir_kernel(cutoff: int = 7000) -> torch.Tensor:
    require(cutoff == 7000, 'Chỉ chấp nhận tần số cắt 7000 Hz theo D035.')
    n = torch.arange(513, dtype=torch.float64) - 256
    kernel = (2 * cutoff / 16000) * torch.sinc(2 * cutoff / 16000 * n)
    kernel *= torch.hann_window(513, periodic=False, dtype=torch.float64)
    return (kernel / kernel.sum()).to(torch.float32)


def verify_coefficients() -> None:
    actual = hashlib.sha256(fir_kernel(7000).numpy().tobytes()).hexdigest()
    require(actual == LOWPASS_7K_COEFFICIENT_SHA256, 'SHA hệ số lowpass_7k không khớp D035.')


@lru_cache(maxsize=1)
def fir_spectrum(cutoff: int = 7000) -> torch.Tensor:
    return torch.fft.rfft(fir_kernel(cutoff), n=65536)


def fir_filter(waveform: torch.Tensor, cutoff: int = 7000) -> torch.Tensor:
    require(waveform.shape == (64000,), 'FIR cần waveform 64000 mẫu.')
    return torch.fft.irfft(torch.fft.rfft(waveform, n=65536) * fir_spectrum(cutoff),
                          n=65536)[256:64256]


def filter_parameters(condition: str = 'lowpass_7k') -> dict[str, Any]:
    require(condition == 'lowpass_7k', 'Chỉ chấp nhận điều kiện lowpass_7k theo D035.')
    cutoff = 7000
    h = fir_kernel(cutoff)
    frequencies = (3000, 4000, 6000, 7000, 7500)
    n = np.arange(513) - 256
    response = {}
    for frequency in frequencies:
        amplitude = abs(np.sum(h.numpy().astype(np.float64) * np.exp(-2j * np.pi * frequency * n / 16000)))
        response[str(frequency)] = 20 * math.log10(max(float(amplitude), np.finfo(float).tiny))
    return {'operation': 'linear_phase_windowed_sinc', 'cutoff_hz': cutoff, 'taps': 513,
            'window': 'symmetric_Hann', 'sample_rate': 16000, 'fft_length': 65536,
            'delay_compensation_samples': 256, 'boundary': 'zero_extension',
            'coefficient_float32_sha256': hashlib.sha256(h.numpy().tobytes()).hexdigest(),
            'response_db': response, 'repeak': True}


def apply_condition(waveform: torch.Tensor, condition: str = 'lowpass_7k') -> torch.Tensor:
    require(waveform.shape == (64000,) and waveform.dtype == torch.float32, 'Cần waveform float32/64000 mẫu.')
    require(condition == 'lowpass_7k', 'Chỉ chấp nhận điều kiện lowpass_7k theo D035.')
    changed = fir_filter(waveform, 7000)
    require(changed.shape == waveform.shape, 'Biến đổi làm sai độ dài waveform.')
    return apply_amplitude_policy(changed, 'peak').waveform


# Dừng ngay lúc nạp module nếu hệ số không còn đúng bộ lọc đã khóa.
verify_coefficients()
