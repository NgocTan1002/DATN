#!/usr/bin/env python3
"""Verify the project's pinned CPU audio/ML environment."""

from __future__ import annotations

import sys

import duckdb
import numpy
import pandas
import psutil
import scipy
import sklearn
import soundfile
import torch
import torchaudio


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if duckdb.connect(":memory:").execute("SELECT 1").fetchone() != (1,):
        raise RuntimeError("DuckDB không hoạt động đúng.")

    source = torch.randn(2, 40_000)
    waveform = torchaudio.functional.resample(source, 40_000, 16_000)
    lfcc = torchaudio.transforms.LFCC(sample_rate=16_000, n_lfcc=40)(waveform)

    if waveform.shape != (2, 16_000):
        raise RuntimeError(f"Resample trả shape không mong đợi: {waveform.shape}")
    if not torch.isfinite(lfcc).all():
        raise RuntimeError("LFCC chứa NaN hoặc Inf.")

    classifier = torch.nn.Linear(1, 1)
    features = lfcc.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
    loss = classifier(features).square().mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Loss kiểm tra không hữu hạn.")

    versions = {
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "soundfile": soundfile.__version__,
        "duckdb": duckdb.__version__,
        "psutil": psutil.__version__,
    }
    print("Môi trường hợp lệ.")
    for name, version in versions.items():
        print(f"- {name}: {version}")
    print(f"- CUDA khả dụng: {torch.cuda.is_available()}")
    print(f"- Resample shape: {tuple(waveform.shape)}")
    print(f"- LFCC shape: {tuple(lfcc.shape)}")
    print(f"- Loss kiểm tra: {loss.item():.6f}")


if __name__ == "__main__":
    main()
