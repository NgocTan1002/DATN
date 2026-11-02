#!/usr/bin/env python3
"""Kiểm chứng X0/X1 bằng Wav2Vec2 thu nhỏ, trọng số ngẫu nhiên, chỉ trên CPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xlsr import XLSRBinaryClassifier  # noqa: E402


def build_tiny_backbone(config: dict[str, Any]) -> nn.Module:
    """Construct a real HF encoder without downloading pretrained weights."""
    try:
        from transformers import Wav2Vec2Config, Wav2Vec2Model
    except ImportError as exc:
        raise RuntimeError("Thiếu dependency XLS-R; hãy cài requirements-xlsr.txt.") from exc
    return Wav2Vec2Model(Wav2Vec2Config(**config["backbone"]))


def build_tiny_classifier(config: dict[str, Any], unfrozen_layers: int) -> XLSRBinaryClassifier:
    return XLSRBinaryClassifier(
        build_tiny_backbone(config),
        num_unfrozen_layers=unfrozen_layers,
        **config["classifier"],
    )


def run_smoke(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audio = json.loads((PROJECT_ROOT / "configs/audio.json").read_text(encoding="utf-8"))
    if config["seed"] != 2026 or config["input"] != {
        "sample_rate_hz": audio["waveform"]["target_sample_rate_hz"],
        "samples": audio["segmentation"]["samples"],
    }:
        raise ValueError("Cấu hình smoke XLS-R phải khớp seed và giao thức âm thanh đang khóa.")

    results: dict[str, Any] = {}
    for name, unfrozen in (("X0", 0), ("X1", config["x1_unfrozen_layers"])):
        torch.manual_seed(config["seed"])
        model = build_tiny_classifier(config, unfrozen)
        # Generated waveforms only. No VSASV utterance or test data is read.
        waveform = torch.randn(2, config["input"]["samples"]) * 0.05
        labels = torch.tensor([0.0, 1.0])
        before = {key: value.detach().clone() for key, value in model.named_parameters()}
        optimizer = torch.optim.Adam(
            model.optimizer_groups(encoder_lr=1e-4, classifier_lr=1e-3)
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(waveform)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        parameters = dict(model.named_parameters())
        gradients_valid = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            if parameter.requires_grad else parameter.grad is None
            for parameter in parameters.values()
        )
        optimizer.step()
        changed = [key for key, parameter in parameters.items() if not torch.equal(before[key], parameter)]
        frozen_unchanged = all(
            key not in changed for key, parameter in parameters.items() if not parameter.requires_grad
        )
        head_changed = any(key.startswith("classifier.") for key in changed)
        opened_layers_changed = all(
            any(key.startswith(f"backbone.encoder.layers.{index}.") for key in changed)
            for index in range(len(model.backbone.encoder.layers) - unfrozen, len(model.backbone.encoder.layers))
        )
        model.eval()
        with torch.no_grad():
            expected = model(waveform)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = build_tiny_classifier(config, unfrozen)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        restored.eval()
        with torch.no_grad():
            difference = (expected - restored(waveform)).abs().max().item()
        passed = (
            logits.shape == (2,) and bool(torch.isfinite(loss)) and gradients_valid
            and frozen_unchanged and head_changed and opened_layers_changed and difference == 0.0
        )
        results[name] = {
            "passed": passed,
            "unfrozen_transformer_layers": unfrozen,
            "total_parameters": sum(parameter.numel() for parameter in parameters.values()),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters.values() if parameter.requires_grad),
            "loss": loss.item(),
            "gradients_valid": gradients_valid,
            "frozen_parameters_unchanged": frozen_unchanged,
            "classifier_updated": head_changed,
            "opened_layers_updated": opened_layers_changed,
            "state_dict_restore_max_difference": difference,
        }
        if not passed:
            raise RuntimeError(f"Smoke {name} không đạt: {results[name]}")

    import transformers

    return {
        "status": "passed",
        "scope": "technical_only_tiny_random_encoder_synthetic_audio_not_scientific_results",
        "seed": config["seed"],
        "dataset_samples": 0,
        "dataset_shards": 0,
        "synthetic_waveforms_per_mode": 2,
        "input": config["input"],
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "audio_config_sha256": hashlib.sha256((PROJECT_ROOT / "configs/audio.json").read_bytes()).hexdigest(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/xlsr_smoke.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/xlsr_cpu_smoke.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_smoke(args.config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Smoke kỹ thuật XLS-R X0/X1 đạt trên CPU; dữ liệu tổng hợp, không phải metric khoa học.")
    print(f"Báo cáo: {args.report}")


if __name__ == "__main__":
    main()
