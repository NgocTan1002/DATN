#!/usr/bin/env python3
"""Đối chiếu độc lập hai run D023 liên tục và ngắt–resume."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.training.full_training import GenerationStore, write_json


def equal_nested(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_nested(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_nested(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def compare_runs(continuous: Path, resumed: Path) -> dict[str, Any]:
    left, right = (GenerationStore(path).load() for path in (continuous, resumed))
    if left["identity"] != right["identity"]:
        raise ValueError("Identity của hai run smoke không khớp.")
    controllers = [copy.deepcopy(p["controller"]) for p in (left, right)]
    for c in controllers:
        if c["stop_reason"] not in {"max_epochs", "early_stopped"} or c["best"] is None:
            raise ValueError("Run smoke chưa kết thúc hoặc thiếu best.")
        c["best"].pop("generation")
        c["best"]["development"].pop("timing", None)
    checks = {
        "model_equal": equal_nested(left["model"], right["model"]),
        "optimizer_equal": equal_nested(left["optimizer"], right["optimizer"]),
        "sampler_equal": equal_nested(left["sampler"], right["sampler"]),
        "rng_equal": equal_nested(left["rng"], right["rng"]),
        "controller_losses_batches_best_history_equal": equal_nested(*controllers),
    }
    if not all(checks.values()):
        raise ValueError("Đối chứng smoke không khớp: " + str(checks))
    return {"passed": True, "mode": left["identity"]["mode"], "checks": checks, "global_step": left["controller"]["global_step"], "epochs": left["controller"]["completed_epochs"], "stop_reason": left["controller"]["stop_reason"], "cuda_rng_states": len(left["rng"]["torch_cuda"]), "max_model_difference": 0.0, "train_samples": len(left["identity"]["train_indices"]), "development_samples": len(left["identity"]["development_indices"]), "source_tree_sha256": left["identity"]["source_tree_sha256"], "scope": "technical_smoke_not_scientific_result"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.continuous, args.resumed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print("ĐẠT: model/optimizer/sampler/RNG/controller của hai run khớp tuyệt đối.")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
