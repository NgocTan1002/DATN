#!/usr/bin/env python3
"""Record the active PyTorch/CUDA runtime without modifying the environment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "gpu_runtime.json",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Báo lỗi nếu PyTorch không nhìn thấy CUDA/GPU.",
    )
    return parser.parse_args()


def current_git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_runtime_info() -> dict[str, Any]:
    """Collect version and accelerator facts from the active Python process."""

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "total_memory_mib": properties.total_memory / (1024**2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": current_git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": devices,
    }


def validate_runtime(
    report: Mapping[str, Any], *, require_cuda: bool
) -> None:
    """Reject a CPU-only runtime when the caller requested the GPU gate."""

    if require_cuda and not bool(report.get("cuda_available")):
        raise RuntimeError(
            "PyTorch không nhìn thấy CUDA; kiểm tra Kaggle Accelerator và không "
            "cài requirements.txt/requirements-lock.txt khóa bản CPU."
        )
    if require_cuda and int(report.get("device_count", 0)) <= 0:
        raise RuntimeError("CUDA được báo khả dụng nhưng không có GPU nào được liệt kê.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    report = collect_runtime_info()
    validate_runtime(report, require_cuda=args.require_cuda)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "GPU runtime: ĐẠT | "
        f"cuda={report['cuda_available']} | devices={report['device_count']}"
    )
    print(f"Báo cáo: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
