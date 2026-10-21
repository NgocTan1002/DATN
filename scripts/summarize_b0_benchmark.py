#!/usr/bin/env python3
"""Tổng hợp benchmark B0 lặp lại và đề xuất step budget vận hành."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--full-development-samples", type=int, default=4115)
    parser.add_argument("--pilot-count", type=int, default=3)
    parser.add_argument("--reserve-run-equivalents", type=int, default=1)
    parser.add_argument("--total-gpu-hours", type=float, default=20.0)
    parser.add_argument("--step-multiple", type=int, default=64)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} phải là một mapping.")
    return value


def _require_positive(name: str, value: int | float) -> None:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} phải là số hữu hạn lớn hơn 0.")


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def build_summary(
    reports: Sequence[Mapping[str, Any]],
    *,
    source_names: Sequence[str],
    full_development_samples: int,
    pilot_count: int,
    reserve_run_equivalents: int,
    total_gpu_hours: float,
    step_multiple: int,
) -> dict[str, Any]:
    """Kiểm tra ba repeat và chọn step budget bảo thủ theo thời gian xấu nhất."""

    if len(reports) < 3:
        raise ValueError("Benchmark phải có ít nhất ba repeat.")
    if len(source_names) != len(reports):
        raise ValueError("source_names phải khớp số report benchmark.")
    for name, value in (
        ("full_development_samples", full_development_samples),
        ("pilot_count", pilot_count),
        ("reserve_run_equivalents", reserve_run_equivalents),
        ("total_gpu_hours", total_gpu_hours),
        ("step_multiple", step_multiple),
    ):
        _require_positive(name, value)

    first_identity = dict(_require_mapping(reports[0].get("identity"), "identity"))
    repeats: list[dict[str, Any]] = []
    training_step_seconds: list[float] = []
    development_sample_seconds: list[float] = []
    selection_sha256: str | None = None

    for index, (report, source_name) in enumerate(
        zip(reports, source_names, strict=True), start=1
    ):
        if report.get("status") != "complete":
            raise ValueError(f"Repeat {index} chưa complete.")
        if report.get("run_mode") != "technical_smoke":
            raise ValueError(f"Repeat {index} không phải technical_smoke.")
        identity = dict(_require_mapping(report.get("identity"), "identity"))
        if identity != first_identity:
            raise ValueError(f"Repeat {index} có identity không khớp.")
        inputs = _require_mapping(report.get("inputs"), "inputs")
        if inputs.get("limit_selection") != "balanced_hash":
            raise ValueError(f"Repeat {index} không dùng balanced_hash.")
        current_selection = str(inputs.get("sample_selection_sha256", ""))
        if len(current_selection) != 64:
            raise ValueError(f"Repeat {index} thiếu selection SHA-256 hợp lệ.")
        if selection_sha256 is None:
            selection_sha256 = current_selection
        elif current_selection != selection_sha256:
            raise ValueError(f"Repeat {index} có sample selection không khớp.")

        training = _require_mapping(report.get("training"), "training")
        development = _require_mapping(report.get("development"), "development")
        environment = _require_mapping(report.get("environment"), "environment")
        resources = _require_mapping(report.get("resources"), "resources")
        if str(environment.get("device")) != "cuda":
            raise ValueError(f"Repeat {index} không chạy trên CUDA.")
        steps = int(training.get("steps_this_run", 0))
        step_budget = int(training.get("step_budget", 0))
        global_step = int(training.get("global_step", -1))
        development_samples = int(development.get("sample_count", 0))
        if steps <= 0 or steps != step_budget or global_step != step_budget:
            raise ValueError(f"Repeat {index} không hoàn thành đúng step budget.")
        if development_samples <= 0:
            raise ValueError(f"Repeat {index} không có development sample.")

        training_seconds = float(resources["training_loader_seconds"]) + float(
            resources["training_model_seconds"]
        )
        development_seconds = float(
            resources["development_loader_seconds"]
        ) + float(resources["development_model_seconds"])
        _require_positive("training_seconds", training_seconds)
        _require_positive("development_seconds", development_seconds)
        step_seconds = training_seconds / steps
        sample_seconds = development_seconds / development_samples
        training_step_seconds.append(step_seconds)
        development_sample_seconds.append(sample_seconds)
        repeats.append(
            {
                "repeat": index,
                "source": source_name,
                "training_seconds": training_seconds,
                "training_step_seconds": step_seconds,
                "development_seconds": development_seconds,
                "development_sample_seconds": sample_seconds,
                "steps": steps,
                "development_samples": development_samples,
                "gpu_name": environment.get("gpu_name"),
            }
        )

    conservative_step_seconds = max(training_step_seconds)
    conservative_development_sample_seconds = max(development_sample_seconds)
    projected_development_seconds = (
        conservative_development_sample_seconds * full_development_samples
    )
    run_equivalents = pilot_count + reserve_run_equivalents
    allowed_per_run_seconds = total_gpu_hours * 3600.0 / run_equivalents
    available_training_seconds = allowed_per_run_seconds - projected_development_seconds
    if available_training_seconds <= 0.0:
        raise ValueError(
            "Ngân sách GPU không đủ cho full development và số run dự phòng."
        )
    unrounded_steps = math.floor(
        available_training_seconds / conservative_step_seconds
    )
    step_budget = unrounded_steps // step_multiple * step_multiple
    if step_budget < step_multiple:
        raise ValueError("Step budget sau làm tròn nhỏ hơn một step multiple.")
    projected_training_seconds = step_budget * conservative_step_seconds
    projected_run_seconds = projected_training_seconds + projected_development_seconds
    projected_total_gpu_hours = projected_run_seconds * run_equivalents / 3600.0

    return {
        "report_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "ready",
        "scope": "operational_budget_not_scientific_result",
        "repeat_count": len(reports),
        "identity": first_identity,
        "sample_selection_sha256": selection_sha256,
        "repeats": repeats,
        "measurement": {
            "training_step_seconds": training_step_seconds,
            "development_sample_seconds": development_sample_seconds,
            "median_training_step_seconds": statistics.median(
                training_step_seconds
            ),
            "median_development_sample_seconds": statistics.median(
                development_sample_seconds
            ),
            "conservative_training_step_seconds": conservative_step_seconds,
            "conservative_development_sample_seconds": (
                conservative_development_sample_seconds
            ),
            "training_step_cv": _coefficient_of_variation(training_step_seconds),
            "development_sample_cv": _coefficient_of_variation(
                development_sample_seconds
            ),
        },
        "budget_policy": {
            "full_development_samples": full_development_samples,
            "pilot_count": pilot_count,
            "reserve_run_equivalents": reserve_run_equivalents,
            "run_equivalents": run_equivalents,
            "total_gpu_hours": float(total_gpu_hours),
            "step_multiple": step_multiple,
            "timing_basis": "maximum_of_three_repeats",
        },
        "recommendation": {
            "step_budget": step_budget,
            "unrounded_step_capacity": unrounded_steps,
            "projected_training_hours_per_run": projected_training_seconds / 3600.0,
            "projected_development_hours_per_run": (
                projected_development_seconds / 3600.0
            ),
            "projected_hours_per_run": projected_run_seconds / 3600.0,
            "projected_total_gpu_hours": projected_total_gpu_hours,
            "remaining_margin_hours": total_gpu_hours - projected_total_gpu_hours,
        },
    }


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    measurement = _require_mapping(summary["measurement"], "measurement")
    recommendation = _require_mapping(summary["recommendation"], "recommendation")
    policy = _require_mapping(summary["budget_policy"], "budget_policy")
    lines = [
        "# Benchmark và ngân sách pilot B0",
        "",
        f"- Trạng thái: `{summary['status']}`",
        f"- Số lần lặp: `{summary['repeat_count']}`",
        f"- Selection SHA-256: `{summary['sample_selection_sha256']}`",
        f"- Thời gian bảo thủ mỗi optimizer step: `{measurement['conservative_training_step_seconds']:.6f}` giây",
        f"- Thời gian bảo thủ mỗi development sample: `{measurement['conservative_development_sample_seconds']:.6f}` giây",
        f"- CV step/development: `{measurement['training_step_cv']:.4f}` / `{measurement['development_sample_cv']:.4f}`",
        f"- Full development: `{policy['full_development_samples']}` mẫu",
        f"- Ngân sách: `{policy['total_gpu_hours']:.2f}` GPU-giờ cho `{policy['run_equivalents']}` run-equivalent",
        f"- Step budget đề xuất: `{recommendation['step_budget']}`",
        f"- Dự kiến mỗi run: `{recommendation['projected_hours_per_run']:.3f}` giờ",
        f"- Dự kiến tổng: `{recommendation['projected_total_gpu_hours']:.3f}` GPU-giờ",
        "",
        "> Đây là ngân sách vận hành dựa trên timing kỹ thuật, không phải kết quả khoa học hoặc căn cứ chọn amplitude policy.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    reports: list[Mapping[str, Any]] = []
    source_names: list[str] = []
    for path in args.reports:
        resolved = path.resolve()
        loaded = json.loads(resolved.read_text(encoding="utf-8"))
        reports.append(_require_mapping(loaded, str(resolved)))
        source_names.append(path.as_posix())
    summary = build_summary(
        reports,
        source_names=source_names,
        full_development_samples=args.full_development_samples,
        pilot_count=args.pilot_count,
        reserve_run_equivalents=args.reserve_run_equivalents,
        total_gpu_hours=args.total_gpu_hours,
        step_multiple=args.step_multiple,
    )
    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_md.resolve(), summary)
    print(
        "B0 benchmark ready: "
        f"step_budget={summary['recommendation']['step_budget']}, "
        f"report={args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
