#!/usr/bin/env python3
"""Tính dự toán B0 từ đặc tả và số đo đã lưu, không chạy huấn luyện."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_budget(
    plan: dict[str, Any], measurements: list[dict[str, Any]]
) -> dict[str, Any]:
    """Dùng phép đo xấu nhất và tính đủ tail batch, development mỗi epoch."""
    train = int(plan["data"]["train_samples"])
    dev = int(plan["data"]["development_samples"])
    training = plan["training"]
    batch = int(training["batch_size"])
    epochs = int(training["max_epochs"])
    chunk = int(plan["sessions"]["max_optimizer_steps_per_invocation"])
    reserve = float(plan["budget"]["reserve_fraction"])
    if min(train, dev, batch, epochs, chunk) <= 0 or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Quy mô, batch, epoch, giới hạn phiên và dự phòng không hợp lệ.")
    if training["drop_last"] or plan["data"]["use_test"]:
        raise ValueError("Đặc tả dự toán không cho phép bỏ tail batch hoặc dùng test.")
    if not plan["evaluation"]["full_development"] or plan["evaluation"]["schedule"] != "after_each_completed_epoch_before_next_epoch":
        raise ValueError("Dự toán yêu cầu full development sau mỗi epoch hoàn chỉnh.")
    steps = math.ceil(train / batch)
    if (
        training["steps_per_epoch"] != steps
        or training["max_optimizer_steps"] != steps * epochs
        or training["max_train_sample_exposures"] != train * epochs
    ):
        raise ValueError("Số bước hoặc lượt mẫu khai báo không khớp phép tính theo epoch.")
    if not measurements:
        raise ValueError("Thiếu số đo để tính dự toán.")
    for measurement in measurements:
        for field in ("training_step_seconds", "development_sample_seconds"):
            if not math.isfinite(float(measurement[field])) or measurement[field] <= 0:
                raise ValueError("Số đo thời gian phải hữu hạn và lớn hơn 0.")
    step_seconds = max(float(row["training_step_seconds"]) for row in measurements)
    sample_seconds = max(float(row["development_sample_seconds"]) for row in measurements)
    train_hours = steps * step_seconds / 3600
    dev_hours = dev * sample_seconds / 3600
    epoch_hours = train_hours + dev_hours
    worst_session_hours = (min(chunk, steps) * step_seconds / 3600 + dev_hours)
    return {
        "conservative_training_step_seconds": step_seconds,
        "conservative_development_sample_seconds": sample_seconds,
        "steps_per_epoch": steps,
        "tail_batch_samples": train % batch or batch,
        "max_optimizer_steps": steps * epochs,
        "max_train_sample_exposures": train * epochs,
        "max_development_evaluations": epochs,
        "sessions_per_epoch_upper_bound": math.ceil(steps / chunk),
        "sessions_upper_bound": math.ceil(steps / chunk) * epochs,
        "projected_training_hours_per_epoch": train_hours,
        "projected_development_hours_per_epoch": dev_hours,
        "projected_hours_per_epoch": epoch_hours,
        "projected_hours_max_epochs": epoch_hours * epochs,
        "projected_hours_max_epochs_with_reserve": epoch_hours * epochs * (1 + reserve),
        "projected_worst_session_hours_with_reserve": worst_session_hours * (1 + reserve),
        "includes_startup_checkpoint_resume_as_measured_cost": False,
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "configs/b0_full_training.json")
    args = parser.parse_args()
    plan = _read(args.plan)
    manifest = PROJECT_ROOT / plan["data"]["manifest"]
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    if _sha(manifest) != plan["data"]["manifest_sha256"] or any(
        sum(row["split"] == split for row in rows) != count
        for split, count in (("closed_train", plan["data"]["train_samples"]), ("closed_dev", plan["data"]["development_samples"]))
    ) or len(rows) != plan["data"]["manifest_samples"] or len({row["shard"] for row in rows}) != plan["data"]["manifest_shards"]:
        raise ValueError("Manifest không khớp checksum, số mẫu hoặc số shard trong đặc tả.")
    for key in ("config", "audio_config"):
        if _sha(PROJECT_ROOT / plan["model"][key]) != plan["model"][key + "_sha256"]:
            raise ValueError("Checksum cấu hình mô hình hoặc audio không khớp đặc tả.")
    sources = [PROJECT_ROOT / "reports/b0_benchmark_summary.json"]
    benchmark = _read(sources[0])
    measurements = [{"source": sources[0].name, "training_step_seconds": benchmark["measurement"]["conservative_training_step_seconds"], "development_sample_seconds": benchmark["measurement"]["conservative_development_sample_seconds"]}]
    for policy in ("none", "peak", "rms_dbfs"):
        source = PROJECT_ROOT / f"reports/b0_pilot_{policy}.json"
        sources.append(source)
        report = _read(source)
        if report["status"] != "complete" or report["identity"]["manifest_sha256"] != plan["data"]["manifest_sha256"] or report["development"]["sample_count"] != plan["data"]["development_samples"]:
            raise ValueError("Report pilot không hoàn thành hoặc khác manifest/development.")
        resource = report["resources"]
        measurements.append({"source": source.name, "training_step_seconds": (resource["training_loader_seconds"] + resource["training_model_seconds"]) / report["training"]["steps_this_run"], "development_sample_seconds": (resource["development_loader_seconds"] + resource["development_model_seconds"]) / report["development"]["sample_count"]})
    projection = project_budget(plan, measurements)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": "projection_not_measurement_not_training_result",
        "experiment_id": plan["experiment_id"],
        "plan_sha256": _sha(args.plan),
        "manifest_samples": len(rows),
        "manifest_shards": plan["data"]["manifest_shards"],
        "local_source_shards": plan["data"]["local_source_shards"],
        "sources": [{"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha(path)} for path in sources],
        "measurements": measurements,
        "reserve_fraction": plan["budget"]["reserve_fraction"],
        "projection": projection,
        "ready_for_training": plan["implementation"]["ready_for_training"],
    }
    output = PROJECT_ROOT / plan["outputs"]["budget_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Dự toán B0 full training", "", "Đây là phép chiếu từ số đo đã lưu, chưa chạy huấn luyện đầy đủ.", "", f"- Đặc tả: `{plan['experiment_id']}`; SHA-256 `{report['plan_sha256']}`.", f"- Manifest {len(rows):,} mẫu/{plan['data']['manifest_shards']} shard, pool cục bộ {plan['data']['local_source_shards']} shard.", f"- {projection['steps_per_epoch']:,} bước/epoch; tail batch {projection['tail_batch_samples']} mẫu.", f"- Tối đa {projection['max_optimizer_steps']:,} bước, {projection['max_train_sample_exposures']:,} lượt mẫu train và {projection['max_development_evaluations']} full-development evaluation.", f"- Dự toán mỗi epoch: {projection['projected_hours_per_epoch']:.3f} giờ.", f"- Trần {plan['training']['max_epochs']} epoch: {projection['projected_hours_max_epochs']:.3f} giờ; cộng {100*plan['budget']['reserve_fraction']:.0f}% dự phòng: {projection['projected_hours_max_epochs_with_reserve']:.3f} giờ.", f"- Một phiên tối đa {plan['sessions']['max_optimizer_steps_per_invocation']} bước và có thể kèm development: dự toán có dự phòng {projection['projected_worst_session_hours_with_reserve']:.3f} giờ.", f"- Tối đa {projection['sessions_upper_bound']} phiên theo cách chia này nếu không gián đoạn ngoài dự kiến hoặc dừng sớm.", "", "Chọn thời gian lớn nhất trên benchmark và ba pilot; training/development maximum có thể đến từ các phép đo khác nhau. Startup/checkpoint/resume chưa đo riêng, được dự trù bằng biên dự phòng; số phiên thực có thể tăng khi gián đoạn. Không coi quota lịch sử là quota hiện tại; kiểm tra lại trước mỗi phiên. Không đổi ngân sách giữa run để vừa quota, không tự dùng tài nguyên trả phí.", "", "Nguồn, checksum và số đo chi tiết trong JSON đi kèm. Tái sinh bằng:", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts/estimate_b0_full_training_budget.py", "```", ""]
    (PROJECT_ROOT / plan["outputs"]["budget_md"]).write_text("\n".join(lines), encoding="utf-8")
    print(f"ĐẠT: đã tính dự toán; ready_for_training={report['ready_for_training']}.")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
        print(f"LỖI: {error}")
        raise SystemExit(1) from error
