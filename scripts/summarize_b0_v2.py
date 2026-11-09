#!/usr/bin/env python3
"""Tổng hợp ba report B0 v2 nối dài D028, chỉ trên dev-full."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


SEEDS = (2026, 2027, 2028)
METRICS = ("eer", "roc_auc", "macro_f1")
NOTES = [
    "dev-full, không phải test",
    "Mỗi seed dùng threshold riêng đã đóng băng từ dev-full; không gộp threshold giữa các seed.",
    "Ba run nối dài là kết quả B0 v2 chính thức theo D028; ba report trần 3 epoch chỉ là kết quả trung gian.",
    "Độ lệch chuẩn mẫu theo seed dùng mẫu số n−1; đây không phải khoảng tin cậy theo speaker.",
]


def reject_test_fields(value: object, location: str = "report") -> None:
    """Từ chối khóa test ở mọi cấp, không nhầm khóa latest hoặc tên source file."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
            tokens = re.split(r"[^a-z0-9]+", normalized)
            if any(token in {"test", "testfull"} for token in tokens):
                raise ValueError(f"Có trường liên quan tới test: {location}.{key}.")
            reject_test_fields(item, f"{location}.{key}")
    elif isinstance(value, list):
        for item in value:
            reject_test_fields(item, location)


def finite_number(value: object, name: str, *, probability: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} phải là số hữu hạn.")
    number = float(value)
    if not math.isfinite(number) or (probability and not 0 <= number <= 1):
        raise ValueError(f"{name} nằm ngoài miền giá trị hợp lệ.")
    return number


def summarize_reports(report_dir: Path) -> dict[str, Any]:
    """Chỉ đọc đúng ba report nối dài; băm chính các byte đã dùng để tổng hợp."""
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"b0_v2_ext_seed{seed}.json"
        raw = (report_dir / name).read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError(f"{name} phải là đối tượng JSON.")
        reject_test_fields(report)
        if report.get("status") not in {"early_stopped", "max_epochs"}:
            raise ValueError(f"{name}: status chưa kết thúc hợp lệ.")
        if (
            report.get("training_seed") != seed
            or report.get("data_seed") != 2026
            or report.get("backend") != "packed"
            or report["environment"]["device"] != "cpu"
            or report.get("run_mode") != "full_training"
        ):
            raise ValueError(f"{name}: identity không đúng phạm vi B0 v2 CPU.")
        if report.get("train_samples") != 15885 or report.get("development_samples") != 38777:
            raise ValueError(f"{name}: số mẫu train/dev-full không đúng giao thức.")
        extensions = report["extension_history"]
        if len(extensions) != 1 or any(
            extensions[0].get(key) != expected
            for key, expected in {
                "old_max_epochs": 3, "new_max_epochs": 10,
                "old_patience": 1, "new_patience": 2,
            }.items()
        ):
            raise ValueError(f"{name}: không phải run nối dài D028.")
        controller = report["controller"]
        if controller["stop_reason"] != report["status"]:
            raise ValueError(f"{name}: trạng thái controller không khớp report.")
        best = controller["best"]
        development = best["development"]
        if development["sample_count"] != 38777:
            raise ValueError(f"{name}: best chưa đủ dev-full.")
        if best["generation"] != report["checkpoints"]["best"]["generation"]:
            raise ValueError(f"{name}: generation best không khớp.")
        history = [
            {"epoch": entry["epoch"], "global_step": entry["global_step"],
             "eer": finite_number(entry["eer"], f"{name}: history.eer")}
            for entry in controller["history"]
        ]
        if not history or len(history) != controller["completed_epochs"]:
            raise ValueError(f"{name}: history không khớp số epoch hoàn tất.")
        if [entry["epoch"] for entry in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"{name}: thứ tự epoch không hợp lệ.")
        metrics = {key: finite_number(development[key], f"{name}: {key}") for key in METRICS}
        selected = next((entry for entry in history if entry["epoch"] == best["epoch"]), None)
        if (
            selected is None or selected["global_step"] != best["global_step"]
            or selected["eer"] != metrics["eer"]
            or min(history, key=lambda entry: entry["eer"])["epoch"] != best["epoch"]
        ):
            raise ValueError(f"{name}: best không khớp lựa chọn EER nghiêm ngặt trong history.")
        rows.append({
            "training_seed": seed, "status": report["status"],
            "final_global_step": controller["global_step"],
            "completed_epochs": controller["completed_epochs"],
            "bad_evaluations": controller["bad_evaluations"],
            "best": {"generation": best["generation"], "epoch": best["epoch"],
                     "global_step": best["global_step"], **metrics,
                     "decision_threshold": finite_number(development["decision_threshold"], f"{name}: threshold")},
            "history": history, "input_report": f"reports/{name}",
            "input_sha256": hashlib.sha256(raw).hexdigest(),
        })
    return {
        "report_version": 1, "decision": "D028", "result_scope": "B0 v2 chính thức sau nối dài",
        "evaluation_scope": NOTES[0], "notes": NOTES,
        "data": {"training_samples": 15885, "training_shards": 57,
                 "development_samples": 38777, "development_shards": 186,
                 "shard_count_source": "Phạm vi artifact đã khóa D027/D028: train 57 shard, dev-full 186 shard; không đọc lại artifact."},
        "seeds": rows,
        "aggregate": {
            key: {"n": len(rows), "mean": statistics.fmean(row["best"][key] for row in rows),
                  "sample_std": statistics.stdev(row["best"][key] for row in rows), "ddof": 1}
            for key in METRICS
        },
        "metric_units": {"eer": "tỉ lệ; nhân 100 để ra phần trăm/điểm phần trăm",
                         "roc_auc": "tỉ lệ", "macro_f1": "tỉ lệ"},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# B0 v2 D028 — kết quả ba seed nối dài", "", *summary["notes"], "",
             "Train: 15.885 mẫu/57 shard. Dev-full: 38.777 mẫu/186 shard.",
             "Số shard theo phạm vi artifact đã khóa; script chỉ đọc ba report đầu vào.", "",
             "| Seed | Status | Best epoch / bước | EER (%) | ROC-AUC | Macro-F1 | Threshold riêng |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["seeds"]:
        best = row["best"]
        lines.append(f"| {row['training_seed']} | {row['status']} | {best['epoch']} / {best['global_step']} | "
                     f"{100 * best['eer']:.6f} | {best['roc_auc']:.9f} | {best['macro_f1']:.9f} | {best['decision_threshold']:.17g} |")
    lines.extend(["", "Trung bình ± độ lệch chuẩn mẫu (n−1), qua ba seed:", ""])
    for key in METRICS:
        stats = summary["aggregate"][key]
        scale = 100 if key == "eer" else 1
        unit = " (%; độ lệch chuẩn tính bằng điểm %)" if key == "eer" else ""
        lines.append(f"- {key}: {scale * stats['mean']:.6f} ± {scale * stats['sample_std']:.6f}{unit}.")
    lines.extend(["", "| Seed | Epoch | Bước | EER dev-full (%) |", "|---|---:|---:|---:|"])
    for row in summary["seeds"]:
        for entry in row["history"]:
            lines.append(f"| {row['training_seed']} | {entry['epoch']} | {entry['global_step']} | {100 * entry['eer']:.6f} |")
    lines.extend(["", "Generation best và SHA-256 report đầu vào:", ""])
    for row in summary["seeds"]:
        lines.extend([f"- Seed {row['training_seed']}: `{row['best']['generation']}`.",
                      f"  Report: `{row['input_report']}`; SHA-256: `{row['input_sha256']}`."])
    return "\n".join(lines) + "\n"


def main() -> int:
    report_dir = Path(__file__).resolve().parents[1] / "reports"
    try:
        summary = summarize_reports(report_dir)
        encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        markdown = render_markdown(summary)
        (report_dir / "b0_v2_summary.json").write_text(encoded, encoding="utf-8")
        (report_dir / "b0_v2_summary.md").write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"KHÔNG ĐẠT: không thể tổng hợp B0 v2: {error}", file=sys.stderr)
        return 1
    print("ĐÃ TẠO: reports/b0_v2_summary.json và .md — dev-full, không phải test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
