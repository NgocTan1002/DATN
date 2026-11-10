"""T23: snapshot registry nháp từ best dev-full, không đọc dữ liệu test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/test_evaluation.json"
DRAFT = ROOT / "reports/model_registry_v1_draft.json"
# src/training/full_training.py::run_training chỉ gán hai stop_reason này;
# khi chưa kết thúc, hàm trả status="paused". scripts/train_b0_full.py ghi
# result["status"] vào report B0 v2. Report/controller XLS-R D028 đã lưu của
# X0 seed2026 và X1-L8-lr3e-5 seed2026 lần lượt xác nhận max_epochs/early_stopped.
TERMINAL_SOURCE_STATUSES: frozenset[str] = frozenset({"early_stopped", "max_epochs"})


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reject_test_path(path: Path) -> None:
    require("closed_test" not in str(path).lower(), "T23 từ chối đọc đường dẫn dữ liệu test.")


def model_id(entry: dict[str, Any]) -> str:
    return f"{entry['run_id']}-seed{entry['training_seed']}"


def protocol_sha256(config: dict[str, Any]) -> str:
    # Gate, digest registry và xác nhận vận hành không tham gia digest này:
    # tránh vòng lặp registry_sha256 <-> checksum của config khi mở gate sau này.
    keys = ("decision", "required_models", "claude_verified", "bootstrap", "conditions_by_family", "pipeline", "dev", "test")
    return hashlib.sha256(canonical({key: config[key] for key in keys})).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    require(config["decision"] == "D032", "Config phải tham chiếu D032 chính thức.")
    expected = {(family, seed) for family in ("B0-v2", "X0", "X1") for seed in (2026, 2027, 2028)}
    specs = config["required_models"]
    require(len(specs) == 9 and {(x["family"], x["training_seed"]) for x in specs} == expected,
            "Config phải có đúng9 mục mô hình-seed D032.")
    for item in specs:
        require(item["run_id"] == {"B0-v2": "B0-v2", "X0": "X0", "X1": "X1-L8-lr3e-5"}[item["family"]],
                "Run không đúng cấu hình D032.")
        reject_test_path(Path(item["source_report"]))
    require(set(config["claude_verified"]).issubset({model_id(x) for x in specs}), "Danh sách Claude kiểm có mục ngoài registry.")
    require(config["bootstrap"] == {"n_resamples": 1000, "seed": 2026, "confidence": .95}, "Bootstrap khác D032.")
    conditions = ["original", "roundtrip_40k", "lowpass_7k", "lowpass_4k"]
    require(config["conditions_by_family"] == {"B0-v2": conditions, "X0": ["original"], "X1": conditions},
            "Điều kiện band-limit khác D032.")
    require(config["pipeline"] == {"amplitude_policy": "peak", "sample_rate": 16000, "segment_samples": 64000,
            "evaluation_crop": "center", "xlsr_normalization": "processor_zero_mean_unit_variance"}, "Pipeline khác D032.")


def build_registry(config: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    validate_config(config)
    entries = []
    for spec in config["required_models"]:
        entry = {**spec, "model_id": model_id(spec), "claude_verified": model_id(spec) in config["claude_verified"],
                 "status": "missing", "missing_reason": "source_report"}
        report_path = root / spec["source_report"]
        reject_test_path(report_path)
        if not report_path.is_file():
            entries.append(entry)
            continue
        report = load_json(report_path)
        source_status = report.get("status")
        terminal = isinstance(source_status, str) and source_status in TERMINAL_SOURCE_STATUSES
        entry.update(source_status=source_status, source_report_sha256=sha256_file(report_path))
        if not terminal:
            entry.update(status="incomplete", missing_reason="source_not_terminal")
        best = report["controller"].get("best")
        point = report["checkpoints"].get("best")
        if best is None or point is None:
            require(not terminal, "Report đã kết thúc nhưng thiếu best checkpoint: " + entry["model_id"])
            entries.append(entry)
            continue
        generation = best["generation"]
        require(re.fullmatch(r"g-[0-9a-f]{32}", generation) is not None and generation == point["generation"],
                "Generation best trong report/checkpoint pointer lệch: " + entry["model_id"])
        identity = report.get("identity")
        require((identity or report)["training_seed"] == spec["training_seed"], "Seed report lệch registry.")
        if identity:
            require(identity["run_id"] == spec["run_id"] and identity["data_seed"] == 2026, "Identity run/data_seed lệch.")
        checkpoint = (root / "checkpoints" / f"b0_v2_seed{spec['training_seed']}" / generation / "checkpoint.pt"
                      if spec["family"] == "B0-v2" else
                      report_path.parent.parent / "checkpoints" / f"xlsr_{spec['run_id']}_seed{spec['training_seed']}" / generation / "checkpoint.pt")
        reject_test_path(checkpoint)
        dev = best["development"]
        require(math.isfinite(dev["decision_threshold"]) and 0 <= dev["eer"] <= 1, "Metric/threshold best không hợp lệ.")
        entry.update(checkpoint=str(checkpoint.relative_to(root)), generation=generation, best_epoch=best["epoch"],
            decision_threshold=dev["decision_threshold"], dev_eer=dev["eer"],
            source_tree_sha256=(identity or report).get("source_tree_sha256"), identity=identity,
            development_samples=report["development_samples"], reported_checkpoint_sha256=point["checkpoint_sha256"],
            missing_reason="checkpoint" if terminal else "source_not_terminal")
        if checkpoint.is_file():
            actual = sha256_file(checkpoint)
            require(actual == point["checkpoint_sha256"], "SHA-256 checkpoint lệch report: " + entry["model_id"])
            entry["checkpoint_sha256"] = actual
            if terminal:
                entry.update(status="present", missing_reason=None)
        entries.append(entry)
    return {"schema_version": 1, "decision": "D032", "role": "draft", "snapshot_utc": utc(),
            "protocol_sha256": protocol_sha256(config), "entries": entries,
            "complete": all(x["status"] == "present" and x["claude_verified"] for x in entries),
            "use_test": False, "test_gate_open": config["test_gate_open"],
            "scope": "Checkpoint chọn trên dev-full, chưa chấm test."}


def validate_registry(registry: dict[str, Any], config: dict[str, Any], *, complete: bool = False) -> None:
    validate_config(config)
    require(registry["decision"] == "D032" and registry["protocol_sha256"] == protocol_sha256(config),
            "Registry không khớp config D032.")
    specs = {model_id(x): x for x in config["required_models"]}
    entries = registry["entries"]
    require(len(entries) == 9 and {x["model_id"] for x in entries} == set(specs), "Registry thiếu/thừa/trùng mục.")
    for entry in entries:
        spec = specs[entry["model_id"]]
        require(all(entry[key] == spec[key] for key in ("family", "run_id", "training_seed", "source_report")),
                "Mục registry lệch cấu hình.")
        require(entry["claude_verified"] == (entry["model_id"] in config["claude_verified"]), "Trạng thái Claude kiểm lệch config.")
    if complete:
        require(registry["complete"] is True and all(x["status"] == "present" and x["claude_verified"] for x in entries),
                "Registry chính thức chưa đủ9 mục present/Claude kiểm.")


def write_markdown(path: Path, registry: dict[str, Any]) -> None:
    lines = ["# Registry mô hình v1 — snapshot chưa đầy đủ", "", "Quyết định: D032; complete: " + str(registry["complete"]),
             "Snapshot UTC: " + registry["snapshot_utc"], "Không phải registry chính thức; không mở cổng test.", "",
             "| Mục | Trạng thái | Claude kiểm | Generation | Threshold dev | EER dev % |", "|---|---|---|---|---:|---:|"]
    for item in registry["entries"]:
        eer = f"{100*item['dev_eer']:.6f}" if "dev_eer" in item else "—"
        lines.append(f"| {item['model_id']} | {item['status']} | {item['claude_verified']} | {item.get('generation','—')} | {item.get('decision_threshold','—')} | {eer} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DRAFT)
    args = parser.parse_args()
    try:
        require(args.output.resolve() != (ROOT / "reports/model_registry_v1.json").resolve(), "T23 không tạo registry chính thức.")
        registry = build_registry(load_json(args.config))
        atomic_json(args.output, registry)
        write_markdown(args.output.with_suffix(".md"), registry)
        print(json.dumps({"status": "draft", "complete": registry["complete"], "entries": len(registry["entries"])}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print("LỖI registry: " + str(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
