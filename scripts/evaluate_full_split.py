"""T23: đánh giá checkpoint best; nhánh test có gate/registry/T22/khóa một lần."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0")

import numpy as np
import torch
from scripts import probe_shortcut_dev as shared
from scripts.build_model_registry import CONFIG, DRAFT, atomic_json, load_json, require, sha256_file, utc, validate_config, validate_registry
from src.data.packed import PackedManifestDataset
from src.data.xlsr import apply_xlsr_waveform_normalization
from src.metrics.bootstrap import confusion_at_threshold, per_utt_type_eer
from src.metrics.classification import macro_f1_at_threshold, roc_auc_score
from src.metrics.eer import equal_error_rate
from src.models.lfcc_lcnn import build_lcnn_from_config, build_lfcc_transform, lfcc_features
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone


def authorize_test(config: dict[str, Any], selected_ids: list[str], root: Path = ROOT) -> dict[str, Any]:
    """Kiểm gate hoàn toàn trước khi chạm đường dẫn manifest/packed test."""
    require(config.get("test_gate_open") is True, "Cổng test đang đóng; không mở dữ liệu test.")
    registry_path = root / config["registry_path"]
    require(registry_path.is_file(), "Thiếu registry chính thức.")
    content = registry_path.read_bytes()
    require(isinstance(config.get("registry_sha256"), str) and hashlib.sha256(content).hexdigest() == config["registry_sha256"],
            "SHA-256 registry không khớp config.")
    registry = json.loads(content)
    validate_registry(registry, config, complete=True)
    require(bool(selected_ids) and len(set(selected_ids)) == len(selected_ids) and
            set(selected_ids).issubset({x["model_id"] for x in registry["entries"]}), "Danh sách mục chấm test không hợp lệ.")
    report_path = root / config["t22_report"]
    require(report_path.is_file(), "Thiếu report T22.")
    report = load_json(report_path)
    indicator = report.get("results", {}).get("X1-L8-lr3e-5-seed2026", {}).get("indicators", {})
    require(report.get("status") == "complete", "T22 chưa hoàn tất; cấm chấm test.")
    require(indicator.get("available_indicator_level") == "weak", "T22 chưa có chỉ báo weak; strong/chưa có thì cấm test.")
    final = next(x for x in registry["entries"] if x["model_id"] == "X1-L8-lr3e-5-seed2026")
    require(report.get("use_test") is False and report.get("subset_files_sha256") == shared.SUBSET_SHA and
            report.get("identity", {}).get("models", {}).get(final["model_id"], {}).get("checkpoint_sha256") == final["checkpoint_sha256"],
            "Report T22 không khớp checkpoint X1 cuối/tập con đã khóa.")
    checks = ("entrypoint_dev_claude_verified", "bootstrap_dev_claude_verified", "t22_claude_verified", "test_dataset_inventory_checked")
    require(all(config.get("pre_test_checks", {}).get(key) is True for key in checks),
            "Thiếu xác nhận kiểm dev/T22/inventory theo D032 mục8.")
    locks = root / config["locks_dir"]
    require(not any((locks / (name + ".lock")).exists() for name in selected_ids), "Đã có khóa chấm một lần; không chấm lại.")
    locks.mkdir(parents=True, exist_ok=True)
    for name in selected_ids:
        # O_EXCL công bố quyền chấm trước mọi open dữ liệu. Không tự xóa khóa khi lỗi.
        with (locks / (name + ".lock")).open("x", encoding="utf-8") as handle:
            handle.write(shared.canonical({"model_id": name, "created_utc": utc(), "registry_sha256": config["registry_sha256"]}).decode() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return registry


def test_data_after_lock(config: dict[str, Any], selected: list[str], opener: Callable[[], Any], root: Path = ROOT) -> Any:
    authorize_test(config, selected, root)
    return opener()


def summarize(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels, scores = [r["label"] for r in records], [r["score"] for r in records]
    require(bool(records) and all(math.isfinite(x) and 0 <= x <= 1 for x in scores), "Thiếu score hoặc score không hợp lệ.")
    require(math.isfinite(threshold), "Threshold không hữu hạn.")
    if set(labels) != {0, 1}:
        tn = sum(y == 0 and p < threshold for y, p in zip(labels, scores))
        fp = sum(y == 0 and p >= threshold for y, p in zip(labels, scores))
        fn = sum(y == 1 and p < threshold for y, p in zip(labels, scores))
        tp = sum(y == 1 and p >= threshold for y, p in zip(labels, scores))
        confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                     "fpr": fp / (tn + fp) if tn + fp else None, "fnr": fn / (tp + fn) if tp + fn else None}
        return {"samples": len(records), "eer": None, "roc_auc": None, "macro_f1": None,
                "confusion": confusion, "fpr": confusion["fpr"], "fnr": confusion["fnr"], "per_attack_eer": {},
                "decision_threshold": threshold, "note": "Tập kỹ thuật chỉ có một lớp; không tính EER/AUC/F1."}
    confusion = confusion_at_threshold(labels, scores, threshold=threshold)
    return {"samples": len(records), "eer": equal_error_rate(labels, scores).eer,
            "roc_auc": roc_auc_score(labels, scores), "macro_f1": macro_f1_at_threshold(labels, scores, threshold=threshold),
            "fpr": confusion["fpr"], "fnr": confusion["fnr"], "confusion": confusion,
            "per_attack_eer": per_utt_type_eer(labels, scores, [r["utt_type"] for r in records]),
            "decision_threshold": threshold, "threshold_source": "registry_fixed_best_dev", "replay_interpretation": "chỉ mô tả"}


def reproduction(records: list[dict[str, Any]], reference: dict[str, dict[str, Any]],
                 threshold: float, family: str) -> dict[str, Any]:
    saved = [reference[r["file"]]["score"] for r in records]
    current = [r["score"] for r in records]
    flips = np.asarray(current) >= threshold
    flips ^= np.asarray(saved) >= threshold
    if family == "B0-v2":
        delta = np.abs(np.asarray(current) - saved)
        return {"passed": bool(delta.max() <= 1e-5 and not flips.any()), "samples": len(records),
                "max_abs_probability_difference": float(delta.max()), "decision_flips": int(flips.sum())}
    delta = np.abs(np.asarray([r["raw_logit"] for r in records]) - shared.stable_logit(saved))
    distances = np.abs(shared.stable_logit(saved) - shared.stable_logit([threshold])[0])
    return {"passed": bool(delta.max() <= .05 and (distances[flips] <= .05).all()), "samples": len(records),
            "max_abs_logit_difference": float(delta.max()), "median_abs_logit_difference": float(np.median(delta)),
            "decision_flips": int(flips.sum()), "flips_outside_0_05_margin": int((distances[flips] > .05).sum()),
            "logit_source": "raw_logit CPU so stable_logit(score lưu)"}


def open_split(config: dict[str, Any], split: str, root: Path = ROOT) -> tuple[list[dict[str, Any]], PackedManifestDataset]:
    paths = config[split]
    manifest, artifact = root / paths["manifest"], root / paths["artifact"]
    if split == "dev":
        shared.require("closed_test" not in str(manifest) and "closed_test" not in str(artifact), "Dev từ chối đường dẫn test.")
    require(sha256_file(manifest) == paths["manifest_sha256"], "Checksum manifest split không khớp.")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = "closed_dev" if split == "dev" else "closed_test"
    require(len(rows) == paths["samples"] and all(r["split"] == expected for r in rows), "Manifest sai số mẫu/split.")
    metadata = load_json(artifact / "metadata.json")
    require(metadata["split"] == expected and metadata["target_sample_rate"] == 16000, "Artifact sai split/sample rate.")
    require(sha256_file(artifact / "waveforms.f32") == metadata["waveforms_sha256"], "Checksum waveform artifact sai.")
    dataset = PackedManifestDataset(artifact, training=False, seed=2026, amplitude_policy="peak")
    try:
        require(len(dataset) == len(rows), "Manifest/packed lệch số mẫu.")
        for packed, row in zip(dataset.records, rows, strict=True):
            require((packed.file, packed.speaker_id, packed.utt_type, packed.binary_label) ==
                    (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])), "Manifest/packed lệch thứ tự/nhãn.")
    except Exception:
        dataset.close()
        raise
    return rows, dataset


class Models:
    def __init__(self, entries: list[dict[str, Any]], root: Path = ROOT) -> None:
        self.root = root
        self.entries = {x["model_id"]: x for x in entries}
        self.b0: dict[str, torch.nn.Module] = {}
        self.lfcc: torch.nn.Module | None = None
        self.xlsr: torch.nn.Module | None = None
        self.xlsr_id: str | None = None

    def load(self, name: str) -> torch.nn.Module:
        item = self.entries[name]
        if name in self.b0:
            return self.b0[name]
        if self.xlsr_id == name:
            return self.xlsr
        path = self.root / item["checkpoint"]
        require(sha256_file(path) == item["checkpoint_sha256"], "Checkpoint đã đổi sau registry: " + name)
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if item["family"] == "B0-v2":
            config = load_json(self.root / "configs/lfcc_lcnn.json")
            model = build_lcnn_from_config(config)
            model.load_state_dict(payload["model"], strict=True)
            if self.lfcc is None:
                self.lfcc = build_lfcc_transform(config["lfcc"]).eval()
            self.b0[name] = model
        else:
            self.xlsr = None
            self.xlsr_id = None
            gc.collect()
            weights = self.root / "data/raw/xlsr/wav2vec2-xls-r-300m"
            require({n: sha256_file(weights / n) for n in shared.PRETRAINED} == shared.PRETRAINED, "Checksum pretrained D021 sai.")
            layers = 0 if item["family"] == "X0" else 8
            model = XLSRBinaryClassifier(load_xlsr_backbone(weights, local_files_only=True),
                                        num_unfrozen_layers=layers, hidden_features=128, dropout=.2)
            saved = payload["model"]
            require(set(saved) == {n for n, p in model.named_parameters() if p.requires_grad}, "Danh sách trainable-only sai.")
            missing, unexpected = model.load_state_dict(saved, strict=False)
            require(not unexpected and set(missing) == {n for n, p in model.named_parameters() if not p.requires_grad},
                    "Checkpoint trainable-only thiếu/sai tham số.")
            require(sum(p.numel() for p in saved.values()) == (262401 if layers == 0 else 101032193), "Số tham số XLS-R sai.")
            self.xlsr, self.xlsr_id = model, name
        model.eval().requires_grad_(False)
        del payload
        gc.collect()
        return model

    def score(self, names: list[str], indices: list[int], rows: list[dict[str, Any]],
              dataset: PackedManifestDataset, condition: str) -> dict[str, list[dict[str, Any]]]:
        models = {name: self.load(name) for name in names}
        b0 = self.entries[names[0]]["family"] == "B0-v2"
        require(b0 or len(names) == 1, "Chỉ gom chung batch cho B0.")
        result = {name: [] for name in names}
        batch_size = 8 if b0 else 2
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                current = indices[start:start + batch_size]
                samples = [dataset[i] for i in current]
                waveform = torch.stack([shared.apply_condition(s["waveform"], condition, s["utt_type"]) for s in samples])
                inputs = (lfcc_features(self.lfcc, waveform).unsqueeze(1) if b0 else
                          apply_xlsr_waveform_normalization(waveform, "processor_zero_mean_unit_variance"))
                for name, model in models.items():
                    logits = model(inputs)
                    require(bool(torch.isfinite(logits).all()), "Logit không hữu hạn.")
                    for index, score, raw in zip(current, torch.sigmoid(logits).tolist(), logits.tolist(), strict=True):
                        row = rows[index]
                        result[name].append({"file": row["file"], "speaker_id": row["speaker_id"], "utt_type": row["utt_type"],
                            "label": int(row["binary_label"]), "score": float(score), "raw_logit": float(raw)})
                if (start + batch_size) % 200 == 0:
                    print(shared.canonical({"utc": utc(), "event": "progress", "models": names,
                          "condition": condition, "completed": start + batch_size, "samples": len(indices)}).decode(), flush=True)
        return result


def run(config: dict[str, Any], registry: dict[str, Any], split: str, selected: list[str],
        limit: int | None, conditions: list[str] | None, output: Path, root: Path = ROOT) -> dict[str, Any]:
    if split == "test":
        require(limit is None and conditions is None, "Test phải đủ split/điều kiện đã khóa, không dùng limit hoặc đổi conditions.")
        # Dùng đúng snapshot vừa kiểm digest, không dùng object do caller cung cấp.
        registry = authorize_test(config, selected, root)
    validate_registry(registry, config)
    require(limit is None or limit > 0, "Limit phải là số nguyên dương.")
    entries = [x for x in registry["entries"] if x["model_id"] in selected]
    require(len(entries) == len(set(selected)) and entries, "Danh sách mô hình không khớp registry.")
    require(all(x["status"] == "present" and x["claude_verified"] for x in entries), "Chỉ chấm mục present đã được Claude kiểm.")
    rows, dataset = open_split(config, split, root)
    began = time.perf_counter()
    report: dict[str, Any] = {"decision": "D032", "status": "running", "split": split, "use_test": split == "test",
        "scope": "dev-full, không phải test" if split == "dev" else "test-full, registry cố định",
        "created_utc": utc(), "artifact_samples": len(rows), "artifact_shards": len({r["shard"] for r in rows}),
        "registry_sha256": shared.digest(registry), "models": {}}
    try:
        models = Models(entries, root)
        groups = [[x["model_id"] for x in entries if x["family"] == "B0-v2"]]
        groups += [[x["model_id"]] for x in entries if x["family"] != "B0-v2"]
        for names in filter(None, groups):
            item = next(x for x in entries if x["model_id"] == names[0])
            indices = list(range(len(rows)))
            selection = "full_manifest" if limit is None else "first_manifest_rows"
            if split == "dev" and item["family"] != "B0-v2" and limit is not None:
                indices = shared.subset_indices(rows)
                require(shared.file_list_sha([rows[i]["file"] for i in indices]) == shared.SUBSET_SHA, "Tập con XLS-R lệch T19b.")
                report["xlsr_full_subset_sha256"] = shared.SUBSET_SHA
                selection = "first_rows_of_T19b_subset"
            if limit is not None:
                indices = indices[:limit]
            allowed = config["conditions_by_family"][item["family"]]
            requested = (allowed if split == "test" else ["original"]) if conditions is None else conditions
            require(requested and len(set(requested)) == len(requested) and set(requested).issubset(allowed), "Điều kiện không hợp lệ cho mô hình.")
            for condition in requested:
                started = time.perf_counter()
                print(shared.canonical({"utc": utc(), "event": "evaluate_start", "models": names, "condition": condition,
                                       "samples": len(indices), "selection": selection}).decode(), flush=True)
                results = models.score(names, indices, rows, dataset, condition)
                for name, records in results.items():
                    entry = next(x for x in entries if x["model_id"] == name)
                    value = {"records": records, "metrics": summarize(records, entry["decision_threshold"]),
                             "samples": len(records), "shards": len({rows[i]["shard"] for i in indices}),
                             "selection": selection, "files_sha256": shared.file_list_sha([r["file"] for r in records]),
                             "checkpoint_sha256": entry["checkpoint_sha256"], "seconds": time.perf_counter() - started}
                    if split == "dev" and condition == "original":
                        require(sha256_file(root / entry["source_report"]) == entry["source_report_sha256"],
                                "Report tham chiếu đã đổi sau registry.")
                        source = load_json(root / entry["source_report"])["controller"]["best"]["development"]["records"]
                        value["reproduction"] = reproduction(records, {r["file"]: r for r in source}, entry["decision_threshold"], entry["family"])
                    report["models"].setdefault(name, {})[condition] = value
                    atomic_json(output.parent / (output.stem + "-" + name + "-" + condition + "-scores.json"),
                                {"decision": "D032", "split": split, "model_id": name, "condition": condition,
                                 "checkpoint_sha256": entry["checkpoint_sha256"], **value})
                    atomic_json(output, report)
                    require(value.get("reproduction", {}).get("passed", True), "Tái tạo không đạt: " + name)
        report.update(status="complete", wall_seconds=time.perf_counter() - began)
        report["seed_summaries"] = {}
        for family in ("B0-v2", "X0", "X1"):
            items = [x for x in entries if x["family"] == family]
            eers = [report["models"][x["model_id"]].get("original", {}).get("metrics", {}).get("eer") for x in items]
            if len(eers) == 3 and all(x is not None for x in eers):
                report["seed_summaries"][family] = {"mean_eer": statistics.mean(eers), "sample_sd_eer": statistics.stdev(eers), "ddof": 1}
        atomic_json(output, report)
        return report
    except Exception as exc:
        report.update(status="blocked", error=str(exc), wall_seconds=time.perf_counter() - began)
        atomic_json(output, report)
        raise
    finally:
        dataset.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="+", choices=shared.CONDITIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        validate_config(config)
        if args.split == "test":
            require(config["test_gate_open"] is True, "Cổng test đang đóng; không mở dữ liệu test.")
            require(args.registry is None or args.registry.resolve() == (ROOT / config["registry_path"]).resolve(),
                    "Test chỉ dùng registry chính thức đã khóa.")
            registry_path = ROOT / config["registry_path"]
        else:
            registry_path = args.registry or DRAFT
        registry = load_json(registry_path)
        selected = args.models or [x["model_id"] for x in registry["entries"] if x["status"] == "present" and x["claude_verified"]]
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        report = run(config, registry, args.split, selected, args.limit, args.conditions, args.output)
        print(shared.canonical({"status": report["status"], "models": list(report["models"]), "seconds": report["wall_seconds"]}).decode())
        return 0
    except Exception as exc:
        print("LỖI đánh giá: " + str(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
