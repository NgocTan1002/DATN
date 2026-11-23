"""T23: đánh giá checkpoint best; nhánh test có gate/registry/T22/khóa một lần (D032 và X1R D035)."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
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
from scripts.build_model_registry import (
    CONFIG,
    DRAFT,
    LOWPASS_7K_COEFFICIENT_SHA256,
    atomic_json,
    load_json,
    protocol_sha256,
    require,
    sha256_file,
    utc,
    validate_config,
    validate_registry,
)
from src.data.packed import PackedManifestDataset
from src.data.xlsr import apply_xlsr_waveform_normalization
from src.metrics.bootstrap import confusion_at_threshold, per_utt_type_eer
from src.metrics.classification import macro_f1_at_threshold, roc_auc_score
from src.metrics.eer import equal_error_rate
from src.models.lfcc_lcnn import build_lcnn_from_config, build_lfcc_transform, lfcc_features
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone

X1R_DRAFT = ROOT / "reports/model_registry_x1r_draft.json"
PRE_TEST_CHECKS = (
    "entrypoint_dev_claude_verified",
    "bootstrap_dev_claude_verified",
    "t22_claude_verified",
    "test_dataset_inventory_checked",
)


def verify_lowpass_7k_filter(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Kiểm tra hệ số và tham số bộ lọc thông thấp 7 kHz theo D035."""
    params = shared.filter_parameters("lowpass_7k")
    require(
        params.get("coefficient_float32_sha256") == LOWPASS_7K_COEFFICIENT_SHA256,
        "SHA-256 hệ số bộ lọc lowpass_7k không khớp D035.",
    )
    if config is not None and config.get("decision") == "D035":
        require(params == config["input_condition"]["filter"], "Tham số bộ lọc lowpass_7k lệch config X1R.")
    return params


def authorize_test(config: dict[str, Any], selected_ids: list[str], root: Path = ROOT) -> dict[str, Any]:
    """Kiểm gate hoàn toàn trước khi chạm đường dẫn manifest/packed test."""
    require(config.get("test_gate_open") is True, "Cổng test đang đóng; không mở dữ liệu test.")
    validate_config(config)
    decision = config["decision"]
    registry_path = root / config["registry_path"]
    require(registry_path.is_file(), "Thiếu registry chính thức.")
    content = registry_path.read_bytes()
    require(isinstance(config.get("registry_sha256"), str) and hashlib.sha256(content).hexdigest() == config["registry_sha256"],
            "SHA-256 registry không khớp config.")
    registry = json.loads(content)
    validate_registry(registry, config, complete=True)
    require(bool(selected_ids) and len(set(selected_ids)) == len(selected_ids) and
            set(selected_ids).issubset({x["model_id"] for x in registry["entries"]}), "Danh sách mục chấm test không hợp lệ.")
    if decision == "D035":
        d032_path = root / "configs/test_evaluation.json"
        require(d032_path.is_file(), "Thiếu cấu hình D032 để kiểm cổng test X1R.")
        d032_config = load_json(d032_path)
        require(d032_config.get("decision") == "D032", "Cấu hình kiểm cổng X1R phải là D032.")
        validate_config(d032_config)
        require(d032_config.get("test_gate_open") is True, "Cổng test D032 đang đóng; không mở dữ liệu test.")
        require(all(d032_config.get("pre_test_checks", {}).get(key) is True for key in PRE_TEST_CHECKS),
                "Thiếu xác nhận kiểm dev/T22/inventory theo D032 mục8.")
    else:
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
        require(all(config.get("pre_test_checks", {}).get(key) is True for key in PRE_TEST_CHECKS),
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
    saved_f64 = np.asarray(saved, dtype=np.float64)
    old_logits = shared.stable_logit(saved_f64)
    delta = np.abs(np.asarray([r["raw_logit"] for r in records], dtype=np.float64) - old_logits)
    distances = np.abs(old_logits - shared.stable_logit([threshold])[0])
    if family == "X1R":
        require(bool((saved_f64.astype(np.float32).astype(np.float64) == saved_f64).all()),
                "Score tham chiếu X1R không biểu diễn chính xác bằng float32; dừng D034.")
        probabilities = saved_f64.astype(np.float32)
        below = np.nextafter(probabilities, np.float32(0), dtype=np.float32).astype(np.float64)
        above = np.nextafter(probabilities, np.float32(1), dtype=np.float32).astype(np.float64)
        gaps = np.maximum(np.abs(shared.stable_logit(above) - old_logits),
                          np.abs(old_logits - shared.stable_logit(below)))
        excess = delta - (.05 + gaps)
        return {"passed": bool((excess <= 0).all() and (distances[flips] <= .05).all()),
                "criterion": "D034", "samples": len(records),
                "max_abs_logit_difference": float(delta.max()), "median_abs_logit_difference": float(np.median(delta)),
                "max_float32_gap_logit": float(gaps.max()), "max_excess_over_tolerance": float(excess.max()),
                "samples_over_0_05_without_gap": int((delta > .05).sum()),
                "decision_flips": int(flips.sum()), "flips_outside_0_05_margin": int((distances[flips] > .05).sum()),
                "logit_source": "raw_logit CPU so stable_logit(score lưu)"}
    return {"passed": bool(delta.max() <= .05 and (distances[flips] <= .05).all()), "samples": len(records),
            "max_abs_logit_difference": float(delta.max()), "median_abs_logit_difference": float(np.median(delta)),
            "decision_flips": int(flips.sum()), "flips_outside_0_05_margin": int((distances[flips] > .05).sum()),
            "logit_source": "raw_logit CPU so stable_logit(score lưu)"}


def open_split(config: dict[str, Any], split: str, root: Path = ROOT) -> tuple[list[dict[str, Any]], PackedManifestDataset]:
    paths = config[split]
    return open_split_paths(split, root / paths["manifest"], root / paths["artifact"], paths["manifest_sha256"], paths["samples"])


def open_split_paths(split: str, manifest: Path, artifact: Path, manifest_sha256: str,
                     samples: int) -> tuple[list[dict[str, Any]], PackedManifestDataset]:
    """Mở split tại đường dẫn tường minh (dùng chung cho đường cũ và đường cấp phép)."""
    if split == "dev":
        shared.require("closed_test" not in str(manifest) and "closed_test" not in str(artifact), "Dev từ chối đường dẫn test.")
    require(sha256_file(manifest) == manifest_sha256, "Checksum manifest split không khớp.")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = "closed_dev" if split == "dev" else "closed_test"
    require(len(rows) == samples and all(r["split"] == expected for r in rows), "Manifest sai số mẫu/split.")
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
    def __init__(self, entries: list[dict[str, Any]], root: Path = ROOT, *, device: str = "cpu",
                 checkpoint_root: Path | None = None, weights_dir: Path | None = None) -> None:
        self.root = root
        self.device = device
        self.checkpoint_root = checkpoint_root or root
        self.weights_dir = weights_dir or root / "data/raw/xlsr/wav2vec2-xls-r-300m"
        self.entries = {x["model_id"]: x for x in entries}
        self.b0: dict[str, torch.nn.Module] = {}
        self.lfcc: torch.nn.Module | None = None
        self.xlsr: torch.nn.Module | None = None
        self.xlsr_id: str | None = None

    def load(self, name: str) -> torch.nn.Module:
        item = self.entries[name]
        require(item["family"] != "B0-v2" or self.device == "cpu", "B0-v2 chỉ được chấm trên CPU (D037 mục 3).")
        if name in self.b0:
            return self.b0[name]
        if self.xlsr_id == name:
            return self.xlsr
        path = self.checkpoint_root / item["checkpoint"]
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
            require(item["family"] in ("X0", "X1", "X1R"), "Họ mô hình không hợp lệ: " + str(item["family"]))
            self.xlsr = None
            self.xlsr_id = None
            gc.collect()
            weights = self.weights_dir
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
        if self.device != "cpu":
            model.to(self.device)
        del payload
        gc.collect()
        return model

    def score(self, names: list[str], indices: list[int], rows: list[dict[str, Any]],
              dataset: PackedManifestDataset, condition: str) -> dict[str, list[dict[str, Any]]]:
        models = {name: self.load(name) for name in names}
        families = {self.entries[name]["family"] for name in names}
        require(len(families) == 1, "Không trộn họ mô hình trong cùng batch.")
        family = next(iter(families))
        b0 = family == "B0-v2"
        require(b0 or len(names) == 1, "Chỉ gom chung batch cho B0.")
        if family == "X1R":
            require(condition == "lowpass_7k", "X1R chỉ chấp nhận điều kiện lowpass_7k.")
            verify_lowpass_7k_filter()
        result = {name: [] for name in names}
        batch_size = 8 if b0 else 2
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                current = indices[start:start + batch_size]
                samples = [dataset[i] for i in current]
                waveform = torch.stack([shared.apply_condition(s["waveform"], condition, s["utt_type"]) for s in samples])
                inputs = (lfcc_features(self.lfcc, waveform).unsqueeze(1) if b0 else
                          apply_xlsr_waveform_normalization(waveform, "processor_zero_mean_unit_variance"))
                if not b0 and self.device != "cpu":
                    inputs = inputs.to(self.device)
                for name, model in models.items():
                    logits = model(inputs).cpu()
                    require(bool(torch.isfinite(logits).all()), "Logit không hữu hạn.")
                    for index, score, raw in zip(current, torch.sigmoid(logits).tolist(), logits.tolist(), strict=True):
                        row = rows[index]
                        result[name].append({"file": row["file"], "speaker_id": row["speaker_id"], "utt_type": row["utt_type"],
                            "label": int(row["binary_label"]), "score": float(score), "raw_logit": float(raw)})
                if (start + batch_size) % 200 == 0:
                    print(shared.canonical({"utc": utc(), "event": "progress", "models": names,
                          "condition": condition, "completed": start + batch_size, "samples": len(indices)}).decode(), flush=True)
        return result


def choose_indices(split: str, family: str, rows: list[dict[str, Any]], limit: int | None) -> tuple[list[int], str, bool]:
    """Chọn chỉ số mẫu; dùng chung cho đường cũ và đường cấp phép. Trả về (chỉ số, selection, dùng tập con T19b)."""
    indices = list(range(len(rows)))
    selection = "full_manifest" if limit is None else "first_manifest_rows"
    subset = False
    if split == "dev" and family != "B0-v2" and limit is not None:
        indices = shared.subset_indices(rows)
        require(shared.file_list_sha([rows[i]["file"] for i in indices]) == shared.SUBSET_SHA, "Tập con XLS-R lệch T19b.")
        selection = "first_rows_of_T19b_subset"
        subset = True
    if limit is not None:
        indices = indices[:limit]
    return indices, selection, subset


def run(config: dict[str, Any], registry: dict[str, Any], split: str, selected: list[str],
        limit: int | None, conditions: list[str] | None, output: Path, root: Path = ROOT,
        device: str = "cpu") -> dict[str, Any]:
    if split == "test":
        require(limit is None and conditions is None, "Test phải đủ split/điều kiện đã khóa, không dùng limit hoặc đổi conditions.")
        # Dùng đúng snapshot vừa kiểm digest, không dùng object do caller cung cấp.
        registry = authorize_test(config, selected, root)
    validate_registry(registry, config)
    decision = config["decision"]
    if decision == "D035":
        verify_lowpass_7k_filter(config)
    require(limit is None or limit > 0, "Limit phải là số nguyên dương.")
    entries = [x for x in registry["entries"] if x["model_id"] in selected]
    require(len(entries) == len(set(selected)) and entries, "Danh sách mô hình không khớp registry.")
    check_device(device, {x["family"] for x in entries})
    allowed_families = {"X1R"} if decision == "D035" else {"B0-v2", "X0", "X1"}
    require(all(x["family"] in allowed_families for x in entries), "Họ mô hình không khớp quyết định của config.")
    require(all(x["status"] == "present" and x["claude_verified"] for x in entries), "Chỉ chấm mục present đã được Claude kiểm.")
    rows, dataset = open_split(config, split, root)
    began = time.perf_counter()
    report: dict[str, Any] = {"decision": decision, "status": "running", "split": split, "use_test": split == "test",
        "scope": "dev-full, không phải test" if split == "dev" else "test-full, registry cố định",
        "created_utc": utc(), "artifact_samples": len(rows), "artifact_shards": len({r["shard"] for r in rows}),
        "registry_sha256": shared.digest(registry), "models": {}}
    if decision == "D035":
        report["input_condition"] = config["input_condition"]
    try:
        models = Models(entries, root, device=device)
        groups = [[x["model_id"] for x in entries if x["family"] == "B0-v2"]]
        groups += [[x["model_id"]] for x in entries if x["family"] != "B0-v2"]
        for names in filter(None, groups):
            item = next(x for x in entries if x["model_id"] == names[0])
            indices, selection, subset = choose_indices(split, item["family"], rows, limit)
            if subset:
                report["xlsr_full_subset_sha256"] = shared.SUBSET_SHA
            allowed = config["conditions_by_family"][item["family"]]
            default_conditions = ["lowpass_7k"] if item["family"] == "X1R" else (allowed if split == "test" else ["original"])
            requested = default_conditions if conditions is None else conditions
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
                    reference_condition = "lowpass_7k" if entry["family"] == "X1R" else "original"
                    if split == "dev" and condition == reference_condition:
                        require(sha256_file(root / entry["source_report"]) == entry["source_report_sha256"],
                                "Report tham chiếu đã đổi sau registry.")
                        source = load_json(root / entry["source_report"])["controller"]["best"]["development"]["records"]
                        value["reproduction"] = reproduction(records, {r["file"]: r for r in source}, entry["decision_threshold"], entry["family"])
                    report["models"].setdefault(name, {})[condition] = value
                    atomic_json(output.parent / (output.stem + "-" + name + "-" + condition + "-scores.json"),
                                {"decision": decision, "split": split, "model_id": name, "condition": condition,
                                 "checkpoint_sha256": entry["checkpoint_sha256"], **value})
                    atomic_json(output, report)
                    require(value.get("reproduction", {}).get("passed", True), "Tái tạo không đạt: " + name)
        report.update(status="complete", wall_seconds=time.perf_counter() - began)
        report["seed_summaries"] = {}
        for family in ("B0-v2", "X0", "X1", "X1R"):
            items = [x for x in entries if x["family"] == family]
            base_condition = "lowpass_7k" if family == "X1R" else "original"
            eers = [report["models"][x["model_id"]].get(base_condition, {}).get("metrics", {}).get("eer") for x in items]
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


PERMIT_SCHEMA_VERSION = 1
COLAB_ENTRY_CANDIDATES: tuple[Path, ...] = (
    ROOT / "colab/xlsr_colab_entry.py",
    Path(r"C:\Users\Admin\Documents\DoAnTTNT-xlsr\colab\xlsr_colab_entry.py"),
)
_PINNED_PATTERN = re.compile(r"^EXPECTED_(TORCH_FULL|CUDA_VERSION|TRANSFORMERS_VERSION|NUMPY_VERSION)\s*=\s*'([^']+)'", re.MULTILINE)
# Chỉ để kiểm thử chèn phiên bản/GPU giả; mặc định rỗng nghĩa là đọc môi trường thật.
_DEVICE_OVERRIDES: dict[str, Any] = {}


def load_colab_pinned_versions(path: Path | None = None) -> dict[str, str]:
    """Đọc hằng phiên bản đã khóa từ entrypoint Colab (không gõ tay)."""
    candidates = [path] if path is not None else list(COLAB_ENTRY_CANDIDATES)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            found = {key: value for key, value in _PINNED_PATTERN.findall(candidate.read_text(encoding="utf-8"))}
            require(set(found) == {"TORCH_FULL", "CUDA_VERSION", "TRANSFORMERS_VERSION", "NUMPY_VERSION"},
                    "Entrypoint Colab thiếu hằng phiên bản đã khóa.")
            return {"torch": found["TORCH_FULL"], "cuda": found["CUDA_VERSION"],
                    "transformers": found["TRANSFORMERS_VERSION"], "numpy": found["NUMPY_VERSION"]}
    raise ValueError("Không tìm thấy xlsr_colab_entry.py để đọc phiên bản thư viện đã khóa.")


def library_versions() -> dict[str, str]:
    import transformers
    return {"torch": torch.__version__, "cuda": str(torch.version.cuda), "transformers": transformers.__version__,
            "numpy": np.__version__}


def check_device(device: str, families: set[str], *, pinned: dict[str, str] | None = None,
                 versions: dict[str, str] | None = None, cuda_available: bool | None = None,
                 gpu_name: str | None = None) -> dict[str, Any]:
    """Kiểm thiết bị chấm: B0-v2 chỉ CPU; cuda cần GPU và đúng phiên bản thư viện đã khóa (D037 mục 3)."""
    require(device in ("cpu", "cuda"), "Thiết bị phải là cpu hoặc cuda.")
    if device == "cpu":
        return {"device": "cpu", "gpu_name": None, "versions": library_versions() if versions is None else versions}
    require("B0-v2" not in families, "B0-v2 chỉ được chấm trên CPU (D037 mục 3); từ chối device=cuda.")
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    require(available, "device=cuda nhưng không có GPU CUDA khả dụng; dừng.")
    name = gpu_name if gpu_name is not None else torch.cuda.get_device_name(0)
    expected = load_colab_pinned_versions() if pinned is None else pinned
    actual = library_versions() if versions is None else versions
    for key in ("torch", "transformers", "numpy", "cuda"):
        require(actual.get(key) == expected[key], f"Phiên bản {key} ({actual.get(key)}) lệch bản đã khóa ({expected[key]}); dừng.")
    return {"device": "cuda", "gpu_name": name, "versions": actual}


def permit_digest(permit: dict[str, Any]) -> str:
    """SHA-256 nội dung chuẩn hóa của cấp phép, không gồm chính trường permit_sha256."""
    return hashlib.sha256(shared.canonical({k: v for k, v in permit.items() if k != "permit_sha256"})).hexdigest()


def split_artifact_hashes(config: dict[str, Any], split: str, root: Path = ROOT) -> dict[str, Any]:
    """SHA ba file artifact của split. Test: lấy từ reports/pre_test_inventory.json, không đọc thư mục test."""
    paths = config[split]
    if split == "test":
        files = load_json(root / "reports/pre_test_inventory.json")["local_test_inventory"]["files"]
        manifest, index, waveforms = (files[key]["sha256"] for key in ("manifest", "index", "waveforms"))
        require(manifest == paths["manifest_sha256"], "SHA manifest test trong inventory lệch config.")
    else:
        metadata = load_json(root / paths["artifact"] / "metadata.json")
        manifest, index, waveforms = paths["manifest_sha256"], metadata["index_sha256"], metadata["waveforms_sha256"]
        require(metadata["source_manifest_sha256"] == manifest, "SHA manifest dev trong artifact lệch config.")
    return {"manifest_name": Path(paths["manifest"]).name, "artifact_name": Path(paths["artifact"]).name,
            "manifest_sha256": manifest, "index_sha256": index, "waveforms_sha256": waveforms, "samples": paths["samples"]}


def write_permit(config: dict[str, Any], split: str, selected: list[str] | None, conditions: list[str] | None,
                 output: Path, root: Path = ROOT) -> dict[str, Any]:
    """Ghi file cấp phép trên máy (D037 mục 4). Test: kiểm cổng + tạo khóa, không chấm. Dev: không cần cổng, không khóa."""
    validate_config(config)
    require(split in ("dev", "test"), "Split phải là dev hoặc test.")
    require(not output.exists(), "File cấp phép đã tồn tại; không ghi đè.")
    registry_path = root / config["registry_path"]
    if split == "test":
        names = selected
        if names is None:
            require(config.get("test_gate_open") is True, "Cổng test đang đóng; không mở dữ liệu test.")
            require(registry_path.is_file(), "Thiếu registry chính thức.")
            names = [x["model_id"] for x in load_json(registry_path)["entries"]]
        registry = authorize_test(config, names, root)
    else:
        require(registry_path.is_file(), "Thiếu registry chính thức.")
        require(isinstance(config.get("registry_sha256"), str) and sha256_file(registry_path) == config["registry_sha256"],
                "SHA-256 registry không khớp config.")
        registry = load_json(registry_path)
        validate_registry(registry, config, complete=True)
        names = selected or [x["model_id"] for x in registry["entries"]]
    by_id = {x["model_id"]: x for x in registry["entries"]}
    require(bool(names) and len(set(names)) == len(names) and set(names).issubset(by_id), "Danh sách mô hình cấp phép không hợp lệ.")
    allowed_families = {"X1R"} if config["decision"] == "D035" else {"B0-v2", "X0", "X1"}
    models = []
    for name in names:
        entry = by_id[name]
        require(entry["family"] in allowed_families and entry["status"] == "present" and entry["claude_verified"],
                "Chỉ cấp phép mục present đã được Claude kiểm: " + name)
        allowed = config["conditions_by_family"][entry["family"]]
        default = ["lowpass_7k"] if entry["family"] == "X1R" else (allowed if split == "test" else ["original"])
        requested = default if conditions is None else conditions
        require(bool(requested) and len(set(requested)) == len(requested) and set(requested).issubset(allowed),
                "Điều kiện cấp phép không thuộc conditions_by_family: " + name)
        models.append({"model_id": name, "family": entry["family"], "checkpoint": entry["checkpoint"],
                       "checkpoint_sha256": entry["checkpoint_sha256"], "conditions": list(requested)})
    permit: dict[str, Any] = {"schema_version": PERMIT_SCHEMA_VERSION, "split": split, "decision": config["decision"],
        "config_sha256": protocol_sha256(config), "registry_sha256": sha256_file(registry_path),
        "models": models, "artifacts": split_artifact_hashes(config, split, root), "created_utc": utc()}
    if split == "test":
        locks = root / config["locks_dir"]
        permit["lock_sha256"] = {name: sha256_file(locks / (name + ".lock")) for name in names}
    permit["permit_sha256"] = permit_digest(permit)
    atomic_json(output, permit)
    return permit


def verify_permit(permit: dict[str, Any], config: dict[str, Any], registry_path: Path, checkpoint_root: Path,
                  data_root: Path, selected: list[str] | None, conditions: list[str] | None, device: str,
                  limit: int | None) -> dict[str, Any]:
    """Kiểm mọi điều kiện của cấp phép TRƯỚC khi mở dữ liệu; lệch bất kỳ điều gì thì ném ValueError."""
    require(permit.get("schema_version") == PERMIT_SCHEMA_VERSION and permit.get("split") in ("dev", "test"),
            "Cấp phép sai schema hoặc split.")
    require(isinstance(permit.get("permit_sha256"), str) and permit_digest(permit) == permit["permit_sha256"],
            "permit_sha256 không khớp nội dung cấp phép.")
    split = permit["split"]
    require(split == "dev" or limit is None, "Test phải chấm đủ split, không dùng limit.")
    require(limit is None or limit > 0, "Limit phải là số nguyên dương.")
    validate_config(config)
    require(permit["decision"] == config["decision"] and permit["config_sha256"] == protocol_sha256(config),
            "Cấp phép không khớp giao thức của config.")
    require(registry_path.is_file() and sha256_file(registry_path) == permit["registry_sha256"],
            "SHA-256 registry không khớp cấp phép.")
    registry = load_json(registry_path)
    validate_registry(registry, config, complete=True)
    by_id = {x["model_id"]: x for x in registry["entries"]}
    permitted = {m["model_id"]: m for m in permit["models"]}
    require(bool(permitted) and set(permitted).issubset(by_id), "Cấp phép chứa mô hình ngoài registry.")
    names = list(permitted) if selected is None else selected
    require(bool(names) and len(set(names)) == len(names) and set(names).issubset(permitted),
            "Mô hình yêu cầu nằm ngoài danh sách cấp phép.")
    if split == "test":
        require(set(permit.get("lock_sha256", {})) >= set(permitted), "Cấp phép test thiếu SHA khóa của mô hình.")
    plan: list[tuple[dict[str, Any], str]] = []
    for name in names:
        item = permitted[name]
        require(item["checkpoint_sha256"] == by_id[name]["checkpoint_sha256"] and item["checkpoint"] == by_id[name]["checkpoint"],
                "Checkpoint trong cấp phép lệch registry: " + name)
        requested = item["conditions"] if conditions is None else conditions
        require(bool(requested) and set(requested).issubset(item["conditions"]),
                "Điều kiện yêu cầu nằm ngoài danh sách cấp phép: " + name)
        plan.extend((by_id[name], c) for c in requested)
    check_device(device, {e["family"] for e, _ in plan}, **_DEVICE_OVERRIDES)
    for name in names:
        path = checkpoint_root / permitted[name]["checkpoint"]
        require(path.is_file() and sha256_file(path) == permitted[name]["checkpoint_sha256"],
                "Checkpoint đã đổi so với cấp phép: " + name)
    art = permit["artifacts"]
    manifest, artifact = data_root / art["manifest_name"], data_root / art["artifact_name"]
    require(manifest.is_file() and sha256_file(manifest) == art["manifest_sha256"], "Manifest split lệch cấp phép.")
    require((artifact / "index.csv").is_file() and sha256_file(artifact / "index.csv") == art["index_sha256"],
            "index.csv lệch cấp phép.")
    require((artifact / "waveforms.f32").is_file() and sha256_file(artifact / "waveforms.f32") == art["waveforms_sha256"],
            "waveforms.f32 lệch cấp phép.")
    return {"registry": registry, "plan": plan, "manifest": manifest, "artifact": artifact}


def result_path(output_dir: Path, split: str, model_id: str, condition: str) -> Path:
    return output_dir / f"{split}__{model_id}__{condition}.json"


def is_complete_result(path: Path, permit_sha256: str) -> bool:
    """Kết quả hoàn chỉnh cùng cấp phép thì bỏ qua (D037 mục 5); cấp phép khác thì không được ghi đè."""
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(value, dict) or value.get("complete") is not True:
        return False
    require(value.get("permit_sha256") == permit_sha256,
            "Đã có kết quả hoàn chỉnh theo cấp phép khác; không ghi đè: " + path.name)
    return True


def run_with_permit(permit_path: Path, config: dict[str, Any], registry_path: Path, data_root: Path,
                    checkpoint_root: Path, device: str, output_dir: Path, *, models: list[str] | None = None,
                    conditions: list[str] | None = None, limit: int | None = None, weights_dir: Path | None = None,
                    reference_root: Path = ROOT, root: Path = ROOT,
                    opener: Callable[..., Any] = open_split_paths) -> dict[str, list[str]]:
    """Chấm bằng cấp phép (D037 mục 4–5): không tạo khóa, không đòi test_gate_open cục bộ."""
    require(permit_path.is_file(), "Thiếu file cấp phép; từ chối chạy.")
    permit = load_json(permit_path)
    require(isinstance(permit, dict), "File cấp phép không hợp lệ.")
    context = verify_permit(permit, config, registry_path, checkpoint_root, data_root, models, conditions, device, limit)
    split, permit_sha = permit["split"], permit["permit_sha256"]
    pending = [(e, c) for e, c in context["plan"]
               if not is_complete_result(result_path(output_dir, split, e["model_id"], c), permit_sha)]
    skipped = [f"{e['model_id']}|{c}" for e, c in context["plan"] if (e, c) not in pending]
    if not pending:
        return {"completed": [], "skipped": skipped}
    env = check_device(device, {e["family"] for e, _ in pending}, **_DEVICE_OVERRIDES)
    art = permit["artifacts"]
    rows, dataset = opener(split, context["manifest"], context["artifact"], art["manifest_sha256"], art["samples"])
    done: list[str] = []
    try:
        scorer = Models([e for e, _ in pending], root, device=device, checkpoint_root=checkpoint_root,
                        weights_dir=weights_dir or data_root / "wav2vec2-xls-r-300m")
        for entry, condition in pending:
            name = entry["model_id"]
            started = time.perf_counter()
            indices, selection, _ = choose_indices(split, entry["family"], rows, limit)
            print(shared.canonical({"utc": utc(), "event": "evaluate_start", "models": [name], "condition": condition,
                                    "samples": len(indices), "selection": selection, "device": device}).decode(), flush=True)
            path = result_path(output_dir, split, name, condition)
            atomic_json(path, {"complete": False, "permit_sha256": permit_sha, "model_id": name, "condition": condition,
                               "split": split, "started_utc": utc()})
            records = scorer.score([name], indices, rows, dataset, condition)[name]
            value: dict[str, Any] = {"schema_version": 1, "complete": False, "permit_sha256": permit_sha, "split": split,
                "decision": permit["decision"], "model_id": name, "condition": condition, "device": env["device"],
                "gpu_name": env["gpu_name"], "versions": env["versions"], "checkpoint_sha256": entry["checkpoint_sha256"],
                "records": records, "metrics": summarize(records, entry["decision_threshold"]), "samples": len(records),
                "selection": selection, "files_sha256": shared.file_list_sha([r["file"] for r in records]),
                "seconds": time.perf_counter() - started, "completed_utc": utc()}
            reference_condition = "lowpass_7k" if entry["family"] == "X1R" else "original"
            source = reference_root / entry["source_report"]
            if split == "dev" and condition == reference_condition and source.is_file() and \
                    sha256_file(source) == entry["source_report_sha256"]:
                reference = load_json(source)["controller"]["best"]["development"]["records"]
                value["reproduction"] = reproduction(records, {r["file"]: r for r in reference},
                                                     entry["decision_threshold"], entry["family"])
                if not value["reproduction"]["passed"]:
                    atomic_json(path, value)
                    raise ValueError("Tái tạo không đạt: " + name)
            value["complete"] = True
            atomic_json(path, value)
            done.append(f"{name}|{condition}")
    finally:
        dataset.close()
    return {"completed": done, "skipped": skipped}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="+", choices=shared.CONDITIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--write-permit", type=Path, help="Ghi file cấp phép trên máy (cần --split).")
    parser.add_argument("--permit", type=Path, help="Chấm bằng file cấp phép (cần --data-root, --checkpoint-root, --output-dir).")
    parser.add_argument("--data-root", type=Path, help="Thư mục chứa manifest .csv và thư mục packed của split.")
    parser.add_argument("--checkpoint-root", type=Path, help="Thư mục gốc để ghép đường dẫn checkpoint trong registry.")
    parser.add_argument("--weights-dir", type=Path, help="Thư mục trọng số XLS-R (mặc định <data-root>/wav2vec2-xls-r-300m).")
    parser.add_argument("--reference-root", type=Path, default=ROOT, help="Gốc chứa report nguồn để kiểm tái tạo dev.")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        validate_config(config)
        if args.permit is not None:
            require(args.write_permit is None and args.split is None and args.output is None,
                    "Chế độ --permit không dùng chung với --write-permit/--split/--output.")
            require(args.data_root is not None and args.checkpoint_root is not None and args.output_dir is not None,
                    "Chế độ --permit cần --data-root, --checkpoint-root và --output-dir.")
            require(args.registry is not None, "Chế độ --permit cần --registry (registry chính thức).")
            if args.device == "cpu":
                torch.set_num_threads(8)
                torch.set_num_interop_threads(1)
            result = run_with_permit(args.permit, config, args.registry, args.data_root, args.checkpoint_root, args.device,
                                     args.output_dir, models=args.models, conditions=args.conditions, limit=args.limit,
                                     weights_dir=args.weights_dir, reference_root=args.reference_root)
            print(shared.canonical({"status": "complete", **result}).decode())
            return 0
        require(args.split is not None, "Cần --split hoặc --permit.")
        if args.write_permit is not None:
            permit = write_permit(config, args.split, args.models, args.conditions, args.write_permit)
            print(shared.canonical({"status": "permit_written", "split": permit["split"], "permit_sha256": permit["permit_sha256"],
                                    "models": [m["model_id"] for m in permit["models"]]}).decode())
            return 0
        require(args.output is not None, "Đường cũ cần --output.")
        if args.split == "test":
            require(config["test_gate_open"] is True, "Cổng test đang đóng; không mở dữ liệu test.")
            require(args.registry is None or args.registry.resolve() == (ROOT / config["registry_path"]).resolve(),
                    "Test chỉ dùng registry chính thức đã khóa.")
            registry_path = ROOT / config["registry_path"]
        else:
            default_draft = X1R_DRAFT if config["decision"] == "D035" else DRAFT
            registry_path = args.registry or default_draft
        registry = load_json(registry_path)
        selected = args.models or [x["model_id"] for x in registry["entries"] if x["status"] == "present" and x["claude_verified"]]
        if args.device == "cpu":
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
        report = run(config, registry, args.split, selected, args.limit, args.conditions, args.output, device=args.device)
        print(shared.canonical({"status": report["status"], "models": list(report["models"]), "seconds": report["wall_seconds"]}).decode())
        return 0
    except Exception as exc:
        print("LỖI đánh giá: " + str(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
