"""SC-02: kiểm mô hình B0 v2, X1 cuối và X1R có dựa vào khoảng lặng, nền nhiễu (và DC với B0) trên dev."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0")

import numpy as np
import torch

from scripts import audit_shortcut_features as sc01
from scripts import probe_shortcut_dev as shared
from scripts import probe_shortcut_x1_final_d034 as d034
from src.data.amplitude import apply_amplitude_policy
from src.data.packed import PackedManifestDataset
from src.data.segment import fixed_length_segment
from src.data.xlsr import apply_xlsr_waveform_normalization
from src.metrics.bootstrap import confusion_at_threshold, fast_eer
from src.models.lfcc_lcnn import build_lcnn_from_config, build_lfcc_transform, lfcc_features
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone

CACHE_DIR = ROOT / ".cache/sc02"
REGISTRY_V1_PATH = ROOT / "reports/model_registry_v1_draft.json"
REGISTRY_X1R_PATH = ROOT / "reports/model_registry_x1r_draft.json"
PRIOR_T22_PATH = ROOT / "reports/shortcut_probe_dev_x1_final.json"
DEV_MANIFEST_PATH = ROOT / "data/manifests/closed_dev_full_v1.csv"
DEV_ARTIFACT_DIR = ROOT / "data/processed/packed/closed_dev_full_v1"
LFCC_CONFIG_PATH = ROOT / "configs/lfcc_lcnn.json"
AUDIO_CONFIG_PATH = ROOT / "configs/audio.json"
WEIGHTS_DIR = ROOT / "data/raw/xlsr/wav2vec2-xls-r-300m"
REPORT_JSON = ROOT / "reports/shortcut_probe_sc02.json"
REPORT_MD = ROOT / "reports/shortcut_probe_sc02.md"

MANIFEST_SHA = shared.MANIFEST_SHA
SUBSET_SHA = shared.SUBSET_SHA
PRETRAINED_SHA = {
    "pytorch_model.bin": "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0",
    "config.json": "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
    "preprocessor_config.json": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
}

SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_000
FRAME_LEN = 400
FRAME_HOP = 160
SILENCE_OFFSET_DB = 40.0
NOISE_OFFSET_DB = 45.0
NOISE_SEED_BASE = 2026
BLOCK_SIZE = 200
B0_MAX_PROB_DIFF = 1e-4

MODEL_SPECS = (
    {
        "model_id": "B0-v2-seed2026",
        "family": "B0-v2",
        "registry": "v1",
        "conditions": ("trim_edges", "noise_floor", "dc_remove"),
        "batch_size": 8,
    },
    {
        "model_id": "X1-L8-lr3e-5-seed2026",
        "family": "X1",
        "registry": "v1",
        "conditions": ("trim_edges", "noise_floor"),
        "batch_size": 2,
    },
    {
        "model_id": "X1-L8-lr3e-5-lp7k-seed2026",
        "family": "X1R",
        "registry": "x1r",
        "conditions": ("trim_edges", "noise_floor"),
        "batch_size": 2,
    },
)
ALL_CONDITIONS = ("original", "trim_edges", "noise_floor", "dc_remove")
LEVEL_ORDER = {"yếu": 0, "trung bình": 1, "mạnh": 2}
require = shared.require


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def reject_test_path(path: Path) -> None:
    require("closed_test" not in str(path).lower(), "SC-02 từ chối đọc đường dẫn dữ liệu test.")


def noise_seed_for_file(file_name: str) -> int:
    """Seed tất định theo file: 2026 + 32 bit đầu của SHA-256 tên file."""
    digest = hashlib.sha256(file_name.encode("utf-8")).digest()
    return NOISE_SEED_BASE + int.from_bytes(digest[:4], "big")


def apply_full_transform(
    raw_waveform: torch.Tensor | np.ndarray,
    condition: str,
    file_name: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Áp biến đổi SC-02 lên waveform gốc dài đầy đủ sau peak, trước khi tạo đoạn 4 giây."""
    require(condition in ALL_CONDITIONS, "Điều kiện SC-02 không hợp lệ: " + str(condition))
    raw_tensor = raw_waveform if isinstance(raw_waveform, torch.Tensor) else torch.from_numpy(np.array(raw_waveform, dtype=np.float32, copy=True))
    require(raw_tensor.ndim == 1 and raw_tensor.numel() > 0, "Waveform phải là 1D khác rỗng.")
    base_full = apply_amplitude_policy(raw_tensor, "peak").waveform
    original_len = int(base_full.numel())
    meta: dict[str, Any] = {"original_samples": original_len, "trimmed_samples": 0, "fallback_kept": False}

    if condition == "original":
        return base_full, meta

    if condition == "trim_edges":
        energies_db = sc01.compute_frame_energies_db(base_full.numpy())
        max_energy_db = float(np.max(energies_db))
        silent = energies_db < (max_energy_db - SILENCE_OFFSET_DB)
        if bool(silent.all()) or max_energy_db <= -119.9999:
            meta["fallback_kept"] = True
            return base_full, meta
        first_active = int(np.argmin(silent))
        last_active = int(silent.size - 1 - np.argmin(silent[::-1]))
        start_sample = first_active * FRAME_HOP
        end_sample = original_len if last_active == silent.size - 1 else last_active * FRAME_HOP + FRAME_LEN
        if end_sample - start_sample < FRAME_LEN:
            meta["fallback_kept"] = True
            return base_full, meta
        trimmed = base_full[start_sample:end_sample]
        meta["trimmed_samples"] = original_len - int(trimmed.numel())
        return apply_amplitude_policy(trimmed, "peak").waveform, meta

    if condition == "noise_floor":
        energies_db = sc01.compute_frame_energies_db(base_full.numpy())
        max_energy_db = float(np.max(energies_db))
        noise_power = 10.0 ** ((max_energy_db - NOISE_OFFSET_DB) / 10.0)
        noise_std = math.sqrt(noise_power)
        rng = np.random.default_rng(noise_seed_for_file(file_name))
        noise = rng.normal(loc=0.0, scale=noise_std, size=original_len)
        noisy = (base_full.numpy().astype(np.float64) + noise).astype(np.float32)
        meta["noise_std"] = float(noise_std)
        return apply_amplitude_policy(torch.from_numpy(noisy), "peak").waveform, meta

    # dc_remove
    x = base_full.to(torch.float64)
    centered = (x - x.mean()).to(torch.float32)
    return apply_amplitude_policy(centered, "peak").waveform, meta


def prepare_model_segment(full_after_transform: torch.Tensor, family: str) -> torch.Tensor:
    """Tạo đoạn 4 giây (center crop / repeat-then-trim) và áp lowpass_7k + peak nếu là X1R."""
    segment = fixed_length_segment(full_after_transform, SEGMENT_SAMPLES, training=False)
    if family == "X1R":
        return shared.apply_condition(segment, "lowpass_7k", "bonafide")
    return segment


def classify_indicator(
    *,
    eer_ratio: float | None,
    eer_diff_pp: float,
    miss_vc_diff_pp: float,
    fpr_bonafide_diff_pp: float,
) -> dict[str, Any]:
    """Phân loại mức chỉ báo SC-02 theo quy tắc đã khóa ở bước 8."""
    strong = bool(
        miss_vc_diff_pp >= 10.0
        or (eer_ratio is not None and eer_ratio >= 3.0 and eer_diff_pp >= 1.0)
    )
    medium = bool(
        not strong
        and (
            miss_vc_diff_pp >= 3.0
            or (eer_ratio is not None and eer_ratio >= 1.5 and eer_diff_pp >= 0.5)
        )
    )
    level = "mạnh" if strong else ("trung bình" if medium else "yếu")
    poor_robustness = bool(fpr_bonafide_diff_pp >= 3.0 and miss_vc_diff_pp < 3.0)
    return {
        "level": level,
        "poor_robustness": poor_robustness,
        "labels": [level, "độ bền kém"] if poor_robustness else [level],
    }


def summarize_condition(
    labels: Sequence[int],
    utt_types: Sequence[str],
    scores: Sequence[float],
    threshold: float,
    *,
    orig_summary: dict[str, Any] | None = None,
    orig_scores: Sequence[float] | None = None,
    trim_meta: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    eer_val, _ = fast_eer(labels, scores)
    conf = confusion_at_threshold(labels, scores, threshold=threshold)
    fpr = float(conf["fpr"])
    miss_vc = float(conf["fnr"])
    out: dict[str, Any] = {
        "samples": len(labels),
        "eer": float(eer_val),
        "decision_threshold": float(threshold),
        "fpr_bonafide": fpr,
        "miss_vc": miss_vc,
        "confusion": conf,
    }
    if orig_summary is not None and orig_scores is not None:
        orig_eer = float(orig_summary["eer"])
        orig_fpr = float(orig_summary["fpr_bonafide"])
        orig_miss = float(orig_summary["miss_vc"])
        eer_ratio = float(eer_val / orig_eer) if orig_eer > 0.0 else None
        eer_diff_pp = float(100.0 * (eer_val - orig_eer))
        miss_diff_pp = float(100.0 * (miss_vc - orig_miss))
        fpr_diff_pp = float(100.0 * (fpr - orig_fpr))

        orig_pred = np.asarray([float(s) >= threshold for s in orig_scores], dtype=bool)
        new_pred = np.asarray([float(s) >= threshold for s in scores], dtype=bool)
        is_bonafide = np.asarray([u == "bonafide" for u in utt_types], dtype=bool)
        is_vc = ~is_bonafide

        flips = {
            "total": int((orig_pred != new_pred).sum()),
            "bonafide_to_spoof": int((is_bonafide & (~orig_pred) & new_pred).sum()),
            "bonafide_spoof_to_bonafide": int((is_bonafide & orig_pred & (~new_pred)).sum()),
            "vc_to_bonafide_missed": int((is_vc & orig_pred & (~new_pred)).sum()),
            "vc_bonafide_to_spoof": int((is_vc & (~orig_pred) & new_pred).sum()),
        }
        out["comparison_vs_original"] = {
            "eer_ratio": eer_ratio,
            "eer_diff_pp": eer_diff_pp,
            "miss_vc_diff_pp": miss_diff_pp,
            "fpr_bonafide_diff_pp": fpr_diff_pp,
            "decision_flips": flips,
        }
        out["indicator"] = classify_indicator(
            eer_ratio=eer_ratio,
            eer_diff_pp=eer_diff_pp,
            miss_vc_diff_pp=miss_diff_pp,
            fpr_bonafide_diff_pp=fpr_diff_pp,
        )

    if trim_meta is not None:
        bf_trimmed = [m["trimmed_samples"] / SAMPLE_RATE for m, u in zip(trim_meta, utt_types, strict=True) if u == "bonafide"]
        vc_trimmed = [m["trimmed_samples"] / SAMPLE_RATE for m, u in zip(trim_meta, utt_types, strict=True) if u == "voice_conversion"]
        bf_nonzero = [v for v in bf_trimmed if v > 0]
        vc_nonzero = [v for v in vc_trimmed if v > 0]
        out["trim_details"] = {
            "bonafide_trimmed_count": len(bf_nonzero),
            "vc_trimmed_count": len(vc_nonzero),
            "total_trimmed_count": len(bf_nonzero) + len(vc_nonzero),
            "fallback_kept_count": sum(1 for m in trim_meta if m.get("fallback_kept")),
            "bonafide_trimmed_duration_s_median_all": float(np.median(bf_trimmed)),
            "vc_trimmed_duration_s_median_all": float(np.median(vc_trimmed)),
            "bonafide_trimmed_duration_s_median_trimmed_only": float(np.median(bf_nonzero)) if bf_nonzero else 0.0,
            "vc_trimmed_duration_s_median_trimmed_only": float(np.median(vc_nonzero)) if vc_nonzero else 0.0,
        }
    return out


class SC02BlockStore:
    def __init__(self, root: Path, identity_sha: str) -> None:
        self.root = root
        self.identity_sha = identity_sha

    def path(self, model_id: str, condition: str, start: int) -> Path:
        return self.root / model_id / condition / f"{start:06d}.json"

    def load(self, model_id: str, condition: str, start: int, files: Sequence[str], checkpoint_sha: str) -> dict[str, Any] | None:
        path = self.path(model_id, condition, start)
        if not path.is_file():
            return None
        value = shared.read_json(path)
        body = {k: v for k, v in value.items() if k != "body_sha256"}
        require(shared.digest(body) == value.get("body_sha256"), "Checksum khối SC-02 hỏng: " + str(path))
        require(value.get("identity_sha256") == self.identity_sha, "Identity khối SC-02 khác.")
        require(value.get("checkpoint_sha256") == checkpoint_sha, "Checkpoint SHA khối SC-02 lệch.")
        require(value.get("model_id") == model_id and value.get("condition") == condition and value.get("start") == start, "Vị trí khối sai.")
        require(value.get("files_sha256") == shared.file_list_sha(files), "Files SHA khối SC-02 lệch.")
        require([r["file"] for r in value["records"]] == list(files), "Thứ tự file trong khối SC-02 lệch.")
        return value

    def save(
        self,
        model_id: str,
        condition: str,
        start: int,
        files: Sequence[str],
        checkpoint_sha: str,
        records: list[dict[str, Any]],
        seconds: float,
    ) -> None:
        require([r["file"] for r in records] == list(files), "Danh sách file ghi khối SC-02 lệch.")
        body = {
            "version": 1,
            "model_id": model_id,
            "condition": condition,
            "start": start,
            "identity_sha256": self.identity_sha,
            "checkpoint_sha256": checkpoint_sha,
            "files_sha256": shared.file_list_sha(files),
            "records": records,
            "compute_seconds_allocated": float(seconds),
            "created_utc": utc(),
        }
        shared.atomic_json(self.path(model_id, condition, start), {**body, "body_sha256": shared.digest(body)}, replace=True)


def check_b0_reproduction(observed_scores: Sequence[float], reference_scores: Sequence[float], threshold: float) -> dict[str, Any]:
    obs = np.asarray(observed_scores, dtype=np.float64)
    ref = np.asarray(reference_scores, dtype=np.float64)
    require(obs.shape == ref.shape and obs.size > 0, "Thiếu score khi kiểm tái tạo B0.")
    diff = np.abs(obs - ref)
    flips = (obs >= threshold) != (ref >= threshold)
    max_diff = float(diff.max())
    return {
        "criterion": "B0_abs_prob_diff_1e-4_and_0_flips",
        "passed": bool(max_diff <= B0_MAX_PROB_DIFF and int(flips.sum()) == 0),
        "samples": int(obs.size),
        "max_abs_probability_difference": max_diff,
        "median_abs_probability_difference": float(np.median(diff)),
        "decision_flips": int(flips.sum()),
    }


def load_model(spec: dict[str, Any], entry: dict[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    checkpoint_path = ROOT / entry["checkpoint"]
    reject_test_path(checkpoint_path)
    require(checkpoint_path.is_file(), "Thiếu checkpoint: " + str(checkpoint_path))
    require(shared.sha256_file(checkpoint_path) == entry["checkpoint_sha256"], "SHA checkpoint lệch registry: " + spec["model_id"])

    if spec["family"] == "B0-v2":
        config = shared.read_json(LFCC_CONFIG_PATH)
        lfcc = build_lfcc_transform(config["lfcc"]).eval()
        model = build_lcnn_from_config(config)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        del payload
        gc.collect()
        return model, lfcc

    torch.manual_seed(2026)
    model = XLSRBinaryClassifier(
        load_xlsr_backbone(WEIGHTS_DIR, local_files_only=True),
        num_unfrozen_layers=8,
        hidden_features=128,
        dropout=0.2,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    saved = payload["model"]
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    require(set(saved) == expected and sum(p.numel() for p in saved.values()) == 101032193, "Tham số checkpoint XLS-R lệch.")
    missing, unexpected = model.load_state_dict(saved, strict=False)
    require(not unexpected and set(missing) == {n for n, p in model.named_parameters() if not p.requires_grad}, "Checkpoint XLS-R thiếu tham số.")
    model.eval().requires_grad_(False)
    del payload, saved
    gc.collect()
    return model, None


def predict_batch_list(
    model: torch.nn.Module,
    lfcc: torch.nn.Module | None,
    family: str,
    dataset: PackedManifestDataset,
    indices: Sequence[int],
    rows: Sequence[dict[str, Any]],
    condition: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            waveforms: list[torch.Tensor] = []
            metas: list[tuple[str, str, dict[str, Any]]] = []
            for idx in batch_indices:
                rec = dataset.records[idx]
                raw = np.asarray(dataset._waveforms[rec.offset_samples:rec.offset_samples + rec.num_samples], dtype="<f4")
                transformed, meta = apply_full_transform(raw, condition, rec.file)
                segment = prepare_model_segment(transformed, family)
                waveforms.append(segment)
                metas.append((rec.file, rec.utt_type, meta))
            batch = torch.stack(waveforms)
            if family == "B0-v2":
                assert lfcc is not None
                feats = lfcc_features(lfcc, batch).unsqueeze(1)
                logits = model(feats)
            else:
                batch = apply_xlsr_waveform_normalization(batch, "processor_zero_mean_unit_variance")
                logits = model(batch)
            require(bool(torch.isfinite(logits).all()), f"Logit không hữu hạn cho {family}/{condition}.")
            probs = torch.sigmoid(logits).tolist()
            for (file_name, utt_type, meta), prob, raw_logit in zip(metas, probs, logits.tolist(), strict=True):
                item: dict[str, Any] = {
                    "file": file_name,
                    "utt_type": utt_type,
                    "score": float(prob),
                    "raw_logit": float(raw_logit),
                }
                if condition == "trim_edges":
                    item["trimmed_samples"] = int(meta["trimmed_samples"])
                    item["fallback_kept"] = bool(meta["fallback_kept"])
                records.append(item)
    return records


def verify_and_prepare() -> tuple[dict[str, Any], list[dict[str, Any]], list[int], list[int], PackedManifestDataset, dict[str, Any], dict[str, Any]]:
    for p in (REGISTRY_V1_PATH, REGISTRY_X1R_PATH, PRIOR_T22_PATH, DEV_MANIFEST_PATH, DEV_ARTIFACT_DIR, LFCC_CONFIG_PATH, AUDIO_CONFIG_PATH, WEIGHTS_DIR):
        reject_test_path(p)
    v1_reg = shared.read_json(REGISTRY_V1_PATH)
    x1r_reg = shared.read_json(REGISTRY_X1R_PATH)
    entries = {e["model_id"]: e for e in (*v1_reg["entries"], *x1r_reg["entries"])}

    prior = shared.read_json(PRIOR_T22_PATH)
    require(prior.get("subset_files_sha256") == SUBSET_SHA and shared.file_list_sha(prior["subset_files"]) == SUBSET_SHA, "SHA tập con T19b lệch.")

    rows = shared.read_dev_manifest(DEV_MANIFEST_PATH, MANIFEST_SHA)
    require(len(rows) == 38777, "Manifest dev-full phải có 38.777 mẫu.")
    selected = shared.subset_indices(rows)
    require([rows[i]["file"] for i in selected] == prior["subset_files"], "Thứ tự tập con khác T19b.")

    bonafide_indices = [i for i in selected if rows[i]["utt_type"] == "bonafide"]
    vc_indices = [i for i in selected if rows[i]["utt_type"] == "voice_conversion"]
    require(len(bonafide_indices) == 1000 and len(vc_indices) == 1000, "Tập con phải có đúng 1.000 bonafide và 1.000 VC.")
    sample_indices = bonafide_indices + vc_indices
    repro_indices = bonafide_indices[:100] + vc_indices[:100]
    sample_files = [rows[i]["file"] for i in sample_indices]
    sample_files_sha = shared.file_list_sha(sample_files)

    metadata = shared.read_json(DEV_ARTIFACT_DIR / "metadata.json")
    require(metadata.get("split") == "closed_dev" and metadata.get("target_sample_rate") == 16000, "Artifact phải là closed_dev 16 kHz.")
    inputs = {
        "manifest_sha256": MANIFEST_SHA,
        "subset_files_sha256": SUBSET_SHA,
        "sc02_2000_files_sha256": sample_files_sha,
        "audio_config_sha256": shared.sha256_file(AUDIO_CONFIG_PATH),
        "lfcc_config_sha256": shared.sha256_file(LFCC_CONFIG_PATH),
        "metadata_sha256": shared.sha256_file(DEV_ARTIFACT_DIR / "metadata.json"),
        "index_sha256": shared.sha256_file(DEV_ARTIFACT_DIR / "index.csv"),
        "waveforms_sha256": shared.sha256_file(DEV_ARTIFACT_DIR / "waveforms.f32"),
    }
    require(inputs["waveforms_sha256"] == metadata["waveforms_sha256"], "Checksum waveforms.f32 lệch metadata.")
    for k in ("audio_config_sha256", "metadata_sha256", "index_sha256", "waveforms_sha256", "manifest_sha256"):
        require(inputs[k] == prior["identity"]["inputs"][k], f"Checksum {k} lệch T19b/T22.")

    pretrained = {name: shared.sha256_file(WEIGHTS_DIR / name) for name in PRETRAINED_SHA}
    require(pretrained == PRETRAINED_SHA, "SHA pretrained XLS-R lệch D021.")

    models_info: dict[str, Any] = {}
    saved_scores: dict[str, list[float]] = {}
    for spec in MODEL_SPECS:
        mid = spec["model_id"]
        entry = entries[mid]
        require(entry["status"] == "present", "Mục registry chưa present: " + mid)
        ckpt = ROOT / entry["checkpoint"]
        require(ckpt.is_file() and shared.sha256_file(ckpt) == entry["checkpoint_sha256"], "SHA checkpoint lệch registry: " + mid)
        src_report = shared.read_json(ROOT / entry["source_report"])
        best = src_report["controller"]["best"]
        dev = best["development"]
        require(best["generation"] == entry["generation"] and dev["decision_threshold"] == entry["decision_threshold"], "Report nguồn lệch registry: " + mid)
        recs = dev["records"]
        require(len(recs) == len(rows), "Report nguồn thiếu bản ghi dev-full: " + mid)
        for rec, row in zip(recs, rows, strict=True):
            require((rec["file"], rec["speaker_id"], rec["utt_type"], int(rec["label"])) == (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])), "Record report lệch manifest: " + mid)
        saved_scores[mid] = [float(recs[i]["score"]) for i in sample_indices]
        models_info[mid] = {
            "model_id": mid,
            "family": spec["family"],
            "generation": entry["generation"],
            "best_epoch": entry["best_epoch"],
            "decision_threshold": float(entry["decision_threshold"]),
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "source_report": entry["source_report"],
            "source_report_sha256": shared.sha256_file(ROOT / entry["source_report"]),
        }

    dataset = shared.open_dev_artifact(DEV_ARTIFACT_DIR)
    try:
        require(len(dataset) == len(rows) and len({r.shard for r in dataset.records}) == 186, "Số mẫu/shard packed sai.")
        for packed, row in zip(dataset.records, rows, strict=True):
            require((packed.file, packed.speaker_id, packed.utt_type, packed.binary_label) == (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])), "Packed lệch manifest.")
    except Exception:
        dataset.close()
        raise

    identity = {
        "version": 1,
        "task": "SC-02",
        "use_test": False,
        "inputs": inputs,
        "pretrained": pretrained,
        "models": models_info,
        "lowpass_7k": shared.filter_parameters("lowpass_7k"),
        "sample_count": len(sample_indices),
        "repro_count": len(repro_indices),
        "sample_shards": len({rows[i]["shard"] for i in sample_indices}),
        "torch_version": str(torch.__version__),
    }
    return identity, rows, sample_indices, repro_indices, dataset, entries, saved_scores


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SC-02 — Kiểm mô hình dựa vào khoảng lặng, nền nhiễu (và DC với B0) trên dev",
        "",
        f"- **Trạng thái:** {report['status']}; cập nhật UTC: {report['created_utc']}",
        f"- **Dữ liệu:** {report['data']['samples']:,} mẫu ({report['data']['bonafide_samples']:,} bonafide + {report['data']['vc_samples']:,} VC) trên {report['data']['shards']} shard từ tập con T19b (`{report['inputs']['sc02_2000_files_sha256']}`)",
        "- **Nguyên tắc:** biến đổi trên waveform gốc đầy đủ sau peak, trước crop/pad 4 giây; không đọc test, không dùng làm tiêu chí chọn mô hình.",
        "",
        "## Kiểm tái tạo trên 200 mẫu đầu (100 bonafide + 100 VC)",
        "",
        "| Mô hình | Tiêu chí | Đạt | Chi tiết |",
        "|---|---|---|---|",
    ]
    for spec in MODEL_SPECS:
        mid = spec["model_id"]
        rep = report["reproduction"][mid]
        if spec["family"] == "B0-v2":
            detail = f"max |Δp|={rep['max_abs_probability_difference']:.3e}, flips={rep['decision_flips']}"
        else:
            detail = f"max |Δlogit|={rep['max_abs_logit_difference']:.5f}, gap={rep['max_float32_gap_logit']:.5f}, flips_out={rep['flips_outside_0_05_margin']}"
        lines.append(f"| {mid} | {rep['criterion']} | {'đạt' if rep['passed'] else 'không đạt'} | {detail} |")

    lines.extend([
        "",
        "## Kết quả theo mô hình × điều kiện",
        "",
        "| Mô hình | Điều kiện | EER % | Miss VC % | FPR % | Tỷ số EER | Δ EER (đ%) | Δ Miss VC (đ%) | Δ FPR (đ%) | Đổi QĐ | Chỉ báo | Thời gian (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ])
    for spec in MODEL_SPECS:
        mid = spec["model_id"]
        mres = report["results"][mid]
        orig = mres["conditions"]["original"]
        lines.append(
            f"| {mid} | original (lưu) | {100*orig['eer']:.4f} | {100*orig['miss_vc']:.4f} | {100*orig['fpr_bonafide']:.4f} | 1.000 | +0.0000 | +0.0000 | +0.0000 | 0 | — | 0.00 |"
        )
        for cond in spec["conditions"]:
            c = mres["conditions"][cond]
            cmp = c["comparison_vs_original"]
            ind = ", ".join(c["indicator"]["labels"])
            ratio_str = f"{cmp['eer_ratio']:.3f}" if cmp["eer_ratio"] is not None else "—"
            sec = mres["timing"][cond]
            lines.append(
                f"| {mid} | {cond} | {100*c['eer']:.4f} | {100*c['miss_vc']:.4f} | {100*c['fpr_bonafide']:.4f} | "
                f"{ratio_str} | {cmp['eer_diff_pp']:+.4f} | {cmp['miss_vc_diff_pp']:+.4f} | {cmp['fpr_bonafide_diff_pp']:+.4f} | "
                f"{cmp['decision_flips']['total']} | {ind} | {sec:.2f} |"
            )

    lines.extend([
        "",
        "## Mức tổng theo mô hình và chi tiết trim_edges",
        "",
        "| Mô hình | Mức tổng | Số mẫu bị cắt (BF / VC) | Giữ nguyên (fallback) | Trung vị bị cắt BF / VC (s, toàn lớp) |",
        "|---|---|---|---:|---|",
    ])
    for spec in MODEL_SPECS:
        mid = spec["model_id"]
        mres = report["results"][mid]
        td = mres["conditions"]["trim_edges"]["trim_details"]
        lines.append(
            f"| {mid} | **{mres['overall_indicator_level']}** | "
            f"{td['bonafide_trimmed_count']} / {td['vc_trimmed_count']} | {td['fallback_kept_count']} | "
            f"{td['bonafide_trimmed_duration_s_median_all']:.3f} / {td['vc_trimmed_duration_s_median_all']:.3f} |"
        )

    lines.extend(["", "## Giới hạn", ""])
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def run_probe(*, cache_dir: Path = CACHE_DIR, report_json: Path = REPORT_JSON, report_md: Path = REPORT_MD) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity, rows, sample_indices, repro_indices, dataset, entries, saved_scores = verify_and_prepare()
    try:
        shared.atomic_json(
            cache_dir / "sample_list.json",
            {
                "subset_files_sha256": SUBSET_SHA,
                "sc02_2000_files_sha256": identity["inputs"]["sc02_2000_files_sha256"],
                "files": [rows[i]["file"] for i in sample_indices],
                "repro_200_files": [rows[i]["file"] for i in repro_indices],
            },
        )
        identity_sha = shared.digest(identity)
        store = SC02BlockStore(cache_dir / "blocks", identity_sha)

        sample_labels = [0 if rows[i]["utt_type"] == "bonafide" else 1 for i in sample_indices]
        sample_types = [rows[i]["utt_type"] for i in sample_indices]
        repro_pos = list(range(100)) + list(range(1000, 1100))

        reproduction_results: dict[str, Any] = {}
        model_results: dict[str, Any] = {}
        repro_failed = False

        for spec in MODEL_SPECS:
            mid = spec["model_id"]
            family = spec["family"]
            entry = entries[mid]
            threshold = float(entry["decision_threshold"])
            orig_sample_scores = saved_scores[mid]
            orig_summary = summarize_condition(sample_labels, sample_types, orig_sample_scores, threshold)

            repro_cache = cache_dir / f"repro_{mid}.json"
            cached_repro = shared.read_json(repro_cache) if repro_cache.is_file() else None
            if cached_repro is not None and cached_repro.get("identity_sha256") == identity_sha and cached_repro.get("reproduction", {}).get("passed") is True:
                repro = cached_repro["reproduction"]
                print(json.dumps({"event": "repro_reused", "model_id": mid, "passed": True}, ensure_ascii=False), flush=True)
                loaded_model = None
                loaded_lfcc = None
            else:
                print(json.dumps({"event": "model_started", "model_id": mid}, ensure_ascii=False), flush=True)
                loaded_model, loaded_lfcc = load_model(spec, entry)
                repro_records = predict_batch_list(
                    loaded_model,
                    loaded_lfcc,
                    family,
                    dataset,
                    repro_indices,
                    rows,
                    "original",
                    spec["batch_size"],
                )
                ref_repro = [orig_sample_scores[p] for p in repro_pos]
                if family == "B0-v2":
                    repro = check_b0_reproduction([r["score"] for r in repro_records], ref_repro, threshold)
                else:
                    repro = d034.check_logit_reproduction_d034(
                        [r["raw_logit"] for r in repro_records],
                        ref_repro,
                        threshold=threshold,
                        observed_scores=[r["score"] for r in repro_records],
                    )
                shared.atomic_json(repro_cache, {"identity_sha256": identity_sha, "reproduction": repro})
                print(json.dumps({"event": "repro_checked", "model_id": mid, **repro}, ensure_ascii=False), flush=True)

            reproduction_results[mid] = repro
            if not repro["passed"]:
                repro_failed = True
                if loaded_model is not None:
                    del loaded_model, loaded_lfcc
                    gc.collect()
                continue

            conditions_out: dict[str, Any] = {"original": orig_summary}
            timing_out: dict[str, float] = {}
            for cond in spec["conditions"]:
                cond_records: list[dict[str, Any]] = []
                cond_seconds = 0.0
                for start in range(0, len(sample_indices), BLOCK_SIZE):
                    block_indices = sample_indices[start:start + BLOCK_SIZE]
                    block_files = [rows[i]["file"] for i in block_indices]
                    loaded_block = store.load(mid, cond, start, block_files, entry["checkpoint_sha256"])
                    if loaded_block is not None:
                        cond_records.extend(loaded_block["records"])
                        cond_seconds += float(loaded_block["compute_seconds_allocated"])
                    else:
                        if loaded_model is None:
                            loaded_model, loaded_lfcc = load_model(spec, entry)
                        t0 = time.perf_counter()
                        recs = predict_batch_list(loaded_model, loaded_lfcc, family, dataset, block_indices, rows, cond, spec["batch_size"])
                        sec = time.perf_counter() - t0
                        store.save(mid, cond, start, block_files, entry["checkpoint_sha256"], recs, sec)
                        cond_records.extend(recs)
                        cond_seconds += sec
                        print(
                            json.dumps(
                                {"event": "block_saved", "model_id": mid, "condition": cond, "start": start, "samples": len(recs), "seconds": sec},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                trim_meta = cond_records if cond == "trim_edges" else None
                conditions_out[cond] = summarize_condition(
                    sample_labels,
                    sample_types,
                    [r["score"] for r in cond_records],
                    threshold,
                    orig_summary=orig_summary,
                    orig_scores=orig_sample_scores,
                    trim_meta=trim_meta,
                )
                timing_out[cond] = float(cond_seconds)
                print(
                    json.dumps(
                        {
                            "event": "condition_done",
                            "model_id": mid,
                            "condition": cond,
                            "eer": conditions_out[cond]["eer"],
                            "indicator": conditions_out[cond]["indicator"],
                            "seconds": cond_seconds,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if loaded_model is not None:
                del loaded_model, loaded_lfcc
                gc.collect()

            overall_level = max(
                (conditions_out[c]["indicator"]["level"] for c in spec["conditions"]),
                key=lambda lvl: LEVEL_ORDER[lvl],
            )
            model_results[mid] = {
                "overall_indicator_level": overall_level,
                "conditions": conditions_out,
                "timing": timing_out,
                "total_compute_seconds": float(sum(timing_out.values())),
            }
    finally:
        dataset.close()

    total_wall = time.perf_counter() - started
    status = "blocked_reproduction" if repro_failed else "complete"
    report = {
        "task": "SC-02",
        "status": status,
        "created_utc": utc(),
        "use_test": False,
        "trained": False,
        "inputs": identity["inputs"],
        "pretrained": identity["pretrained"],
        "models": identity["models"],
        "data": {
            "dev_samples": 38777,
            "dev_shards": 186,
            "samples": len(sample_indices),
            "bonafide_samples": 1000,
            "vc_samples": 1000,
            "shards": identity["sample_shards"],
            "repro_samples": len(repro_indices),
        },
        "definitions": {
            "transform_stage": "waveform gốc dài đầy đủ sau peak, trước fixed_length_segment(64000, training=False), rồi re-peak",
            "trim_edges": "khung 400 mẫu bước 160, dB=10*log10(mean(x^2)+1e-12), lặng < max-40 dB; cắt chuỗi lặng đầu/cuối; giữ nguyên nếu toàn lặng hoặc còn < 400 mẫu",
            "noise_floor": "cộng nhiễu trắng Gauss công suất = (max_frame_energy_db - 45 dB), default_rng(2026 + uint32_be(SHA256(file)[:4]))",
            "dc_remove": "trừ trung bình toàn waveform gốc rồi peak lại (chỉ cho B0 v2)",
            "indicator_rules": {
                "strong": "miss_vc_diff_pp >= 10 hoặc (eer_ratio >= 3 và eer_diff_pp >= 1)",
                "medium": "không mạnh và (miss_vc_diff_pp >= 3 hoặc (eer_ratio >= 1.5 và eer_diff_pp >= 0.5))",
                "weak": "còn lại",
                "poor_robustness": "fpr_bonafide_diff_pp >= 3 và miss_vc_diff_pp < 3",
            },
        },
        "reproduction": reproduction_results,
        "results": model_results,
        "limitations": [
            "Chỉ kiểm tra seed huấn luyện 2026 cho mỗi họ mô hình.",
            "Chỉ đánh giá 2.000 mẫu (1.000 bonafide + 1.000 VC) thuộc tập con T19b trên closed_dev; không đọc test.",
            "Cắt khoảng lặng (trim_edges) và cộng nhiễu (noise_floor) cũng làm thay đổi nội dung hợp lệ hoặc biên phân đoạn 4 giây, nên EER tăng không thuần túy do lối tắt.",
            "noise_floor đồng thời là phép thử độ bền trước nhiễu cộng.",
            "Kết quả chỉ áp dụng cho VSASV-HF-public-snapshot-v1 và tập 15.885 mẫu closed_train.",
        ],
        "wall_seconds": float(total_wall),
    }
    shared.atomic_json(report_json, report)
    report_md.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return (3 if repro_failed else 0), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    args = parser.parse_args()
    for p in (args.cache_dir, args.report_json, args.report_md):
        reject_test_path(p)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    try:
        code, report = run_probe(cache_dir=args.cache_dir, report_json=args.report_json, report_md=args.report_md)
        summary = {mid: res["overall_indicator_level"] for mid, res in report["results"].items()}
        print(json.dumps({"status": report["status"], "overall_levels": summary, "wall_seconds": report["wall_seconds"]}, ensure_ascii=False), flush=True)
        return code
    except Exception as exc:
        print("LỖI SC-02: " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
