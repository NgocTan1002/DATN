"""REG-X1R: kiểm tính hợp lệ round-trip trên dev cho 3 seed X1R (D035 mục 6)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0")

import numpy as np
import torch

from scripts import build_model_registry as registry_mod
from scripts import probe_shortcut_dev as shared
from scripts import probe_shortcut_x1_final_d034 as d034
from src.data.packed import PackedManifestDataset
from src.data.xlsr import apply_xlsr_waveform_normalization
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone

CACHE_DIR = ROOT / ".cache/reg_x1r"
CONFIG_PATH = ROOT / "configs/test_evaluation_x1r.json"
REGISTRY_PATH = ROOT / "reports/model_registry_x1r_draft.json"
PRIOR_T22_PATH = ROOT / "reports/shortcut_probe_dev_x1_final.json"
DEV_MANIFEST_PATH = ROOT / "data/manifests/closed_dev_full_v1.csv"
DEV_ARTIFACT_DIR = ROOT / "data/processed/packed/closed_dev_full_v1"
WEIGHTS_DIR = ROOT / "data/raw/xlsr/wav2vec2-xls-r-300m"
REPORT_JSON = ROOT / "reports/x1r_roundtrip_dev.json"
REPORT_MD = ROOT / "reports/x1r_roundtrip_dev.md"

MANIFEST_SHA = shared.MANIFEST_SHA
SUBSET_SHA = shared.SUBSET_SHA
LOWPASS_7K_COEFFICIENT_SHA256 = registry_mod.LOWPASS_7K_COEFFICIENT_SHA256
PRETRAINED_SHA = {
    "pytorch_model.bin": "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0",
    "config.json": "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
    "preprocessor_config.json": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
}
X1R_CONDITIONS = ("x1r_original", "x1r_roundtrip_40k")
MAX_FPR_INCREASE_PP = 0.5
X1R_SEEDS = (2026, 2027, 2028)
require = shared.require


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def reject_test_path(path: Path) -> None:
    require("closed_test" not in str(path).lower(), "Từ chối đọc đường dẫn dữ liệu test.")


def transform_x1r_waveform(waveform: torch.Tensor, condition: str) -> torch.Tensor:
    """Biến đổi 1 đoạn waveform 64.000 mẫu (đã center-crop + peak) theo điều kiện X1R.

    - ``x1r_original``: lowpass_7k + peak (đúng pipeline X1R lúc train/eval).
    - ``x1r_roundtrip_40k``: resample 16->40->16 kHz + peak, rồi đi tiếp qua lowpass_7k + peak.
    """
    require(condition in X1R_CONDITIONS, "Điều kiện X1R không hợp lệ: " + str(condition))
    if condition == "x1r_original":
        return shared.apply_condition(waveform, "lowpass_7k", "bonafide")
    after_roundtrip = shared.apply_condition(waveform, "roundtrip_40k", "bonafide")
    return shared.apply_condition(after_roundtrip, "lowpass_7k", "bonafide")


def summarize_bonafide_condition(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Tổng hợp FPR bonafide và phân vị xác suất/logit trên tập mẫu bonafide."""
    require(bool(records), "Thiếu bản ghi để tổng hợp điều kiện X1R.")
    scores = np.asarray([float(r["score"]) for r in records], dtype=np.float64)
    raw_logits = np.asarray([float(r["raw_logit"]) for r in records], dtype=np.float64)
    stable_logits = shared.stable_logit(scores)
    require(bool(np.isfinite(raw_logits).all()), "Logit chứa giá trị không hữu hạn.")
    errors = int((scores >= threshold).sum())
    total = int(scores.size)
    return {
        "samples": total,
        "errors": errors,
        "fpr": float(errors / total),
        "decision_threshold": float(threshold),
        "probability_p5_p50_p95": np.quantile(scores, [0.05, 0.5, 0.95]).tolist(),
        "logit_p5_p50_p95": np.quantile(stable_logits, [0.05, 0.5, 0.95]).tolist(),
        "raw_logit_p5_p50_p95": np.quantile(raw_logits, [0.05, 0.5, 0.95]).tolist(),
    }


def evaluate_roundtrip_criterion(
    original_records: Sequence[dict[str, Any]],
    roundtrip_records: Sequence[dict[str, Any]],
    threshold: float,
    *,
    max_increase_pp: float = MAX_FPR_INCREASE_PP,
) -> dict[str, Any]:
    """Đánh giá hiệu FPR (điểm %) và số mẫu đổi quyết định giữa x1r_original và x1r_roundtrip_40k."""
    require(len(original_records) == len(roundtrip_records) and len(original_records) > 0,
            "Số lượng mẫu giữa hai điều kiện không khớp hoặc rỗng.")
    require(
        [r["file"] for r in original_records] == [r["file"] for r in roundtrip_records],
        "Thứ tự file giữa hai điều kiện không khớp.",
    )
    orig_summary = summarize_bonafide_condition(original_records, threshold)
    rt_summary = summarize_bonafide_condition(roundtrip_records, threshold)
    total = orig_summary["samples"]
    orig_errors = orig_summary["errors"]
    rt_errors = rt_summary["errors"]
    # Tính theo hiệu số nguyên lỗi * 100 / total để tránh nhiễu float tại biên 0,5 điểm %.
    delta_pp = float((rt_errors - orig_errors) * 100.0 / total)
    passed = bool((rt_errors - orig_errors) * 100.0 <= max_increase_pp * total)
    orig_pred = np.asarray([float(r["score"]) >= threshold for r in original_records], dtype=bool)
    rt_pred = np.asarray([float(r["score"]) >= threshold for r in roundtrip_records], dtype=bool)
    flips = orig_pred != rt_pred
    return {
        "fpr_x1r_original": orig_summary["fpr"],
        "fpr_x1r_roundtrip_40k": rt_summary["fpr"],
        "fpr_increase_percentage_points": delta_pp,
        "max_allowed_fpr_increase_percentage_points": float(max_increase_pp),
        "passed": passed,
        "status_label": "đạt" if passed else "không đạt",
        "decision_flips_vs_original": int(flips.sum()),
        "flips_bonafide_to_spoof": int((~orig_pred & rt_pred).sum()),
        "flips_spoof_to_bonafide": int((orig_pred & ~rt_pred).sum()),
    }


def verify_and_prepare_inputs(
    *,
    root: Path = ROOT,
    config_path: Path = CONFIG_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[int], PackedManifestDataset, dict[str, Any]]:
    """Kiểm toàn bộ đầu vào, bộ lọc, registry X1R và trích đúng 1.000 mẫu bonafide T19b."""
    for path in (config_path, registry_path, PRIOR_T22_PATH, DEV_MANIFEST_PATH, DEV_ARTIFACT_DIR, WEIGHTS_DIR):
        reject_test_path(path)
    config = registry_mod.load_json(config_path)
    registry = registry_mod.load_json(registry_path)
    registry_mod.validate_registry(registry, config, complete=False)

    lp7_params = shared.filter_parameters("lowpass_7k")
    require(
        lp7_params["coefficient_float32_sha256"] == LOWPASS_7K_COEFFICIENT_SHA256,
        "SHA-256 hệ số bộ lọc lowpass_7k không khớp D035.",
    )
    require(lp7_params == config["input_condition"]["filter"], "Tham số lowpass_7k lệch config X1R.")
    rt_params = shared.filter_parameters("roundtrip_40k")

    prior = shared.read_json(PRIOR_T22_PATH)
    require(
        prior.get("subset_files_sha256") == SUBSET_SHA
        and len(prior.get("subset_files", [])) == 3120
        and shared.file_list_sha(prior["subset_files"]) == SUBSET_SHA,
        "Tập con T19b trong report T22 không khớp SHA đăng ký.",
    )

    rows = shared.read_dev_manifest(DEV_MANIFEST_PATH, MANIFEST_SHA)
    require(len(rows) == 38777, "Manifest dev-full phải có đúng 38.777 mẫu.")
    selected = shared.subset_indices(rows)
    require([rows[i]["file"] for i in selected] == prior["subset_files"], "Thứ tự tập con khác T19b/T22.")

    bonafide_indices = [i for i in selected if rows[i]["utt_type"] == "bonafide"]
    require(len(bonafide_indices) == 1000, "Tập con T19b phải có đúng 1.000 mẫu bonafide.")
    bonafide_files = [rows[i]["file"] for i in bonafide_indices]
    bonafide_files_sha = shared.file_list_sha(bonafide_files)

    metadata = shared.read_json(DEV_ARTIFACT_DIR / "metadata.json")
    require(
        metadata.get("split") == "closed_dev" and metadata.get("target_sample_rate") == 16000,
        "Chỉ chấp nhận artifact closed_dev 16 kHz.",
    )
    inputs = {
        "manifest_sha256": MANIFEST_SHA,
        "subset_files_sha256": SUBSET_SHA,
        "bonafide_files_sha256": bonafide_files_sha,
        "audio_config_sha256": shared.sha256_file(root / "configs/audio.json"),
        "metadata_sha256": shared.sha256_file(DEV_ARTIFACT_DIR / "metadata.json"),
        "index_sha256": shared.sha256_file(DEV_ARTIFACT_DIR / "index.csv"),
        "waveforms_sha256": shared.sha256_file(DEV_ARTIFACT_DIR / "waveforms.f32"),
    }
    require(inputs["waveforms_sha256"] == metadata["waveforms_sha256"], "Checksum waveform dev-full lệch metadata.")
    for key in ("audio_config_sha256", "metadata_sha256", "index_sha256", "waveforms_sha256", "manifest_sha256", "subset_files_sha256"):
        require(inputs[key] == prior["identity"]["inputs"][key], f"Checksum {key} lệch T19b/T22.")

    pretrained = {name: shared.sha256_file(WEIGHTS_DIR / name) for name in PRETRAINED_SHA}
    require(pretrained == PRETRAINED_SHA, "SHA-256 pretrained wav2vec2-xls-r-300m không khớp D021.")

    dataset = shared.open_dev_artifact(DEV_ARTIFACT_DIR)
    try:
        require(len(dataset) == len(rows) and len({r.shard for r in dataset.records}) == 186, "Số mẫu/shard packed sai.")
        for packed, row in zip(dataset.records, rows, strict=True):
            require(
                (packed.file, packed.speaker_id, packed.utt_type, packed.binary_label)
                == (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])),
                "Metadata packed khác manifest.",
            )
    except Exception:
        dataset.close()
        raise

    shared_identity = {
        "version": 1,
        "task": "REG-X1R-roundtrip-dev",
        "decision": "D035",
        "use_test": False,
        "device": "cpu",
        "torch_threads": 8,
        "xlsr_batch_size": 2,
        "data_seed": 2026,
        "normalization": "processor_zero_mean_unit_variance",
        "inputs": inputs,
        "pretrained": pretrained,
        "filters": {"lowpass_7k": lp7_params, "roundtrip_40k": rt_params},
        "bonafide_samples": len(bonafide_indices),
        "bonafide_shards": len({rows[i]["shard"] for i in bonafide_indices}),
        "torch_version": str(torch.__version__),
        "torchaudio_version": str(torch.__version__ if False else __import__("torchaudio").__version__),
    }
    return config, registry, rows, bonafide_indices, dataset, shared_identity


def load_x1r_model(checkpoint_path: Path, expected_sha256: str) -> torch.nn.Module:
    reject_test_path(checkpoint_path)
    require(checkpoint_path.is_file(), "Thiếu checkpoint X1R: " + str(checkpoint_path))
    require(shared.sha256_file(checkpoint_path) == expected_sha256, "SHA-256 checkpoint X1R không khớp trước khi nạp.")
    torch.manual_seed(2026)
    model = XLSRBinaryClassifier(
        load_xlsr_backbone(WEIGHTS_DIR, local_files_only=True),
        num_unfrozen_layers=8,
        hidden_features=128,
        dropout=0.2,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    saved = payload["model"]
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    require(set(saved) == expected and sum(param.numel() for param in saved.values()) == 101032193,
            "Checkpoint X1R không đúng tập trainable-only 8 tầng.")
    require(all(param.dtype == torch.float32 for param in saved.values()), "Checkpoint X1R phải là float32.")
    missing, unexpected = model.load_state_dict(saved, strict=False)
    require(
        not unexpected and set(missing) == {name for name, param in model.named_parameters() if not param.requires_grad},
        "Checkpoint X1R thiếu hoặc sai tham số.",
    )
    model.eval().requires_grad_(False)
    del payload, saved
    gc.collect()
    return model


def predict_condition(
    model: torch.nn.Module,
    dataset: PackedManifestDataset,
    indices: Sequence[int],
    condition: str,
    *,
    batch_size: int = 2,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            samples = [dataset[i] for i in batch_indices]
            waveforms = torch.stack([transform_x1r_waveform(sample["waveform"], condition) for sample in samples])
            waveforms = apply_xlsr_waveform_normalization(waveforms, "processor_zero_mean_unit_variance")
            logits = model(waveforms)
            require(bool(torch.isfinite(logits).all()), "Logit X1R không hữu hạn ở điều kiện " + condition)
            scores = torch.sigmoid(logits).tolist()
            for sample, score, raw in zip(samples, scores, logits.tolist(), strict=True):
                records.append({"file": sample["file"], "score": float(score), "raw_logit": float(raw)})
    return records


def build_seed_identity(
    entry: dict[str, Any],
    shared_identity: dict[str, Any],
    *,
    source_report_sha256: str,
) -> dict[str, Any]:
    return {
        **shared_identity,
        "model": {
            "model_id": entry["model_id"],
            "training_seed": entry["training_seed"],
            "generation": entry["generation"],
            "best_epoch": entry["best_epoch"],
            "decision_threshold": entry["decision_threshold"],
            "dev_eer": entry["dev_eer"],
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "source_report": entry["source_report"],
            "source_report_sha256": source_report_sha256,
        },
    }


def seed_cache_path(cache_dir: Path, seed: int) -> Path:
    return cache_dir / f"seed_{seed}_result.json"


def load_cached_seed_result(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        cached = shared.read_json(path)
        body_sha = cached.get("body_sha256")
        body = {k: v for k, v in cached.items() if k != "body_sha256"}
        if shared.digest(body) != body_sha:
            return None
        if cached.get("identity") != expected_identity:
            return None
        repro = cached.get("reproduction")
        if not isinstance(repro, dict) or repro.get("passed") is not True:
            return None
        if set(cached.get("conditions", {})) != set(X1R_CONDITIONS):
            return None
        return cached
    except (ValueError, KeyError, TypeError, OSError):
        return None


def save_seed_result(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in result.items() if k != "body_sha256"}
    payload = {**body, "body_sha256": shared.digest(body)}
    shared.atomic_json(path, payload)
    return payload


def extract_reference_scores(
    report_path: Path,
    rows: Sequence[dict[str, Any]],
    bonafide_indices: Sequence[int],
    entry: dict[str, Any],
) -> list[float]:
    reject_test_path(report_path)
    report = shared.read_json(report_path)
    best = report["controller"]["best"]
    development = best["development"]
    require(best["generation"] == entry["generation"], "Generation best trong report lệch registry.")
    require(development["decision_threshold"] == entry["decision_threshold"], "Threshold trong report lệch registry.")
    records = development["records"]
    require(len(records) == len(rows), "Report nguồn không đủ 38.777 bản ghi dev-full.")
    for record, row in zip(records, rows, strict=True):
        require(
            (record["file"], record["speaker_id"], record["utt_type"], int(record["label"]))
            == (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])),
            "Record dev-full trong report lệch manifest.",
        )
    scores = [float(records[i]["score"]) for i in bonafide_indices]
    shared.stable_logit(scores)
    return scores


def run_seed_check(
    entry: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    bonafide_indices: Sequence[int],
    dataset: PackedManifestDataset,
    shared_identity: dict[str, Any],
    *,
    cache_dir: Path = CACHE_DIR,
    root: Path = ROOT,
    batch_size: int = 2,
) -> dict[str, Any]:
    seed = int(entry["training_seed"])
    report_path = root / entry["source_report"]
    checkpoint_path = root / entry["checkpoint"]
    source_report_sha256 = shared.sha256_file(report_path)
    require(source_report_sha256 == entry["source_report_sha256"], "SHA report nguồn lệch registry: " + entry["model_id"])
    seed_identity = build_seed_identity(entry, shared_identity, source_report_sha256=source_report_sha256)
    cache_path = seed_cache_path(cache_dir, seed)
    cached = load_cached_seed_result(cache_path, seed_identity)
    if cached is not None:
        print(json.dumps({"event": "seed_reused", "seed": seed, "model_id": entry["model_id"]}, ensure_ascii=False), flush=True)
        return {**cached, "reused_from_cache": True}

    reference_scores = extract_reference_scores(report_path, rows, bonafide_indices, entry)
    threshold = float(entry["decision_threshold"])
    print(json.dumps({"event": "seed_started", "seed": seed, "model_id": entry["model_id"]}, ensure_ascii=False), flush=True)
    seed_started = time.perf_counter()
    model = load_x1r_model(checkpoint_path, entry["checkpoint_sha256"])
    try:
        t0 = time.perf_counter()
        orig_records = predict_condition(model, dataset, bonafide_indices, "x1r_original", batch_size=batch_size)
        orig_seconds = time.perf_counter() - t0
        repro = d034.check_logit_reproduction_d034(
            [r["raw_logit"] for r in orig_records],
            reference_scores,
            threshold=threshold,
            observed_scores=[r["score"] for r in orig_records],
        )
        orig_summary = summarize_bonafide_condition(orig_records, threshold)

        if not repro["passed"]:
            total_seconds = time.perf_counter() - seed_started
            failed_result = {
                "model_id": entry["model_id"],
                "training_seed": seed,
                "generation": entry["generation"],
                "best_epoch": entry["best_epoch"],
                "decision_threshold": threshold,
                "checkpoint": entry["checkpoint"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "source_report": entry["source_report"],
                "source_report_sha256": source_report_sha256,
                "identity": seed_identity,
                "reproduction": repro,
                "conditions": {"x1r_original": orig_summary},
                "comparison": None,
                "records": {"x1r_original": orig_records},
                "timing": {
                    "x1r_original_seconds": orig_seconds,
                    "x1r_roundtrip_40k_seconds": None,
                    "total_seconds": total_seconds,
                },
                "updated_utc": utc(),
            }
            saved = save_seed_result(cache_path, failed_result)
            print(json.dumps({"event": "seed_reproduction_failed", "seed": seed, "reproduction": repro}, ensure_ascii=False), flush=True)
            return {**saved, "reused_from_cache": False}

        t1 = time.perf_counter()
        rt_records = predict_condition(model, dataset, bonafide_indices, "x1r_roundtrip_40k", batch_size=batch_size)
        rt_seconds = time.perf_counter() - t1
        rt_summary = summarize_bonafide_condition(rt_records, threshold)
        comparison = evaluate_roundtrip_criterion(orig_records, rt_records, threshold)
        total_seconds = time.perf_counter() - seed_started

        complete_result = {
            "model_id": entry["model_id"],
            "training_seed": seed,
            "generation": entry["generation"],
            "best_epoch": entry["best_epoch"],
            "decision_threshold": threshold,
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "source_report": entry["source_report"],
            "source_report_sha256": source_report_sha256,
            "identity": seed_identity,
            "reproduction": repro,
            "conditions": {
                "x1r_original": orig_summary,
                "x1r_roundtrip_40k": rt_summary,
            },
            "comparison": comparison,
            "records": {
                "x1r_original": orig_records,
                "x1r_roundtrip_40k": rt_records,
            },
            "timing": {
                "x1r_original_seconds": orig_seconds,
                "x1r_roundtrip_40k_seconds": rt_seconds,
                "total_seconds": total_seconds,
            },
            "updated_utc": utc(),
        }
        saved = save_seed_result(cache_path, complete_result)
        print(
            json.dumps(
                {
                    "event": "seed_completed",
                    "seed": seed,
                    "reproduction_passed": repro["passed"],
                    "roundtrip_passed": comparison["passed"],
                    "fpr_increase_pp": comparison["fpr_increase_percentage_points"],
                    "total_seconds": total_seconds,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return {**saved, "reused_from_cache": False}
    finally:
        del model
        gc.collect()


def public_seed_view(result: dict[str, Any]) -> dict[str, Any]:
    """Loại bỏ mảng 1.000 bản ghi chi tiết khỏi report tổng hợp (vẫn giữ nguyên trong .cache)."""
    return {
        "model_id": result["model_id"],
        "training_seed": result["training_seed"],
        "generation": result["generation"],
        "best_epoch": result["best_epoch"],
        "decision_threshold": result["decision_threshold"],
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "source_report": result["source_report"],
        "source_report_sha256": result["source_report_sha256"],
        "reproduction": result["reproduction"],
        "conditions": result["conditions"],
        "comparison": result["comparison"],
        "timing": {**result["timing"], "reused_from_cache": bool(result.get("reused_from_cache", False))},
        "updated_utc": result["updated_utc"],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# REG-X1R — Kiểm tính hợp lệ round-trip trên dev (D035 mục 6)",
        "",
        f"Trạng thái: {report['status']} ({report['summary']['overall_status']}); cập nhật UTC: {report['updated_utc']}",
        f"Dữ liệu: {report['data']['bonafide_samples']} mẫu bonafide trong tập con T19b ({report['data']['bonafide_shards']} shard); "
        f"artifact dev-full {report['data']['dev_samples']} mẫu/186 shard.",
        f"SHA-256 tập con T19b: `{report['data']['subset_files_sha256']}`",
        f"SHA-256 1.000 mẫu bonafide: `{report['data']['bonafide_files_sha256']}`",
        f"Ngưỡng kỳ vọng D035 mục 6: tăng FPR bonafide ≤ {report['max_fpr_increase_percentage_points']} điểm %.",
        f"Tiêu chí tái tạo: {report['reproduction_rule']}. Chỉ đánh giá CPU trên dev, không đọc dữ liệu test.",
        "",
        "## Kết quả theo seed",
        "",
        "| Seed | Epoch best | Threshold | FPR x1r_original % | FPR x1r_roundtrip_40k % | Hiệu (điểm %) | Đổi quyết định | Tái tạo D034 | Kết quả | Thời gian (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for seed_str in sorted(report["seeds"]):
        item = report["seeds"][seed_str]
        repro = item["reproduction"]
        repro_str = f"{'đạt' if repro['passed'] else 'không đạt'} (max Δ={repro['max_abs_logit_difference']:.5f}, gap={repro['max_float32_gap_logit']:.5f})"
        orig_fpr = f"{100.0 * item['conditions']['x1r_original']['fpr']:.4f}"
        comp = item["comparison"]
        if comp is None:
            rt_fpr = "—"
            delta_str = "—"
            flips_str = "—"
            status_str = "dừng (tái tạo không đạt)"
        else:
            rt_fpr = f"{100.0 * item['conditions']['x1r_roundtrip_40k']['fpr']:.4f}"
            delta_str = f"{comp['fpr_increase_percentage_points']:+.4f}"
            flips_str = str(comp["decision_flips_vs_original"])
            status_str = comp["status_label"]
        lines.append(
            f"| {item['training_seed']} | {item['best_epoch']} | {item['decision_threshold']} | "
            f"{orig_fpr} | {rt_fpr} | {delta_str} | {flips_str} | {repro_str} | {status_str} | "
            f"{item['timing']['total_seconds']:.2f} |"
        )
    lines.extend([
        "",
        "## Phân vị logit (P5 / P50 / P95)",
        "",
        "| Seed | Điều kiện | P5 | P50 | P95 |",
        "|---|---|---:|---:|---:|",
    ])
    for seed_str in sorted(report["seeds"]):
        item = report["seeds"][seed_str]
        for cond_name, cond_data in item["conditions"].items():
            p5, p50, p95 = cond_data["raw_logit_p5_p50_p95"]
            lines.append(f"| {item['training_seed']} | {cond_name} | {p5:.6f} | {p50:.6f} | {p95:.6f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_all(
    *,
    seeds: Sequence[int] = X1R_SEEDS,
    batch_size: int = 2,
    cache_dir: Path = CACHE_DIR,
    report_json: Path = REPORT_JSON,
    report_md: Path = REPORT_MD,
) -> tuple[int, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    config, registry, rows, bonafide_indices, dataset, shared_identity = verify_and_prepare_inputs()
    try:
        entries_by_seed = {int(entry["training_seed"]): entry for entry in registry["entries"]}
        seed_results: dict[str, Any] = {}
        for seed in seeds:
            entry = entries_by_seed[int(seed)]
            require(entry["status"] == "present", "Mục X1R chưa present trong registry: " + entry["model_id"])
            seed_results[str(seed)] = public_seed_view(
                run_seed_check(
                    entry,
                    rows,
                    bonafide_indices,
                    dataset,
                    shared_identity,
                    cache_dir=cache_dir,
                    batch_size=batch_size,
                )
            )
    finally:
        dataset.close()

    all_repro = all(v["reproduction"]["passed"] for v in seed_results.values())
    all_rt = all(v["comparison"] is not None and v["comparison"]["passed"] for v in seed_results.values())
    deltas = [v["comparison"]["fpr_increase_percentage_points"] for v in seed_results.values() if v["comparison"] is not None]
    status = "complete" if (all_repro and all_rt) else ("blocked_reproduction" if not all_repro else "failed_threshold")
    report = {
        "task": "REG-X1R-roundtrip-dev",
        "decision": "D035",
        "status": status,
        "updated_utc": utc(),
        "use_test": False,
        "trained": False,
        "gates_changed": False,
        "reproduction_rule": "D034",
        "max_fpr_increase_percentage_points": MAX_FPR_INCREASE_PP,
        "registry_path": str(REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "registry_protocol_sha256": registry["protocol_sha256"],
        "data": {
            "dev_samples": 38777,
            "dev_shards": 186,
            "subset_total_samples": 3120,
            "bonafide_samples": len(bonafide_indices),
            "bonafide_shards": shared_identity["bonafide_shards"],
            "subset_files_sha256": SUBSET_SHA,
            "bonafide_files_sha256": shared_identity["inputs"]["bonafide_files_sha256"],
        },
        "inputs": shared_identity["inputs"],
        "pretrained": shared_identity["pretrained"],
        "filters": shared_identity["filters"],
        "seeds": seed_results,
        "summary": {
            "seeds_checked": [int(s) for s in seeds],
            "all_reproduction_passed": all_repro,
            "all_roundtrip_passed": all_rt,
            "overall_status": "đạt" if (all_repro and all_rt) else "không đạt",
            "max_observed_fpr_increase_percentage_points": max(deltas) if deltas else None,
        },
    }
    shared.atomic_json(report_json, report)
    write_markdown(report_md, report)
    if not all_repro:
        return 3, report
    if not all_rt:
        return 4, report
    return 0, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(X1R_SEEDS))
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    try:
        code, report = run_all(seeds=args.seeds, batch_size=args.batch_size)
        print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False), flush=True)
        return code
    except Exception as exc:
        print("LỖI kiểm round-trip X1R: " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
