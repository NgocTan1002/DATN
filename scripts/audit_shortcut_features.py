"""SC-01: thống kê đặc trưng lối tắt thô theo lớp trên closed_train và closed_dev (chỉ đọc, CPU)."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.amplitude import apply_amplitude_policy
from src.data.packed import PackedManifestDataset
from src.data.segment import fixed_length_segment
from src.metrics.bootstrap import confusion_at_threshold, fast_eer, speaker_bootstrap

CACHE_DIR = ROOT / ".cache/sc01"
AUDIO_CONFIG_PATH = ROOT / "configs/audio.json"
TRAIN_MANIFEST_PATH = ROOT / "data/manifests/development_20k_v2.csv"
TRAIN_ARTIFACT_DIR = ROOT / "data/processed/packed/train_development_20k_v2"
DEV_MANIFEST_PATH = ROOT / "data/manifests/closed_dev_full_v1.csv"
DEV_ARTIFACT_DIR = ROOT / "data/processed/packed/closed_dev_full_v1"
REPORT_JSON = ROOT / "reports/shortcut_feature_audit.json"
REPORT_MD = ROOT / "reports/shortcut_feature_audit.md"

DEV_MANIFEST_SHA256 = "a79f1bb2595bc46b08ce30b8652234b1858cc26e558e2ccd33632435df1a5360"
EXPECTED_TRAIN_COUNTS = {
    "bonafide": 7134,
    "voice_conversion": 4348,
    "adversarial_attack": 4348,
    "replay": 55,
}
EXPECTED_DEV_COUNTS = {
    "bonafide": 14151,
    "voice_conversion": 12253,
    "adversarial_attack": 12253,
    "replay": 120,
}
EXPECTED_TRAIN_SHARDS = 57
EXPECTED_DEV_SHARDS = 186

SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 64_000
FRAME_LEN = 400
FRAME_HOP = 160
ENERGY_EPS = 1e-12
SILENCE_OFFSET_DB = 40.0
LOWBAND_CUTOFF_HZ = 100.0
POWER_EPS = 1e-300

STRONG_EER_MAX = 0.10
MEDIUM_EER_MAX = 0.25
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000
LOGISTIC_SEED = 2026
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 2026
BOOTSTRAP_CONFIDENCE = 0.95

UTT_TYPES = ("bonafide", "voice_conversion", "adversarial_attack", "replay")
REGION_METRICS = (
    "silence_ratio",
    "leading_silence_s",
    "trailing_silence_s",
    "noise_floor_db",
    "dc_offset",
    "lowband_ratio_db",
)
CONTINUOUS_FEATURES = (
    "duration_s",
    "pad_fraction",
    *(f"full_{name}" for name in REGION_METRICS),
    *(f"crop_{name}" for name in REGION_METRICS),
)
FULL_COMBINED_FEATURES = ("duration_s", "pad_fraction", *(f"full_{name}" for name in REGION_METRICS))
CROP_COMBINED_FEATURES = ("duration_s", "pad_fraction", *(f"crop_{name}" for name in REGION_METRICS))
CSV_COLUMNS = (
    "split",
    "file",
    "speaker_id",
    "utt_type",
    "binary_label",
    "native_sample_rate",
    "shard",
    "num_samples",
    "is_padded",
    *CONTINUOUS_FEATURES,
)
TASKS = (
    ("bonafide_vs_spoof", None),
    ("bonafide_vs_vc", "voice_conversion"),
    ("bonafide_vs_ap", "adversarial_attack"),
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def reject_test_path(path: Path) -> None:
    require("closed_test" not in str(path).lower(), "SC-01 từ chối đọc đường dẫn dữ liệu test.")


def sha256_file(path: Path) -> str:
    reject_test_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    reject_test_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def compute_frame_energies_db(waveform: np.ndarray) -> np.ndarray:
    """Tính năng lượng dB từng khung chữ nhật 25 ms (400 mẫu), bước 10 ms (160 mẫu)."""
    x = np.asarray(waveform, dtype=np.float64)
    require(x.ndim == 1 and x.size > 0 and bool(np.isfinite(x).all()), "Waveform phải là mảng 1D hữu hạn khác rỗng.")
    if x.size >= FRAME_LEN:
        frames = np.lib.stride_tricks.sliding_window_view(x, FRAME_LEN)[::FRAME_HOP]
        mean_sq = np.mean(frames * frames, axis=1)
    else:
        mean_sq = np.array([float(np.mean(x * x))], dtype=np.float64)
    return 10.0 * np.log10(mean_sq + ENERGY_EPS)


def compute_region_features(
    waveform: np.ndarray,
    *,
    pad_start_sample: int | None = None,
) -> dict[str, float]:
    """Tính silence, noise floor, DC offset và tỷ lệ dải thấp <100 Hz cho một vùng đo."""
    energies_db = compute_frame_energies_db(waveform)
    max_energy_db = float(np.max(energies_db))
    silent = energies_db < (max_energy_db - SILENCE_OFFSET_DB)
    if pad_start_sample is not None:
        require(pad_start_sample > 0, "pad_start_sample phải lớn hơn 0.")
        frame_starts = np.arange(silent.size, dtype=np.int64) * FRAME_HOP
        silent = silent | (frame_starts >= pad_start_sample)

    silence_ratio = float(np.mean(silent))
    if bool(silent.all()):
        leading_frames = int(silent.size)
        trailing_frames = int(silent.size)
    else:
        leading_frames = int(np.argmin(silent))
        trailing_frames = int(np.argmin(silent[::-1]))
    hop_seconds = FRAME_HOP / SAMPLE_RATE
    leading_silence_s = float(leading_frames * hop_seconds)
    trailing_silence_s = float(trailing_frames * hop_seconds)
    noise_floor_db = float(np.percentile(energies_db, 10) - max_energy_db)

    x = np.asarray(waveform, dtype=np.float64)
    dc_offset = float(abs(np.mean(x)))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / SAMPLE_RATE)
    power = np.abs(np.fft.rfft(x)) ** 2
    if x.size % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    total_power = float(np.sum(power))
    low_power = float(np.sum(power[freqs < LOWBAND_CUTOFF_HZ]))
    lowband_ratio_db = float(10.0 * math.log10(max(low_power, POWER_EPS) / max(total_power, POWER_EPS)))

    return {
        "silence_ratio": silence_ratio,
        "leading_silence_s": leading_silence_s,
        "trailing_silence_s": trailing_silence_s,
        "noise_floor_db": noise_floor_db,
        "dc_offset": dc_offset,
        "lowband_ratio_db": lowband_ratio_db,
    }


def extract_sample_features(raw_waveform: np.ndarray, *, num_samples: int) -> dict[str, float | bool]:
    """Trích đặc trưng vùng full (sau peak) và vùng crop 4s (center crop/repeat-then-trim sau peak)."""
    require(num_samples > 0 and raw_waveform.shape == (num_samples,), "Độ dài waveform không khớp num_samples.")
    duration_s = float(num_samples / SAMPLE_RATE)
    is_padded = bool(num_samples < SEGMENT_SAMPLES)
    pad_fraction = float(max(0, SEGMENT_SAMPLES - num_samples) / SEGMENT_SAMPLES)

    tensor_raw = torch.from_numpy(np.array(raw_waveform, dtype=np.float32, copy=True))
    full_wave = apply_amplitude_policy(tensor_raw, "peak").waveform
    crop_wave = fixed_length_segment(full_wave, SEGMENT_SAMPLES, training=False)

    full_feats = compute_region_features(full_wave.numpy(), pad_start_sample=None)
    crop_feats = compute_region_features(
        crop_wave.numpy(),
        pad_start_sample=num_samples if is_padded else None,
    )
    out: dict[str, float | bool] = {
        "duration_s": duration_s,
        "is_padded": is_padded,
        "pad_fraction": pad_fraction,
    }
    for key, value in full_feats.items():
        out[f"full_{key}"] = value
    for key, value in crop_feats.items():
        out[f"crop_{key}"] = value
    return out


def classify_severity(eer: float) -> str:
    """Phân loại mức độ chỉ báo theo ngưỡng đã khóa (≤ 10% mạnh; ≤ 25% trung bình; > 25% yếu)."""
    require(math.isfinite(eer) and 0.0 <= eer <= 1.0, "EER phải nằm trong [0, 1].")
    if eer <= STRONG_EER_MAX:
        return "mạnh"
    if eer <= MEDIUM_EER_MAX:
        return "trung bình"
    return "yếu"


def evaluate_single_feature(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    dev_values: np.ndarray,
    dev_labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Chọn hướng và threshold tại EER trên train, áp nguyên threshold đó lên dev."""
    eer_pos, thr_pos = fast_eer(train_labels, train_values)
    eer_neg, thr_neg = fast_eer(train_labels, -train_values)
    if eer_pos <= eer_neg:
        direction = 1
        direction_label = "higher_is_spoof"
        train_eer, train_threshold = float(eer_pos), float(thr_pos)
    else:
        direction = -1
        direction_label = "lower_is_spoof"
        train_eer, train_threshold = float(eer_neg), float(thr_neg)

    dev_scores = np.asarray(dev_values, dtype=np.float64) * direction
    dev_eer, _ = fast_eer(dev_labels, dev_scores)
    confusion = confusion_at_threshold(dev_labels.tolist(), dev_scores.tolist(), threshold=train_threshold)
    result = {
        "direction": direction_label,
        "direction_sign": direction,
        "train_eer": float(train_eer),
        "train_threshold_signed": float(train_threshold),
        "train_threshold_raw": float(direction * train_threshold),
        "dev_eer": float(dev_eer),
        "dev_fpr_bonafide": float(confusion["fpr"]),
        "dev_miss_spoof": float(confusion["fnr"]),
        "dev_confusion": confusion,
    }
    return result, dev_scores


def evaluate_logistic_combination(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    dev_matrix: np.ndarray,
    dev_labels: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[dict[str, Any], np.ndarray]:
    """Chuẩn hóa z-score và huấn luyện hồi quy logistic trên train, chấm dev."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_matrix)
    x_dev = scaler.transform(dev_matrix)
    clf = LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, random_state=LOGISTIC_SEED)
    clf.fit(x_train, train_labels)

    train_scores = clf.predict_proba(x_train)[:, 1]
    dev_scores = clf.predict_proba(x_dev)[:, 1]
    train_eer, train_threshold = fast_eer(train_labels, train_scores)
    dev_eer, _ = fast_eer(dev_labels, dev_scores)
    confusion = confusion_at_threshold(dev_labels.tolist(), dev_scores.tolist(), threshold=train_threshold)
    weights = {name: float(coef) for name, coef in zip(feature_names, clf.coef_[0], strict=True)}
    result = {
        "features": list(feature_names),
        "train_eer": float(train_eer),
        "train_threshold": float(train_threshold),
        "dev_eer": float(dev_eer),
        "dev_fpr_bonafide": float(confusion["fpr"]),
        "dev_miss_spoof": float(confusion["fnr"]),
        "dev_confusion": confusion,
        "intercept": float(clf.intercept_[0]),
        "standardized_weights": weights,
        "n_iter": [int(v) for v in clf.n_iter_],
    }
    return result, dev_scores


def verify_split_inputs(
    *,
    manifest_path: Path,
    artifact_dir: Path,
    expected_split: str,
    expected_counts: dict[str, int],
    expected_shards: int,
    expected_manifest_sha: str | None = None,
) -> tuple[dict[str, Any], PackedManifestDataset, list[dict[str, Any]], dict[str, str]]:
    reject_test_path(manifest_path)
    reject_test_path(artifact_dir)
    metadata = load_json(artifact_dir / "metadata.json")
    require(metadata.get("split") == expected_split, f"Split metadata không khớp {expected_split}.")
    require(
        metadata.get("preprocessing_stage") == "post_resample_pre_amplitude_pre_crop",
        "Preprocessing stage artifact không khớp giao thức.",
    )
    require(metadata.get("target_sample_rate") == SAMPLE_RATE, "Sample rate artifact không phải 16 kHz.")
    require(metadata.get("dtype") == "float32-le", "Dtype artifact không phải float32-le.")

    actual_manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha is not None:
        require(actual_manifest_sha == expected_manifest_sha, f"SHA manifest {manifest_path.name} lệch giá trị khóa.")
    require(actual_manifest_sha == metadata["source_manifest_sha256"], f"SHA manifest lệch metadata của {artifact_dir.name}.")

    actual_index_sha = sha256_file(artifact_dir / "index.csv")
    require(actual_index_sha == metadata["index_sha256"], f"SHA index.csv lệch metadata của {artifact_dir.name}.")
    actual_wave_sha = sha256_file(artifact_dir / "waveforms.f32")
    require(actual_wave_sha == metadata["waveforms_sha256"], f"SHA waveforms.f32 lệch metadata của {artifact_dir.name}.")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = [row for row in csv.DictReader(handle) if row["split"] == expected_split]

    expected_total = sum(expected_counts.values())
    require(len(manifest_rows) == expected_total, f"Số mẫu {expected_split} khác {expected_total}: {len(manifest_rows)}.")
    actual_counts = dict(Counter(row["utt_type"] for row in manifest_rows))
    require(actual_counts == expected_counts, f"Phân bố lớp {expected_split} khác kỳ vọng: {actual_counts}.")
    actual_shards = len({row["shard"] for row in manifest_rows})
    require(actual_shards == expected_shards, f"Số shard {expected_split} khác {expected_shards}: {actual_shards}.")

    dataset = PackedManifestDataset(artifact_dir, training=False, seed=2026, amplitude_policy="peak")
    try:
        require(len(dataset) == len(manifest_rows), f"Số mẫu packed {expected_split} lệch manifest.")
        manifest_by_file = {row["file"]: row for row in manifest_rows}
        require(len(manifest_by_file) == len(manifest_rows), f"Manifest {expected_split} có file trùng.")
        for record in dataset.records:
            row = manifest_by_file.get(record.file)
            require(row is not None, f"File trong packed không có trong manifest {expected_split}: {record.file}")
            require(
                (record.speaker_id, record.utt_type, record.binary_label, record.native_sample_rate, record.shard)
                == (row["speaker_id"], row["utt_type"], int(row["binary_label"]), int(row["native_sample_rate"]), row["shard"]),
                f"Metadata packed lệch manifest cho {record.file}.",
            )
    except Exception:
        dataset.close()
        raise

    checksums = {
        "manifest_sha256": actual_manifest_sha,
        "metadata_sha256": sha256_file(artifact_dir / "metadata.json"),
        "index_sha256": actual_index_sha,
        "waveforms_sha256": actual_wave_sha,
    }
    return metadata, dataset, manifest_rows, checksums


def write_feature_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def read_feature_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            parsed: dict[str, Any] = {
                "split": raw["split"],
                "file": raw["file"],
                "speaker_id": raw["speaker_id"],
                "utt_type": raw["utt_type"],
                "binary_label": int(raw["binary_label"]),
                "native_sample_rate": int(raw["native_sample_rate"]),
                "shard": raw["shard"],
                "num_samples": int(raw["num_samples"]),
                "is_padded": raw["is_padded"] == "True",
            }
            for col in CONTINUOUS_FEATURES:
                parsed[col] = float(raw[col])
            rows.append(parsed)
    return rows


def extract_or_load_split(
    split_label: str,
    dataset: PackedManifestDataset,
    cache_csv: Path,
    meta_json: Path,
    expected_identity: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    if cache_csv.is_file() and meta_json.is_file():
        saved_meta = load_json(meta_json)
        if saved_meta.get("identity") == expected_identity and saved_meta.get("csv_sha256") == sha256_file(cache_csv):
            rows = read_feature_csv(cache_csv)
            if len(rows) == len(dataset):
                print(f"[{split_label}] Dùng lại đặc trưng cache ({len(rows)} mẫu).", flush=True)
                return rows, True

    rows = []
    total = len(dataset)
    started = time.perf_counter()
    for idx, record in enumerate(dataset.records):
        start = record.offset_samples
        stop = start + record.num_samples
        raw_waveform = np.asarray(dataset._waveforms[start:stop], dtype="<f4")
        feats = extract_sample_features(raw_waveform, num_samples=record.num_samples)
        rows.append(
            {
                "split": split_label,
                "file": record.file,
                "speaker_id": record.speaker_id,
                "utt_type": record.utt_type,
                "binary_label": record.binary_label,
                "native_sample_rate": record.native_sample_rate,
                "shard": record.shard,
                "num_samples": record.num_samples,
                **feats,
            }
        )
        if (idx + 1) % 5000 == 0 or (idx + 1) == total:
            elapsed = time.perf_counter() - started
            print(f"[{split_label}] Đã trích {idx + 1}/{total} mẫu ({elapsed:.1f}s).", flush=True)

    write_feature_csv(cache_csv, rows)
    meta_json.write_text(
        json.dumps({"identity": expected_identity, "csv_sha256": sha256_file(cache_csv)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows, False


def descriptive_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    for utt_type in UTT_TYPES:
        subset = [r for r in rows if r["utt_type"] == utt_type]
        n = len(subset)
        require(n > 0, f"Thiếu lớp {utt_type} khi tổng hợp thống kê mô tả.")
        padded_count = sum(1 for r in subset if r["is_padded"])
        features: dict[str, Any] = {}
        for feat in CONTINUOUS_FEATURES:
            vals = np.asarray([float(r[feat]) for r in subset], dtype=np.float64)
            p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95]).tolist()
            features[feat] = {"median": p50, "p5": p5, "p25": p25, "p75": p75, "p95": p95}
        by_class[utt_type] = {
            "n": n,
            "is_padded_count": padded_count,
            "is_padded_rate": float(padded_count / n),
            "features": features,
        }

    rates = sorted({int(r["native_sample_rate"]) for r in rows})
    crosstab: dict[str, Any] = {}
    for utt_type in UTT_TYPES:
        subset = [r for r in rows if r["utt_type"] == utt_type]
        counts = {str(rate): sum(1 for r in subset if int(r["native_sample_rate"]) == rate) for rate in rates}
        crosstab[utt_type] = {
            "n": len(subset),
            "counts": counts,
            "rates": {k: float(v / len(subset)) for k, v in counts.items()},
        }
    return {"by_class": by_class, "native_sample_rate_crosstab": crosstab}


def _filter_task_rows(rows: Sequence[dict[str, Any]], attack_filter: str | None) -> tuple[list[dict[str, Any]], np.ndarray]:
    if attack_filter is None:
        selected = list(rows)
    else:
        selected = [r for r in rows if r["utt_type"] in ("bonafide", attack_filter)]
    labels = np.asarray([int(r["binary_label"]) for r in selected], dtype=np.int64)
    return selected, labels


def run_separation_indicators(
    train_rows: Sequence[dict[str, Any]],
    dev_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    single_results: dict[str, dict[str, Any]] = {}
    logistic_results: dict[str, dict[str, Any]] = {}
    candidates: list[tuple[float, dict[str, Any], np.ndarray, np.ndarray, list[str]]] = []

    for task_name, attack_filter in TASKS:
        tr_rows, tr_y = _filter_task_rows(train_rows, attack_filter)
        dv_rows, dv_y = _filter_task_rows(dev_rows, attack_filter)
        dv_speakers = [str(r["speaker_id"]) for r in dv_rows]

        task_single: dict[str, Any] = {}
        for feat in CONTINUOUS_FEATURES:
            tr_x = np.asarray([float(r[feat]) for r in tr_rows], dtype=np.float64)
            dv_x = np.asarray([float(r[feat]) for r in dv_rows], dtype=np.float64)
            res, dv_scores = evaluate_single_feature(tr_x, tr_y, dv_x, dv_y)
            task_single[feat] = res
            candidates.append(
                (
                    res["dev_eer"],
                    {
                        "method_type": "single_feature",
                        "task": task_name,
                        "feature_or_region": feat,
                        "region": "utterance" if feat in ("duration_s", "pad_fraction") else feat.split("_", 1)[0],
                        "dev_eer": res["dev_eer"],
                        "train_eer": res["train_eer"],
                        "dev_fpr_bonafide": res["dev_fpr_bonafide"],
                        "dev_miss_spoof": res["dev_miss_spoof"],
                    },
                    dv_y,
                    dv_scores,
                    dv_speakers,
                )
            )
        single_results[task_name] = task_single

        task_logistic: dict[str, Any] = {}
        for region_name, feat_list in (("crop", CROP_COMBINED_FEATURES), ("full", FULL_COMBINED_FEATURES)):
            tr_mat = np.asarray([[float(r[f]) for f in feat_list] for r in tr_rows], dtype=np.float64)
            dv_mat = np.asarray([[float(r[f]) for f in feat_list] for r in dv_rows], dtype=np.float64)
            res, dv_scores = evaluate_logistic_combination(tr_mat, tr_y, dv_mat, dv_y, feat_list)
            task_logistic[region_name] = res
            candidates.append(
                (
                    res["dev_eer"],
                    {
                        "method_type": "logistic_regression",
                        "task": task_name,
                        "feature_or_region": f"logistic_{region_name}",
                        "region": region_name,
                        "dev_eer": res["dev_eer"],
                        "train_eer": res["train_eer"],
                        "dev_fpr_bonafide": res["dev_fpr_bonafide"],
                        "dev_miss_spoof": res["dev_miss_spoof"],
                    },
                    dv_y,
                    dv_scores,
                    dv_speakers,
                )
            )
        logistic_results[task_name] = task_logistic

    # Sắp xếp tất cả phép theo dev_eer tăng dần
    candidates.sort(key=lambda item: item[0])
    _, best_meta, best_y, best_scores, best_speakers = candidates[0]
    boot = speaker_bootstrap(
        best_y.tolist(),
        best_scores.tolist(),
        best_speakers,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    level = classify_severity(best_meta["dev_eer"])

    # Top 3 đơn đặc trưng có EER dev thấp nhất
    single_ranked = [meta for _, meta, _, _, _ in candidates if meta["method_type"] == "single_feature"]
    return {
        "single_feature": single_results,
        "logistic_regression": logistic_results,
        "summary": {
            "indicator_level": level,
            "lowest_dev_eer_method": {
                **best_meta,
                "speaker_bootstrap_95_ci": {
                    "eer_estimate": boot.eer.estimate,
                    "ci_low": boot.eer.ci_low,
                    "ci_high": boot.eer.ci_high,
                    "roc_auc_estimate": boot.roc_auc.estimate,
                    "roc_auc_ci_low": boot.roc_auc.ci_low,
                    "roc_auc_ci_high": boot.roc_auc.ci_high,
                    "speakers": boot.speakers,
                    "samples": boot.samples,
                    "n_resamples": boot.requested_resamples,
                    "valid_resamples": boot.valid_resamples,
                    "seed": boot.seed,
                },
            },
            "top_3_single_features": single_ranked[:3],
            "combined_bonafide_vs_spoof": {
                "crop_dev_eer": logistic_results["bonafide_vs_spoof"]["crop"]["dev_eer"],
                "full_dev_eer": logistic_results["bonafide_vs_spoof"]["full"]["dev_eer"],
            },
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["indicators"]["summary"]
    lowest = summary["lowest_dev_eer_method"]
    ci = lowest["speaker_bootstrap_95_ci"]
    lines = [
        "# SC-01 — Thống kê lối tắt theo lớp trên train và dev-full",
        "",
        f"- **Cập nhật UTC:** {report['created_utc']}",
        f"- **Mức phân loại chỉ báo:** **{summary['indicator_level']}** "
        f"(ngưỡng khóa: mạnh ≤ 10%, trung bình > 10% và ≤ 25%, yếu > 25%)",
        f"- **Phép đạt EER dev thấp nhất:** `{lowest['task']}` / `{lowest['feature_or_region']}` "
        f"(vùng `{lowest['region']}`) — EER dev = **{100.0 * lowest['dev_eer']:.4f}%** "
        f"(CI 95% theo người nói: [{100.0 * ci['ci_low']:.4f}%; {100.0 * ci['ci_high']:.4f}%], {ci['speakers']} speaker)",
        f"- **Dữ liệu train (`closed_train`):** {report['data']['train']['samples']:,} mẫu / {report['data']['train']['shards']} shard",
        f"- **Dữ liệu dev (`closed_dev`):** {report['data']['dev']['samples']:,} mẫu / {report['data']['dev']['shards']} shard",
        f"- **Thời gian chạy:** {report['timing']['total_seconds']:.2f} giây (CPU, chỉ đọc)",
        "",
        "## Bảng chéo sample rate gốc × lớp",
        "",
        "| Split | Lớp | n | 16 kHz | 40 kHz |",
        "|---|---|---:|---:|---:|",
    ]
    for split_key in ("train", "dev"):
        crosstab = report["descriptive"][split_key]["native_sample_rate_crosstab"]
        for utt_type in UTT_TYPES:
            row = crosstab[utt_type]
            c16 = row["counts"].get("16000", 0)
            c40 = row["counts"].get("40000", 0)
            lines.append(f"| {split_key} | {utt_type} | {row['n']:,} | {c16:,} | {c40:,} |")

    for split_key in ("train", "dev"):
        lines.extend([
            "",
            f"## Thống kê mô tả trên {split_key}",
            "",
            "| Lớp | n | Tỷ lệ pad | Đặc trưng | P5 | P25 | Trung vị | P75 | P95 |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ])
        by_class = report["descriptive"][split_key]["by_class"]
        for utt_type in UTT_TYPES:
            cls = by_class[utt_type]
            pad_str = f"{100.0 * cls['is_padded_rate']:.2f}%"
            for feat in CONTINUOUS_FEATURES:
                st = cls["features"][feat]
                lines.append(
                    f"| {utt_type} | {cls['n']:,} | {pad_str} | `{feat}` | "
                    f"{st['p5']:.4g} | {st['p25']:.4g} | {st['median']:.4g} | {st['p75']:.4g} | {st['p95']:.4g} |"
                )

    lines.extend([
        "",
        "## EER đơn đặc trưng (ngưỡng chọn trên train, áp nguyên lên dev)",
        "",
        "| Đặc trưng | Phép | Hướng (train) | EER train % | EER dev % | FPR bonafide dev % | Miss spoof dev % |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for task_name, _ in TASKS:
        for feat in CONTINUOUS_FEATURES:
            r = report["indicators"]["single_feature"][task_name][feat]
            lines.append(
                f"| `{feat}` | {task_name} | {r['direction']} | {100.0 * r['train_eer']:.4f} | "
                f"{100.0 * r['dev_eer']:.4f} | {100.0 * r['dev_fpr_bonafide']:.4f} | {100.0 * r['dev_miss_spoof']:.4f} |"
            )

    lines.extend([
        "",
        "## Kết hợp hồi quy logistic (chuẩn hóa z-score trên train, chấm dev)",
        "",
        "| Phép | Vùng | EER train % | EER dev % | FPR bonafide dev % | Miss spoof dev % |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for task_name, _ in TASKS:
        for region_name in ("crop", "full"):
            r = report["indicators"]["logistic_regression"][task_name][region_name]
            lines.append(
                f"| {task_name} | {region_name} | {100.0 * r['train_eer']:.4f} | "
                f"{100.0 * r['dev_eer']:.4f} | {100.0 * r['dev_fpr_bonafide']:.4f} | {100.0 * r['dev_miss_spoof']:.4f} |"
            )

    lines.extend(["", "## Giới hạn", ""])
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def run_audit(
    *,
    cache_dir: Path = CACHE_DIR,
    report_json: Path = REPORT_JSON,
    report_md: Path = REPORT_MD,
) -> dict[str, Any]:
    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_sha = sha256_file(AUDIO_CONFIG_PATH)

    _, train_ds, train_manifest_rows, train_checksums = verify_split_inputs(
        manifest_path=TRAIN_MANIFEST_PATH,
        artifact_dir=TRAIN_ARTIFACT_DIR,
        expected_split="closed_train",
        expected_counts=EXPECTED_TRAIN_COUNTS,
        expected_shards=EXPECTED_TRAIN_SHARDS,
    )
    try:
        _, dev_ds, dev_manifest_rows, dev_checksums = verify_split_inputs(
            manifest_path=DEV_MANIFEST_PATH,
            artifact_dir=DEV_ARTIFACT_DIR,
            expected_split="closed_dev",
            expected_counts=EXPECTED_DEV_COUNTS,
            expected_shards=EXPECTED_DEV_SHARDS,
            expected_manifest_sha=DEV_MANIFEST_SHA256,
        )
        try:
            t_extract = time.perf_counter()
            script_sha = sha256_file(Path(__file__))
            train_rows, train_cached = extract_or_load_split(
                "closed_train",
                train_ds,
                cache_dir / "train_features.csv",
                cache_dir / "train_features_meta.json",
                {"split": "closed_train", "audio_sha256": audio_sha, "script_sha256": script_sha, **train_checksums},
            )
            dev_rows, dev_cached = extract_or_load_split(
                "closed_dev",
                dev_ds,
                cache_dir / "dev_features.csv",
                cache_dir / "dev_features_meta.json",
                {"split": "closed_dev", "audio_sha256": audio_sha, "script_sha256": script_sha, **dev_checksums},
            )
            extract_seconds = time.perf_counter() - t_extract
        finally:
            dev_ds.close()
    finally:
        train_ds.close()

    t_stats = time.perf_counter()
    descriptive = {
        "train": descriptive_stats(train_rows),
        "dev": descriptive_stats(dev_rows),
    }
    indicators = run_separation_indicators(train_rows, dev_rows)
    stats_seconds = time.perf_counter() - t_stats
    total_seconds = time.perf_counter() - started

    report = {
        "task": "SC-01",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "use_test": False,
        "inputs": {
            "audio_config_sha256": audio_sha,
            "train": train_checksums,
            "dev": dev_checksums,
        },
        "data": {
            "snapshot": "VSASV-HF-public-snapshot-v1",
            "train": {
                "split": "closed_train",
                "samples": len(train_rows),
                "shards": len({r["shard"] for r in train_rows}),
                "by_utt_type": dict(Counter(r["utt_type"] for r in train_rows)),
            },
            "dev": {
                "split": "closed_dev",
                "samples": len(dev_rows),
                "shards": len({r["shard"] for r in dev_rows}),
                "by_utt_type": dict(Counter(r["utt_type"] for r in dev_rows)),
            },
        },
        "feature_definitions": {
            "sample_rate_hz": SAMPLE_RATE,
            "segment_samples": SEGMENT_SAMPLES,
            "amplitude_policy": "peak (target=0.95, near_silence=-50 dBFS)",
            "crop_policy": "evaluation center crop / repeat-then-trim (src/data/segment.py)",
            "frame_length_samples": FRAME_LEN,
            "frame_hop_samples": FRAME_HOP,
            "frame_window": "rectangular",
            "frame_energy_db": "10 * log10(mean(x^2) + 1e-12)",
            "silent_frame_rule": "frame_energy_db < (max_frame_energy_db - 40.0); trên vùng crop mọi khung có frame_start >= num_samples (phần pad) tính là lặng",
            "leading_trailing_silence_unit": "consecutive_silent_frames * (160 / 16000) seconds",
            "noise_floor_db": "percentile_10(frame_energy_db) - max(frame_energy_db)",
            "dc_offset": "|mean(x)|",
            "lowband_ratio_db": "10 * log10(sum(power[freq < 100 Hz]) / total_power) via float64 one-sided rFFT",
        },
        "locked_thresholds": {
            "strong_eer_max": STRONG_EER_MAX,
            "medium_eer_max": MEDIUM_EER_MAX,
            "logistic_regression": {"scaler": "StandardScaler", "C": LOGISTIC_C, "max_iter": LOGISTIC_MAX_ITER, "random_state": LOGISTIC_SEED},
            "bootstrap": {"n_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "confidence": BOOTSTRAP_CONFIDENCE},
        },
        "descriptive": descriptive,
        "indicators": indicators,
        "limitations": [
            "Không có cột nguồn corpus hay codec trong metadata cục bộ nên không kiểm được hai lối tắt này.",
            "Đây là chỉ báo dữ liệu thô, không đo hành vi mô hình hay chứng minh mô hình dùng lối tắt.",
            "Crop train lúc huấn luyện là ngẫu nhiên theo seed/epoch, nên đặc trưng vùng crop (center crop) chỉ xấp xỉ cái mô hình thấy khi train.",
            "Kết quả chỉ áp dụng cho VSASV-HF-public-snapshot-v1 và tập con 15.885 mẫu closed_train / 38.777 mẫu closed_dev.",
        ],
        "timing": {
            "extract_seconds": extract_seconds,
            "stats_and_models_seconds": stats_seconds,
            "total_seconds": total_seconds,
            "train_reused_from_cache": train_cached,
            "dev_reused_from_cache": dev_cached,
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = report_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp_json.replace(report_json)

    tmp_md = report_md.with_suffix(".md.tmp")
    tmp_md.write_text(render_markdown(report), encoding="utf-8")
    tmp_md.replace(report_md)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    args = parser.parse_args()
    for path in (args.cache_dir, args.report_json, args.report_md):
        reject_test_path(path)
    try:
        report = run_audit(cache_dir=args.cache_dir, report_json=args.report_json, report_md=args.report_md)
        summary = report["indicators"]["summary"]
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "indicator_level": summary["indicator_level"],
                    "lowest": summary["lowest_dev_eer_method"],
                    "total_seconds": report["timing"]["total_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        print("LỖI SC-01: " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
