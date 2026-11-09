"""T19b: đánh giá dev-only theo khối, không huấn luyện hoặc truy cập mạng."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator, Sequence
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0")

import csv
import numpy as np
import torch
import torchaudio
from src.data.amplitude import apply_amplitude_policy
from src.data.packed import PackedManifestDataset
from src.data.xlsr import apply_xlsr_waveform_normalization
from src.metrics.eer import equal_error_rate
from src.metrics.bootstrap import per_utt_type_eer
from src.models.lfcc_lcnn import build_lcnn_from_config, build_lfcc_transform, lfcc_features
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone

CACHE = ROOT / ".cache/t19b"
MANIFEST_SHA = "a79f1bb2595bc46b08ce30b8652234b1858cc26e558e2ccd33632435df1a5360"
SUBSET_SHA = "869d8f87c8ffa6c52efc6bc619048dc85dd1eda6d9148fffeefc71ec7d802dfe"
GROUPS = ("bonafide", "voice_conversion", "adversarial_attack", "replay")
CONDITIONS = ("original", "roundtrip_40k", "lowpass_7k", "lowpass_4k")
XMODEL = "X1-L8-lr1e-5-seed2026"
BMODELS = tuple(f"B0-v2-seed{s}" for s in (2026, 2027, 2028))
WEIGHTS = ROOT / "data/raw/xlsr/wav2vec2-xls-r-300m"
PRETRAINED = {
    "pytorch_model.bin": "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0",
    "config.json": "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
    "preprocessor_config.json": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
}
X_CHECKPOINT_SHA = "d9a1cf41a592ea8f6825a1a21e63c1cdab6c981f8dd9ea9291f344fd291d20dd"
BEST_GENERATIONS = {
    2026: "g-951dc37b31054c3ab33a74816a20423c",
    2027: "g-1f9266acc9704954850fdafaa2be1b61",
    2028: "g-86bc3dc8b0cd4dbf91b4f84e4bc62c4b",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), "Không được ghi đè tệp đã có: " + str(path))
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


def file_list_sha(files: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(files) + "\n").encode("utf-8")).hexdigest()


def stable_logit(probabilities: Any) -> np.ndarray:
    """Chỉ thay đúng 0/1 bằng số float64 kề bên; giữ mọi xác suất nội miền."""
    values = np.asarray(probabilities, dtype=np.float64)
    require(bool(np.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()),
            "Score phải hữu hạn và nằm trong [0,1].")
    safe = np.where(values == 0, np.nextafter(0.0, 1.0), values)
    safe = np.where(safe == 1, np.nextafter(1.0, 0.0), safe)
    return np.log(safe) - np.log1p(-safe)


def read_dev_manifest(path: Path, expected_sha: str | None = MANIFEST_SHA) -> list[dict[str, Any]]:
    require("test" not in path.name.lower(), "T19b từ chối manifest test.")
    if expected_sha is not None:
        require(sha256_file(path) == expected_sha, "Checksum manifest dev-full không khớp.")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows) and {r["split"] for r in rows} == {"closed_dev"}, "Chỉ chấp nhận manifest closed_dev.")
    require(len({r["file"] for r in rows}) == len(rows), "Manifest có file trùng.")
    for row in rows:
        require(row["utt_type"] in GROUPS, "Loại mẫu không hợp lệ.")
        require(int(row["binary_label"]) == int(row["utt_type"] != "bonafide"), "Nhãn không khớp loại mẫu.")
    return rows


def subset_indices(rows: Sequence[dict[str, Any]]) -> list[int]:
    selected: list[int] = []
    for group, count in zip(GROUPS, (1000, 1000, 1000, 120)):
        pool = sorted((i for i, r in enumerate(rows) if r["utt_type"] == group),
                      key=lambda i: hashlib.sha256(("2026|" + rows[i]["file"]).encode()).hexdigest())
        require(len(pool) >= count, "Không đủ quota tập con: " + group)
        selected.extend(pool[:count])
    return selected


def open_dev_artifact(path: Path) -> PackedManifestDataset:
    require("test" not in path.name.lower(), "T19b từ chối artifact test.")
    metadata = read_json(path / "metadata.json")
    require(metadata.get("split") == "closed_dev", "Chỉ chấp nhận artifact closed_dev; không mở test.")
    return PackedManifestDataset(path, training=False, seed=2026, amplitude_policy="peak")


@lru_cache(maxsize=2)
def fir_kernel(cutoff: int) -> torch.Tensor:
    require(cutoff in (4000, 7000), "Tần số cắt chưa đăng ký.")
    n = torch.arange(513, dtype=torch.float64) - 256
    kernel = (2 * cutoff / 16000) * torch.sinc(2 * cutoff / 16000 * n)
    kernel *= torch.hann_window(513, periodic=False, dtype=torch.float64)
    return (kernel / kernel.sum()).to(torch.float32)


@lru_cache(maxsize=2)
def fir_spectrum(cutoff: int) -> torch.Tensor:
    return torch.fft.rfft(fir_kernel(cutoff), n=65536)


def fir_filter(waveform: torch.Tensor, cutoff: int) -> torch.Tensor:
    require(waveform.shape == (64000,), "FIR cần waveform 64000 mẫu.")
    return torch.fft.irfft(torch.fft.rfft(waveform, n=65536) * fir_spectrum(cutoff),
                          n=65536)[256:64256]


def filter_parameters(condition: str) -> dict[str, Any]:
    if condition == "original":
        return {"operation": "identity", "repeak": False}
    if condition == "roundtrip_40k":
        return {"operation": "torchaudio.functional.resample", "sample_rates": [16000, 40000, 16000],
                "lowpass_filter_width": 6, "rolloff": 0.99, "resampling_method": "sinc_interp_hann",
                "beta": None, "only_utt_type": "bonafide", "repeak": True}
    require(condition in ("lowpass_7k", "lowpass_4k"), "Điều kiện chưa đăng ký.")
    cutoff = 7000 if condition == "lowpass_7k" else 4000
    h = fir_kernel(cutoff)
    frequencies = (3000, 4000, 6000, 7000, 7500)
    n = np.arange(513) - 256
    response = {}
    for frequency in frequencies:
        amplitude = abs(np.sum(h.numpy().astype(np.float64) * np.exp(-2j * np.pi * frequency * n / 16000)))
        response[str(frequency)] = 20 * math.log10(max(float(amplitude), np.finfo(float).tiny))
    return {"operation": "linear_phase_windowed_sinc", "cutoff_hz": cutoff, "taps": 513,
            "window": "symmetric_Hann", "sample_rate": 16000, "fft_length": 65536,
            "delay_compensation_samples": 256, "boundary": "zero_extension",
            "coefficient_float32_sha256": hashlib.sha256(h.numpy().tobytes()).hexdigest(),
            "response_db": response, "repeak": True}


def apply_condition(waveform: torch.Tensor, condition: str, utt_type: str) -> torch.Tensor:
    require(waveform.shape == (64000,) and waveform.dtype == torch.float32, "Cần waveform float32/64000 mẫu.")
    require(condition in CONDITIONS and utt_type in GROUPS, "Điều kiện hoặc loại mẫu chưa đăng ký.")
    if condition == "original" or (condition == "roundtrip_40k" and utt_type != "bonafide"):
        return waveform
    if condition == "roundtrip_40k":
        changed = torchaudio.functional.resample(
            torchaudio.functional.resample(waveform, 16000, 40000), 40000, 16000)
    else:
        cutoff = 7000 if condition == "lowpass_7k" else 4000
        # Tích chập tuyến tính, bỏ trễ nhóm 256; không lọc hai chiều/square đáp ứng.
        changed = fir_filter(waveform, cutoff)
    require(changed.shape == waveform.shape, "Biến đổi làm sai độ dài waveform.")
    return apply_amplitude_policy(changed, "peak").waveform


class DeadlineReached(RuntimeError):
    pass


class ReproductionFailure(RuntimeError):
    pass


class Deadline:
    def __init__(self, hours: float, clock: Callable[[], float] = time.monotonic) -> None:
        require(math.isfinite(hours) and hours > 0, "Hạn giờ phải hữu hạn và dương.")
        self.clock = clock
        self.end = clock() + hours * 3600

    def expired(self) -> bool:
        return self.clock() >= self.end


class BlockStore:
    def __init__(self, root: Path, identity: dict[str, Any]) -> None:
        self.root = root
        self.identity = identity
        self.identity_sha = digest(identity)

    def path(self, model: str, condition: str, start: int) -> Path:
        return self.root / model / condition / f"{start:06d}.json"

    def load(self, model: str, condition: str, start: int, files: list[str]) -> dict[str, Any] | None:
        path = self.path(model, condition, start)
        if not path.exists():
            return None
        try:
            value = read_json(path)
            checksum = value["body_sha256"]
            body = {k: v for k, v in value.items() if k != "body_sha256"}
            require(digest(body) == checksum, "Checksum nội dung khối không khớp.")
            require(value["identity_sha256"] == self.identity_sha, "Identity khối khác checkpoint/đầu vào.")
            require(value["checkpoint_sha256"] == self.identity["models"][model]["checkpoint_sha256"], "Checkpoint khối lệch.")
            require(value["filter_parameters"] == self.identity["filters"][condition], "Tham số bộ lọc khối lệch.")
            require(value["model"] == model and value["condition"] == condition and value["start"] == start, "Sai vị trí khối.")
            require(value["files_sha256"] == file_list_sha(files), "Checksum danh sách file trong khối lệch.")
            require([r["file"] for r in value["records"]] == files and 0 < len(files) <= 200, "Sai file/thứ tự/kích thước khối.")
            stable_logit([r["score"] for r in value["records"]])
            require(all(math.isfinite(r["raw_logit"]) for r in value["records"]), "Khối chứa logit không hữu hạn.")
            return value
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ValueError("Khối hỏng hoặc lệch; không ghi đè: " + str(path)) from exc

    def save(self, model: str, condition: str, start: int, files: list[str],
             records: list[dict[str, Any]], seconds: float) -> None:
        require(0 < len(files) <= 200 and [r["file"] for r in records] == files, "Sai kích thước/thứ tự khối.")
        stable_logit([r["score"] for r in records])
        require(all(math.isfinite(r["raw_logit"]) for r in records), "Logit khối không hữu hạn.")
        require(math.isfinite(seconds) and seconds >= 0, "Thời gian khối phải không âm và hữu hạn.")
        body = {"version": 1, "model": model, "condition": condition, "start": start,
                "identity_sha256": self.identity_sha,
                "checkpoint_sha256": self.identity["models"][model]["checkpoint_sha256"],
                "filter_parameters": self.identity["filters"][condition], "files_sha256": file_list_sha(files),
                "records": records, "compute_seconds_allocated": seconds, "created_utc": utc()}
        atomic_json(self.path(model, condition, start), {**body, "body_sha256": digest(body)}, replace=False)


def process_blocks(store: BlockStore, models: Sequence[str], condition: str, indices: list[int],
                   rows: Sequence[dict[str, Any]], predict: Callable[[list[int], list[str], str], dict[str, list[dict[str, Any]]]],
                   deadline: Deadline, emit: Callable[..., None], *, limit: int | None = None) -> None:
    count = len(indices) if limit is None else min(limit, len(indices))
    for start in range(0, count, 200):
        if deadline.expired():
            raise DeadlineReached("Hết hạn trước khối tiếp theo.")
        current = indices[start:min(start + 200, count)]
        files = [rows[i]["file"] for i in current]
        missing = [model for model in models if store.load(model, condition, start, files) is None]
        if missing:
            began = time.perf_counter()
            values = predict(current, missing, condition)
            seconds = time.perf_counter() - began
            for model in missing:
                store.save(model, condition, start, files, values[model], seconds / len(missing))
            emit("block_committed", models=missing, condition=condition, start=start,
                 samples=len(files), completed=start + len(files), total=len(indices), seconds=seconds)
        else:
            emit("block_reused", models=list(models), condition=condition, start=start, samples=len(files))
        if deadline.expired():
            raise DeadlineReached("Hết hạn; đã ghi nguyên tử khối vừa chạy.")


def check_reproduction(observed: Sequence[float], reference: Sequence[float],
                       threshold: float, *, model: str) -> dict[str, Any]:
    current = np.asarray(observed, dtype=np.float64)
    old = np.asarray(reference, dtype=np.float64)
    require(current.shape == old.shape and current.size > 0, "Thiếu score khi kiểm tái tạo.")
    flips = (current >= threshold) != (old >= threshold)
    if model == XMODEL:
        difference = np.abs(stable_logit(current) - stable_logit(old))
        distances = np.abs(stable_logit(old) - stable_logit([threshold])[0])
        passed = bool(difference.max() <= .05 and (distances[flips] <= .05).all())
        return {"passed": passed, "samples": int(current.size), "max_abs_logit_difference": float(difference.max()),
                "median_abs_logit_difference": float(np.median(difference)), "decision_flips": int(flips.sum()),
                "flips_outside_0_05_margin": int((distances[flips] > .05).sum())}
    difference = np.abs(current - old)
    return {"passed": bool(difference.max() <= 1e-5 and not flips.any()), "samples": int(current.size),
            "max_abs_probability_difference": float(difference.max()), "decision_flips": int(flips.sum())}


def summarize_scores(rows: Sequence[dict[str, Any]], scores: list[float], threshold: float) -> dict[str, Any]:
    require(len(rows) == len(scores), "Số score không khớp mẫu đánh giá.")
    labels = [int(r["binary_label"]) for r in rows]
    types = [r["utt_type"] for r in rows]
    result: dict[str, Any] = {"samples": len(rows), "shards": len({r["shard"] for r in rows}),
        "overall_eer": equal_error_rate(labels, scores).eer,
        "per_attack_eer": per_utt_type_eer(labels, scores, types), "decision_threshold": threshold,
        "threshold_source": "fixed_checkpoint_original_dev", "groups": {}}
    for group in GROUPS:
        values = np.asarray([score for score, r in zip(scores, rows) if r["utt_type"] == group])
        require(values.size > 0, "Thiếu nhóm khi tổng hợp metric.")
        wrong = values >= threshold if group == "bonafide" else values < threshold
        result["groups"][group] = {"samples": int(values.size), "errors": int(wrong.sum()),
            "fpr" if group == "bonafide" else "miss_rate": float(wrong.mean()),
            "probability_p5_p50_p95": np.quantile(values, [.05, .5, .95]).tolist(),
            "logit_p5_p50_p95": np.quantile(stable_logit(values), [.05, .5, .95]).tolist()}
    return result


def indicators(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "original" not in results:
        return {"status": "not_available"}
    original = results["original"]
    result: dict[str, Any] = {"lowpass_4k_is_robustness_only": True}
    strong: list[bool] = []
    if "roundtrip_40k" in results:
        delta = 100 * (results["roundtrip_40k"]["groups"]["bonafide"]["fpr"] - original["groups"]["bonafide"]["fpr"])
        result["roundtrip_40k"] = {"fpr_increase_percentage_points": delta, "strong_threshold_hit": delta >= 10}
        strong.append(delta >= 10)
    if "lowpass_7k" in results:
        ratios = {}
        for attack in ("voice_conversion", "adversarial_attack"):
            before = original["per_attack_eer"][attack]["eer"]
            after = results["lowpass_7k"]["per_attack_eer"][attack]["eer"]
            ratios[attack] = after / before if before > 0 else None
        valid = all(value is not None for value in ratios.values())
        hit = valid and ratios["voice_conversion"] >= 3 and ratios["adversarial_attack"] < 1.5
        result["lowpass_7k"] = {"eer_ratios": ratios, "strong_threshold_hit": bool(hit) if valid else None}
        if valid:
            strong.append(bool(hit))
    result["available_indicator_level"] = "strong" if any(strong) else "weak" if strong else "not_available"
    result["complete_for_shortcut_indicators"] = "roundtrip_40k" in results and "lowpass_7k" in results
    result["interpretation"] = "Chỉ báo theo ngưỡng khai báo trước, không kết luận có/không shortcut."
    return result


def prepare_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[int],
                             PackedManifestDataset, dict[str, list[float]]]:
    rows = read_dev_manifest(ROOT / "data/manifests/closed_dev_full_v1.csv")
    require(len(rows) == 38777, "Số mẫu dev-full không đúng 38.777.")
    selected = subset_indices(rows)
    files = [rows[i]["file"] for i in selected]
    require(file_list_sha(files) == SUBSET_SHA, "Danh sách 3.120 mẫu không khớp đăng ký T19.")
    artifact = ROOT / "data/processed/packed/closed_dev_full_v1"
    locked = read_json(ROOT / ".cache/t19/report_stop_t19.json")["input_checksums"]
    inputs = {"manifest_sha256": MANIFEST_SHA, "subset_files_sha256": SUBSET_SHA}
    for key, path in (("audio_config_sha256", ROOT / "configs/audio.json"),
                      ("metadata_sha256", artifact / "metadata.json"), ("index_sha256", artifact / "index.csv")):
        actual = sha256_file(path)
        require(actual == locked[key], "Checksum đầu vào khác T19: " + key)
        inputs[key] = actual
    metadata = read_json(artifact / "metadata.json")
    require(metadata["split"] == "closed_dev" and metadata["target_sample_rate"] == 16000,
            "Không chấp nhận artifact ngoài closed_dev/16kHz.")
    inputs["waveforms_sha256"] = sha256_file(artifact / "waveforms.f32")
    require(inputs["waveforms_sha256"] == metadata["waveforms_sha256"] == locked["waveforms_sha256_from_metadata"],
            "Checksum toàn waveform dev-full sai.")
    models: dict[str, Any] = {}
    references: dict[str, list[float]] = {}
    for seed, model_name in zip((2026, 2027, 2028), BMODELS):
        report_path = ROOT / f"reports/b0_v2_ext_seed{seed}.json"
        report = read_json(report_path)
        point = report["checkpoints"]["best"]
        require(point["generation"] == BEST_GENERATIONS[seed], "Generation best B0 khác đăng ký.")
        checkpoint = ROOT / f"checkpoints/b0_v2_seed{seed}" / point["generation"] / "checkpoint.pt"
        actual = sha256_file(checkpoint)
        require(actual == point["checkpoint_sha256"], "Checksum checkpoint B0 sai.")
        development = report["controller"]["best"]["development"]
        _validate_reference(development["records"], rows)
        references[model_name] = [r["score"] for r in development["records"]]
        models[model_name] = {"checkpoint": str(checkpoint), "checkpoint_sha256": actual,
            "reference_report_sha256": sha256_file(report_path), "generation": point["generation"],
            "decision_threshold": development["decision_threshold"]}
    report_path = ROOT / f"kaggle_outputs/xlsr-d028-X1-L8-lr1e-5-seed2026/reports/xlsr_X1-L8-lr1e-5_seed2026.json"
    report = read_json(report_path)
    generation = "g-ce724d6e82c8457d8ce3e3d7d2154349"
    checkpoint = ROOT / f"kaggle_outputs/xlsr-d028-X1-L8-lr1e-5-seed2026/checkpoints/xlsr_X1-L8-lr1e-5_seed2026/{generation}/checkpoint.pt"
    require(sha256_file(checkpoint) == X_CHECKPOINT_SHA, "Checksum checkpoint XLS-R sai.")
    development = report["controller"]["best"]["development"]
    require(report["controller"]["best"]["generation"] == generation, "Best XLS-R khác đăng ký.")
    require(development["decision_threshold"] == 7.74885248054705e-05, "Threshold XLS-R khác đăng ký.")
    _validate_reference(development["records"], rows)
    references[XMODEL] = [r["score"] for r in development["records"]]
    models[XMODEL] = {"checkpoint": str(checkpoint), "checkpoint_sha256": X_CHECKPOINT_SHA,
        "reference_report_sha256": sha256_file(report_path), "generation": generation,
        "decision_threshold": development["decision_threshold"]}
    pretrained = {name: sha256_file(WEIGHTS / name) for name in PRETRAINED}
    require(pretrained == PRETRAINED, "SHA-256 pretrained không khớp D021.")
    source_paths = ("src/data/packed.py", "src/data/amplitude.py", "src/data/segment.py",
                    "src/data/xlsr.py", "src/models/lfcc_lcnn.py", "src/models/xlsr.py", "configs/lfcc_lcnn.json")
    identity = {"version": 1, "task": "T19b", "use_test": False, "device": "cpu", "torch_threads": 8,
        "b0_batch_size": 8, "xlsr_batch_size": 2, "maximum_block_samples": 200,
        "inputs": inputs, "models": models, "pretrained": pretrained,
        "filters": {condition: filter_parameters(condition) for condition in CONDITIONS},
        "runner_sha256": sha256_file(Path(__file__)), "source_sha256": {p: sha256_file(ROOT / p) for p in source_paths},
        "torch_version": str(torch.__version__), "torchaudio_version": str(torchaudio.__version__),
        "logit_convention": "float64 log(p)-log1p(-p); chỉ đúng0/1 thay bằng nextafter(0,1)/nextafter(1,0)",
        "normalization": "processor_zero_mean_unit_variance", "data_seed": 2026}
    ds = open_dev_artifact(artifact)
    require(len(ds) == len(rows) and len({r.shard for r in ds.records}) == 186, "Sai mẫu/shard packed.")
    for packed, row in zip(ds.records, rows):
        require((packed.file, packed.speaker_id, packed.utt_type, packed.binary_label) ==
                (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])), "Metadata packed/manifest lệch.")
    return identity, rows, selected, ds, references


def _validate_reference(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    require(len(records) == len(rows), "Report gốc không đủ dev-full.")
    for record, row in zip(records, rows):
        require((record["file"], record["speaker_id"], record["utt_type"], int(record["label"])) ==
                (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])),
                "Record gốc không khớp thứ tự/nhãn manifest dev-full.")
    stable_logit([r["score"] for r in records])


class Models:
    def __init__(self, identity: dict[str, Any], dataset: PackedManifestDataset) -> None:
        self.identity, self.dataset = identity, dataset
        self.xlsr: torch.nn.Module | None = None
        self.b0: dict[str, torch.nn.Module] = {}
        self.lfcc: torch.nn.Module | None = None

    def load_xlsr(self) -> torch.nn.Module:
        if self.xlsr is None:
            torch.manual_seed(2026)
            model = XLSRBinaryClassifier(load_xlsr_backbone(WEIGHTS, local_files_only=True),
                                        num_unfrozen_layers=8, hidden_features=128, dropout=.2)
            payload = torch.load(self.identity["models"][XMODEL]["checkpoint"], map_location="cpu",
                                 weights_only=False, mmap=True)
            saved = payload["model"]
            require(set(saved) == {n for n, p in model.named_parameters() if p.requires_grad},
                    "Tập tham số trainable-only XLS-R lệch.")
            missing, unexpected = model.load_state_dict(saved, strict=False)
            require(not unexpected and set(missing) == {n for n, p in model.named_parameters() if not p.requires_grad},
                    "Checkpoint XLS-R thiếu/sai shape tham số.")
            require(sum(p.numel() for n, p in model.named_parameters() if n in saved) == 101032193,
                    "Số tham số checkpoint L8 không khớp.")
            model.eval().requires_grad_(False)
            self.xlsr = model
            del payload, saved
            gc.collect()
        return self.xlsr

    def load_b0(self, names: list[str]) -> None:
        config = read_json(ROOT / "configs/lfcc_lcnn.json")
        if self.lfcc is None:
            self.lfcc = build_lfcc_transform(config["lfcc"]).eval()
        for name in names:
            if name not in self.b0:
                payload = torch.load(self.identity["models"][name]["checkpoint"], map_location="cpu",
                                     weights_only=False, mmap=True)
                model = build_lcnn_from_config(config)
                model.load_state_dict(payload["model"], strict=True)
                model.eval().requires_grad_(False)
                self.b0[name] = model
                del payload
        gc.collect()

    def predict(self, indices: list[int], names: list[str], condition: str) -> dict[str, list[dict[str, Any]]]:
        is_xlsr = names == [XMODEL]
        require(is_xlsr or set(names).issubset(BMODELS), "Trộn hai họ mô hình trong một khối.")
        if is_xlsr:
            model = self.load_xlsr()
        else:
            self.load_b0(names)
        results: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        batch_size = 2 if is_xlsr else 8
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                samples = [self.dataset[i] for i in batch_indices]
                waveform = torch.stack([apply_condition(s["waveform"], condition, s["utt_type"]) for s in samples])
                if is_xlsr:
                    waveform = apply_xlsr_waveform_normalization(waveform, "processor_zero_mean_unit_variance")
                    outputs = {XMODEL: model(waveform)}
                else:
                    features = lfcc_features(self.lfcc, waveform).unsqueeze(1)
                    outputs = {name: self.b0[name](features) for name in names}
                for name, logits in outputs.items():
                    require(bool(torch.isfinite(logits).all()), "Logit không hữu hạn: " + name)
                    probabilities = torch.sigmoid(logits).tolist()
                    for sample, probability, raw in zip(samples, probabilities, logits.tolist()):
                        results[name].append({"file": sample["file"], "score": float(probability), "raw_logit": float(raw)})
        return results


@contextmanager
def job_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ValueError("Đang có job T19b khác; không chạy trùng.") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class Probe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.deadline = Deadline(args.deadline_hours)
        self.started_clock = time.perf_counter()
        self.stage_clock: float | None = None
        self.stage: str | None = None
        self.reproduction: dict[str, Any] = {}
        self.completed: list[str] = []
        self.invocation_path = CACHE / "invocations" / (uuid.uuid4().hex + ".json")
        self.invocation = {"started_utc": utc(), "pid": os.getpid(), "status": "running",
                           "resume": args.resume, "resume_reason": args.resume_reason, "stages": {}}
        for previous in (CACHE / "invocations").glob("*.json"):
            item = read_json(previous)
            if item["status"] == "running":
                require(args.resume, "Phiên trước chưa kết thúc; cần --resume và ghi lý do.")
                item.update(status="interrupted", detected_utc=utc(), interruption_reason=args.resume_reason)
                atomic_json(previous, item)
        atomic_json(self.invocation_path, self.invocation)
        self.emit("preflight_started")
        began = time.perf_counter()
        self.identity, self.rows, self.selected, self.dataset, references = prepare_inputs()
        self.rehash_and_setup_seconds = time.perf_counter() - began
        self.references = references
        identity_path = CACHE / "identity.json"
        if identity_path.exists():
            require(args.resume, "Đã có identity T19b; phải dùng --resume.")
            require(read_json(identity_path) == self.identity, "Identity/checkpoint/bộ lọc khác; từ chối resume.")
        else:
            require(not any((CACHE / "blocks").rglob("*.json")), "Có khối nhưng thiếu identity.")
            atomic_json(identity_path, self.identity, replace=False)
        self.store = BlockStore(CACHE / "blocks", self.identity)
        self.validate_existing()
        self.models = Models(self.identity, self.dataset)
        self.emit("preflight_complete", seconds=self.rehash_and_setup_seconds,
                  identity_sha256=self.store.identity_sha, subset_sha256=SUBSET_SHA)
        self.publish("running")

    def indices(self, name: str, condition: str) -> list[int]:
        values = self.selected if name == XMODEL else list(range(len(self.rows)))
        return [i for i in values if self.rows[i]["utt_type"] == "bonafide"] if condition == "roundtrip_40k" else values

    def validate_existing(self) -> None:
        expected: dict[Path, tuple[str, str, int, list[str]]] = {}
        for name in (*BMODELS, XMODEL):
            for condition in CONDITIONS:
                indices = self.indices(name, condition)
                for start in range(0, len(indices), 200):
                    files = [self.rows[i]["file"] for i in indices[start:start + 200]]
                    expected[self.store.path(name, condition, start)] = (name, condition, start, files)
        for path in self.store.root.rglob("*.json"):
            require(path in expected, "Có khối ngoài kế hoạch: " + str(path))
            self.store.load(*expected[path])

    def emit(self, event: str, **fields: Any) -> None:
        record = {"utc": utc(), "event": event, "stage": self.stage, **fields}
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        with (CACHE / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.invocation["last_event_utc"] = record["utc"]
        if self.stage_clock is not None:
            self.invocation["stages"][self.stage]["elapsed_seconds"] = time.perf_counter() - self.stage_clock
        atomic_json(self.invocation_path, self.invocation)
        atomic_json(CACHE / "progress.json", record)

    def read_scores(self, name: str, condition: str, *, count: int | None = None) -> list[float] | None:
        indices = self.indices(name, condition)
        if count is not None:
            indices = indices[:count]
        scores: list[float] = []
        for start in range(0, len(indices), 200):
            files = [self.rows[i]["file"] for i in indices[start:start + 200]]
            value = self.store.load(name, condition, start, files)
            if value is None:
                return None
            scores.extend(r["score"] for r in value["records"])
        return scores

    def verify_original(self, name: str, count: int | None = None) -> None:
        indices = self.indices(name, "original")
        if count is not None:
            indices = indices[:count]
        observed = self.read_scores(name, "original", count=count)
        require(observed is not None, "Thiếu khối original để kiểm tái tạo.")
        result = check_reproduction(observed, [self.references[name][i] for i in indices],
                                   self.identity["models"][name]["decision_threshold"], model=name)
        key = name + ("_first200" if count == 200 else "")
        self.reproduction[key] = result
        atomic_json(CACHE / ("reproduction_" + key + ".json"), result)
        self.emit("reproduction_checked", model=name, **result)
        if not result["passed"]:
            raise ReproductionFailure("Tái tạo original không đạt: " + key)

    def collect_results(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name in (*BMODELS, XMODEL):
            full = self.indices(name, "original")
            original = self.read_scores(name, "original")
            available = {}
            for condition in CONDITIONS:
                scores = self.read_scores(name, condition)
                if scores is None:
                    continue
                newly_evaluated = self.indices(name, condition)
                if condition == "roundtrip_40k":
                    if original is None:
                        continue
                    overlay = dict(zip(newly_evaluated, scores))
                    scores = [overlay.get(i, value) for i, value in zip(full, original)]
                summary = summarize_scores([self.rows[i] for i in full], scores,
                                           self.identity["models"][name]["decision_threshold"])
                summary.update(forward_evaluated_samples=len(newly_evaluated),
                               forward_evaluated_shards=len({self.rows[i]["shard"] for i in newly_evaluated}),
                               scores_reused_from_original=len(full) - len(newly_evaluated))
                available[condition] = summary
            if available:
                results[name] = {"conditions": available, "indicators": indicators(available)}
        return results

    def publish(self, status: str, error: str | None = None) -> dict[str, Any]:
        invocations = [read_json(p) for p in sorted((CACHE / "invocations").glob("*.json"))]
        stage_times = {phase: sum(item.get("stages", {}).get(phase, {}).get("elapsed_seconds", 0) for item in invocations)
                       for phase in ("i", "ii", "iii", "iv", "v", "vi")}
        result = {"task": "T19b", "status": status, "updated_utc": utc(), "error": error,
            "use_test": False, "trained": False, "kaggle_operations": False, "gates_changed": False,
            "completed_stages": self.completed, "current_stage": self.stage, "identity": self.identity,
            "subset_files": [self.rows[i]["file"] for i in self.selected], "subset_files_sha256": SUBSET_SHA,
            "data": {"dev_samples": 38777, "dev_shards": 186, "xlsr_samples": 3120,
                     "xlsr_shards": len({self.rows[i]["shard"] for i in self.selected})},
            "reproduction": self.reproduction, "results": self.collect_results(),
            "timing": {"invocation_wall_seconds": time.perf_counter() - self.started_clock,
                       "rehash_and_setup_seconds_this_invocation": self.rehash_and_setup_seconds,
                       "recorded_active_stage_seconds": stage_times,
                       "note": "Tổng thời gian pha ghi đến event cuối; không tính downtime/đuôi chưa commit khi mất điện.",
                       "deadline_hours_this_invocation": self.args.deadline_hours,
                       "seconds_left": self.deadline.end - self.deadline.clock()},
            "sessions": invocations, "resume_count": sum(bool(v["resume"]) for v in invocations),
            "interruption_count": sum(v["status"] == "interrupted" for v in invocations),
            "blocks_root": str(self.store.root), "filters": self.identity["filters"],
            "logit_convention": self.identity["logit_convention"],
            "indicator_rules": {"roundtrip_fpr_increase_percentage_points": 10,
                               "lowpass7_vc_eer_ratio_at_least": 3, "lowpass7_ap_eer_ratio_less_than": 1.5,
                               "lowpass4_robustness_only": True},
            "interpretation_limit": "Chỉ báo theo ngưỡng khai báo trước; không kết luận có/không shortcut."}
        atomic_json(ROOT / "reports/shortcut_probe_dev.json", result)
        write_markdown(result)
        atomic_json(CACHE / "status.json", {"status": status, "current_stage": self.stage,
            "completed_stages": self.completed, "updated_utc": utc(), "error": error})
        return result

    def phase(self, phase: str, action: Callable[[], None]) -> None:
        self.stage, self.stage_clock = phase, time.perf_counter()
        self.invocation["stages"][phase] = {"started_utc": utc(), "elapsed_seconds": 0}
        self.emit("stage_started")
        action()
        self.emit("stage_completed")
        self.completed.append(phase)
        report = self.publish("running")
        if phase != "i":
            atomic_json(CACHE / f"interim_{phase}.json", report)
        self.stage_clock = None

    def blocks(self, names: Sequence[str], condition: str, limit: int | None = None) -> None:
        process_blocks(self.store, names, condition, self.indices(names[0], condition), self.rows,
                       self.models.predict, self.deadline, self.emit, limit=limit)

    def run(self) -> int:
        try:
            def first() -> None:
                self.blocks([XMODEL], "original", limit=200)
                self.verify_original(XMODEL, count=200)
            self.phase("i", first)
            def rest_original() -> None:
                self.blocks([XMODEL], "original")
                self.verify_original(XMODEL)
            self.phase("ii", rest_original)
            self.phase("iii", lambda: self.blocks([XMODEL], "roundtrip_40k"))
            self.phase("iv", lambda: self.blocks([XMODEL], "lowpass_7k"))
            def b0() -> None:
                self.blocks(BMODELS, "original")
                for name in BMODELS:
                    self.verify_original(name)
                for condition in CONDITIONS[1:]:
                    self.blocks(BMODELS, condition)
            self.phase("v", b0)
            self.phase("vi", lambda: self.blocks([XMODEL], "lowpass_4k"))
            self.invocation.update(status="complete", ended_utc=utc())
            self.emit("job_completed")
            self.publish("complete")
            return 0
        except DeadlineReached as exc:
            self.invocation.update(status="paused_deadline", ended_utc=utc())
            self.emit("deadline_stop", error=str(exc))
            self.publish("paused_deadline", str(exc))
            return 2
        except Exception as exc:
            self.invocation.update(status="blocked_reproduction" if isinstance(exc, ReproductionFailure) else "blocked_error",
                                   ended_utc=utc())
            self.emit("job_stopped", error=str(exc))
            self.publish(self.invocation["status"], str(exc))
            return 3 if isinstance(exc, ReproductionFailure) else 1
        finally:
            self.dataset.close()


def write_markdown(report: dict[str, Any]) -> None:
    lines = ["# T19b — Phép thử shortcut trên dev-full", "",
        "Trạng thái: " + report["status"], "Chỉ đánh giá CPU; không train/test/Kaggle.", "",
        "B0: 38.777 mẫu/186 shard mỗi seed; XLS-R: 3.120 mẫu/169 shard. Threshold cố định từ checkpoint.",
        "Tập con SHA-256: " + SUBSET_SHA, "Giai đoạn đã xong: " + ", ".join(report["completed_stages"]), "",
        "## Tái tạo original", "", json.dumps(report["reproduction"], ensure_ascii=False), "",
        "## Kết quả", "", "| Mô hình | Điều kiện | EER tổng % | EER VC % | EER AP % | EER replay % | FPR % | Bỏ lọt VC % | AP % | Replay % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["results"].items():
        for condition, result in value["conditions"].items():
            e = result["per_attack_eer"]; g = result["groups"]
            numbers = [result["overall_eer"], e["voice_conversion"]["eer"], e["adversarial_attack"]["eer"], e["replay"]["eer"],
                       g["bonafide"]["fpr"], g["voice_conversion"]["miss_rate"], g["adversarial_attack"]["miss_rate"], g["replay"]["miss_rate"]]
            lines.append("| " + name + " | " + condition + " | " + " | ".join(f"{100 * n:.6f}" for n in numbers) + " |")
        lines += ["", name + ": " + json.dumps(value["indicators"], ensure_ascii=False), ""]
    lines += ["", "## Bộ lọc và số đo", "", json.dumps(report["filters"], ensure_ascii=False, indent=2), "",
        "FIR Hann513tap, pha tuyến tính, tích chập FFT tuyến tính/bù trễ256mẫu, zero-extension. Sau biến đổi áp lại peak.",
        "Đáp ứng dB tại3/4/6/7/7,5kHz nằm ở khối bộ lọc và JSON. Tham số roundtrip là mặc định torchaudio của pipeline.",
        "P5/P50/P95 xác suất và logit từng nhóm, checksum đầu vào/checkpoint/pretrained, threshold, số khối và thời gian có trong JSON/blocks.",
        "Số lần ngắt: " + str(report["interruption_count"]) + "; số lần resume: " + str(report["resume_count"]),
        json.dumps(report["timing"], ensure_ascii=False, indent=2), "",
        "Lowpass4k chỉ kiểm độ bền. Không kết luận có hoặc không shortcut."]
    if report.get("error"):
        lines += ["", "Điểm dừng: " + report["error"]]
    path = ROOT / "reports/shortcut_probe_dev.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="T19b: CPU dev-only, theo khối và tiếp tục được.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-reason", default="Tiếp tục phiên trước chưa có bản ghi kết thúc.")
    parser.add_argument("--deadline-hours", type=float, default=11)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        with job_lock(CACHE / "job.lock"):
            return Probe(args).run()
    except Exception as exc:
        # Không ghi đè khối. Giữ kết quả đã công bố nếu setup/resume bị từ chối.
        report_path = ROOT / "reports/shortcut_probe_dev.json"
        previous = read_json(report_path) if report_path.exists() else {}
        previous.update(task="T19b", status="blocked_setup_or_resume", error=str(exc), updated_utc=utc(), use_test=False)
        atomic_json(report_path, previous)
        atomic_json(CACHE / "status.json", {"status": "blocked_setup_or_resume", "error": str(exc), "updated_utc": utc()})
        print(json.dumps({"utc": utc(), "event": "setup_or_resume_rejected", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
