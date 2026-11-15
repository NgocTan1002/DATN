"""T22: phép thử dev-only cho X1 cuối; CPU, theo khối, không train/mạng."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0")

import numpy as np
import torch
import torchaudio
from scripts import probe_shortcut_dev as shared
from src.data.packed import PackedManifestDataset
from src.data.xlsr import apply_xlsr_waveform_normalization
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone

CACHE = ROOT / ".cache/t22"
REPORT_JSON = ROOT / "reports/shortcut_probe_dev_x1_final.json"
REPORT_MD = ROOT / "reports/shortcut_probe_dev_x1_final.md"
MODEL = "X1-L8-lr3e-5-seed2026"
GENERATION = "g-eb35f3c2379f4a47bd22744c9be1dd8f"
CHECKPOINT_SHA = "d4a60373a88e0e72192d6040441d6adee8928eddb3a8a6129ccf5ac18dc5da50"
THRESHOLD = 0.00015767214536590907
RUN_ROOT = ROOT / "kaggle_outputs/xlsr-d028-X1-L8-lr3e-5-seed2026"
REFERENCE_REPORT = RUN_ROOT / "reports/xlsr_X1-L8-lr3e-5_seed2026.json"
CHECKPOINT = RUN_ROOT / "checkpoints/xlsr_X1-L8-lr3e-5_seed2026" / GENERATION / "checkpoint.pt"
WEIGHTS = ROOT / "data/raw/xlsr/wav2vec2-xls-r-300m"
PRETRAINED = {
    "pytorch_model.bin": "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0",
    "config.json": "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
    "preprocessor_config.json": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
}
MANIFEST_SHA = "a79f1bb2595bc46b08ce30b8652234b1858cc26e558e2ccd33632435df1a5360"
SUBSET_SHA = "869d8f87c8ffa6c52efc6bc619048dc85dd1eda6d9148fffeefc71ec7d802dfe"
CONDITIONS = ("original", "roundtrip_40k", "lowpass_7k", "lowpass_4k")
INDICATOR_RULES = {
    "roundtrip_fpr_increase_percentage_points": 10,
    "lowpass7_vc_eer_ratio_at_least": 3,
    "lowpass7_ap_eer_ratio_less_than": 1.5,
    "lowpass4_robustness_only": True,
}
STAGES = ("i", "ii", "iii", "iv", "v")
require = shared.require


def locked_specification(prior: dict[str, Any]) -> dict[str, Any]:
    filters = {condition: shared.filter_parameters(condition) for condition in CONDITIONS}
    require(filters == prior["filters"], "Bộ lọc khác đặc tả T19b; dừng.")
    require(INDICATOR_RULES == prior["indicator_rules"], "Ngưỡng chỉ báo khác T19b; dừng.")
    return filters


def check_logit_reproduction(raw_logits: Sequence[float], reference: Sequence[float],
                            threshold: float = THRESHOLD, *,
                            observed_scores: Sequence[float] | None = None) -> dict[str, Any]:
    current = np.asarray(raw_logits, dtype=np.float64)
    old = shared.stable_logit(reference)
    require(current.shape == old.shape and current.size > 0 and bool(np.isfinite(current).all()),
            "Thiếu hoặc sai logit khi kiểm tái tạo XLS-R.")
    threshold_logit = shared.stable_logit([threshold])[0]
    if observed_scores is None:
        decisions = current >= threshold_logit
    else:
        probabilities = np.asarray(observed_scores, dtype=np.float64)
        require(probabilities.shape == current.shape, "Số xác suất/logit không khớp.")
        shared.stable_logit(probabilities)
        decisions = probabilities >= threshold
    flips = decisions != (np.asarray(reference) >= threshold)
    differences = np.abs(current - old)
    outside = np.abs(old - threshold_logit)[flips] > .05
    return {"passed": bool(differences.max() <= .05 and not outside.any()),
            "samples": int(current.size), "max_abs_logit_difference": float(differences.max()),
            "median_abs_logit_difference": float(np.median(differences)),
            "decision_flips": int(flips.sum()), "flips_outside_0_05_margin": int(outside.sum()),
            "observed_logit_source": "block.raw_logit", "reference_logit_source": "stable_logit(saved_score)"}


def prepare_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[int], PackedManifestDataset, list[float], dict[str, Any]]:
    prior = shared.read_json(ROOT / "reports/shortcut_probe_dev.json")
    filters = locked_specification(prior)
    rows = shared.read_dev_manifest(ROOT / "data/manifests/closed_dev_full_v1.csv", MANIFEST_SHA)
    selected = shared.subset_indices(rows)
    files = [rows[i]["file"] for i in selected]
    require(len(rows) == 38777 and len(files) == 3120 and shared.file_list_sha(files) == SUBSET_SHA,
            "Manifest/tập con không khớp T19b.")
    require(files == prior["subset_files"] and prior["subset_files_sha256"] == SUBSET_SHA,
            "Thứ tự tập con khác T19b.")
    require(len({rows[i]["shard"] for i in selected}) == 169, "Số shard tập con khác T19b.")
    artifact = ROOT / "data/processed/packed/closed_dev_full_v1"
    metadata = shared.read_json(artifact / "metadata.json")
    require(metadata["split"] == "closed_dev" and metadata["target_sample_rate"] == 16000,
            "Chỉ chấp nhận artifact dev-full 16kHz, không đọc test.")
    inputs = {"manifest_sha256": MANIFEST_SHA, "subset_files_sha256": SUBSET_SHA}
    for key, path in (("audio_config_sha256", ROOT / "configs/audio.json"),
                      ("metadata_sha256", artifact / "metadata.json"),
                      ("index_sha256", artifact / "index.csv"),
                      ("waveforms_sha256", artifact / "waveforms.f32")):
        inputs[key] = shared.sha256_file(path)
    require(inputs == prior["identity"]["inputs"] and inputs["waveforms_sha256"] == metadata["waveforms_sha256"],
            "Checksum artifact/audio khác T19b.")
    require(shared.sha256_file(CHECKPOINT) == CHECKPOINT_SHA, "Checksum best checkpoint X1 cuối sai.")
    pretrained = {name: shared.sha256_file(WEIGHTS / name) for name in PRETRAINED}
    require(pretrained == PRETRAINED, "Checksum pretrained khác D021.")
    report = shared.read_json(REFERENCE_REPORT)
    best = report["controller"]["best"]
    development = best["development"]
    require(best["generation"] == GENERATION and best["epoch"] == 2 and
            development["decision_threshold"] == THRESHOLD, "Best/threshold khác giá trị khóa T22.")
    require(report["identity"]["run_id"] == "X1-L8-lr3e-5" and report["identity"]["training_seed"] == 2026,
            "Run/seed tham chiếu sai.")
    require(len(development["records"]) == len(rows), "Report tham chiếu không đủ dev-full.")
    for record, row in zip(development["records"], rows, strict=True):
        require((record["file"], record["speaker_id"], record["utt_type"], record["label"]) ==
                (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])),
                "Record tham chiếu lệch thứ tự/nhãn manifest dev-full.")
    references = [record["score"] for record in development["records"]]
    shared.stable_logit(references)
    sources = ["scripts/probe_shortcut_dev.py", "src/data/packed.py", "src/data/amplitude.py",
               "src/data/segment.py", "src/data/xlsr.py", "src/models/xlsr.py"]
    identity = {"version": 1, "task": "T22", "use_test": False, "device": "cpu", "torch_threads": 8,
        "xlsr_batch_size": 2, "maximum_block_samples": 200, "inputs": inputs, "pretrained": pretrained,
        "filters": filters, "indicator_rules": INDICATOR_RULES,
        "models": {MODEL: {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA,
                           "generation": GENERATION, "decision_threshold": THRESHOLD,
                           "reference_report_sha256": shared.sha256_file(REFERENCE_REPORT),
                           "unfrozen_layers": 8, "trainable_parameters": 101032193,
                           "head_hidden_features": 128, "dropout": .2}},
        "runner_sha256": shared.sha256_file(Path(__file__)),
        "source_sha256": {name: shared.sha256_file(ROOT / name) for name in sources},
        "prior_T19b_report_sha256": shared.sha256_file(ROOT / "reports/shortcut_probe_dev.json"),
        "torch_version": str(torch.__version__), "torchaudio_version": str(torchaudio.__version__),
        "normalization": "processor_zero_mean_unit_variance", "data_seed": 2026,
        "logit_convention": "raw_logit CPU so với float64 stable_logit(score lưu); ngưỡng0,05"}
    dataset = shared.open_dev_artifact(artifact)
    try:
        require(len(dataset) == len(rows) and len({r.shard for r in dataset.records}) == 186,
                "Số mẫu/shard packed sai.")
        for packed, row in zip(dataset.records, rows, strict=True):
            require((packed.file, packed.speaker_id, packed.utt_type, packed.binary_label) ==
                    (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])),
                    "Metadata packed khác manifest.")
    except Exception:
        dataset.close()
        raise
    return identity, rows, selected, dataset, references, prior


class FinalModel:
    def __init__(self, identity: dict[str, Any], dataset: PackedManifestDataset) -> None:
        self.identity, self.dataset = identity, dataset
        self.model: torch.nn.Module | None = None

    def load(self) -> torch.nn.Module:
        if self.model is None:
            require(shared.sha256_file(CHECKPOINT) == CHECKPOINT_SHA, "Checkpoint đổi trước khi nạp.")
            torch.manual_seed(2026)
            model = XLSRBinaryClassifier(load_xlsr_backbone(WEIGHTS, local_files_only=True),
                                        num_unfrozen_layers=8, hidden_features=128, dropout=.2)
            payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
            saved = payload["model"]
            expected = {n for n, p in model.named_parameters() if p.requires_grad}
            require(set(saved) == expected and sum(p.numel() for p in saved.values()) == 101032193,
                    "Checkpoint không đúng tập trainable-only 8 tầng.")
            require(all(p.dtype == torch.float32 for p in saved.values()), "Checkpoint không phải float32.")
            missing, unexpected = model.load_state_dict(saved, strict=False)
            require(not unexpected and set(missing) == {n for n, p in model.named_parameters() if not p.requires_grad},
                    "Checkpoint thiếu/sai tham số.")
            model.eval().requires_grad_(False)
            self.model = model
            del payload, saved
            gc.collect()
        return self.model

    def predict(self, indices: list[int], names: list[str], condition: str) -> dict[str, list[dict[str, Any]]]:
        require(names == [MODEL], "T22 chỉ chấm X1 cuối, không chạy B0.")
        model = self.load()
        records = []
        with torch.inference_mode():
            for start in range(0, len(indices), 2):
                samples = [self.dataset[i] for i in indices[start:start + 2]]
                waveform = torch.stack([shared.apply_condition(s["waveform"], condition, s["utt_type"]) for s in samples])
                waveform = apply_xlsr_waveform_normalization(waveform, "processor_zero_mean_unit_variance")
                logits = model(waveform)
                require(bool(torch.isfinite(logits).all()), "Logit X1 cuối không hữu hạn.")
                for sample, score, raw in zip(samples, torch.sigmoid(logits).tolist(), logits.tolist(), strict=True):
                    records.append({"file": sample["file"], "score": float(score), "raw_logit": float(raw)})
        return {MODEL: records}


class Probe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.deadline = shared.Deadline(args.deadline_hours)
        self.started_clock = time.perf_counter()
        self.stage: str | None = None
        self.stage_clock: float | None = None
        self.completed: list[str] = []
        self.reproduction: dict[str, Any] = {}
        self.invocation_path = CACHE / "invocations" / (uuid.uuid4().hex + ".json")
        self.invocation = {"started_utc": shared.utc(), "pid": os.getpid(), "status": "running",
                           "resume": args.resume, "resume_reason": args.resume_reason, "stages": {}}
        for path in (CACHE / "invocations").glob("*.json"):
            previous = shared.read_json(path)
            if previous["status"] == "running":
                require(args.resume, "Phiên trước bị ngắt; dùng --resume và ghi lý do.")
                previous.update(status="interrupted", interruption_reason=args.resume_reason, detected_utc=shared.utc())
                shared.atomic_json(path, previous)
        shared.atomic_json(self.invocation_path, self.invocation)
        self.emit("preflight_started")
        began = time.perf_counter()
        self.identity, self.rows, self.selected, self.dataset, self.references, self.prior = prepare_inputs()
        self.setup_seconds = time.perf_counter() - began
        try:
            path = CACHE / "identity.json"
            if path.exists():
                require(args.resume, "Đã có kết quả T22; cần --resume.")
                require(shared.read_json(path) == self.identity, "Identity/checkpoint/bộ lọc khác; từ chối resume.")
            else:
                require(not any((CACHE / "blocks").rglob("*.json")), "Có khối nhưng thiếu identity T22.")
                shared.atomic_json(path, self.identity, replace=False)
            self.store = shared.BlockStore(CACHE / "blocks", self.identity)
            self.validate_existing()
            self.model = FinalModel(self.identity, self.dataset)
            self.emit("preflight_complete", seconds=self.setup_seconds, subset_sha256=SUBSET_SHA)
            self.publish("running")
        except Exception:
            self.dataset.close()
            raise

    def indices(self, condition: str) -> list[int]:
        return [i for i in self.selected if self.rows[i]["utt_type"] == "bonafide"] if condition == "roundtrip_40k" else self.selected

    def validate_existing(self) -> None:
        expected = {}
        for condition in CONDITIONS:
            indices = self.indices(condition)
            for start in range(0, len(indices), 200):
                files = [self.rows[i]["file"] for i in indices[start:start + 200]]
                expected[self.store.path(MODEL, condition, start)] = (MODEL, condition, start, files)
        for path in self.store.root.rglob("*.json"):
            require(path in expected, "Có khối ngoài đặc tả T22: " + str(path))
            self.store.load(*expected[path])

    def emit(self, event: str, **fields: Any) -> None:
        record = {"utc": shared.utc(), "event": event, "stage": self.stage, **fields}
        print(shared.canonical(record).decode(), flush=True)
        with (CACHE / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(shared.canonical(record).decode() + "\n")
        self.invocation["last_event_utc"] = record["utc"]
        if self.stage_clock is not None:
            self.invocation["stages"][self.stage]["elapsed_seconds"] = time.perf_counter() - self.stage_clock
        shared.atomic_json(self.invocation_path, self.invocation)
        shared.atomic_json(CACHE / "progress.json", record)

    def records(self, condition: str, count: int | None = None) -> list[dict[str, Any]] | None:
        indices = self.indices(condition)[:count]
        result = []
        for start in range(0, len(indices), 200):
            files = [self.rows[i]["file"] for i in indices[start:start + 200]]
            block = self.store.load(MODEL, condition, start, files)
            if block is None:
                return None
            result.extend(block["records"])
        return result

    def verify_original(self, count: int | None = None) -> None:
        records = self.records("original", count)
        require(records is not None, "Thiếu khối original để kiểm tái tạo.")
        indices = self.indices("original")[:count]
        value = check_logit_reproduction([r["raw_logit"] for r in records],
                    [self.references[i] for i in indices], observed_scores=[r["score"] for r in records])
        key = "first200" if count == 200 else "full_subset"
        self.reproduction[key] = value
        shared.atomic_json(CACHE / ("reproduction_" + key + ".json"), value)
        self.emit("reproduction_checked", scope=key, **value)
        if not value["passed"]:
            raise shared.ReproductionFailure("Tái tạo logit X1 cuối không đạt: " + key)

    def collect_results(self) -> dict[str, Any]:
        original = self.records("original")
        available = {}
        for condition in CONDITIONS:
            records = self.records(condition)
            if records is None:
                continue
            indices = self.indices(condition)
            if condition == "roundtrip_40k":
                if original is None:
                    continue
                overlay = dict(zip(indices, records, strict=True))
                records = [overlay.get(i, r) for i, r in zip(self.selected, original, strict=True)]
            summary = shared.summarize_scores([self.rows[i] for i in self.selected], [r["score"] for r in records], THRESHOLD)
            summary.update(forward_evaluated_samples=len(indices),
                           forward_evaluated_shards=len({self.rows[i]["shard"] for i in indices}),
                           scores_reused_from_original=len(self.selected) - len(indices))
            available[condition] = summary
        return {"conditions": available, "indicators": shared.indicators(available)}

    def publish(self, status: str, error: str | None = None) -> dict[str, Any]:
        sessions = [shared.read_json(p) for p in sorted((CACHE / "invocations").glob("*.json"))]
        result = {"task": "T22", "model": MODEL, "status": status, "updated_utc": shared.utc(), "error": error,
            "use_test": False, "trained": False, "kaggle_operations": False, "gates_changed": False,
            "completed_stages": self.completed, "current_stage": self.stage, "identity": self.identity,
            "subset_files": [self.rows[i]["file"] for i in self.selected], "subset_files_sha256": SUBSET_SHA,
            "data": {"dev_samples": 38777, "dev_shards": 186, "xlsr_samples": 3120,
                     "xlsr_shards": len({self.rows[i]["shard"] for i in self.selected})},
            "filters": self.identity["filters"], "indicator_rules": INDICATOR_RULES,
            "reproduction": self.reproduction, "results": {MODEL: self.collect_results()},
            "comparison_data_only": {shared.XMODEL: self.prior["results"][shared.XMODEL]},
            "sessions": sessions, "resume_count": sum(bool(s["resume"]) for s in sessions),
            "interruption_count": sum(s["status"] == "interrupted" for s in sessions),
            "timing": {"invocation_wall_seconds": time.perf_counter() - self.started_clock,
                       "rehash_and_setup_seconds_this_invocation": self.setup_seconds,
                       "recorded_active_stage_seconds": {stage: sum(s["stages"].get(stage, {}).get("elapsed_seconds", 0)
                                                                   for s in sessions) for stage in STAGES},
                       "deadline_hours_this_invocation": self.args.deadline_hours,
                       "seconds_left": self.deadline.end - self.deadline.clock()},
            "blocks_root": str(self.store.root),
            "interpretation_limit": "Chỉ báo khai báo trước; lowpass4k chỉ đo độ bền. Không kết luận có/không shortcut."}
        shared.atomic_json(REPORT_JSON, result)
        write_markdown(result)
        shared.atomic_json(CACHE / "status.json", {k: result[k] for k in ("status", "current_stage", "completed_stages", "updated_utc", "error")})
        return result

    def blocks(self, condition: str, limit: int | None = None) -> None:
        shared.process_blocks(self.store, [MODEL], condition, self.indices(condition), self.rows,
                              self.model.predict, self.deadline, self.emit, limit=limit)

    def phase(self, stage: str, action: Callable[[], None]) -> None:
        self.stage, self.stage_clock = stage, time.perf_counter()
        self.invocation["stages"][stage] = {"started_utc": shared.utc(), "elapsed_seconds": 0}
        self.emit("stage_started")
        action()
        self.emit("stage_completed")
        self.completed.append(stage)
        shared.atomic_json(CACHE / f"interim_{stage}.json", self.publish("running"))
        self.stage_clock = None

    def run(self) -> int:
        try:
            def first() -> None:
                self.blocks("original", 200)
                self.verify_original(200)
            def original() -> None:
                self.blocks("original")
                self.verify_original()
            self.phase("i", first)
            self.phase("ii", original)
            self.phase("iii", lambda: self.blocks("roundtrip_40k"))
            self.phase("iv", lambda: self.blocks("lowpass_7k"))
            self.phase("v", lambda: self.blocks("lowpass_4k"))
            self.invocation.update(status="complete", ended_utc=shared.utc())
            self.emit("job_completed")
            self.publish("complete")
            return 0
        except Exception as exc:
            status = ("paused_deadline" if isinstance(exc, shared.DeadlineReached) else
                      "blocked_reproduction" if isinstance(exc, shared.ReproductionFailure) else "blocked_error")
            self.invocation.update(status=status, ended_utc=shared.utc())
            self.emit("job_stopped", error=str(exc))
            self.publish(status, str(exc))
            return 2 if status == "paused_deadline" else 3 if status == "blocked_reproduction" else 1
        finally:
            self.dataset.close()


def write_markdown(report: dict[str, Any]) -> None:
    lines = ["# T22 — Phép thử shortcut cho X1 cuối", "", "Trạng thái: " + report["status"],
        "CPU/dev-full, không train/test/Kaggle. Tập con 3.120 mẫu/169 shard; artifact 38.777 mẫu/186 shard.",
        "Threshold X1 cuối cố định: " + str(THRESHOLD), "Tập con SHA-256: " + SUBSET_SHA, "",
        "## Tái tạo logit", "", shared.canonical(report["reproduction"]).decode(), "",
        "## Kết quả đặt cạnh LR1e-5 (mỗi checkpoint có threshold riêng đã khóa)", "",
        "| Mô hình | Điều kiện | EER tổng % | VC % | AP % | Replay % | FPR % | Bỏ lọt VC % | AP % | Replay % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in (report["comparison_data_only"] | report["results"]).items():
        for condition, result in value["conditions"].items():
            e, g = result["per_attack_eer"], result["groups"]
            numbers = [result["overall_eer"], *[e[a]["eer"] for a in shared.GROUPS[1:]],
                       g["bonafide"]["fpr"], *[g[a]["miss_rate"] for a in shared.GROUPS[1:]]]
            lines.append("| " + name + " | " + condition + " | " + " | ".join(f"{100*n:.6f}" for n in numbers) + " |")
        lines += ["", name + ": " + shared.canonical(value["indicators"]).decode(), ""]
    lines += ["## Bộ lọc, checksum và thời gian", "", shared.canonical(report["filters"]).decode(), "",
              shared.canonical(report["timing"]).decode(), "",
              f"Ngắt: {report['interruption_count']}; resume: {report['resume_count']}.",
              "JSON chứa P5/P50/P95 xác suất và logit theo nhóm, SHA đầu vào/checkpoint/pretrained và danh sách file.",
              report["interpretation_limit"]]
    if report.get("error"):
        lines += ["", "Điểm dừng: " + report["error"]]
    temporary = CACHE / "report.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, REPORT_MD)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-reason", default="Tiếp tục sau gián đoạn cục bộ; giữ nguyên identity.")
    parser.add_argument("--deadline-hours", type=float, default=8)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        with shared.job_lock(CACHE / "job.lock"):
            return Probe(args).run()
    except Exception as exc:
        # Setup/resume bị từ chối: giữ nguyên mọi khối và report đã công bố.
        shared.atomic_json(CACHE / "setup_error.json", {"utc": shared.utc(), "error": str(exc)})
        print("LỖI T22: " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
