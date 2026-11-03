#!/usr/bin/env python3
"""Runner B0 theo D023 (Parquet, dev cũ) và B0 v2 theo D027 (packed, dev-full).

Technical smoke tách khỏi lượt khoa học. Với D027, seed huấn luyện (khởi tạo mô hình,
sampler, random crop) được truyền bằng `--training-seed` và tách khỏi seed dữ liệu 2026
của split/manifest; thiết bị chạy được ghi vào identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import torch
from torch.utils.data import Dataset, Subset, default_collate

from scripts import train_b0 as common
from scripts.kaggle_workflow import source_files
from src.data import PackedManifestDataset, verify_packed_artifact
from src.training import ResumableBatchSampler
from src.training.full_training import GenerationStore, run_training, sha256_file, write_json


DATA_SEED = 2026
B0_V2_TRAINING_SEEDS = (2026, 2027, 2028)
_TRAINING_EXPECTED = {
    "batch_size": 8, "num_workers": 0, "optimizer": "Adam",
    "learning_rate": 0.001, "max_epochs": 3, "steps_per_epoch": 1986,
    "max_optimizer_steps": 5958, "max_train_sample_exposures": 47655,
    "drop_last": False, "shuffle": True, "scheduler": None, "gradient_clipping": None,
    "loss": "BCEWithLogitsLoss",
}
_EVALUATION_EXPECTED = {
    "schedule": "after_each_completed_epoch_before_next_epoch", "full_development": True,
    "partial_epoch_evaluation": False, "threshold_source": "development_eer_of_selected_checkpoint",
    "selection_metric": "eer", "mode": "min", "strict_improvement": True,
    "equal_eer_action": "keep_earliest_checkpoint", "patience": 1,
    "minimum_completed_epochs": 1, "min_delta": 0.0,
    "allow_loss_tie_break": False, "nonfinite_action": "fail_without_committing_evaluation",
}


def semantic_plan(plan: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(plan)
    for key in ("implementation", "status", "outputs", "budget"):
        result.pop(key, None)
    return result


def _validate_shared(plan: dict[str, Any]) -> None:
    t, e, m = (plan[k] for k in ("training", "evaluation", "model"))
    if any(t.get(k) != v for k, v in _TRAINING_EXPECTED.items()):
        raise ValueError("Cấu hình training không khớp D023.")
    if (
        m["amplitude_policy"] != "peak"
        or m["initialization"] != "fresh_seeded_no_pilot_weights"
        or m["precision"] != "float32" or m["amp"]
        or e != _EVALUATION_EXPECTED
        or plan["sessions"]["max_optimizer_steps_per_invocation"] != 512
        or plan["sessions"]["checkpoint_every_completed_steps"] != 64
    ):
        raise ValueError("Đặc tả model/evaluation/phiên không khớp D023.")


def validate_plan(plan: dict[str, Any]) -> None:
    """Đặc tả D023: seed 2026, Parquet, dev cũ 4.115 mẫu."""

    d = plan["data"]
    if plan["training"].get("seed") != 2026:
        raise ValueError("Cấu hình training không khớp D023.")
    _validate_shared(plan)
    if (
        plan["decision"] != "D023"
        or d["use_test"] or d["backend"] != "parquet"
        or d["train_split"] != "closed_train" or d["development_split"] != "closed_dev"
        or (d["train_samples"], d["development_samples"], d["manifest_samples"], d["manifest_shards"]) != (15885, 4115, 20000, 66)
    ):
        raise ValueError("Đặc tả dữ liệu/model/evaluation/phiên không khớp D023.")


def validate_plan_v2(plan: dict[str, Any]) -> None:
    """Đặc tả D027: cấu hình D023, nhưng packed, dev-full và ba seed huấn luyện."""

    d, t = plan["data"], plan["training"]
    if "seed" in t:
        raise ValueError("B0 v2 không dùng trường seed chung; cần training_seeds tách khỏi data_seed.")
    if tuple(t.get("training_seeds", ())) != B0_V2_TRAINING_SEEDS:
        raise ValueError("Seed huấn luyện B0 v2 phải đúng 2026, 2027, 2028 (D026/D027).")
    if t.get("training_seed_scope") != ["model_initialization", "sampler", "random_crop"]:
        raise ValueError("Phạm vi seed huấn luyện không khớp D026.")
    _validate_shared(plan)
    if (
        plan["decision"] != "D027"
        or plan["model"].get("lfcc_features") != "per_sample"
        or d["use_test"] or d["backend"] != "packed" or d["data_seed"] != DATA_SEED
        or d["train_split"] != "closed_train" or d["development_split"] != "closed_dev"
        or (d["train_samples"], d["development_samples"]) != (15885, 38777)
    ):
        raise ValueError("Đặc tả dữ liệu/model của B0 v2 không khớp D027.")
    for key in ("train_artifact_waveforms_sha256", "development_artifact_waveforms_sha256"):
        value = d.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Checksum artifact {key} chưa được khóa trong đặc tả B0 v2.")


@dataclass
class RunData:
    """Dữ liệu và phần identity riêng của từng đặc tả."""

    train: Any
    dev: Any
    train_indices: list[int]
    dev_indices: list[int]
    training_seed: int
    identity_extra: dict[str, Any] = field(default_factory=dict)
    report_extra: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        self.train.close()
        self.dev.close()


def _open_d023(args: argparse.Namespace, plan: dict[str, Any], audio: dict[str, Any]) -> RunData:
    if args.manifest is None or args.parquet_dir is None:
        raise ValueError("D023 cần --manifest và --parquet-dir.")
    if args.training_seed not in (None, 2026):
        raise ValueError("D023 chỉ có seed 2026; seed huấn luyện khác thuộc B0 v2 (D027).")
    if sha256_file(args.manifest) != plan["data"]["manifest_sha256"]:
        raise ValueError("Checksum manifest/model/audio không khớp D023.")
    train = common._build_dataset(manifest=args.manifest, parquet_dir=args.parquet_dir, split="closed_train", training=True, audio_config=audio, seed=2026, amplitude_policy="peak")
    dev = common._build_dataset(manifest=args.manifest, parquet_dir=args.parquet_dir, split="closed_dev", training=False, audio_config=audio, seed=2026, amplitude_policy="peak")
    try:
        if len(train) != 15885 or len(dev) != 4115 or len({r.shard for r in train.records + dev.records}) != 66:
            raise ValueError("Số mẫu/shard không khớp manifest D023.")
        for record in train.records + dev.records:
            if not (args.parquet_dir / record.shard).is_file():
                raise ValueError("Không đủ coverage shard để chạy D023.")
        train_indices, dev_indices = _select(args, plan, train.records, dev.records)
    except Exception:
        train.close()
        dev.close()
        raise
    used = len({train.records[i].shard for i in train_indices} | {dev.records[i].shard for i in dev_indices})
    return RunData(
        train=train, dev=dev, train_indices=train_indices, dev_indices=dev_indices,
        training_seed=2026,
        report_extra={"manifest_sha256": sha256_file(args.manifest), "source_manifest_samples": 20000, "source_manifest_shards": 66, "used_shards": used},
    )


def _packed(path: Path, *, training: bool, seed: int, audio: dict[str, Any]) -> PackedManifestDataset:
    amplitude = audio["waveform"]["amplitude_policy"]
    return PackedManifestDataset(
        path, training=training, target_samples=int(audio["segmentation"]["samples"]),
        seed=seed, amplitude_policy="peak", peak_target=float(amplitude["peak_target"]),
        rms_target_dbfs=float(amplitude["rms_target_dbfs"]),
        minimum_input_rms_dbfs=float(amplitude["minimum_input_rms_dbfs"]),
    )


def _check_artifact(dataset: PackedManifestDataset, *, split: str, samples: int, manifest_sha256: str, waveforms_sha256: str, name: str) -> None:
    if (
        dataset.split != split or len(dataset) != samples
        or dataset.source_manifest_sha256 != manifest_sha256
        or dataset.waveforms_sha256 != waveforms_sha256
        or dataset.target_sample_rate != 16_000
    ):
        raise ValueError(f"Artifact {name} không khớp split/số mẫu/manifest/checksum của D027.")
    if [r.source_index for r in dataset.records] != list(range(len(dataset))):
        raise ValueError(f"Artifact {name} phải giữ đúng thứ tự manifest.")


def _open_d027(args: argparse.Namespace, plan: dict[str, Any], audio: dict[str, Any], device: torch.device) -> RunData:
    d = plan["data"]
    if args.training_seed not in B0_V2_TRAINING_SEEDS:
        raise ValueError("B0 v2 cần --training-seed thuộc 2026, 2027, 2028.")
    if args.manifest is not None or args.parquet_dir is not None:
        raise ValueError("B0 v2 đọc packed artifact; không nhận --manifest/--parquet-dir.")
    train_dir = args.train_artifact or ROOT / d["train_artifact"]
    dev_dir = args.dev_artifact or ROOT / d["development_artifact"]
    if not args.resume:
        # Băm lại toàn bộ waveform một lần khi bắt đầu run; resume chỉ kiểm metadata/kích thước.
        for path in (train_dir, dev_dir):
            verify_packed_artifact(path)
    train = _packed(train_dir, training=True, seed=args.training_seed, audio=audio)
    dev = _packed(dev_dir, training=False, seed=args.training_seed, audio=audio)
    try:
        _check_artifact(train, split="closed_train", samples=15885, manifest_sha256=d["train_manifest_sha256"], waveforms_sha256=d["train_artifact_waveforms_sha256"], name="train")
        _check_artifact(dev, split="closed_dev", samples=38777, manifest_sha256=d["development_manifest_sha256"], waveforms_sha256=d["development_artifact_waveforms_sha256"], name="dev-full")
        train_indices, dev_indices = _select(args, plan, train.records, dev.records)
    except Exception:
        train.close()
        dev.close()
        raise
    artifacts = {
        name: {"metadata_sha256": sha256_file(path / "metadata.json"), "waveforms_sha256": dataset.waveforms_sha256, "source_manifest_sha256": dataset.source_manifest_sha256}
        for name, path, dataset in (("train", train_dir, train), ("development", dev_dir, dev))
    }
    return RunData(
        train=train, dev=dev, train_indices=train_indices, dev_indices=dev_indices,
        training_seed=args.training_seed,
        identity_extra={"training_seed": args.training_seed, "data_seed": DATA_SEED, "backend": "packed", "device": device.type, "artifacts": artifacts},
        report_extra={"training_seed": args.training_seed, "data_seed": DATA_SEED, "backend": "packed", "artifacts": artifacts, "train_manifest_sha256": d["train_manifest_sha256"], "development_manifest_sha256": d["development_manifest_sha256"]},
    )


def _select(args: argparse.Namespace, plan: dict[str, Any], train_records: Sequence[Any], dev_records: Sequence[Any]) -> tuple[list[int], list[int]]:
    train_indices = common._limited_indices(train_records, 4 if args.technical_smoke else None, require_both_labels=True, selection="head", seed=DATA_SEED, split_name="closed_train")
    dev_indices = common._limited_indices(dev_records, 2 if args.technical_smoke else None, require_both_labels=True, selection="head", seed=DATA_SEED, split_name="closed_dev")
    selection = common._sample_selection_sha256(selection="head", train_records=train_records, train_indices=list(range(len(train_records))), development_records=dev_records, development_indices=list(range(len(dev_records))))
    if selection != plan["data"]["sample_selection_sha256"]:
        raise ValueError(f"Exact selection của manifest không khớp {plan['decision']}.")
    return train_indices, dev_indices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/b0_full_training.json")
    parser.add_argument("--manifest", type=Path, help="D023: manifest Parquet.")
    parser.add_argument("--parquet-dir", type=Path, help="D023: thư mục shard Parquet.")
    parser.add_argument("--train-artifact", type=Path, help="D027: ghi đè đường dẫn artifact train.")
    parser.add_argument("--dev-artifact", type=Path, help="D027: ghi đè đường dẫn artifact dev-full.")
    parser.add_argument("--training-seed", type=int, help="D027: seed huấn luyện 2026/2027/2028.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--smoke-session-steps", type=int, default=128)
    args = parser.parse_args(argv)
    plan = common._load_json(args.plan, "đặc tả full training")
    is_v2 = plan.get("decision") == "D027"
    (validate_plan_v2 if is_v2 else validate_plan)(plan)
    label = "B0 v2 D027" if is_v2 else "B0 D023"
    if not args.technical_smoke and not plan["implementation"]["ready_for_training"]:
        raise ValueError(f"Chưa mở cổng CPU/CUDA cho full training {plan['decision']}.")
    if args.smoke_session_steps <= 0 or (not args.technical_smoke and args.smoke_session_steps != 128):
        raise ValueError("Giới hạn phiên smoke không hợp lệ cho lần chạy này.")
    for path, expected in (
        (ROOT / plan["model"]["config"], plan["model"]["config_sha256"]),
        (ROOT / plan["model"]["audio_config"], plan["model"]["audio_config_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Checksum manifest/model/audio không khớp {plan['decision']}.")
    device = common._select_device(args.device)
    torch.set_num_threads(1)
    config = common._load_json(ROOT / plan["model"]["config"], "mô hình")
    audio = common._load_json(ROOT / plan["model"]["audio_config"], "audio")
    data = _open_d027(args, plan, audio, device) if is_v2 else _open_d023(args, plan, audio)
    try:
        seed = data.training_seed
        train, dev, train_indices, dev_indices = data.train, data.dev, data.train_indices, data.dev_indices
        # Seed huấn luyện quyết định khởi tạo mô hình; đặt ngay trước khi dựng mô hình.
        common._set_seed(seed)
        batch_size = 2 if args.technical_smoke else 8
        model = common.build_lcnn_from_config(config).to(device)
        lfcc = common.build_lfcc_transform(config["lfcc"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sampler = ResumableBatchSampler(dataset_size=len(train_indices), batch_size=batch_size, seed=seed)
        files = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p)} for p in source_files()]
        source_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        identity = {"plan": semantic_plan(plan), "source_tree_sha256": source_sha, "mode": "technical_smoke" if args.technical_smoke else "full_training", "train_indices": train_indices, "development_indices": dev_indices, "batch_size": batch_size, **data.identity_extra}
        process = psutil.Process()
        started = time.perf_counter()
        timing = {"training_loader_seconds": 0.0, "training_model_seconds": 0.0}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def step(indices: list[int], epoch: int) -> float:
            train.set_epoch(epoch)
            stamp = time.perf_counter()
            batch = default_collate([train[train_indices[i]] for i in indices])
            timing["training_loader_seconds"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(common._model_input(lfcc, batch["waveform"].to(device)))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["label"].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Training loss không hữu hạn.")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Gradient không hữu hạn.")
            optimizer.step()
            timing["training_model_seconds"] += time.perf_counter() - stamp
            return float(loss.detach().cpu())

        def evaluate() -> dict[str, Any]:
            dataset: Dataset[dict[str, Any]] = Subset(dev, dev_indices)
            result, records, resources = common._evaluate(model=model, lfcc=lfcc, dataset=dataset, batch_size=batch_size, device=device, process=process, seed=seed)
            result["records"] = records
            result["timing"] = resources
            return result

        result = run_training(model=model, optimizer=optimizer, sampler=sampler, identity=identity, store=GenerationStore(args.run_root), train_step=step, evaluate=evaluate, max_epochs=3, patience=1, development_samples=len(dev_indices), session_steps=args.smoke_session_steps if args.technical_smoke else 512, checkpoint_interval=1 if args.technical_smoke else 64, resume=args.resume)
        pointer = json.loads((args.run_root / "LATEST.json").read_text(encoding="utf-8"))
        report = {"report_version": 1, "run_mode": identity["mode"], "experiment_id": plan["experiment_id"], "decision": plan["decision"], "status": result["status"], "plan_sha256": sha256_file(args.plan), "source_tree_sha256": source_sha, **data.report_extra, "train_samples": len(train_indices), "development_samples": len(dev_indices), "steps_this_invocation": result["steps_this_invocation"], "controller": result["controller"], "checkpoints": pointer, "environment": common._device_details(device), "resources": {**timing, "elapsed_seconds": time.perf_counter()-started, "rss_mib": process.memory_info().rss/1024**2, "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else 0.0}}
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report_json, report)
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        if is_v2:
            scope = f"- Seed huấn luyện/dữ liệu: {seed}/{DATA_SEED}; backend packed; thiết bị {device.type}\n"
        else:
            scope = f"- Used shards: {report['used_shards']} (source manifest 20.000/66 shard)\n"
        args.report_md.write_text(f"# {label}\n\n- Mode: {identity['mode']}\n- Status: {result['status']}\n- Train/development: {len(train_indices)}/{len(dev_indices)}\n{scope}- Global step: {result['controller']['global_step']}\n- Chi tiết best/latest/history và checksum: report JSON đi kèm.\n", encoding="utf-8")
        print(f"ĐẠT: {label} {result['status']}, step={result['controller']['global_step']}.")
        return 0
    finally:
        data.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
