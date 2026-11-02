#!/usr/bin/env python3
"""Runner B0 theo D023; technical smoke tách khỏi lượt khoa học."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import torch
from torch.utils.data import Subset, default_collate

from scripts import train_b0 as common
from scripts.kaggle_workflow import source_files
from src.training import ResumableBatchSampler
from src.training.full_training import GenerationStore, run_training, sha256_file, write_json


def semantic_plan(plan: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(plan)
    for key in ("implementation", "status", "outputs", "budget"):
        result.pop(key, None)
    return result


def validate_plan(plan: dict[str, Any]) -> None:
    t, e, d, m = (plan[k] for k in ("training", "evaluation", "data", "model"))
    expected = {
        "seed": 2026, "batch_size": 8, "num_workers": 0, "optimizer": "Adam",
        "learning_rate": 0.001, "max_epochs": 3, "steps_per_epoch": 1986,
        "max_optimizer_steps": 5958, "max_train_sample_exposures": 47655,
        "drop_last": False, "shuffle": True, "scheduler": None, "gradient_clipping": None,
        "loss": "BCEWithLogitsLoss",
    }
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError("Cấu hình training không khớp D023.")
    if (
        plan["decision"] != "D023" or m["amplitude_policy"] != "peak"
        or m["initialization"] != "fresh_seeded_no_pilot_weights"
        or m["precision"] != "float32" or m["amp"]
        or d["use_test"] or d["backend"] != "parquet"
        or d["train_split"] != "closed_train" or d["development_split"] != "closed_dev"
        or (d["train_samples"], d["development_samples"], d["manifest_samples"], d["manifest_shards"]) != (15885, 4115, 20000, 66)
        or e != {
            "schedule": "after_each_completed_epoch_before_next_epoch", "full_development": True,
            "partial_epoch_evaluation": False, "threshold_source": "development_eer_of_selected_checkpoint",
            "selection_metric": "eer", "mode": "min", "strict_improvement": True,
            "equal_eer_action": "keep_earliest_checkpoint", "patience": 1,
            "minimum_completed_epochs": 1, "min_delta": 0.0,
            "allow_loss_tie_break": False, "nonfinite_action": "fail_without_committing_evaluation",
        }
        or plan["sessions"]["max_optimizer_steps_per_invocation"] != 512
        or plan["sessions"]["checkpoint_every_completed_steps"] != 64
    ):
        raise ValueError("Đặc tả dữ liệu/model/evaluation/phiên không khớp D023.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/b0_full_training.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--smoke-session-steps", type=int, default=128)
    args = parser.parse_args()
    plan = common._load_json(args.plan, "đặc tả full training")
    validate_plan(plan)
    if not args.technical_smoke and not plan["implementation"]["ready_for_training"]:
        raise ValueError("Chưa mở cổng CPU/CUDA cho full training D023.")
    if args.smoke_session_steps <= 0 or (not args.technical_smoke and args.smoke_session_steps != 128):
        raise ValueError("Giới hạn phiên smoke không hợp lệ cho lần chạy này.")
    for path, expected in (
        (args.manifest, plan["data"]["manifest_sha256"]),
        (ROOT / plan["model"]["config"], plan["model"]["config_sha256"]),
        (ROOT / plan["model"]["audio_config"], plan["model"]["audio_config_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError("Checksum manifest/model/audio không khớp D023.")
    device = common._select_device(args.device)
    common._set_seed(2026)
    torch.set_num_threads(1)
    config = common._load_json(ROOT / plan["model"]["config"], "mô hình")
    audio = common._load_json(ROOT / plan["model"]["audio_config"], "audio")
    train = common._build_dataset(manifest=args.manifest, parquet_dir=args.parquet_dir, split="closed_train", training=True, audio_config=audio, seed=2026, amplitude_policy="peak")
    dev = common._build_dataset(manifest=args.manifest, parquet_dir=args.parquet_dir, split="closed_dev", training=False, audio_config=audio, seed=2026, amplitude_policy="peak")
    try:
        if len(train) != 15885 or len(dev) != 4115 or len({r.shard for r in train.records + dev.records}) != 66:
            raise ValueError("Số mẫu/shard không khớp manifest D023.")
        for record in train.records + dev.records:
            if not (args.parquet_dir / record.shard).is_file():
                raise ValueError("Không đủ coverage shard để chạy D023.")
        train_indices = common._limited_indices(train.records, 4 if args.technical_smoke else None, require_both_labels=True, selection="head", seed=2026, split_name="closed_train")
        dev_indices = common._limited_indices(dev.records, 2 if args.technical_smoke else None, require_both_labels=True, selection="head", seed=2026, split_name="closed_dev")
        selection = common._sample_selection_sha256(selection="head", train_records=train.records, train_indices=list(range(len(train))), development_records=dev.records, development_indices=list(range(len(dev))))
        if selection != plan["data"]["sample_selection_sha256"]:
            raise ValueError("Exact selection của manifest không khớp D023.")
        batch_size = 2 if args.technical_smoke else 8
        model = common.build_lcnn_from_config(config).to(device)
        lfcc = common.build_lfcc_transform(config["lfcc"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sampler = ResumableBatchSampler(dataset_size=len(train_indices), batch_size=batch_size, seed=2026)
        files = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p)} for p in source_files()]
        source_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        identity = {"plan": semantic_plan(plan), "source_tree_sha256": source_sha, "mode": "technical_smoke" if args.technical_smoke else "full_training", "train_indices": train_indices, "development_indices": dev_indices, "batch_size": batch_size}
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
            result, records, resources = common._evaluate(model=model, lfcc=lfcc, dataset=Subset(dev, dev_indices), batch_size=batch_size, device=device, process=process, seed=2026)
            result["records"] = records
            result["timing"] = resources
            return result

        result = run_training(model=model, optimizer=optimizer, sampler=sampler, identity=identity, store=GenerationStore(args.run_root), train_step=step, evaluate=evaluate, max_epochs=3, patience=1, development_samples=len(dev_indices), session_steps=args.smoke_session_steps if args.technical_smoke else 512, checkpoint_interval=1 if args.technical_smoke else 64, resume=args.resume)
        pointer = json.loads((args.run_root / "LATEST.json").read_text(encoding="utf-8"))
        report = {"report_version": 1, "run_mode": identity["mode"], "experiment_id": plan["experiment_id"], "status": result["status"], "plan_sha256": sha256_file(args.plan), "source_tree_sha256": source_sha, "manifest_sha256": sha256_file(args.manifest), "source_manifest_samples": 20000, "source_manifest_shards": 66, "train_samples": len(train_indices), "development_samples": len(dev_indices), "used_shards": len({train.records[i].shard for i in train_indices} | {dev.records[i].shard for i in dev_indices}), "steps_this_invocation": result["steps_this_invocation"], "controller": result["controller"], "checkpoints": pointer, "environment": common._device_details(device), "resources": {**timing, "elapsed_seconds": time.perf_counter()-started, "rss_mib": process.memory_info().rss/1024**2, "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else 0.0}}
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report_json, report)
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(f"# B0 D023\n\n- Mode: {identity['mode']}\n- Status: {result['status']}\n- Train/development: {len(train_indices)}/{len(dev_indices)}\n- Used shards: {report['used_shards']} (source manifest 20.000/66 shard)\n- Global step: {result['controller']['global_step']}\n- Chi tiết best/latest/history và checksum: report JSON đi kèm.\n", encoding="utf-8")
        print(f"ĐẠT: B0 D023 {result['status']}, step={result['controller']['global_step']}.")
        return 0
    finally:
        train.close()
        dev.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
