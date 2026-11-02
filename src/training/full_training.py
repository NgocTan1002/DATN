"""Điều phối epoch và checkpoint generation nguyên tử cho B0 đầy đủ."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Callable

import torch

from .checkpoint import _capture_rng_state, _restore_rng_state
from .sampler import ResumableBatchSampler


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class GenerationStore:
    """Chỉ LATEST.json đã commit có quyền quyết định trạng thái khôi phục."""

    def __init__(self, root: Path, fault: Callable[[str], None] | None = None) -> None:
        self.root = root.resolve()
        self.fault = fault or (lambda _stage: None)

    def _file(self, generation: str, filename: str) -> Path:
        if Path(generation).name != generation or not generation.startswith("g-"):
            raise ValueError("Tên generation checkpoint không hợp lệ.")
        return self.root / generation / filename

    def commit(self, payload: dict[str, Any], *, new_best: bool = False) -> dict[str, Any]:
        generation = "g-" + uuid.uuid4().hex
        folder = self.root / generation
        folder.mkdir(parents=True)
        payload = copy.deepcopy(payload)
        # Dataclass metric dùng tuple; JSON dùng list. Lưu cùng một biểu diễn.
        payload["controller"] = json.loads(json.dumps(payload["controller"], allow_nan=False))
        if new_best:
            payload["controller"]["best"]["generation"] = generation
        checkpoint = folder / "checkpoint.pt"
        with checkpoint.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        write_json(folder / "state.json", payload["controller"])
        current = {
            "generation": generation,
            "checkpoint_sha256": sha256_file(checkpoint),
            "state_sha256": sha256_file(folder / "state.json"),
        }
        best = payload["controller"]["best"]
        best_ref = None
        if best is not None:
            best_folder = self._file(best["generation"], "checkpoint.pt").parent
            best_ref = {
                "generation": best["generation"],
                "checkpoint_sha256": sha256_file(best_folder / "checkpoint.pt"),
                "state_sha256": sha256_file(best_folder / "state.json"),
            }
        pointer = {"checkpoint_version": 4, "latest": current, "best": best_ref}
        staging = self.root / ("LATEST-" + uuid.uuid4().hex + ".tmp")
        write_json(staging, pointer)
        self.fault("before_pointer")
        os.replace(staging, self.root / "LATEST.json")
        self.fault("after_pointer")
        return payload

    def load(self) -> dict[str, Any]:
        pointer = json.loads((self.root / "LATEST.json").read_text(encoding="utf-8"))
        if pointer.get("checkpoint_version") != 4:
            raise ValueError("Checkpoint full training phải có phiên bản 4.")
        for ref in (pointer["latest"], pointer["best"]):
            if ref is None:
                continue
            for filename, field in (("checkpoint.pt", "checkpoint_sha256"), ("state.json", "state_sha256")):
                path = self._file(ref["generation"], filename)
                if not path.is_file() or sha256_file(path) != ref[field]:
                    raise ValueError("Checkpoint/report generation thiếu hoặc sai checksum.")
        latest = pointer["latest"]["generation"]
        payload = torch.load(self._file(latest, "checkpoint.pt"), map_location="cpu", weights_only=True)
        if payload.get("checkpoint_version") != 4:
            raise ValueError("Payload checkpoint full training sai phiên bản.")
        state = json.loads(self._file(latest, "state.json").read_text(encoding="utf-8"))
        if payload["controller"] != state:
            raise ValueError("Controller checkpoint không khớp report generation.")
        best = state["best"]
        if (best is None) != (pointer["best"] is None) or (
            best is not None and best["generation"] != pointer["best"]["generation"]
        ):
            raise ValueError("Tham chiếu best không nhất quán với controller.")
        return payload


def run_training(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: ResumableBatchSampler,
    identity: dict[str, Any],
    store: GenerationStore,
    train_step: Callable[[list[int], int], float],
    evaluate: Callable[[], dict[str, Any]],
    max_epochs: int,
    patience: int,
    development_samples: int,
    session_steps: int,
    checkpoint_interval: int,
    resume: bool,
    fault: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Huấn luyện batch đã commit; evaluation pending có ưu tiên trước bước mới."""
    fault = fault or (lambda _stage: None)
    if min(max_epochs, patience, development_samples, session_steps, checkpoint_interval) <= 0:
        raise ValueError("Giới hạn epoch/phiên/checkpoint/patience phải lớn hơn 0.")
    steps_per_epoch = math.ceil(sampler.dataset_size / sampler.batch_size)
    state: dict[str, Any] = {
        "global_step": 0, "completed_epochs": 0, "last_evaluated_step": 0,
        "needs_evaluation": False, "best": None, "bad_evaluations": 0,
        "stop_reason": None, "history": [], "losses": [], "batch_trace": [],
    }
    if resume:
        payload = store.load()
        if payload["identity"] != identity:
            raise ValueError("Identity checkpoint full training không khớp.")
        sampler.load_state_dict(payload["sampler"])
        state = payload["controller"]
        if (
            state["global_step"] != sampler.epoch * steps_per_epoch + sampler.batch_cursor
            or len(state["history"]) != state["completed_epochs"]
            or state["completed_epochs"] != sampler.epoch + int(len(sampler) == 0 and not state["needs_evaluation"])
            or state["needs_evaluation"] != (len(sampler) == 0 and state["last_evaluated_step"] < state["global_step"])
            or state["global_step"] > max_epochs * steps_per_epoch
        ):
            raise ValueError("Controller/progress/sampler full training không nhất quán.")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng_state(payload["rng"])
    elif (store.root / "LATEST.json").exists():
        raise ValueError("Thư mục run đã có checkpoint; cần yêu cầu resume rõ ràng.")

    def commit(new_best: bool = False) -> None:
        nonlocal state
        payload = {
            "checkpoint_version": 4, "identity": identity,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(), "rng": _capture_rng_state(),
            "controller": state,
        }
        state = store.commit(payload, new_best=new_best)["controller"]

    if not resume:
        commit()
    performed = 0
    iterator = iter(sampler)
    while True:
        if state["needs_evaluation"]:
            fault("before_evaluation")
            development = evaluate()
            for key in ("eer", "loss", "eer_threshold", "roc_auc", "macro_f1", "decision_threshold"):
                if not math.isfinite(float(development[key])):
                    raise ValueError("Development metric chứa NaN hoặc Inf; chưa commit evaluation.")
            if development["sample_count"] != development_samples:
                raise ValueError("Evaluation không đủ số mẫu development.")
            epoch = sampler.epoch + 1
            improved = state["best"] is None or development["eer"] < state["best"]["development"]["eer"]
            state["history"].append({"epoch": epoch, "global_step": state["global_step"], "eer": development["eer"], "loss": development["loss"]})
            state["completed_epochs"] = epoch
            state["last_evaluated_step"] = state["global_step"]
            state["needs_evaluation"] = False
            if improved:
                state["best"] = {"generation": None, "epoch": epoch, "global_step": state["global_step"], "development": development}
                state["bad_evaluations"] = 0
            else:
                state["bad_evaluations"] += 1
            if state["bad_evaluations"] >= patience:
                state["stop_reason"] = "early_stopped"
            elif epoch >= max_epochs:
                state["stop_reason"] = "max_epochs"
            commit(new_best=improved)
            continue
        if state["stop_reason"] is not None:
            break
        if performed >= session_steps:
            commit()
            break
        if len(sampler) == 0:
            sampler.set_epoch(sampler.epoch + 1)
            iterator = iter(sampler)
        batch = next(iterator)
        loss = train_step(batch, sampler.epoch)
        if not math.isfinite(loss):
            raise ValueError("Training loss chứa NaN hoặc Inf.")
        performed += 1
        state["global_step"] += 1
        state["losses"].append(loss)
        state["batch_trace"].append(batch)
        state["needs_evaluation"] = len(sampler) == 0
        fault("after_step")
        if state["needs_evaluation"] or state["global_step"] % checkpoint_interval == 0:
            commit()
    return {"status": state["stop_reason"] or "paused", "steps_this_invocation": performed, "controller": state}
