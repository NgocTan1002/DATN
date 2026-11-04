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


def extended_identity(
    previous: dict[str, Any], current: dict[str, Any], *, max_epochs: int,
    patience: int, source_transition: dict[str, Any] | None,
) -> dict[str, Any]:
    """Chỉ nới trần/ngưỡng patience đã duyệt; mọi identity khác phải giữ nguyên."""
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    history = old.pop("extension_history", [])
    if "extension_history" in new:
        raise ValueError("Identity yêu cầu nối dài không được tự điền lịch sử chuyển đổi.")
    old_training, new_training = old["plan"]["training"], new["plan"]["training"]
    old_cap = old_training["max_epochs"]
    old_patience = old["plan"]["evaluation"]["patience"]
    if max_epochs <= old_cap or new_training["max_epochs"] != max_epochs:
        raise ValueError("Trần epoch nối dài phải lớn hơn trần cũ và khớp identity.")
    if patience < old_patience or new["plan"]["evaluation"]["patience"] != patience:
        raise ValueError("Patience nối dài không được giảm và phải khớp identity.")
    for identity in (old, new):
        training = identity["plan"]["training"]
        samples = identity["plan"]["data"]["train_samples"]
        if (
            training["steps_per_epoch"] != math.ceil(samples / training["batch_size"])
            or training["max_optimizer_steps"] != training["max_epochs"] * training["steps_per_epoch"]
            or training["max_train_sample_exposures"] != training["max_epochs"] * samples
        ):
            raise ValueError("Ngân sách bước/lượt mẫu của identity nối dài không hợp lệ.")
    # Chỉ bỏ bốn trường thay đổi có chủ đích sau khi kiểm tra giá trị dẫn xuất.
    for identity in (old, new):
        for key in ("max_epochs", "max_optimizer_steps", "max_train_sample_exposures"):
            identity["plan"]["training"].pop(key)
        identity["plan"]["evaluation"].pop("patience")
    if old.get("source_tree_sha256") != new.get("source_tree_sha256"):
        if (
            source_transition is None
            or source_transition.get("old_source_tree_sha256") != old.get("source_tree_sha256")
            or source_transition.get("new_source_tree_sha256") != new.get("source_tree_sha256")
        ):
            raise ValueError("Identity source khác nhưng chưa có chuyển nguồn được kiểm chứng.")
        old_files = source_transition.get("old_source_files", [])
        new_files = source_transition.get("new_source_files", [])
        old_inventory = {item["path"]: item["sha256"] for item in old_files}
        new_inventory = {item["path"]: item["sha256"] for item in new_files}
        changed = sorted(path for path in old_inventory.keys() | new_inventory.keys()
                         if old_inventory.get(path) != new_inventory.get(path))
        allowed = {"scripts/train_b0_full.py", "src/training/full_training.py",
                   "configs/b0_v2_training.json", "configs/xlsr_pretrained_candidate.json",
                   "scripts/benchmark_xlsr_cuda.py"}
        if (
            hashlib.sha256(json.dumps(old_files, sort_keys=True).encode()).hexdigest() != old["source_tree_sha256"]
            or hashlib.sha256(json.dumps(new_files, sort_keys=True).encode()).hexdigest() != new["source_tree_sha256"]
            or current.get("source_files") != new_files
            or changed != source_transition.get("changed_files")
            or not set(changed) <= allowed or not old_inventory.keys() <= new_inventory.keys()
        ):
            raise ValueError("Bằng chứng chuyển source không khớp inventory/hash/danh sách thay đổi.")
        for identity in (old, new):
            identity.pop("source_tree_sha256", None)
            identity.pop("source_files", None)
    elif old.get("source_files") != new.get("source_files"):
        # Checkpoint cũ chưa có inventory: chỉ được bổ sung inventory đúng cùng hash.
        if "source_files" in old or "source_files" not in new:
            raise ValueError("Identity inventory source không khớp.")
        files = new.pop("source_files")
        if hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() != new["source_tree_sha256"]:
            raise ValueError("Identity inventory source sai checksum.")
    if old != new:
        raise ValueError("Identity ngoài trần epoch/patience/chuyển nguồn không được thay đổi.")
    result = copy.deepcopy(current)
    result["extension_history"] = history + [{
        "old_max_epochs": old_cap, "new_max_epochs": max_epochs,
        "old_patience": old_patience, "new_patience": patience,
        "source_transition": copy.deepcopy(source_transition),
    }]
    return result


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
    extend_max_epochs: bool = False,
    source_transition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Huấn luyện batch đã commit; evaluation pending có ưu tiên trước bước mới."""
    fault = fault or (lambda _stage: None)
    if min(max_epochs, patience, development_samples, session_steps, checkpoint_interval) <= 0:
        raise ValueError("Giới hạn epoch/phiên/checkpoint/patience phải lớn hơn 0.")
    if extend_max_epochs and not resume:
        raise ValueError("Nối dài cần --resume từ checkpoint đã chạm trần.")
    steps_per_epoch = math.ceil(sampler.dataset_size / sampler.batch_size)
    state: dict[str, Any] = {
        "global_step": 0, "completed_epochs": 0, "last_evaluated_step": 0,
        "needs_evaluation": False, "best": None, "bad_evaluations": 0,
        "stop_reason": None, "history": [], "losses": [], "batch_trace": [],
    }
    if resume:
        payload = store.load()
        previous_identity = payload["identity"]
        if extend_max_epochs:
            state = payload["controller"]
            old_cap = previous_identity["plan"]["training"]["max_epochs"]
            if (
                state["stop_reason"] != "max_epochs" or state["bad_evaluations"] != 0
                or state["completed_epochs"] != old_cap
                or state["global_step"] != old_cap * steps_per_epoch
                or state["last_evaluated_step"] != state["global_step"]
                or state["needs_evaluation"]
            ):
                raise ValueError("Chỉ nối dài khi đã chạm trần max_epochs, đánh giá xong và bộ đếm patience bằng 0.")
            identity = extended_identity(previous_identity, identity, max_epochs=max_epochs,
                                         patience=patience, source_transition=source_transition)
        else:
            expected = copy.deepcopy(previous_identity)
            audit = expected.pop("extension_history", None)
            if identity != previous_identity and identity != expected:
                raise ValueError("Identity checkpoint full training không khớp.")
            if audit is not None:
                identity = previous_identity
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
        if extend_max_epochs:
            state["stop_reason"] = None
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

    if not resume or extend_max_epochs:
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
