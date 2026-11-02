"""Chuyển checkpoint giữa các phiên mà không đổi source huấn luyện đã khóa."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.training.full_training import GenerationStore


MANIFEST_NAME = "b0_resume_manifest.json"
PATH_PATTERN = re.compile(r"(?:LATEST\.json|g-[0-9a-f]{32}/(?:checkpoint\.pt|state\.json))\Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_payload(source: Path, destination: Path, dataset_ref: str) -> dict[str, Any]:
    """Đóng gói pointer và toàn bộ generation thành file phẳng cho private dataset."""
    payload = GenerationStore(source).load()
    paths = [source / "LATEST.json"]
    for folder in sorted(source.glob("g-*")):
        paths.extend([folder / "checkpoint.pt", folder / "state.json"])
    if destination.exists():
        raise ValueError("Thư mục gói checkpoint đã tồn tại; từ chối ghi đè.")
    entries: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(source).as_posix()
        if not PATH_PATTERN.fullmatch(relative) or not path.is_file() or path.is_symlink():
            raise ValueError("Generation checkpoint thiếu file hoặc có đường dẫn không hợp lệ.")
        entries.append({"path": relative, "file": relative.replace("/", "__"), "sha256": digest(path), "bytes": path.stat().st_size})
    destination.mkdir(parents=True)
    for path, entry in zip(paths, entries):
        shutil.copyfile(path, destination / entry["file"])
    manifest = {
        "schema_version": 1,
        "global_step": payload["controller"]["global_step"],
        "identity_sha256": json_hash(payload["identity"]),
        "source_tree_sha256": payload["identity"]["source_tree_sha256"],
        "pointer": json.loads((source / "LATEST.json").read_text(encoding="utf-8")),
        "files": entries,
    }
    write_json(destination / MANIFEST_NAME, manifest)
    write_json(destination / "dataset-metadata.json", {
        "title": "DoAnTTNT B0 Full Resume",
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}],
        "description": "Private B0 checkpoint transfer; no audio or credentials. Keep private.",
    })
    return manifest


def restore_payload(
    input_root: Path, run_root: Path, manifest_sha256: str,
    expected_step: int, source_tree_sha256: str, report_path: Path,
) -> dict[str, Any]:
    """Kiểm tra mount, checksum và identity trước khi công bố thư mục resume."""
    candidates = sorted(input_root.rglob(MANIFEST_NAME))
    if len(candidates) != 1:
        raise ValueError("Không xác định được duy nhất manifest checkpoint đầu vào.")
    manifest_path = candidates[0]
    if digest(manifest_path) != manifest_sha256:
        raise ValueError("Checksum manifest checkpoint đầu vào không khớp.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("global_step") != expected_step
            or manifest.get("source_tree_sha256") != source_tree_sha256):
        raise ValueError("Bước hoặc source của checkpoint đầu vào không khớp.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest checkpoint không có danh sách file hợp lệ.")
    seen: set[str] = set()
    for entry in entries:
        relative = entry["path"]
        if not isinstance(relative, str) or not PATH_PATTERN.fullmatch(relative) or relative in seen:
            raise ValueError("Đường dẫn checkpoint không an toàn hoặc bị lặp.")
        seen.add(relative)
        if entry["file"] != relative.replace("/", "__"):
            raise ValueError("Tên file truyền checkpoint không hợp lệ.")
        candidate = manifest_path.parent / entry["file"]
        if (candidate.is_symlink() or not candidate.is_file()
                or candidate.stat().st_size != entry["bytes"] or digest(candidate) != entry["sha256"]):
            raise ValueError("File checkpoint đầu vào thiếu hoặc sai checksum.")
    if "LATEST.json" not in seen:
        raise ValueError("Gói checkpoint thiếu pointer LATEST.json.")
    if run_root.exists():
        raise ValueError("Thư mục run đã tồn tại; từ chối ghi đè checkpoint.")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resume-stage-", dir=run_root.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        for entry in entries:
            target = staging / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest_path.parent / entry["file"], target)
            if digest(target) != entry["sha256"]:
                raise ValueError("Checksum checkpoint thay đổi khi sao chép.")
        payload = GenerationStore(staging).load()
        pointer = json.loads((staging / "LATEST.json").read_text(encoding="utf-8"))
        if (pointer != manifest["pointer"] or payload["controller"]["global_step"] != expected_step
                or json_hash(payload["identity"]) != manifest["identity_sha256"]
                or payload["identity"]["source_tree_sha256"] != source_tree_sha256):
            raise ValueError("Pointer, controller hoặc identity checkpoint không khớp.")
        os.replace(staging, run_root)
    report = {
        "passed": True, "restored_global_step": expected_step,
        "manifest_sha256": manifest_sha256, "source_tree_sha256": source_tree_sha256,
        "identity_sha256": manifest["identity_sha256"], "files_verified": len(entries),
        "pointer": pointer, "run_root": str(run_root),
        "training_resumed": False,
    }
    write_json(report_path, report)
    print(f"Đã kiểm chứng và khôi phục checkpoint bước {expected_step}; sẵn sàng resume.", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from scripts.kaggle_workflow import PROJECT_ROOT, source_files
    files = [{"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": digest(path)} for path in source_files()]
    restore_payload(args.input_root, args.run_root, args.manifest_sha256, args.expected_step, json_hash(files), args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
