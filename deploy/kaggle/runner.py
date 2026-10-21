#!/usr/bin/env python3
"""Run a prepared DoAnTTNT command profile inside a Kaggle kernel."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
PROJECT_ROOT = KAGGLE_WORKING_ROOT / "DoAnTTNT"
REPORT_PATH = KAGGLE_WORKING_ROOT / "reports" / "kaggle_cli_workflow.json"
RUN_CONFIG_PATH = Path(__file__).resolve().with_name("run-config.json")
RUN_CONFIG_SENTINEL = "__KAGGLE_RUN_CONFIG_B64__"
EMBEDDED_RUN_CONFIG_B64 = "__KAGGLE_RUN_CONFIG_B64__"
DATA_ROOT_TOKEN = "__KAGGLE_DATA_ROOT__"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_config(path: Path) -> dict[str, Any]:
    if EMBEDDED_RUN_CONFIG_B64 != RUN_CONFIG_SENTINEL:
        try:
            raw = base64.b64decode(
                EMBEDDED_RUN_CONFIG_B64.encode("ascii"), validate=True
            ).decode("utf-8")
            value = json.loads(raw)
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cấu hình kernel nhúng không hợp lệ.") from exc
    else:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Không thể đọc cấu hình kernel: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Cấu hình kernel phải là một JSON object.")
    commands = value.get("commands")
    if not isinstance(commands, list) or not commands:
        raise RuntimeError("Cấu hình kernel phải có ít nhất một lệnh.")
    for command in commands:
        if not isinstance(command, list) or not command:
            raise RuntimeError("Mỗi lệnh kernel phải là một danh sách chuỗi không rỗng.")
        if not all(isinstance(part, str) and part for part in command):
            raise RuntimeError("Mọi thành phần lệnh kernel phải là chuỗi không rỗng.")
    return value


def find_source_payload(source_dataset_slug: str) -> Path:
    expected = KAGGLE_INPUT_ROOT / source_dataset_slug / "project_source.zip"
    if expected.is_file():
        return expected
    expected_directory = KAGGLE_INPUT_ROOT / source_dataset_slug / "project_source"
    if expected_directory.is_dir():
        return expected_directory
    candidates = sorted(KAGGLE_INPUT_ROOT.rglob("project_source.zip"))
    candidates.extend(
        sorted(
            path
            for path in KAGGLE_INPUT_ROOT.rglob("project_source")
            if path.is_dir()
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Không xác định được duy nhất source payload trong /kaggle/input."
        )
    return candidates[0]


def find_data_root(data_dataset_slug: str) -> Path:
    """Tìm mount của data dataset trên cả layout Kaggle cũ và mới."""

    expected = KAGGLE_INPUT_ROOT / data_dataset_slug
    if expected.is_dir():
        return expected
    candidates = sorted(
        path
        for path in KAGGLE_INPUT_ROOT.rglob(data_dataset_slug)
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Không xác định được duy nhất data root trong /kaggle/input."
        )
    return candidates[0]


def resolve_runtime_commands(
    commands: Sequence[Sequence[str]], data_root: Path
) -> list[list[str]]:
    """Thay data-root token sau khi Kaggle đã mount dataset."""

    replacement = data_root.as_posix()
    return [
        [part.replace(DATA_ROOT_TOKEN, replacement) for part in command]
        for command in commands
    ]


def extract_source(bundle_path: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Thư mục project đã tồn tại, từ chối ghi đè: {destination}")
    destination.mkdir(parents=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(bundle_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Bundle chứa đường dẫn không an toàn: {info.filename}"
                ) from exc
        archive.extractall(destination)


def load_source_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Không thể đọc source manifest: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise RuntimeError("Source manifest không hợp lệ.")
    return value


def copy_verified_source_tree(
    source: Path, destination: Path, manifest: dict[str, Any]
) -> None:
    if destination.exists():
        raise RuntimeError(f"Thư mục project đã tồn tại, từ chối ghi đè: {destination}")
    source_root = source.resolve()
    expected_files: set[str] = set()
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise RuntimeError("Source manifest chứa mục file không hợp lệ.")
        relative_value = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(relative_value, str) or not isinstance(expected_hash, str):
            raise RuntimeError("Source manifest thiếu path hoặc SHA-256 hợp lệ.")
        relative = Path(relative_value)
        candidate = (source_root / relative).resolve()
        try:
            candidate.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Source manifest chứa đường dẫn không an toàn: {relative_value}"
            ) from exc
        normalized = relative.as_posix()
        if normalized in expected_files or not candidate.is_file():
            raise RuntimeError(f"Source file thiếu hoặc bị lặp: {relative_value}")
        if sha256_file(candidate) != expected_hash:
            raise RuntimeError(f"SHA-256 source file không khớp: {relative_value}")
        expected_files.add(normalized)

    actual_files = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError("Danh sách file source không khớp source manifest.")
    shutil.copytree(source_root, destination)


def run_command(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(cwd)},
        check=False,
    )
    return {
        "command": list(command),
        "return_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
    }


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def cleanup_project(project_root: Path, working_root: Path) -> None:
    resolved_project = project_root.resolve()
    resolved_working = working_root.resolve()
    if (
        resolved_project.parent != resolved_working
        or resolved_project.name != "DoAnTTNT"
    ):
        raise RuntimeError(f"Từ chối dọn thư mục project không an toàn: {project_root}")
    if resolved_project.exists():
        shutil.rmtree(resolved_project)


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    configure_console_encoding()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at,
        "status": "running",
        "commands": [],
    }
    try:
        config = load_run_config(RUN_CONFIG_PATH)
        source_slug = str(config.get("source_dataset_slug", ""))
        if not source_slug:
            raise RuntimeError("Thiếu source_dataset_slug trong cấu hình kernel.")
        source_payload = find_source_payload(source_slug)
        expected_sha256 = str(config.get("source_bundle_sha256", ""))
        if source_payload.is_file():
            actual_sha256 = sha256_file(source_payload)
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise RuntimeError("SHA-256 của source bundle không khớp cấu hình kernel.")
            extract_source(source_payload, PROJECT_ROOT)
            source_layout = "zip"
        else:
            manifest = load_source_manifest(source_payload.parent / "source_manifest.json")
            actual_sha256 = str(manifest.get("bundle_sha256", ""))
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise RuntimeError("SHA-256 trong source manifest không khớp cấu hình kernel.")
            copy_verified_source_tree(source_payload, PROJECT_ROOT, manifest)
            source_layout = "directory"
        report.update(
            {
                "profile": config.get("profile"),
                "source_dataset_slug": source_slug,
                "source_bundle_sha256": actual_sha256,
                "source_layout": source_layout,
            }
        )
        configured_commands = config["commands"]
        uses_data = any(
            DATA_ROOT_TOKEN in part
            for command in configured_commands
            for part in command
        )
        if uses_data:
            data_slug = str(config.get("data_dataset_slug", ""))
            if not data_slug:
                raise RuntimeError("Thiếu data_dataset_slug trong cấu hình kernel.")
            data_root = find_data_root(data_slug)
            commands = resolve_runtime_commands(configured_commands, data_root)
            report["data_dataset_slug"] = data_slug
            report["data_root"] = data_root.as_posix()
        else:
            commands = [list(command) for command in configured_commands]
        for command in commands:
            result = run_command(command, PROJECT_ROOT)
            report["commands"].append(result)
            write_report(report)
            if result["return_code"] != 0:
                raise RuntimeError(
                    "Lệnh Kaggle thất bại với mã "
                    f"{result['return_code']}: {' '.join(command)}"
                )
        report["status"] = "complete"
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        write_report(report)
        try:
            cleanup_project(PROJECT_ROOT, KAGGLE_WORKING_ROOT)
        except (OSError, RuntimeError) as exc:
            report["cleanup_warning"] = str(exc)
            write_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
