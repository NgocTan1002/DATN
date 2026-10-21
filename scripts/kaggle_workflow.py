#!/usr/bin/env python3
"""Manage the reproducible Kaggle dataset and kernel workflow for DoAnTTNT."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_TOKEN = "__KAGGLE_DATA_ROOT__"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / ".kaggle" / "project.json"
RUNNER_TEMPLATE_PATH = PROJECT_ROOT / "deploy" / "kaggle" / "runner.py"
KAGGLE_REF_PATTERN = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
SOURCE_PATTERNS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "configs/*.json",
    "deploy/kaggle/runner.py",
    "requirements-kaggle.txt",
)
RUN_CONFIG_ASSIGNMENT = (
    'EMBEDDED_RUN_CONFIG_B64 = "__KAGGLE_RUN_CONFIG_B64__"'
)


class WorkflowError(RuntimeError):
    """A user-facing Kaggle workflow failure."""


@dataclass(frozen=True)
class WorkflowConfig:
    data_dataset_ref: str
    source_dataset_ref: str
    source_dataset_title: str
    kernel_ref: str
    kernel_title: str
    workspace_dir: Path
    enable_gpu: bool
    enable_internet: bool
    default_profile: str
    profiles: Mapping[str, tuple[tuple[str, ...], ...]]

    @property
    def data_dataset_slug(self) -> str:
        return self.data_dataset_ref.split("/", maxsplit=1)[1]

    @property
    def source_dataset_slug(self) -> str:
        return self.source_dataset_ref.split("/", maxsplit=1)[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"Trường cấu hình '{key}' phải là chuỗi không rỗng.")
    return value.strip()


def require_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise WorkflowError(f"Trường cấu hình '{key}' phải là boolean.")
    return value


def load_config(path: Path) -> WorkflowConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkflowError(
            f"Không tìm thấy cấu hình Kaggle cục bộ: {path}. "
            "Sao chép configs/kaggle_cli.example.json vào .kaggle/project.json."
        ) from exc
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Cấu hình Kaggle không phải JSON hợp lệ: {path}") from exc
    if not isinstance(raw, dict):
        raise WorkflowError("Cấu hình Kaggle phải là một JSON object.")
    if raw.get("schema_version") != 1:
        raise WorkflowError("Chỉ hỗ trợ schema_version=1 cho cấu hình Kaggle.")

    refs = {
        key: require_string(raw, key)
        for key in ("data_dataset_ref", "source_dataset_ref", "kernel_ref")
    }
    for key, value in refs.items():
        if not KAGGLE_REF_PATTERN.fullmatch(value):
            raise WorkflowError(
                f"Trường '{key}' phải có dạng <owner>/<slug>: {value}"
            )
    owners = {value.split("/", maxsplit=1)[0] for value in refs.values()}
    if len(owners) != 1:
        raise WorkflowError("Dataset nguồn và kernel phải dùng cùng một owner Kaggle.")

    workspace_value = require_string(raw, "workspace_dir")
    workspace_dir = (PROJECT_ROOT / workspace_value).resolve()
    try:
        workspace_dir.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise WorkflowError("workspace_dir phải nằm bên trong repository.") from exc

    profiles_value = raw.get("profiles")
    if not isinstance(profiles_value, dict) or not profiles_value:
        raise WorkflowError("Cấu hình phải có ít nhất một profile lệnh.")
    profiles: dict[str, tuple[tuple[str, ...], ...]] = {}
    for name, commands_value in profiles_value.items():
        if not isinstance(name, str) or not name:
            raise WorkflowError("Tên profile phải là chuỗi không rỗng.")
        if not isinstance(commands_value, list) or not commands_value:
            raise WorkflowError(f"Profile '{name}' phải có ít nhất một lệnh.")
        commands: list[tuple[str, ...]] = []
        for command_value in commands_value:
            if not isinstance(command_value, list) or not command_value:
                raise WorkflowError(
                    f"Mỗi lệnh trong profile '{name}' phải là danh sách không rỗng."
                )
            if not all(isinstance(part, str) and part for part in command_value):
                raise WorkflowError(
                    f"Lệnh trong profile '{name}' chỉ được chứa chuỗi không rỗng."
                )
            commands.append(tuple(command_value))
        profiles[name] = tuple(commands)

    default_profile = require_string(raw, "default_profile")
    if default_profile not in profiles:
        raise WorkflowError(
            f"default_profile không tồn tại trong profiles: {default_profile}"
        )
    return WorkflowConfig(
        data_dataset_ref=refs["data_dataset_ref"],
        source_dataset_ref=refs["source_dataset_ref"],
        source_dataset_title=require_string(raw, "source_dataset_title"),
        kernel_ref=refs["kernel_ref"],
        kernel_title=require_string(raw, "kernel_title"),
        workspace_dir=workspace_dir,
        enable_gpu=require_bool(raw, "enable_gpu"),
        enable_internet=require_bool(raw, "enable_internet"),
        default_profile=default_profile,
        profiles=profiles,
    )


def run_process(
    command: Sequence[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            },
            check=False,
            capture_output=capture_output,
            text=True,
        )
    except OSError as exc:
        raise WorkflowError(f"Không thể chạy lệnh: {' '.join(command)}") from exc


def require_success(
    completed: subprocess.CompletedProcess[str], command_name: str
) -> None:
    if completed.returncode == 0:
        return
    detail = (completed.stderr or completed.stdout or "").strip()
    if len(detail) > 1000:
        detail = detail[-1000:]
    raise WorkflowError(
        f"Lệnh {command_name} thất bại với mã {completed.returncode}. {detail}"
    )


def current_git_state() -> tuple[str | None, bool | None]:
    commit_result = run_process(
        ["git", "rev-parse", "HEAD"], capture_output=True
    )
    status_result = run_process(
        ["git", "status", "--porcelain"], capture_output=True
    )
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    dirty = bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
    return commit, dirty


def source_files() -> list[Path]:
    found: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        found.update(path for path in PROJECT_ROOT.glob(pattern) if path.is_file())
    filtered = [
        path
        for path in found
        if "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    if not filtered:
        raise WorkflowError("Không tìm thấy file nguồn để đóng gói.")
    return sorted(filtered, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def write_deterministic_zip(output_path: Path, files: Sequence[Path]) -> None:
    temporary_path = output_path.with_suffix(".zip.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    with zipfile.ZipFile(
        temporary_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    os.replace(temporary_path, output_path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def bundle_source(config: WorkflowConfig) -> dict[str, Any]:
    output_dir = config.workspace_dir / "source_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "project_source.zip"
    files = source_files()
    write_deterministic_zip(bundle_path, files)
    commit, dirty = current_git_state()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "file_count": len(files),
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256_file(bundle_path),
        "files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    write_json(output_dir / "source_manifest.json", manifest)
    metadata = {
        "title": config.source_dataset_title,
        "id": config.source_dataset_ref,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Private reproducible source snapshot for the DoAnTTNT Kaggle workflow."
        ),
    }
    write_json(output_dir / "dataset-metadata.json", metadata)
    return manifest


def parse_json_output(completed: subprocess.CompletedProcess[str]) -> Any:
    require_success(completed, "Kaggle CLI")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise WorkflowError("Kaggle CLI không trả về JSON hợp lệ.") from exc


def kaggle_dataset_exists(dataset_ref: str) -> bool:
    owner, slug = dataset_ref.split("/", maxsplit=1)
    completed = run_process(
        [
            "kaggle",
            "datasets",
            "list",
            "--mine",
            "--search",
            slug,
            "--page-size",
            "100",
            "--format",
            "json",
        ],
        capture_output=True,
    )
    require_success(completed, "Kaggle CLI")
    if completed.stdout.strip().lower() == "no datasets found":
        return False
    values = parse_json_output(completed)
    if not isinstance(values, list):
        raise WorkflowError("Danh sách dataset Kaggle có định dạng không hợp lệ.")
    expected = f"{owner}/{slug}".lower()
    return any(
        isinstance(value, dict) and str(value.get("ref", "")).lower() == expected
        for value in values
    )


def publish_source(config: WorkflowConfig, message: str) -> str:
    manifest = bundle_source(config)
    source_dir = config.workspace_dir / "source_dataset"
    if kaggle_dataset_exists(config.source_dataset_ref):
        command = [
            "kaggle",
            "datasets",
            "version",
            "--path",
            str(source_dir),
            "--message",
            message,
            "--dir-mode",
            "skip",
            "--keep-tabular",
        ]
        action = "versioned"
    else:
        command = [
            "kaggle",
            "datasets",
            "create",
            "--path",
            str(source_dir),
            "--dir-mode",
            "skip",
            "--keep-tabular",
        ]
        action = "created"
    completed = run_process(command)
    require_success(completed, "publish-source")
    print(
        "Đã cập nhật source dataset riêng tư: "
        f"{config.source_dataset_ref} ({manifest['bundle_sha256']})."
    )
    return action


def render_commands(
    config: WorkflowConfig, profile: str
) -> tuple[tuple[str, ...], ...]:
    if profile not in config.profiles:
        raise WorkflowError(f"Không tồn tại profile Kaggle: {profile}")
    replacements = {
        "data_root": DATA_ROOT_TOKEN,
        "source_root": f"/kaggle/input/{config.source_dataset_slug}",
        "project_root": "/kaggle/working/DoAnTTNT",
        "working_root": "/kaggle/working",
    }
    rendered: list[tuple[str, ...]] = []
    for command in config.profiles[profile]:
        try:
            rendered.append(tuple(part.format(**replacements) for part in command))
        except KeyError as exc:
            raise WorkflowError(
                f"Profile '{profile}' dùng placeholder không được hỗ trợ: {exc}"
            ) from exc
    return tuple(rendered)


def required_dataset_sources(
    config: WorkflowConfig, commands: Sequence[Sequence[str]]
) -> list[str]:
    uses_data = any(
        DATA_ROOT_TOKEN in part for command in commands for part in command
    )
    sources = [config.source_dataset_ref]
    if uses_data:
        sources.insert(0, config.data_dataset_ref)
    return sources


def prepare_kernel(config: WorkflowConfig, profile: str) -> Path:
    source_manifest_path = (
        config.workspace_dir / "source_dataset" / "source_manifest.json"
    )
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            "Chưa có source bundle hợp lệ; hãy chạy bundle-source hoặc publish-source."
        ) from exc
    bundle_sha256 = source_manifest.get("bundle_sha256")
    if not isinstance(bundle_sha256, str) or len(bundle_sha256) != 64:
        raise WorkflowError("source_manifest.json thiếu bundle_sha256 hợp lệ.")

    kernel_dir = config.workspace_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    commands = render_commands(config, profile)
    run_config = {
        "schema_version": 1,
        "profile": profile,
        "source_dataset_slug": config.source_dataset_slug,
        "data_dataset_slug": config.data_dataset_slug,
        "source_bundle_sha256": bundle_sha256,
        "commands": [list(command) for command in commands],
    }
    try:
        runner_source = RUNNER_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkflowError(
            f"Không thể đọc runner template: {RUNNER_TEMPLATE_PATH}"
        ) from exc
    if runner_source.count(RUN_CONFIG_ASSIGNMENT) != 1:
        raise WorkflowError("Runner template thiếu marker cấu hình nhúng duy nhất.")
    encoded_config = base64.b64encode(
        json.dumps(run_config, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    rendered_runner = runner_source.replace(
        RUN_CONFIG_ASSIGNMENT,
        f'EMBEDDED_RUN_CONFIG_B64 = "{encoded_config}"',
    )
    (kernel_dir / "runner.py").write_text(rendered_runner, encoding="utf-8")
    write_json(kernel_dir / "run-config.json", run_config)
    metadata = {
        "id": config.kernel_ref,
        "title": config.kernel_title,
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": config.enable_gpu,
        "enable_tpu": False,
        "enable_internet": config.enable_internet,
        "dataset_sources": required_dataset_sources(config, commands),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return kernel_dir


def push_kernel(config: WorkflowConfig, profile: str) -> None:
    kernel_dir = prepare_kernel(config, profile)
    completed = run_process(
        ["kaggle", "kernels", "push", "--path", str(kernel_dir)]
    )
    require_success(completed, "push-kernel")
    print(f"Đã gửi kernel riêng tư: {config.kernel_ref} (profile={profile}).")


def read_dataset_files(dataset_ref: str) -> list[dict[str, Any]]:
    completed = run_process(
        [
            "kaggle",
            "datasets",
            "files",
            dataset_ref,
            "--page-size",
            "200",
            "--format",
            "json",
        ],
        capture_output=True,
    )
    values = parse_json_output(completed)
    if not isinstance(values, list):
        raise WorkflowError("Danh sách file Kaggle có định dạng không hợp lệ.")
    return [value for value in values if isinstance(value, dict)]


def doctor(config: WorkflowConfig, offline: bool) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        raise WorkflowError("Không tìm thấy Kaggle CLI trong PATH.")
    version = run_process(["kaggle", "--version"], capture_output=True)
    require_success(version, "kaggle --version")
    print(version.stdout.strip())
    print(f"Cấu hình cục bộ hợp lệ: {config.workspace_dir}")
    if offline:
        print("Đã bỏ qua kiểm tra API theo yêu cầu --offline.")
        return

    files = read_dataset_files(config.data_dataset_ref)
    names = {str(value.get("name", "")) for value in files}
    parquet_count = sum(name.lower().endswith(".parquet") for name in names)
    required = {"development_20k_v2.csv", "smoke_train.csv", "smoke_dev.csv"}
    missing = sorted(required - names)
    if parquet_count != 68 or missing:
        raise WorkflowError(
            "Dataset đầu vào không đạt: "
            f"parquet={parquet_count}, thiếu={missing or 'không'}."
        )
    print(
        "Kết nối Kaggle đạt; dataset đầu vào có 68 Parquet và đủ manifest/smoke split."
    )
    source_exists = kaggle_dataset_exists(config.source_dataset_ref)
    print(
        "Source dataset: "
        + ("đã tồn tại." if source_exists else "chưa tạo; chạy publish-source.")
    )


def kernel_status(config: WorkflowConfig) -> None:
    completed = run_process(
        ["kaggle", "kernels", "status", config.kernel_ref]
    )
    require_success(completed, "kernel-status")


def wait_for_kernel(
    config: WorkflowConfig, *, timeout_seconds: int, interval_seconds: int
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        completed = run_process(
            ["kaggle", "kernels", "status", config.kernel_ref],
            capture_output=True,
        )
        require_success(completed, "kernel-status")
        output = completed.stdout.strip()
        print(output)
        normalized = output.lower()
        if any(word in normalized for word in ("complete", "error", "cancel")):
            if "complete" not in normalized:
                raise WorkflowError("Kernel kết thúc nhưng không thành công.")
            return
        if time.monotonic() >= deadline:
            raise WorkflowError("Hết thời gian chờ kernel hoàn thành.")
        time.sleep(interval_seconds)


def download_output(config: WorkflowConfig, output_path: Path, force: bool) -> None:
    destination = output_path.resolve()
    try:
        destination.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise WorkflowError("Thư mục output phải nằm bên trong repository.") from exc
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "kernels",
        "output",
        config.kernel_ref,
        "--path",
        str(destination),
    ]
    if force:
        command.append("--force")
    completed = run_process(command)
    require_success(completed, "download-output")
    print(f"Đã tải output Kaggle về: {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="action", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Kiểm tra CLI và dataset.")
    doctor_parser.add_argument("--offline", action="store_true")

    subparsers.add_parser("bundle-source", help="Đóng gói source hiện hành.")

    publish_parser = subparsers.add_parser(
        "publish-source", help="Tạo hoặc cập nhật private source dataset."
    )
    publish_parser.add_argument("--message", required=True)
    publish_parser.add_argument(
        "--confirm-private-upload",
        action="store_true",
        required=True,
        help="Xác nhận cho phép upload source bundle đã kiểm tra lên private dataset.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-kernel", help="Sinh kernel-metadata và run config."
    )
    prepare_parser.add_argument("--profile")

    push_parser = subparsers.add_parser(
        "push-kernel", help="Chuẩn bị và gửi kernel lên Kaggle."
    )
    push_parser.add_argument("--profile")

    deploy_parser = subparsers.add_parser(
        "deploy", help="Publish source rồi gửi kernel trong một lệnh."
    )
    deploy_parser.add_argument("--profile")
    deploy_parser.add_argument("--message", required=True)
    deploy_parser.add_argument(
        "--confirm-private-upload",
        action="store_true",
        required=True,
        help="Xác nhận cho phép upload source bundle đã kiểm tra lên private dataset.",
    )

    subparsers.add_parser("status", help="Đọc trạng thái kernel hiện tại.")

    wait_parser = subparsers.add_parser("wait", help="Chờ kernel kết thúc.")
    wait_parser.add_argument("--timeout-seconds", type=int, default=1800)
    wait_parser.add_argument("--interval-seconds", type=int, default=20)

    output_parser = subparsers.add_parser(
        "download-output", help="Tải output kernel về workspace."
    )
    output_parser.add_argument(
        "--path",
        type=Path,
        default=PROJECT_ROOT / "kaggle_outputs" / "latest",
    )
    output_parser.add_argument("--force", action="store_true")
    return parser


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config.resolve())
        profile = getattr(args, "profile", None) or config.default_profile
        if args.action == "doctor":
            doctor(config, args.offline)
        elif args.action == "bundle-source":
            manifest = bundle_source(config)
            print(
                "Đã đóng gói "
                f"{manifest['file_count']} file, SHA-256={manifest['bundle_sha256']}."
            )
        elif args.action == "publish-source":
            publish_source(config, args.message)
        elif args.action == "prepare-kernel":
            path = prepare_kernel(config, profile)
            print(f"Đã chuẩn bị kernel tại: {path}")
        elif args.action == "push-kernel":
            push_kernel(config, profile)
        elif args.action == "deploy":
            publish_source(config, args.message)
            push_kernel(config, profile)
        elif args.action == "status":
            kernel_status(config)
        elif args.action == "wait":
            if args.timeout_seconds <= 0 or args.interval_seconds <= 0:
                raise WorkflowError("Thời gian chờ và chu kỳ kiểm tra phải dương.")
            wait_for_kernel(
                config,
                timeout_seconds=args.timeout_seconds,
                interval_seconds=args.interval_seconds,
            )
        elif args.action == "download-output":
            download_output(config, args.path, args.force)
        else:
            parser.error(f"Action không được hỗ trợ: {args.action}")
        return 0
    except WorkflowError as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
