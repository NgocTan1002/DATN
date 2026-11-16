#!/usr/bin/env python3
"""Entrypoint chạy XLS-R trên Google Colab (D033 mục 2, 3, 4): kiểm phiên bản, chép về ổ cục bộ, kiểm checksum và đồng bộ Drive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_xlsr as runner
from src.data.packed import verify_packed_artifact

EXPECTED_TORCH_BASE = '2.11.0'
EXPECTED_TORCH_FULL = '2.11.0+cu130'
EXPECTED_CUDA_VERSION = '13.0'
EXPECTED_TRANSFORMERS_VERSION = '4.57.6'
EXPECTED_NUMPY_VERSION = '2.1.3'
DEFAULT_WORK_DIR = Path('/content/work')


def reject_test_paths(*values: Path | str | None) -> None:
    """Không bao giờ cho phép truyền đường dẫn hoặc tham số liên quan tới tập test."""
    for value in values:
        if value is None:
            continue
        text = str(value).replace('\\', '/').lower()
        parts = [part for part in text.split('/') if part]
        for part in parts:
            if part in ('test', 'closed_test', 'open_test') or 'closed_test' in part or 'open_test' in part or part.startswith('test_') or part.endswith('_test'):
                raise ValueError(f'Entrypoint Colab cấm truyền hoặc truy cập artifact test: {value}')


def verify_local_work_dir_not_on_drive(work_dir: Path, drive_data_dir: Path, drive_output_dir: Path) -> None:
    """Chặn đọc memmap trực tiếp trên Google Drive."""
    work_resolved = work_dir.resolve()
    for drive_dir in (drive_data_dir.resolve(), drive_output_dir.resolve()):
        if work_resolved == drive_dir or work_resolved.is_relative_to(drive_dir):
            raise ValueError('Thư mục làm việc cục bộ không được nằm trong thư mục Google Drive (cấm đọc memmap trực tiếp trên Drive).')
    normalized = work_resolved.as_posix().lower()
    if normalized.startswith('/content/drive/') or normalized == '/content/drive':
        raise ValueError('Thư mục làm việc cục bộ không được nằm dưới /content/drive (cấm đọc memmap trực tiếp trên Drive).')


def load_expected_environment(env_reference_path: Path | None = None) -> dict[str, str]:
    expected = {
        'torch': EXPECTED_TORCH_FULL,
        'cuda': EXPECTED_CUDA_VERSION,
        'numpy': EXPECTED_NUMPY_VERSION,
        'transformers': EXPECTED_TRANSFORMERS_VERSION,
    }
    if env_reference_path is not None and env_reference_path.is_file():
        payload = json.loads(env_reference_path.read_text(encoding='utf-8'))
        eff = payload.get('effective_environment') or {}
        dep = ((payload.get('dependency_and_cpu_check') or {}).get('versions')) or {}
        if eff.get('torch'):
            expected['torch'] = str(eff['torch'])
        if eff.get('cuda'):
            expected['cuda'] = str(eff['cuda'])
        if eff.get('numpy'):
            expected['numpy'] = str(eff['numpy'])
        if dep.get('transformers'):
            expected['transformers'] = str(dep['transformers'])
    return expected


def record_pip_freeze(log_path: Path | None = None) -> list[str]:
    """Ghi pip freeze vào log mà không tự nâng hay hạ phiên bản."""
    try:
        raw = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
    except Exception:
        import importlib.metadata as metadata
        lines = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in metadata.distributions()
            if dist.metadata.get('Name')
        )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'event': 'pip_freeze',
                'packages': lines,
            }, ensure_ascii=False) + '\n')
    return lines


def verify_colab_environment(
    *,
    torch_version: str | None = None,
    cuda_version: str | None = None,
    transformers_version: str | None = None,
    numpy_version: str | None = None,
    env_reference_path: Path | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Kiểm phiên bản thư viện theo effective_environment của T14; sai thì dừng ngay."""
    import numpy as np
    import torch
    import transformers

    actual_torch = torch.__version__ if torch_version is None else torch_version
    actual_cuda = torch.version.cuda if cuda_version is None else cuda_version
    actual_transformers = transformers.__version__ if transformers_version is None else transformers_version
    actual_numpy = np.__version__ if numpy_version is None else numpy_version

    expected = load_expected_environment(env_reference_path)
    expected_torch_base = expected['torch'].split('+', maxsplit=1)[0]
    actual_torch_base = actual_torch.split('+', maxsplit=1)[0]

    if actual_torch_base != expected_torch_base:
        raise ValueError(
            f"Phiên bản PyTorch ({actual_torch}) không khớp effective_environment (yêu cầu {expected['torch']}); không tự nâng hay hạ phiên bản."
        )
    if actual_transformers != expected['transformers']:
        raise ValueError(
            f"Phiên bản transformers ({actual_transformers}) không khớp D028/T14 (yêu cầu {expected['transformers']}); không tự nâng hay hạ phiên bản."
        )
    if actual_numpy != expected['numpy']:
        raise ValueError(
            f"Phiên bản numpy ({actual_numpy}) không khớp effective_environment (yêu cầu {expected['numpy']}); không tự nâng hay hạ phiên bản."
        )

    exact_cuda_build = actual_torch == expected['torch']
    cuda_note = (
        f"Khớp đúng {expected['torch']}."
        if exact_cuda_build
        else f"Chấp nhận PyTorch {actual_torch_base} với biến thể CUDA khác ({actual_torch}, cuda={actual_cuda}); chuẩn T14 là {expected['torch']}."
    )
    freeze_lines = record_pip_freeze(log_path)
    summary = {
        'torch': actual_torch,
        'cuda': actual_cuda,
        'transformers': actual_transformers,
        'numpy': actual_numpy,
        'expected': expected,
        'exact_cuda_build': exact_cuda_build,
        'cuda_note': cuda_note,
        'pip_freeze_count': len(freeze_lines),
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'event': 'environment_verified',
                **summary,
            }, ensure_ascii=False) + '\n')
    return summary


def _copy_directory_fresh(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f'Không tìm thấy thư mục nguồn trên Drive: {source}')
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def resolve_drive_weights_dir(drive_data_dir: Path) -> Path:
    for candidate in (
        drive_data_dir / 'wav2vec2-xls-r-300m',
        drive_data_dir / 'weights',
        drive_data_dir / 'pretrained',
    ):
        if (candidate / 'pytorch_model.bin').is_file():
            return candidate
    if (drive_data_dir / 'pytorch_model.bin').is_file():
        return drive_data_dir
    raise ValueError(f'Không tìm thấy trọng số pretrained (pytorch_model.bin) trong {drive_data_dir}.')


def stage_and_verify_artifacts(
    *,
    drive_data_dir: Path,
    work_dir: Path,
    plan: Mapping[str, Any],
    training_seed: int,
    include_train: bool = True,
) -> dict[str, Path]:
    """Chép packed train, dev-full và trọng số pretrained từ Drive sang ổ cục bộ rồi kiểm checksum."""
    reject_test_paths(drive_data_dir, work_dir)
    if plan.get('data', {}).get('use_test') is not False:
        raise ValueError('Plan cấu hình use_test khác false; từ chối chạy.')

    local_data_root = work_dir / 'data'
    local_weights_dir = work_dir / 'pretrained' / 'wav2vec2-xls-r-300m'

    drive_weights = resolve_drive_weights_dir(drive_data_dir)
    _copy_directory_fresh(drive_weights, local_weights_dir)
    runner.verify_pretrained(local_weights_dir, dict(plan))

    drive_dev = drive_data_dir / 'closed_dev_full_v1'
    local_dev = local_data_root / 'closed_dev_full_v1'
    _copy_directory_fresh(drive_dev, local_dev)
    dev_ds = runner.open_artifact(local_dev, dict(plan), training=False, seed=training_seed)
    dev_ds.close()
    verify_packed_artifact(local_dev)

    staged: dict[str, Path] = {
        'dev_artifact': local_dev,
        'pretrained_dir': local_weights_dir,
    }
    if include_train:
        drive_train = drive_data_dir / 'train_development_20k_v2'
        local_train = local_data_root / 'train_development_20k_v2'
        _copy_directory_fresh(drive_train, local_train)
        train_ds = runner.open_artifact(local_train, dict(plan), training=True, seed=training_seed)
        train_ds.close()
        verify_packed_artifact(local_train)
        staged['train_artifact'] = local_train
    return staged


def sync_outputs_to_drive(
    *,
    local_run_root: Path | None,
    local_report_json: Path | None,
    local_log_path: Path | None,
    drive_output_dir: Path,
    extra_files: Sequence[Path] = (),
) -> None:
    """Đồng bộ checkpoint, report và log từ ổ cục bộ về Google Drive."""
    drive_output_dir.mkdir(parents=True, exist_ok=True)
    if local_run_root is not None and local_run_root.exists():
        target_ckpt = drive_output_dir / 'checkpoints' / local_run_root.name
        target_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_run_root, target_ckpt, dirs_exist_ok=True)
    if local_report_json is not None and local_report_json.is_file():
        target_report = drive_output_dir / 'reports' / local_report_json.name
        target_report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_report_json, target_report)
    if local_log_path is not None and local_log_path.is_file():
        target_log = drive_output_dir / 'logs' / local_log_path.name
        target_log.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_log_path, target_log)
    for extra in extra_files:
        if extra.is_file():
            target_extra = drive_output_dir / 'reports' / extra.name
            target_extra.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, target_extra)


def build_colab_runner_command(
    *,
    python_exe: str,
    plan_path: Path,
    run_id: str,
    training_seed: int,
    device: str,
    expected_gpu: str,
    train_artifact: Path,
    dev_artifact: Path,
    pretrained_dir: Path,
    run_root: Path,
    report_json: Path,
    session_steps: int,
    deadline_unix: float,
    resume: bool,
) -> list[str]:
    reject_test_paths(plan_path, train_artifact, dev_artifact, pretrained_dir, run_root, report_json)
    command = [
        python_exe,
        str(ROOT / 'scripts/train_xlsr.py'),
        '--plan', str(plan_path),
        '--run-id', run_id,
        '--training-seed', str(training_seed),
        '--device', device,
        '--expected-gpu', expected_gpu,
        '--train-artifact', str(train_artifact),
        '--dev-artifact', str(dev_artifact),
        '--pretrained-dir', str(pretrained_dir),
        '--run-root', str(run_root),
        '--report-json', str(report_json),
        '--session-steps', str(session_steps),
        '--deadline-unix', str(deadline_unix),
    ]
    if resume:
        command.append('--resume')
    return command


def run_runner_with_checkpoint_sync(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    local_run_root: Path,
    local_report_json: Path,
    local_log_path: Path,
    drive_output_dir: Path,
    on_checkpoint_sync: Callable[[], None] | None = None,
) -> int:
    local_log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        list(command),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    assert proc.stdout is not None
    with local_log_path.open('a', encoding='utf-8') as log_handle:
        for line in proc.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end='', flush=True)
            stripped = line.strip()
            if stripped.startswith('{'):
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    event = {}
                if isinstance(event, dict) and event.get('event') == 'checkpoint':
                    sync_outputs_to_drive(
                        local_run_root=local_run_root,
                        local_report_json=local_report_json,
                        local_log_path=local_log_path,
                        drive_output_dir=drive_output_dir,
                    )
                    if on_checkpoint_sync is not None:
                        on_checkpoint_sync()
    return_code = int(proc.wait())
    sync_outputs_to_drive(
        local_run_root=local_run_root,
        local_report_json=local_report_json,
        local_log_path=local_log_path,
        drive_output_dir=drive_output_dir,
    )
    if return_code != 0:
        raise RuntimeError(f'Runner XLS-R trên Colab kết thúc với mã lỗi {return_code}.')
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-data-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument('--drive-output-dir', type=Path, required=True)
    parser.add_argument('--run-id', type=str, required=True)
    parser.add_argument('--training-seed', type=int, required=True)
    parser.add_argument('--expected-gpu', type=str, required=True)
    parser.add_argument('--mode', choices=('train', 'resume', 'parity'), required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max-session-hours', type=float, default=11.0)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/xlsr_training.json')
    parser.add_argument('--env-reference', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--generation', type=str)
    parser.add_argument('--reference-report', type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started_unix = time.time()
    args = build_parser().parse_args(argv)
    reject_test_paths(
        args.drive_data_dir,
        args.work_dir,
        args.drive_output_dir,
        args.plan,
        args.checkpoint,
        args.reference_report,
    )
    verify_local_work_dir_not_on_drive(args.work_dir, args.drive_data_dir, args.drive_output_dir)

    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    runner.validate_plan(plan, args.run_id, args.training_seed)

    local_log_path = args.work_dir / 'logs' / f'colab_{args.mode}_{args.run_id}_seed{args.training_seed}.log'
    verify_colab_environment(
        env_reference_path=args.env_reference,
        log_path=local_log_path,
    )

    staged = stage_and_verify_artifacts(
        drive_data_dir=args.drive_data_dir,
        work_dir=args.work_dir,
        plan=plan,
        training_seed=args.training_seed,
        include_train=args.mode in ('train', 'resume'),
    )

    env = dict(
        os.environ,
        PYTHONUTF8='1',
        PYTHONIOENCODING='utf-8',
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        USE_TF='0',
        USE_FLAX='0',
    )

    if args.mode == 'parity':
        if args.checkpoint is None or args.reference_report is None:
            raise ValueError('Chế độ parity cần đủ --checkpoint và --reference-report.')
        parity_json = args.work_dir / 'reports' / f'xlsr_parity_{args.run_id}_seed{args.training_seed}.json'
        parity_md = args.work_dir / 'reports' / f'xlsr_parity_{args.run_id}_seed{args.training_seed}.md'
        parity_cmd = [
            sys.executable,
            str(ROOT / 'scripts/check_xlsr_env_parity.py'),
            '--plan', str(args.plan),
            '--checkpoint', str(args.checkpoint),
            '--reference-report', str(args.reference_report),
            '--dev-artifact', str(staged['dev_artifact']),
            '--pretrained-dir', str(staged['pretrained_dir']),
            '--device', args.device,
            '--expected-gpu', args.expected_gpu,
            '--output-json', str(parity_json),
            '--output-md', str(parity_md),
        ]
        if args.generation:
            parity_cmd.extend(['--generation', args.generation])
        completed = subprocess.run(parity_cmd, env=env, check=False)
        sync_outputs_to_drive(
            local_run_root=None,
            local_report_json=parity_json,
            local_log_path=local_log_path,
            drive_output_dir=args.drive_output_dir,
            extra_files=(parity_md,),
        )
        if completed.returncode != 0:
            raise RuntimeError(f'Kiểm tương đương môi trường thất bại với mã {completed.returncode}.')
        return 0

    local_run_root = args.work_dir / 'checkpoints' / f'xlsr_{args.run_id}_seed{args.training_seed}'
    local_report_json = args.work_dir / 'reports' / f'xlsr_{args.run_id}_seed{args.training_seed}.json'
    if args.mode == 'resume':
        drive_ckpt = args.props_checkpoint if hasattr(args, 'props_checkpoint') else (args.checkpoint or (args.drive_output_dir / 'checkpoints' / local_run_root.name))
        if drive_ckpt.exists() and not local_run_root.exists():
            local_run_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(drive_ckpt, local_run_root)

    deadline_unix = started_unix + float(args.max_session_hours) * 3600.0
    command = build_colab_runner_command(
        python_exe=sys.executable,
        plan_path=args.plan,
        run_id=args.run_id,
        training_seed=args.training_seed,
        device=args.device,
        expected_gpu=args.expected_gpu,
        train_artifact=staged['train_artifact'],
        dev_artifact=staged['dev_artifact'],
        pretrained_dir=staged['pretrained_dir'],
        run_root=local_run_root,
        report_json=local_report_json,
        session_steps=int(plan['training']['max_optimizer_steps']),
        deadline_unix=deadline_unix,
        resume=(args.mode == 'resume'),
    )
    return run_runner_with_checkpoint_sync(
        command,
        env=env,
        local_run_root=local_run_root,
        local_report_json=local_report_json,
        local_log_path=local_log_path,
        drive_output_dir=args.drive_output_dir,
    )


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f'LỖI: {error}', file=sys.stderr)
        raise SystemExit(1) from error
