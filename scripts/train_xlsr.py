#!/usr/bin/env python3
"""T15: runner XLS-R D028, packed train/dev-full, không dùng test."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import tempfile
import platform
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import default_collate
from scripts.train_b0 import _set_seed, _sample_selection_sha256
from scripts.kaggle_workflow import source_files
from scripts.benchmark_xlsr_cuda import sha256
from src.data.packed import PackedManifestDataset, verify_packed_artifact
from src.data.xlsr import apply_xlsr_waveform_normalization
from src.data import lowpass
from src.metrics.evaluation import evaluate_development_scores
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone
from src.training.full_training import GenerationStore, run_training, write_json
from src.training.sampler import ResumableBatchSampler

NORMALIZATION = 'processor_zero_mean_unit_variance'
GRID = [{'run_id': 'X0', 'layers': 0, 'encoder_lr': None}] + [
    {'run_id': f'X1-L{n}-lr{lr}', 'layers': n, 'encoder_lr': float(lr)}
    for n in (2, 4, 8) for lr in ('1e-5', '3e-5')]
VARIANTS = {'X1-L8-lr3e-5-lp7k': {'base_run_id': 'X1-L8-lr3e-5',
            'input_condition': 'lowpass_7k', 'decision': 'D035'}}
SCALE2_GRID = [{'run_id': 'X1-L8-lr3e-5', 'layers': 8, 'encoder_lr': 3e-05}]
SCALE2_DATA_SPEC = {
    'train_manifest': 'data/manifests/development_scale2_v1.csv',
    'train_manifest_sha256': '6533fced4b3a25ebab9f037a62d34774b44c2fa1f00d17040fd815bd17fa51a3',
    'train_artifact': 'data/processed/packed/train_development_scale2_v1',
    'train_artifact_waveforms_sha256': '260d93d3762f20b564400d023d599d4151d8e74be618441aed255675f3e3da5e',
    'train_artifact_index_sha256': '7eb18914d7d2ea3b101273d0ab4a4e882a0cac4c8676ae5c018a281086aa8080',
    'train_metadata_sha256': '3341f75b45f089c4b6b2f71ca3659233a87c6232d416a247b5886d7e0ecec48d',
    'train_samples': 31770,
}
SCALE2_OUTPUTS = {
    'root_template': 'checkpoints/xlsr_scale2_{run_id}_seed{training_seed}',
    'report_json_template': 'reports/xlsr_scale2_{run_id}_seed{training_seed}.json',
}


class InputConditionDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Chỉ thay waveform đã crop/peak; giữ nguyên metadata và crop stream packed."""

    def __init__(self, dataset: PackedManifestDataset, condition: str) -> None:
        self.dataset = dataset
        self.condition = condition

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item['waveform'] = lowpass.apply_condition(item['waveform'], self.condition)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def input_condition_identity(run_id: str) -> dict[str, Any] | None:
    variant = VARIANTS.get(run_id)
    if variant is None:
        return None
    return {'decision': variant['decision'], 'condition': variant['input_condition'],
            'base_run_id': variant['base_run_id'],
            'filter': lowpass.filter_parameters(variant['input_condition']),
            'implementation_sha256': sha256(Path(lowpass.__file__))}


def _validate_shared_d028_fields(plan: dict[str, Any]) -> None:
    """Kiểm tra các trường bất biến dùng chung giữa D028 (×1) và D036 (scale2)."""
    d, t, m, e, p = (plan[k] for k in ('data', 'training', 'model', 'evaluation', 'pretrained'))
    shared_training = {
        'batch_size': 16, 'num_workers': 0, 'optimizer': 'Adam',
        'classifier_learning_rate': .001, 'encoder_learning_rates': [1e-5, 3e-5],
        'max_epochs': 10, 'scheduler': None, 'gradient_clipping': None,
        'drop_last': False, 'shuffle': True, 'loss': 'BCEWithLogitsLoss', 'gpus_per_run': 1,
        'training_seeds': [2026, 2027, 2028],
    }
    if (any(t.get(k) != v for k, v in shared_training.items())
        or d['data_seed'] != 2026 or d['use_test'] is not False
        or d['backend'] != 'packed' or d['train_split'] != 'closed_train'
        or d['development_split'] != 'closed_dev'
        or d['development_samples'] != 38777
        or m['amplitude_policy'] != 'peak' or m['precision'] != 'float32' or m['amp']
        or m['classifier_hidden_features'] != 128 or m['classifier_dropout'] != .2
        or p['waveform_normalization'] != NORMALIZATION
        or p['backbone_training'] or p['frozen_blocks_training'] or not p['opened_blocks_training']
        or p['effective_layerdrop'] != 0 or p['effective_apply_spec_augment']
        or p['layerdrop'] != .1 or p['configured_apply_spec_augment'] is not True
        or p['mode_decision'] != 'D029'
        or e['patience'] != 2 or not e['strict_improvement'] or e['min_delta'] != 0
        or e['equal_eer_action'] != 'keep_earliest_checkpoint'
        or not e['full_development'] or e['partial_epoch_evaluation']
        or e['selection_metric'] != 'eer' or e['mode'] != 'min'
        or e['minimum_completed_epochs'] != 1 or e['allow_loss_tie_break']):
        raise ValueError('Đặc tả không khớp D028.')
    if (p['model_id'] != 'facebook/wav2vec2-xls-r-300m'
        or p['revision'] != '1a640f32ac3e39899438a2931f9924c02f080a54'
        or p['weights_sha256'] != 'd5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0'
        or p['config_sha256'] != '0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac'
        or p['preprocessor_config_sha256'] != 'a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335'):
        raise ValueError('Pretrained không khớp revision/checksum khóa D021.')


def validate_plan(plan: dict[str, Any], run_id: str, seed: int) -> dict[str, Any]:
    """Từ chối thay đổi giao thức D028/D036 hoặc run ngoài lưới."""
    _validate_shared_d028_fields(plan)
    d, t = plan['data'], plan['training']
    decision = plan.get('decision')
    if decision == 'D036':
        if (plan.get('base_decision') != 'D028'
            or plan.get('experiment_id') != 'xlsr-d036-scale2'
            or plan.get('grid') != SCALE2_GRID
            or any(d.get(k) != v for k, v in SCALE2_DATA_SPEC.items())
            or t.get('steps_per_epoch') != 1986
            or t.get('max_optimizer_steps') != 19860
            or t.get('max_train_sample_exposures') != 317700
            or plan.get('outputs') != SCALE2_OUTPUTS):
            raise ValueError('Đặc tả scale2 không khớp D036/D028.')
        if seed not in (2026, 2027, 2028):
            raise ValueError('Seed huấn luyện phải thuộc 2026/2027/2028.')
        if run_id != 'X1-L8-lr3e-5':
            raise ValueError('Cấu hình scale2 (D036) chỉ cho phép run X1-L8-lr3e-5.')
        return dict(SCALE2_GRID[0])
    if (decision != 'D028' or plan['grid'] != GRID
        or d['train_samples'] != 15885
        or t.get('steps_per_epoch') != 993
        or t.get('max_optimizer_steps') != 9930
        or t.get('max_train_sample_exposures') != 158850):
        raise ValueError('Đặc tả không khớp D028.')
    if seed not in (2026, 2027, 2028):
        raise ValueError('Seed huấn luyện phải thuộc 2026/2027/2028.')
    base_run_id = VARIANTS[run_id]['base_run_id'] if run_id in VARIANTS else run_id
    for run in GRID:
        if run['run_id'] == base_run_id:
            return dict(run)
    raise ValueError('Run không thuộc lưới D028.')


def verify_pretrained(directory: Path, plan: dict[str, Any]) -> dict[str, str]:
    p = plan['pretrained']
    checks = {'pytorch_model.bin': p['weights_sha256'], 'config.json':p['config_sha256'],
              'preprocessor_config.json':p['preprocessor_config_sha256']}
    for name, expected in checks.items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Checksum pretrained không khớp: {name}')
    return checks


def open_artifact(path: Path, plan: dict[str, Any], *, training: bool, seed: int) -> PackedManifestDataset:
    d = plan['data']
    prefix = 'train' if training else 'development'
    # Kiểm metadata trước khi loader mở memmap; không băm waveform trong dry-run.
    if sha256(path / 'metadata.json') != d[prefix+'_metadata_sha256']:
        raise ValueError('Checksum metadata packed không khớp plan.')
    index_key = prefix + '_artifact_index_sha256'
    if index_key in d:
        metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        if sha256(path / metadata['index_file']) != d[index_key]:
            raise ValueError('Checksum index packed không khớp plan.')
    ds = PackedManifestDataset(path, training=training, amplitude_policy='peak', seed=seed)
    if (ds.split != d[prefix+'_split'] or ds.split not in ('closed_train','closed_dev')
        or len(ds) != d[prefix+'_samples'] or ds.target_sample_rate != 16000
        or ds.source_manifest_sha256 != d[prefix+'_manifest_sha256']
        or ds.waveforms_sha256 != d[prefix+'_artifact_waveforms_sha256']
        or [r.source_index for r in ds.records] != list(range(len(ds)))):
        ds.close()
        raise ValueError('Artifact không khớp split/số mẫu/checksum plan.')
    return ds


def build_model(backbone: torch.nn.Module, run: dict[str, Any], seed: int) -> XLSRBinaryClassifier:
    _set_seed(seed)
    return XLSRBinaryClassifier(backbone, num_unfrozen_layers=run['layers'], hidden_features=128, dropout=.2)


class SessionBudget:
    """Ước lượng bảo thủ T14; kiểm tra trước mỗi bước và toàn lượt dev."""
    def __init__(self, hours: float, *, evaluation_seconds: float, step_seconds: float,
                 deadline_unix: float | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (evaluation_seconds,step_seconds)):
            raise ValueError('Ước lượng thời gian bước và đánh giá phải hữu hạn và dương.')
        # Hạn Unix dùng wall clock; ngân sách cục bộ mặc định giữ monotonic như cũ.
        self.clock = clock if clock is not None else (time.time if deadline_unix is not None else time.monotonic)
        now = self.clock()
        if deadline_unix is not None:
            if not math.isfinite(deadline_unix) or deadline_unix <= 0 or deadline_unix <= now:
                raise ValueError('Hạn Unix phải hữu hạn, dương và còn ở tương lai.')
            self.deadline = deadline_unix
        else:
            if not math.isfinite(hours) or hours <= 0 or not math.isfinite(hours*3600):
                raise ValueError('Ngân sách giờ phải hữu hạn và dương.')
            self.deadline = now + hours*3600
        self.evaluation_seconds = evaluation_seconds
        self.step_seconds = step_seconds

    def can_start(self, phase: str, count: int) -> bool:
        needed = self.evaluation_seconds if phase == 'evaluation' else count*self.step_seconds
        return self.deadline - self.clock() >= needed + 120  # dành thời gian commit


def effective_modes(model: XLSRBinaryClassifier) -> dict[str, Any]:
    """D029: giữ config pretrained và kiểm mode thực sự, không gán config."""
    model.train()
    b = model.backbone
    result = {'configured_layerdrop': b.config.layerdrop,
              'configured_apply_spec_augment': b.config.apply_spec_augment,
              'effective_layerdrop': b.config.layerdrop if b.encoder.training else 0.0,
              'effective_apply_spec_augment': bool(b.config.apply_spec_augment and b.training),
              'backbone_training': b.training, 'encoder_training': b.encoder.training,
              'block_training': [layer.training for layer in b.encoder.layers]}
    if result['effective_layerdrop'] != 0 or result['effective_apply_spec_augment']:
        raise ValueError('Mode backbone không khớp D029.')
    return result


def artifact_identity(dataset: PackedManifestDataset) -> dict[str, Any]:
    metadata = json.loads((dataset.artifact_dir / 'metadata.json').read_text(encoding='utf-8'))
    return {'split': dataset.split, 'samples': len(dataset),
            'shards': len({r.shard for r in dataset.records}),
            'metadata_sha256': sha256(dataset.artifact_dir / 'metadata.json'),
            'index_sha256': sha256(dataset.artifact_dir / metadata['index_file']),
            'waveforms_sha256': dataset.waveforms_sha256,
            'manifest_sha256': dataset.source_manifest_sha256}


def smoke_indices(dataset: PackedManifestDataset, count: int) -> list[int]:
    """Chọn cân bằng từ metadata, chỉ train/dev, không đọc test."""
    if dataset.split not in ('closed_train', 'closed_dev'):
        raise ValueError('Không được đọc split test.')
    groups = [[i for i, record in enumerate(dataset.records) if record.binary_label == label]
              for label in (0, 1)]
    if any(len(group) < count // 2 for group in groups):
        raise ValueError('Smoke cần đủ hai nhãn từ train/dev.')
    return sorted(groups[0][:count // 2] + groups[1][:count // 2])


def progress(event: str, **fields: Any) -> None:
    """Log quan sát; không đọc/ghi RNG hoặc controller."""
    print(json.dumps({'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                      'event': event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def get_gpu_name(device: torch.device, gpu_name_fn: Callable[[int], str | None] | None = None) -> str | None:
    if gpu_name_fn is not None or device.type == 'cuda':
        raw = (gpu_name_fn or torch.cuda.get_device_name)(device.index or 0)
        return str(raw) if raw is not None else None
    return None


def get_nvidia_driver() -> str | None:
    proc_version = Path('/proc/driver/nvidia/version')
    if proc_version.is_file():
        try:
            text = proc_version.read_text(encoding='utf-8', errors='replace').strip()
            if text:
                return text.splitlines()[0].strip()
        except OSError:
            pass
    if torch.cuda.is_available() and hasattr(torch._C, '_cuda_getDriverVersion'):
        try:
            ver = int(torch._C._cuda_getDriverVersion())
            if ver > 0:
                return f'{ver // 1000}.{(ver % 1000) // 10}'
        except Exception:
            pass
    return None


def collect_hardware_info(
    device: torch.device,
    gpu_name_fn: Callable[[int], str | None] | None = None,
    device_count_fn: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Ghi nhận thông tin phần cứng/thư viện theo D033 mục 3, nằm ngoài identity."""
    import numpy as np
    import transformers
    gpu_name = get_gpu_name(device, gpu_name_fn)
    visible_gpus = int(device_count_fn()) if device_count_fn is not None else (
        int(torch.cuda.device_count()) if device.type == 'cuda' else 0
    )
    return {
        'device': str(device),
        'has_gpu': gpu_name is not None,
        'gpu_name': gpu_name,
        'gpu_status': gpu_name if gpu_name is not None else 'không có GPU',
        'visible_gpu_count': visible_gpus,
        'nvidia_driver': get_nvidia_driver() if gpu_name is not None else None,
        'cuda_version': torch.version.cuda,
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'numpy_version': np.__version__,
        'tf32': {
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
        },
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
    }


def verify_expected_and_resume_hardware(
    hardware: dict[str, Any],
    *,
    expected_gpu: str | None,
    resume: bool,
    run_root: Path | None,
) -> None:
    actual_gpu = hardware.get('gpu_name')
    if expected_gpu is not None:
        needle = expected_gpu.strip()
        if not needle or actual_gpu is None or needle not in actual_gpu:
            raise ValueError(
                f"Tên GPU thực tế ({actual_gpu or 'không có GPU'}) không chứa --expected-gpu ({expected_gpu!r})."
            )
    if resume and run_root is not None and (run_root / 'LATEST.json').is_file():
        pointer = json.loads((run_root / 'LATEST.json').read_text(encoding='utf-8'))
        latest = (pointer.get('latest') or {}).get('generation')
        if isinstance(latest, str):
            state_path = run_root / latest / 'state.json'
            if state_path.is_file():
                state_obj = json.loads(state_path.read_text(encoding='utf-8'))
                saved_hw = state_obj.get('hardware')
                if isinstance(saved_hw, dict) and 'gpu_name' in saved_hw:
                    saved_gpu = saved_hw.get('gpu_name')
                    if saved_gpu != actual_gpu:
                        raise ValueError(
                            f'Không được resume trên loại GPU khác với checkpoint đã lưu ({saved_gpu!r} != {actual_gpu!r}).'
                        )


class InvocationTiming:
    """Số đo của riêng lần gọi, không tham gia identity hoặc điều khiển dừng."""
    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self.clock = clock or time.perf_counter
        self.phases = {name: 0.0 for name in
                       ('rehash_train', 'rehash_dev', 'backbone_load', 'model_build')}
        self.rehash_performed = {'train': False, 'dev': False}
        self.steps: list[tuple[float, float]] = []
        self.evaluations: list[dict[str, Any]] = []
        self.pause_phase: str | None = None

    def phase(self, name: str, action: Callable[[], Any],
              device: torch.device | None = None) -> Any:
        progress(name + '_start')
        if device is not None:
            synchronize(device)
        started = self.clock()
        status = 'error'
        try:
            result = action()
            if device is not None:
                synchronize(device)
            status = 'ok'
            if name.startswith('rehash_'):
                self.rehash_performed[name.removeprefix('rehash_')] = True
            return result
        finally:
            elapsed = self.clock() - started
            self.phases[name] += elapsed
            progress(name + '_end', seconds=elapsed, status=status)

    def record_step(self, loader_seconds: float, model_seconds: float) -> None:
        self.steps.append((loader_seconds, model_seconds))

    def report(self, *, wall_seconds: float, deadline_unix: float | None,
               budget: SessionBudget, plan: dict[str, Any], layers: int) -> dict[str, Any]:
        loader = sum(x[0] for x in self.steps)
        model = sum(x[1] for x in self.steps)
        remaining = budget.deadline - budget.clock()
        return {
            'rehash_train_seconds': self.phases['rehash_train'],
            'rehash_dev_seconds': self.phases['rehash_dev'],
            'rehash_performed': self.rehash_performed.copy(),
            'backbone_load_seconds': self.phases['backbone_load'],
            'model_build_seconds': self.phases['model_build'],
            'train_steps_this_invocation': len(self.steps),
            'train_loader_seconds': loader, 'train_model_seconds': model,
            'train_seconds': loader + model,
            'mean_step_seconds': (loader + model) / len(self.steps) if self.steps else 0.0,
            'evaluations_this_invocation': self.evaluations.copy(),
            'wall_seconds': wall_seconds, 'deadline_unix': deadline_unix,
            'seconds_left_at_exit': max(0.0, remaining),
            'deadline_overrun_seconds': max(0.0, -remaining),
            'estimates': {
                'train_step_seconds': plan['sessions']['train_step_estimated_seconds'][str(layers)],
                'dev_full_seconds': plan['sessions']['dev_full_estimated_seconds'],
                'max_session_hours': plan['sessions']['max_session_hours']},
        }


class ProgressGenerationStore(GenerationStore):
    """Bọc commit đã có: chỉ log SAU khi commit thành công, giữ nguyên payload/prune."""
    def __init__(self, root: Path, *, timing: InvocationTiming,
                 hardware: dict[str, Any] | None = None, prune: bool = True) -> None:
        super().__init__(root, prune=prune)
        self.timing = timing
        self.hardware = copy.deepcopy(hardware) if hardware is not None else None

    def load(self) -> dict[str, Any]:
        if self.hardware is not None:
            verify_expected_and_resume_hardware(
                self.hardware,
                expected_gpu=None,
                resume=True,
                run_root=self.root,
            )
        payload = super().load()
        if self.hardware is not None:
            saved_hw = payload.get('controller', {}).get('hardware')
            if isinstance(saved_hw, dict) and 'gpu_name' in saved_hw:
                saved_gpu = saved_hw.get('gpu_name')
                current_gpu = self.hardware.get('gpu_name')
                if saved_gpu != current_gpu:
                    raise ValueError(
                        f'Không được resume trên loại GPU khác với checkpoint đã lưu ({saved_gpu!r} != {current_gpu!r}).'
                    )
        return payload

    def commit(self, payload: dict[str, Any], *, new_best: bool = False) -> dict[str, Any]:
        if self.hardware is not None:
            payload = copy.deepcopy(payload)
            payload['controller']['hardware'] = copy.deepcopy(self.hardware)
        result = super().commit(payload, new_best=new_best)
        state = result['controller']
        losses = state['losses'][-64:]
        recent = self.timing.steps[-64:]
        progress('checkpoint', global_step=state['global_step'],
                 epoch=result['sampler']['epoch'] + 1,
                 completed_epochs=state['completed_epochs'],
                 mean_loss_last_64=sum(losses) / len(losses) if losses else None,
                 mean_step_seconds=sum(a+b for a,b in recent) / len(recent) if recent else 0.0,
                 observed_steps_this_invocation=len(self.timing.steps),
                 timing_window_steps=len(recent), new_best=new_best)
        return result


def evaluate(model: XLSRBinaryClassifier, dev: PackedManifestDataset, device: torch.device,
             batch_size: int = 16, indices: list[int] | None = None,
             clock: Callable[[], float] | None = None) -> dict[str, Any]:
    timer = clock or time.perf_counter
    progress('evaluate_start', samples=len(dev) if indices is None else len(indices))
    synchronize(device)
    started = timer()
    loader_seconds = model_seconds = 0.0
    records: list[dict[str, Any]] = []
    loss_sum = 0.0
    model.eval()
    indices = list(range(len(dev))) if indices is None else indices
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            synchronize(device)
            load_started = timer()
            batch = default_collate([dev[i] for i in indices[start:start+batch_size]])
            synchronize(device)
            loader_seconds += timer() - load_started
            synchronize(device)
            model_started = timer()
            x = apply_xlsr_waveform_normalization(batch['waveform'].to(device), NORMALIZATION)
            targets = batch['label'].to(device)
            logits = model(x)
            loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='sum'))
            for file, speaker, utt, label, score in zip(batch['file'],batch['speaker_id'],batch['utt_type'],
                                                        targets.cpu().tolist(),logits.sigmoid().cpu().tolist(),strict=True):
                records.append({'file':file,'speaker_id':speaker,'utt_type':utt,'label':int(label),'score':score})
            synchronize(device)
            model_seconds += timer() - model_started
    result = asdict(evaluate_development_scores([r['label'] for r in records],[r['score'] for r in records],
                  loss=loss_sum/len(records),utt_types=[r['utt_type'] for r in records]))
    result['records'] = records
    synchronize(device)
    total_seconds = timer() - started
    result['timing'] = {'loader_seconds':loader_seconds,'model_seconds':model_seconds,
                        'total_seconds':total_seconds,'samples':len(records),
                        'throughput_samples_per_second':len(records)/max(total_seconds,1e-12)}
    progress('evaluate_end', eer=result['eer'], loss=result['loss'], **result['timing'])
    return result


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('Số bước mỗi phiên phải là số nguyên dương.') from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError('Số bước mỗi phiên phải lớn hơn 0.')
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/xlsr_training.json')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--training-seed',type=int,required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--expected-gpu',type=str)
    parser.add_argument('--train-artifact',type=Path)
    parser.add_argument('--dev-artifact',type=Path)
    parser.add_argument('--pretrained-dir',type=Path,required=True)
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--report-json',type=Path)
    parser.add_argument('--max-session-hours',type=float,default=11.0)
    parser.add_argument('--session-steps',type=positive_int,default=512)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--technical-smoke', action='store_true')
    parser.add_argument('--smoke-session-steps', type=int, choices=(1, 2), default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    run = validate_plan(plan,args.run_id,args.training_seed)
    if not re.fullmatch(r'cpu|cuda(?::\d+)?',args.device):
        raise ValueError('Thiết bị phải là cpu, cuda hoặc cuda:N.')
    device = torch.device(args.device)
    if args.technical_smoke and args.dry_run:
        raise ValueError('Không ghép dry-run và technical-smoke.')
    if args.technical_smoke and (args.device != 'cpu' or args.run_id != 'X1-L2-lr1e-5' or args.training_seed != 2026):
        raise ValueError('Technical smoke chỉ dùng CPU, X1-L2-lr1e-5, seed 2026.')
    if not args.dry_run and not args.technical_smoke and not plan['implementation']['ready_for_training']:
        raise ValueError('Chưa mở ready_for_training cho XLS-R.')
    budget = SessionBudget(args.max_session_hours,deadline_unix=args.deadline_unix,
                evaluation_seconds=plan['sessions']['dev_full_estimated_seconds'],
                step_seconds=plan['sessions']['train_step_estimated_seconds'][str(run['layers'])])
    started = time.perf_counter()
    timing = InvocationTiming()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hardware = collect_hardware_info(device)
    candidate_root = args.run_root or ROOT/plan['outputs']['root_template'].format(run_id=args.run_id,training_seed=args.training_seed)
    verify_expected_and_resume_hardware(
        hardware,
        expected_gpu=args.expected_gpu,
        resume=args.resume,
        run_root=candidate_root,
    )
    pretrained = verify_pretrained(args.pretrained_dir,plan)
    if sha256(ROOT/plan['model']['audio_config']) != plan['model']['audio_config_sha256']:
        raise ValueError('Checksum cấu hình audio không khớp.')
    train = open_artifact(args.train_artifact or ROOT/plan['data']['train_artifact'],plan,training=True,seed=args.training_seed)
    try:
        dev = open_artifact(args.dev_artifact or ROOT/plan['data']['development_artifact'],plan,training=False,seed=args.training_seed)
        try:
            condition = input_condition_identity(args.run_id)
            if condition is not None:
                train = InputConditionDataset(train, condition['condition'])
                dev = InputConditionDataset(dev, condition['condition'])
            selection = _sample_selection_sha256(selection='head',train_records=train.records,train_indices=list(range(len(train))),development_records=dev.records,development_indices=list(range(len(dev))))
            if selection != plan['data']['sample_selection_sha256']:
                raise ValueError('Exact selection không khớp plan.')
            backbone = timing.phase('backbone_load', lambda: load_xlsr_backbone(args.pretrained_dir))
            if (backbone.config.layerdrop != plan['pretrained']['layerdrop']
                    or backbone.config.apply_spec_augment != plan['pretrained']['configured_apply_spec_augment']):
                raise ValueError('Layerdrop cấu hình pretrain không khớp T14.')
            if args.dry_run:
                counts = []
                for configuration in ([run] if condition is not None else plan['grid']):
                    model = timing.phase('model_build', lambda: build_model(backbone,configuration,args.training_seed))
                    counts.append({**configuration,'modes':effective_modes(model),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
                report = {'status':'dry_run','use_test':False,'training_seed':args.training_seed,'data_seed':2026,
                    'device':str(device),'pretrained_checksums':pretrained,'runs':counts,
                    'train_samples':len(train),'development_samples':len(dev),
                    'train_shards':len({r.shard for r in train.records}),'development_shards':len({r.shard for r in dev.records}),
                    'waveforms_rehashed':False,'plan_sha256':sha256(args.plan),
                    'checksums_match':True,'artifacts':{'train':artifact_identity(train),'dev':artifact_identity(dev)},
                    'ready_for_training':False}
                if condition is not None:
                    report.update(run_id=args.run_id, input_condition=condition)
            else:
                # Một lần băm toàn waveform khi bắt đầu, resume kiểm metadata/index/kích thước.
                if not args.resume and not args.technical_smoke:
                    timing.phase('rehash_train', lambda: verify_packed_artifact(train.artifact_dir))
                    timing.phase('rehash_dev', lambda: verify_packed_artifact(dev.artifact_dir))
                model = timing.phase('model_build', lambda: build_model(backbone,run,args.training_seed).to(device),device)
                optimizer = torch.optim.Adam(model.optimizer_groups(encoder_lr=run['encoder_lr'] or 1e-5,classifier_lr=.001))
                train_indices = smoke_indices(train, 4) if args.technical_smoke else list(range(len(train)))
                dev_indices = smoke_indices(dev, 8) if args.technical_smoke else list(range(len(dev)))
                batch_size = 2 if args.technical_smoke else 16
                sampler = ResumableBatchSampler(dataset_size=len(train_indices),batch_size=batch_size,seed=args.training_seed)
                files = [{'path':p.relative_to(ROOT).as_posix(),'sha256':sha256(p)} for p in source_files()]
                identity = {'plan':plan,'run':run,'run_id':args.run_id,'training_seed':args.training_seed,'data_seed':2026,
                            'device':str(device),'backend':'packed','pretrained':pretrained,'source_files':files,
                            'modes':effective_modes(model),'trainable_only':True,
                            'technical_smoke':args.technical_smoke,'batch_size':batch_size,
                            'train_indices':train_indices,'development_indices':dev_indices,
                            'artifacts':{'train':artifact_identity(train),'dev':artifact_identity(dev)},
                            'source_tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
                if condition is not None:
                    identity['input_condition'] = condition
                root = args.run_root or ROOT/plan['outputs']['root_template'].format(run_id=args.run_id,training_seed=args.training_seed)
                if args.technical_smoke:
                    allowed = (ROOT/'.cache/xlsr-technical-smoke').resolve()
                    if args.run_root is None:
                        if args.resume:
                            raise ValueError('Resume smoke cần --run-root đã tạo trước.')
                        allowed.mkdir(parents=True,exist_ok=True)
                        root = Path(tempfile.mkdtemp(prefix='run-',dir=allowed))
                    else:
                        root = args.run_root.resolve()
                    if not root.is_relative_to(allowed) or root == allowed:
                        raise ValueError('Run root smoke phải nằm riêng dưới .cache/xlsr-technical-smoke/.')
                store = ProgressGenerationStore(root,timing=timing,hardware=hardware,prune=True)
                def step(indices: list[int], epoch: int) -> float:
                    synchronize(device)
                    load_started = timing.clock()
                    train.set_epoch(epoch)
                    batch = default_collate([train[train_indices[i]] for i in indices])
                    synchronize(device)
                    loader_seconds = timing.clock() - load_started
                    synchronize(device)
                    model_started = timing.clock()
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    x = apply_xlsr_waveform_normalization(batch['waveform'].to(device),NORMALIZATION)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x),batch['label'].to(device))
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError('Gradient XLS-R không hữu hạn.')
                    optimizer.step()
                    value = float(loss.detach())
                    synchronize(device)
                    timing.record_step(loader_seconds, timing.clock() - model_started)
                    return value
                def evaluate_once() -> dict[str, Any]:
                    development = evaluate(model,dev,device,batch_size,dev_indices)
                    timing.evaluations.append(dict(development['timing']))
                    return development

                def can_start(phase: str, count: int) -> bool:
                    allowed = budget.can_start(phase,count)
                    if not allowed:
                        timing.pause_phase = phase
                    return allowed

                progress('train_start',run_id=args.run_id,training_seed=args.training_seed,
                         resume=args.resume,run_root=str(root))
                report = run_training(model=model,optimizer=optimizer,sampler=sampler,identity=identity,store=store,
                    train_step=step,evaluate=evaluate_once,max_epochs=10,patience=2,
                    development_samples=len(dev_indices),session_steps=args.smoke_session_steps if args.technical_smoke else args.session_steps,checkpoint_interval=64,resume=args.resume,
                    trainable_only=True,can_start=None if args.technical_smoke else can_start,retain_evaluations=True)
                progress('stop',reason=report['status'],
                         pause_cause=('budget_before_'+timing.pause_phase if timing.pause_phase else 'session_steps') if report['status']=='paused' else None,
                         global_step=report['controller']['global_step'])
                report['controller'].setdefault('hardware', copy.deepcopy(hardware))
                report.update(technical_smoke=args.technical_smoke,run_root=str(root),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    evaluated_samples=len(dev_indices),checksums_match=True,
                    pretrained_checksums=pretrained,identity=identity,checkpoints=json.loads((root/'LATEST.json').read_text()),device=str(device),use_test=False,
                    train_samples=len(train),development_samples=len(dev),train_shards=len({r.shard for r in train.records}),
                    development_shards=len({r.shard for r in dev.records}))
            report['hardware'] = copy.deepcopy(hardware)
            report['session_controls'] = {
                'session_steps': args.smoke_session_steps if args.technical_smoke else args.session_steps,
                'deadline_unix': args.deadline_unix, 'max_session_hours': args.max_session_hours,
                'commit_safety_seconds': 120}
            report['wall_seconds'] = time.perf_counter() - started
            report['timing'] = timing.report(wall_seconds=report['wall_seconds'],
                deadline_unix=args.deadline_unix,budget=budget,plan=plan,layers=run['layers'])
            report['environment'] = {'python':platform.python_version(),'torch':torch.__version__}
            try:
                import psutil
                memory = psutil.Process().memory_info()
                report['peak_rss_bytes'] = getattr(memory,'peak_wset',memory.rss)
            except ImportError:
                raise RuntimeError('Thiếu psutil để ghi RSS đỉnh.')
            output = args.report_json or (root/'smoke-report.json' if args.technical_smoke else None) or (ROOT/'reports/xlsr_training_dry_run.json' if args.dry_run else ROOT/plan['outputs']['report_json_template'].format(run_id=args.run_id,training_seed=args.training_seed))
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
            print(json.dumps(report if args.dry_run else {'status':report['status'],'report':str(output)},ensure_ascii=False))
            return 0
        finally:
            dev.close()
    finally:
        train.close()

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,RuntimeError,KeyError) as error:
        print(f'LỖI: {error}',file=sys.stderr)
        raise SystemExit(1) from error
