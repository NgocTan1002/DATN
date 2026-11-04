#!/usr/bin/env python3
"""T14: đo chi phí XLS-R trên packed train/dev; không chọn cấu hình khoa học."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.packed import PackedManifestDataset, verify_packed_artifact
from src.data.xlsr import apply_xlsr_waveform_normalization
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone

SEED = 2026
NORMALIZATION = "processor_zero_mean_unit_variance"


@dataclass(frozen=True)
class Configuration:
    unfrozen_layers: int
    batch_size: int

    @property
    def mode(self) -> str:
        return "X0" if self.unfrozen_layers == 0 else "X1"


@dataclass(frozen=True)
class Measurement:
    warmup_steps: int = 5
    train_steps: int = 30
    dev_samples: int = 512
    eval_batch_size: int = 8

    def __post_init__(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("Số bước và số mẫu đo phải dương.")


def configuration_grid() -> list[Configuration]:
    return [Configuration(n, b) for n in (0, 1, 2, 4, 8, 12) for b in (8, 16)]


def summarize_seconds(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    return {"mean": mean, "median": statistics.median(values),
            "cv": statistics.pstdev(values) / mean if mean else 0.0}


def project_cost(batch_size: int, train_seconds: float, dev_seconds: float) -> dict[str, float]:
    if batch_size <= 0 or not all(math.isfinite(v) and v >= 0 for v in (train_seconds, dev_seconds)):
        raise ValueError("Đầu vào phép chiếu phải hữu hạn và không âm, batch phải dương.")
    return {"train_hours_per_epoch": math.ceil(15885 / batch_size) * train_seconds / 3600,
            "dev_full_hours_per_pass": 38777 * dev_seconds / 3600}


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    if not 0 < count <= size:
        raise ValueError("Không đủ mẫu dev để chọn tập rải đều.")
    return [0] if count == 1 else [i * (size - 1) // (count - 1) for i in range(count)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_hashes(model: nn.Module) -> dict[str, str]:
    # Chỉ băm bản sao CPU theo từng tensor, không clone cả backbone trên GPU.
    return {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def verify_updates(model: XLSRBinaryClassifier, before: dict[str, str]) -> dict[str, Any]:
    after = parameter_hashes(model)
    changed = {name for name in before if before[name] != after[name]}
    frozen_changed = [name for name, p in model.named_parameters() if not p.requires_grad and name in changed]
    head_changed = any(name.startswith("classifier.") for name in changed)
    first = len(model.backbone.encoder.layers) - model.num_unfrozen_layers
    opened = {str(i): any(name.startswith(f"backbone.encoder.layers.{i}.") for name in changed)
              for i in range(first, len(model.backbone.encoder.layers))}
    if frozen_changed or not head_changed or not all(opened.values()):
        raise RuntimeError("Kiểm tra tham số đóng băng/mở không đạt.")
    return {"frozen_parameters_unchanged": True, "classifier_updated": head_changed,
            "opened_layers_updated": opened, "changed_parameter_tensors": len(changed)}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory(device: torch.device) -> dict[str, int]:
    return {"allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}


def measure_configuration(
    configuration: Configuration, *, factory: Callable[[int], XLSRBinaryClassifier],
    train: PackedManifestDataset, dev: PackedManifestDataset, device: torch.device,
    measurement: Measurement = Measurement(),
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED)
        torch.cuda.reset_peak_memory_stats(device)
    model = factory(configuration.unfrozen_layers).to(device=device, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.Adam(model.optimizer_groups(encoder_lr=1e-5, classifier_lr=1e-3))
    before = parameter_hashes(model)
    count = (measurement.warmup_steps + measurement.train_steps) * configuration.batch_size
    if len(train) < count:
        raise ValueError("Không đủ mẫu train cho lượt đo không lặp mẫu.")
    train.set_epoch(0)
    train_indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED))[:count].tolist()
    dev_indices = evenly_spaced_indices(len(dev), measurement.dev_samples)
    loader = DataLoader(Subset(train, train_indices), batch_size=configuration.batch_size,
                        num_workers=0, generator=torch.Generator().manual_seed(SEED))
    iterator = iter(loader)
    loader_times: list[float] = []
    compute_times: list[float] = []
    losses: list[float] = []
    for step in range(measurement.warmup_steps + measurement.train_steps):
        start = time.perf_counter()
        batch = next(iterator)
        waveform = apply_xlsr_waveform_normalization(batch["waveform"], NORMALIZATION)
        loaded = time.perf_counter() - start
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(waveform.to(device)), batch["label"].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Loss train không hữu hạn.")
        loss.backward()
        optimizer.step()
        synchronize(device)
        computed = time.perf_counter() - start
        losses.append(float(loss.detach().cpu()))
        if step >= measurement.warmup_steps:
            loader_times.append(loaded)
            compute_times.append(computed)
    model.eval()
    dev_loader_seconds = 0.0
    dev_compute_seconds = 0.0
    dev_losses: list[float] = []
    iterator = iter(DataLoader(Subset(dev, dev_indices), batch_size=measurement.eval_batch_size,
                              num_workers=0, generator=torch.Generator().manual_seed(SEED)))
    with torch.no_grad():
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            waveform = apply_xlsr_waveform_normalization(batch["waveform"], NORMALIZATION)
            dev_loader_seconds += time.perf_counter() - start
            synchronize(device)
            start = time.perf_counter()
            loss = nn.functional.binary_cross_entropy_with_logits(model(waveform.to(device)), batch["label"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Loss dev không hữu hạn.")
            synchronize(device)
            dev_compute_seconds += time.perf_counter() - start
            dev_losses.append(float(loss.cpu()))
    memory = peak_memory(device)
    checks = verify_updates(model, before)
    total = [a + b for a, b in zip(loader_times, compute_times, strict=True)]
    dev_per_sample = (dev_loader_seconds + dev_compute_seconds) / len(dev_indices)
    return {"status": "ok", "mode": configuration.mode, **asdict(configuration),
            "measurement": asdict(measurement), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_seconds_per_step": summarize_seconds(total),
            "loader_seconds_per_step": summarize_seconds(loader_times),
            "compute_seconds_per_step": summarize_seconds(compute_times),
            "train_step_seconds": total, "loader_step_seconds": loader_times, "compute_step_seconds": compute_times,
            "dev_seconds_per_sample": dev_per_sample, "dev_loader_seconds": dev_loader_seconds,
            "dev_compute_seconds": dev_compute_seconds, "peak_vram": memory,
            "checks": {"all_train_and_dev_losses_finite": True, **checks},
            "technical_train_losses": losses, "technical_dev_batch_losses": dev_losses,
            "selected_train_samples": len(train_indices), "selected_train_shards": len({train.records[i].shard for i in train_indices}),
            "selected_dev_samples": len(dev_indices), "selected_dev_shards": len({dev.records[i].shard for i in dev_indices}),
            "train_indices_sha256": hashlib.sha256(json.dumps(train_indices).encode()).hexdigest(),
            "dev_indices": dev_indices, "projection": project_cost(configuration.batch_size, statistics.mean(total), dev_per_sample)}


def run_grid(
    configurations: list[Configuration], measure: Callable[[Configuration], dict[str, Any]], *,
    device: torch.device, on_result: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    results: list[dict[str, Any]] = []
    blocked = False
    for configuration in configurations:
        try:
            result = measure(configuration)
        except torch.cuda.OutOfMemoryError as error:
            result = {"mode": configuration.mode, **asdict(configuration), "status": "oom",
                      "message": str(error), "peak_vram": peak_memory(device)}
        # Traceback và các tensor trong lượt lỗi được giải phóng trước empty_cache.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results.append(result)
        if on_result:
            on_result(results)
        if result["status"] == "oom" and configuration == Configuration(0, 8):
            blocked = True
            break
    return results, blocked


def write_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# XLS-R T14 — đo chi phí kỹ thuật", "", f"- Status: {report['status']}",
             "- Chỉ đo chi phí; head/LR là giá trị kỹ thuật giữ chỗ. Không chọn bậc, epoch hoặc LR, chưa có D028.",
             f"- Môi trường: {json.dumps(report['environment'], ensure_ascii=False)}",
             f"- Dữ liệu: {json.dumps(report.get('data', {}), ensure_ascii=False)}", "",
             "| Mode | Tầng mở | Batch | Status | s/bước mean | median | CV | loader mean | compute mean | s/mẫu dev | VRAM allocated/reserved byte | Trainable | giờ/epoch train | giờ/dev-full |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report['results']:
        prefix = f"| {r['mode']} | {r['unfrozen_layers']} | {r['batch_size']} | {r['status']}"
        if r['status'] != 'ok':
            lines.append(prefix + " | — | — | — | — | — | — | — | — | — | — |")
            continue
        t, m, p = r['train_seconds_per_step'], r['peak_vram'], r['projection']
        lines.append(prefix + f" | {t['mean']:.8g} | {t['median']:.8g} | {t['cv']:.8g} | {r['loader_seconds_per_step']['mean']:.8g} | {r['compute_seconds_per_step']['mean']:.8g} | {r['dev_seconds_per_sample']:.8g} | {m['allocated_bytes']}/{m['reserved_bytes']} | {r['trainable_parameters']} | {p['train_hours_per_epoch']:.8g} | {p['dev_full_hours_per_pass']:.8g} |")
    if 'error' in report:
        lines += ['', f"- Lỗi/cổng dừng: {report['error']}"]
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def check_weights(directory: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    source = candidate['source']
    if not source['weights_downloaded'] or not source['weights_sha256']:
        raise ValueError("Chưa qua G1: trọng số ứng viên chưa được tải/băm và khóa checksum.")
    for filename, key in (('config.json', 'config_sha256'), ('preprocessor_config.json', 'preprocessor_config_sha256')):
        if sha256(directory / filename) != candidate['published_config'][key]:
            raise ValueError(f"Checksum {filename} lệch; cần dừng và hỏi người dùng.")
    files = sorted([*directory.glob('*.safetensors'), *directory.glob('pytorch_model*.bin')])
    if not files:
        raise ValueError("Không có tệp trọng số trong thư mục offline.")
    actual = {p.name: sha256(p) for p in files}
    expected = source['weights_sha256']
    valid = actual == expected if isinstance(expected, dict) else len(actual) == 1 and next(iter(actual.values())) == expected
    if not valid:
        raise ValueError("Checksum trọng số lệch; cần dừng và hỏi người dùng.")
    return {"model_id": source['model_id'], "revision": source['revision'], "weights_sha256": actual,
            "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": actual[p.name]} for p in files]}


def open_datasets(train_path: Path, dev_path: Path, *, locked: bool) -> tuple[PackedManifestDataset, PackedManifestDataset]:
    audio = json.loads((ROOT/'configs/audio.json').read_text(encoding='utf-8'))
    policy = audio['waveform']['amplitude_policy']
    opened: list[PackedManifestDataset] = []
    try:
        plan = json.loads((ROOT/'configs/b0_v2_training.json').read_text(encoding='utf-8'))['data'] if locked else {}
        for path, split, name in ((train_path, 'closed_train', 'train'), (dev_path, 'closed_dev', 'development')):
            dataset = PackedManifestDataset(path, training=name == 'train', seed=SEED, amplitude_policy='peak',
                peak_target=policy['peak_target'], rms_target_dbfs=policy['rms_target_dbfs'], minimum_input_rms_dbfs=policy['minimum_input_rms_dbfs'])
            opened.append(dataset)
            if dataset.split != split or dataset.target_sample_rate != 16000:
                raise ValueError("Artifact phải là train/dev mono 16 kHz đúng split.")
            if locked:
                if (len(dataset) != plan[f'{name}_samples'] or dataset.source_manifest_sha256 != plan[f'{name}_manifest_sha256']
                        or dataset.waveforms_sha256 != plan[f'{name}_artifact_waveforms_sha256']):
                    raise ValueError("Artifact không khớp số mẫu/manifest/checksum đã khóa.")
                verify_packed_artifact(path)
        return opened[0], opened[1]
    except Exception:
        for dataset in opened:
            dataset.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT/'data/raw/xlsr/wav2vec2-xls-r-300m')
    parser.add_argument('--candidate', type=Path, default=ROOT/'configs/xlsr_pretrained_candidate.json')
    parser.add_argument('--train-artifact', type=Path, default=ROOT/'data/processed/packed/train_development_20k_v2')
    parser.add_argument('--dev-artifact', type=Path, default=ROOT/'data/processed/packed/closed_dev_full_v1')
    parser.add_argument('--report-json', type=Path)
    parser.add_argument('--report-md', type=Path)
    parser.add_argument('--technical-smoke', action='store_true', help='CPU, backbone thu nhỏ ngẫu nhiên và chỉ artifact tổng hợp.')
    args = parser.parse_args(argv)
    if args.technical_smoke and (args.report_json is None or args.report_md is None):
        parser.error('Smoke cần đường dẫn report riêng tường minh để không ghi đè report CUDA.')
    json_path = args.report_json or ROOT/'reports/xlsr_t14_cuda_benchmark.json'
    md_path = args.report_md or ROOT/'reports/xlsr_t14_cuda_benchmark.md'
    report: dict[str, Any] = {"schema_version": 1, "status": "preflight", "technical_smoke": args.technical_smoke,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "transformers": None}, "results": [],
        "protocol": {"seed": SEED, "normalization": NORMALIZATION, "amplitude_policy": "peak", "samples": 64000,
                     "sample_rate": 16000, "precision": "float32", "amp": False, "num_workers": 0,
                     "encoder_lr_placeholder": 1e-5, "classifier_lr_placeholder": 1e-3,
                     "classifier_hidden_features_technical": 128, "classifier_dropout_technical": 0.2,
                     "compute_timing_includes": "H2D, forward, kiểm loss hữu hạn, backward, Adam step; đồng bộ CUDA",
                     "loader_timing_includes": "packed I/O, peak, crop, collate và processor normalization trên CPU",
                     "warmup_excluded_from_timing_but_included_in_peak_vram": True}}
    train = dev = None
    try:
        try:
            import transformers
        except ImportError as error:
            print('Transformers hiện có: không import được', flush=True)
            raise RuntimeError('Không import được transformers; dừng và hỏi người dùng, không cài thêm.') from error
        report['environment']['transformers'] = transformers.__version__
        print(f'Transformers hiện có: {transformers.__version__}', flush=True)
        if transformers.__version__ != '4.57.6':
            raise RuntimeError("Transformers thiếu hoặc khác 4.57.6; dừng và hỏi người dùng, không cài thêm.")
        torch.set_num_threads(1)
        if args.technical_smoke:
            from scripts.smoke_test_xlsr import build_tiny_classifier
            config = json.loads((ROOT/'configs/xlsr_smoke.json').read_text(encoding='utf-8'))
            factory = lambda n: build_tiny_classifier(config, n)
            configurations = [Configuration(0, 2), Configuration(1, 2)]
            measurement = Measurement(1, 2, 4, 2)
            device = torch.device('cpu')
            report['protocol']['classifier_hidden_features_technical'] = config['classifier']['hidden_features']
            report['protocol']['classifier_dropout_technical'] = config['classifier']['dropout']
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("T14 thật cần một GPU CUDA; CPU chỉ dùng technical-smoke.")
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            device = torch.device('cuda:0')
            candidate = json.loads(args.candidate.read_text(encoding='utf-8'))
            if (candidate['source']['model_id'] != 'facebook/wav2vec2-xls-r-300m'
                    or candidate['source']['revision'] != '1a640f32ac3e39899438a2931f9924c02f080a54'):
                raise ValueError("Model/revision XLS-R khác mốc đã ghim.")
            report['pretrained'] = check_weights(args.weights, candidate)
            def factory(n: int) -> XLSRBinaryClassifier:
                backbone = load_xlsr_backbone(args.weights, local_files_only=True)
                if len(backbone.encoder.layers) != 24 or backbone.config.hidden_size != 1024:
                    raise ValueError("Backbone pretrained không đúng 24 tầng/hidden 1024.")
                return XLSRBinaryClassifier(backbone, num_unfrozen_layers=n)
            configurations, measurement = configuration_grid(), Measurement()
        report['environment'].update(device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                                     cuda_device_count_used=1 if device.type == 'cuda' else 0)
        train, dev = open_datasets(args.train_artifact, args.dev_artifact, locked=not args.technical_smoke)
        if args.technical_smoke and (train.source_snapshot != 'fixture' or dev.source_snapshot != 'fixture'):
            raise ValueError("Technical-smoke chỉ nhận artifact tổng hợp fixture, không dùng VSASV thật.")
        report['data'] = {name: {"samples": len(ds), "shards": len({r.shard for r in ds.records}),
            "manifest_sha256": ds.source_manifest_sha256, "waveforms_sha256": ds.waveforms_sha256}
            for name, ds in (('train', train), ('dev_full', dev))}
        report['status'] = 'running'
        def save(results: list[dict[str, Any]]) -> None:
            report['results'] = results
            write_reports(report, json_path, md_path)
        results, blocked = run_grid(configurations, lambda c: measure_configuration(c, factory=factory,
            train=train, dev=dev, device=device, measurement=measurement), device=device, on_result=save)
        report.update(results=results, status='blocked_x0_batch8_oom' if blocked else 'complete')
        if blocked:
            report['error'] = 'X0 batch 8 OOM: dừng, cần hỏi người dùng trước khi chạy tiếp.'
        write_reports(report, json_path, md_path)
        return 2 if blocked else 0
    except (OSError, ValueError, RuntimeError) as error:
        report.update(status='failed', error=str(error))
        write_reports(report, json_path, md_path)
        print(f'LỖI: {error}', file=sys.stderr)
        return 1
    finally:
        for dataset in (train, dev):
            if dataset is not None:
                dataset.close()


if __name__ == '__main__':
    raise SystemExit(main())
