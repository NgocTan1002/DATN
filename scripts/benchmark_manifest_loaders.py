#!/usr/bin/env python3
"""So sánh loader split cũ và loader development manifest trên cùng file."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_data_pipeline import (  # noqa: E402
    summarize_seconds,
    waveform_sha256,
)
from src.data import VSASVManifestDataset, VSASVParquetDataset  # noqa: E402
from src.data.vsasv import VSASVRecord  # noqa: E402


DEFAULT_SPLIT = PROJECT_ROOT / "data" / "splits" / "closed_train.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "development_20k_v2.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "manifest_loader_benchmark.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "manifest_loader_benchmark.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-improvement-percent", type=float, default=15.0)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args()


def stable_file_key(file_name: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}|{file_name}".encode("utf-8")).hexdigest()
    return digest, file_name


def select_benchmark_files(
    rows: Iterable[Mapping[str, str]],
    *,
    samples: int,
    seed: int,
    split: str = "closed_train",
) -> list[str]:
    """Chọn danh sách file xác định từ một partition của manifest."""

    if samples <= 0:
        raise ValueError("samples phải lớn hơn 0.")
    seen: set[str] = set()
    candidates: list[str] = []
    for row in rows:
        file_name = str(row.get("file", "")).strip()
        if not file_name:
            raise ValueError("Manifest benchmark có file rỗng.")
        if file_name in seen:
            raise ValueError(f"Manifest benchmark có file bị lặp: {file_name}")
        seen.add(file_name)
        if str(row.get("split", "")).strip() == split:
            candidates.append(file_name)
    if len(candidates) < samples:
        raise ValueError(
            f"Không đủ file {split} để benchmark: cần {samples}, có {len(candidates)}."
        )
    return sorted(candidates, key=lambda name: stable_file_key(name, seed))[:samples]


def _record_map(records: Sequence[VSASVRecord], source_name: str) -> dict[str, VSASVRecord]:
    result: dict[str, VSASVRecord] = {}
    for record in records:
        if record.file in result:
            raise ValueError(f"{source_name} có file bị lặp: {record.file}")
        result[record.file] = record
    return result


def align_common_records(
    legacy_records: Sequence[VSASVRecord],
    manifest_records: Sequence[VSASVRecord],
    file_names: Sequence[str],
) -> tuple[list[VSASVRecord], list[VSASVRecord]]:
    """Align records so both datasets use the same index-based crop seed."""

    if len(set(file_names)) != len(file_names):
        raise ValueError("Danh sách file benchmark bị lặp.")
    legacy_by_file = _record_map(legacy_records, "Loader split cũ")
    manifest_by_file = _record_map(manifest_records, "Loader manifest")
    aligned_legacy: list[VSASVRecord] = []
    aligned_manifest: list[VSASVRecord] = []
    for file_name in file_names:
        legacy = legacy_by_file.get(file_name)
        manifest = manifest_by_file.get(file_name)
        if legacy is None or manifest is None:
            missing_sources = []
            if legacy is None:
                missing_sources.append("split cũ")
            if manifest is None:
                missing_sources.append("manifest")
            raise ValueError(
                f"Thiếu file {file_name} trong {' và '.join(missing_sources)}."
            )
        legacy_metadata = (
            legacy.speaker_id,
            legacy.utt_type,
            legacy.shard.name,
        )
        manifest_metadata = (
            manifest.speaker_id,
            manifest.utt_type,
            manifest.shard.name,
        )
        if legacy_metadata != manifest_metadata:
            raise ValueError(
                "Metadata không tương đương cho "
                f"{file_name}: split={legacy_metadata}, manifest={manifest_metadata}."
            )
        aligned_legacy.append(legacy)
        aligned_manifest.append(manifest)
    return aligned_legacy, aligned_manifest


def _scalar_label(sample: Mapping[str, Any]) -> float:
    value = sample["label"]
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def validate_sample_equivalence(
    legacy_sample: Mapping[str, Any],
    manifest_sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact post-processing equivalence and return a reference row."""

    metadata_fields = ("file", "speaker_id", "utt_type")
    for field in metadata_fields:
        if legacy_sample[field] != manifest_sample[field]:
            raise RuntimeError(
                f"Metadata mẫu không tương đương ở {field}: "
                f"split={legacy_sample[field]!r}, manifest={manifest_sample[field]!r}."
            )
    legacy_label = _scalar_label(legacy_sample)
    manifest_label = _scalar_label(manifest_sample)
    if legacy_label != manifest_label:
        raise RuntimeError(
            f"Nhãn không tương đương cho {legacy_sample['file']}: "
            f"split={legacy_label}, manifest={manifest_label}."
        )
    legacy_waveform = legacy_sample["waveform"]
    manifest_waveform = manifest_sample["waveform"]
    if not isinstance(legacy_waveform, torch.Tensor) or not isinstance(
        manifest_waveform, torch.Tensor
    ):
        raise TypeError("Waveform benchmark phải là torch.Tensor.")
    if legacy_waveform.shape != manifest_waveform.shape or not torch.equal(
        legacy_waveform, manifest_waveform
    ):
        raise RuntimeError(
            f"Waveform không tương đương cho {legacy_sample['file']}."
        )
    return {
        "file": str(legacy_sample["file"]),
        "speaker_id": str(legacy_sample["speaker_id"]),
        "utt_type": str(legacy_sample["utt_type"]),
        "label": legacy_label,
        "waveform_sha256": waveform_sha256(legacy_waveform),
    }


def calculate_improvement_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0.0 or candidate <= 0.0:
        raise ValueError("Throughput baseline và candidate phải lớn hơn 0.")
    return (candidate / baseline - 1.0) * 100.0


def calculate_time_reduction_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0.0 or candidate <= 0.0:
        raise ValueError("Thời gian baseline và candidate phải lớn hơn 0.")
    return (1.0 - candidate / baseline) * 100.0


def summarize_runs(
    runs: Sequence[Mapping[str, float]],
    *,
    samples_per_run: int,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("Không thể tổng hợp danh sách lượt chạy rỗng.")
    if samples_per_run <= 0:
        raise ValueError("samples_per_run phải lớn hơn 0.")
    seconds = [float(run["seconds"]) for run in runs]
    if any(value <= 0.0 for value in seconds):
        raise ValueError("Thời gian mỗi lượt chạy phải lớn hơn 0.")
    startup = [float(run["startup_seconds"]) for run in runs]
    first_batch = [float(run["first_batch_seconds"]) for run in runs]
    return {
        "samples_per_run": samples_per_run,
        "repeats": len(runs),
        "throughput_samples_per_second": (
            samples_per_run * len(runs) / sum(seconds)
        ),
        "run_seconds": summarize_seconds(seconds),
        "startup_seconds": summarize_seconds(startup),
        "first_batch_seconds": summarize_seconds(first_batch),
        "peak_rss_mib": max(float(run["peak_rss_mib"]) for run in runs),
        "runs": [dict(run) for run in runs],
    }


def process_tree_rss_mib(process: psutil.Process) -> float:
    total = 0
    candidates = [process]
    try:
        candidates.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for candidate in candidates:
        try:
            total += candidate.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024**2)


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Manifest rỗng: {path}")
    return rows


def validate_protocol(args: argparse.Namespace) -> list[int]:
    if args.samples <= 0:
        raise ValueError("samples phải lớn hơn 0.")
    if args.repeats < 3:
        raise ValueError("repeats phải ít nhất bằng 3.")
    if args.batch_size <= 0:
        raise ValueError("batch-size phải lớn hơn 0.")
    workers = list(dict.fromkeys(int(value) for value in args.workers))
    if not workers or any(value < 0 for value in workers):
        raise ValueError("workers phải gồm các số nguyên không âm.")
    if args.minimum_improvement_percent <= 0.0:
        raise ValueError("minimum-improvement-percent phải lớn hơn 0.")
    return workers


def build_dataset(
    loader_name: str,
    *,
    split_path: Path,
    manifest_path: Path,
    parquet_dir: Path,
    seed: int,
) -> VSASVParquetDataset:
    common = {
        "training": True,
        "seed": seed,
        "amplitude_policy": "none",
    }
    if loader_name == "legacy":
        return VSASVParquetDataset(split_path, parquet_dir, **common)
    if loader_name == "manifest":
        return VSASVManifestDataset(
            manifest_path,
            parquet_dir,
            split="closed_train",
            **common,
        )
    raise ValueError(f"Loader benchmark không hợp lệ: {loader_name}")


def benchmark_initialization(
    *,
    repeats: int,
    split_path: Path,
    manifest_path: Path,
    parquet_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    timings: dict[str, list[float]] = {"legacy": [], "manifest": []}
    orders: list[list[str]] = []
    counts: dict[str, dict[str, float | int]] = {}
    for repeat in range(repeats):
        order = ["legacy", "manifest"]
        if repeat % 2:
            order.reverse()
        orders.append(order.copy())
        for loader_name in order:
            gc.collect()
            started = time.perf_counter()
            dataset = build_dataset(
                loader_name,
                split_path=split_path,
                manifest_path=manifest_path,
                parquet_dir=parquet_dir,
                seed=seed,
            )
            elapsed = time.perf_counter() - started
            timings[loader_name].append(elapsed)
            counts[loader_name] = {
                "available_records": len(dataset),
                "declared_split_rows": dataset.total_split_rows,
                "local_coverage": dataset.local_coverage,
            }
            dataset.close()
    summaries = {
        loader_name: {
            **counts[loader_name],
            "timing_seconds": summarize_seconds(values),
        }
        for loader_name, values in timings.items()
    }
    summaries["manifest"]["time_reduction_vs_legacy_percent"] = (
        calculate_time_reduction_percent(
            summaries["legacy"]["timing_seconds"]["median_ms"],
            summaries["manifest"]["timing_seconds"]["median_ms"],
        )
    )
    return summaries, orders


def verify_batch(
    batch: Mapping[str, Any],
    reference_by_file: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    files = [str(value) for value in batch["file"]]
    for position, file_name in enumerate(files):
        expected = reference_by_file.get(file_name)
        if expected is None:
            raise RuntimeError(f"Batch đọc file ngoài danh sách benchmark: {file_name}")
        label = float(batch["label"][position].item())
        speaker_id = str(batch["speaker_id"][position])
        utt_type = str(batch["utt_type"][position])
        checksum = waveform_sha256(batch["waveform"][position])
        if label != expected["label"]:
            raise RuntimeError(f"Sai nhãn benchmark cho {file_name}.")
        if speaker_id != expected["speaker_id"] or utt_type != expected["utt_type"]:
            raise RuntimeError(f"Sai metadata benchmark cho {file_name}.")
        if checksum != expected["waveform_sha256"]:
            raise RuntimeError(f"Sai checksum waveform benchmark cho {file_name}.")
    return files


def run_loader_once(
    *,
    dataset: VSASVParquetDataset,
    reference_by_file: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    num_workers: int,
    seed: int,
    process: psutil.Process,
) -> dict[str, float]:
    dataset.close()
    dataset.set_epoch(0)
    expected_order = [record.file for record in dataset.records]
    observed_files: list[str] = []
    rss_values = [process_tree_rss_mib(process)]
    batch_waits: list[float] = []
    started = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator_started = time.perf_counter()
    iterator = iter(loader)
    startup_seconds = time.perf_counter() - iterator_started
    try:
        while True:
            wait_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch_waits.append(time.perf_counter() - wait_started)
            observed_files.extend(verify_batch(batch, reference_by_file))
            rss_values.append(process_tree_rss_mib(process))
        seconds = time.perf_counter() - started
        if observed_files != expected_order:
            raise RuntimeError("Thứ tự file benchmark không tái lập.")
        if not batch_waits:
            raise RuntimeError("Benchmark không đọc được batch nào.")
    finally:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        del iterator, loader
    gc.collect()
    rss_values.append(process_tree_rss_mib(process))
    return {
        "seconds": seconds,
        "startup_seconds": startup_seconds,
        "first_batch_seconds": batch_waits[0],
        "all_batch_wait_seconds": sum(batch_waits),
        "peak_rss_mib": max(rss_values),
    }


def benchmark_variants(
    *,
    datasets: Mapping[str, VSASVParquetDataset],
    reference_by_file: Mapping[str, Mapping[str, Any]],
    workers: Sequence[int],
    repeats: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    process = psutil.Process()
    names = [
        f"{loader_name}_workers{worker_count}"
        for worker_count in workers
        for loader_name in ("legacy", "manifest")
    ]
    runs: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    run_orders: list[list[str]] = []
    for repeat in range(repeats):
        order = names.copy()
        if repeat % 2:
            order.reverse()
        run_orders.append(order.copy())
        for name in order:
            loader_name, worker_text = name.rsplit("_workers", maxsplit=1)
            print(
                f"[benchmark] lượt {repeat + 1}/{repeats}: {name}",
                flush=True,
            )
            run = run_loader_once(
                dataset=datasets[loader_name],
                reference_by_file=reference_by_file,
                batch_size=batch_size,
                num_workers=int(worker_text),
                seed=seed,
                process=process,
            )
            run["repeat"] = float(repeat + 1)
            runs[name].append(run)
            print(
                f"[benchmark] hoàn tất {name}: {run['seconds']:.3f} giây",
                flush=True,
            )
    samples_per_run = len(next(iter(datasets.values())))
    return (
        {
            name: {
                "loader": name.rsplit("_workers", maxsplit=1)[0],
                "num_workers": int(name.rsplit("_workers", maxsplit=1)[1]),
                **summarize_runs(values, samples_per_run=samples_per_run),
            }
            for name, values in runs.items()
        },
        run_orders,
    )


def build_comparisons(
    variants: Mapping[str, Mapping[str, Any]],
    workers: Sequence[int],
) -> dict[str, Any]:
    by_worker: dict[str, Any] = {}
    for worker_count in workers:
        legacy = variants[f"legacy_workers{worker_count}"]
        manifest = variants[f"manifest_workers{worker_count}"]
        by_worker[str(worker_count)] = {
            "manifest_vs_legacy_throughput_percent": calculate_improvement_percent(
                float(legacy["throughput_samples_per_second"]),
                float(manifest["throughput_samples_per_second"]),
            ),
            "manifest_minus_legacy_peak_rss_mib": (
                float(manifest["peak_rss_mib"]) - float(legacy["peak_rss_mib"])
            ),
        }
    worker_comparison = None
    if 0 in workers and 2 in workers:
        workers0 = variants["manifest_workers0"]
        workers2 = variants["manifest_workers2"]
        worker_comparison = {
            "workers2_vs_workers0_throughput_percent": calculate_improvement_percent(
                float(workers0["throughput_samples_per_second"]),
                float(workers2["throughput_samples_per_second"]),
            ),
            "workers2_minus_workers0_peak_rss_mib": (
                float(workers2["peak_rss_mib"])
                - float(workers0["peak_rss_mib"])
            ),
        }
    return {
        "manifest_vs_legacy_by_worker": by_worker,
        "manifest_worker_comparison": worker_comparison,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    initialization = report["initialization"]
    variants = report["variants"]
    rows = []
    for name, variant in variants.items():
        rows.append(
            f"| `{name}` | {variant['throughput_samples_per_second']:.3f} | "
            f"{variant['run_seconds']['median_ms']:.2f} ms | "
            f"{variant['first_batch_seconds']['median_ms']:.2f} ms | "
            f"{variant['peak_rss_mib']:.2f} MiB |"
        )
    comparisons = report["comparisons"]
    comparison_lines = []
    for worker, values in comparisons["manifest_vs_legacy_by_worker"].items():
        comparison_lines.append(
            f"- Worker {worker}: manifest so với split cũ "
            f"{values['manifest_vs_legacy_throughput_percent']:+.2f}% throughput; "
            f"peak RSS chênh {values['manifest_minus_legacy_peak_rss_mib']:+.2f} MiB."
        )
    worker_comparison = comparisons["manifest_worker_comparison"]
    if worker_comparison is not None:
        comparison_lines.append(
            "- Trên loader manifest, worker 2 so với worker 0: "
            f"{worker_comparison['workers2_vs_workers0_throughput_percent']:+.2f}% "
            "throughput; peak RSS chênh "
            f"{worker_comparison['workers2_minus_workers0_peak_rss_mib']:+.2f} MiB."
        )
    return f"""# Benchmark A/B loader split cũ và manifest v2

## Trạng thái

- Kết quả: **{report['status']}**
- Thời điểm: `{report['created_at']}`
- Phạm vi: benchmark kỹ thuật, không phải kết quả khoa học.
- Manifest: `{report['protocol']['manifest_version']}`
- Seed: `{report['protocol']['seed']}`
- Mẫu mỗi lượt: {report['protocol']['samples']}
- Số lần lặp: {report['protocol']['repeats']}
- Batch size: {report['protocol']['batch_size']}
- Worker: {report['protocol']['workers']}
- Policy: `{report['protocol']['amplitude_policy']}`
- Train mode/crop epoch: `{report['protocol']['training']}` / `{report['protocol']['epoch']}`

## Tương đương đầu ra

- File, speaker, `utt_type`, nhãn và waveform sau tiền xử lý: **ĐẠT**.
- Hai dataset được sắp theo cùng danh sách file trước khi đọc để cùng vị trí nhận cùng crop seed.
- Checksum danh sách file: `{report['selection']['file_list_sha256']}`
- Số shard trong mẫu benchmark: {report['selection']['shards']}

## Khởi tạo dataset

| Loader | Trung vị | P95 | Bản ghi khả dụng | Coverage |
|---|---:|---:|---:|---:|
| Split cũ | {initialization['legacy']['timing_seconds']['median_ms']:.2f} ms | {initialization['legacy']['timing_seconds']['p95_ms']:.2f} ms | {initialization['legacy']['available_records']:,} | {initialization['legacy']['local_coverage']:.2%} |
| Manifest v2 | {initialization['manifest']['timing_seconds']['median_ms']:.2f} ms | {initialization['manifest']['timing_seconds']['p95_ms']:.2f} ms | {initialization['manifest']['available_records']:,} | {initialization['manifest']['local_coverage']:.2%} |

Manifest giảm thời gian khởi tạo so với split cũ **{initialization['manifest']['time_reduction_vs_legacy_percent']:.2f}%**.

## Đọc batch

| Biến thể | Mẫu/giây | Trung vị/lượt | First batch | Peak RSS |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## So sánh

{chr(10).join(comparison_lines)}

## Quyết định

- Trạng thái D014: **{report['decision']['status']}**.
- Ngưỡng cải thiện worker: {report['decision']['minimum_improvement_percent']:.2f}%.
- Khuyến nghị: {report['decision']['recommendation']}

## Cache và giới hạn

- Không xóa cache hệ điều hành; không gọi lượt đầu là cold cache.
- Thứ tự loader được đảo xen kẽ giữa các lần lặp và được lưu đầy đủ trong JSON.
- Kết quả chỉ áp dụng cho máy, {report['protocol']['samples']} file và các shard đang có.
- Hai loader dùng chung logic truy vấn/decode từng waveform; phép đo này không tự chứng minh một layout/cache mới đã tối ưu đường đọc.
"""


def write_reports(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    workers = validate_protocol(args)
    manifest_path = args.manifest.resolve()
    split_path = args.split.resolve()
    parquet_dir = args.parquet_dir.resolve()
    rows = read_manifest_rows(manifest_path)
    selected_files = select_benchmark_files(
        rows,
        samples=args.samples,
        seed=args.seed,
    )
    selected_rows = {row["file"]: row for row in rows if row["file"] in selected_files}

    print("[benchmark] đo thời gian khởi tạo dataset", flush=True)
    initialization, initialization_order = benchmark_initialization(
        repeats=args.repeats,
        split_path=split_path,
        manifest_path=manifest_path,
        parquet_dir=parquet_dir,
        seed=args.seed,
    )

    print("[benchmark] tạo hai dataset đối chứng", flush=True)
    legacy = build_dataset(
        "legacy",
        split_path=split_path,
        manifest_path=manifest_path,
        parquet_dir=parquet_dir,
        seed=args.seed,
    )
    manifest = build_dataset(
        "manifest",
        split_path=split_path,
        manifest_path=manifest_path,
        parquet_dir=parquet_dir,
        seed=args.seed,
    )
    try:
        aligned_legacy, aligned_manifest = align_common_records(
            legacy.records,
            manifest.records,
            selected_files,
        )
        legacy.records = aligned_legacy
        manifest.records = aligned_manifest
        legacy.set_epoch(0)
        manifest.set_epoch(0)
        references = []
        for index in range(len(selected_files)):
            print(
                f"[benchmark] đối chiếu waveform {index + 1}/{len(selected_files)}",
                flush=True,
            )
            references.append(
                validate_sample_equivalence(legacy[index], manifest[index])
            )
        reference_by_file = {row["file"]: row for row in references}
        legacy.close()
        manifest.close()
        variants, run_orders = benchmark_variants(
            datasets={"legacy": legacy, "manifest": manifest},
            reference_by_file=reference_by_file,
            workers=workers,
            repeats=args.repeats,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    finally:
        legacy.close()
        manifest.close()

    comparisons = build_comparisons(variants, workers)
    worker_comparison = comparisons["manifest_worker_comparison"]
    worker_improvement = (
        worker_comparison["workers2_vs_workers0_throughput_percent"]
        if worker_comparison is not None
        else None
    )
    speed_threshold_met = (
        worker_improvement is not None
        and worker_improvement >= args.minimum_improvement_percent
    )
    if speed_threshold_met:
        status = "CẦN ĐÁNH GIÁ CHI PHÍ RAM TRƯỚC KHI ĐỔI D014"
        recommendation = (
            "Worker 2 vượt ngưỡng tốc độ nhưng chỉ thay D014 sau khi chấp nhận "
            "mức tăng peak RSS được ghi trong báo cáo."
        )
    else:
        status = "GIỮ D014"
        recommendation = (
            "Giữ batch_size=8, num_workers=0 vì worker 2 chưa vượt ngưỡng "
            "cải thiện 15% trên loader manifest."
        )

    split_counts = Counter(row["utt_type"] for row in selected_rows.values())
    sample_rate_counts = Counter(
        int(row["native_sample_rate"]) for row in selected_rows.values()
    )
    report = {
        "status": "ĐẠT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "technical_manifest_loader_ab_benchmark_not_scientific_result",
        "protocol": {
            "seed": args.seed,
            "samples": len(selected_files),
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "workers": workers,
            "amplitude_policy": "none",
            "training": True,
            "epoch": 0,
            "manifest_version": manifest.manifest_version,
            "split_path": str(split_path),
            "manifest_path": str(manifest_path),
            "parquet_dir": str(parquet_dir),
        },
        "cache_protocol": {
            "os_cache_flushed": False,
            "claim": "warm_or_uncontrolled_os_cache",
            "initialization_order": initialization_order,
            "run_order": run_orders,
        },
        "selection": {
            "files": selected_files,
            "file_list_sha256": hashlib.sha256(
                ("\n".join(selected_files) + "\n").encode("utf-8")
            ).hexdigest(),
            "shards": len({selected_rows[file_name]["shard"] for file_name in selected_files}),
            "utt_type_counts": dict(sorted(split_counts.items())),
            "native_sample_rate_counts": {
                str(key): value for key, value in sorted(sample_rate_counts.items())
            },
        },
        "equivalence": {
            "status": "ĐẠT",
            "checked_files": len(references),
            "fields": ["file", "speaker_id", "utt_type", "label", "waveform_sha256"],
            "reference": references,
        },
        "initialization": initialization,
        "variants": variants,
        "comparisons": comparisons,
        "decision": {
            "minimum_improvement_percent": args.minimum_improvement_percent,
            "manifest_workers2_speed_threshold_met": speed_threshold_met,
            "status": status,
            "recommendation": recommendation,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "duckdb": __import__("duckdb").__version__,
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    write_reports(report, args.report_json.resolve(), args.report_md.resolve())
    print(
        f"ĐẠT: benchmark {len(selected_files)} file, {args.repeats} lần lặp; "
        f"quyết định={status}."
    )
    print(f"JSON: {args.report_json.resolve()}")
    print(f"Markdown: {args.report_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
