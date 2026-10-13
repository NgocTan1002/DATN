#!/usr/bin/env python3
"""Benchmark đường đọc từng item và ứng viên đọc batch theo shard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import psutil
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_data_pipeline import (  # noqa: E402
    summarize_seconds,
    waveform_sha256,
)
from src.data import VSASVManifestDataset  # noqa: E402
from src.data.vsasv import VSASVRecord  # noqa: E402
from src.models import build_lcnn_from_config, build_lfcc_transform  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "lfcc_lcnn.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "development_20k_v2.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "batch_read_benchmark.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "batch_read_benchmark.md"
VARIANTS = ("per_item", "batch_read")


class CountingDuckDBConnection:
    """Đếm truy vấn Parquet trong khi ủy quyền cho kết nối DuckDB thật."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self.parquet_query_count = 0

    def execute(
        self,
        query: str,
        parameters: Sequence[Any] | None = None,
    ) -> Any:
        if "from read_parquet" in query.lower():
            self.parquet_query_count += 1
        if parameters is None:
            return self.connection.execute(query)
        return self.connection.execute(query, parameters)

    def close(self) -> None:
        self.connection.close()


class PerItemView(Dataset[dict[str, Any]]):
    """Expose a fixed index list without the optional batched-fetch hook."""

    def __init__(
        self,
        dataset: VSASVManifestDataset,
        indices: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.dataset[self.indices[position]]


class BatchReadCandidateView(PerItemView):
    """Benchmark-only view that reproduces the P0.1 batch-read candidate."""

    def __getitems__(self, positions: list[int]) -> list[dict[str, Any]]:
        indices = [self.indices[position] for position in positions]
        return load_batch_candidate(self.dataset, indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-improvement-percent", type=float, default=15.0)
    parser.add_argument("--maximum-throughput-cv-percent", type=float, default=15.0)
    parser.add_argument("--maximum-rss-increase-mib", type=float, default=256.0)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args()


def validate_protocol(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise ValueError("samples phải lớn hơn 0.")
    if args.repeats < 3:
        raise ValueError("repeats phải ít nhất bằng 3.")
    if args.batch_size <= 0:
        raise ValueError("batch-size phải lớn hơn 0.")
    if args.samples < args.batch_size or args.samples % args.batch_size != 0:
        raise ValueError("samples phải là bội số của batch-size và có ít nhất một batch.")
    for field in (
        "minimum_improvement_percent",
        "maximum_throughput_cv_percent",
        "maximum_rss_increase_mib",
    ):
        value = float(getattr(args, field))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} phải là số hữu hạn lớn hơn 0.")


def stable_key(text: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()
    return digest, text


def select_index_sets(
    records: Sequence[VSASVRecord],
    *,
    samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, list[int]]:
    """Select fixed same-shard, dispersed, and realistic shuffled workloads."""

    if samples < batch_size or samples % batch_size != 0:
        raise ValueError("samples phải là bội số của batch_size.")
    if len(records) < samples:
        raise ValueError(
            f"Không đủ record để benchmark: cần {samples}, có {len(records)}."
        )

    by_shard: dict[Path, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_shard[record.shard].append(index)
    eligible = [shard for shard, indices in by_shard.items() if len(indices) >= batch_size]
    if not eligible:
        raise ValueError(f"Không có shard nào chứa đủ {batch_size} file benchmark.")
    same_shard = min(eligible, key=lambda path: stable_key(path.name, seed))
    same_indices = sorted(
        by_shard[same_shard],
        key=lambda index: stable_key(records[index].file, seed),
    )[:batch_size]

    ordered_shards = sorted(by_shard, key=lambda path: stable_key(path.name, seed))
    if len(ordered_shards) < batch_size:
        raise ValueError(
            f"Không đủ shard để tạo batch phân tán: cần {batch_size}, "
            f"có {len(ordered_shards)}."
        )
    dispersed_indices = [
        min(
            by_shard[shard],
            key=lambda index: stable_key(records[index].file, seed),
        )
        for shard in ordered_shards[:batch_size]
    ]

    generator = torch.Generator().manual_seed(seed)
    workload = torch.randperm(len(records), generator=generator)[:samples].tolist()
    return {
        "same_shard": same_indices,
        "dispersed": dispersed_indices,
        "end_to_end": workload,
    }


def expected_query_count(
    records: Sequence[VSASVRecord],
    indices: Sequence[int],
    *,
    batch_size: int,
    variant: str,
) -> int:
    if variant == "per_item":
        return len(indices)
    if variant != "batch_read":
        raise ValueError(f"Biến thể benchmark không hợp lệ: {variant}")
    return sum(
        len({records[index].shard for index in indices[start : start + batch_size]})
        for start in range(0, len(indices), batch_size)
    )


def load_batch_candidate(
    dataset: VSASVManifestDataset,
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Load one candidate batch with at most one query per distinct shard."""

    if not indices:
        return []
    grouped: dict[Path, list[tuple[int, int, VSASVRecord]]] = defaultdict(list)
    for output_position, index in enumerate(indices):
        try:
            record = dataset.records[index]
        except IndexError as error:
            raise IndexError(f"Index dataset không hợp lệ: {index}") from error
        grouped[record.shard].append((output_position, index, record))

    samples: list[dict[str, Any] | None] = [None] * len(indices)
    for entries in grouped.values():
        records_by_file: dict[str, VSASVRecord] = {}
        for _, _, record in entries:
            previous = records_by_file.setdefault(record.file, record)
            if previous != record:
                raise ValueError(
                    f"Cùng file có metadata không nhất quán trong batch: {record.file}"
                )
        requested_files = list(records_by_file)
        shard = entries[0][2].shard
        rows = dataset._get_connection().execute(
            """
            SELECT file, audio, label, utt_type
            FROM read_parquet(?)
            WHERE file = ANY(?)
            """,
            [shard.as_posix(), requested_files],
        ).fetchall()
        rows_by_file: dict[str, list[tuple[Any, Any, Any]]] = {
            file_name: [] for file_name in requested_files
        }
        for logical_file, audio, speaker_id, utt_type in rows:
            file_name = str(logical_file)
            if file_name in rows_by_file:
                rows_by_file[file_name].append((audio, speaker_id, utt_type))

        for output_position, index, record in entries:
            waveform, native_sample_rate = dataset._audio_from_rows(
                record,
                rows_by_file[record.file],
            )
            samples[output_position] = dataset._prepare_sample(
                index,
                record,
                waveform,
                native_sample_rate,
            )

    completed: list[dict[str, Any]] = []
    for output_position, sample in enumerate(samples):
        if sample is None:
            raise RuntimeError(
                f"Thiếu sample đầu ra tại vị trí batch {output_position}."
            )
        completed.append(sample)
    return completed


def sample_fingerprints(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = [str(value) for value in batch["file"]]
    return [
        {
            "file": file_name,
            "speaker_id": str(batch["speaker_id"][position]),
            "utt_type": str(batch["utt_type"][position]),
            "label": float(batch["label"][position].item()),
            "native_sample_rate": int(batch["native_sample_rate"][position].item()),
            "waveform_sha256": waveform_sha256(batch["waveform"][position]),
        }
        for position, file_name in enumerate(files)
    ]


def fingerprint_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        [dict(row) for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def process_rss_mib(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024**2)


def build_dataset(
    manifest_path: Path,
    parquet_dir: Path,
    *,
    seed: int,
) -> VSASVManifestDataset:
    return VSASVManifestDataset(
        manifest_path,
        parquet_dir,
        split="closed_train",
        training=True,
        seed=seed,
        amplitude_policy="none",
    )


def attach_query_counter(dataset: VSASVManifestDataset) -> CountingDuckDBConnection:
    counter = CountingDuckDBConnection(dataset._get_connection())
    dataset._connection = counter  # type: ignore[assignment]
    return counter


def build_view(
    variant: str,
    dataset: VSASVManifestDataset,
    indices: Sequence[int],
) -> PerItemView:
    if variant == "per_item":
        return PerItemView(dataset, indices)
    if variant == "batch_read":
        return BatchReadCandidateView(dataset, indices)
    raise ValueError(f"Biến thể benchmark không hợp lệ: {variant}")


def run_once(
    *,
    variant: str,
    indices: Sequence[int],
    batch_size: int,
    seed: int,
    manifest_path: Path,
    parquet_dir: Path,
    config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run loader-only when config is None, otherwise full B0 training steps."""

    torch.manual_seed(seed)
    dataset = build_dataset(manifest_path, parquet_dir, seed=seed)
    dataset.set_epoch(0)
    counter = attach_query_counter(dataset)
    view = build_view(variant, dataset, indices)
    loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=0)
    process = psutil.Process()
    rss_start = process_rss_mib(process)
    rss_values = [rss_start]
    fingerprints: list[dict[str, Any]] = []
    loader_waits: list[float] = []
    batch_seconds: list[float] = []

    lfcc = None
    model = None
    loss_function = None
    optimizer = None
    if config is not None:
        lfcc = build_lfcc_transform(config["lfcc"])
        model = build_lcnn_from_config(config)
        loss_function = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["smoke_optimization"]["learning_rate"]),
        )

    iterator = iter(loader)
    started = time.perf_counter()
    try:
        while True:
            batch_started = time.perf_counter()
            wait_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            loader_waits.append(time.perf_counter() - wait_started)
            fingerprints.extend(sample_fingerprints(batch))

            if config is not None:
                assert lfcc is not None
                assert model is not None
                assert loss_function is not None
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                features = lfcc(batch["waveform"]).unsqueeze(1)
                logits = model(features)
                loss = loss_function(logits, batch["label"])
                if not torch.isfinite(features).all() or not torch.isfinite(loss):
                    raise RuntimeError("Pipeline benchmark tạo tensor không hữu hạn.")
                loss.backward()
                optimizer.step()
            batch_seconds.append(time.perf_counter() - batch_started)
            rss_values.append(process_rss_mib(process))
        elapsed = time.perf_counter() - started
    finally:
        dataset.close()
        del iterator, loader, view, dataset
        gc.collect()

    if len(fingerprints) != len(indices) or not batch_seconds:
        raise RuntimeError("Benchmark không trả đủ mẫu hoặc batch.")
    run = {
        "seconds": elapsed,
        "throughput_samples_per_second": len(indices) / elapsed,
        "first_batch_seconds": batch_seconds[0],
        "batch_seconds": batch_seconds,
        "loader_wait_seconds": loader_waits,
        "parquet_queries": counter.parquet_query_count,
        "rss_start_mib": rss_start,
        "peak_rss_mib": max(rss_values),
        "rss_growth_mib": max(rss_values) - rss_start,
        "fingerprint_sha256": fingerprint_sha256(fingerprints),
    }
    return run, fingerprints


def throughput_cv_percent(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("Throughput phải gồm các số hữu hạn lớn hơn 0.")
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values) / statistics.fmean(values) * 100.0


def summarize_variant(
    runs: Sequence[Mapping[str, Any]],
    *,
    samples: int,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("Không thể tổng hợp danh sách lượt chạy rỗng.")
    throughputs = [float(run["throughput_samples_per_second"]) for run in runs]
    batch_values = [
        float(value)
        for run in runs
        for value in run["batch_seconds"]
    ]
    loader_waits = [
        float(value)
        for run in runs
        for value in run["loader_wait_seconds"]
    ]
    return {
        "samples_per_run": samples,
        "repeats": len(runs),
        "throughput_samples_per_second": statistics.median(throughputs),
        "throughput_cv_percent": throughput_cv_percent(throughputs),
        "run_seconds": summarize_seconds([float(run["seconds"]) for run in runs]),
        "first_batch_seconds": summarize_seconds(
            [float(run["first_batch_seconds"]) for run in runs]
        ),
        "batch_seconds": summarize_seconds(batch_values),
        "loader_wait_seconds": summarize_seconds(loader_waits),
        "parquet_queries": [int(run["parquet_queries"]) for run in runs],
        "peak_rss_mib": max(float(run["peak_rss_mib"]) for run in runs),
        "rss_growth_mib": max(float(run["rss_growth_mib"]) for run in runs),
        "runs": [dict(run) for run in runs],
    }


def benchmark_pair(
    *,
    indices: Sequence[int],
    repeats: int,
    batch_size: int,
    seed: int,
    manifest_path: Path,
    parquet_dir: Path,
    config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[list[str]], list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    orders: list[list[str]] = []
    reference: list[dict[str, Any]] | None = None
    for repeat in range(repeats):
        order = list(VARIANTS)
        if repeat % 2:
            order.reverse()
        orders.append(order.copy())
        for variant in order:
            print(
                f"[benchmark] lượt {repeat + 1}/{repeats}: {variant}",
                flush=True,
            )
            run, fingerprints = run_once(
                variant=variant,
                indices=indices,
                batch_size=batch_size,
                seed=seed,
                manifest_path=manifest_path,
                parquet_dir=parquet_dir,
                config=config,
            )
            if reference is None:
                reference = fingerprints
            elif fingerprints != reference:
                raise RuntimeError(
                    f"Đầu ra {variant} không tương đương danh sách đối chứng."
                )
            run["repeat"] = repeat + 1
            runs[variant].append(run)
    if reference is None:
        raise RuntimeError("Benchmark không tạo được đối chứng đầu ra.")
    return (
        {
            variant: summarize_variant(values, samples=len(indices))
            for variant, values in runs.items()
        },
        orders,
        reference,
    )


def improvement_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0.0 or candidate <= 0.0:
        raise ValueError("Throughput baseline và candidate phải lớn hơn 0.")
    return (candidate / baseline - 1.0) * 100.0


def decide_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_improvement_percent: float,
    maximum_throughput_cv_percent: float,
    maximum_rss_increase_mib: float,
    query_contract_met: bool,
    output_equivalent: bool,
) -> dict[str, Any]:
    improvement = improvement_percent(
        float(baseline["throughput_samples_per_second"]),
        float(candidate["throughput_samples_per_second"]),
    )
    rss_increase = float(candidate["peak_rss_mib"]) - float(
        baseline["peak_rss_mib"]
    )
    speed_met = improvement >= minimum_improvement_percent
    stability_met = max(
        float(baseline["throughput_cv_percent"]),
        float(candidate["throughput_cv_percent"]),
    ) <= maximum_throughput_cv_percent
    memory_met = rss_increase <= maximum_rss_increase_mib
    accepted = all(
        (speed_met, stability_met, memory_met, query_contract_met, output_equivalent)
    )
    return {
        "status": "GIỮ BATCH-READ" if accepted else "LOẠI BATCH-READ",
        "accepted": accepted,
        "throughput_improvement_percent": improvement,
        "peak_rss_increase_mib": rss_increase,
        "criteria": {
            "minimum_improvement_percent": minimum_improvement_percent,
            "speed_met": speed_met,
            "maximum_throughput_cv_percent": maximum_throughput_cv_percent,
            "stability_met": stability_met,
            "maximum_rss_increase_mib": maximum_rss_increase_mib,
            "memory_met": memory_met,
            "query_contract_met": query_contract_met,
            "output_equivalent": output_equivalent,
            "protocol_unchanged": True,
        },
        "next_action": (
            "Giữ __getitems__ và tiếp tục P0.3 Kaggle."
            if accepted
            else "Không dùng __getitems__; giữ D014 và kích hoạt P1 artifact đóng gói."
        ),
    }


def shard_counts_by_batch(
    records: Sequence[VSASVRecord],
    indices: Sequence[int],
    batch_size: int,
) -> list[int]:
    return [
        len({records[index].shard for index in indices[start : start + batch_size]})
        for start in range(0, len(indices), batch_size)
    ]


def render_markdown(report: Mapping[str, Any]) -> str:
    micro = report["micro_benchmark"]
    end_to_end = report["end_to_end"]
    decision = report["decision"]
    rows = []
    for case_name in ("same_shard", "dispersed"):
        case = micro[case_name]
        for variant in VARIANTS:
            values = case["variants"][variant]
            rows.append(
                f"| `{case_name}` | `{variant}` | {case['files']} | {case['shards']} | "
                f"{values['parquet_queries']} | "
                f"{values['batch_seconds']['median_ms']:.2f} ms | "
                f"{values['throughput_samples_per_second']:.3f} |"
            )
    end_rows = []
    for variant in VARIANTS:
        values = end_to_end["variants"][variant]
        end_rows.append(
            f"| `{variant}` | {values['throughput_samples_per_second']:.3f} | "
            f"{values['throughput_cv_percent']:.2f}% | "
            f"{values['batch_seconds']['median_ms']:.2f} ms | "
            f"{values['first_batch_seconds']['median_ms']:.2f} ms | "
            f"{values['parquet_queries']} | {values['peak_rss_mib']:.2f} MiB |"
        )
    action_state = (
        "đã gỡ"
        if not report["implementation"]["production_batch_hook_active"]
        else "còn hoạt động tại thời điểm đo"
    )
    return f"""# Benchmark P0.2 — batch-read theo shard

## Kết luận

- Quyết định: **{decision['status']}**.
- Cải thiện throughput end-to-end: **{decision['throughput_improvement_percent']:+.2f}%**; cổng yêu cầu ít nhất {decision['criteria']['minimum_improvement_percent']:.2f}%.
- Chênh peak RSS: **{decision['peak_rss_increase_mib']:+.2f} MiB**.
- Hook batch-read trong Dataset: **{action_state}**.
- Đây là benchmark kỹ thuật trên CPU, không phải kết quả khoa học.

## Giao thức đo

- Manifest: `{report['protocol']['manifest_version']}`; split `closed_train`.
- Seed/epoch/policy: `{report['protocol']['seed']}` / `0` / `none`.
- Workload end-to-end: {report['protocol']['samples']} mẫu, batch {report['protocol']['batch_size']}, worker 0, {report['protocol']['repeats']} lần lặp.
- Số shard từng batch theo thứ tự shuffle cố định: {end_to_end['shards_per_batch']}.
- Đường end-to-end gồm DataLoader, LFCC, LCNN forward, BCE loss, backward và Adam step.
- Không xóa cache hệ điều hành; thứ tự hai biến thể được đảo xen kẽ. Vì vậy không gọi lượt đầu là cold cache.

## Micro-benchmark

| Case | Biến thể | File | Shard | Truy vấn từng lượt | Trung vị batch | Mẫu/giây |
|---|---|---:|---:|---|---:|---:|
{chr(10).join(rows)}

Batch cùng shard chứng minh ứng viên giảm 8 truy vấn còn 1; batch phân tán giữ 8 truy vấn. Cả hai case dùng cùng file và checksum đầu ra giữa hai đường đọc.

## A/B pipeline end-to-end

| Biến thể | Mẫu/giây | CV throughput | Trung vị batch | First batch | Truy vấn từng lượt | Peak RSS |
|---|---:|---:|---:|---:|---|---:|
{chr(10).join(end_rows)}

- File, speaker, `utt_type`, nhãn, native sample rate và checksum waveform: **ĐẠT**.
- Checksum danh sách đầu ra: `{end_to_end['fingerprint_sha256']}`.
- Cổng ổn định (CV ≤ {decision['criteria']['maximum_throughput_cv_percent']:.2f}%): **{'ĐẠT' if decision['criteria']['stability_met'] else 'KHÔNG ĐẠT'}**.
- Cổng bộ nhớ (tăng ≤ {decision['criteria']['maximum_rss_increase_mib']:.2f} MiB): **{'ĐẠT' if decision['criteria']['memory_met'] else 'KHÔNG ĐẠT'}**.
- Cổng số truy vấn: **{'ĐẠT' if decision['criteria']['query_contract_met'] else 'KHÔNG ĐẠT'}**.

## Quyết định và phạm vi

{decision['next_action']}

Không cập nhật `reports/runtime_projection.*`: workload 24 mẫu chỉ dùng để ra quyết định A/B, và cấu hình được giữ sau P0.2 vẫn là đường đọc từng item đã dùng cho dự báo ngày 02/10. P1 tiếp theo là đề xuất artifact dẫn xuất đúng manifest 20k, cache sau decode/resample nhưng trước amplitude policy và crop.
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
    validate_protocol(args)
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if int(config["seed"]) != args.seed:
        raise ValueError("Seed benchmark không khớp cấu hình LFCC-LCNN.")

    probe = build_dataset(args.manifest.resolve(), args.parquet_dir.resolve(), seed=args.seed)
    try:
        index_sets = select_index_sets(
            probe.records,
            samples=args.samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        records = list(probe.records)
        manifest_version = probe.manifest_version
        source_snapshot = probe.source_snapshot
    finally:
        probe.close()

    micro_report: dict[str, Any] = {}
    micro_references: dict[str, list[dict[str, Any]]] = {}
    for case_name in ("same_shard", "dispersed"):
        print(f"[benchmark] micro case: {case_name}", flush=True)
        variants, orders, reference = benchmark_pair(
            indices=index_sets[case_name],
            repeats=args.repeats,
            batch_size=args.batch_size,
            seed=args.seed,
            manifest_path=args.manifest.resolve(),
            parquet_dir=args.parquet_dir.resolve(),
            config=None,
        )
        indices = index_sets[case_name]
        micro_report[case_name] = {
            "files": len(indices),
            "shards": len({records[index].shard for index in indices}),
            "indices": indices,
            "run_orders": orders,
            "fingerprint_sha256": fingerprint_sha256(reference),
            "variants": variants,
        }
        micro_references[case_name] = reference

    print("[benchmark] pipeline end-to-end", flush=True)
    end_variants, end_orders, end_reference = benchmark_pair(
        indices=index_sets["end_to_end"],
        repeats=args.repeats,
        batch_size=args.batch_size,
        seed=args.seed,
        manifest_path=args.manifest.resolve(),
        parquet_dir=args.parquet_dir.resolve(),
        config=config,
    )

    query_contract_met = True
    for case_name in ("same_shard", "dispersed"):
        indices = index_sets[case_name]
        for variant in VARIANTS:
            expected = expected_query_count(
                records,
                indices,
                batch_size=args.batch_size,
                variant=variant,
            )
            observed = micro_report[case_name]["variants"][variant]["parquet_queries"]
            query_contract_met &= observed == [expected] * args.repeats
    for variant in VARIANTS:
        expected = expected_query_count(
            records,
            index_sets["end_to_end"],
            batch_size=args.batch_size,
            variant=variant,
        )
        observed = end_variants[variant]["parquet_queries"]
        query_contract_met &= observed == [expected] * args.repeats

    decision = decide_candidate(
        end_variants["per_item"],
        end_variants["batch_read"],
        minimum_improvement_percent=args.minimum_improvement_percent,
        maximum_throughput_cv_percent=args.maximum_throughput_cv_percent,
        maximum_rss_increase_mib=args.maximum_rss_increase_mib,
        query_contract_met=query_contract_met,
        output_equivalent=True,
    )
    report = {
        "status": "ĐẠT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "technical_batch_read_ab_benchmark_not_scientific_result",
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "manifest_version": manifest_version,
            "source_snapshot": source_snapshot,
            "seed": args.seed,
            "epoch": 0,
            "amplitude_policy": "none",
            "samples": args.samples,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "cache_state": "os_cache_not_cleared_order_alternated",
        },
        "environment": {
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "duckdb": duckdb.__version__,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "torch_threads": torch.get_num_threads(),
        },
        "selection": {
            "same_shard_files": [records[index].file for index in index_sets["same_shard"]],
            "dispersed_files": [records[index].file for index in index_sets["dispersed"]],
            "end_to_end_indices": index_sets["end_to_end"],
            "end_to_end_files_sha256": hashlib.sha256(
                "\n".join(records[index].file for index in index_sets["end_to_end"]).encode(
                    "utf-8"
                )
            ).hexdigest(),
        },
        "micro_benchmark": micro_report,
        "end_to_end": {
            "indices": index_sets["end_to_end"],
            "shards_per_batch": shard_counts_by_batch(
                records,
                index_sets["end_to_end"],
                args.batch_size,
            ),
            "run_orders": end_orders,
            "fingerprint_sha256": fingerprint_sha256(end_reference),
            "reference": end_reference,
            "variants": end_variants,
        },
        "equivalence": {
            "status": "ĐẠT",
            "checked_fields": [
                "file",
                "speaker_id",
                "utt_type",
                "label",
                "native_sample_rate",
                "waveform_sha256",
            ],
            "micro_fingerprints": {
                name: fingerprint_sha256(rows)
                for name, rows in micro_references.items()
            },
        },
        "implementation": {
            "candidate": "benchmark_local_equivalent_of_p0_1_getitems",
            "production_batch_hook_active": hasattr(
                VSASVManifestDataset,
                "__getitems__",
            ),
        },
        "decision": decision,
        "runtime_projection": {
            "updated": False,
            "reason": (
                "Workload 24 mẫu dùng cho A/B; đường cấu hình sau quyết định vẫn là "
                "baseline đã dùng trong dự báo ngày 02/10."
            ),
        },
    }
    write_reports(report, args.report_json.resolve(), args.report_md.resolve())
    print(f"Quyết định: {decision['status']}")
    print(
        "Cải thiện end-to-end: "
        f"{decision['throughput_improvement_percent']:+.2f}%"
    )
    print(f"JSON: {args.report_json.resolve()}")
    print(f"Markdown: {args.report_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
