#!/usr/bin/env python3
"""Build and benchmark a deterministic packed-manifest proof of concept."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psutil
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    PackedManifestDataset,
    VSASVManifestDataset,
    build_packed_artifact,
    verify_packed_artifact,
)
from src.models import build_lcnn_from_config, build_lfcc_transform  # noqa: E402


VARIANTS = ("parquet", "packed")


class IndexedView(Dataset[dict[str, Any]]):
    """Expose a fixed, ordered index list without changing the source dataset."""

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        indices: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "development_20k_v2.csv",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data",
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "batch_read_benchmark.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "packed_manifest_poc_20261013"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lfcc_lcnn.json",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "packed_artifact_benchmark.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "packed_artifact_benchmark.md",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Dùng artifact đã có sau khi kiểm tra toàn bộ checksum.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_indices(report_path: Path) -> list[int]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_indices = report["selection"]["end_to_end_indices"]
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("Báo cáo lựa chọn không có end_to_end_indices hợp lệ.")
    indices = [int(value) for value in raw_indices]
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("end_to_end_indices phải không âm và không trùng lặp.")
    return indices


def sample_fingerprints(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, file_name in enumerate(batch["file"]):
        waveform = batch["waveform"][position].detach().cpu().contiguous()
        rows.append(
            {
                "file": str(file_name),
                "speaker_id": str(batch["speaker_id"][position]),
                "utt_type": str(batch["utt_type"][position]),
                "label": float(batch["label"][position].item()),
                "native_sample_rate": int(
                    batch["native_sample_rate"][position].item()
                ),
                "waveform_sha256": hashlib.sha256(
                    waveform.numpy().tobytes()
                ).hexdigest(),
            }
        )
    return rows


def fingerprint_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        [dict(row) for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rss_mib(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024**2)


def build_dataset_factory(
    *,
    variant: str,
    manifest: Path,
    parquet_dir: Path,
    artifact_dir: Path,
    source_indices: Sequence[int],
    seed: int,
) -> Callable[[], tuple[Dataset[dict[str, Any]], Sequence[int]]]:
    if variant == "parquet":
        def build_parquet() -> tuple[Dataset[dict[str, Any]], Sequence[int]]:
            dataset = VSASVManifestDataset(
                manifest,
                parquet_dir,
                split="closed_train",
                training=True,
                seed=seed,
                amplitude_policy="none",
            )
            dataset.set_epoch(0)
            return dataset, source_indices

        return build_parquet
    if variant == "packed":
        def build_packed() -> tuple[Dataset[dict[str, Any]], Sequence[int]]:
            dataset = PackedManifestDataset(
                artifact_dir,
                training=True,
                seed=seed,
                amplitude_policy="none",
            )
            dataset.set_epoch(0)
            return dataset, range(len(dataset))

        return build_packed
    raise ValueError(f"Biến thể benchmark không hợp lệ: {variant}")


def run_once(
    *,
    dataset_factory: Callable[
        [], tuple[Dataset[dict[str, Any]], Sequence[int]]
    ],
    config: Mapping[str, Any] | None,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    dataset, indices = dataset_factory()
    view = IndexedView(dataset, indices)
    loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=0)
    process = psutil.Process()
    rss_start = rss_mib(process)
    rss_values = [rss_start]
    fingerprints: list[dict[str, Any]] = []
    batch_seconds: list[float] = []
    loader_wait_seconds: list[float] = []

    lfcc: torch.nn.Module | None = None
    model: torch.nn.Module | None = None
    loss_function: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
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
            loader_wait_seconds.append(time.perf_counter() - wait_started)
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
            rss_values.append(rss_mib(process))
        elapsed = time.perf_counter() - started
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        del iterator, loader, view, dataset
        gc.collect()

    if not batch_seconds or not fingerprints:
        raise RuntimeError("Benchmark không tạo được batch hoặc fingerprint.")
    return (
        {
            "seconds": elapsed,
            "throughput_samples_per_second": len(fingerprints) / elapsed,
            "first_batch_seconds": batch_seconds[0],
            "batch_seconds": batch_seconds,
            "loader_wait_seconds": loader_wait_seconds,
            "rss_start_mib": rss_start,
            "peak_rss_mib": max(rss_values),
            "rss_growth_mib": max(rss_values) - rss_start,
            "fingerprint_sha256": fingerprint_sha256(fingerprints),
        },
        fingerprints,
    )


def summarize_seconds(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Không thể tổng hợp danh sách thời gian rỗng.")
    ordered = sorted(float(value) for value in values)
    p95_position = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered) * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "p95_ms": ordered[p95_position] * 1000.0,
        "minimum_ms": min(ordered) * 1000.0,
        "maximum_ms": max(ordered) * 1000.0,
        "values_seconds": [float(value) for value in values],
    }


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    throughputs = [float(run["throughput_samples_per_second"]) for run in runs]
    cv = (
        statistics.stdev(throughputs) / statistics.fmean(throughputs) * 100.0
        if len(throughputs) > 1
        else 0.0
    )
    return {
        "repeats": len(runs),
        "throughput_samples_per_second": statistics.median(throughputs),
        "throughput_cv_percent": cv,
        "run_seconds": summarize_seconds(
            [float(run["seconds"]) for run in runs]
        ),
        "first_batch_seconds": summarize_seconds(
            [float(run["first_batch_seconds"]) for run in runs]
        ),
        "batch_seconds": summarize_seconds(
            [float(value) for run in runs for value in run["batch_seconds"]]
        ),
        "loader_wait_seconds": summarize_seconds(
            [
                float(value)
                for run in runs
                for value in run["loader_wait_seconds"]
            ]
        ),
        "peak_rss_mib": max(float(run["peak_rss_mib"]) for run in runs),
        "rss_growth_mib": max(float(run["rss_growth_mib"]) for run in runs),
        "runs": [dict(run) for run in runs],
    }


def benchmark_pair(
    *,
    manifest: Path,
    parquet_dir: Path,
    artifact_dir: Path,
    indices: Sequence[int],
    config: Mapping[str, Any] | None,
    repeats: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], list[list[str]], list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    orders: list[list[str]] = []
    reference: list[dict[str, Any]] | None = None
    for repeat in range(repeats):
        order = list(VARIANTS)
        if repeat % 2:
            order.reverse()
        orders.append(order.copy())
        for variant in order:
            print(
                f"[packed-poc] lượt {repeat + 1}/{repeats}: {variant}",
                flush=True,
            )
            run, fingerprints = run_once(
                dataset_factory=build_dataset_factory(
                    variant=variant,
                    manifest=manifest,
                    parquet_dir=parquet_dir,
                    artifact_dir=artifact_dir,
                    source_indices=indices,
                    seed=seed,
                ),
                config=config,
                batch_size=batch_size,
                seed=seed,
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
        {variant: summarize_runs(values) for variant, values in runs.items()},
        orders,
        reference,
    )


def improvement_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0.0 or candidate <= 0.0:
        raise ValueError("Throughput phải lớn hơn 0.")
    return (candidate / baseline - 1.0) * 100.0


def write_reports(report: Mapping[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    loader = report["loader_only"]
    end_to_end = report["end_to_end"]
    decision = report["decision"]
    artifact = report["artifact"]
    markdown = f"""# Benchmark PoC artifact đóng gói

## Kết luận

- Trạng thái: **{report['status']}**
- Quyết định: **{decision['status']}**
- Phạm vi: benchmark kỹ thuật trên {report['protocol']['samples']} mẫu cố định; không phải kết quả khoa học.
- Artifact: `{artifact['path']}` ({artifact['size_mib']:.2f} MiB), tạo trong {artifact['build_seconds']:.2f} giây.
- Kiểm tra toàn vẹn: {artifact['verification_status']}; {artifact['verified_waveforms']} waveform được băm lại độc lập.

## Kết quả A/B

| Đường đo | Parquet (mẫu/s) | Packed (mẫu/s) | Cải thiện |
|---|---:|---:|---:|
| Loader-only | {loader['variants']['parquet']['throughput_samples_per_second']:.3f} | {loader['variants']['packed']['throughput_samples_per_second']:.3f} | {loader['improvement_percent']:.2f}% |
| B0 LFCC + LCNN | {end_to_end['variants']['parquet']['throughput_samples_per_second']:.3f} | {end_to_end['variants']['packed']['throughput_samples_per_second']:.3f} | {end_to_end['improvement_percent']:.2f}% |

Mỗi biến thể chạy {report['protocol']['repeats']} lượt, batch {report['protocol']['batch_size']}, worker 0 và luân phiên thứ tự. Fingerprint metadata/waveform khớp tuyệt đối giữa hai đường đọc.

## Cổng quyết định

- Tương đương đầu ra: {decision['criteria']['output_equivalent']}.
- Cải thiện B0 tối thiểu {decision['criteria']['minimum_improvement_percent']:.0f}%: {decision['criteria']['speed_met']}.
- CV packed không quá {decision['criteria']['maximum_throughput_cv_percent']:.0f}%: {decision['criteria']['stability_met']}.
- RSS tăng không quá {decision['criteria']['maximum_rss_increase_mib']:.0f} MiB: {decision['criteria']['memory_met']}.
- Manifest/split/seed/amplitude/crop không đổi: {decision['criteria']['protocol_unchanged']}.

## Diễn giải

PoC chỉ chứng minh đường đọc artifact cho cùng 24 mẫu và cùng B0. Không dùng kết quả này để suy ra thời gian toàn bộ 20k; cần benchmark lại sau khi tạo artifact đầy đủ.
"""
    md_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("repeats phải từ 2 trở lên để đo độ ổn định.")
    if args.batch_size <= 0:
        raise ValueError("batch-size phải lớn hơn 0.")
    if args.seed != 2026:
        raise ValueError("PoC phải giữ seed đã khóa là 2026.")

    manifest = args.manifest.resolve()
    parquet_dir = args.parquet_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    selection_report = args.selection_report.resolve()
    config_path = args.config.resolve()
    for path in (manifest, selection_report, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {path}")
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục Parquet: {parquet_dir}")
    indices = load_indices(selection_report)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if artifact_dir.exists() and not args.reuse_existing:
        raise FileExistsError(
            "Artifact đã tồn tại; dùng --reuse-existing sau khi xác nhận đúng mục tiêu: "
            f"{artifact_dir}"
        )
    build_seconds = 0.0
    if not artifact_dir.exists():
        print(f"[packed-poc] tạo artifact {len(indices)} mẫu", flush=True)
        build_started = time.perf_counter()
        build_packed_artifact(
            manifest,
            parquet_dir,
            artifact_dir,
            split="closed_train",
            indices=indices,
            seed=args.seed,
        )
        build_seconds = time.perf_counter() - build_started
    verification = verify_packed_artifact(artifact_dir)
    with (artifact_dir / "index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        stored_source_indices = [
            int(row["source_index"]) for row in csv.DictReader(handle)
        ]
    if stored_source_indices != indices:
        raise ValueError("Artifact hiện có không khớp danh sách mẫu PoC đã khóa.")

    print("[packed-poc] benchmark loader-only", flush=True)
    loader_variants, loader_orders, loader_reference = benchmark_pair(
        manifest=manifest,
        parquet_dir=parquet_dir,
        artifact_dir=artifact_dir,
        indices=indices,
        config=None,
        repeats=args.repeats,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("[packed-poc] benchmark B0 LFCC + LCNN", flush=True)
    end_variants, end_orders, end_reference = benchmark_pair(
        manifest=manifest,
        parquet_dir=parquet_dir,
        artifact_dir=artifact_dir,
        indices=indices,
        config=config,
        repeats=args.repeats,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if loader_reference != end_reference:
        raise RuntimeError("Fingerprint loader-only và B0 không tương đương.")

    loader_improvement = improvement_percent(
        float(loader_variants["parquet"]["throughput_samples_per_second"]),
        float(loader_variants["packed"]["throughput_samples_per_second"]),
    )
    end_improvement = improvement_percent(
        float(end_variants["parquet"]["throughput_samples_per_second"]),
        float(end_variants["packed"]["throughput_samples_per_second"]),
    )
    packed_cvs = (
        float(loader_variants["packed"]["throughput_cv_percent"]),
        float(end_variants["packed"]["throughput_cv_percent"]),
    )
    rss_increases = (
        float(loader_variants["packed"]["peak_rss_mib"])
        - float(loader_variants["parquet"]["peak_rss_mib"]),
        float(end_variants["packed"]["peak_rss_mib"])
        - float(end_variants["parquet"]["peak_rss_mib"]),
    )
    criteria = {
        "minimum_improvement_percent": 15.0,
        "speed_met": end_improvement >= 15.0,
        "maximum_throughput_cv_percent": 15.0,
        "stability_met": max(packed_cvs) <= 15.0,
        "maximum_rss_increase_mib": 256.0,
        "memory_met": max(rss_increases) <= 256.0,
        "output_equivalent": True,
        "integrity_verified": True,
        "protocol_unchanged": True,
    }
    accepted = all(
        bool(criteria[name])
        for name in (
            "speed_met",
            "stability_met",
            "memory_met",
            "output_equivalent",
            "integrity_verified",
            "protocol_unchanged",
        )
    )
    artifact_size = sum(
        path.stat().st_size for path in artifact_dir.iterdir() if path.is_file()
    )
    report: dict[str, Any] = {
        "status": "ĐẠT" if accepted else "KHÔNG ĐẠT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "technical_packed_artifact_poc_not_scientific_result",
        "protocol": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "split": "closed_train",
            "seed": args.seed,
            "epoch": 0,
            "amplitude_policy": "none",
            "target_sample_rate": 16_000,
            "target_samples": 64_000,
            "samples": len(indices),
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "cache_state": "os_cache_not_cleared_order_alternated",
            "selection_indices": indices,
        },
        "environment": {
            "git_commit": git_commit(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": "cpu",
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "torch_threads": torch.get_num_threads(),
        },
        "artifact": {
            "path": str(artifact_dir),
            "build_seconds": build_seconds,
            "size_bytes": artifact_size,
            "size_mib": artifact_size / (1024**2),
            "verification_status": "ĐẠT",
            "verified_waveforms": verification.artifact_count,
            "verified_total_samples": verification.total_samples,
            "waveforms_sha256": verification.waveforms_sha256,
        },
        "loader_only": {
            "run_orders": loader_orders,
            "fingerprint_sha256": fingerprint_sha256(loader_reference),
            "variants": loader_variants,
            "improvement_percent": loader_improvement,
        },
        "end_to_end": {
            "model": "B0_LFCC_LCNN",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "run_orders": end_orders,
            "fingerprint_sha256": fingerprint_sha256(end_reference),
            "variants": end_variants,
            "improvement_percent": end_improvement,
        },
        "decision": {
            "status": "CHẤP NHẬN PoC" if accepted else "TỪ CHỐI PoC",
            "accepted": accepted,
            "criteria": criteria,
            "loader_improvement_percent": loader_improvement,
            "end_to_end_improvement_percent": end_improvement,
            "maximum_packed_cv_percent": max(packed_cvs),
            "maximum_rss_increase_mib_observed": max(rss_increases),
            "next_action": (
                "Xin xác nhận trước khi tạo artifact đầy đủ development-20k-v2."
                if accepted
                else "Giữ đường đọc Parquet và phân tích nguyên nhân PoC không đạt."
            ),
        },
        "runtime_projection": {
            "updated": False,
            "reason": "PoC 24 mẫu không đại diện cho toàn bộ development-20k-v2.",
        },
    }
    write_reports(report, args.report_json.resolve(), args.report_md.resolve())
    print(
        f"[packed-poc] {report['status']}: B0 {end_improvement:.2f}%",
        flush=True,
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
