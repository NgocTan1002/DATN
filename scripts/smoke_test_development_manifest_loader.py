#!/usr/bin/env python3
"""Smoke test loader trực tiếp trên development manifest đã khóa."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ManifestSplit, VSASVManifestDataset  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "development_20k_v2.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
EXPECTED_COUNTS: dict[ManifestSplit, int] = {
    "closed_train": 15_885,
    "closed_dev": 4_115,
}
DEFAULT_EXPECTED_MANIFEST_VERSION = "development-20k-v2"
EXPECTED_SOURCE_SNAPSHOT = "VSASV-HF-public-snapshot-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--expected-manifest-version",
        default=DEFAULT_EXPECTED_MANIFEST_VERSION,
    )
    return parser.parse_args()


def select_representative_indices(dataset: VSASVManifestDataset) -> list[int]:
    """Chọn một bonafide và một spoof từ shard nhỏ để smoke nhanh."""

    groups = (
        [index for index, record in enumerate(dataset.records) if record.binary_label == 0],
        [index for index, record in enumerate(dataset.records) if record.binary_label == 1],
    )
    if any(not group for group in groups):
        raise RuntimeError("Mỗi split manifest phải có cả bonafide và spoof để smoke.")
    return [
        min(
            group,
            key=lambda index: (
                dataset.records[index].shard.stat().st_size,
                dataset.records[index].file,
            ),
        )
        for group in groups
    ]


def load_batch(dataset: VSASVManifestDataset, indices: list[int]) -> dict[str, Any]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
    )
    return next(iter(loader))


def assert_reproducible_batch(
    first: dict[str, Any], second: dict[str, Any], split: ManifestSplit
) -> None:
    if not torch.equal(first["waveform"], second["waveform"]):
        raise AssertionError(f"Waveform {split} không tái lập với cùng seed/epoch.")
    if not torch.equal(first["label"], second["label"]):
        raise AssertionError(f"Nhãn {split} thay đổi giữa hai lần đọc.")
    for field in ("file", "speaker_id", "utt_type"):
        if first[field] != second[field]:
            raise AssertionError(f"Trường {field} của {split} không tái lập.")


def smoke_split(
    manifest: Path,
    parquet_dir: Path,
    split: ManifestSplit,
    seed: int,
    expected_manifest_version: str,
) -> None:
    started = time.perf_counter()
    dataset = VSASVManifestDataset(
        manifest,
        parquet_dir,
        split=split,
        training=split == "closed_train",
        seed=seed,
    )
    initialization_seconds = time.perf_counter() - started

    try:
        expected_count = EXPECTED_COUNTS[split]
        if len(dataset) != expected_count:
            raise AssertionError(
                f"Số mẫu {split} sai: nhận {len(dataset)}, cần {expected_count}."
            )
        if dataset.total_split_rows != expected_count or dataset.local_coverage != 1.0:
            raise AssertionError(f"Coverage {split} không đạt 100%.")
        if dataset.manifest_version != expected_manifest_version:
            raise AssertionError(
                "manifest_version sai: "
                f"nhận {dataset.manifest_version}, cần {expected_manifest_version}"
            )
        if dataset.source_snapshot != EXPECTED_SOURCE_SNAPSHOT:
            raise AssertionError(f"source_snapshot sai: {dataset.source_snapshot}")

        indices = select_representative_indices(dataset)
        batch_started = time.perf_counter()
        first = load_batch(dataset, indices)
        second = load_batch(dataset, indices)
        batch_seconds = time.perf_counter() - batch_started
        assert_reproducible_batch(first, second, split)

        expected_shape = (len(indices), 64_000)
        if tuple(first["waveform"].shape) != expected_shape:
            raise AssertionError(
                f"Shape batch {split} sai: {tuple(first['waveform'].shape)}"
            )
        if not torch.isfinite(first["waveform"]).all():
            raise AssertionError(f"Batch {split} chứa NaN hoặc Inf.")
        if set(first["label"].tolist()) != {0.0, 1.0}:
            raise AssertionError(f"Batch {split} không có đủ hai nhãn nhị phân.")
        if not torch.all(first["sample_rate"] == 16_000):
            raise AssertionError(f"Sample rate đích của {split} không phải 16 kHz.")

        details = ", ".join(
            f"{file_name} ({utt_type}, {int(sample_rate)} Hz)"
            for file_name, utt_type, sample_rate in zip(
                first["file"],
                first["utt_type"],
                first["native_sample_rate"].tolist(),
            )
        )
        print(
            f"[ĐẠT] {split}: {len(dataset):,} mẫu, coverage=100%, "
            f"khởi tạo={initialization_seconds:.3f}s, "
            f"đọc 2 batch={batch_seconds:.3f}s"
        )
        print(f"       Mẫu kiểm tra: {details}")
    finally:
        dataset.close()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    for split in ("closed_train", "closed_dev"):
        smoke_split(
            args.manifest,
            args.parquet_dir,
            split,
            args.seed,
            args.expected_manifest_version,
        )
    print(
        "[ĐẠT] Loader development manifest "
        f"{args.expected_manifest_version} tái lập trên train và development."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
