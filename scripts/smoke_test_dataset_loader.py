#!/usr/bin/env python3
"""Run a small end-to-end check of the local closed-set data loaders."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import VSASVParquetDataset  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parquet_dir = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
    split_dir = PROJECT_ROOT / "data" / "splits"

    for split_name in ("train", "dev", "test"):
        dataset = VSASVParquetDataset(
            split_dir / f"closed_{split_name}.csv",
            parquet_dir,
            training=split_name == "train",
        )
        selected_indices: list[int] = []
        for desired_type in ("bonafide", "voice_conversion"):
            match = next(
                (
                    index
                    for index, record in enumerate(dataset.records)
                    if record.utt_type == desired_type
                ),
                None,
            )
            if match is not None:
                selected_indices.append(match)
        if not any(dataset.records[index].binary_label == 1 for index in selected_indices):
            spoof_index = next(
                index
                for index, record in enumerate(dataset.records)
                if record.binary_label == 1
            )
            selected_indices.append(spoof_index)

        loader = DataLoader(
            Subset(dataset, selected_indices),
            batch_size=len(selected_indices),
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))
        waveform = batch["waveform"]
        labels = batch["label"]

        if waveform.ndim != 2 or waveform.shape[1] != 64_000:
            raise RuntimeError(f"Batch {split_name} có shape sai: {waveform.shape}")
        if not torch.isfinite(waveform).all():
            raise RuntimeError(f"Batch {split_name} chứa NaN hoặc Inf.")
        if not set(labels.tolist()).issubset({0.0, 1.0}):
            raise RuntimeError(f"Batch {split_name} có nhãn không hợp lệ: {labels}")
        if set(labels.tolist()) != {0.0, 1.0}:
            raise RuntimeError(
                f"Batch {split_name} phải có cả bonafide và spoof: {labels}"
            )
        if set(batch["sample_rate"].tolist()) != {16_000}:
            raise RuntimeError(
                f"Batch {split_name} chưa chuẩn hóa về 16 kHz: {batch['sample_rate']}"
            )

        print(
            f"{split_name}: local={len(dataset):,}/{dataset.total_split_rows:,} "
            f"({dataset.local_coverage:.2%}), batch={tuple(waveform.shape)}, "
            f"types={list(batch['utt_type'])}, "
            f"native_hz={batch['native_sample_rate'].tolist()}, "
            f"labels={labels.tolist()}"
        )
        dataset.close()

    print("Dataset loader smoke test: ĐẠT")


if __name__ == "__main__":
    main()
