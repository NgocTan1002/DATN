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


SMOKE_EXPECTED_ROWS = {"train": 256, "dev": 128, "test": 64}


def selected_indices(dataset: VSASVParquetDataset) -> list[int]:
    indices: list[int] = []
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
            indices.append(match)
    if not any(dataset.records[index].binary_label == 1 for index in indices):
        spoof_index = next(
            index
            for index, record in enumerate(dataset.records)
            if record.binary_label == 1
        )
        indices.append(spoof_index)
    return indices


def verify_seeded_smoke_batch(
    split_name: str,
    split_dir: Path,
    parquet_dir: Path,
    seed: int = 2026,
) -> None:
    """Load the fixed smoke manifest twice and require an identical first batch."""

    batches: list[dict[str, object]] = []
    for _ in range(2):
        dataset = VSASVParquetDataset(
            split_dir / f"smoke_{split_name}.csv",
            parquet_dir,
            training=split_name == "train",
            seed=seed,
            amplitude_policy="none",
        )
        expected = SMOKE_EXPECTED_ROWS[split_name]
        if len(dataset) != expected or dataset.total_split_rows != expected:
            raise RuntimeError(
                f"Smoke {split_name} phải có {expected} file cục bộ, "
                f"nhận {len(dataset)}/{dataset.total_split_rows}."
            )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        batch = next(iter(loader))
        batches.append(batch)
        dataset.close()

    first, second = batches
    if list(first["file"]) != list(second["file"]):
        raise RuntimeError(f"Thứ tự batch smoke {split_name} không tái lập.")
    if not torch.equal(first["label"], second["label"]):
        raise RuntimeError(f"Nhãn batch smoke {split_name} không tái lập.")
    if not torch.equal(first["waveform"], second["waveform"]):
        raise RuntimeError(f"Waveform batch smoke {split_name} không tái lập.")
    if not torch.isfinite(first["waveform"]).all():
        raise RuntimeError(f"Waveform batch smoke {split_name} có NaN/Inf.")
    print(
        f"smoke/{split_name}: {SMOKE_EXPECTED_ROWS[split_name]} mẫu, "
        f"batch đầu tái lập={tuple(first['waveform'].shape)}, seed={seed}"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parquet_dir = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
    split_dir = PROJECT_ROOT / "data" / "splits"

    for policy in ("none", "peak", "rms_dbfs"):
        for split_name in ("train", "dev", "test"):
            dataset = VSASVParquetDataset(
                split_dir / f"closed_{split_name}.csv",
                parquet_dir,
                training=split_name == "train",
                amplitude_policy=policy,
            )
            indices = selected_indices(dataset)
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=len(indices),
                shuffle=False,
                num_workers=0,
            )
            batch = next(iter(loader))
            waveform = batch["waveform"]
            labels = batch["label"]

            if waveform.ndim != 2 or waveform.shape[1] != 64_000:
                raise RuntimeError(
                    f"Batch {split_name}/{policy} có shape sai: {waveform.shape}"
                )
            if not torch.isfinite(waveform).all():
                raise RuntimeError(f"Batch {split_name}/{policy} chứa NaN hoặc Inf.")
            if set(labels.tolist()) != {0.0, 1.0}:
                raise RuntimeError(
                    f"Batch {split_name}/{policy} phải có cả bonafide và spoof: "
                    f"{labels}"
                )
            if set(batch["sample_rate"].tolist()) != {16_000}:
                raise RuntimeError(
                    f"Batch {split_name}/{policy} chưa chuẩn hóa về 16 kHz: "
                    f"{batch['sample_rate']}"
                )
            if set(batch["amplitude_policy"]) != {policy}:
                raise RuntimeError(
                    f"Batch {split_name}/{policy} trả sai policy: "
                    f"{batch['amplitude_policy']}"
                )
            for field in (
                "input_peak",
                "output_peak",
                "input_rms",
                "output_rms",
                "input_rms_dbfs",
                "output_rms_dbfs",
                "applied_gain",
            ):
                if not torch.isfinite(batch[field]).all():
                    raise RuntimeError(
                        f"Batch {split_name}/{policy} có {field} không hữu hạn."
                    )

            print(
                f"{policy}/{split_name}: "
                f"local={len(dataset):,}/{dataset.total_split_rows:,} "
                f"({dataset.local_coverage:.2%}), batch={tuple(waveform.shape)}, "
                f"types={list(batch['utt_type'])}, "
                f"native_hz={batch['native_sample_rate'].tolist()}, "
                f"gain={[round(value, 4) for value in batch['applied_gain'].tolist()]}, "
                f"labels={labels.tolist()}"
            )
            dataset.close()

    for split_name in ("train", "dev", "test"):
        verify_seeded_smoke_batch(split_name, split_dir, parquet_dir)

    print("Dataset loader amplitude-policy smoke test: ĐẠT")


if __name__ == "__main__":
    main()
