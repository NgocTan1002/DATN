#!/usr/bin/env python3
"""Đo ảnh hưởng của lỗi LFCC phụ thuộc batch lên một checkpoint B0 đã huấn luyện (D027).

Chấm lại các mẫu development của report nguồn theo hai cách: batch 8 như lúc huấn luyện
(đặc trưng cũ) và từng mẫu riêng (đặc trưng đã sửa, `lfcc_features`). Waveform đọc từ chunk
float32 do `src.data.eval_ingest` ghi, đi qua cùng `peak` và center crop 4 giây. Script chỉ
đọc; trước khi so sánh nó kiểm tra rằng chấm theo batch tái dựng đúng score đã lưu.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.amplitude import apply_amplitude_policy  # noqa: E402
from src.data.segment import fixed_length_segment  # noqa: E402
from src.metrics import equal_error_rate  # noqa: E402
from src.metrics.bootstrap import paired_speaker_bootstrap, per_utt_type_eer  # noqa: E402
from src.models.lfcc_lcnn import build_lcnn_from_config, build_lfcc_transform  # noqa: E402

DEFAULT_SOURCE = PROJECT_ROOT / "reports" / "b0_peak_full_session_10.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "kaggle_outputs" / "b0-full-session-10-v23" / "checkpoints"
    / "b0_peak_full_v1" / "g-4174b60757a44324a9b864a4931e7a18" / "checkpoint.pt"
)
DEFAULT_CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "eval_chunks"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "lfcc_lcnn.json"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "lfcc_batch_effect.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "reports" / "lfcc_batch_effect.md"
CHANGED_SCORE_TOLERANCE = 1e-4
RECONSTRUCTION_TOLERANCE = 1e-4


def score_batched(
    model: torch.nn.Module,
    lfcc: Callable[[torch.Tensor], torch.Tensor],
    waveforms: torch.Tensor,
    *,
    batch_size: int,
) -> list[float]:
    """Xác suất spoof khi gom `batch_size` mẫu liên tiếp thành một batch."""

    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, waveforms.shape[0], batch_size):
            features = lfcc(waveforms[start : start + batch_size]).unsqueeze(1)
            scores += torch.sigmoid(model(features)).tolist()
    return scores


def score_single(
    model: torch.nn.Module,
    lfcc: Callable[[torch.Tensor], torch.Tensor],
    waveforms: torch.Tensor,
) -> list[float]:
    """Xác suất spoof khi mỗi mẫu chạy riêng; tương đương `lfcc_features`."""

    return score_batched(model, lfcc, waveforms, batch_size=1)


def compare_scores(
    labels: Sequence[int],
    stored: Sequence[float],
    single: Sequence[float],
    speakers: Sequence[str],
    utt_types: Sequence[str],
    *,
    threshold: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """So sánh score theo batch (cũ) với score riêng từng mẫu (đã sửa)."""

    sizes = {len(labels), len(stored), len(single), len(speakers), len(utt_types)}
    if len(sizes) != 1:
        raise ValueError("labels, scores, speakers và utt_types phải có cùng số phần tử.")
    stored_array = np.asarray(stored, dtype=np.float64)
    single_array = np.asarray(single, dtype=np.float64)
    difference = np.abs(stored_array - single_array)
    flipped = (stored_array >= threshold) != (single_array >= threshold)
    by_utt: dict[str, int] = {}
    for index in np.flatnonzero(flipped):
        by_utt[utt_types[index]] = by_utt.get(utt_types[index], 0) + 1
    eer_old = equal_error_rate(list(labels), list(stored))
    eer_new = equal_error_rate(list(labels), list(single))
    paired = paired_speaker_bootstrap(
        labels, single, stored, speakers, n_resamples=n_resamples, seed=seed
    )
    return {
        "samples": len(labels),
        "threshold": threshold,
        "max_abs_score_difference": float(difference.max()),
        "median_abs_score_difference": float(np.median(difference)),
        "changed_scores": int((difference > CHANGED_SCORE_TOLERANCE).sum()),
        "changed_scores_by_utt_type": _count_by(
            utt_types, difference > CHANGED_SCORE_TOLERANCE
        ),
        "decisions_changed": int(flipped.sum()),
        "decisions_changed_by_utt_type": dict(sorted(by_utt.items())),
        "eer_batched": eer_old.eer,
        "eer_per_sample": eer_new.eer,
        "eer_difference": {
            "definition": "EER(per_sample) - EER(batched); âm nghĩa là chấm riêng tốt hơn",
            "estimate": paired.estimate,
            "ci_low": paired.ci_low,
            "ci_high": paired.ci_high,
            "confidence": paired.confidence,
            "fraction_per_sample_lower": paired.fraction_a_lower,
            "resamples_requested": paired.requested_resamples,
            "resamples_valid": paired.valid_resamples,
            "seed": paired.seed,
        },
        "eer_by_utt_type_batched": {
            key: value["eer"] for key, value in per_utt_type_eer(labels, stored, utt_types).items()
        },
        "eer_by_utt_type_per_sample": {
            key: value["eer"] for key, value in per_utt_type_eer(labels, single, utt_types).items()
        },
    }


def _count_by(utt_types: Sequence[str], mask: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in np.flatnonzero(mask):
        counts[utt_types[index]] = counts.get(utt_types[index], 0) + 1
    return dict(sorted(counts.items()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locate_chunks(chunk_dir: Path) -> dict[str, tuple[Path, int, int]]:
    location: dict[str, tuple[Path, int, int]] = {}
    for index_path in sorted(chunk_dir.glob("shard-*.index.csv")):
        stem = index_path.name.removesuffix(".index.csv")
        if not (chunk_dir / f"{stem}.done.json").is_file():
            continue
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                location[row["file"]] = (
                    chunk_dir / f"{stem}.f32",
                    int(row["offset_samples"]),
                    int(row["num_samples"]),
                )
    return location


def render_markdown(report: dict[str, Any]) -> str:
    result = report["comparison"]
    diff = result["eer_difference"]
    lines = [
        "# Ảnh hưởng của LFCC phụ thuộc batch lên B0 (D027)",
        "",
        f"- Checkpoint: `{report['checkpoint']['generation']}` (SHA-256 `{report['checkpoint']['sha256']}`).",
        f"- Nguồn score: `{report['source_report']}`; {result['samples']} mẫu development cũ (manifest v2), batch {report['batch_size']}.",
        f"- Tái dựng chấm theo batch so với score lưu: lệch tối đa {report['reconstruction_max_abs_difference']:.3e}.",
        "",
        "| | Chấm theo batch (cũ) | Chấm riêng từng mẫu (đã sửa) |",
        "|---|---:|---:|",
        f"| EER | {100 * result['eer_batched']:.4f}% | {100 * result['eer_per_sample']:.4f}% |",
    ]
    for key in sorted(result["eer_by_utt_type_batched"]):
        lines.append(
            f"| EER {key} | {100 * result['eer_by_utt_type_batched'][key]:.3f}% | "
            f"{100 * result['eer_by_utt_type_per_sample'][key]:.3f}% |"
        )
    lines += [
        "",
        f"- Score đổi hơn {CHANGED_SCORE_TOLERANCE:g}: {result['changed_scores']} ({result['changed_scores_by_utt_type']}); lớn nhất {result['max_abs_score_difference']:.4f}, trung vị {result['median_abs_score_difference']:.2e}.",
        f"- Quyết định đảo tại threshold {result['threshold']:.6f}: {result['decisions_changed']} ({result['decisions_changed_by_utt_type']}).",
        f"- Hiệu EER (riêng từng mẫu − theo batch): {100 * diff['estimate']:+.3f} điểm, CI {100 * diff['confidence']:.0f}% theo speaker [{100 * diff['ci_low']:+.3f}; {100 * diff['ci_high']:+.3f}], {diff['resamples_valid']}/{diff['resamples_requested']} lần lấy mẫu hợp lệ, seed {diff['seed']}.",
        "",
        "## Giới hạn",
        "",
        "- Chỉ đo trên một checkpoint và dev cũ (31 speaker, VC từ 3 speaker); không thay thế đánh giá B0 v2 trên dev-full.",
        "- Mô hình được huấn luyện với đặc trưng cũ, nên số liệu cho thấy độ nhạy của score với cách ghép batch, không phải hiệu năng của mô hình sửa lỗi.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    development = source["controller"]["best"]["development"]
    records = development["records"]
    threshold = float(development["decision_threshold"])
    location = _locate_chunks(args.chunk_dir)
    missing = [row["file"] for row in records if row["file"] not in location]
    if missing:
        raise ValueError(f"Thiếu {len(missing)} mẫu development trong chunk, ví dụ {missing[0]}.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = build_lcnn_from_config(config)
    lfcc = build_lfcc_transform(config["lfcc"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])

    memmaps: dict[Path, np.memmap] = {}
    waves: list[torch.Tensor] = []
    for row in records:
        path, offset, count = location[row["file"]]
        if path not in memmaps:
            memmaps[path] = np.memmap(path, dtype="<f4", mode="r")
        waveform = torch.from_numpy(np.array(memmaps[path][offset : offset + count]))
        waveform = apply_amplitude_policy(waveform, "peak").waveform
        waves.append(fixed_length_segment(waveform, 64_000, training=False))
    waveforms = torch.stack(waves)

    batched = score_batched(model, lfcc, waveforms, batch_size=args.batch_size)
    stored = [float(row["score"]) for row in records]
    reconstruction = float(np.abs(np.asarray(batched) - np.asarray(stored)).max())
    if reconstruction > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"Không tái dựng được score đã lưu (lệch tối đa {reconstruction:.3e}); "
            "đường tiền xử lý không khớp lúc huấn luyện."
        )
    single = score_single(model, lfcc, waveforms)
    comparison = compare_scores(
        [int(row["label"]) for row in records],
        stored,
        single,
        [row["speaker_id"] for row in records],
        [row["utt_type"] for row in records],
        threshold=threshold,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "D027",
        "scope": "lfcc_batch_dependence_measurement_not_scientific_result",
        "source_report": args.source_report.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint": {
            "generation": args.checkpoint.parent.name,
            "sha256": _sha256_file(args.checkpoint),
        },
        "batch_size": args.batch_size,
        "reconstruction_max_abs_difference": reconstruction,
        "comparison": comparison,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"ĐẠT: {comparison['samples']} mẫu; score đổi {comparison['changed_scores']}; "
        f"quyết định đảo {comparison['decisions_changed']}; EER "
        f"{100 * comparison['eer_batched']:.4f}% → {100 * comparison['eer_per_sample']:.4f}%."
    )
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
