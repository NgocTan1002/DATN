#!/usr/bin/env python3
"""Kiểm parity giữa packed artifact và đường Parquet production (T08, D026/D027).

Với mỗi dòng của artifact có shard Parquet cục bộ, script đọc lại waveform bằng đúng
các hàm của `VSASVManifestDataset` (`_audio_from_rows`, `_prepare_sample`), gom truy vấn
theo shard để không phải mở Parquet từng file, rồi so:

1. metadata/thứ tự dòng của artifact với manifest (`source_index` = vị trí trong split);
2. waveform float32 sau resample, so từng byte với lát lưu trong artifact;
3. toàn bộ sample sau amplitude `peak` và crop: chế độ đánh giá, và với train thêm
   crop ngẫu nhiên ở từng cặp (seed huấn luyện, epoch) khai báo.

Train phải phủ 100% Parquet cục bộ; dev-full/test-full chỉ so được phần shard còn
Parquet cục bộ, phần còn lại dựa vào checksum lúc nạp. Script không chạy mô hình, không
tính metric và không ghi thống kê nhãn của test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from src.data import (  # noqa: E402
    MANIFEST_COLUMNS,
    PackedManifestDataset,
    VSASVManifestDataset,
)
from src.data.packed import _read_manifest_split, _sha256_file  # noqa: E402
from src.data.vsasv import configure_duckdb  # noqa: E402


DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_PACKED_ROOT = PROJECT_ROOT / "data" / "processed" / "packed"
DEFAULT_AUDIO_CONFIG = PROJECT_ROOT / "configs" / "audio.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "packed_parity_v1.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "packed_parity_v1.md"
MANIFESTS = PROJECT_ROOT / "data" / "manifests"
TRAIN_CHECKS: tuple[tuple[int, int], ...] = ((2026, 0), (2027, 1), (2028, 2))
TARGETS: dict[str, tuple[str, str, str]] = {
    "train": ("train_development_20k_v2", "development_20k_v2.csv", "closed_train"),
    "dev-full": ("closed_dev_full_v1", "closed_dev_full_v1.csv", "closed_dev"),
    "test-full": ("closed_test_full_v1", "closed_test_full_v1.csv", "closed_test"),
}
ITEM_FIELDS = (
    "label", "file", "speaker_id", "utt_type", "native_sample_rate", "sample_rate",
    "amplitude_policy", "input_peak", "output_peak", "input_rms", "output_rms",
    "input_rms_dbfs", "output_rms_dbfs", "applied_gain", "near_silence", "peak_limited",
)


def _items_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right) or not torch.equal(left["waveform"], right["waveform"]):
        return False
    for field in ITEM_FIELDS:
        a, b = left[field], right[field]
        if isinstance(a, torch.Tensor):
            if not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True


def _write_reference_manifest(
    rows: Sequence[Mapping[str, str]], parquet_dir: Path, target: Path
) -> list[Mapping[str, str]]:
    """Ghi manifest tạm chỉ gồm dòng có shard cục bộ để dựng reader Parquet.

    `VSASVManifestDataset` chỉ nhận split train/dev; với test, cột split được ghi là
    `closed_dev` trong tệp tạm. Split không tham gia đọc hay tiền xử lý waveform.
    """

    local = [row for row in rows if (parquet_dir / row["shard"]).is_file()]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in local:
            out = {column: row[column] for column in MANIFEST_COLUMNS}
            if out["split"] == "closed_test":
                out["split"] = "closed_dev"
            writer.writerow(out)
    return local


def check_artifact_parity(
    *,
    artifact_dir: Path,
    manifest_csv: Path,
    split: str,
    parquet_dir: Path,
    audio_config: Mapping[str, Any],
    train_checks: Sequence[tuple[int, int]] = TRAIN_CHECKS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """So một artifact với đường Parquet; trả thống kê và số lệch theo từng phép so."""

    say = progress or (lambda _message: None)
    waveform_config = audio_config["waveform"]
    amplitude = waveform_config["amplitude_policy"]
    target_rate = int(waveform_config["target_sample_rate_hz"])
    target_samples = int(audio_config["segmentation"]["samples"])
    policy_args = dict(
        amplitude_policy="peak",
        peak_target=float(amplitude["peak_target"]),
        rms_target_dbfs=float(amplitude["rms_target_dbfs"]),
        minimum_input_rms_dbfs=float(amplitude["minimum_input_rms_dbfs"]),
    )
    allow_test = split == "closed_test"
    manifest_rows, _version, _snapshot = _read_manifest_split(manifest_csv, split)  # type: ignore[arg-type]
    packed_eval = PackedManifestDataset(
        artifact_dir, training=False, target_samples=target_samples,
        allow_test=allow_test, **policy_args,  # type: ignore[arg-type]
    )
    packed_train = {
        (seed, epoch): PackedManifestDataset(
            artifact_dir, training=True, target_samples=target_samples, seed=seed,
            allow_test=allow_test, **policy_args,  # type: ignore[arg-type]
        )
        for seed, epoch in (train_checks if split == "closed_train" else ())
    }
    for (seed, epoch), dataset in packed_train.items():
        dataset.set_epoch(epoch)

    result: dict[str, Any] = {
        "split": split,
        "artifact_rows": len(packed_eval),
        "manifest_rows": len(manifest_rows),
        "manifest_sha256_matches": _sha256_file(manifest_csv) == packed_eval.source_manifest_sha256,
        "metadata_mismatches": 0,
        "local_rows": 0,
        "local_shards": 0,
        "missing_local_shards": 0,
        "raw_mismatches": 0,
        "raw_max_abs_diff": 0.0,
        "eval_item_mismatches": 0,
        "train_item_mismatches": {f"seed{s}_epoch{e}": 0 for s, e in packed_train},
        "examples": [],
    }
    reference: VSASVManifestDataset | None = None
    connection = duckdb.connect(database=":memory:")
    configure_duckdb(connection)
    try:
        if len(manifest_rows) != len(packed_eval):
            result["metadata_mismatches"] += abs(len(manifest_rows) - len(packed_eval))
        for position, (row, record) in enumerate(zip(manifest_rows, packed_eval.records)):
            expected = (
                position, row["file"], row["speaker_id"], row["utt_type"],
                int(row["binary_label"]), int(row["native_sample_rate"]), row["shard"],
            )
            actual = (
                record.source_index, record.file, record.speaker_id, record.utt_type,
                record.binary_label, record.native_sample_rate, record.shard,
            )
            if expected != actual:
                result["metadata_mismatches"] += 1
                if len(result["examples"]) < 5:
                    result["examples"].append({"file": row["file"], "kind": "metadata"})

        shards = sorted({row["shard"] for row in manifest_rows})
        result["missing_local_shards"] = sum(
            1 for shard in shards if not (parquet_dir / shard).is_file()
        )
        with tempfile.TemporaryDirectory() as scratch:
            reference_manifest = Path(scratch) / "reference.csv"
            local_rows = _write_reference_manifest(manifest_rows, parquet_dir, reference_manifest)
            result["local_rows"] = len(local_rows)
            if not local_rows:
                return result
            reference = VSASVManifestDataset(
                reference_manifest, parquet_dir, split="closed_train" if split == "closed_train" else "closed_dev",
                training=False, target_sample_rate=target_rate, target_samples=target_samples,
                **policy_args,  # type: ignore[arg-type]
            )
        packed_position = {record.file: index for index, record in enumerate(packed_eval.records)}
        by_shard: dict[Path, list[int]] = defaultdict(list)
        for index, record in enumerate(reference.records):
            by_shard[record.shard].append(index)
        result["local_shards"] = len(by_shard)
        for shard_number, (shard_path, indices) in enumerate(sorted(by_shard.items()), start=1):
            files = [reference.records[i].file for i in indices]
            fetched = {
                str(file_name): (audio, speaker, utt)
                for file_name, audio, speaker, utt in connection.execute(
                    "SELECT file, audio, label, utt_type FROM read_parquet(?) "
                    "WHERE list_contains(?, file)",
                    [shard_path.as_posix(), files],
                ).fetchall()
            }
            for ref_index in indices:
                record = reference.records[ref_index]
                packed_index = packed_position[record.file]
                packed_record = packed_eval.records[packed_index]
                row = fetched.get(record.file)
                waveform, native_rate = reference._audio_from_rows(record, [row] if row else [])
                resampled = (
                    torchaudio.functional.resample(waveform, native_rate, target_rate)
                    if native_rate != target_rate else waveform
                ).to(torch.float32)
                start = packed_record.offset_samples
                stored = torch.from_numpy(
                    np.asarray(packed_eval._waveforms[start : start + packed_record.num_samples])
                )
                if resampled.shape != stored.shape or not torch.equal(resampled, stored):
                    result["raw_mismatches"] += 1
                    if resampled.shape == stored.shape:
                        diff = float((resampled - stored).abs().max())
                        result["raw_max_abs_diff"] = max(result["raw_max_abs_diff"], diff)
                    if len(result["examples"]) < 5:
                        result["examples"].append({"file": record.file, "kind": "raw"})
                reference.training = False
                if not _items_equal(
                    reference._prepare_sample(ref_index, record, waveform, native_rate),
                    packed_eval[packed_index],
                ):
                    result["eval_item_mismatches"] += 1
                for (seed, epoch), dataset in packed_train.items():
                    if ref_index != packed_index:
                        raise ValueError("Train reference phải giữ đúng thứ tự manifest.")
                    reference.training = True
                    reference.seed = seed
                    reference.set_epoch(epoch)
                    if not _items_equal(
                        reference._prepare_sample(ref_index, record, waveform, native_rate),
                        dataset[packed_index],
                    ):
                        result["train_item_mismatches"][f"seed{seed}_epoch{epoch}"] += 1
            if shard_number % 10 == 0 or shard_number == len(by_shard):
                say(f"  {split}: {shard_number}/{len(by_shard)} shard cục bộ đã so.")
    finally:
        connection.close()
        packed_eval.close()
        for dataset in packed_train.values():
            dataset.close()
        if reference is not None:
            reference.close()

    result["local_coverage"] = result["local_rows"] / max(result["artifact_rows"], 1)
    mismatches = (
        result["metadata_mismatches"] + result["raw_mismatches"]
        + result["eval_item_mismatches"] + sum(result["train_item_mismatches"].values())
    )
    full_coverage_required = split == "closed_train"
    result["passed"] = bool(
        mismatches == 0
        and result["manifest_sha256_matches"]
        and result["local_rows"] > 0
        and (not full_coverage_required or result["local_rows"] == result["artifact_rows"])
    )
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Parity packed artifact với đường Parquet (T08)",
        "",
        f"- Ngày chạy: {report['generated_at']}; trạng thái: **{report['status']}**.",
        "- So bằng hàm production của `VSASVManifestDataset`; amplitude `peak`, đoạn 64.000 mẫu.",
        f"- Crop train so ở các cặp (seed, epoch): {report['train_checks']}.",
        "",
        "| Artifact | Dòng | Dòng so được | Phủ | Shard thiếu Parquet | Lệch metadata | "
        "Lệch waveform | Lệch sample eval | Lệch sample train | Đạt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in report["artifacts"].items():
        train = sum(item["train_item_mismatches"].values()) if item["train_item_mismatches"] else "—"
        coverage = f"{item['local_coverage'] * 100:.2f}".replace(".", ",")
        lines.append(
            f"| {name} | {item['artifact_rows']:,} | {item['local_rows']:,} | ".replace(",", ".")
            + f"{coverage}% | {item['missing_local_shards']} | "
            f"{item['metadata_mismatches']} | {item['raw_mismatches']} | "
            f"{item['eval_item_mismatches']} | {train} | {'có' if item['passed'] else 'KHÔNG'} |"
        )
    lines += [
        "",
        "## Giới hạn",
        "",
        "- Dev-full/test-full chỉ so được dòng thuộc 68 shard còn Parquet cục bộ; phần còn lại"
        " dựa vào SHA-256 shard theo tree Hugging Face và checksum waveform ghi lúc nạp.",
        "- Test-full chỉ được đọc để so dữ liệu; không chạy mô hình, metric hay thống kê nhãn.",
        "- Parity chứng minh đầu vào mô hình giống hệt; không nói gì về kết quả huấn luyện.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS), default=list(TARGETS))
    parser.add_argument("--packed-root", type=Path, default=DEFAULT_PACKED_ROOT)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--audio-config", type=Path, default=DEFAULT_AUDIO_CONFIG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    audio_config = json.loads(args.audio_config.read_text(encoding="utf-8"))
    artifacts: dict[str, Any] = {}
    started = time.perf_counter()
    for name in args.targets:
        directory, manifest_name, split = TARGETS[name]
        print(f"Kiểm parity {name} ...", flush=True)
        item_started = time.perf_counter()
        artifacts[name] = check_artifact_parity(
            artifact_dir=args.packed_root / directory,
            manifest_csv=MANIFESTS / manifest_name,
            split=split,
            parquet_dir=args.parquet_dir,
            audio_config=audio_config,
            progress=lambda message: print(message, flush=True),
        )
        artifacts[name]["seconds"] = round(time.perf_counter() - item_started, 3)
    status = "ĐẠT" if all(item["passed"] for item in artifacts.values()) else "KHÔNG ĐẠT"
    report = {
        "report_version": 1,
        "task": "T08",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "amplitude_policy": "peak",
        "train_checks": [list(pair) for pair in TRAIN_CHECKS],
        "audio_config_sha256": _sha256_file(args.audio_config),
        "seconds": round(time.perf_counter() - started, 3),
        "artifacts": artifacts,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"{status}: parity packed/Parquet cho {', '.join(artifacts)}.")
    return 0 if status == "ĐẠT" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
