#!/usr/bin/env python3
"""Lắp packed artifact cho train / dev-full / test-full từ chunk từng shard (T07).

Dev-full và test-full dùng chunk đã nạp trong `data/processed/eval_chunks/`. Train
(15.885 mẫu của `development-20k-v2`) được nạp thành chunk từ Parquet cục bộ bằng
cùng `src.data.eval_ingest`, không xóa Parquet, rồi lắp bằng cùng một hàm. Mỗi artifact
được băm lại toàn bộ sau khi ghi. Test-full chỉ được đóng gói và băm lại; không có
mô hình hay metric nào chạy trên test trong script này.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_evaluation_shards import TreeEntry, load_tree  # noqa: E402
from scripts.plan_evaluation_shards import OFFICIAL_GEOMETRY  # noqa: E402
from src.data import (  # noqa: E402
    MANIFEST_COLUMNS,
    PackedManifestDataset,
    assemble_packed_from_chunks,
    verify_packed_artifact,
)
from src.data.eval_ingest import WantedRecord, process_shard  # noqa: E402
from src.data.packed import _SHARD_NAME  # noqa: E402


DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_EVAL_CHUNKS = PROJECT_ROOT / "data" / "processed" / "eval_chunks"
DEFAULT_TRAIN_CHUNKS = PROJECT_ROOT / "data" / "processed" / "train_chunks"
DEFAULT_TRAIN_SCALE2_CHUNKS = PROJECT_ROOT / "data" / "processed" / "train_scale2_chunks"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "packed"
DEFAULT_TREE_CACHE = PROJECT_ROOT / "reports" / "vsasv_shard_tree.json"
DEFAULT_AUDIO_CONFIG = PROJECT_ROOT / "configs" / "audio.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "packed_artifacts_v1.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "packed_artifacts_v1.md"
DEFAULT_SCALE2_REPORT_JSON = PROJECT_ROOT / "reports" / "packed_artifact_scale2_v1.json"
DEFAULT_SCALE2_REPORT_MD = PROJECT_ROOT / "reports" / "packed_artifact_scale2_v1.md"
DEFAULT_SCALE2_PARITY_JSON = PROJECT_ROOT / "reports" / "packed_parity_train_scale2_v1.json"
DEFAULT_SCALE2_KAGGLE_META = (
    PROJECT_ROOT / ".cache" / "scale2_03" / "kaggle_dataset" / "dataset-metadata.json"
)
MANIFESTS = PROJECT_ROOT / "data" / "manifests"

# Tên artifact → (manifest, split, nguồn chunk).
TARGETS: dict[str, tuple[str, str, str]] = {
    "train": ("development_20k_v2.csv", "closed_train", "train"),
    "dev-full": ("closed_dev_full_v1.csv", "closed_dev", "eval"),
    "test-full": ("closed_test_full_v1.csv", "closed_test", "eval"),
    "train-scale2": ("development_scale2_v1.csv", "closed_train", "train-scale2"),
}
DEFAULT_TARGETS: tuple[str, ...] = ("train", "dev-full", "test-full")
ARTIFACT_DIRS = {
    "train": "train_development_20k_v2",
    "dev-full": "closed_dev_full_v1",
    "test-full": "closed_test_full_v1",
    "train-scale2": "train_development_scale2_v1",
}
SCALE2_PREFIX_COUNT = 15_885
SCALE2_KAGGLE_DATASET_ID = "ngctnnguyn1111/doanttnt-scale2-packed-v1"
SCALE2_KAGGLE_DATASET_TITLE = "DoAnTTNT scale2 packed v1"
SCALE2_KAGGLE_LICENSE = "CC0-1.0"
PREFIX_IDENTITY_FIELDS = (
    "artifact_index",
    "source_index",
    "offset_samples",
    "num_samples",
    "waveform_sha256",
    "file",
    "speaker_id",
    "split",
    "binary_label",
    "utt_type",
    "native_sample_rate",
    "shard",
)


def shard_index(shard: str) -> int:
    match = _SHARD_NAME.match(shard)
    if match is None:
        raise ValueError(f"Tên shard không hợp lệ: {shard}")
    return int(match.group(1))


def train_wanted_by_shard(
    manifest_csv: Path, split: str = "closed_train"
) -> dict[int, dict[str, WantedRecord]]:
    """Nhóm các dòng train của manifest theo shard để nạp chunk."""

    wanted: dict[int, dict[str, WantedRecord]] = {}
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError("Schema manifest train không hợp lệ.")
        for row in reader:
            if row["split"] != split:
                continue
            bucket = wanted.setdefault(shard_index(row["shard"]), {})
            if row["file"] in bucket:
                raise ValueError(f"Manifest train lặp file {row['file']}.")
            bucket[row["file"]] = WantedRecord(
                file=row["file"],
                split=row["split"],
                speaker_id=row["speaker_id"],
                utt_type=row["utt_type"],
            )
    if not wanted:
        raise ValueError(f"Manifest không có dòng {split}.")
    return wanted


def build_train_chunks(
    *,
    wanted_by_shard: Mapping[int, Mapping[str, WantedRecord]],
    parquet_dir: Path,
    chunk_dir: Path,
    tree: Mapping[int, TreeEntry],
    shard_name: Callable[[int], str],
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Nạp chunk train từ Parquet cục bộ; không bao giờ xóa Parquet."""

    say = progress or (lambda _message: None)
    counts = {"ingested": 0, "already_done": 0}
    for index in sorted(wanted_by_shard):
        if index not in tree:
            raise ValueError(f"Thiếu thông tin tree Hugging Face cho shard {index}.")
        path = parquet_dir / shard_name(index)
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu shard Parquet cục bộ cho train: {path}")
        result = process_shard(
            shard_path=path,
            shard_index=index,
            wanted=wanted_by_shard[index],
            chunk_dir=chunk_dir,
            delete_after=False,
            expected_size=tree[index].size,
            expected_sha256=tree[index].sha256,
        )
        counts[str(result["status"])] += 1
        say(f"Chunk train shard {index}: {result['status']} ({result['records']} mẫu).")
    return counts


def build_one(
    *,
    name: str,
    manifest: Path,
    split: str,
    chunk_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    summary = assemble_packed_from_chunks(manifest, chunk_dir, output_dir, split=split)  # type: ignore[arg-type]
    built_seconds = time.perf_counter() - started
    started = time.perf_counter()
    verification = verify_packed_artifact(output_dir, allow_test=split == "closed_test")
    verify_seconds = time.perf_counter() - started
    if verification.waveforms_sha256 != summary.waveforms_sha256:
        raise ValueError(f"Băm lại artifact {name} không khớp lúc ghi.")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "name": name,
        "split": split,
        "artifact_dir": output_dir.relative_to(PROJECT_ROOT).as_posix()
        if output_dir.is_relative_to(PROJECT_ROOT)
        else str(output_dir),
        "manifest": manifest.name,
        "manifest_version": metadata["manifest_version"],
        "source_manifest_sha256": summary.source_manifest_sha256,
        "artifact_count": summary.artifact_count,
        "total_samples": summary.total_samples,
        "waveform_bytes": summary.total_samples * 4,
        "chunk_shard_count": metadata["chunk_shard_count"],
        "index_sha256": summary.index_sha256,
        "waveforms_sha256": summary.waveforms_sha256,
        "metadata_sha256": _sha256(output_dir / "metadata.json"),
        "build_seconds": round(built_seconds, 3),
        "verify_seconds": round(verify_seconds, 3),
        "verified_rows": verification.artifact_count,
    }


def _reject_closed_test_context(
    *,
    path: Path | None = None,
    split: str | None = None,
    manifest_csv: Path | None = None,
    context: str,
) -> None:
    if split == "closed_test":
        raise ValueError(f"Từ chối truy cập dữ liệu closed_test trong {context}.")
    if path is not None and "closed_test" in path.as_posix().lower():
        raise ValueError(f"Từ chối đường dẫn closed_test trong {context}: {path}")
    if manifest_csv is not None and "closed_test" in manifest_csv.as_posix().lower():
        raise ValueError(f"Từ chối manifest closed_test trong {context}: {manifest_csv}")


def verify_manifest_shards_against_tree(
    *,
    manifest_csv: Path,
    parquet_dir: Path,
    tree: Mapping[int, TreeEntry],
    expected_split: str = "closed_train",
) -> dict[str, Any]:
    """Kiểm đủ shard mà manifest cần có trong thư mục cục bộ và khớp SHA vs tree."""
    _reject_closed_test_context(
        split=expected_split, manifest_csv=manifest_csv, context="verify_manifest_shards_against_tree"
    )
    shard_indices: set[int] = set()
    row_count = 0
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError("Schema manifest không hợp lệ khi kiểm shard.")
        for row in reader:
            row_count += 1
            if row["split"] == "closed_test":
                raise ValueError("Từ chối manifest chứa dòng closed_test trong scale2.")
            if row["split"] != expected_split:
                continue
            shard_indices.add(shard_index(row["shard"]))
    if not shard_indices:
        raise ValueError(f"Manifest không có dòng {expected_split} nào: {manifest_csv}")

    total_bytes = 0
    for index in sorted(shard_indices):
        if index not in tree:
            raise ValueError(f"Thiếu thông tin tree Hugging Face cho shard {index}.")
        shard_path = parquet_dir / f"train-{index:05d}-of-{OFFICIAL_GEOMETRY.total_shards:05d}.parquet"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Thiếu shard Parquet cục bộ: {shard_path}")
        actual_size = shard_path.stat().st_size
        entry = tree[index]
        if actual_size != entry.size:
            raise ValueError(
                f"Sai kích thước shard {shard_path.name}: nhận {actual_size}, kỳ vọng {entry.size}."
            )
        actual_sha = _sha256(shard_path)
        if actual_sha != entry.sha256:
            raise ValueError(
                f"SHA-256 shard {shard_path.name} không khớp vsasv_shard_tree.json: {actual_sha} != {entry.sha256}."
            )
        total_bytes += actual_size
    return {
        "checked_shards": len(shard_indices),
        "manifest_rows_scanned": row_count,
        "total_shard_bytes": total_bytes,
        "all_matched": True,
    }


def _read_packed_metadata_no_test(artifact_dir: Path, context: str) -> dict[str, Any]:
    _reject_closed_test_context(path=artifact_dir, context=context)
    meta_path = artifact_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metadata artifact: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    _reject_closed_test_context(split=str(metadata.get("split", "")), context=context)
    return metadata


def compare_prefix_with_v1_packed(
    *,
    scale2_dir: Path,
    v1_dir: Path,
    prefix_count: int = SCALE2_PREFIX_COUNT,
) -> dict[str, Any]:
    """Đối chiếu từng dòng index và từng byte float32 cho `prefix_count` mẫu đầu giữa ×2 và ×1."""
    started = time.perf_counter()
    s_meta = _read_packed_metadata_no_test(scale2_dir, "compare_prefix_with_v1_packed")
    v_meta = _read_packed_metadata_no_test(v1_dir, "compare_prefix_with_v1_packed")
    for field in ("format_version", "preprocessing_stage", "dtype", "target_sample_rate", "crop_seed", "split"):
        if s_meta.get(field) != v_meta.get(field):
            raise ValueError(
                f"Metadata {field} lệch giữa ×2 ({s_meta.get(field)}) và ×1 ({v_meta.get(field)})."
            )

    with (scale2_dir / "index.csv").open("r", encoding="utf-8", newline="") as s_handle:
        s_rows = list(csv.DictReader(s_handle))
    with (v1_dir / "index.csv").open("r", encoding="utf-8", newline="") as v_handle:
        v_rows = list(csv.DictReader(v_handle))

    if len(v_rows) < prefix_count or len(s_rows) < prefix_count:
        raise ValueError(
            f"Số dòng index không đủ {prefix_count}: ×2={len(s_rows)}, ×1={len(v_rows)}."
        )

    prefix_samples = 0
    prefix_bytes = 0
    with (
        (scale2_dir / "waveforms.f32").open("rb") as s_bin,
        (v1_dir / "waveforms.f32").open("rb") as v_bin,
    ):
        for i in range(prefix_count):
            sr = s_rows[i]
            vr = v_rows[i]
            for field in PREFIX_IDENTITY_FIELDS:
                if sr.get(field) != vr.get(field):
                    raise ValueError(
                        f"Lệch index dòng {i} ở trường {field}: ×2={sr.get(field)!r}, ×1={vr.get(field)!r}."
                    )
            if int(sr["artifact_index"]) != i or int(sr["source_index"]) != i:
                raise ValueError(
                    f"Chỉ số dòng {i} không liên tiếp: artifact_index={sr['artifact_index']}, source_index={sr['source_index']}."
                )
            num_samples = int(sr["num_samples"])
            byte_len = num_samples * 4
            s_bytes = s_bin.read(byte_len)
            v_bytes = v_bin.read(byte_len)
            if len(s_bytes) != byte_len or len(v_bytes) != byte_len:
                raise ValueError(f"Thiếu byte waveform tại dòng {i} ({sr['file']}).")
            if s_bytes != v_bytes:
                raise ValueError(
                    f"Lệch byte waveform tại dòng {i} ({sr['file']}): ×2 không trùng từng bit với ×1."
                )
            prefix_samples += num_samples
            prefix_bytes += byte_len

    return {
        "checked_prefix_rows": prefix_count,
        "matched_index_rows": prefix_count,
        "matched_waveform_byte_slices": prefix_count,
        "prefix_total_samples": prefix_samples,
        "prefix_total_bytes": prefix_bytes,
        "passed": True,
        "seconds": round(time.perf_counter() - started, 3),
    }


def smoke_load_packed_artifact(
    *,
    artifact_dir: Path,
    manifest_csv: Path,
    audio_config: Mapping[str, Any],
    sample_count: int = 200,
    seed: int = 2026,
    v1_dir: Path | None = None,
) -> dict[str, Any]:
    """Đọc thử tất định `sample_count` mẫu ở chế độ training và eval bằng PackedManifestDataset."""
    _read_packed_metadata_no_test(artifact_dir, "smoke_load_packed_artifact")
    _reject_closed_test_context(manifest_csv=manifest_csv, context="smoke_load_packed_artifact")

    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    for row in manifest_rows:
        if row["split"] == "closed_test":
            raise ValueError("Từ chối manifest chứa closed_test trong smoke_load_packed_artifact.")

    target_samples = int(audio_config["segmentation"]["samples"])
    amp = audio_config["waveform"]["amplitude_policy"]
    policy_kwargs = dict(
        target_samples=target_samples,
        amplitude_policy="peak",
        peak_target=float(amp["peak_target"]),
        rms_target_dbfs=float(amp["rms_target_dbfs"]),
        minimum_input_rms_dbfs=float(amp["minimum_input_rms_dbfs"]),
    )

    ds_eval = PackedManifestDataset(artifact_dir, training=False, **policy_kwargs)
    ds_train = PackedManifestDataset(artifact_dir, training=True, seed=seed, **policy_kwargs)
    ds_train.set_epoch(0)
    v1_eval = PackedManifestDataset(v1_dir, training=False, **policy_kwargs) if v1_dir is not None else None
    v1_train = PackedManifestDataset(v1_dir, training=True, seed=seed, **policy_kwargs) if v1_dir is not None else None
    if v1_train is not None:
        v1_train.set_epoch(0)

    try:
        if len(ds_eval) != len(manifest_rows):
            raise ValueError(f"Số mẫu PackedManifestDataset ({len(ds_eval)}) khác manifest ({len(manifest_rows)}).")
        actual_count = min(sample_count, len(ds_eval))
        sampled_indices = random.Random(seed).sample(range(len(ds_eval)), actual_count)

        def _verify_mode(dataset: PackedManifestDataset, v1_ds: PackedManifestDataset | None, mode_name: str) -> tuple[float, int]:
            t0 = time.perf_counter()
            v1_checked = 0
            for idx in sampled_indices:
                item = dataset[idx]
                wf = item["waveform"]
                if wf.shape != (target_samples,) or wf.dtype != torch.float32:
                    raise ValueError(f"Shape/dtype sai ({wf.shape}, {wf.dtype}) ở {mode_name} idx={idx}.")
                if not torch.isfinite(wf).all():
                    raise ValueError(f"Waveform chứa NaN/Inf ở {mode_name} idx={idx}.")
                row = manifest_rows[idx]
                if (
                    item["file"] != row["file"]
                    or item["speaker_id"] != row["speaker_id"]
                    or item["utt_type"] != row["utt_type"]
                    or int(round(float(item["label"].item()))) != int(row["binary_label"])
                    or int(item["native_sample_rate"]) != int(row["native_sample_rate"])
                ):
                    raise ValueError(f"Nhãn/metadata lệch manifest tại {mode_name} idx={idx} ({row['file']}).")
                if v1_ds is not None and idx < len(v1_ds):
                    v1_item = v1_ds[idx]
                    if not torch.equal(wf, v1_item["waveform"]):
                        raise ValueError(f"Sample tải từ ×2 khác ×1 tại {mode_name} idx={idx} ({row['file']}).")
                    v1_checked += 1
            return round(time.perf_counter() - t0, 3), v1_checked

        eval_seconds, eval_v1_checked = _verify_mode(ds_eval, v1_eval, "eval")
        train_seconds, train_v1_checked = _verify_mode(ds_train, v1_train, "training")
        return {
            "sample_count": actual_count,
            "seed": seed,
            "sampled_indices_preview": sampled_indices[:10],
            "expected_shape": [target_samples],
            "expected_dtype": "torch.float32",
            "all_finite": True,
            "labels_and_metadata_matched": True,
            "eval_seconds": eval_seconds,
            "training_seconds": train_seconds,
            "total_read_seconds": round(eval_seconds + train_seconds, 3),
            "v1_prefix_indices_checked": eval_v1_checked,
            "v1_prefix_training_checked": train_v1_checked,
            "passed": True,
        }
    finally:
        ds_eval.close()
        ds_train.close()
        if v1_eval is not None:
            v1_eval.close()
        if v1_train is not None:
            v1_train.close()


def prepare_scale2_kaggle_metadata(
    *,
    artifact_dir: Path,
    output_meta_path: Path,
    dataset_id: str = SCALE2_KAGGLE_DATASET_ID,
    title: str = SCALE2_KAGGLE_DATASET_TITLE,
    license_name: str = SCALE2_KAGGLE_LICENSE,
) -> dict[str, Any]:
    """Ghi dataset-metadata.json cho Kaggle dataset riêng tư từ artifact ×2 (chưa upload)."""
    metadata = _read_packed_metadata_no_test(artifact_dir, "prepare_scale2_kaggle_metadata")
    files_info: list[dict[str, Any]] = []
    total_bytes = 0
    for name in ("index.csv", "metadata.json", "waveforms.f32"):
        fpath = artifact_dir / name
        if not fpath.is_file():
            raise FileNotFoundError(f"Thiếu file artifact để chuẩn bị Kaggle metadata: {fpath}")
        fsize = fpath.stat().st_size
        total_bytes += fsize
        files_info.append({"file": name, "size_bytes": fsize, "sha256": _sha256(fpath)})

    payload = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
        "isPrivate": True,
        "description": (
            f"Private VSASV scale2 packed training dataset ({metadata.get('manifest_version')}, "
            f"{metadata.get('artifact_count')} samples, {metadata.get('chunk_shard_count')} shards, "
            f"seed {metadata.get('crop_seed')}). Source snapshot: {metadata.get('source_snapshot')}."
        ),
    }
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset_metadata_path": output_meta_path.relative_to(PROJECT_ROOT).as_posix()
        if output_meta_path.is_relative_to(PROJECT_ROOT)
        else str(output_meta_path),
        "dataset_metadata_sha256": _sha256(output_meta_path),
        "id": dataset_id,
        "title": title,
        "licenses": [license_name],
        "isPrivate": True,
        "planned_content_dir": artifact_dir.relative_to(PROJECT_ROOT).as_posix()
        if artifact_dir.is_relative_to(PROJECT_ROOT)
        else str(artifact_dir),
        "files": files_info,
        "expected_total_bytes": total_bytes,
        "expected_total_gb": round(total_bytes / 1e9, 3),
        "expected_total_gib": round(total_bytes / (1024**3), 3),
    }


def _render_scale2_markdown(report: Mapping[str, Any]) -> str:
    art = report["artifact"]
    shards = report["shard_verification"]
    bit = report["bit_comparison_with_v1"]
    smoke = report["smoke_load"]
    kmeta = report["kaggle_dataset"]
    lines = [
        "# Báo cáo đóng gói packed artifact train ×2 (`train_development_scale2_v1`, D036)",
        "",
        f"- Ngày tạo: `{report['generated_at']}`",
        f"- Trạng thái tổng thể: **{report['status']}**",
        f"- Định dạng & giai đoạn: `{art['format_version']}`, `{art['preprocessing_stage']}`, `dtype={art['dtype']}`, `crop_seed={art['crop_seed']}`.",
        f"- Manifest nguồn: `{art['manifest']}` (`manifest_version={art['manifest_version']}`, SHA `{art['source_manifest_sha256']}`).",
        f"- Số mẫu / tổng số mẫu float32 / số shard: `{_vn_int(art['artifact_count'])}` / `{_vn_int(art['total_samples'])}` / `{art['chunk_shard_count']}` (`verified_rows={_vn_int(art['verified_rows'])}`).",
        f"- Kiểm `{shards['checked_shards']}` shard Parquet vs `vsasv_shard_tree.json` trước đóng gói: `all_matched={shards['all_matched']}` (`{_vn_int(shards['total_shard_bytes'])}` B).",
        f"- Thời gian đóng gói & băm lại: build `{_vn_float(art['build_seconds'], 3)} s`, verify `{_vn_float(art['verify_seconds'], 3)} s`.",
        "",
        "## 1. Kích thước và SHA-256 3 file artifact",
        "",
        "| File | Kích thước (B) | GiB | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for finfo in kmeta["files"]:
        lines.append(
            f"| `{finfo['file']}` | {_vn_int(finfo['size_bytes'])} | "
            f"{_vn_float(finfo['size_bytes'] / 1024**3, 3)} | `{finfo['sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"- Tổng kích thước dự kiến upload Kaggle: `{_vn_int(kmeta['expected_total_bytes'])}` B (`{kmeta['expected_total_gb']:.3f} GB` / `{kmeta['expected_total_gib']:.3f} GiB`).",
            "",
            "## 2. So bit 15.885 mẫu đầu với ×1 (`train_development_20k_v2`)",
            "",
            f"- Kết quả: **{bit['matched_waveform_byte_slices']}/{bit['checked_prefix_rows']}** dòng trùng toàn bộ trường `index.csv` và trùng từng byte `waveforms.f32` (`passed={bit['passed']}`, thời gian `{bit['seconds']} s`).",
            "",
            "## 3. Thử nạp ngẫu nhiên tất định 200 mẫu (`PackedManifestDataset`)",
            "",
            f"- Số mẫu kiểm: `{smoke['sample_count']}` (seed `{smoke['seed']}`); shape `{smoke['expected_shape']}`, dtype `{smoke['expected_dtype']}`, hữu hạn `all_finite={smoke['all_finite']}`, nhãn & metadata khớp `True`.",
            f"- Thời gian đọc: `eval={smoke['eval_seconds']} s`, `training={smoke['training_seconds']} s` (tổng `{smoke['total_read_seconds']} s`).",
            f"- Số mẫu tiền tố (< 15.885) trùng khớp tuyệt đối với ×1 khi nạp: `eval={smoke['v1_prefix_indices_checked']}`, `training={smoke['v1_prefix_training_checked']}`.",
            "",
            "## 4. Parity và kiểm chứng D017",
            "",
        ]
    )
    parity = report.get("parity_and_d017")
    if parity:
        lines.extend(
            [
                f"- Báo cáo chi tiết: `{parity['report_path']}` (trạng thái: **{parity['status']}**, tổng thời gian `{parity['seconds']} s`).",
                f"- Parity Parquet ↔ Artifact (`31.770/31.770`): `raw_mismatches={parity['raw_mismatches']}`, `metadata_mismatches={parity['metadata_mismatches']}`, `eval_item_mismatches={parity['eval_item_mismatches']}`, `train_item_mismatches={parity['train_item_mismatches']}`.",
                f"- Hash D017 trực tiếp từ `waveforms.f32` (nhóm `native_sample_rate=16.000 Hz`): `{parity['d017_direct_16k_matched']}/{parity['d017_direct_16k_total']}` khớp (nhóm `40.000 Hz` không áp dụng trực tiếp do artifact đã resample xuống `16.000 Hz` `float32`).",
                f"- Hash D017 bắc cầu toàn vẹn Parquet (`float64`) ↔ audit D017 & ↔ lát packed (`float32` `16.000 Hz`): `{parity['d017_transitive_matched']}/{parity['d017_transitive_total']}` khớp (`0` lệch).",
            ]
        )
    else:
        lines.append("- Chưa đính kèm kết quả parity (chờ bước `check_packed_parity.py --targets train-scale2`).")
    lines.extend(
        [
            "",
            "## 5. Chuẩn bị metadata dataset Kaggle (chưa upload)",
            "",
            f"- File: `{kmeta['dataset_metadata_path']}` (SHA `{kmeta['dataset_metadata_sha256']}`).",
            f"- `id`: `{kmeta['id']}` | `title`: `{kmeta['title']}` | `licenses`: `{kmeta['licenses']}` | `isPrivate`: `{kmeta['isPrivate']}`.",
            f"- Nội dung dự kiến: thư mục `{kmeta['planned_content_dir']}` (3 file, tổng `{kmeta['expected_total_gib']:.3f} GiB`).",
            "",
        ]
    )
    return "\n".join(lines)


def _load_scale2_parity_summary(parity_report_path: Path | None) -> dict[str, Any] | None:
    if parity_report_path is None or not parity_report_path.is_file():
        return None
    data = json.loads(parity_report_path.read_text(encoding="utf-8"))
    art = data.get("artifacts", {}).get("train-scale2")
    if not art:
        return None
    d017 = art.get("d017_content_hash_audit", {})
    direct_16k = d017.get("direct_on_packed_float32", {}).get("by_sample_rate", {}).get("16000", {})
    bridge = d017.get("transitive_parquet_bridge", {})
    return {
        "report_path": parity_report_path.relative_to(PROJECT_ROOT).as_posix()
        if parity_report_path.is_relative_to(PROJECT_ROOT)
        else str(parity_report_path),
        "status": data.get("status"),
        "passed": bool(art.get("passed")),
        "seconds": data.get("seconds"),
        "raw_mismatches": art.get("raw_mismatches"),
        "metadata_mismatches": art.get("metadata_mismatches"),
        "eval_item_mismatches": art.get("eval_item_mismatches"),
        "train_item_mismatches": art.get("train_item_mismatches"),
        "d017_direct_16k_matched": direct_16k.get("matched", 0),
        "d017_direct_16k_total": direct_16k.get("total", 0),
        "d017_transitive_matched": bridge.get("d017_audit_matched", 0),
        "d017_transitive_total": bridge.get("total_rows", 0),
        "d017_passed": bool(d017.get("passed")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vn_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _vn_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Packed artifact train / dev-full / test-full (T07)",
        "",
        f"- Ngày tạo: {report['generated_at']}",
        "- Định dạng: `vsasv-packed-v1`, float32 sau resample 16 kHz, trước amplitude/crop.",
        "- Nguồn: chunk từng shard (`src/data/eval_ingest.py`); train nạp từ Parquet cục bộ,"
        " dev/test từ `data/processed/eval_chunks/`.",
        f"- Chunk train mới nạp/đã có: {report['train_chunks']}.",
        "",
        "| Artifact | Split | Manifest | Mẫu | Shard | GiB | Build (s) | Băm lại (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["artifacts"]:
        lines.append(
            f"| {item['name']} | {item['split']} | `{item['manifest']}` | "
            f"{_vn_int(item['artifact_count'])} | {item['chunk_shard_count']} | "
            f"{_vn_float(item['waveform_bytes'] / 1024**3, 2)} | "
            f"{_vn_float(item['build_seconds'], 1)} | {_vn_float(item['verify_seconds'], 1)} |"
        )
    lines += ["", "## Checksum", ""]
    for item in report["artifacts"]:
        lines.append(
            f"- {item['name']}: waveforms `{item['waveforms_sha256']}`, index "
            f"`{item['index_sha256']}`, metadata `{item['metadata_sha256']}`, manifest "
            f"`{item['source_manifest_sha256']}`."
        )
    lines += [
        "",
        "## Giới hạn",
        "",
        "- Băm lại chỉ chứng minh artifact khớp chunk; đối chiếu với Parquet nằm ở báo cáo"
        " parity (T08).",
        "- Test-full chỉ được đóng gói và băm lại; chưa chạy mô hình hay metric trên test.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS), default=list(DEFAULT_TARGETS))
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--eval-chunks", type=Path, default=DEFAULT_EVAL_CHUNKS)
    parser.add_argument("--train-chunks", type=Path, default=DEFAULT_TRAIN_CHUNKS)
    parser.add_argument("--train-scale2-chunks", type=Path, default=DEFAULT_TRAIN_SCALE2_CHUNKS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tree-cache", type=Path, default=DEFAULT_TREE_CACHE)
    parser.add_argument("--audio-config", type=Path, default=DEFAULT_AUDIO_CONFIG)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_SCALE2_PARITY_JSON)
    parser.add_argument("--kaggle-meta-out", type=Path, default=DEFAULT_SCALE2_KAGGLE_META)
    parser.add_argument("--finalize-scale2-report", action="store_true")
    parser.add_argument("--reserve-gib", type=float, default=15.0)
    return parser.parse_args(argv)


def _estimated_bytes(targets: Sequence[str], eval_chunks: Path) -> int:
    # Ước tính trên: dev+test ≤ tổng chunk eval; train ≤ 15.885 × 400 KB cho chunk + packed; train-scale2 ~ 2 × train.
    total = 0
    if {"dev-full", "test-full"} & set(targets):
        total += sum(path.stat().st_size for path in eval_chunks.glob("*.f32"))
    if "train" in targets:
        total += 2 * 15_885 * 400_000
    if "train-scale2" in targets:
        total += 2 * 31_770 * 400_000
    return total


def _run_scale2_mode(args: argparse.Namespace) -> int:
    if set(args.targets) != {"train-scale2"}:
        raise ValueError("Target `train-scale2` phải chạy riêng bằng `--targets train-scale2`, không gộp với target cũ.")
    report_json = args.report_json if args.report_json is not None else DEFAULT_SCALE2_REPORT_JSON
    report_md = args.report_md if args.report_md is not None else DEFAULT_SCALE2_REPORT_MD
    if report_json.resolve() == DEFAULT_REPORT_JSON.resolve() or report_md.resolve() == DEFAULT_REPORT_MD.resolve():
        raise ValueError("Từ chối ghi đè báo cáo packed_artifacts_v1.* hiện có.")

    manifest_csv = MANIFESTS / TARGETS["train-scale2"][0]
    scale2_dir = args.output_root / ARTIFACT_DIRS["train-scale2"]
    v1_dir = args.output_root / ARTIFACT_DIRS["train"]
    audio_config = json.loads(args.audio_config.read_text(encoding="utf-8"))
    tree = load_tree(args.tree_cache, OFFICIAL_GEOMETRY)
    from scripts.plan_evaluation_shards import shard_filename

    shard_verif = verify_manifest_shards_against_tree(
        manifest_csv=manifest_csv,
        parquet_dir=args.parquet_dir,
        tree=tree,
        expected_split="closed_train",
    )

    if args.finalize_scale2_report:
        if not report_json.is_file():
            raise FileNotFoundError(f"Chưa có báo cáo scale2 ban đầu để hoàn tất: {report_json}")
        report = json.loads(report_json.read_text(encoding="utf-8"))
        parity_summary = _load_scale2_parity_summary(args.parity_report)
        if parity_summary is None or not parity_summary.get("passed"):
            raise ValueError(f"Báo cáo parity scale2 chưa đạt hoặc không tồn tại: {args.parity_report}")
        report["parity_and_d017"] = parity_summary
        report["status"] = "ĐẠT" if parity_summary["passed"] else "KHÔNG ĐẠT"
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_md.write_text(_render_scale2_markdown(report), encoding="utf-8")
        print(f"ĐẠT: đã cập nhật parity & D017 vào {report_json.name}.")
        return 0

    needed = _estimated_bytes(args.targets, args.eval_chunks) + round(args.reserve_gib * 1024**3)
    args.output_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output_root).free
    if free < needed:
        raise ValueError(f"Không đủ dung lượng trống: còn {free} byte, cần khoảng {needed} byte.")
    if scale2_dir.exists():
        raise FileExistsError(f"Artifact train-scale2 đã tồn tại; không ghi đè: {scale2_dir}")

    print(f"Kiểm {shard_verif['checked_shards']} shard Parquet cho scale2 đạt.", flush=True)
    train_chunks = build_train_chunks(
        wanted_by_shard=train_wanted_by_shard(manifest_csv, split="closed_train"),
        parquet_dir=args.parquet_dir,
        chunk_dir=args.train_scale2_chunks,
        tree=tree,
        shard_name=lambda index: shard_filename(OFFICIAL_GEOMETRY, index),
        progress=lambda message: print(message, flush=True),
    )
    print("Lắp artifact train-scale2 (closed_train) ...", flush=True)
    art = build_one(
        name="train-scale2",
        manifest=manifest_csv,
        split="closed_train",
        chunk_dir=args.train_scale2_chunks,
        output_dir=scale2_dir,
    )
    meta = json.loads((scale2_dir / "metadata.json").read_text(encoding="utf-8"))
    art.update(
        {
            "format_version": meta["format_version"],
            "preprocessing_stage": meta["preprocessing_stage"],
            "dtype": meta["dtype"],
            "crop_seed": meta["crop_seed"],
        }
    )

    print("So bit 15.885 mẫu đầu với train_development_20k_v2 ...", flush=True)
    bit_compare = compare_prefix_with_v1_packed(
        scale2_dir=scale2_dir,
        v1_dir=v1_dir,
        prefix_count=SCALE2_PREFIX_COUNT,
    )
    print("Thử nạp tất định 200 mẫu bằng PackedManifestDataset ...", flush=True)
    smoke_load = smoke_load_packed_artifact(
        artifact_dir=scale2_dir,
        manifest_csv=manifest_csv,
        audio_config=audio_config,
        sample_count=200,
        seed=2026,
        v1_dir=v1_dir,
    )
    kaggle_meta = prepare_scale2_kaggle_metadata(
        artifact_dir=scale2_dir,
        output_meta_path=args.kaggle_meta_out,
    )
    parity_summary = _load_scale2_parity_summary(args.parity_report)

    report = {
        "report_version": 1,
        "task": "SCALE2-03",
        "decision": "D036",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "ĐẠT" if (parity_summary is None or parity_summary.get("passed")) else "KHÔNG ĐẠT",
        "train_chunks": train_chunks,
        "shard_verification": shard_verif,
        "artifact": art,
        "bit_comparison_with_v1": bit_compare,
        "smoke_load": smoke_load,
        "parity_and_d017": parity_summary,
        "kaggle_dataset": kaggle_meta,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(_render_scale2_markdown(report), encoding="utf-8")
    print(f"ĐẠT: đã đóng gói train-scale2 ({art['artifact_count']} mẫu), so bit 15.885/15.885 và thử nạp 200 mẫu.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if "train-scale2" in args.targets:
        return _run_scale2_mode(args)

    report_json = args.report_json if args.report_json is not None else DEFAULT_REPORT_JSON
    report_md = args.report_md if args.report_md is not None else DEFAULT_REPORT_MD
    needed = _estimated_bytes(args.targets, args.eval_chunks) + round(args.reserve_gib * 1024**3)
    args.output_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output_root).free
    if free < needed:
        raise ValueError(f"Không đủ dung lượng trống: còn {free} byte, cần khoảng {needed} byte.")
    for name in args.targets:
        target = args.output_root / ARTIFACT_DIRS[name]
        if target.exists():
            raise FileExistsError(f"Artifact {name} đã tồn tại; không ghi đè: {target}")

    train_chunks: dict[str, int] = {}
    if "train" in args.targets:
        tree = load_tree(args.tree_cache, OFFICIAL_GEOMETRY)
        from scripts.plan_evaluation_shards import shard_filename

        train_chunks = build_train_chunks(
            wanted_by_shard=train_wanted_by_shard(MANIFESTS / TARGETS["train"][0]),
            parquet_dir=args.parquet_dir,
            chunk_dir=args.train_chunks,
            tree=tree,
            shard_name=lambda index: shard_filename(OFFICIAL_GEOMETRY, index),
            progress=lambda message: print(message, flush=True),
        )

    artifacts = []
    for name in args.targets:
        manifest_name, split, source = TARGETS[name]
        chunk_dir = args.train_chunks if source == "train" else args.eval_chunks
        print(f"Lắp artifact {name} ({split}) ...", flush=True)
        item = build_one(
            name=name,
            manifest=MANIFESTS / manifest_name,
            split=split,
            chunk_dir=chunk_dir,
            output_dir=args.output_root / ARTIFACT_DIRS[name],
        )
        print(
            f"  {item['artifact_count']} mẫu, {item['waveform_bytes'] / 1024**3:.2f} GiB, "
            f"build {item['build_seconds']} s, băm lại {item['verify_seconds']} s.",
            flush=True,
        )
        artifacts.append(item)

    report = {
        "report_version": 1,
        "task": "T07",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_chunks": train_chunks,
        "artifacts": artifacts,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"ĐẠT: {len(artifacts)} artifact đã lắp và băm lại.")
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
