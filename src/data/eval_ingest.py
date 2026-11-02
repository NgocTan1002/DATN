"""Nạp từng shard VSASV cho dev/test: băm nội dung, đóng gói float32, rồi xóa Parquet.

Mỗi shard tạo một chunk độc lập gồm waveform float32 sau resample (trước amplitude
policy và crop, cùng hợp đồng với `src.data.packed`), chỉ mục có hash nội dung D017
tính trên waveform float64 gốc, và một done marker ghi cuối cùng. Chỉ khi done marker
đã nằm trên đĩa thì Parquet mới được phép xóa.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch
import torchaudio

from .content_hash import waveform_sha256


CHUNK_FORMAT_VERSION = 1
CHUNK_INDEX_COLUMNS = (
    "file",
    "split",
    "speaker_id",
    "utt_type",
    "binary_label",
    "native_sample_rate",
    "offset_samples",
    "num_samples",
    "content_sha256",
    "waveform_sha256",
)
FETCH_BATCH = 16


@dataclass(frozen=True)
class WantedRecord:
    """Một dòng split cần lấy từ shard và metadata kỳ vọng của nó."""

    file: str
    split: str
    speaker_id: str
    utt_type: str


@dataclass(frozen=True)
class ChunkPaths:
    waveforms: Path
    index: Path
    done: Path


def chunk_paths(chunk_dir: Path, shard_index: int) -> ChunkPaths:
    stem = f"shard-{shard_index:05d}"
    return ChunkPaths(
        waveforms=chunk_dir / f"{stem}.f32",
        index=chunk_dir / f"{stem}.index.csv",
        done=chunk_dir / f"{stem}.done.json",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wanted_digest(wanted: Mapping[str, WantedRecord]) -> str:
    material = "\n".join(
        f"{r.file}|{r.split}|{r.speaker_id}|{r.utt_type}"
        for r in sorted(wanted.values(), key=lambda item: item.file)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _fsync_write(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def ingest_shard(
    *,
    shard_path: Path,
    shard_index: int,
    wanted: Mapping[str, WantedRecord],
    chunk_dir: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    target_sample_rate: int = 16_000,
) -> dict[str, Any]:
    """Tạo chunk cho một shard; không để lại chunk dở khi gặp lỗi."""

    if not wanted:
        raise ValueError("Shard phải có ít nhất một mẫu cần lấy.")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate phải lớn hơn 0.")
    if expected_size is not None and shard_path.stat().st_size != expected_size:
        raise ValueError(
            f"Sai kích thước shard {shard_path.name}: nhận "
            f"{shard_path.stat().st_size}, cần {expected_size}."
        )
    shard_sha256 = _sha256_file(shard_path)
    if expected_sha256 is not None and shard_sha256 != expected_sha256:
        raise ValueError(f"SHA-256 shard {shard_path.name} không khớp nguồn.")

    paths = chunk_paths(chunk_dir, shard_index)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    staging = {
        "waveforms": paths.waveforms.with_suffix(".f32.tmp"),
        "index": paths.index.with_suffix(".csv.tmp"),
    }
    seen: set[str] = set()
    rows: list[dict[str, str | int]] = []
    offset = 0
    data_digest = hashlib.sha256()
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads = 1")
        cursor = connection.execute(
            "SELECT file, audio, label, utt_type FROM read_parquet(?) "
            "WHERE list_contains(?, file)",
            [shard_path.as_posix(), list(wanted)],
        )
        with staging["waveforms"].open("wb") as data_handle:
            while True:
                batch = cursor.fetchmany(FETCH_BATCH)
                if not batch:
                    break
                for file_name, audio, speaker_id, utt_type in batch:
                    file_name = str(file_name)
                    if file_name in seen:
                        raise ValueError(f"Shard lặp file {file_name}.")
                    seen.add(file_name)
                    record = wanted[file_name]
                    if (str(speaker_id), str(utt_type)) != (
                        record.speaker_id,
                        record.utt_type,
                    ):
                        raise ValueError(
                            f"Metadata không khớp Parquet cho {file_name}: nguồn="
                            f"({record.speaker_id}, {record.utt_type}), parquet="
                            f"({speaker_id}, {utt_type})."
                        )
                    if not isinstance(audio, dict) or not {
                        "array",
                        "sampling_rate",
                    }.issubset(audio):
                        raise ValueError(f"Cấu trúc audio không hợp lệ cho {file_name}.")
                    native_rate = int(audio["sampling_rate"])
                    raw = audio["array"]
                    if native_rate <= 0 or not raw:
                        raise ValueError(f"Audio rỗng hoặc sample rate sai: {file_name}.")
                    content_hash = waveform_sha256(raw, native_rate)
                    waveform = torch.as_tensor(raw, dtype=torch.float32)
                    if not torch.isfinite(waveform).all():
                        raise ValueError(f"Waveform chứa NaN hoặc Inf: {file_name}.")
                    if native_rate != target_sample_rate:
                        waveform = torchaudio.functional.resample(
                            waveform, native_rate, target_sample_rate
                        )
                    block = np.asarray(
                        waveform.to(torch.float32).contiguous().numpy(), dtype="<f4"
                    ).tobytes(order="C")
                    data_handle.write(block)
                    data_digest.update(block)
                    count = len(block) // 4
                    rows.append(
                        {
                            "file": file_name,
                            "split": record.split,
                            "speaker_id": record.speaker_id,
                            "utt_type": record.utt_type,
                            "binary_label": 0 if record.utt_type == "bonafide" else 1,
                            "native_sample_rate": native_rate,
                            "offset_samples": offset,
                            "num_samples": count,
                            "content_sha256": content_hash,
                            "waveform_sha256": hashlib.sha256(block).hexdigest(),
                        }
                    )
                    offset += count
            data_handle.flush()
            os.fsync(data_handle.fileno())
        missing = sorted(set(wanted) - seen)
        if missing:
            raise ValueError(
                f"Shard {shard_path.name} thiếu {len(missing)} mẫu cần lấy, ví dụ "
                f"{missing[0]}."
            )
        with staging["index"].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CHUNK_INDEX_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging["waveforms"], paths.waveforms)
        os.replace(staging["index"], paths.index)
        marker = {
            "chunk_format_version": CHUNK_FORMAT_VERSION,
            "shard": shard_path.name,
            "shard_index": shard_index,
            "shard_size_bytes": shard_path.stat().st_size,
            "shard_sha256": shard_sha256,
            "records": len(rows),
            "total_samples": offset,
            "target_sample_rate": target_sample_rate,
            "waveforms_sha256": data_digest.hexdigest(),
            "index_sha256": _sha256_file(paths.index),
            "wanted_sha256": _wanted_digest(wanted),
        }
        _fsync_write(
            paths.done,
            (json.dumps(marker, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        return marker
    except Exception:
        for path in (*staging.values(), paths.waveforms, paths.index, paths.done):
            path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()


def verify_chunk(chunk_dir: Path, shard_index: int) -> dict[str, Any]:
    """Băm lại chunk và đối chiếu với done marker và hash từng waveform."""

    paths = chunk_paths(chunk_dir, shard_index)
    if not (paths.done.is_file() and paths.index.is_file() and paths.waveforms.is_file()):
        return {"passed": False, "reason": "thiếu tệp chunk"}
    marker = json.loads(paths.done.read_text(encoding="utf-8"))
    if _sha256_file(paths.index) != marker["index_sha256"]:
        return {"passed": False, "reason": "chỉ mục sai checksum"}
    if paths.waveforms.stat().st_size != marker["total_samples"] * 4:
        return {"passed": False, "reason": "sai kích thước waveform"}
    if _sha256_file(paths.waveforms) != marker["waveforms_sha256"]:
        return {"passed": False, "reason": "waveform sai checksum"}
    data = np.memmap(paths.waveforms, dtype="<f4", mode="r")
    with paths.index.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start, size = int(row["offset_samples"]), int(row["num_samples"])
            if hashlib.sha256(data[start : start + size].tobytes()).hexdigest() != row[
                "waveform_sha256"
            ]:
                return {"passed": False, "reason": f"sai hash waveform {row['file']}"}
    del data
    return {"passed": True, "records": marker["records"]}


def process_shard(
    *,
    shard_path: Path,
    shard_index: int,
    wanted: Mapping[str, WantedRecord],
    chunk_dir: Path,
    delete_after: bool,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    target_sample_rate: int = 16_000,
) -> dict[str, Any]:
    """Nạp một shard (hoặc bỏ qua nếu đã xong); chỉ xóa Parquet sau khi chunk hợp lệ."""

    paths = chunk_paths(chunk_dir, shard_index)
    if paths.done.is_file():
        marker = json.loads(paths.done.read_text(encoding="utf-8"))
        if marker["wanted_sha256"] != _wanted_digest(wanted):
            raise ValueError(
                f"Shard {shard_index} đã nạp với tập mẫu cần lấy khác; không ghi đè."
            )
        if not verify_chunk(chunk_dir, shard_index)["passed"]:
            raise ValueError(f"Chunk shard {shard_index} đã hỏng; cần xử lý thủ công.")
        if delete_after and shard_path.exists():
            shard_path.unlink()
        return {"status": "already_done", **marker}
    marker = ingest_shard(
        shard_path=shard_path,
        shard_index=shard_index,
        wanted=wanted,
        chunk_dir=chunk_dir,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        target_sample_rate=target_sample_rate,
    )
    if not verify_chunk(chunk_dir, shard_index)["passed"]:
        raise ValueError(f"Chunk shard {shard_index} không qua kiểm tra sau khi ghi.")
    if delete_after:
        shard_path.unlink()
    return {"status": "ingested", **marker}
