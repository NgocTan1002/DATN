"""Packed waveform artifact built from a verified development manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .amplitude import (
    AmplitudePolicy,
    VALID_AMPLITUDE_POLICIES,
    apply_amplitude_policy,
)
from .segment import fixed_length_segment
from .vsasv import ManifestSplit, VSASVManifestDataset


PACKED_FORMAT_VERSION = "vsasv-packed-v1"
PACKED_PREPROCESSING_STAGE = "post_resample_pre_amplitude_pre_crop"
PACKED_INDEX_COLUMNS = (
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


@dataclass(frozen=True)
class PackedArtifactSummary:
    """Small, serializable summary of one completed artifact build."""

    artifact_dir: Path
    artifact_count: int
    total_samples: int
    source_manifest_sha256: str
    index_sha256: str
    waveforms_sha256: str


@dataclass(frozen=True)
class PackedArtifactVerification:
    """Result of independently hashing every waveform in an artifact."""

    artifact_count: int
    total_samples: int
    waveforms_sha256: str


@dataclass(frozen=True)
class PackedRecord:
    """One logical waveform slice and its source-manifest identity."""

    source_index: int
    offset_samples: int
    num_samples: int
    waveform_sha256: str
    file: str
    speaker_id: str
    split: ManifestSplit
    binary_label: int
    utt_type: str
    native_sample_rate: int
    shard: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_indices(indices: Sequence[int], record_count: int) -> list[int]:
    selected = list(indices)
    if not selected:
        raise ValueError("indices phải chứa ít nhất một vị trí trong split.")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in selected):
        raise TypeError("Mọi phần tử của indices phải là số nguyên.")
    if len(set(selected)) != len(selected):
        raise ValueError("indices không được chứa vị trí trùng lặp.")
    for index in selected:
        if index < 0 or index >= record_count:
            raise IndexError(
                f"Vị trí split nằm ngoài phạm vi: {index}; số mẫu={record_count}."
            )
    return selected


def build_packed_artifact(
    manifest_csv: str | Path,
    parquet_dir: str | Path,
    output_dir: str | Path,
    *,
    split: ManifestSplit,
    indices: Sequence[int],
    target_sample_rate: int = 16_000,
    seed: int = 2026,
) -> PackedArtifactSummary:
    """Create a deterministic packed subset before amplitude policy and crop.

    ``indices`` address the selected split exactly as ``VSASVManifestDataset``
    does. Their original values are retained so deterministic training crops
    remain identical even when the artifact is reordered or contains a subset.
    """

    manifest_path = Path(manifest_csv).resolve()
    parquet_path = Path(parquet_dir).resolve()
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(
            f"Thư mục artifact đã tồn tại; không ghi đè dữ liệu: {target}"
        )
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate phải lớn hơn 0.")
    if seed < 0:
        raise ValueError("seed không được âm.")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent)
    )
    source: VSASVManifestDataset | None = None
    try:
        source = VSASVManifestDataset(
            manifest_path,
            parquet_path,
            split=split,
            training=False,
            target_sample_rate=target_sample_rate,
            seed=seed,
            amplitude_policy="none",
        )
        selected = _validate_indices(indices, len(source))
        data_path = staging / "waveforms.f32"
        index_path = staging / "index.csv"
        index_rows: list[dict[str, str | int]] = []
        offset_samples = 0
        data_digest = hashlib.sha256()

        with data_path.open("wb") as data_handle:
            for artifact_index, source_index in enumerate(selected):
                record = source.records[source_index]
                waveform, native_sample_rate = source._load_audio(record)
                if native_sample_rate != target_sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform,
                        native_sample_rate,
                        target_sample_rate,
                    )
                waveform = waveform.to(torch.float32).contiguous()
                if waveform.ndim != 1 or waveform.numel() == 0:
                    raise ValueError(
                        f"Waveform sau resample không hợp lệ: {record.file}"
                    )
                if not torch.isfinite(waveform).all():
                    raise ValueError(
                        f"Waveform sau resample chứa NaN hoặc Inf: {record.file}"
                    )

                waveform_array = np.asarray(waveform.cpu().numpy(), dtype="<f4")
                waveform_bytes = waveform_array.tobytes(order="C")
                waveform_sha256 = hashlib.sha256(waveform_bytes).hexdigest()
                data_handle.write(waveform_bytes)
                data_digest.update(waveform_bytes)
                num_samples = int(waveform_array.size)
                index_rows.append(
                    {
                        "artifact_index": artifact_index,
                        "source_index": source_index,
                        "offset_samples": offset_samples,
                        "num_samples": num_samples,
                        "waveform_sha256": waveform_sha256,
                        "file": record.file,
                        "speaker_id": record.speaker_id,
                        "split": split,
                        "binary_label": record.binary_label,
                        "utt_type": record.utt_type,
                        "native_sample_rate": native_sample_rate,
                        "shard": record.shard.name,
                    }
                )
                offset_samples += num_samples

        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PACKED_INDEX_COLUMNS)
            writer.writeheader()
            writer.writerows(index_rows)

        source_manifest_sha256 = _sha256_file(manifest_path)
        index_sha256 = _sha256_file(index_path)
        waveforms_sha256 = data_digest.hexdigest()
        metadata = {
            "format_version": PACKED_FORMAT_VERSION,
            "preprocessing_stage": PACKED_PREPROCESSING_STAGE,
            "dtype": "float32-le",
            "target_sample_rate": target_sample_rate,
            "crop_seed": seed,
            "split": split,
            "artifact_count": len(index_rows),
            "total_samples": offset_samples,
            "manifest_version": source.manifest_version,
            "source_snapshot": source.source_snapshot,
            "source_manifest": manifest_path.name,
            "source_manifest_sha256": source_manifest_sha256,
            "index_file": index_path.name,
            "index_sha256": index_sha256,
            "waveforms_file": data_path.name,
            "waveforms_sha256": waveforms_sha256,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
        return PackedArtifactSummary(
            artifact_dir=target,
            artifact_count=len(index_rows),
            total_samples=offset_samples,
            source_manifest_sha256=source_manifest_sha256,
            index_sha256=index_sha256,
            waveforms_sha256=waveforms_sha256,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if source is not None:
            source.close()


class PackedManifestDataset(Dataset[dict[str, Any]]):
    """Read a packed manifest artifact with the existing sample contract."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        training: bool,
        target_samples: int = 64_000,
        seed: int | None = None,
        amplitude_policy: AmplitudePolicy = "none",
        peak_target: float = 0.95,
        rms_target_dbfs: float = -25.0,
        minimum_input_rms_dbfs: float = -50.0,
    ) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self.training = training
        self.target_samples = target_samples
        self.amplitude_policy = amplitude_policy
        self.peak_target = peak_target
        self.rms_target_dbfs = rms_target_dbfs
        self.minimum_input_rms_dbfs = minimum_input_rms_dbfs
        self.epoch = 0

        if target_samples <= 0:
            raise ValueError("target_samples phải lớn hơn 0.")
        if amplitude_policy not in VALID_AMPLITUDE_POLICIES:
            raise ValueError(f"Chính sách biên độ không hợp lệ: {amplitude_policy}")
        if not math.isfinite(peak_target) or not 0.0 < peak_target <= 1.0:
            raise ValueError("peak_target phải hữu hạn và thuộc khoảng (0, 1].")
        if not math.isfinite(rms_target_dbfs):
            raise ValueError("rms_target_dbfs phải hữu hạn.")
        if not math.isfinite(minimum_input_rms_dbfs):
            raise ValueError("minimum_input_rms_dbfs phải hữu hạn.")
        if minimum_input_rms_dbfs >= rms_target_dbfs:
            raise ValueError(
                "minimum_input_rms_dbfs phải nhỏ hơn rms_target_dbfs."
            )
        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục artifact: {self.artifact_dir}"
            )

        metadata_path = self.artifact_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy metadata artifact: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"Không đọc được metadata artifact: {metadata_path}") from error
        if metadata.get("format_version") != PACKED_FORMAT_VERSION:
            raise ValueError(
                "Phiên bản artifact không được hỗ trợ: "
                f"{metadata.get('format_version')}"
            )
        if metadata.get("preprocessing_stage") != PACKED_PREPROCESSING_STAGE:
            raise ValueError("Artifact không nằm ở bước tiền xử lý đã khóa.")
        if metadata.get("dtype") != "float32-le":
            raise ValueError("Artifact phải dùng dtype float32 little-endian.")

        try:
            self.target_sample_rate = int(metadata["target_sample_rate"])
            artifact_seed = int(metadata["crop_seed"])
            expected_count = int(metadata["artifact_count"])
            expected_total_samples = int(metadata["total_samples"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Metadata artifact thiếu trường số hợp lệ.") from error
        if self.target_sample_rate <= 0 or expected_count <= 0:
            raise ValueError("Metadata artifact có số lượng hoặc sample rate không hợp lệ.")
        if expected_total_samples <= 0:
            raise ValueError("Metadata artifact có tổng số mẫu không hợp lệ.")
        self.seed = artifact_seed if seed is None else seed
        if self.seed < 0:
            raise ValueError("seed không được âm.")
        self.manifest_version = str(metadata.get("manifest_version", ""))
        self.source_snapshot = str(metadata.get("source_snapshot", ""))
        self.source_manifest_sha256 = str(
            metadata.get("source_manifest_sha256", "")
        )
        self.waveforms_sha256 = str(metadata.get("waveforms_sha256", ""))
        if len(self.source_manifest_sha256) != 64:
            raise ValueError("Checksum manifest nguồn trong metadata không hợp lệ.")

        index_path = self.artifact_dir / str(metadata.get("index_file", ""))
        data_path = self.artifact_dir / str(metadata.get("waveforms_file", ""))
        if not index_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy chỉ mục artifact: {index_path}")
        if not data_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy waveform artifact: {data_path}")
        if _sha256_file(index_path) != metadata.get("index_sha256"):
            raise ValueError("Checksum chỉ mục artifact không khớp metadata.")

        self.records = self._read_index(index_path)
        if len(self.records) != expected_count:
            raise ValueError(
                "Số hàng chỉ mục không khớp metadata: "
                f"nhận {len(self.records)}, cần {expected_count}."
            )
        expected_bytes = expected_total_samples * np.dtype("<f4").itemsize
        actual_bytes = data_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                "Sai kích thước waveform artifact: "
                f"nhận {actual_bytes} byte, cần {expected_bytes} byte."
            )
        if self.records[-1].offset_samples + self.records[-1].num_samples != (
            expected_total_samples
        ):
            raise ValueError("Chỉ mục không phủ đúng tổng số mẫu waveform.")
        if len(self.waveforms_sha256) != 64:
            raise ValueError("Checksum waveform tổng trong metadata không hợp lệ.")
        self._data_path = data_path
        self._waveforms = np.memmap(self._data_path, dtype="<f4", mode="c")

    @staticmethod
    def _read_index(index_path: Path) -> list[PackedRecord]:
        records: list[PackedRecord] = []
        seen_files: set[str] = set()
        seen_source_indices: set[int] = set()
        expected_offset = 0
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PACKED_INDEX_COLUMNS:
                raise ValueError(
                    "Schema chỉ mục artifact không hợp lệ; cần đúng các cột "
                    f"{list(PACKED_INDEX_COLUMNS)}."
                )
            for row_number, row in enumerate(reader):
                try:
                    artifact_index = int(row["artifact_index"])
                    source_index = int(row["source_index"])
                    offset_samples = int(row["offset_samples"])
                    num_samples = int(row["num_samples"])
                    binary_label = int(row["binary_label"])
                    native_sample_rate = int(row["native_sample_rate"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Chỉ mục artifact có trường số sai ở hàng {row_number + 2}."
                    ) from error
                if artifact_index != row_number:
                    raise ValueError("artifact_index phải liên tiếp từ 0.")
                if source_index < 0 or source_index in seen_source_indices:
                    raise ValueError("source_index phải không âm và không trùng lặp.")
                if offset_samples != expected_offset or num_samples <= 0:
                    raise ValueError("Offset/độ dài waveform trong chỉ mục không hợp lệ.")
                if row["file"] in seen_files:
                    raise ValueError(f"Chỉ mục artifact bị lặp file: {row['file']}")
                if binary_label not in {0, 1}:
                    raise ValueError("binary_label trong artifact phải là 0 hoặc 1.")
                if native_sample_rate <= 0:
                    raise ValueError("native_sample_rate trong artifact phải lớn hơn 0.")
                waveform_sha256 = row["waveform_sha256"]
                if len(waveform_sha256) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in waveform_sha256
                ):
                    raise ValueError("waveform_sha256 trong chỉ mục không hợp lệ.")
                split_text = row["split"]
                if split_text not in {"closed_train", "closed_dev"}:
                    raise ValueError(f"Split artifact không hợp lệ: {split_text}")
                split = cast(ManifestSplit, split_text)
                record = PackedRecord(
                    source_index=source_index,
                    offset_samples=offset_samples,
                    num_samples=num_samples,
                    waveform_sha256=waveform_sha256,
                    file=row["file"],
                    speaker_id=row["speaker_id"],
                    split=split,
                    binary_label=binary_label,
                    utt_type=row["utt_type"],
                    native_sample_rate=native_sample_rate,
                    shard=row["shard"],
                )
                records.append(record)
                seen_files.add(record.file)
                seen_source_indices.add(source_index)
                expected_offset += num_samples
        if not records:
            raise ValueError("Chỉ mục artifact không được rỗng.")
        return records

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic crop stream for a training epoch."""

        if epoch < 0:
            raise ValueError("epoch không được âm.")
        self.epoch = epoch

    def _crop_seed(self, source_index: int) -> int:
        material = f"{self.seed}:{self.epoch}:{source_index}".encode("ascii")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        start = record.offset_samples
        stop = start + record.num_samples
        waveform = torch.from_numpy(np.asarray(self._waveforms[start:stop]))
        if waveform.shape != (record.num_samples,):
            raise RuntimeError(f"Không đọc đủ waveform cho {record.file}.")
        amplitude_result = apply_amplitude_policy(
            waveform,
            self.amplitude_policy,
            peak_target=self.peak_target,
            rms_target_dbfs=self.rms_target_dbfs,
            minimum_input_rms_dbfs=self.minimum_input_rms_dbfs,
        )
        crop_generator = (
            torch.Generator().manual_seed(self._crop_seed(record.source_index))
            if self.training
            else None
        )
        prepared = fixed_length_segment(
            amplitude_result.waveform,
            self.target_samples,
            training=self.training,
            rng=crop_generator,
        ).contiguous()
        if prepared.shape != (self.target_samples,):
            raise RuntimeError(
                f"Tiền xử lý trả shape sai cho {record.file}: {prepared.shape}"
            )
        if not torch.isfinite(prepared).all():
            raise ValueError(f"Waveform sau tiền xử lý không hữu hạn: {record.file}")
        return {
            "waveform": prepared,
            "label": torch.tensor(record.binary_label, dtype=torch.float32),
            "file": record.file,
            "speaker_id": record.speaker_id,
            "utt_type": record.utt_type,
            "native_sample_rate": record.native_sample_rate,
            "sample_rate": self.target_sample_rate,
            "amplitude_policy": self.amplitude_policy,
            "input_peak": amplitude_result.input_peak,
            "output_peak": amplitude_result.output_peak,
            "input_rms": amplitude_result.input_rms,
            "output_rms": amplitude_result.output_rms,
            "input_rms_dbfs": amplitude_result.input_rms_dbfs,
            "output_rms_dbfs": amplitude_result.output_rms_dbfs,
            "applied_gain": amplitude_result.applied_gain,
            "near_silence": amplitude_result.near_silence,
            "peak_limited": amplitude_result.peak_limited,
        }

    def close(self) -> None:
        """Release the memory-mapped file handle."""

        waveforms = getattr(self, "_waveforms", None)
        memory_map = getattr(waveforms, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
        self._waveforms = None  # type: ignore[assignment]

    def __getstate__(self) -> dict[str, Any]:
        """Reopen the memory map inside DataLoader worker processes."""

        state = self.__dict__.copy()
        state["_waveforms"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the memory map after DataLoader worker unpickling."""

        self.__dict__.update(state)
        self._waveforms = np.memmap(self._data_path, dtype="<f4", mode="c")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def verify_packed_artifact(
    artifact_dir: str | Path,
) -> PackedArtifactVerification:
    """Hash every stored waveform and compare row and whole-file checksums."""

    dataset = PackedManifestDataset(
        artifact_dir,
        training=False,
        target_samples=1,
    )
    total_digest = hashlib.sha256()
    total_samples = 0
    try:
        for record in dataset.records:
            start = record.offset_samples
            stop = start + record.num_samples
            waveform_bytes = np.asarray(
                dataset._waveforms[start:stop], dtype="<f4"
            ).tobytes(order="C")
            actual_sha256 = hashlib.sha256(waveform_bytes).hexdigest()
            if actual_sha256 != record.waveform_sha256:
                raise ValueError(
                    f"Checksum waveform không khớp cho {record.file}."
                )
            total_digest.update(waveform_bytes)
            total_samples += record.num_samples
        waveforms_sha256 = total_digest.hexdigest()
        if waveforms_sha256 != dataset.waveforms_sha256:
            raise ValueError("Checksum waveform tổng không khớp metadata.")
        return PackedArtifactVerification(
            artifact_count=len(dataset),
            total_samples=total_samples,
            waveforms_sha256=waveforms_sha256,
        )
    finally:
        dataset.close()
