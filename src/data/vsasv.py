"""Lazy VSASV Parquet dataset with deterministic waveform preprocessing."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import torch
import torchaudio
from torch.utils.data import Dataset

from .amplitude import (
    AmplitudePolicy,
    VALID_AMPLITUDE_POLICIES,
    apply_amplitude_policy,
)


VALID_UTT_TYPES = frozenset(
    {"bonafide", "voice_conversion", "adversarial_attack", "replay"}
)


@dataclass(frozen=True)
class VSASVRecord:
    """Metadata required to retrieve one local VSASV utterance."""

    file: str
    speaker_id: str
    utt_type: str
    shard: Path

    @property
    def binary_label(self) -> int:
        """Return 0 for bonafide and 1 for every spoofing condition."""

        return 0 if self.utt_type == "bonafide" else 1


class VSASVParquetDataset(Dataset[dict[str, Any]]):
    """Read a speaker-disjoint CSV split from local VSASV Parquet shards.

    The split CSV is authoritative for membership and metadata. Only rows with
    audio present in ``parquet_dir`` are exposed. Waveforms are loaded lazily,
    converted to float32, resampled to 16 kHz by default, and cropped or
    repeated to a fixed number of samples.
    """

    def __init__(
        self,
        split_csv: str | Path,
        parquet_dir: str | Path,
        *,
        training: bool,
        target_sample_rate: int = 16_000,
        target_samples: int = 64_000,
        seed: int = 2026,
        amplitude_policy: AmplitudePolicy = "none",
        peak_target: float = 0.95,
        rms_target_dbfs: float = -25.0,
        minimum_input_rms_dbfs: float = -50.0,
    ) -> None:
        self.split_csv = Path(split_csv).resolve()
        self.parquet_dir = Path(parquet_dir).resolve()
        self.training = training
        self.target_sample_rate = target_sample_rate
        self.target_samples = target_samples
        self.seed = seed
        self.amplitude_policy = amplitude_policy
        self.peak_target = peak_target
        self.rms_target_dbfs = rms_target_dbfs
        self.minimum_input_rms_dbfs = minimum_input_rms_dbfs
        self.epoch = 0
        self._connection: duckdb.DuckDBPyConnection | None = None

        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate phải lớn hơn 0.")
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
        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Không tìm thấy split CSV: {self.split_csv}")
        if not self.parquet_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục Parquet: {self.parquet_dir}")

        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"Không tìm thấy shard Parquet trong {self.parquet_dir}"
            )

        local_index = self._build_local_index()
        self.records, self.total_split_rows = self._read_split(local_index)
        if not self.records:
            raise ValueError(
                f"Split {self.split_csv.name} không giao với audio Parquet cục bộ."
            )

    @property
    def local_coverage(self) -> float:
        """Fraction of split rows whose audio is available locally."""

        return len(self.records) / self.total_split_rows

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic random-crop stream for a training epoch."""

        if epoch < 0:
            raise ValueError("epoch không được âm.")
        self.epoch = epoch

    def _build_local_index(self) -> dict[str, tuple[Path, str, str]]:
        parquet_glob = (self.parquet_dir / "*.parquet").as_posix()
        connection = duckdb.connect(database=":memory:")
        try:
            rows = connection.execute(
                """
                SELECT file, label, utt_type, filename
                FROM read_parquet(?, filename = true)
                ORDER BY file
                """,
                [parquet_glob],
            ).fetchall()
        finally:
            connection.close()

        index: dict[str, tuple[Path, str, str]] = {}
        for logical_file, speaker_id, utt_type, shard_name in rows:
            if logical_file in index:
                raise ValueError(f"Audio cục bộ bị lặp file: {logical_file}")
            index[logical_file] = (
                Path(shard_name).resolve(),
                str(speaker_id),
                str(utt_type),
            )
        return index

    def _read_split(
        self, local_index: dict[str, tuple[Path, str, str]]
    ) -> tuple[list[VSASVRecord], int]:
        records: list[VSASVRecord] = []
        seen_files: set[str] = set()

        with self.split_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"file", "label", "utt_type"}
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"Split phải có các cột {sorted(required_columns)}; "
                    f"nhận được {reader.fieldnames}."
                )

            total_rows = 0
            for row in reader:
                total_rows += 1
                logical_file = row["file"]
                speaker_id = row["label"]
                utt_type = row["utt_type"]

                if logical_file in seen_files:
                    raise ValueError(f"Split bị lặp file: {logical_file}")
                seen_files.add(logical_file)
                if utt_type not in VALID_UTT_TYPES:
                    raise ValueError(
                        f"utt_type không hợp lệ cho {logical_file}: {utt_type}"
                    )

                local = local_index.get(logical_file)
                if local is None:
                    continue
                shard, local_speaker, local_type = local
                if (speaker_id, utt_type) != (local_speaker, local_type):
                    raise ValueError(
                        "Metadata split không khớp Parquet cho "
                        f"{logical_file}: split=({speaker_id}, {utt_type}), "
                        f"parquet=({local_speaker}, {local_type})."
                    )
                records.append(
                    VSASVRecord(
                        file=logical_file,
                        speaker_id=speaker_id,
                        utt_type=utt_type,
                        shard=shard,
                    )
                )

        if total_rows == 0:
            raise ValueError(f"Split CSV rỗng: {self.split_csv}")
        return records, total_rows

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            self._connection = duckdb.connect(database=":memory:")
        return self._connection

    def _load_audio(self, record: VSASVRecord) -> tuple[torch.Tensor, int]:
        rows = self._get_connection().execute(
            "SELECT audio FROM read_parquet(?) WHERE file = ?",
            [record.shard.as_posix(), record.file],
        ).fetchall()
        if len(rows) != 1:
            raise RuntimeError(
                f"Cần đúng một waveform cho {record.file}, nhận được {len(rows)}."
            )

        audio = rows[0][0]
        if not isinstance(audio, dict) or not {"array", "sampling_rate"}.issubset(
            audio
        ):
            raise ValueError(f"Cấu trúc audio không hợp lệ cho {record.file}.")

        waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
        sample_rate = int(audio["sampling_rate"])
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError(f"Waveform phải là mono và không rỗng: {record.file}")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate không hợp lệ cho {record.file}: {sample_rate}")
        if not torch.isfinite(waveform).all():
            raise ValueError(f"Waveform chứa NaN hoặc Inf: {record.file}")
        return waveform, sample_rate

    def _crop_seed(self, index: int) -> int:
        material = f"{self.seed}:{self.epoch}:{index}".encode("ascii")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def _fixed_length(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        length = waveform.numel()
        if length == self.target_samples:
            return waveform
        if length > self.target_samples:
            maximum_start = length - self.target_samples
            if self.training:
                generator = torch.Generator().manual_seed(self._crop_seed(index))
                start = int(
                    torch.randint(
                        maximum_start + 1, size=(1,), generator=generator
                    ).item()
                )
            else:
                start = maximum_start // 2
            return waveform[start : start + self.target_samples]

        repeats = math.ceil(self.target_samples / length)
        return waveform.repeat(repeats)[: self.target_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform, native_sample_rate = self._load_audio(record)
        if native_sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, native_sample_rate, self.target_sample_rate
            )
        amplitude_result = apply_amplitude_policy(
            waveform,
            self.amplitude_policy,
            peak_target=self.peak_target,
            rms_target_dbfs=self.rms_target_dbfs,
            minimum_input_rms_dbfs=self.minimum_input_rms_dbfs,
        )
        waveform = amplitude_result.waveform
        waveform = self._fixed_length(waveform, index).contiguous()

        if waveform.shape != (self.target_samples,):
            raise RuntimeError(
                f"Tiền xử lý trả shape sai cho {record.file}: {waveform.shape}"
            )
        if not torch.isfinite(waveform).all():
            raise ValueError(f"Waveform sau tiền xử lý không hữu hạn: {record.file}")

        return {
            "waveform": waveform,
            "label": torch.tensor(record.binary_label, dtype=torch.float32),
            "file": record.file,
            "speaker_id": record.speaker_id,
            "utt_type": record.utt_type,
            "native_sample_rate": native_sample_rate,
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

    def __getstate__(self) -> dict[str, Any]:
        """Drop process-local DuckDB state before DataLoader worker pickling."""

        state = self.__dict__.copy()
        state["_connection"] = None
        return state

    def close(self) -> None:
        """Close the lazy DuckDB connection, if it was opened."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
