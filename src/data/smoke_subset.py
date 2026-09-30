"""Deterministic selection and validation for local smoke manifests."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable


SPOOF_TYPE_ORDER = ("voice_conversion", "adversarial_attack", "replay")


@dataclass(frozen=True)
class SmokeRow:
    file: str
    speaker_id: str
    utt_type: str

    @property
    def binary_label(self) -> int:
        return 0 if self.utt_type == "bonafide" else 1

    def as_csv_row(self) -> dict[str, str]:
        return {
            "file": self.file,
            "label": self.speaker_id,
            "utt_type": self.utt_type,
        }


def _rank(row: SmokeRow, seed: int, split_name: str, purpose: str) -> tuple[bytes, str]:
    material = f"{seed}\0{split_name}\0{purpose}\0{row.file}".encode("utf-8")
    return hashlib.sha256(material).digest(), row.file


def select_smoke_rows(
    rows: Iterable[SmokeRow],
    *,
    split_name: str,
    per_class: int,
    seed: int,
) -> list[SmokeRow]:
    """Select a balanced subset and spread spoof rows across available types."""

    if per_class <= 0:
        raise ValueError("per_class phải lớn hơn 0.")
    candidates = list(rows)
    if len({row.file for row in candidates}) != len(candidates):
        raise ValueError(f"Ứng viên {split_name} có file bị lặp.")

    bonafide = sorted(
        (row for row in candidates if row.binary_label == 0),
        key=lambda row: _rank(row, seed, split_name, "bonafide"),
    )
    if len(bonafide) < per_class:
        raise ValueError(
            f"{split_name} chỉ có {len(bonafide)} bonafide, cần {per_class}."
        )

    spoof_by_type: dict[str, deque[SmokeRow]] = {}
    for utt_type in SPOOF_TYPE_ORDER:
        ordered = sorted(
            (
                row
                for row in candidates
                if row.binary_label == 1 and row.utt_type == utt_type
            ),
            key=lambda row: _rank(row, seed, split_name, utt_type),
        )
        if ordered:
            spoof_by_type[utt_type] = deque(ordered)

    available_spoof = sum(len(group) for group in spoof_by_type.values())
    if available_spoof < per_class:
        raise ValueError(
            f"{split_name} chỉ có {available_spoof} spoof, cần {per_class}."
        )

    selected_spoof: list[SmokeRow] = []
    active_types = [name for name in SPOOF_TYPE_ORDER if name in spoof_by_type]
    while len(selected_spoof) < per_class:
        made_progress = False
        for utt_type in active_types:
            group = spoof_by_type[utt_type]
            if group and len(selected_spoof) < per_class:
                selected_spoof.append(group.popleft())
                made_progress = True
        if not made_progress:
            raise RuntimeError("Không thể hoàn tất quota spoof dù số lượng đã được kiểm tra.")

    selected = bonafide[:per_class] + selected_spoof
    return sorted(
        selected,
        key=lambda row: _rank(row, seed, split_name, "manifest"),
    )


def validate_smoke_partitions(
    partitions: dict[str, list[SmokeRow]],
) -> dict[str, object]:
    """Validate balance, duplicate files and cross-partition speaker isolation."""

    all_files: set[str] = set()
    speakers_by_split: dict[str, set[str]] = {}
    for split_name, rows in partitions.items():
        counts = Counter(row.binary_label for row in rows)
        if counts[0] != counts[1] or counts[0] == 0:
            raise ValueError(
                f"{split_name} không cân bằng nhãn: bonafide={counts[0]}, spoof={counts[1]}."
            )
        files = {row.file for row in rows}
        if len(files) != len(rows):
            raise ValueError(f"{split_name} có file bị lặp.")
        overlap = all_files & files
        if overlap:
            raise ValueError(f"File xuất hiện ở nhiều partition: {min(overlap)}")
        all_files.update(files)
        speakers_by_split[split_name] = {row.speaker_id for row in rows}

    split_names = sorted(partitions)
    speaker_overlaps: dict[str, int] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = speakers_by_split[left] & speakers_by_split[right]
            key = f"{left}_vs_{right}"
            speaker_overlaps[key] = len(overlap)
            if overlap:
                raise ValueError(
                    f"Speaker overlap giữa {left} và {right}: {min(overlap)}"
                )

    return {
        "balanced_within_each_partition": True,
        "duplicate_files": 0,
        "speaker_overlaps": speaker_overlaps,
        "total_samples": len(all_files),
    }


def summarize_smoke_rows(rows: Iterable[SmokeRow]) -> dict[str, object]:
    materialized = list(rows)
    type_counts = Counter(row.utt_type for row in materialized)
    return {
        "samples": len(materialized),
        "speakers": len({row.speaker_id for row in materialized}),
        "bonafide": sum(row.binary_label == 0 for row in materialized),
        "spoof": sum(row.binary_label == 1 for row in materialized),
        "utt_type_counts": {
            name: type_counts.get(name, 0)
            for name in ("bonafide", *SPOOF_TYPE_ORDER)
        },
    }
