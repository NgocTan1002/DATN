"""Deterministic fixed-length waveform segmentation without ML dependencies."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T", int, float)


def fixed_length_segment(
    waveform: Sequence[T],
    target_samples: int,
    *,
    training: bool,
    rng: random.Random | None = None,
) -> list[T]:
    """Crop or repeat a mono waveform to an exact number of samples.

    Training uses a random crop for long utterances. Evaluation uses a center
    crop so repeated runs are identical. Short utterances are repeated and
    trimmed; an empty waveform is rejected instead of silently padding it.
    """

    if target_samples <= 0:
        raise ValueError("target_samples phải lớn hơn 0.")
    length = len(waveform)
    if length == 0:
        raise ValueError("Waveform rỗng không thể chia đoạn.")
    if length == target_samples:
        return list(waveform)
    if length > target_samples:
        maximum_start = length - target_samples
        if training:
            generator = rng if rng is not None else random
            start = generator.randint(0, maximum_start)
        else:
            start = maximum_start // 2
        return list(waveform[start : start + target_samples])

    repeats, remainder = divmod(target_samples, length)
    return list(waveform) * repeats + list(waveform[:remainder])
