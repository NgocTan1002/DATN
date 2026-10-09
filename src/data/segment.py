"""Deterministic fixed-length waveform segmentation for lists and tensors."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TypeVar, overload

import torch


T = TypeVar("T", int, float)


@overload
def fixed_length_segment(
    waveform: torch.Tensor,
    target_samples: int,
    *,
    training: bool,
    rng: random.Random | torch.Generator | None = None,
) -> torch.Tensor: ...


@overload
def fixed_length_segment(
    waveform: Sequence[T],
    target_samples: int,
    *,
    training: bool,
    rng: random.Random | torch.Generator | None = None,
) -> list[T]: ...


def fixed_length_segment(
    waveform: Sequence[T] | torch.Tensor,
    target_samples: int,
    *,
    training: bool,
    rng: random.Random | torch.Generator | None = None,
) -> list[T] | torch.Tensor:
    """Crop or repeat a mono waveform to an exact number of samples.

    Training uses a random crop for long utterances. Evaluation uses a center
    crop so repeated runs are identical. Short utterances are repeated and
    trimmed; an empty waveform is rejected instead of silently padding it.
    """

    if target_samples <= 0:
        raise ValueError("target_samples phải lớn hơn 0.")
    is_tensor = isinstance(waveform, torch.Tensor)
    length = waveform.numel() if is_tensor else len(waveform)
    if length == 0:
        raise ValueError("Waveform rỗng không thể chia đoạn.")
    if length == target_samples:
        return waveform if is_tensor else list(waveform)
    if length > target_samples:
        maximum_start = length - target_samples
        if training:
            if isinstance(rng, torch.Generator):
                start = int(
                    torch.randint(
                        maximum_start + 1,
                        size=(1,),
                        generator=rng,
                    ).item()
                )
            elif rng is None:
                start = random.randint(0, maximum_start)
            else:
                start = rng.randint(0, maximum_start)
        else:
            start = maximum_start // 2
        segment = waveform[start : start + target_samples]
        return segment if is_tensor else list(segment)

    if is_tensor:
        repeats = math.ceil(target_samples / length)
        return waveform.repeat(repeats)[:target_samples]

    repeats, remainder = divmod(target_samples, length)
    return list(waveform) * repeats + list(waveform[:remainder])
