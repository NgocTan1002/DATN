"""Stable content hashes for decoded VSASV waveforms."""

from __future__ import annotations

import array
import hashlib
import sys
from collections.abc import Iterable


def waveform_sha256(values: Iterable[float], sampling_rate: int) -> str:
    """Hash one sample rate and all float64 samples using little-endian bytes."""

    samples = array.array("d", values)
    if sys.byteorder != "little":
        samples.byteswap()
    digest = hashlib.sha256()
    digest.update(int(sampling_rate).to_bytes(8, "little", signed=True))
    digest.update(samples.tobytes())
    return digest.hexdigest()
