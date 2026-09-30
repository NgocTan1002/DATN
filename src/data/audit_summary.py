"""Pure aggregation helpers for amplitude audit observations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


AUDIT_NUMERIC_FIELDS = (
    "input_peak",
    "output_peak",
    "input_rms_dbfs",
    "output_rms_dbfs",
    "applied_gain",
)


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated quantile for a non-empty sequence."""

    if not values:
        raise ValueError("Không thể tính quantile của dãy rỗng.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Xác suất quantile phải thuộc khoảng [0, 1].")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Quantile chỉ nhận giá trị hữu hạn.")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": quantile(values, 0.5),
        "mean": sum(values) / len(values),
        "p95": quantile(values, 0.95),
        "maximum": max(values),
    }


def summarize_observations(
    observations: Iterable[Mapping[str, Any]],
    dimensions: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate audit observations by the requested dimensions."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        try:
            key = tuple(observation[dimension] for dimension in dimensions)
            for field in AUDIT_NUMERIC_FIELDS:
                value = float(observation[field])
                if not math.isfinite(value):
                    raise ValueError(f"{field} không hữu hạn: {value}")
        except KeyError as error:
            raise ValueError(f"Thiếu trường audit bắt buộc: {error.args[0]}") from error
        groups[key].append(observation)

    summaries: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        rows = groups[key]
        summary: dict[str, Any] = {
            dimension: value for dimension, value in zip(dimensions, key)
        }
        summary.update(
            {
                "samples": len(rows),
                "near_silence": sum(bool(row["near_silence"]) for row in rows),
                "peak_limited": sum(bool(row["peak_limited"]) for row in rows),
                "statistics": {
                    field: _numeric_summary([float(row[field]) for row in rows])
                    for field in AUDIT_NUMERIC_FIELDS
                },
            }
        )
        summaries.append(summary)
    return summaries


def build_amplitude_summaries(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build every grouping required by audio protocol version 3."""

    if not observations:
        raise ValueError("Không có observation để tổng hợp amplitude audit.")
    return {
        "by_policy": summarize_observations(observations, ("policy",)),
        "by_split_and_policy": summarize_observations(
            observations, ("split", "policy")
        ),
        "by_label_and_policy": summarize_observations(
            observations, ("binary_label", "policy")
        ),
        "by_utt_type_and_policy": summarize_observations(
            observations, ("utt_type", "policy")
        ),
        "by_native_sample_rate_and_policy": summarize_observations(
            observations, ("native_sample_rate", "policy")
        ),
    }
