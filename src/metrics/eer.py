"""Dependency-free binary detection metrics.

Scores are assumed to increase with confidence in the positive class.  For the
anti-spoofing experiments the positive class is ``spoof`` by convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EERResult:
    """Equal-error-rate estimate and its interpolated operating point."""

    eer: float
    threshold: float
    false_accept_rate: float
    false_reject_rate: float
    positives: int
    negatives: int


def _validated_pairs(
    labels: Iterable[int | bool], scores: Iterable[float]
) -> list[tuple[bool, float]]:
    label_values = list(labels)
    score_values = list(scores)
    if len(label_values) != len(score_values):
        raise ValueError("labels và scores phải có cùng số phần tử.")
    if not label_values:
        raise ValueError("Không thể tính EER trên tập rỗng.")

    pairs: list[tuple[bool, float]] = []
    for index, (label, raw_score) in enumerate(zip(label_values, score_values)):
        if label not in (0, 1, False, True):
            raise ValueError(f"Nhãn tại vị trí {index} phải là 0 hoặc 1.")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"Score tại vị trí {index} không hữu hạn.")
        pairs.append((bool(label), score))

    positives = sum(label for label, _ in pairs)
    if positives == 0 or positives == len(pairs):
        raise ValueError("EER cần có cả mẫu dương và mẫu âm.")
    return pairs


def rates_at_threshold(
    labels: Iterable[int | bool],
    scores: Iterable[float],
    threshold: float,
) -> tuple[float, float]:
    """Return false-accept and false-reject rates for ``score >= threshold``."""

    pairs = _validated_pairs(labels, scores)
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    false_accepts = sum(
        (not label) and score >= threshold for label, score in pairs
    )
    false_rejects = sum(label and score < threshold for label, score in pairs)
    return false_accepts / negatives, false_rejects / positives


def equal_error_rate(
    labels: Sequence[int | bool] | Iterable[int | bool],
    scores: Sequence[float] | Iterable[float],
) -> EERResult:
    """Compute EER with linear interpolation between adjacent ROC points.

    ``1`` means spoof and larger scores must indicate stronger spoof evidence.
    Tied scores are processed as a group so the result does not depend on input
    ordering.
    """

    pairs = _validated_pairs(labels, scores)
    pairs.sort(key=lambda item: item[1], reverse=True)
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives

    max_score = pairs[0][1]
    points: list[tuple[float, float, float]] = [
        (math.nextafter(max_score, math.inf), 0.0, 1.0)
    ]
    true_positives = 0
    false_positives = 0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][1]
        while index < len(pairs) and pairs[index][1] == threshold:
            if pairs[index][0]:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        false_accept_rate = false_positives / negatives
        false_reject_rate = (positives - true_positives) / positives
        points.append((threshold, false_accept_rate, false_reject_rate))

    for point in points:
        if math.isclose(point[1], point[2], abs_tol=1e-15):
            return EERResult(
                eer=(point[1] + point[2]) / 2.0,
                threshold=point[0],
                false_accept_rate=point[1],
                false_reject_rate=point[2],
                positives=positives,
                negatives=negatives,
            )

    for left, right in zip(points, points[1:]):
        left_delta = left[1] - left[2]
        right_delta = right[1] - right[2]
        if left_delta <= 0.0 <= right_delta:
            denominator = right_delta - left_delta
            weight = 0.0 if denominator == 0.0 else -left_delta / denominator
            threshold = left[0] + weight * (right[0] - left[0])
            false_accept_rate = left[1] + weight * (right[1] - left[1])
            false_reject_rate = left[2] + weight * (right[2] - left[2])
            return EERResult(
                eer=(false_accept_rate + false_reject_rate) / 2.0,
                threshold=threshold,
                false_accept_rate=false_accept_rate,
                false_reject_rate=false_reject_rate,
                positives=positives,
                negatives=negatives,
            )

    raise RuntimeError("Không tìm thấy giao điểm EER; dữ liệu đầu vào không hợp lệ.")
