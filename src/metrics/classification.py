"""Các metric phân loại nhị phân không phụ thuộc thư viện ngoài."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .eer import _validated_pairs


def roc_auc_score(
    labels: Iterable[int | bool], scores: Iterable[float]
) -> float:
    """Tính ROC-AUC khi score lớn hơn biểu thị khả năng spoof cao hơn."""

    pairs = sorted(_validated_pairs(labels, scores), key=lambda item: item[1])
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        group_end = index + 1
        while group_end < len(pairs) and pairs[group_end][1] == pairs[index][1]:
            group_end += 1
        average_rank = ((index + 1) + group_end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for label, _ in pairs[index:group_end]
        )
        index = group_end

    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def macro_f1_at_threshold(
    labels: Iterable[int | bool],
    scores: Iterable[float],
    *,
    threshold: float,
) -> float:
    """Tính macro-F1 với quy tắc ``score >= threshold`` là spoof."""

    if not math.isfinite(threshold):
        raise ValueError("Threshold phải là số hữu hạn.")
    pairs = _validated_pairs(labels, scores)
    class_scores: list[float] = []
    for target in (False, True):
        true_positives = sum(
            label == target and (score >= threshold) == target
            for label, score in pairs
        )
        false_positives = sum(
            label != target and (score >= threshold) == target
            for label, score in pairs
        )
        false_negatives = sum(
            label == target and (score >= threshold) != target
            for label, score in pairs
        )
        denominator = 2 * true_positives + false_positives + false_negatives
        class_scores.append(
            0.0 if denominator == 0 else 2 * true_positives / denominator
        )
    return sum(class_scores) / len(class_scores)
