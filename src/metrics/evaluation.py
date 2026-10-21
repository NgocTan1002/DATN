"""Hợp đồng tổng hợp metric cho các lần đánh giá nhị phân."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .classification import macro_f1_at_threshold, roc_auc_score
from .eer import equal_error_rate


ThresholdSource = Literal["development_eer", "provided"]


@dataclass(frozen=True)
class BinaryEvaluationResult:
    """Metric và đầu vào cần thiết để tái kiểm một lần đánh giá."""

    labels: tuple[int, ...]
    scores: tuple[float, ...]
    loss: float
    eer: float
    eer_threshold: float
    roc_auc: float
    macro_f1: float
    decision_threshold: float
    threshold_source: ThresholdSource
    sample_count: int
    utt_type_counts: dict[str, int]


def evaluate_development_scores(
    labels: Iterable[int | bool],
    scores: Iterable[float],
    *,
    loss: float,
    utt_types: Iterable[str],
) -> BinaryEvaluationResult:
    """Tổng hợp development metric và khóa threshold tại điểm EER."""

    label_values = tuple(labels)
    score_values = tuple(scores)
    utt_type_values = tuple(utt_types)
    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("Development loss phải là số hữu hạn không âm.")
    if len(utt_type_values) != len(label_values):
        raise ValueError("utt_types và labels phải có cùng số phần tử.")
    if any(not value for value in utt_type_values):
        raise ValueError("utt_type không được rỗng.")

    eer_result = equal_error_rate(label_values, score_values)
    decision_threshold = eer_result.threshold
    return BinaryEvaluationResult(
        labels=tuple(int(value) for value in label_values),
        scores=tuple(float(value) for value in score_values),
        loss=float(loss),
        eer=eer_result.eer,
        eer_threshold=eer_result.threshold,
        roc_auc=roc_auc_score(label_values, score_values),
        macro_f1=macro_f1_at_threshold(
            label_values,
            score_values,
            threshold=decision_threshold,
        ),
        decision_threshold=decision_threshold,
        threshold_source="development_eer",
        sample_count=len(label_values),
        utt_type_counts=dict(sorted(Counter(utt_type_values).items())),
    )


def evaluate_scores_at_threshold(
    labels: Iterable[int | bool],
    scores: Iterable[float],
    *,
    loss: float,
    utt_types: Iterable[str],
    threshold: float,
) -> BinaryEvaluationResult:
    """Tổng hợp metric nhưng giữ nguyên threshold development đã cung cấp."""

    label_values = tuple(labels)
    score_values = tuple(scores)
    utt_type_values = tuple(utt_types)
    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("Loss đánh giá phải là số hữu hạn không âm.")
    if len(utt_type_values) != len(label_values):
        raise ValueError("utt_types và labels phải có cùng số phần tử.")
    if any(not value for value in utt_type_values):
        raise ValueError("utt_type không được rỗng.")

    eer_result = equal_error_rate(label_values, score_values)
    macro_f1 = macro_f1_at_threshold(
        label_values,
        score_values,
        threshold=threshold,
    )
    return BinaryEvaluationResult(
        labels=tuple(int(value) for value in label_values),
        scores=tuple(float(value) for value in score_values),
        loss=float(loss),
        eer=eer_result.eer,
        eer_threshold=eer_result.threshold,
        roc_auc=roc_auc_score(label_values, score_values),
        macro_f1=macro_f1,
        decision_threshold=float(threshold),
        threshold_source="provided",
        sample_count=len(label_values),
        utt_type_counts=dict(sorted(Counter(utt_type_values).items())),
    )
