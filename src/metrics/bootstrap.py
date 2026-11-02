"""Khoảng tin cậy bootstrap theo speaker và các bảng phân rã cho đánh giá nhị phân.

Quy ước giống `src.metrics.eer`: `1 = spoof`, score lớn hơn nghĩa là bằng chứng
spoof mạnh hơn. Bootstrap lấy mẫu lại theo cụm speaker (có hoàn lại) vì các mẫu của
cùng một speaker không độc lập; CI này đo dao động do dữ liệu đánh giá, không đo dao
động do seed huấn luyện.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .eer import _validated_pairs, equal_error_rate, rates_at_threshold


@dataclass(frozen=True)
class Interval:
    """Ước lượng điểm trên toàn dữ liệu và khoảng percentile bootstrap."""

    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class SpeakerBootstrapResult:
    eer: Interval
    roc_auc: Interval
    confidence: float
    seed: int
    requested_resamples: int
    valid_resamples: int
    speakers: int
    samples: int


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Hiệu EER(a) - EER(b) trên cùng các lần lấy mẫu; âm nghĩa là a tốt hơn."""

    estimate: float
    ci_low: float
    ci_high: float
    fraction_a_lower: float
    confidence: float
    seed: int
    requested_resamples: int
    valid_resamples: int


def fast_eer(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """EER và threshold theo đúng thuật toán của `equal_error_rate`, dùng numpy."""

    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("EER cần có cả mẫu dương và mẫu âm.")
    order = np.argsort(-s, kind="mergesort")
    s = s[order]
    y = y[order]
    last_of_group = np.r_[s[1:] != s[:-1], True]
    true_positives = np.cumsum(y)[last_of_group]
    false_positives = np.cumsum(~y)[last_of_group]
    thresholds = np.r_[math.nextafter(float(s[0]), math.inf), s[last_of_group]]
    far = np.r_[0.0, false_positives / negatives]
    frr = np.r_[1.0, (positives - true_positives) / positives]
    tolerance = np.maximum(1e-9 * np.maximum(np.abs(far), np.abs(frr)), 1e-15)
    equal = np.abs(far - frr) <= tolerance
    if equal.any():
        k = int(np.argmax(equal))
        return float((far[k] + frr[k]) / 2.0), float(thresholds[k])
    delta = far - frr
    crossing = (delta[:-1] <= 0.0) & (delta[1:] >= 0.0)
    if not crossing.any():
        raise RuntimeError("Không tìm thấy giao điểm EER; dữ liệu đầu vào không hợp lệ.")
    k = int(np.argmax(crossing))
    denominator = delta[k + 1] - delta[k]
    weight = 0.0 if denominator == 0.0 else -delta[k] / denominator
    far_k = far[k] + weight * (far[k + 1] - far[k])
    frr_k = frr[k] + weight * (frr[k + 1] - frr[k])
    threshold = thresholds[k] + weight * (thresholds[k + 1] - thresholds[k])
    return float((far_k + frr_k) / 2.0), float(threshold)


def fast_roc_auc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """ROC-AUC theo hạng trung bình khi hòa score, dùng numpy."""

    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC cần có cả mẫu dương và mẫu âm.")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    positive_rank_sum = float(average_rank[inverse][y].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _validate(
    labels: Sequence[int],
    scores: Sequence[float],
    speakers: Sequence[str],
    n_resamples: int,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    if not (len(labels) == len(scores) == len(speakers)):
        raise ValueError("labels, scores và speakers phải có cùng số phần tử.")
    pairs = _validated_pairs(labels, scores)
    if any(not str(speaker) for speaker in speakers):
        raise ValueError("speaker_id không được rỗng.")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples phải là số nguyên dương.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence phải thuộc khoảng (0, 1).")
    y = np.array([label for label, _ in pairs], dtype=bool)
    s = np.array([score for _, score in pairs], dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for index, speaker in enumerate(speakers):
        groups.setdefault(str(speaker), []).append(index)
    clusters = [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]
    return y, s, clusters


def _resample_indices(
    rng: np.random.Generator, clusters: list[np.ndarray]
) -> np.ndarray:
    chosen = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in chosen])


def _interval(estimate: float, values: list[float], confidence: float) -> Interval:
    alpha = 1.0 - confidence
    low, high = np.quantile(np.asarray(values), [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(estimate=float(estimate), ci_low=float(low), ci_high=float(high))


def speaker_bootstrap(
    labels: Sequence[int],
    scores: Sequence[float],
    speakers: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> SpeakerBootstrapResult:
    """CI percentile cho EER và ROC-AUC, lấy mẫu lại cụm speaker có hoàn lại.

    Lần lấy mẫu chỉ có một lớp bị bỏ qua và được đếm qua `valid_resamples`.
    """

    y, s, clusters = _validate(labels, scores, speakers, n_resamples, confidence)
    eer_estimate = equal_error_rate(y.astype(int).tolist(), s.tolist()).eer
    auc_estimate = fast_roc_auc(y, s)
    rng = np.random.default_rng(seed)
    eers: list[float] = []
    aucs: list[float] = []
    for _ in range(n_resamples):
        index = _resample_indices(rng, clusters)
        sample_y = y[index]
        if sample_y.all() or not sample_y.any():
            continue
        eers.append(fast_eer(sample_y, s[index])[0])
        aucs.append(fast_roc_auc(sample_y, s[index]))
    if not eers:
        raise ValueError("Không có lần lấy mẫu nào chứa đủ cả hai lớp.")
    return SpeakerBootstrapResult(
        eer=_interval(eer_estimate, eers, confidence),
        roc_auc=_interval(auc_estimate, aucs, confidence),
        confidence=confidence,
        seed=seed,
        requested_resamples=n_resamples,
        valid_resamples=len(eers),
        speakers=len(clusters),
        samples=int(y.size),
    )


def paired_speaker_bootstrap(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    speakers: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> PairedBootstrapResult:
    """CI cho hiệu EER(a) - EER(b) khi hai hệ thống chấm cùng một tập mẫu."""

    y, s_a, clusters = _validate(labels, scores_a, speakers, n_resamples, confidence)
    _, s_b, _ = _validate(labels, scores_b, speakers, n_resamples, confidence)
    estimate = (
        equal_error_rate(y.astype(int).tolist(), s_a.tolist()).eer
        - equal_error_rate(y.astype(int).tolist(), s_b.tolist()).eer
    )
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(n_resamples):
        index = _resample_indices(rng, clusters)
        sample_y = y[index]
        if sample_y.all() or not sample_y.any():
            continue
        differences.append(fast_eer(sample_y, s_a[index])[0] - fast_eer(sample_y, s_b[index])[0])
    if not differences:
        raise ValueError("Không có lần lấy mẫu nào chứa đủ cả hai lớp.")
    interval = _interval(estimate, differences, confidence)
    return PairedBootstrapResult(
        estimate=float(estimate),
        ci_low=interval.ci_low,
        ci_high=interval.ci_high,
        fraction_a_lower=sum(value < 0.0 for value in differences) / len(differences),
        confidence=confidence,
        seed=seed,
        requested_resamples=n_resamples,
        valid_resamples=len(differences),
    )


def per_utt_type_eer(
    labels: Sequence[int],
    scores: Sequence[float],
    utt_types: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """EER của toàn bộ bonafide so với từng loại tấn công riêng."""

    if not (len(labels) == len(scores) == len(utt_types)):
        raise ValueError("labels, scores và utt_types phải có cùng số phần tử.")
    bonafide = [i for i, utt in enumerate(utt_types) if utt == "bonafide"]
    if not bonafide:
        raise ValueError("Cần có mẫu bonafide để tính EER theo utt_type.")
    if any(int(labels[i]) != 0 for i in bonafide):
        raise ValueError("Mẫu bonafide phải có nhãn 0.")
    result: dict[str, dict[str, Any]] = {}
    for attack in sorted({utt for utt in utt_types if utt != "bonafide"}):
        spoof = [i for i, utt in enumerate(utt_types) if utt == attack]
        if any(int(labels[i]) != 1 for i in spoof):
            raise ValueError(f"Mẫu {attack} phải có nhãn 1.")
        subset = bonafide + spoof
        eer = equal_error_rate([labels[i] for i in subset], [scores[i] for i in subset])
        result[attack] = {
            "eer": eer.eer,
            "threshold": eer.threshold,
            "bonafide_samples": len(bonafide),
            "spoof_samples": len(spoof),
        }
    return result


def confusion_at_threshold(
    labels: Sequence[int], scores: Sequence[float], *, threshold: float
) -> dict[str, float | int]:
    """Ma trận nhầm lẫn với quy tắc `score >= threshold` là spoof."""

    if not math.isfinite(threshold):
        raise ValueError("Threshold phải là số hữu hạn.")
    pairs = _validated_pairs(labels, scores)
    tp = sum(label and score >= threshold for label, score in pairs)
    fn = sum(label and score < threshold for label, score in pairs)
    fp = sum((not label) and score >= threshold for label, score in pairs)
    tn = sum((not label) and score < threshold for label, score in pairs)
    far, frr = rates_at_threshold(labels, scores, threshold)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "fpr": far, "fnr": frr}
