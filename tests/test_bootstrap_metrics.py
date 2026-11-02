from __future__ import annotations

import random
import unittest

from src.metrics import equal_error_rate, roc_auc_score
from src.metrics.bootstrap import (
    confusion_at_threshold,
    fast_eer,
    fast_roc_auc,
    paired_speaker_bootstrap,
    per_utt_type_eer,
    speaker_bootstrap,
)


def _synthetic(seed: int, size: int = 400, ties: bool = False):
    rng = random.Random(seed)
    labels = [rng.random() < 0.6 for _ in range(size)]
    scores = []
    for label in labels:
        value = rng.gauss(1.0 if label else 0.0, 1.0)
        scores.append(round(value, 1) if ties else value)
    speakers = [f"spk{rng.randrange(25)}" for _ in range(size)]
    return [int(v) for v in labels], scores, speakers


class FastMetricTests(unittest.TestCase):
    def test_fast_eer_matches_reference_implementation(self) -> None:
        for seed in range(8):
            for ties in (False, True):
                labels, scores, _ = _synthetic(seed, ties=ties)
                reference = equal_error_rate(labels, scores)
                eer, threshold = fast_eer(labels, scores)
                with self.subTest(seed=seed, ties=ties):
                    self.assertAlmostEqual(eer, reference.eer, places=12)
                    self.assertAlmostEqual(threshold, reference.threshold, places=12)

    def test_fast_eer_perfect_separation_is_zero(self) -> None:
        eer, _ = fast_eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(eer, 0.0)

    def test_fast_auc_matches_reference_implementation(self) -> None:
        for seed in range(8):
            for ties in (False, True):
                labels, scores, _ = _synthetic(seed, ties=ties)
                with self.subTest(seed=seed, ties=ties):
                    self.assertAlmostEqual(
                        fast_roc_auc(labels, scores), roc_auc_score(labels, scores), places=12
                    )


class SpeakerBootstrapTests(unittest.TestCase):
    def test_same_seed_is_deterministic_and_seed_changes_result(self) -> None:
        labels, scores, speakers = _synthetic(1)
        first = speaker_bootstrap(labels, scores, speakers, n_resamples=200, seed=2026)
        second = speaker_bootstrap(labels, scores, speakers, n_resamples=200, seed=2026)
        other = speaker_bootstrap(labels, scores, speakers, n_resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertNotEqual(first.eer.ci_low, other.eer.ci_low)

    def test_point_estimates_use_full_data(self) -> None:
        labels, scores, speakers = _synthetic(2)
        result = speaker_bootstrap(labels, scores, speakers, n_resamples=50, seed=2026)
        self.assertAlmostEqual(result.eer.estimate, equal_error_rate(labels, scores).eer, places=12)
        self.assertAlmostEqual(result.roc_auc.estimate, roc_auc_score(labels, scores), places=12)
        self.assertEqual(result.speakers, len(set(speakers)))
        self.assertLessEqual(result.eer.ci_low, result.eer.ci_high)

    def test_perfect_separation_has_degenerate_interval(self) -> None:
        labels = [0, 0, 1, 1, 0, 1]
        scores = [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]
        speakers = ["a", "b", "a", "b", "c", "c"]
        result = speaker_bootstrap(labels, scores, speakers, n_resamples=100, seed=2026)
        self.assertEqual((result.eer.ci_low, result.eer.ci_high), (0.0, 0.0))

    def test_single_class_resamples_are_skipped_and_counted(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.6, 0.4, 0.9]
        speakers = ["only_bonafide", "only_bonafide", "only_spoof", "only_spoof"]
        result = speaker_bootstrap(labels, scores, speakers, n_resamples=200, seed=2026)
        self.assertLess(result.valid_resamples, 200)
        self.assertGreater(result.valid_resamples, 0)

    def test_paired_identical_systems_have_zero_difference(self) -> None:
        labels, scores, speakers = _synthetic(3)
        result = paired_speaker_bootstrap(
            labels, scores, scores, speakers, n_resamples=100, seed=2026
        )
        self.assertEqual(result.estimate, 0.0)
        self.assertEqual((result.ci_low, result.ci_high), (0.0, 0.0))
        self.assertEqual(result.fraction_a_lower, 0.0)

    def test_paired_better_system_has_negative_difference(self) -> None:
        labels, scores, speakers = _synthetic(4)
        better = [score + (1.5 if label else -1.5) for label, score in zip(labels, scores)]
        result = paired_speaker_bootstrap(
            labels, better, scores, speakers, n_resamples=200, seed=2026
        )
        self.assertLess(result.estimate, 0.0)
        self.assertLess(result.ci_high, 0.0)
        self.assertEqual(result.fraction_a_lower, 1.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cùng số phần tử"):
            speaker_bootstrap([0, 1], [0.1], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "speaker"):
            speaker_bootstrap([0, 1], [0.1, 0.2], ["a", ""])
        with self.assertRaisesRegex(ValueError, "n_resamples"):
            speaker_bootstrap([0, 1], [0.1, 0.2], ["a", "b"], n_resamples=0)
        with self.assertRaisesRegex(ValueError, "confidence"):
            speaker_bootstrap([0, 1], [0.1, 0.2], ["a", "b"], confidence=1.0)
        with self.assertRaisesRegex(ValueError, "hữu hạn"):
            speaker_bootstrap([0, 1], [0.1, float("nan")], ["a", "b"])

    def test_single_class_data_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cả mẫu dương và mẫu âm"):
            speaker_bootstrap([1, 1], [0.1, 0.9], ["a", "b"], n_resamples=10)


class BreakdownTests(unittest.TestCase):
    def test_per_utt_type_eer_pairs_bonafide_with_each_attack(self) -> None:
        labels = [0, 0, 0, 1, 1, 1, 1]
        scores = [0.1, 0.4, 0.2, 0.9, 0.3, 0.8, 0.7]
        utt = ["bonafide"] * 3 + ["voice_conversion", "voice_conversion", "adversarial_attack", "adversarial_attack"]
        result = per_utt_type_eer(labels, scores, utt)
        self.assertEqual(sorted(result), ["adversarial_attack", "voice_conversion"])
        vc = equal_error_rate([0, 0, 0, 1, 1], [0.1, 0.4, 0.2, 0.9, 0.3])
        self.assertAlmostEqual(result["voice_conversion"]["eer"], vc.eer, places=12)
        self.assertEqual(result["voice_conversion"]["spoof_samples"], 2)
        self.assertEqual(result["voice_conversion"]["bonafide_samples"], 3)

    def test_per_utt_type_requires_bonafide(self) -> None:
        with self.assertRaisesRegex(ValueError, "bonafide"):
            per_utt_type_eer([1, 1], [0.1, 0.2], ["voice_conversion", "replay"])

    def test_confusion_counts_use_score_at_or_above_threshold_as_spoof(self) -> None:
        result = confusion_at_threshold([0, 0, 1, 1, 0], [0.1, 0.5, 0.5, 0.9, 0.7], threshold=0.5)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 2, 0, 2))
        self.assertAlmostEqual(result["fpr"], 2 / 3)
        self.assertAlmostEqual(result["fnr"], 0.0)


if __name__ == "__main__":
    unittest.main()
