from __future__ import annotations

import math
import unittest

from src.metrics import (
    equal_error_rate,
    evaluate_development_scores,
    evaluate_scores_at_threshold,
    macro_f1_at_threshold,
    rates_at_threshold,
    roc_auc_score,
)


class EqualErrorRateTests(unittest.TestCase):
    def test_perfect_separation_has_zero_eer(self) -> None:
        result = equal_error_rate([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(result.eer, 0.0)
        self.assertEqual(result.positives, 2)
        self.assertEqual(result.negatives, 2)

    def test_known_equal_operating_point(self) -> None:
        result = equal_error_rate([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(result.eer, 0.5)
        self.assertAlmostEqual(result.false_accept_rate, 0.5)
        self.assertAlmostEqual(result.false_reject_rate, 0.5)

    def test_tied_scores_do_not_depend_on_order(self) -> None:
        first = equal_error_rate([1, 0, 1, 0], [0.5, 0.5, 0.2, 0.1])
        second = equal_error_rate([0, 1, 0, 1], [0.5, 0.5, 0.1, 0.2])
        self.assertAlmostEqual(first.eer, second.eer)
        self.assertAlmostEqual(first.threshold, second.threshold)

    def test_rates_at_threshold(self) -> None:
        false_accept, false_reject = rates_at_threshold(
            [1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], threshold=0.5
        )
        self.assertAlmostEqual(false_accept, 0.5)
        self.assertAlmostEqual(false_reject, 0.5)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            equal_error_rate([], [])
        with self.assertRaises(ValueError):
            equal_error_rate([1, 0], [0.1])
        with self.assertRaises(ValueError):
            equal_error_rate([1, 1], [0.9, 0.8])
        with self.assertRaises(ValueError):
            equal_error_rate([1, 0], [math.nan, 0.2])


class RocAucTests(unittest.TestCase):
    def test_perfect_separation_has_unit_area(self) -> None:
        area = roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(area, 1.0)

    def test_tied_scores_do_not_depend_on_order(self) -> None:
        first = roc_auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        second = roc_auc_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(first, 0.5)
        self.assertAlmostEqual(second, 0.5)


class MacroF1Tests(unittest.TestCase):
    def test_known_confusion_matrix_has_expected_macro_f1(self) -> None:
        score = macro_f1_at_threshold(
            [0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5
        )
        self.assertAlmostEqual(score, 0.5)

    def test_non_finite_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Threshold"):
            macro_f1_at_threshold([0, 1], [0.1, 0.9], threshold=math.nan)


class BinaryEvaluationTests(unittest.TestCase):
    def test_development_result_freezes_its_eer_threshold(self) -> None:
        result = evaluate_development_scores(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            loss=0.25,
            utt_types=["bonafide", "bonafide", "voice_conversion", "replay"],
        )

        self.assertEqual(result.threshold_source, "development_eer")
        self.assertAlmostEqual(result.decision_threshold, result.eer_threshold)
        self.assertAlmostEqual(result.roc_auc, 1.0)
        self.assertAlmostEqual(result.macro_f1, 1.0)
        self.assertEqual(result.sample_count, 4)
        self.assertEqual(
            result.utt_type_counts,
            {"bonafide": 2, "replay": 1, "voice_conversion": 1},
        )
        self.assertEqual(result.labels, (0, 0, 1, 1))
        self.assertEqual(result.scores, (0.1, 0.2, 0.8, 0.9))

    def test_evaluation_uses_provided_development_threshold(self) -> None:
        development = evaluate_development_scores(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            loss=0.25,
            utt_types=["bonafide", "bonafide", "voice_conversion", "replay"],
        )
        evaluated = evaluate_scores_at_threshold(
            [1, 0, 1, 0],
            [0.8, 0.7, 0.2, 0.1],
            loss=0.5,
            utt_types=["voice_conversion", "bonafide", "replay", "bonafide"],
            threshold=development.decision_threshold,
        )

        self.assertEqual(evaluated.threshold_source, "provided")
        self.assertAlmostEqual(
            evaluated.decision_threshold, development.decision_threshold
        )
        self.assertNotAlmostEqual(
            evaluated.decision_threshold, evaluated.eer_threshold
        )

    def test_utt_type_count_must_match_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "utt_types"):
            evaluate_development_scores(
                [0, 1],
                [0.1, 0.9],
                loss=0.25,
                utt_types=["bonafide"],
            )


if __name__ == "__main__":
    unittest.main()
