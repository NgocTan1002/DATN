from __future__ import annotations

import math
import unittest

from src.metrics import equal_error_rate, rates_at_threshold


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


if __name__ == "__main__":
    unittest.main()
