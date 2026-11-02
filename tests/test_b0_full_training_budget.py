"""Kiểm chứng phép chiếu thời gian và bảo toàn ngân sách epoch."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from scripts.estimate_b0_full_training_budget import project_budget


class FullTrainingBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan: dict[str, Any] = {
            "data": {"train_samples": 9, "development_samples": 3, "use_test": False},
            "training": {"batch_size": 4, "max_epochs": 3, "drop_last": False, "steps_per_epoch": 3, "max_optimizer_steps": 9, "max_train_sample_exposures": 27},
            "sessions": {"max_optimizer_steps_per_invocation": 2},
            "evaluation": {"full_development": True, "schedule": "after_each_completed_epoch_before_next_epoch"},
            "budget": {"reserve_fraction": 0.2},
        }
        self.measurements = [
            {"training_step_seconds": 10.0, "development_sample_seconds": 2.0},
            {"training_step_seconds": 8.0, "development_sample_seconds": 4.0},
        ]

    def test_keeps_tail_and_uses_independent_worst_times_with_full_dev_each_epoch(self) -> None:
        result = project_budget(self.plan, self.measurements)
        self.assertEqual(result["steps_per_epoch"], 3)
        self.assertEqual(result["tail_batch_samples"], 1)
        self.assertEqual(result["max_optimizer_steps"], 9)
        self.assertEqual(result["max_train_sample_exposures"], 27)
        self.assertEqual(result["sessions_upper_bound"], 6)
        self.assertAlmostEqual(result["projected_hours_per_epoch"], 42 / 3600)
        self.assertAlmostEqual(result["projected_hours_max_epochs_with_reserve"], 126 * 1.2 / 3600)
        self.assertAlmostEqual(result["projected_worst_session_hours_with_reserve"], 32 * 1.2 / 3600)

    def test_rejects_inconsistent_budget_or_excluding_tail_or_development(self) -> None:
        for group, key, value in [
            ("training", "steps_per_epoch", 2),
            ("training", "max_optimizer_steps", 8),
            ("training", "max_train_sample_exposures", 26),
            ("training", "drop_last", True),
            ("data", "use_test", True),
            ("evaluation", "full_development", False),
        ]:
            with self.subTest(key=key):
                plan = copy.deepcopy(self.plan)
                plan[group][key] = value
                with self.assertRaises(ValueError):
                    project_budget(plan, self.measurements)

    def test_rejects_missing_nonfinite_or_nonpositive_timing(self) -> None:
        for measurements in ([], [{"training_step_seconds": float("nan"), "development_sample_seconds": 2}], [{"training_step_seconds": 10, "development_sample_seconds": 0}]):
            with self.subTest(measurements=measurements):
                with self.assertRaises(ValueError):
                    project_budget(self.plan, measurements)


if __name__ == "__main__":
    unittest.main()
