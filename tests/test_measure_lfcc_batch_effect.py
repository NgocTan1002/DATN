from __future__ import annotations

import json
import unittest

import torch

from scripts.measure_lfcc_batch_effect import (
    compare_scores,
    score_batched,
    score_single,
)


class _Probe(torch.nn.Module):
    """Mô hình giả: logit là trung bình đặc trưng, đủ để kiểm tra đường tính."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=(1, 2, 3))


def _lfcc(waveform: torch.Tensor) -> torch.Tensor:
    # LFCC giả có hành vi phụ thuộc batch: trừ mức lớn nhất của cả batch.
    features = waveform.unsqueeze(1).expand(-1, 3, -1)[..., :4].clone()
    return features - features.amax()


class ScoreTests(unittest.TestCase):
    def test_batched_scores_depend_on_batch_members(self) -> None:
        waves = torch.stack([torch.full((8,), 1.0), torch.full((8,), 5.0)])
        together = score_batched(_Probe(), _lfcc, waves, batch_size=2)
        apart = score_batched(_Probe(), _lfcc, waves, batch_size=1)
        self.assertNotEqual(together, apart)

    def test_single_scores_equal_batch_size_one(self) -> None:
        waves = torch.stack([torch.full((8,), 1.0), torch.full((8,), 5.0)])
        self.assertEqual(
            score_single(_Probe(), _lfcc, waves),
            score_batched(_Probe(), _lfcc, waves, batch_size=1),
        )

    def test_scores_are_probabilities(self) -> None:
        waves = torch.randn(3, 8)
        scores = score_single(_Probe(), _lfcc, waves)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in scores))


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = [0, 0, 0, 1, 1, 1, 0, 1]
        self.stored = [0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.25, 0.6]
        self.speakers = ["a", "a", "b", "b", "c", "c", "d", "d"]
        self.utt = ["bonafide"] * 3 + ["voice_conversion"] * 3 + ["bonafide", "replay"]

    def test_identical_scores_report_no_change(self) -> None:
        result = compare_scores(
            self.labels, self.stored, list(self.stored), self.speakers, self.utt,
            threshold=0.5, n_resamples=50, seed=2026,
        )
        self.assertEqual(result["changed_scores"], 0)
        self.assertEqual(result["decisions_changed"], 0)
        self.assertEqual(result["eer_difference"]["estimate"], 0.0)
        self.assertEqual(result["max_abs_score_difference"], 0.0)

    def test_changed_scores_and_decisions_are_counted(self) -> None:
        single = list(self.stored)
        single[1] = 0.9  # bonafide vượt threshold
        single[5] = 0.2  # spoof rơi dưới threshold
        result = compare_scores(
            self.labels, self.stored, single, self.speakers, self.utt,
            threshold=0.5, n_resamples=50, seed=2026,
        )
        self.assertEqual(result["changed_scores"], 2)
        self.assertEqual(result["decisions_changed"], 2)
        self.assertEqual(result["decisions_changed_by_utt_type"], {"bonafide": 1, "voice_conversion": 1})
        self.assertAlmostEqual(result["max_abs_score_difference"], 0.7)
        json.dumps(result)  # phải tuần tự hóa được thành JSON

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cùng số phần tử"):
            compare_scores(
                self.labels, self.stored[:-1], self.stored, self.speakers, self.utt,
                threshold=0.5, n_resamples=10, seed=2026,
            )


if __name__ == "__main__":
    unittest.main()
