from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from scripts.train_b0 import _model_input
from src.models.lfcc_lcnn import build_lfcc_transform, lfcc_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-3


def _transform() -> torch.nn.Module:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "lfcc_lcnn.json").read_text(encoding="utf-8")
    )
    return build_lfcc_transform(config["lfcc"])


class LFCCBatchIndependenceTests(unittest.TestCase):
    """Đặc trưng của một mẫu không được phụ thuộc mẫu khác trong batch."""

    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        self.loud = torch.randn(64_000, generator=generator) * 0.9
        self.quiet = torch.randn(64_000, generator=generator) * 0.001
        self.lfcc = _transform()

    def test_raw_transform_couples_batch_members(self) -> None:
        # Bảo đảm dữ liệu kiểm thử thực sự kích hoạt cutoff top_db chung của torchaudio.
        batched = self.lfcc(torch.stack([self.loud, self.quiet]))[1]
        alone = self.lfcc(self.quiet.unsqueeze(0))[0]
        self.assertGreater(float((batched - alone).abs().max()), 0.1)

    def test_quiet_sample_is_unchanged_by_loud_partner(self) -> None:
        batched = lfcc_features(self.lfcc, torch.stack([self.loud, self.quiet]))[1]
        alone = lfcc_features(self.lfcc, self.quiet.unsqueeze(0))[0]
        torch.testing.assert_close(batched, alone, atol=ATOL, rtol=0.0)

    def test_matches_unbatched_transform_for_each_sample(self) -> None:
        batch = torch.stack([self.loud, self.quiet])
        features = lfcc_features(self.lfcc, batch)
        for index in range(batch.shape[0]):
            reference = self.lfcc(batch[index].unsqueeze(0))[0]
            torch.testing.assert_close(features[index], reference, atol=ATOL, rtol=0.0)

    def test_batch_order_does_not_change_features(self) -> None:
        forward = lfcc_features(self.lfcc, torch.stack([self.loud, self.quiet]))
        reverse = lfcc_features(self.lfcc, torch.stack([self.quiet, self.loud]))
        torch.testing.assert_close(forward[0], reverse[1], atol=ATOL, rtol=0.0)
        torch.testing.assert_close(forward[1], reverse[0], atol=ATOL, rtol=0.0)

    def test_locked_shape_is_preserved(self) -> None:
        features = lfcc_features(self.lfcc, torch.stack([self.loud, self.quiet]))
        self.assertEqual(tuple(features.shape), (2, 60, 401))
        self.assertTrue(torch.isfinite(features).all())

    def test_invalid_waveform_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch, samples"):
            lfcc_features(self.lfcc, self.quiet)

    def test_b0_model_input_uses_per_sample_features(self) -> None:
        batched = _model_input(self.lfcc, torch.stack([self.loud, self.quiet]))[1]
        alone = _model_input(self.lfcc, self.quiet.unsqueeze(0))[0]
        self.assertEqual(tuple(batched.shape), (1, 60, 401))
        torch.testing.assert_close(batched, alone, atol=ATOL, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
