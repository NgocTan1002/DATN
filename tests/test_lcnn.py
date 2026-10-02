from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from src.models import MaxFeatureMap, build_lcnn_from_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MaxFeatureMapTests(unittest.TestCase):
    def test_takes_maximum_of_paired_channels(self) -> None:
        layer = MaxFeatureMap(dimension=1)
        inputs = torch.tensor([[[1.0, 4.0], [3.0, 2.0], [2.0, 1.0], [0.0, 5.0]]])

        outputs = layer(inputs)

        expected = torch.tensor([[[2.0, 4.0], [3.0, 5.0]]])
        self.assertTrue(torch.equal(outputs, expected))

    def test_rejects_odd_feature_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "số đặc trưng chẵn"):
            MaxFeatureMap(dimension=1)(torch.zeros(2, 3, 4, 4))


class LightCNNBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (PROJECT_ROOT / "configs" / "lfcc_lcnn.json").read_text(encoding="utf-8")
        )

    def test_forward_returns_one_finite_logit_per_utterance(self) -> None:
        model = build_lcnn_from_config(self.config)
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                features = torch.randn(batch_size, 1, 60, 401)
                logits = model(features)
                self.assertEqual(logits.shape, (batch_size,))
                self.assertTrue(torch.isfinite(logits).all())

    def test_rejects_wrong_shape_and_nonfinite_input(self) -> None:
        model = build_lcnn_from_config(self.config)
        with self.assertRaisesRegex(ValueError, "LCNN cần input"):
            model(torch.zeros(2, 60, 401))
        with self.assertRaisesRegex(ValueError, "LCNN cần input"):
            model(torch.zeros(2, 1, 61, 401))

        features = torch.zeros(2, 1, 60, 401)
        features[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN hoặc Inf"):
            model(features)

    def test_loss_backward_and_optimizer_step_are_finite(self) -> None:
        torch.manual_seed(2026)
        model = build_lcnn_from_config(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = torch.nn.BCEWithLogitsLoss()
        features = torch.randn(2, 1, 60, 401)
        labels = torch.tensor([0.0, 1.0])
        before = next(model.parameters()).detach().clone()

        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        optimizer.step()
        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))


if __name__ == "__main__":
    unittest.main()
