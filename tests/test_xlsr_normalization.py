from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from src.data import (
    XLSR_PROCESSOR_NORMALIZATION_EPSILON,
    apply_xlsr_waveform_normalization,
)


ROOT = Path(__file__).resolve().parents[1]


class XLSRWaveformNormalizationTests(unittest.TestCase):
    def test_pipeline_only_returns_input_unchanged(self) -> None:
        waveform = torch.linspace(-0.7, 0.9, 64_000, dtype=torch.float32).repeat(2, 1)
        result = apply_xlsr_waveform_normalization(waveform, "pipeline_only")
        self.assertIs(result, waveform)
        self.assertTrue(torch.equal(result, waveform))

    def test_processor_mode_matches_published_formula_per_segment(self) -> None:
        first = torch.linspace(-0.4, 0.8, 64_000, dtype=torch.float32)
        second = torch.linspace(0.2, 1.1, 64_000, dtype=torch.float32)
        waveform = torch.stack((first, second))
        result = apply_xlsr_waveform_normalization(
            waveform, "processor_zero_mean_unit_variance"
        )
        variance, mean = torch.var_mean(
            waveform, dim=-1, correction=0, keepdim=True
        )
        expected = (waveform - mean) / torch.sqrt(
            variance + XLSR_PROCESSOR_NORMALIZATION_EPSILON
        )
        self.assertTrue(torch.allclose(result, expected, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(result.mean(dim=-1), torch.zeros(2), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                result.var(dim=-1, correction=0), torch.ones(2), atol=2e-6
            )
        )

    def test_processor_mode_keeps_constant_segments_finite(self) -> None:
        waveform = torch.full((2, 64_000), 0.25, dtype=torch.float32)
        result = apply_xlsr_waveform_normalization(
            waveform, "processor_zero_mean_unit_variance"
        )
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_invalid_mode_shape_dtype_and_values_are_rejected(self) -> None:
        valid = torch.zeros((1, 64_000), dtype=torch.float32)
        invalid_cases = (
            (valid, "unknown"),
            (torch.zeros(64_000), "pipeline_only"),
            (torch.zeros((1, 63_999)), "pipeline_only"),
            (valid.to(torch.float64), "pipeline_only"),
            (valid.clone().index_fill(1, torch.tensor([0]), float("nan")), "pipeline_only"),
        )
        for waveform, mode in invalid_cases:
            with self.subTest(mode=mode, shape=tuple(waveform.shape)):
                with self.assertRaises(ValueError):
                    apply_xlsr_waveform_normalization(waveform, mode)  # type: ignore[arg-type]

    def test_d024_config_is_paired_development_only_and_not_ready(self) -> None:
        plan = json.loads(
            (ROOT / "configs/xlsr_normalization_pilot.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(plan["decision"], "D024")
        self.assertFalse(plan["ready_for_gpu_smoke"])
        self.assertFalse(plan["ready_for_pilot_training"])
        self.assertFalse(plan["data"]["use_test"])
        self.assertFalse(plan["comparison"]["selection"]["test_set_used"])
        self.assertTrue(plan["comparison"]["paired"])
        self.assertEqual(
            {arm["waveform_normalization"] for arm in plan["comparison"]["arms"]},
            {"pipeline_only", "processor_zero_mean_unit_variance"},
        )
        self.assertEqual(plan["model"]["mode"], "X0")
        self.assertEqual(plan["model"]["num_unfrozen_layers"], 0)
        self.assertIsNone(plan["pretrained"]["weights_sha256"])
        self.assertIsNone(plan["training"]["batch_size"])
        self.assertIsNone(plan["training"]["max_optimizer_steps"])


if __name__ == "__main__":
    unittest.main()
