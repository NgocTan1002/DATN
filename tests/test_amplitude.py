from __future__ import annotations

import unittest

import torch

from src.data import apply_amplitude_policy


class AmplitudePolicyTests(unittest.TestCase):
    def test_none_preserves_waveform(self) -> None:
        waveform = torch.tensor([0.1, -0.2, 0.3])

        result = apply_amplitude_policy(waveform, "none")

        self.assertTrue(torch.equal(result.waveform, waveform))
        self.assertEqual(result.applied_gain, 1.0)
        self.assertFalse(result.peak_limited)

    def test_peak_reaches_target(self) -> None:
        waveform = torch.tensor([0.1, -0.5, 0.25])

        result = apply_amplitude_policy(waveform, "peak")

        self.assertAlmostEqual(result.output_peak, 0.95, places=5)
        self.assertFalse(result.peak_limited)
        self.assertTrue(torch.isfinite(result.waveform).all())

    def test_rms_reaches_target(self) -> None:
        waveform = torch.full((16_000,), 0.1)

        result = apply_amplitude_policy(waveform, "rms_dbfs")

        self.assertAlmostEqual(
            result.output_rms_dbfs,
            -25.0,
            places=3,
        )
        self.assertFalse(result.peak_limited)

    def test_near_silence_is_not_amplified(self) -> None:
        waveform = torch.full((1_000,), 1e-4)

        for policy in ("peak", "rms_dbfs"):
            with self.subTest(policy=policy):
                result = apply_amplitude_policy(waveform, policy)

                self.assertTrue(result.near_silence)
                self.assertEqual(result.applied_gain, 1.0)
                self.assertTrue(
                    torch.equal(result.waveform, waveform)
                )

    def test_rms_gain_is_limited_by_peak(self) -> None:
        waveform = torch.zeros(10_000)
        waveform[0] = 1.0

        result = apply_amplitude_policy(waveform, "rms_dbfs")

        self.assertTrue(result.peak_limited)
        self.assertLessEqual(result.output_peak, 0.950001)
        self.assertFalse(result.near_silence)

    def test_zero_waveform_remains_finite(self) -> None:
        waveform = torch.zeros(1_000)

        result = apply_amplitude_policy(waveform, "rms_dbfs")

        self.assertTrue(result.near_silence)
        self.assertEqual(result.applied_gain, 1.0)
        self.assertTrue(torch.isfinite(result.waveform).all())

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_amplitude_policy(torch.tensor([]), "none")

        with self.assertRaises(ValueError):
            apply_amplitude_policy(
                torch.tensor([0.0, float("nan")]),
                "none",
            )

        with self.assertRaises(ValueError):
            apply_amplitude_policy(
                torch.tensor([0.0, float("inf")]),
                "none",
            )

        with self.assertRaises(ValueError):
            apply_amplitude_policy(
                torch.zeros(2, 10),
                "none",
            )

        with self.assertRaises(ValueError):
            apply_amplitude_policy(
                torch.zeros(10),
                "unknown",  # type: ignore[arg-type]
            )

    def test_policy_is_deterministic(self) -> None:
        waveform = torch.linspace(-0.5, 0.5, 1_000)

        first = apply_amplitude_policy(waveform, "rms_dbfs")
        second = apply_amplitude_policy(waveform, "rms_dbfs")

        self.assertTrue(
            torch.equal(first.waveform, second.waveform)
        )
        self.assertEqual(first.applied_gain, second.applied_gain)


if __name__ == "__main__":
    unittest.main()