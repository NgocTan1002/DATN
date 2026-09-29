from __future__ import annotations

import math
import unittest

from scripts.audio_smoke_test import summarize_waveform


class WaveformSummaryTests(unittest.TestCase):
    def test_regular_waveform(self) -> None:
        result = summarize_waveform([0.0, 0.5, -0.5, 1.0], sampling_rate=4)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["nonfinite_samples"], 0)
        self.assertAlmostEqual(result["duration_seconds"], 1.0)
        self.assertAlmostEqual(result["peak_absolute"], 1.0)
        self.assertAlmostEqual(result["rms"], math.sqrt(1.5 / 4.0))
        self.assertAlmostEqual(result["clipped_fraction"], 0.25)
        self.assertFalse(result["silent"])

    def test_nonfinite_values_are_counted(self) -> None:
        result = summarize_waveform([0.1, math.nan, math.inf], sampling_rate=16_000)
        self.assertEqual(result["finite_samples"], 1)
        self.assertEqual(result["nonfinite_samples"], 2)

    def test_empty_waveform_is_silent(self) -> None:
        result = summarize_waveform([], sampling_rate=16_000)
        self.assertTrue(result["silent"])
        self.assertIsNone(result["rms"])


if __name__ == "__main__":
    unittest.main()
