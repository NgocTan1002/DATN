from __future__ import annotations

import random
import unittest

import torch

from src.data import fixed_length_segment


class FixedLengthSegmentTests(unittest.TestCase):
    def test_exact_length_is_preserved(self) -> None:
        self.assertEqual(
            fixed_length_segment([1, 2, 3], 3, training=False), [1, 2, 3]
        )

    def test_short_waveform_is_repeated_then_trimmed(self) -> None:
        self.assertEqual(
            fixed_length_segment([1, 2, 3], 8, training=False),
            [1, 2, 3, 1, 2, 3, 1, 2],
        )

    def test_evaluation_uses_center_crop(self) -> None:
        self.assertEqual(
            fixed_length_segment(list(range(10)), 4, training=False), [3, 4, 5, 6]
        )

    def test_training_crop_is_seeded_when_rng_is_supplied(self) -> None:
        first = fixed_length_segment(
            list(range(20)), 5, training=True, rng=random.Random(2026)
        )
        second = fixed_length_segment(
            list(range(20)), 5, training=True, rng=random.Random(2026)
        )
        self.assertEqual(first, second)

    def test_tensor_short_waveform_is_repeated_then_trimmed(self) -> None:
        waveform = torch.tensor([1.0, 2.0, 3.0])

        segment = fixed_length_segment(waveform, 8, training=False)

        self.assertIsInstance(segment, torch.Tensor)
        self.assertTrue(
            torch.equal(segment, torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]))
        )

    def test_tensor_training_crop_uses_torch_generator(self) -> None:
        waveform = torch.arange(20)

        first = fixed_length_segment(
            waveform,
            5,
            training=True,
            rng=torch.Generator().manual_seed(2026),
        )
        second = fixed_length_segment(
            waveform,
            5,
            training=True,
            rng=torch.Generator().manual_seed(2026),
        )

        self.assertTrue(torch.equal(first, second))

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fixed_length_segment([], 4, training=False)
        with self.assertRaises(ValueError):
            fixed_length_segment([1], 0, training=False)


if __name__ == "__main__":
    unittest.main()
