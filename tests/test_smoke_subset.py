from __future__ import annotations

import unittest
from collections import Counter

from src.data.smoke_subset import (
    SmokeRow,
    select_smoke_rows,
    validate_smoke_partitions,
)


def rows_for_speaker(speaker: str, utt_type: str, count: int) -> list[SmokeRow]:
    return [
        SmokeRow(
            file=f"{speaker}/{utt_type}/{index:03d}.wav",
            speaker_id=speaker,
            utt_type=utt_type,
        )
        for index in range(count)
    ]


class SmokeSubsetTests(unittest.TestCase):
    def test_selection_is_balanced_deterministic_and_order_independent(self) -> None:
        candidates = (
            rows_for_speaker("real", "bonafide", 12)
            + rows_for_speaker("vc", "voice_conversion", 8)
            + rows_for_speaker("ap", "adversarial_attack", 8)
            + rows_for_speaker("replay", "replay", 8)
        )

        first = select_smoke_rows(
            candidates, split_name="train", per_class=9, seed=2026
        )
        second = select_smoke_rows(
            reversed(candidates), split_name="train", per_class=9, seed=2026
        )

        self.assertEqual(first, second)
        labels = Counter(row.binary_label for row in first)
        self.assertEqual(labels, {0: 9, 1: 9})
        spoof_types = Counter(row.utt_type for row in first if row.binary_label == 1)
        self.assertEqual(
            spoof_types,
            {"voice_conversion": 3, "adversarial_attack": 3, "replay": 3},
        )

    def test_selection_falls_back_when_one_spoof_type_is_small(self) -> None:
        candidates = (
            rows_for_speaker("real", "bonafide", 10)
            + rows_for_speaker("vc", "voice_conversion", 1)
            + rows_for_speaker("ap", "adversarial_attack", 10)
        )

        selected = select_smoke_rows(
            candidates, split_name="dev", per_class=6, seed=2026
        )

        spoof_types = Counter(row.utt_type for row in selected if row.binary_label == 1)
        self.assertEqual(spoof_types["voice_conversion"], 1)
        self.assertEqual(spoof_types["adversarial_attack"], 5)

    def test_validation_accepts_disjoint_partitions(self) -> None:
        partitions = {
            "train": [
                SmokeRow("train/r.wav", "train-real", "bonafide"),
                SmokeRow("train/s.wav", "train-spoof", "replay"),
            ],
            "dev": [
                SmokeRow("dev/r.wav", "dev-real", "bonafide"),
                SmokeRow("dev/s.wav", "dev-spoof", "voice_conversion"),
            ],
        }

        checks = validate_smoke_partitions(partitions)

        self.assertEqual(checks["total_samples"], 4)
        self.assertEqual(checks["speaker_overlaps"]["dev_vs_train"], 0)

    def test_validation_rejects_speaker_overlap(self) -> None:
        partitions = {
            "train": [
                SmokeRow("train/r.wav", "same", "bonafide"),
                SmokeRow("train/s.wav", "train-spoof", "replay"),
            ],
            "test": [
                SmokeRow("test/r.wav", "same", "bonafide"),
                SmokeRow("test/s.wav", "test-spoof", "replay"),
            ],
        }

        with self.assertRaisesRegex(ValueError, "Speaker overlap"):
            validate_smoke_partitions(partitions)


if __name__ == "__main__":
    unittest.main()
