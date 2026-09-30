from __future__ import annotations

import unittest

from src.data.audit_summary import (
    build_amplitude_summaries,
    quantile,
    summarize_observations,
)


def observation(
    *,
    split: str,
    label: str,
    utt_type: str,
    rate: int,
    policy: str,
    gain: float,
    near_silence: bool = False,
    peak_limited: bool = False,
) -> dict[str, object]:
    return {
        "split": split,
        "binary_label": label,
        "utt_type": utt_type,
        "native_sample_rate": rate,
        "policy": policy,
        "input_peak": 0.5,
        "output_peak": 0.5 * gain,
        "input_rms_dbfs": -30.0,
        "output_rms_dbfs": -30.0 + gain,
        "applied_gain": gain,
        "near_silence": near_silence,
        "peak_limited": peak_limited,
    }


class AmplitudeAuditSummaryTests(unittest.TestCase):
    def test_quantile_interpolates(self) -> None:
        self.assertEqual(quantile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(quantile([0.0, 10.0], 0.95), 9.5)

    def test_groups_counts_flags_and_statistics(self) -> None:
        rows = [
            observation(
                split="train",
                label="bonafide",
                utt_type="bonafide",
                rate=16_000,
                policy="none",
                gain=1.0,
            ),
            observation(
                split="train",
                label="spoof",
                utt_type="voice_conversion",
                rate=40_000,
                policy="none",
                gain=1.0,
                near_silence=True,
            ),
            observation(
                split="train",
                label="bonafide",
                utt_type="bonafide",
                rate=16_000,
                policy="rms_dbfs",
                gain=2.0,
                peak_limited=True,
            ),
        ]

        summaries = summarize_observations(rows, ("policy",))

        none = next(row for row in summaries if row["policy"] == "none")
        self.assertEqual(none["samples"], 2)
        self.assertEqual(none["near_silence"], 1)
        self.assertEqual(none["statistics"]["applied_gain"]["median"], 1.0)
        rms = next(row for row in summaries if row["policy"] == "rms_dbfs")
        self.assertEqual(rms["peak_limited"], 1)

    def test_builds_all_protocol_groupings(self) -> None:
        rows = [
            observation(
                split="dev",
                label="spoof",
                utt_type="replay",
                rate=16_000,
                policy="peak",
                gain=1.5,
            )
        ]

        summaries = build_amplitude_summaries(rows)

        self.assertEqual(
            set(summaries),
            {
                "by_policy",
                "by_split_and_policy",
                "by_label_and_policy",
                "by_utt_type_and_policy",
                "by_native_sample_rate_and_policy",
            },
        )
        self.assertEqual(
            summaries["by_native_sample_rate_and_policy"][0][
                "native_sample_rate"
            ],
            16_000,
        )

    def test_rejects_missing_or_nonfinite_fields(self) -> None:
        with self.assertRaises(ValueError):
            summarize_observations([{"policy": "none"}], ("policy",))
        with self.assertRaises(ValueError):
            quantile([float("nan")], 0.5)


if __name__ == "__main__":
    unittest.main()
