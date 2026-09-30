from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LFCCConfigurationTests(unittest.TestCase):
    def test_locked_input_produces_expected_finite_shape(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs" / "lfcc_lcnn.json").read_text(
                encoding="utf-8"
            )
        )
        lfcc_config = config["lfcc"]
        spectrum = lfcc_config["spectrum"]
        transform = torchaudio.transforms.LFCC(
            sample_rate=lfcc_config["sample_rate"],
            n_filter=lfcc_config["n_filter"],
            n_lfcc=lfcc_config["n_lfcc"],
            f_min=lfcc_config["f_min"],
            f_max=lfcc_config["f_max"],
            dct_type=lfcc_config["dct_type"],
            norm=lfcc_config["norm"],
            log_lf=lfcc_config["log_lf"],
            speckwargs={
                "n_fft": spectrum["n_fft"],
                "win_length": spectrum["win_length"],
                "hop_length": spectrum["hop_length"],
                "window_fn": torch.hann_window,
                "power": spectrum["power"],
                "center": spectrum["center"],
                "pad_mode": spectrum["pad_mode"],
            },
        )
        waveform = torch.zeros(2, config["input"]["samples"])

        features = transform(waveform)

        self.assertEqual(features.shape, (2, 60, 401))
        self.assertTrue(torch.isfinite(features).all())
        model_input = features.unsqueeze(1)
        self.assertEqual(model_input.shape, (2, 1, 60, 401))


if __name__ == "__main__":
    unittest.main()
