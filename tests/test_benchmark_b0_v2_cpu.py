from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.benchmark_b0_v2_cpu import measure, project
from scripts.train_b0 import _load_json
from scripts.train_b0_full import _packed
from tests.test_b0_v2_runner import _write_artifact


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkB0V2Tests(unittest.TestCase):
    def test_measure_and_project_on_tiny_artifacts(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(root / "train", split="closed_train", rows=40, manifest_sha256="0" * 64, offset=1)
            _write_artifact(root / "dev", split="closed_dev", rows=30, manifest_sha256="1" * 64, offset=2)
            audio = _load_json(ROOT / "configs/audio.json", "audio")
            config = _load_json(ROOT / "configs/lfcc_lcnn.json", "mô hình")
            train = _packed(root / "train", training=True, seed=2026, audio=audio)
            dev = _packed(root / "dev", training=False, seed=2026, audio=audio)
            try:
                result = measure(train=train, dev=dev, config=config, train_steps=3, dev_samples=10, warmup_steps=1)
                with self.assertRaises(ValueError):
                    measure(train=train, dev=dev, config=config, train_steps=0, dev_samples=10)
            finally:
                train.close()
                dev.close()
        self.assertEqual(result["train_steps_measured"], 3)
        self.assertEqual(result["dev_samples_measured"], 10)
        self.assertGreater(result["seconds_per_step_mean"], 0.0)
        self.assertGreater(result["seconds_per_dev_sample"], 0.0)
        projection = project(
            {"seconds_per_step_mean": 1.0, "seconds_per_dev_sample": 0.1},
            steps_per_epoch=10, max_epochs=3, dev_total=100,
        )
        self.assertAlmostEqual(projection["train_hours_max_epochs"], 30 / 3600)
        self.assertAlmostEqual(projection["dev_full_hours_max_epochs"], 30 / 3600)
        self.assertAlmostEqual(projection["hours_three_seeds_max"], 3 * 60 / 3600)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
