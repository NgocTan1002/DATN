"""T19b: kiểm lọc, chọn mẫu, tái tạo và tiếp tục theo khối; không mạng/test thật."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
import torch
from scripts import probe_shortcut_dev as probe


class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.old_threads)

    def sine(self, frequency: int) -> torch.Tensor:
        return torch.sin(2 * torch.pi * frequency * torch.arange(64000, dtype=torch.float64) / 16000).float()

    def test_fir_pass_and_stop_bands(self) -> None:
        # 4k: pass3k, stop5k; 7k: pass6k, stop7.5k. Cách cutoff >=500Hz.
        for cutoff, pass_frequency, stop_frequency in ((4000, 3000, 5000), (7000, 6000, 7500)):
            for frequency, is_pass in ((pass_frequency, True), (stop_frequency, False)):
                with self.subTest(cutoff=cutoff, frequency=frequency):
                    x = self.sine(frequency)
                    y = probe.fir_filter(x, cutoff)
                    gain_db = float(20 * torch.log10(y[1024:-1024].square().mean().sqrt() /
                                                     x[1024:-1024].square().mean().sqrt()))
                    self.assertLessEqual(abs(gain_db), .5) if is_pass else self.assertLessEqual(gain_db, -40)

    def test_kernel_symmetric_and_frequency_response(self) -> None:
        for condition, cutoff in (("lowpass_4k", 4000), ("lowpass_7k", 7000)):
            h = probe.fir_kernel(cutoff)
            torch.testing.assert_close(h, h.flip(0), atol=1e-7, rtol=0)
            self.assertAlmostEqual(float(h.sum()), 1, places=6)
            params = probe.filter_parameters(condition)
            self.assertEqual(params["taps"], 513)
            self.assertEqual(set(params["response_db"]), {"3000", "4000", "6000", "7000", "7500"})

    def test_transforms_length_and_alignment(self) -> None:
        x = torch.zeros(64000)
        x[32000] = 1
        for condition in probe.CONDITIONS:
            with self.subTest(condition=condition):
                y = probe.apply_condition(x, condition, "bonafide")
                self.assertEqual(y.shape, x.shape)
                self.assertLessEqual(abs(int(y.abs().argmax()) - 32000), 1)
                self.assertTrue(bool(torch.isfinite(y).all()))

    def test_original_and_nonbonafide_roundtrip_identical(self) -> None:
        x = self.sine(1000) * .2
        self.assertTrue(torch.equal(probe.apply_condition(x, "original", "bonafide"), x))
        for group in probe.GROUPS[1:]:
            self.assertTrue(torch.equal(probe.apply_condition(x, "roundtrip_40k", group), x))

    def test_transforms_reapply_peak_without_clipping(self) -> None:
        x = self.sine(1000) * .1 + self.sine(7500) * .05
        for condition in probe.CONDITIONS[1:]:
            y = probe.apply_condition(x, condition, "bonafide")
            self.assertAlmostEqual(float(y.abs().max()), .95, places=5)

    def test_invalid_filter_or_shape_rejected(self) -> None:
        with self.assertRaises(ValueError):
            probe.fir_kernel(5000)
        with self.assertRaises(ValueError):
            probe.apply_condition(torch.zeros(100), "original", "bonafide")


class SelectionAndMetricsTests(unittest.TestCase):
    def test_actual_registered_subset_sha_and_counts(self) -> None:
        rows = probe.read_dev_manifest(probe.ROOT / "data/manifests/closed_dev_full_v1.csv")
        selected = probe.subset_indices(rows)
        files = [rows[i]["file"] for i in selected]
        self.assertEqual(len(files), 3120)
        self.assertEqual(len(set(files)), 3120)
        self.assertEqual(probe.file_list_sha(files), probe.SUBSET_SHA)
        for group, count in zip(probe.GROUPS, (1000, 1000, 1000, 120)):
            self.assertEqual(sum(rows[i]["utt_type"] == group for i in selected), count)
        reversed_rows = rows[::-1]
        self.assertEqual(files, [reversed_rows[i]["file"] for i in probe.subset_indices(reversed_rows)])

    def test_insufficient_quota_rejected(self) -> None:
        with self.assertRaises(ValueError):
            probe.subset_indices([])

    def test_logit_stable_endpoints_and_small_probabilities(self) -> None:
        values = probe.stable_logit([0, 1e-30, .5, 1 - 1e-12, 1])
        self.assertTrue(bool(np.isfinite(values).all()))
        self.assertTrue(bool((np.diff(values) > 0).all()))
        self.assertAlmostEqual(values[1], math.log(1e-30), places=10)
        self.assertEqual(values[2], 0)

    def test_logit_rejects_invalid_probabilities(self) -> None:
        for value in (-.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.stable_logit([value])

    def test_b0_reproduction_fixed_threshold(self) -> None:
        self.assertTrue(probe.check_reproduction([.1, .9], [.1, .9], .5, model="B0")["passed"])
        self.assertFalse(probe.check_reproduction([.1001, .9], [.1, .9], .5, model="B0")["passed"])
        result = probe.check_reproduction([.500001], [.499999], .5, model="B0")
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision_flips"], 1)

    def test_xlsr_logit_reproduction_and_flip_margin(self) -> None:
        sigmoid = lambda x: 1 / (1 + math.exp(-x))
        threshold = sigmoid(-9.5)
        result = probe.check_reproduction([sigmoid(-9.49)], [sigmoid(-9.51)], threshold, model=probe.XMODEL)
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision_flips"], 1)
        result = probe.check_reproduction([sigmoid(-9.4)], [sigmoid(-9.6)], threshold, model=probe.XMODEL)
        self.assertFalse(result["passed"])
        self.assertEqual(result["flips_outside_0_05_margin"], 1)

    def test_fixed_threshold_metrics(self) -> None:
        rows = [{"file": str(i), "utt_type": group, "binary_label": int(group != "bonafide"), "shard": "a"}
                for i, group in enumerate(probe.GROUPS)]
        result = probe.summarize_scores(rows, [.6, .4, .9, .1], .5)
        self.assertEqual(result["decision_threshold"], .5)
        self.assertEqual(result["groups"]["bonafide"]["fpr"], 1)
        self.assertEqual(result["groups"]["voice_conversion"]["miss_rate"], 1)
        self.assertEqual(result["groups"]["adversarial_attack"]["miss_rate"], 0)

    def test_refuses_test_manifest_before_open(self) -> None:
        with patch.object(Path, "open", side_effect=AssertionError("Không được mở")):
            with self.assertRaisesRegex(ValueError, "test"):
                probe.read_dev_manifest(Path("closed_test_full_v1.csv"))

    def test_refuses_test_artifact_before_memmap(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "candidate"
            path.mkdir()
            (path / "metadata.json").write_text(json.dumps({"split": "closed_test"}))
            with patch.object(probe, "PackedManifestDataset", side_effect=AssertionError("Không được mở")):
                with self.assertRaisesRegex(ValueError, "closed_dev"):
                    probe.open_dev_artifact(path)

    def test_indicator_thresholds_and_lowpass4_not_used(self) -> None:
        def metric(fpr: float, vc: float, ap: float) -> dict:
            return {"groups": {"bonafide": {"fpr": fpr}},
                    "per_attack_eer": {"voice_conversion": {"eer": vc}, "adversarial_attack": {"eer": ap}}}
        values = {"original": metric(.01, .02, .03), "roundtrip_40k": metric(.12, .02, .03),
                  "lowpass_7k": metric(.01, .061, .04), "lowpass_4k": metric(.9, .8, .8)}
        result = probe.indicators(values)
        self.assertTrue(result["roundtrip_40k"]["strong_threshold_hit"])
        self.assertTrue(result["lowpass_7k"]["strong_threshold_hit"])
        self.assertNotIn("lowpass_4k", result)
        self.assertEqual(probe.indicators({"original": values["original"], "lowpass_4k": values["lowpass_4k"]})
                         ["available_indicator_level"], "not_available")


class BlockAndDeadlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.identity = {"models": {"m": {"checkpoint_sha256": "a" * 64}},
                         "filters": {"original": {"operation": "identity"}}}
        self.store = probe.BlockStore(Path(self.temp.name), self.identity)
        self.records = [{"file": "a", "score": .25, "raw_logit": -1.0}]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_atomic_save_and_valid_resume(self) -> None:
        self.store.save("m", "original", 0, ["a"], self.records, 1)
        value = self.store.load("m", "original", 0, ["a"])
        self.assertEqual(value["records"], self.records)
        self.assertEqual(value["files_sha256"], hashlib.sha256(b"a\n").hexdigest())
        self.assertFalse(list(Path(self.temp.name).rglob("*.tmp")))
        with self.assertRaises(ValueError):
            self.store.save("m", "original", 0, ["a"], self.records, 1)

    def test_atomic_failure_leaves_no_committed_block(self) -> None:
        with patch.object(probe.os, "replace", side_effect=OSError("Giả lập ngắt")):
            with self.assertRaises(OSError):
                self.store.save("m", "original", 0, ["a"], self.records, 1)
        self.assertIsNone(self.store.load("m", "original", 0, ["a"]))
        self.assertFalse(list(Path(self.temp.name).rglob("*.tmp")))

    def test_resume_rejects_corrupt_content(self) -> None:
        self.store.save("m", "original", 0, ["a"], self.records, 1)
        path = self.store.path("m", "original", 0)
        value = probe.read_json(path); value["records"][0]["score"] = .4
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "không ghi đè"):
            self.store.load("m", "original", 0, ["a"])

    def test_resume_rejects_wrong_file_checkpoint_and_filter(self) -> None:
        self.store.save("m", "original", 0, ["a"], self.records, 1)
        with self.assertRaises(ValueError):
            self.store.load("m", "original", 0, ["b"])
        for field in ("models", "filters"):
            changed = deepcopy(self.identity)
            if field == "models":
                changed["models"]["m"]["checkpoint_sha256"] = "b" * 64
            else:
                changed["filters"]["original"]["operation"] = "different"
            with self.assertRaises(ValueError):
                probe.BlockStore(self.store.root, changed).load("m", "original", 0, ["a"])

    def test_process_skips_valid_block_without_predicting(self) -> None:
        self.store.save("m", "original", 0, ["a"], self.records, 1)
        def forbidden(*args: object) -> dict:
            raise AssertionError("Không được chấm lại khối hợp lệ.")
        probe.process_blocks(self.store, ["m"], "original", [0], [{"file": "a"}], forbidden,
                             probe.Deadline(1), lambda *a, **kw: None)

    def test_deadline_stops_after_committing_one_block(self) -> None:
        clock = [0.0]
        deadline = probe.Deadline(1 / 3600, clock=lambda: clock[0])
        rows = [{"file": str(i)} for i in range(201)]
        def predict(indices: list[int], names: list[str], condition: str) -> dict:
            clock[0] = 2
            return {"m": [{"file": str(i), "score": .25, "raw_logit": -1.0} for i in indices]}
        with self.assertRaises(probe.DeadlineReached):
            probe.process_blocks(self.store, ["m"], "original", list(range(201)), rows, predict,
                                 deadline, lambda *a, **kw: None)
        self.assertEqual(len(self.store.load("m", "original", 0, [str(i) for i in range(200)])["records"]), 200)
        self.assertIsNone(self.store.load("m", "original", 200, ["200"]))

    def test_invalid_deadline_rejected(self) -> None:
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                probe.Deadline(value)


class StageOrderTests(unittest.TestCase):
    def fake_probe(self) -> tuple[probe.Probe, list[tuple]]:
        value = probe.Probe.__new__(probe.Probe)
        calls: list[tuple] = []
        value.invocation = {}
        value.dataset = Mock()
        value.emit = lambda *a, **kw: None
        value.publish = lambda *a, **kw: calls.append(("report", a[0]))
        def phase(name: str, action: object) -> None:
            calls.append(("phase", name))
            action()
        value.phase = phase
        value.blocks = lambda names, condition, limit=None: calls.append(("blocks", tuple(names), condition, limit))
        value.verify_original = lambda name, count=None: calls.append(("verify", name, count))
        return value, calls

    def test_fixed_stage_order_and_b0_gate(self) -> None:
        value, calls = self.fake_probe()
        self.assertEqual(value.run(), 0)
        self.assertEqual([x[1] for x in calls if x[0] == "phase"], ["i", "ii", "iii", "iv", "v", "vi"])
        first = next(x for x in calls if x[0] == "blocks")
        self.assertEqual(first, ("blocks", (probe.XMODEL,), "original", 200))
        b0_round = calls.index(("blocks", probe.BMODELS, "roundtrip_40k", None))
        for name in probe.BMODELS:
            self.assertLess(calls.index(("verify", name, None)), b0_round)
        self.assertEqual([x for x in calls if x[0] == "blocks"][-1],
                         ("blocks", (probe.XMODEL,), "lowpass_4k", None))

    def test_first200_reproduction_failure_stops_everything(self) -> None:
        value, calls = self.fake_probe()
        def fail(name: str, count: int | None = None) -> None:
            raise probe.ReproductionFailure("Giả lập tái tạo lệch.")
        value.verify_original = fail
        self.assertEqual(value.run(), 3)
        self.assertEqual(len([x for x in calls if x[0] == "blocks"]), 1)
        self.assertIn(("report", "blocked_reproduction"), calls)
        value.dataset.close.assert_called_once()

    def test_deadline_publishes_partial_and_stops(self) -> None:
        value, calls = self.fake_probe()
        def stop(*args: object, **kwargs: object) -> None:
            raise probe.DeadlineReached("Giả lập hết giờ.")
        value.blocks = stop
        self.assertEqual(value.run(), 2)
        self.assertEqual(value.invocation["status"], "paused_deadline")
        self.assertIn(("report", "paused_deadline"), calls)


if __name__ == "__main__":
    unittest.main()
