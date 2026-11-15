"""T22: đặc tả, logit, khối/resume và thứ tự; không mạng/mô hình thật."""
from __future__ import annotations

from copy import deepcopy
import argparse
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scripts import probe_shortcut_x1_final as probe


class SpecificationTests(unittest.TestCase):
    def test_filters_and_coefficient_hashes_match_locked_T19b(self) -> None:
        prior = probe.shared.read_json(probe.ROOT / "reports/shortcut_probe_dev.json")
        filters = probe.locked_specification(prior)
        self.assertEqual(filters, prior["filters"])
        self.assertEqual(filters["lowpass_7k"]["coefficient_float32_sha256"],
                         "973b82a36db8dab4f802de2102c5feed2bfb8cf6bb5846472c07d9984ab4c977")
        self.assertEqual(filters["lowpass_4k"]["coefficient_float32_sha256"],
                         "72a4f352a618490a0d5f21e59d890d6841236aca8d2321b457f9c98953a7d276")
        self.assertEqual(probe.INDICATOR_RULES, prior["indicator_rules"])

    def test_changed_specification_is_rejected(self) -> None:
        prior = probe.shared.read_json(probe.ROOT / "reports/shortcut_probe_dev.json")
        prior["filters"]["lowpass_7k"]["taps"] = 511
        with self.assertRaisesRegex(ValueError, "Bộ lọc"):
            probe.locked_specification(prior)

    def test_subset_is_deterministic_and_exact(self) -> None:
        rows = probe.shared.read_dev_manifest(probe.ROOT / "data/manifests/closed_dev_full_v1.csv")
        selected = probe.shared.subset_indices(rows)
        files = [rows[i]["file"] for i in selected]
        self.assertEqual(probe.shared.file_list_sha(files), probe.SUBSET_SHA)
        self.assertEqual(len(set(files)), 3120)
        self.assertEqual([sum(rows[i]["utt_type"] == g for i in selected) for g in probe.shared.GROUPS],
                         [1000, 1000, 1000, 120])
        reversed_rows = rows[::-1]
        self.assertEqual(files, [reversed_rows[i]["file"] for i in probe.shared.subset_indices(reversed_rows)])

    def test_test_manifest_and_artifact_are_rejected_before_open(self) -> None:
        with patch.object(Path, "open", side_effect=AssertionError("Không được mở test")):
            for call in (probe.shared.read_dev_manifest, probe.shared.open_dev_artifact):
                with self.assertRaisesRegex(ValueError, "test"):
                    call(Path("closed_test_full_v1"))

    def test_no_B0_inference_for_final_model(self) -> None:
        model = probe.FinalModel({}, None)
        with patch.object(model, "load", side_effect=AssertionError("Không được nạp")):
            with self.assertRaisesRegex(ValueError, "không chạy B0"):
                model.predict([], ["B0-v2-seed2026"], "original")


class ReproductionTests(unittest.TestCase):
    def test_tiny_probability_difference_but_large_logit_is_rejected(self) -> None:
        # Δp=1e-9 đạt tiêu chí B0, nhưng Δlogit≈log(2)>0,05 phải bị bắt.
        value = probe.check_logit_reproduction([float(probe.shared.stable_logit([2e-9])[0])], [1e-9],
                                               observed_scores=[2e-9])
        self.assertFalse(value["passed"])
        self.assertGreater(value["max_abs_logit_difference"], .05)
        self.assertEqual(value["observed_logit_source"], "block.raw_logit")

    def test_threshold_flip_inside_margin_is_accepted(self) -> None:
        threshold = .00015767214536590907
        boundary = probe.shared.stable_logit([threshold])[0]
        reference = 1 / (1 + math.exp(-(boundary - .01)))
        result = probe.check_logit_reproduction([boundary + .01], [reference], threshold)
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision_flips"], 1)

    def test_flip_outside_margin_is_rejected(self) -> None:
        boundary = probe.shared.stable_logit([probe.THRESHOLD])[0]
        reference = 1 / (1 + math.exp(-(boundary - .06)))
        result = probe.check_logit_reproduction([boundary + .001], [reference])
        self.assertFalse(result["passed"])
        self.assertEqual(result["flips_outside_0_05_margin"], 1)

    def test_raw_logit_is_used_instead_of_recovering_observed_probability(self) -> None:
        result = probe.check_logit_reproduction([-19.], [1e-9], observed_scores=[1e-9])
        self.assertFalse(result["passed"])
        self.assertGreater(result["max_abs_logit_difference"], 1)

    def test_invalid_or_missing_logits_are_rejected(self) -> None:
        for logits, reference in (([], []), ([float("nan")], [.1]), ([0., 1.], [.5])):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                probe.check_logit_reproduction(logits, reference)


class BlocksTests(unittest.TestCase):
    def setUp(self) -> None:
        (probe.CACHE / "temp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=probe.CACHE / "temp")
        self.root = Path(self.temp.name)
        self.identity = {"models": {probe.MODEL: {"checkpoint_sha256": probe.CHECKPOINT_SHA}},
                         "filters": {"original": probe.shared.filter_parameters("original")}}
        self.store = probe.shared.BlockStore(self.root / "blocks", self.identity)
        self.records = [{"file": "a", "score": .5, "raw_logit": 0.}]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_resume_reuses_block_without_prediction(self) -> None:
        self.store.save(probe.MODEL, "original", 0, ["a"], self.records, .1)
        predict = Mock(side_effect=AssertionError("Không được chấm lại"))
        events = Mock()
        probe.shared.process_blocks(self.store, [probe.MODEL], "original", [0], [{"file": "a"}],
                                    predict, probe.shared.Deadline(8, clock=lambda: 0.), events)
        predict.assert_not_called()
        self.assertEqual(events.call_args.args[0], "block_reused")
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_mismatched_checkpoint_or_filter_is_rejected_without_overwrite(self) -> None:
        self.store.save(probe.MODEL, "original", 0, ["a"], self.records, .1)
        path = self.store.path(probe.MODEL, "original", 0)
        before = path.read_bytes()
        for field in ("checkpoint", "filter"):
            identity = deepcopy(self.identity)
            if field == "checkpoint":
                identity["models"][probe.MODEL]["checkpoint_sha256"] = "f" * 64
            else:
                identity["filters"]["original"]["repeak"] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "không ghi đè"):
                probe.shared.BlockStore(self.root / "blocks", identity).load(probe.MODEL, "original", 0, ["a"])
            self.assertEqual(path.read_bytes(), before)

    def test_changed_file_order_is_rejected(self) -> None:
        self.store.save(probe.MODEL, "original", 0, ["a"], self.records, .1)
        with self.assertRaisesRegex(ValueError, "không ghi đè"):
            self.store.load(probe.MODEL, "original", 0, ["b"])

    def test_atomic_failure_does_not_publish_partial_block(self) -> None:
        with patch.object(probe.shared.os, "replace", side_effect=OSError("Giả lập ngắt")):
            with self.assertRaises(OSError):
                self.store.save(probe.MODEL, "original", 0, ["a"], self.records, .1)
        self.assertFalse(self.store.path(probe.MODEL, "original", 0).exists())
        self.assertFalse(list(self.root.rglob("*.tmp")))


class OrchestrationTests(unittest.TestCase):
    def fake_probe(self) -> tuple[probe.Probe, list[tuple]]:
        value = probe.Probe.__new__(probe.Probe)
        calls: list[tuple] = []
        value.invocation = {}
        value.dataset = Mock()
        value.blocks = lambda condition, limit=None: calls.append(("blocks", condition, limit))
        value.verify_original = lambda count=None: calls.append(("verify", count))
        value.phase = lambda stage, action: (calls.append(("stage", stage)), action())
        value.emit = lambda *args, **kwargs: None
        value.publish = lambda status, *args: calls.append(("publish", status))
        return value, calls

    def test_stage_order_has_both_reproduction_gates(self) -> None:
        value, calls = self.fake_probe()
        self.assertEqual(value.run(), 0)
        self.assertEqual([c for c in calls if c[0] in ("blocks", "verify")],
            [("blocks", "original", 200), ("verify", 200), ("blocks", "original", None),
             ("verify", None), ("blocks", "roundtrip_40k", None),
             ("blocks", "lowpass_7k", None), ("blocks", "lowpass_4k", None)])
        value.dataset.close.assert_called_once()

    def test_first200_failure_stops_before_any_other_condition(self) -> None:
        value, calls = self.fake_probe()
        value.verify_original = Mock(side_effect=probe.shared.ReproductionFailure("Sai tái tạo"))
        self.assertEqual(value.run(), 3)
        self.assertEqual([c for c in calls if c[0] == "blocks"], [("blocks", "original", 200)])
        self.assertIn(("publish", "blocked_reproduction"), calls)

    def test_deadline_publishes_partial_and_stops(self) -> None:
        value, calls = self.fake_probe()
        value.blocks = Mock(side_effect=probe.shared.DeadlineReached("Hết8giờ"))
        self.assertEqual(value.run(), 2)
        self.assertIn(("publish", "paused_deadline"), calls)
        value.dataset.close.assert_called_once()

    def test_report_destinations_are_separate_from_T19b(self) -> None:
        self.assertEqual(probe.CACHE, probe.ROOT / ".cache/t22")
        self.assertEqual(probe.REPORT_JSON.name, "shortcut_probe_dev_x1_final.json")
        self.assertEqual(probe.REPORT_MD.name, "shortcut_probe_dev_x1_final.md")
        old_paths = [probe.ROOT / "scripts/probe_shortcut_dev.py",
                     probe.ROOT / "reports/shortcut_probe_dev.json", probe.ROOT / "reports/shortcut_probe_dev.md"]
        before = [probe.shared.sha256_file(p) for p in old_paths]
        (probe.CACHE / "temp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=probe.CACHE / "temp") as directory:
            root = Path(directory)
            report = {"status": "running", "reproduction": {}, "comparison_data_only": {}, "results": {},
                      "filters": {}, "timing": {}, "interruption_count": 0, "resume_count": 0,
                      "interpretation_limit": "Chỉ báo, không kết luận."}
            with patch.object(probe, "CACHE", root), patch.object(probe, "REPORT_MD", root / "final.md"):
                probe.write_markdown(report)
            self.assertTrue((root / "final.md").exists())
        self.assertEqual(before, [probe.shared.sha256_file(p) for p in old_paths])


if __name__ == "__main__":
    unittest.main()
