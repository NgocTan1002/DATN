"""T23: gate từ chối và khóa trước opener, chỉ dùng file giả lập."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scripts import evaluate_full_split as evaluate
from scripts import build_model_registry as registry


class TestGateTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = evaluate.ROOT / ".cache/t23/temp"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.root = Path(self.temporary.name)
        self.config = registry.load_json(registry.CONFIG)
        self.config.update(test_gate_open=True, registry_path="official.json", t22_report="t22.json", locks_dir="locks")
        self.config["claude_verified"] = [registry.model_id(x) for x in self.config["required_models"]]
        self.config["pre_test_checks"] = {k: True for k in self.config["pre_test_checks"]}
        self.entries = [{**x, "model_id": registry.model_id(x), "status": "present", "claude_verified": True,
                         "checkpoint_sha256": "a" * 64} for x in self.config["required_models"]]
        self.value = {"decision": "D032", "complete": True, "protocol_sha256": registry.protocol_sha256(self.config), "entries": self.entries}
        self.write_registry()
        self.t22 = {"status": "complete", "use_test": False, "subset_files_sha256": evaluate.shared.SUBSET_SHA,
                    "identity": {"models": {"X1-L8-lr3e-5-seed2026": {"checkpoint_sha256": "a" * 64}}},
                    "results": {"X1-L8-lr3e-5-seed2026": {"indicators": {"available_indicator_level": "weak"}}}}
        self.write_t22()
        self.selected = [self.entries[0]["model_id"]]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_registry(self) -> None:
        path = self.root / self.config["registry_path"]
        path.write_text(json.dumps(self.value), encoding="utf-8")
        self.config["registry_sha256"] = registry.sha256_file(path)

    def write_t22(self) -> None:
        (self.root / self.config["t22_report"]).write_text(json.dumps(self.t22), encoding="utf-8")

    def refused(self, pattern: str) -> None:
        opener = Mock(side_effect=AssertionError("Không được mở dữ liệu"))
        with self.assertRaisesRegex(ValueError, pattern):
            evaluate.test_data_after_lock(self.config, self.selected, opener, self.root)
        opener.assert_not_called()

    def test_closed_gate_rejects_before_any_file_open(self) -> None:
        self.config["test_gate_open"] = False
        with patch.object(Path, "open", side_effect=AssertionError("Không mở file")):
            self.refused("đóng")

    def test_missing_registry(self) -> None:
        (self.root / self.config["registry_path"]).unlink()
        self.refused("Thiếu registry")

    def test_incomplete_registry(self) -> None:
        self.value["complete"] = False
        self.write_registry()
        self.refused("chưa đủ9")

    def test_complete_flag_does_not_allow_missing_entry(self) -> None:
        self.value["entries"][0]["status"] = "missing"
        self.write_registry()
        self.refused("chưa đủ9")

    def test_registry_sha_mismatch(self) -> None:
        self.config["registry_sha256"] = "b" * 64
        self.refused("SHA-256")

    def test_registry_must_match_config(self) -> None:
        self.value["entries"][0]["training_seed"] = 2028
        self.write_registry()
        self.refused("lệch")

    def test_missing_T22(self) -> None:
        (self.root / self.config["t22_report"]).unlink()
        self.refused("Thiếu report T22")

    def test_T22_incomplete_or_blocked(self) -> None:
        self.t22["status"] = "blocked_reproduction"
        self.write_t22()
        self.refused("chưa hoàn tất")

    def test_T22_strong_or_unavailable(self) -> None:
        for level in ("strong", "not_available"):
            self.t22["results"]["X1-L8-lr3e-5-seed2026"]["indicators"]["available_indicator_level"] = level
            self.write_t22()
            self.refused("weak")

    def test_T22_wrong_checkpoint(self) -> None:
        self.t22["identity"]["models"]["X1-L8-lr3e-5-seed2026"]["checkpoint_sha256"] = "b" * 64
        self.write_t22()
        self.refused("checkpoint")

    def test_missing_D032_clause8_attestations(self) -> None:
        self.config["pre_test_checks"]["bootstrap_dev_claude_verified"] = False
        self.refused("D032 mục8")

    def test_existing_lock_refuses_repeated_read(self) -> None:
        directory = self.root / "locks"
        directory.mkdir()
        (directory / (self.selected[0] + ".lock")).write_text("old", encoding="utf-8")
        self.refused("Đã có khóa")

    def test_exclusive_lock_exists_before_opener_and_is_retained(self) -> None:
        lock = self.root / "locks" / (self.selected[0] + ".lock")
        def opener() -> str:
            self.assertTrue(lock.is_file())
            return "fixture opened"
        self.assertEqual(evaluate.test_data_after_lock(self.config, self.selected, opener, self.root), "fixture opened")
        self.refused("Đã có khóa")

    def test_failure_after_lock_never_deletes_it(self) -> None:
        with self.assertRaises(RuntimeError):
            evaluate.test_data_after_lock(self.config, self.selected, Mock(side_effect=RuntimeError("Giả lập lỗi đọc")), self.root)
        self.assertTrue((self.root / "locks" / (self.selected[0] + ".lock")).is_file())

    def test_gate_closed_in_run_never_calls_dataset(self) -> None:
        self.config["test_gate_open"] = False
        with patch.object(evaluate, "open_split", side_effect=AssertionError("Không mở packed")) as loader:
            with self.assertRaisesRegex(ValueError, "đóng"):
                evaluate.run(self.config, self.value, "test", self.selected, None, None, self.root / "out.json", self.root)
        loader.assert_not_called()

    def test_run_uses_verified_registry_bytes_not_callers_checkpoint(self) -> None:
        altered = deepcopy(self.value)
        altered["entries"][0]["checkpoint_sha256"] = "b" * 64
        dataset = Mock()
        with patch.object(evaluate, "open_split", return_value=([], dataset)), patch.object(evaluate, "Models", side_effect=RuntimeError("Dừng trước inference")) as models:
            with self.assertRaisesRegex(RuntimeError, "Dừng trước inference"):
                evaluate.run(self.config, altered, "test", self.selected, None, None, self.root / "out.json", self.root)
        self.assertEqual(models.call_args.args[0][0]["checkpoint_sha256"], "a" * 64)
        dataset.close.assert_called_once()

    def test_dev_refuses_test_path_before_hash_or_open(self) -> None:
        config = deepcopy(self.config)
        config["dev"]["manifest"] = "closed_test_full_v1.csv"
        with patch.object(evaluate, "sha256_file", side_effect=AssertionError("Không được băm")):
            with self.assertRaisesRegex(ValueError, "test"):
                evaluate.open_split(config, "dev", self.root)


class MetricTests(unittest.TestCase):
    def test_fixed_threshold_metrics(self) -> None:
        records = [{"label": y, "score": p, "utt_type": group} for y, p, group in
                   [(0, .3, "bonafide"), (0, .8, "bonafide"), (1, .7, "voice_conversion"), (1, .2, "adversarial_attack")]]
        result = evaluate.summarize(records, .4)
        self.assertEqual(result["decision_threshold"], .4)
        self.assertEqual(result["macro_f1"], .5)
        self.assertEqual(result["fpr"], .5)
        self.assertEqual(result["fnr"], .5)

    def test_one_class_technical_subset_has_no_scientific_metrics(self) -> None:
        result = evaluate.summarize([{"label": 0, "score": .1, "utt_type": "bonafide"}], .5)
        self.assertIsNone(result["eer"])
        self.assertIsNone(result["roc_auc"])
        self.assertEqual(result["fpr"], 0)

    def test_b0_reproduction_and_flips(self) -> None:
        reference = {"a": {"score": .5}}
        result = evaluate.reproduction([{"file": "a", "score": .500001}], reference, .5000005, "B0-v2")
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision_flips"], 1)

    def test_xlsr_reproduction_uses_logit_for_both_X_families(self) -> None:
        reference = {"a": {"score": 1e-9}}
        for family in ("X0", "X1"):
            result = evaluate.reproduction([{"file": "a", "score": 2e-9, "raw_logit": float(evaluate.shared.stable_logit([2e-9])[0])}], reference, .5, family)
            self.assertFalse(result["passed"])
            self.assertGreater(result["max_abs_logit_difference"], .05)

    def test_xlsr_threshold_flip_inside_logit_margin_is_accepted(self) -> None:
        boundary = evaluate.shared.stable_logit([.001])[0]
        old, new = 1 / (1 + np.exp(-(boundary - .01))), 1 / (1 + np.exp(-(boundary + .01)))
        result = evaluate.reproduction([{"file": "a", "score": new, "raw_logit": boundary + .01}], {"a": {"score": old}}, .001, "X1")
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision_flips"], 1)


if __name__ == "__main__":
    unittest.main()
