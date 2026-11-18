"""T23: gate từ chối và khóa trước opener, chỉ dùng file giả lập."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from scripts import evaluate_full_split as evaluate
from scripts import build_model_registry as registry
from src.data.xlsr import apply_xlsr_waveform_normalization


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


class _FakePackedDataset:
    def __init__(self, samples: list[dict[str, object]]) -> None:
        self._samples = samples
        self.closed = False

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._samples[index]

    def close(self) -> None:
        self.closed = True


class _CapturingDummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.captured_inputs: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.captured_inputs.append(inputs.detach().clone())
        # Tạo logit tất định khác nhau theo mẫu để kiểm EER/reproduction.
        return inputs.mean(dim=-1) * 10.0 - 0.5


class X1RGateAndEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = evaluate.ROOT / ".cache/test_x1r/temp_tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.root = Path(self.temporary.name)

        self.d032_config = registry.load_json(registry.CONFIG)
        self.d032_config["dev"] = {"manifest": "fake_dev.csv", "manifest_sha256": "0" * 64, "artifact": "fake_dev_packed", "samples": 4, "shards": 1}
        self.d032_config["test"] = {"manifest": "fake_eval_split.csv", "manifest_sha256": "0" * 64, "artifact": "fake_eval_packed", "samples": 4}
        self.d032_config.update(test_gate_open=True, registry_path="official_d032.json", t22_report="t22.json", locks_dir="locks_d032")
        self.d032_config["claude_verified"] = [registry.model_id(x) for x in self.d032_config["required_models"]]
        self.d032_config["pre_test_checks"] = {k: True for k in self.d032_config["pre_test_checks"]}
        self.write_d032_config()

        self.x1r_config = registry.load_json(evaluate.ROOT / "configs/test_evaluation_x1r.json")
        self.x1r_config["dev"] = {"manifest": "fake_dev.csv", "manifest_sha256": "0" * 64, "artifact": "fake_dev_packed", "samples": 4, "shards": 1}
        self.x1r_config["test"] = {"manifest": "fake_eval_split.csv", "manifest_sha256": "0" * 64, "artifact": "fake_eval_packed", "samples": 4}
        self.x1r_config.update(test_gate_open=True, registry_path="official_x1r.json", locks_dir="locks_x1r")
        self.x1r_config["claude_verified"] = [registry.model_id(x) for x in self.x1r_config["required_models"]]
        self.entries = [
            {
                **x,
                "model_id": registry.model_id(x),
                "input_condition": "lowpass_7k",
                "status": "present",
                "claude_verified": True,
                "checkpoint": f"fake_ckpt_{x['training_seed']}.pt",
                "checkpoint_sha256": "c" * 64,
                "decision_threshold": 0.4,
                "dev_eer": 0.01,
            }
            for x in self.x1r_config["required_models"]
        ]
        self.x1r_registry = {
            "decision": "D035",
            "complete": True,
            "protocol_sha256": registry.protocol_sha256(self.x1r_config),
            "input_condition": deepcopy(self.x1r_config["input_condition"]),
            "entries": deepcopy(self.entries),
        }
        self.write_x1r_registry()
        self.selected = [self.entries[0]["model_id"]]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_d032_config(self) -> None:
        path = self.root / "configs/test_evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.d032_config), encoding="utf-8")

    def write_x1r_registry(self) -> None:
        path = self.root / self.x1r_config["registry_path"]
        path.write_text(json.dumps(self.x1r_registry), encoding="utf-8")
        self.x1r_config["registry_sha256"] = registry.sha256_file(path)

    def refused_x1r(self, pattern: str) -> None:
        opener = Mock(side_effect=AssertionError("Không được mở dữ liệu"))
        with self.assertRaisesRegex(ValueError, pattern):
            evaluate.test_data_after_lock(self.x1r_config, self.selected, opener, self.root)
        opener.assert_not_called()

    def test_x1r_closed_gate_rejects_before_any_file_open(self) -> None:
        self.x1r_config["test_gate_open"] = False
        with patch.object(Path, "open", side_effect=AssertionError("Không mở file")):
            self.refused_x1r("đóng")

    def test_x1r_rejected_when_d032_gate_closed(self) -> None:
        self.d032_config["test_gate_open"] = False
        self.write_d032_config()
        self.refused_x1r("D032.*đóng")
        self.assertFalse((self.root / "locks_x1r").exists())

    def test_x1r_rejected_when_d032_missing_any_pre_test_check(self) -> None:
        for key in evaluate.PRE_TEST_CHECKS:
            with self.subTest(missing_check=key):
                self.d032_config["pre_test_checks"] = {k: True for k in evaluate.PRE_TEST_CHECKS}
                self.d032_config["pre_test_checks"][key] = False
                self.write_d032_config()
                self.refused_x1r("D032 mục8")
                self.assertFalse((self.root / "locks_x1r").exists())

    def test_x1r_rejected_when_d032_config_missing(self) -> None:
        (self.root / "configs/test_evaluation.json").unlink()
        self.refused_x1r("Thiếu cấu hình D032")

    def test_x1r_rejected_when_registry_sha_mismatch(self) -> None:
        self.x1r_config["registry_sha256"] = "d" * 64
        self.refused_x1r("SHA-256")
        self.assertFalse((self.root / "locks_x1r").exists())

    def test_x1r_rejected_when_registry_incomplete(self) -> None:
        self.x1r_registry["complete"] = False
        self.write_x1r_registry()
        self.refused_x1r("chưa đủ3")

    def test_x1r_rejected_when_existing_lock_present(self) -> None:
        locks = self.root / self.x1r_config["locks_dir"]
        locks.mkdir(parents=True, exist_ok=True)
        (locks / (self.selected[0] + ".lock")).write_text("old", encoding="utf-8")
        self.refused_x1r("Đã có khóa")

    def test_x1r_lock_created_before_opener_and_retained_without_separate_t22_file(self) -> None:
        # Không có file t22.json hay key t22_report trong x1r_config; chỉ cần điều kiện (b) trên D032.
        lock = self.root / self.x1r_config["locks_dir"] / (self.selected[0] + ".lock")

        def opener() -> str:
            self.assertTrue(lock.is_file())
            return "x1r opened"

        self.assertEqual(evaluate.test_data_after_lock(self.x1r_config, self.selected, opener, self.root), "x1r opened")
        self.refused_x1r("Đã có khóa")

        # Lỗi sau khi tạo khóa không xóa khóa.
        second = [self.entries[1]["model_id"]]
        with self.assertRaises(RuntimeError):
            evaluate.test_data_after_lock(self.x1r_config, second, Mock(side_effect=RuntimeError("Lỗi giả lập")), self.root)
        self.assertTrue((self.root / self.x1r_config["locks_dir"] / (second[0] + ".lock")).is_file())

    def _build_dev_fixture(self) -> tuple[list[dict[str, str]], _FakePackedDataset, _CapturingDummyModel]:
        torch.manual_seed(2026)
        rows = [
            {"file": "bf_1.wav", "speaker_id": "spk_b1", "utt_type": "bonafide", "binary_label": "0", "shard": "shard-000"},
            {"file": "vc_1.wav", "speaker_id": "spk_v1", "utt_type": "voice_conversion", "binary_label": "1", "shard": "shard-000"},
            {"file": "bf_2.wav", "speaker_id": "spk_b2", "utt_type": "bonafide", "binary_label": "0", "shard": "shard-000"},
            {"file": "ap_1.wav", "speaker_id": "spk_a1", "utt_type": "adversarial_attack", "binary_label": "1", "shard": "shard-000"},
        ]
        samples = []
        for idx, row in enumerate(rows):
            t = torch.linspace(0.0, 1.0, 64_000, dtype=torch.float32)
            freq = 400.0 + idx * 1500.0
            wave = (0.5 * torch.sin(2.0 * np.pi * freq * t) + 0.1 * (idx + 1)).to(torch.float32)
            samples.append({"file": row["file"], "waveform": wave, "utt_type": row["utt_type"]})
        dataset = _FakePackedDataset(samples)
        dummy_model = _CapturingDummyModel()

        # Tính score float32 khớp đúng 1 lần lowpass_7k + peak + XLS-R norm để lập source_report giả.
        filtered = torch.stack([evaluate.shared.apply_condition(s["waveform"], "lowpass_7k", s["utt_type"]) for s in samples])
        normed = apply_xlsr_waveform_normalization(filtered, "processor_zero_mean_unit_variance")
        with torch.inference_mode():
            logits = dummy_model(normed)
            scores = torch.sigmoid(logits).to(torch.float32).tolist()
        dummy_model.captured_inputs.clear()

        dev_records = [
            {
                "file": row["file"],
                "speaker_id": row["speaker_id"],
                "utt_type": row["utt_type"],
                "label": int(row["binary_label"]),
                "score": float(score),
            }
            for row, score in zip(rows, scores, strict=True)
        ]
        for entry in self.x1r_registry["entries"]:
            report_path = self.root / entry["source_report"]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps({"controller": {"best": {"development": {"records": dev_records}}}}),
                encoding="utf-8",
            )
            entry["source_report_sha256"] = registry.sha256_file(report_path)
        self.write_x1r_registry()
        return rows, dataset, dummy_model

    def test_x1r_dev_applies_lowpass_7k_once_and_computes_d034_and_seed_summary(self) -> None:
        rows, dataset, dummy_model = self._build_dev_fixture()
        all_ids = [e["model_id"] for e in self.x1r_registry["entries"]]
        expected_filtered = torch.stack(
            [evaluate.shared.apply_condition(dataset[i]["waveform"], "lowpass_7k", dataset[i]["utt_type"]) for i in range(4)]
        )
        expected_normed = apply_xlsr_waveform_normalization(expected_filtered, "processor_zero_mean_unit_variance")

        with patch.object(evaluate, "open_split", return_value=(rows, dataset)), patch.object(
            evaluate.Models, "load", return_value=dummy_model
        ), patch.object(evaluate.shared, "apply_condition", wraps=evaluate.shared.apply_condition) as spy_apply:
            report = evaluate.run(
                self.x1r_config,
                self.x1r_registry,
                "dev",
                all_ids,
                None,
                None,
                self.root / "eval_x1r_out.json",
                self.root,
            )

        self.assertEqual(report["decision"], "D035")
        self.assertEqual(report["status"], "complete")
        # 3 mô hình × 4 mẫu = 12 lần gọi apply_condition, tất cả đều là "lowpass_7k".
        self.assertEqual(spy_apply.call_count, 12)
        self.assertTrue(all(call.args[1] == "lowpass_7k" for call in spy_apply.call_args_list))
        # Batch đầu tiên (2 mẫu đầu) của model đầu khớp tuyệt đối với 1 lần lọc + norm (không lọc 2 lần).
        observed_inputs = torch.cat(dummy_model.captured_inputs[:2], dim=0)
        self.assertTrue(torch.allclose(observed_inputs, expected_normed, atol=1e-6))

        for mid in all_ids:
            self.assertEqual(list(report["models"][mid].keys()), ["lowpass_7k"])
            repro = report["models"][mid]["lowpass_7k"]["reproduction"]
            self.assertEqual(repro["criterion"], "D034")
            self.assertTrue(repro["passed"])
        self.assertIn("X1R", report["seed_summaries"])
        self.assertEqual(report["seed_summaries"]["X1R"]["ddof"], 1)
        self.assertTrue(dataset.closed)

    def test_x1r_rejects_other_conditions_and_bad_filter_sha(self) -> None:
        rows, dataset, dummy_model = self._build_dev_fixture()
        for bad_condition in ("original", "roundtrip_40k", "lowpass_4k"):
            with self.subTest(condition=bad_condition), patch.object(
                evaluate, "open_split", return_value=(rows, dataset)
            ), patch.object(evaluate.Models, "load", return_value=dummy_model):
                with self.assertRaisesRegex(ValueError, "Điều kiện không hợp lệ"):
                    evaluate.run(
                        self.x1r_config,
                        self.x1r_registry,
                        "dev",
                        self.selected,
                        None,
                        [bad_condition],
                        self.root / "bad_cond.json",
                        self.root,
                    )

        models = evaluate.Models(self.x1r_registry["entries"], self.root)
        with patch.object(models, "load", return_value=dummy_model):
            with self.assertRaisesRegex(ValueError, "lowpass_7k"):
                models.score(self.selected, [0, 1], rows, dataset, "original")

        bad_filter = deepcopy(evaluate.shared.filter_parameters("lowpass_7k"))
        bad_filter["coefficient_float32_sha256"] = "0" * 64
        with patch.object(evaluate.shared, "filter_parameters", return_value=bad_filter):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                evaluate.verify_lowpass_7k_filter(self.x1r_config)

    def test_x1r_d034_reproduction_uses_float32_quantization_gap(self) -> None:
        saturated = float(np.float32(0.99999905))
        base_logit = float(evaluate.shared.stable_logit([saturated])[0])
        # Độ lệch 0.06 > 0.05 nhưng nhỏ hơn 0.05 + g (~0.074 tại vùng bão hòa này).
        raw_logit = base_logit + 0.06
        score_now = float(1.0 / (1.0 + np.exp(-raw_logit)))
        records = [{"file": "a", "score": score_now, "raw_logit": raw_logit}]
        reference = {"a": {"score": saturated}}

        x1r_repro = evaluate.reproduction(records, reference, 0.5, "X1R")
        self.assertEqual(x1r_repro["criterion"], "D034")
        self.assertTrue(x1r_repro["passed"])
        self.assertGreater(x1r_repro["max_float32_gap_logit"], 0.05)

        x1_repro = evaluate.reproduction(records, reference, 0.5, "X1")
        self.assertFalse(x1_repro["passed"])

        too_far = [{"file": "a", "score": score_now, "raw_logit": base_logit + 0.2}]
        x1r_fail = evaluate.reproduction(too_far, reference, 0.5, "X1R")
        self.assertFalse(x1r_fail["passed"])

    def test_d032_rejects_x1r_and_d035_rejects_d032_families(self) -> None:
        d032_entries = [
            {
                **x,
                "model_id": registry.model_id(x),
                "status": "present",
                "claude_verified": True,
                "checkpoint_sha256": "a" * 64,
            }
            for x in self.d032_config["required_models"]
        ]
        d032_registry = {
            "decision": "D032",
            "complete": True,
            "protocol_sha256": registry.protocol_sha256(self.d032_config),
            "entries": d032_entries,
        }
        (self.root / self.d032_config["registry_path"]).write_text(json.dumps(d032_registry), encoding="utf-8")
        self.d032_config["registry_sha256"] = registry.sha256_file(self.root / self.d032_config["registry_path"])

        # D032 config không chấp nhận registry X1R hay model_id X1R.
        with self.assertRaises(ValueError):
            evaluate.run(self.d032_config, self.x1r_registry, "dev", self.selected, None, None, self.root / "m1.json", self.root)
        with self.assertRaisesRegex(ValueError, "không khớp"):
            evaluate.run(self.d032_config, d032_registry, "dev", self.selected, None, None, self.root / "m2.json", self.root)

        # D035 config không chấp nhận registry D032 hay model_id D032.
        with self.assertRaises(ValueError):
            evaluate.run(self.x1r_config, d032_registry, "dev", ["X1-L8-lr3e-5-seed2026"], None, None, self.root / "m3.json", self.root)
        with self.assertRaisesRegex(ValueError, "không khớp"):
            evaluate.run(self.x1r_config, self.x1r_registry, "dev", ["X1-L8-lr3e-5-seed2026"], None, None, self.root / "m4.json", self.root)

        bad_d032_cfg = deepcopy(self.d032_config)
        bad_d032_cfg["required_models"][-1]["family"] = "X1R"
        with self.assertRaises(ValueError):
            registry.validate_config(bad_d032_cfg)

        bad_d035_cfg = deepcopy(self.x1r_config)
        bad_d035_cfg["required_models"][0]["family"] = "X1"
        with self.assertRaises(ValueError):
            registry.validate_config(bad_d035_cfg)


if __name__ == "__main__":
    unittest.main()
