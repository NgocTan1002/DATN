"""PREGATE-01: file cấp phép chấm test/dev (D037 mục 4–5); chỉ dùng fixture trong thư mục tạm, không đọc dữ liệu test thật."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import build_model_registry as registry
from scripts import evaluate_full_split as evaluate

MODEL = "X1-L8-lr3e-5-seed2026"
B0 = "B0-v2-seed2026"
PINNED = {"torch": "2.11.0+cu130", "cuda": "13.0", "transformers": "4.57.6", "numpy": "2.1.3"}


class _FakeScorer:
    """Thay Models: trả score tất định, đếm số lần chấm."""
    calls: list[tuple[str, str]] = []

    def __init__(self, entries: list[dict], root: Path, **kwargs: object) -> None:
        self.entries = entries

    def score(self, names: list[str], indices: list[int], rows: list[dict], dataset: object, condition: str) -> dict:
        _FakeScorer.calls.append((names[0], condition))
        return {names[0]: [{"file": rows[i]["file"], "speaker_id": "s", "utt_type": rows[i]["utt_type"],
                            "label": int(rows[i]["binary_label"]), "score": 0.2 + 0.6 * int(rows[i]["binary_label"]),
                            "raw_logit": 0.0} for i in indices]}


class PermitBase(unittest.TestCase):
    def setUp(self) -> None:
        temp = evaluate.ROOT / ".cache/pregate01/temp_tests"
        temp.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temp)
        self.root = Path(self.directory.name)
        (self.root / "data/fake_dev_packed").mkdir(parents=True)
        (self.root / "data/fake_dev.csv").write_text("file,speaker_id,utt_type,binary_label,shard\n", encoding="utf-8")
        art = self.root / "data/fake_dev_packed"
        (art / "index.csv").write_bytes(b"index fixture")
        (art / "waveforms.f32").write_bytes(b"waveform fixture")
        manifest_sha = registry.sha256_file(self.root / "data/fake_dev.csv")
        (art / "metadata.json").write_text(json.dumps({
            "source_manifest_sha256": manifest_sha, "index_sha256": registry.sha256_file(art / "index.csv"),
            "waveforms_sha256": registry.sha256_file(art / "waveforms.f32")}), encoding="utf-8")
        self.config = registry.load_json(registry.CONFIG)
        self.config.update(registry_path="reports/model_registry_v1.json", t22_report="t22.json", locks_dir="locks")
        self.config["dev"] = {"manifest": "data/fake_dev.csv", "manifest_sha256": manifest_sha,
                              "artifact": "data/fake_dev_packed", "samples": 4, "shards": 1}
        self.config["test"] = {"manifest": "data/fake_test.csv", "manifest_sha256": "c" * 64,
                               "artifact": "data/fake_test_packed", "samples": 4}
        self.entries = []
        (self.root / "ckpt").mkdir()
        for spec in self.config["required_models"]:
            name = registry.model_id(spec)
            (self.root / "ckpt" / (name + ".pt")).write_bytes(b"ckpt " + name.encode())
            self.entries.append({**spec, "model_id": name, "status": "present", "claude_verified": True,
                "checkpoint": f"ckpt/{name}.pt", "checkpoint_sha256": registry.sha256_file(self.root / "ckpt" / (name + ".pt")),
                "decision_threshold": 0.5, "source_report_sha256": "d" * 64})
        self.registry = {"decision": "D032", "role": "official", "complete": True,
                         "protocol_sha256": registry.protocol_sha256(self.config), "entries": self.entries}
        self.write_registry()
        self.permit_path = self.root / "permit.json"
        _FakeScorer.calls = []

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_registry(self) -> None:
        path = self.root / self.config["registry_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.registry), encoding="utf-8")
        self.config["registry_sha256"] = registry.sha256_file(path)

    def make_permit(self, selected: list[str] | None = None, conditions: list[str] | None = None) -> dict:
        return evaluate.write_permit(self.config, "dev", selected or [MODEL], conditions, self.permit_path, self.root)

    def resign(self, mutate) -> None:
        permit = registry.load_json(self.permit_path)
        mutate(permit)
        permit["permit_sha256"] = evaluate.permit_digest(permit)
        self.permit_path.write_text(json.dumps(permit), encoding="utf-8")

    def refused(self, pattern: str, **overrides: object) -> None:
        opener = Mock(side_effect=AssertionError("Không được mở dữ liệu"))
        args = dict(permit_path=self.permit_path, config=self.config, registry_path=self.root / self.config["registry_path"],
                    data_root=self.root / "data", checkpoint_root=self.root, device="cpu", output_dir=self.root / "out",
                    root=self.root, reference_root=self.root, opener=opener)
        args.update(overrides)
        with patch.object(evaluate, "Models", side_effect=AssertionError("Không được tạo mô hình")):
            with self.assertRaisesRegex(ValueError, pattern):
                evaluate.run_with_permit(**args)
        opener.assert_not_called()

    def rows(self) -> list[dict]:
        return [{"file": f"f{i}", "speaker_id": "s", "utt_type": "bonafide" if i % 2 == 0 else "voice_conversion",
                 "binary_label": str(i % 2), "shard": "0"} for i in range(4)]

    def run_ok(self, **overrides: object) -> tuple[dict, Mock]:
        dataset = Mock()
        opener = Mock(return_value=(self.rows(), dataset))
        args = dict(permit_path=self.permit_path, config=self.config, registry_path=self.root / self.config["registry_path"],
                    data_root=self.root / "data", checkpoint_root=self.root, device="cpu", output_dir=self.root / "out",
                    root=self.root, reference_root=self.root, opener=opener)
        args.update(overrides)
        with patch.object(evaluate, "Models", _FakeScorer):
            return evaluate.run_with_permit(**args), opener


class WritePermitTests(PermitBase):
    def test_dev_permit_written_without_gate_or_locks(self) -> None:
        self.assertFalse(self.config["test_gate_open"])
        permit = self.make_permit()
        self.assertEqual(permit["split"], "dev")
        self.assertEqual(permit["permit_sha256"], evaluate.permit_digest(permit))
        self.assertEqual(permit["models"][0]["checkpoint_sha256"], self.entries[6]["checkpoint_sha256"])
        self.assertEqual(permit["models"][0]["conditions"], ["original"])
        self.assertEqual(permit["registry_sha256"], registry.sha256_file(self.root / self.config["registry_path"]))
        self.assertNotIn("lock_sha256", permit)
        self.assertFalse((self.root / "locks").exists())

    def test_permit_conditions_must_belong_to_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "conditions_by_family"):
            self.make_permit(conditions=["nonexistent"])
        with self.assertRaisesRegex(ValueError, "conditions_by_family"):
            self.make_permit(selected=["X0-seed2026"], conditions=["lowpass_4k"])
        self.assertFalse(self.permit_path.exists())

    def test_permit_not_overwritten(self) -> None:
        self.make_permit()
        with self.assertRaisesRegex(ValueError, "không ghi đè"):
            self.make_permit()

    def test_test_permit_refused_when_gate_closed_and_no_lock(self) -> None:
        self.assertFalse(self.config["test_gate_open"])
        for selected in (None, [MODEL]):
            with self.subTest(selected=selected), patch.object(Path, "open", side_effect=AssertionError("Không mở file")):
                with self.assertRaisesRegex(ValueError, "đóng"):
                    evaluate.write_permit(self.config, "test", selected, None, self.permit_path, self.root)
        self.assertFalse(self.permit_path.exists())
        self.assertFalse((self.root / "locks").exists())

    def test_test_permit_with_open_gate_creates_locks_and_records_hashes(self) -> None:
        config = deepcopy(self.config)
        config["test_gate_open"] = True
        config["pre_test_checks"] = {k: True for k in config["pre_test_checks"]}
        t22 = {"status": "complete", "use_test": False, "subset_files_sha256": evaluate.shared.SUBSET_SHA,
               "identity": {"models": {MODEL: {"checkpoint_sha256": self.entries[6]["checkpoint_sha256"]}}},
               "results": {MODEL: {"indicators": {"available_indicator_level": "weak"}}}}
        (self.root / "t22.json").write_text(json.dumps(t22), encoding="utf-8")
        (self.root / "reports").mkdir(exist_ok=True)
        inventory = {"local_test_inventory": {"files": {k: {"sha256": v} for k, v in
                     (("manifest", "c" * 64), ("index", "1" * 64), ("waveforms", "2" * 64))}}}
        (self.root / "reports/pre_test_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
        permit = evaluate.write_permit(config, "test", [MODEL], None, self.permit_path, self.root)
        lock = self.root / "locks" / (MODEL + ".lock")
        self.assertTrue(lock.is_file())
        self.assertEqual(permit["lock_sha256"], {MODEL: registry.sha256_file(lock)})
        self.assertEqual(permit["artifacts"]["waveforms_sha256"], "2" * 64)
        self.assertEqual(permit["models"][0]["conditions"], config["conditions_by_family"]["X1"])
        with self.assertRaisesRegex(ValueError, "không ghi đè|khóa"):
            evaluate.write_permit(config, "test", [MODEL], None, self.root / "permit2.json", self.root)


class RunWithPermitRefusalTests(PermitBase):
    def setUp(self) -> None:
        super().setUp()
        self.make_permit()

    def test_missing_permit(self) -> None:
        self.refused("Thiếu file cấp phép", permit_path=self.root / "none.json")

    def test_tampered_permit_sha(self) -> None:
        permit = registry.load_json(self.permit_path)
        permit["registry_sha256"] = "0" * 64
        self.permit_path.write_text(json.dumps(permit), encoding="utf-8")
        self.refused("permit_sha256")

    def test_registry_sha_mismatch(self) -> None:
        path = self.root / self.config["registry_path"]
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.refused("registry không khớp cấp phép")

    def test_checkpoint_sha_mismatch_on_disk(self) -> None:
        (self.root / self.entries[6]["checkpoint"]).write_bytes(b"changed")
        self.refused("Checkpoint đã đổi")

    def test_checkpoint_sha_mismatch_in_resigned_permit(self) -> None:
        self.resign(lambda p: p["models"][0].update(checkpoint_sha256="0" * 64))
        self.refused("Checkpoint trong cấp phép lệch registry")

    def test_artifact_sha_mismatch(self) -> None:
        for name, pattern in (("index.csv", "index.csv"), ("waveforms.f32", "waveforms.f32")):
            path = self.root / "data/fake_dev_packed" / name
            original = path.read_bytes()
            path.write_bytes(b"changed")
            with self.subTest(name=name):
                self.refused(pattern)
            path.write_bytes(original)
        manifest = self.root / "data/fake_dev.csv"
        manifest.write_text("khác\n", encoding="utf-8")
        self.refused("Manifest split lệch")

    def test_model_outside_permit(self) -> None:
        self.refused("ngoài danh sách cấp phép", models=["X0-seed2026"])

    def test_condition_outside_permit(self) -> None:
        self.refused("Điều kiện yêu cầu nằm ngoài", conditions=["lowpass_4k"])

    def test_config_protocol_mismatch(self) -> None:
        config = deepcopy(self.config)
        config["bootstrap"] = {"n_resamples": 10, "seed": 1, "confidence": .9}
        self.refused("giao thức|bootstrap|Bootstrap", config=config)

    def test_cuda_without_gpu_stops(self) -> None:
        with patch.dict(evaluate._DEVICE_OVERRIDES, {"cuda_available": False}):
            self.refused("không có GPU", device="cuda")

    def test_cuda_version_mismatch_stops(self) -> None:
        bad = {**PINNED, "torch": "2.0.0"}
        with patch.dict(evaluate._DEVICE_OVERRIDES, {"cuda_available": True, "gpu_name": "L4", "pinned": PINNED, "versions": bad}):
            self.refused("lệch bản đã khóa", device="cuda")

    def test_limit_not_allowed_for_test_permit(self) -> None:
        self.resign(lambda p: p.update(split="test"))
        self.refused("không dùng limit", limit=10)


class B0DeviceTests(PermitBase):
    def test_b0_with_cuda_rejected_before_data(self) -> None:
        self.make_permit(selected=[B0])
        with patch.dict(evaluate._DEVICE_OVERRIDES, {"cuda_available": True, "gpu_name": "L4", "pinned": PINNED, "versions": PINNED}):
            self.refused("B0-v2 chỉ được chấm trên CPU", device="cuda")

    def test_models_load_b0_on_cuda_rejected(self) -> None:
        models = evaluate.Models([self.entries[0]], self.root, device="cuda")
        with self.assertRaisesRegex(ValueError, "B0-v2 chỉ được chấm trên CPU"):
            models.load(B0)

    def test_check_device_cpu_and_unknown(self) -> None:
        self.assertEqual(evaluate.check_device("cpu", {"B0-v2"}, versions=PINNED)["device"], "cpu")
        with self.assertRaisesRegex(ValueError, "cpu hoặc cuda"):
            evaluate.check_device("tpu", set())

    def test_pinned_versions_read_from_entry_file(self) -> None:
        text = ("EXPECTED_TORCH_FULL = '9.9.9+cu1'\nEXPECTED_CUDA_VERSION = '1.0'\n"
                "EXPECTED_TRANSFORMERS_VERSION = '1.2.3'\nEXPECTED_NUMPY_VERSION = '4.5.6'\n")
        path = self.root / "entry.py"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(evaluate.load_colab_pinned_versions(path),
                         {"torch": "9.9.9+cu1", "cuda": "1.0", "transformers": "1.2.3", "numpy": "4.5.6"})
        with self.assertRaisesRegex(ValueError, "Không tìm thấy"):
            evaluate.load_colab_pinned_versions(self.root / "missing.py")


class ResumeTests(PermitBase):
    def setUp(self) -> None:
        super().setUp()
        self.make_permit()
        self.result = self.root / "out" / f"dev__{MODEL}__original.json"

    def test_complete_result_skipped_and_partial_rescored(self) -> None:
        outcome, opener = self.run_ok()
        self.assertEqual(outcome["completed"], [f"{MODEL}|original"])
        self.assertEqual(_FakeScorer.calls, [(MODEL, "original")])
        first = registry.load_json(self.result)
        self.assertTrue(first["complete"])
        self.assertEqual(first["permit_sha256"], registry.load_json(self.permit_path)["permit_sha256"])
        self.assertEqual(first["samples"], 4)
        self.assertEqual(first["device"], "cpu")
        before = self.result.read_bytes()
        outcome, opener = self.run_ok()
        self.assertEqual(outcome["completed"], [])
        self.assertEqual(outcome["skipped"], [f"{MODEL}|original"])
        opener.assert_not_called()
        self.assertEqual(len(_FakeScorer.calls), 1)
        self.assertEqual(self.result.read_bytes(), before)
        partial = dict(first, complete=False)
        self.result.write_text(json.dumps(partial), encoding="utf-8")
        outcome, _ = self.run_ok()
        self.assertEqual(outcome["completed"], [f"{MODEL}|original"])
        self.assertEqual(len(_FakeScorer.calls), 2)
        self.assertTrue(registry.load_json(self.result)["complete"])

    def test_stub_without_flag_is_rescored(self) -> None:
        self.result.parent.mkdir(parents=True)
        self.result.write_text("{không phải json", encoding="utf-8")
        outcome, _ = self.run_ok()
        self.assertEqual(outcome["completed"], [f"{MODEL}|original"])

    def test_complete_result_of_other_permit_not_overwritten(self) -> None:
        self.run_ok()
        before = self.result.read_bytes()
        other = registry.load_json(self.result)
        other["permit_sha256"] = "0" * 64
        self.result.write_text(json.dumps(other), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cấp phép khác"):
            self.run_ok()
        self.assertEqual(registry.load_json(self.result)["permit_sha256"], "0" * 64)
        self.assertNotEqual(before, b"")

    def test_run_never_creates_locks(self) -> None:
        self.run_ok()
        self.assertFalse((self.root / "locks").exists())


class LegacyAndRealConfigTests(unittest.TestCase):
    def test_legacy_test_split_still_refused_when_gate_closed(self) -> None:
        locks = evaluate.ROOT / ".cache/test_once"
        before = sorted(p.name for p in locks.rglob("*")) if locks.exists() else []
        argv = ["evaluate_full_split.py", "--split", "test", "--output", str(evaluate.ROOT / ".cache/pregate01/should_not_exist.json")]
        with patch.object(sys, "argv", argv), patch.object(evaluate, "open_split", side_effect=AssertionError("Không mở dữ liệu")):
            self.assertEqual(evaluate.main(), 1)
        after = sorted(p.name for p in locks.rglob("*")) if locks.exists() else []
        self.assertEqual(before, after)
        self.assertFalse((evaluate.ROOT / ".cache/pregate01/should_not_exist.json").exists())

    def test_real_write_permit_test_refused_with_closed_gate(self) -> None:
        for name in ("test_evaluation.json", "test_evaluation_x1r.json"):
            config = registry.load_json(evaluate.ROOT / "configs" / name)
            registry.validate_config(config)
            self.assertFalse(config["test_gate_open"])
            target = evaluate.ROOT / ".cache/pregate01/should_not_exist_permit.json"
            with self.assertRaisesRegex(ValueError, "đóng"):
                evaluate.write_permit(config, "test", None, None, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
