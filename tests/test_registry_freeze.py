"""PREGATE-01: đóng băng registry nháp thành registry chính thức (D037 mục 6); chỉ dùng fixture trong thư mục tạm."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import build_model_registry as registry


def make_source(root: Path, spec: dict, index: int, x1r_condition: dict | None = None) -> None:
    generation = f"g-{index:032x}"
    report_path = root / spec["source_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    path = (root / "checkpoints" / f"b0_v2_seed{spec['training_seed']}" / generation / "checkpoint.pt"
            if spec["family"] == "B0-v2" else report_path.parent.parent / "checkpoints" /
            f"xlsr_{spec['run_id']}_seed{spec['training_seed']}" / generation / "checkpoint.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint fixture " + str(index).encode())
    identity = {"run_id": spec["run_id"], "training_seed": spec["training_seed"], "data_seed": 2026, "source_tree_sha256": "a" * 64}
    if x1r_condition is not None:
        identity["input_condition"] = x1r_condition
    report = {"status": "early_stopped", "training_seed": spec["training_seed"], "development_samples": 38777,
              "source_tree_sha256": "a" * 64,
              "controller": {"best": {"generation": generation, "epoch": 2, "development": {"eer": .1, "decision_threshold": .25}}},
              "checkpoints": {"best": {"generation": generation, "checkpoint_sha256": registry.sha256_file(path)}}}
    if spec["family"] != "B0-v2":
        report["identity"] = identity
    report_path.write_text(json.dumps(report), encoding="utf-8")


class FreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = registry.ROOT / ".cache/pregate01/temp_tests"
        temp.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temp)
        self.root = Path(self.directory.name)
        self.config = registry.load_json(registry.CONFIG)
        for i, spec in enumerate(self.config["required_models"], 1):
            make_source(self.root, spec, i)
        self.draft = registry.build_registry(self.config, self.root)
        self.draft_path = self.root / "draft.json"
        registry.atomic_json(self.draft_path, self.draft)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def freeze(self, draft: dict | None = None) -> dict:
        return registry.freeze_registry(self.config, draft if draft is not None else self.draft, self.draft_path, self.root)

    def test_freeze_success_keeps_entries_and_adds_provenance(self) -> None:
        frozen = self.freeze()
        self.assertEqual(frozen["role"], "official")
        self.assertEqual(frozen["entries"], self.draft["entries"])
        self.assertEqual(frozen["draft_sha256"], registry.sha256_file(self.draft_path))
        self.assertIn("frozen_utc", frozen)
        self.assertTrue(frozen["complete"])
        self.assertEqual(len(frozen["entries"]), 9)
        registry.validate_registry(frozen, self.config, complete=True)

    def test_freeze_markdown_header_is_official(self) -> None:
        frozen = self.freeze()
        path = self.root / "official.md"
        registry.write_official_markdown(path, frozen)
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Registry chính thức"))
        self.assertNotIn("snapshot chưa đầy đủ", text)
        self.assertIn(frozen["draft_sha256"], text)

    def test_freeze_rejects_draft_diverging_from_source(self) -> None:
        for key, value in (("dev_eer", .2), ("decision_threshold", .3), ("checkpoint_sha256", "0" * 64),
                           ("generation", "g-" + "0" * 32), ("source_report_sha256", "1" * 64), ("identity", {"x": 1})):
            draft = deepcopy(self.draft)
            draft["entries"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "lệch nguồn"):
                self.freeze(draft)

    def test_freeze_rejects_missing_entry_or_incomplete(self) -> None:
        draft = deepcopy(self.draft)
        draft["entries"].pop()
        with self.assertRaises(ValueError):
            self.freeze(draft)
        draft = deepcopy(self.draft)
        draft["complete"] = False
        with self.assertRaises(ValueError):
            self.freeze(draft)
        draft = deepcopy(self.draft)
        draft["entries"][0]["status"] = "missing"
        with self.assertRaises(ValueError):
            self.freeze(draft)

    def test_freeze_rejects_protocol_mismatch_and_non_draft(self) -> None:
        draft = deepcopy(self.draft)
        draft["protocol_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.freeze(draft)
        draft = deepcopy(self.draft)
        draft["role"] = "official"
        with self.assertRaisesRegex(ValueError, "role=draft"):
            self.freeze(draft)

    def test_freeze_rejects_source_changed_after_draft(self) -> None:
        spec = self.config["required_models"][3]
        report_path = self.root / spec["source_report"]
        report = registry.load_json(report_path)
        report["controller"]["best"]["development"]["eer"] = .2
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lệch nguồn"):
            self.freeze()

    def test_freeze_x1r(self) -> None:
        config = registry.load_json(registry.ROOT / "configs/test_evaluation_x1r.json")
        for i, spec in enumerate(config["required_models"], 1):
            make_source(self.root, spec, 100 + i, config["input_condition"])
        draft = registry.build_registry(config, self.root)
        path = self.root / "draft_x1r.json"
        registry.atomic_json(path, draft)
        frozen = registry.freeze_registry(config, draft, path, self.root)
        self.assertEqual(len(frozen["entries"]), 3)
        self.assertEqual(frozen["input_condition"], config["input_condition"])

    def test_real_official_registries_and_configs(self) -> None:
        for config_name in ("test_evaluation.json", "test_evaluation_x1r.json"):
            config = registry.load_json(registry.ROOT / "configs" / config_name)
            path = registry.ROOT / config["registry_path"]
            self.assertTrue(path.is_file())
            self.assertFalse(config["test_gate_open"])
            self.assertEqual(config["registry_sha256"], registry.sha256_file(path))
            official = registry.load_json(path)
            self.assertEqual(official["role"], "official")
            registry.validate_registry(official, config, complete=True)
            draft_path = registry.ROOT / official["draft_path"]
            self.assertEqual(official["draft_sha256"], registry.sha256_file(draft_path))
            self.assertEqual(registry.load_json(draft_path)["entries"], official["entries"])
        d032 = registry.load_json(registry.CONFIG)
        self.assertTrue(all(d032["pre_test_checks"].values()))


if __name__ == "__main__":
    unittest.main()
