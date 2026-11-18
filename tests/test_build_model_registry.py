"""T23: registry snapshot, checksum và trạng thái kiểm độc lập."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_model_registry as registry


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = registry.ROOT / ".cache/t23/temp"
        temp.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temp)
        self.root = Path(self.directory.name)
        self.config = registry.load_json(registry.CONFIG)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def create(self, spec: dict, index: int = 1) -> tuple[Path, Path]:
        generation = f"g-{index:032x}"
        report_path = self.root / spec["source_report"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        path = (self.root / "checkpoints" / f"b0_v2_seed{spec['training_seed']}" / generation / "checkpoint.pt"
                if spec["family"] == "B0-v2" else report_path.parent.parent / "checkpoints" /
                f"xlsr_{spec['run_id']}_seed{spec['training_seed']}" / generation / "checkpoint.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint fixture")
        identity = {"run_id": spec["run_id"], "training_seed": spec["training_seed"], "data_seed": 2026,
                    "source_tree_sha256": "a" * 64}
        report = {"status": "early_stopped", "training_seed": spec["training_seed"], "development_samples": 38777,
                  "source_tree_sha256": "a" * 64, "controller": {"best": {"generation": generation, "epoch": 2,
                    "development": {"eer": .1, "decision_threshold": .25}}},
                  "checkpoints": {"best": {"generation": generation, "checkpoint_sha256": registry.sha256_file(path)}}}
        if spec["family"] != "B0-v2":
            report["identity"] = identity
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, path

    def test_real_config_gate_closed_and_exact_verified_set(self) -> None:
        registry.validate_config(self.config)
        self.assertFalse(self.config["test_gate_open"])
        self.assertEqual(self.config["decision"], "D032")
        self.assertEqual(set(self.config["claude_verified"]), {
            "B0-v2-seed2026", "B0-v2-seed2027", "B0-v2-seed2028", "X0-seed2026", "X1-L8-lr3e-5-seed2026", "X1-L8-lr3e-5-seed2027", "X1-L8-lr3e-5-seed2028", "X0-seed2027", "X0-seed2028"})

    def test_hash_recomputed_and_best_threshold_retained(self) -> None:
        report, checkpoint = self.create(self.config["required_models"][0])
        value = registry.build_registry(self.config, self.root)
        item = value["entries"][0]
        self.assertEqual(item["status"], "present")
        self.assertEqual(item["checkpoint_sha256"], registry.sha256_file(checkpoint))
        self.assertEqual(item["decision_threshold"], .25)
        self.assertEqual(item["source_report_sha256"], registry.sha256_file(report))
        self.assertFalse(value["complete"])

    def test_checkpoint_digest_mismatch_is_fatal(self) -> None:
        _, checkpoint = self.create(self.config["required_models"][0])
        checkpoint.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            registry.build_registry(self.config, self.root)

    def test_missing_report_and_missing_checkpoint(self) -> None:
        _, checkpoint = self.create(self.config["required_models"][0])
        checkpoint.unlink()
        value = registry.build_registry(self.config, self.root)
        self.assertEqual(value["entries"][0]["missing_reason"], "checkpoint")
        self.assertEqual(value["entries"][1]["missing_reason"], "source_report")
        self.assertTrue(all(x["status"] == "missing" for x in value["entries"]))

    def test_nine_present_not_complete_until_all_claude_verified(self) -> None:
        self.config["claude_verified"] = []
        for i, spec in enumerate(self.config["required_models"], 1):
            self.create(spec, i)
        value = registry.build_registry(self.config, self.root)
        self.assertTrue(all(x["status"] == "present" for x in value["entries"]))
        self.assertFalse(value["complete"])
        self.config["claude_verified"] = [registry.model_id(x) for x in self.config["required_models"]]
        self.assertTrue(registry.build_registry(self.config, self.root)["complete"])

    def test_paused_is_incomplete_even_with_nine_checkpoints_and_verified(self) -> None:
        paths = [self.create(spec, i)[0] for i, spec in enumerate(self.config["required_models"], 1)]
        self.config["claude_verified"] = [registry.model_id(x) for x in self.config["required_models"]]
        report = registry.load_json(paths[0])
        report["status"] = "paused"
        paths[0].write_text(json.dumps(report), encoding="utf-8")
        value = registry.build_registry(self.config, self.root)
        item = value["entries"][0]
        self.assertEqual(item["status"], "incomplete")
        self.assertEqual(item["missing_reason"], "source_not_terminal")
        self.assertEqual(item["source_status"], "paused")
        self.assertEqual(item["checkpoint_sha256"], report["checkpoints"]["best"]["checkpoint_sha256"])
        self.assertEqual(item["generation"], report["controller"]["best"]["generation"])
        self.assertEqual(item["decision_threshold"], .25)
        self.assertEqual(item["dev_eer"], .1)
        self.assertFalse(value["complete"])

    def test_only_early_stopped_and_max_epochs_are_terminal_present(self) -> None:
        self.assertEqual(registry.TERMINAL_SOURCE_STATUSES, {"early_stopped", "max_epochs"})
        path, _ = self.create(self.config["required_models"][0])
        for status in ("early_stopped", "max_epochs"):
            report = registry.load_json(path)
            report["status"] = status
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.subTest(status=status):
                item = registry.build_registry(self.config, self.root)["entries"][0]
                self.assertEqual(item["status"], "present")
                self.assertIsNone(item["missing_reason"])

    def test_unknown_source_status_is_incomplete(self) -> None:
        path, _ = self.create(self.config["required_models"][0])
        for status in ("unknown", "running", "complete", None):
            report = registry.load_json(path)
            report["status"] = status
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.subTest(status=status):
                value = registry.build_registry(self.config, self.root)
                self.assertEqual(value["entries"][0]["status"], "incomplete")
                self.assertEqual(value["entries"][0]["missing_reason"], "source_not_terminal")
                self.assertFalse(value["complete"])

    def test_paused_without_checkpoint_remains_incomplete(self) -> None:
        path, checkpoint = self.create(self.config["required_models"][0])
        report = registry.load_json(path)
        report["status"] = "paused"
        path.write_text(json.dumps(report), encoding="utf-8")
        checkpoint.unlink()
        item = registry.build_registry(self.config, self.root)["entries"][0]
        self.assertEqual(item["status"], "incomplete")
        self.assertEqual(item["missing_reason"], "source_not_terminal")
        self.assertIn("reported_checkpoint_sha256", item)
        self.assertNotIn("checkpoint_sha256", item)

    def test_paused_before_first_best_is_incomplete(self) -> None:
        path, _ = self.create(self.config["required_models"][0])
        report = registry.load_json(path)
        report["status"] = "paused"
        report["controller"]["best"] = None
        report["checkpoints"]["best"] = None
        path.write_text(json.dumps(report), encoding="utf-8")
        value = registry.build_registry(self.config, self.root)
        self.assertEqual(value["entries"][0]["status"], "incomplete")
        self.assertEqual(value["entries"][0]["source_status"], "paused")
        self.assertFalse(value["complete"])

    def test_best_pointer_mismatch_not_replaced_by_latest(self) -> None:
        path, _ = self.create(self.config["required_models"][0])
        report = registry.load_json(path)
        report["checkpoints"]["best"]["generation"] = "g-" + "f" * 32
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Generation"):
            registry.build_registry(self.config, self.root)

    def test_registry_config_mismatch_or_duplicate_is_rejected(self) -> None:
        value = registry.build_registry(self.config, self.root)
        altered = deepcopy(value)
        altered["entries"][1] = deepcopy(altered["entries"][0])
        with self.assertRaises(ValueError):
            registry.validate_registry(altered, self.config)
        altered = deepcopy(value)
        altered["entries"][0]["source_report"] = "different.json"
        with self.assertRaises(ValueError):
            registry.validate_registry(altered, self.config)

    def test_gate_and_registry_digest_not_in_protocol_digest_cycle(self) -> None:
        old = registry.protocol_sha256(self.config)
        self.config.update(test_gate_open=True, registry_sha256="f" * 64)
        self.assertEqual(old, registry.protocol_sha256(self.config))

    def test_cli_refuses_official_registry_without_writing(self) -> None:
        for official in ("reports/model_registry_v1.json", "reports/model_registry_x1r.json"):
            with self.subTest(official=official), patch.object(registry, "ROOT", self.root), patch(
                "sys.argv", ["registry", "--output", str(self.root / official)]
            ), patch.object(registry, "atomic_json") as write:
                self.assertEqual(registry.main(), 1)
            write.assert_not_called()


class X1RRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = registry.ROOT / ".cache/reg_x1r/temp_tests"
        temp.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temp)
        self.root = Path(self.directory.name)
        self.config_path = registry.ROOT / "configs/test_evaluation_x1r.json"
        self.config = registry.load_json(self.config_path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def create_x1r_fixture(self, spec: dict, index: int = 1, *, include_ic: bool = True) -> tuple[Path, Path]:
        generation = f"g-{index:032x}"
        report_path = self.root / spec["source_report"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path = (
            report_path.parent.parent
            / "checkpoints"
            / f"xlsr_{spec['run_id']}_seed{spec['training_seed']}"
            / generation
            / "checkpoint.pt"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_bytes(f"x1r-{spec['training_seed']}".encode("utf-8"))
        identity = {
            "run_id": spec["run_id"],
            "training_seed": spec["training_seed"],
            "data_seed": 2026,
            "source_tree_sha256": "b" * 64,
        }
        if include_ic:
            identity["input_condition"] = deepcopy(self.config["input_condition"])
        report = {
            "status": "early_stopped",
            "training_seed": spec["training_seed"],
            "development_samples": 38777,
            "source_tree_sha256": "b" * 64,
            "identity": identity,
            "controller": {
                "best": {
                    "generation": generation,
                    "epoch": 6,
                    "development": {"eer": 0.009, "decision_threshold": 0.01},
                }
            },
            "checkpoints": {
                "best": {
                    "generation": generation,
                    "checkpoint_sha256": registry.sha256_file(ckpt_path),
                }
            },
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, ckpt_path

    def test_real_x1r_config_valid_three_seeds_and_gate_closed(self) -> None:
        registry.validate_config(self.config)
        self.assertEqual(self.config["decision"], "D035")
        self.assertFalse(self.config["test_gate_open"])
        self.assertEqual(
            set(self.config["claude_verified"]),
            {
                "X1-L8-lr3e-5-lp7k-seed2026",
                "X1-L8-lr3e-5-lp7k-seed2027",
                "X1-L8-lr3e-5-lp7k-seed2028",
            },
        )
        self.assertEqual(len(self.config["required_models"]), 3)
        self.assertEqual([x["training_seed"] for x in self.config["required_models"]], [2026, 2027, 2028])
        self.assertEqual(
            self.config["input_condition"]["filter"]["coefficient_float32_sha256"],
            registry.LOWPASS_7K_COEFFICIENT_SHA256,
        )

    def test_x1r_build_three_entries_incomplete_when_claude_verified_empty(self) -> None:
        cfg = deepcopy(self.config)
        cfg["claude_verified"] = []
        for idx, spec in enumerate(cfg["required_models"], 1):
            self.create_x1r_fixture(spec, idx)
        value = registry.build_registry(cfg, self.root)
        self.assertEqual(value["decision"], "D035")
        self.assertFalse(value["complete"])
        self.assertEqual(len(value["entries"]), 3)
        for entry in value["entries"]:
            self.assertEqual(entry["status"], "present")
            self.assertEqual(entry["input_condition"], "lowpass_7k")
            self.assertFalse(entry["claude_verified"])
        seed2026 = next(x for x in value["entries"] if x["training_seed"] == 2026)
        self.assertIn("T30B 9 epoch + T32B resume", seed2026["verification_note"])
        registry.validate_registry(value, cfg, complete=False)
        verified_value = registry.build_registry(self.config, self.root)
        self.assertTrue(verified_value["complete"])
        registry.validate_registry(verified_value, self.config, complete=True)

    def test_x1r_rejects_missing_seed_and_test_path_and_wrong_filter_sha(self) -> None:
        missing_seed = deepcopy(self.config)
        missing_seed["required_models"] = missing_seed["required_models"][:2]
        with self.assertRaisesRegex(ValueError, "3 mục"):
            registry.validate_config(missing_seed)

        test_path_cfg = deepcopy(self.config)
        test_path_cfg["required_models"][0]["source_report"] = "data/closed_test_report.json"
        with self.assertRaisesRegex(ValueError, "dữ liệu test"):
            registry.validate_config(test_path_cfg)

        bad_sha_cfg = deepcopy(self.config)
        bad_sha_cfg["input_condition"]["filter"]["coefficient_float32_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "coefficient_float32_sha256"):
            registry.validate_config(bad_sha_cfg)

    def test_x1r_rejects_report_missing_input_condition(self) -> None:
        self.create_x1r_fixture(self.config["required_models"][0], 1, include_ic=False)
        with self.assertRaisesRegex(ValueError, "input_condition"):
            registry.build_registry(self.config, self.root)

    def test_d032_protocol_sha_unchanged_by_d035_support(self) -> None:
        d032_config = registry.load_json(registry.CONFIG)
        d032_draft = registry.load_json(registry.DRAFT)
        self.assertEqual(registry.protocol_sha256(d032_config), d032_draft["protocol_sha256"])
        registry.validate_registry(d032_draft, d032_config, complete=True)


if __name__ == "__main__":
    unittest.main()
