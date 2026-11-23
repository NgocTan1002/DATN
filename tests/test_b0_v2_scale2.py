"""Kiểm thử đặc tả và runner cho B0 v2 scale2 (D036)."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from scripts.train_b0_full import (
    RunData,
    _open_d027,
    main,
    validate_plan,
    validate_plan_scale2,
    validate_plan_v2,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_scale2_plan() -> dict[str, Any]:
    return json.loads((ROOT / "configs" / "b0_v2_training_scale2.json").read_text(encoding="utf-8"))


def _load_v2_plan() -> dict[str, Any]:
    return json.loads((ROOT / "configs" / "b0_v2_training.json").read_text(encoding="utf-8"))


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out


class B0V2Scale2PlanTests(unittest.TestCase):
    def test_scale2_config_passes_validate_plan_scale2(self) -> None:
        plan = _load_scale2_plan()
        validate_plan_scale2(plan)

    def test_cross_rejection_between_v2_and_scale2_validators(self) -> None:
        v2_plan = _load_v2_plan()
        scale2_plan = _load_scale2_plan()

        validate_plan_v2(v2_plan)
        with self.assertRaises(ValueError):
            validate_plan_v2(scale2_plan)
        with self.assertRaises(ValueError):
            validate_plan_scale2(v2_plan)
        with self.assertRaises(ValueError):
            validate_plan(scale2_plan)

    def test_rejects_invalid_scale2_plan_fields(self) -> None:
        base = _load_scale2_plan()
        mutations: list[tuple[str, Any]] = [
            ("data.train_samples", 15885),
            ("training.steps_per_epoch", 1986),
            ("training.max_optimizer_steps", 19860),
            ("training.max_train_sample_exposures", 158850),
            ("training.max_epochs", 3),
            ("evaluation.patience", 1),
            ("decision", "D027"),
            ("base_decision", "D023"),
            ("data.use_test", True),
            ("training.training_seeds", [2026, 2027]),
            ("training.seed", 2026),
        ]
        for dotted_key, bad_value in mutations:
            with self.subTest(key=dotted_key, value=bad_value):
                mutated = copy.deepcopy(base)
                parts = dotted_key.split(".")
                target = mutated
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = bad_value
                with self.assertRaises(ValueError):
                    validate_plan_scale2(mutated)

        # Cặp (max_epochs=3, patience=1) phải bị từ chối ở D036
        mutated_ep = copy.deepcopy(base)
        mutated_ep["training"]["max_epochs"] = 3
        mutated_ep["training"]["max_optimizer_steps"] = 3972 * 3
        mutated_ep["training"]["max_train_sample_exposures"] = 31770 * 3
        mutated_ep["evaluation"]["patience"] = 1
        with self.assertRaises(ValueError):
            validate_plan_scale2(mutated_ep)

    def test_train_manifest_sha256_matches_disk(self) -> None:
        plan = _load_scale2_plan()
        manifest_path = ROOT / plan["data"]["train_manifest"]
        actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(plan["data"]["train_manifest_sha256"], actual_sha)

    def test_diff_keys_from_v2_config_match_expected_set(self) -> None:
        v2_flat = _flatten_dict(_load_v2_plan())
        scale2_flat = _flatten_dict(_load_scale2_plan())
        diff_keys = sorted(
            k for k in (v2_flat.keys() | scale2_flat.keys())
            if v2_flat.get(k) != scale2_flat.get(k)
        )
        expected = sorted([
            "experiment_id",
            "decision",
            "base_decision",
            "data.train_manifest",
            "data.train_manifest_sha256",
            "data.train_artifact",
            "data.train_artifact_waveforms_sha256",
            "data.sample_selection_sha256",
            "data.train_samples",
            "training.steps_per_epoch",
            "training.max_optimizer_steps",
            "training.max_train_sample_exposures",
            "outputs.report_json_template",
            "outputs.report_md_template",
            "outputs.root_template",
        ])
        self.assertEqual(diff_keys, expected)


class _DummyDataset:
    def __init__(self, split: str, count: int, manifest_sha: str, waveforms_sha: str) -> None:
        self.split = split
        self._count = count
        self.source_manifest_sha256 = manifest_sha
        self.waveforms_sha256 = waveforms_sha
        self.target_sample_rate = 16_000
        self.records = [
            type("_Rec", (), {"source_index": i, "file": f"{split}/spk0/{i:05d}.wav"})()
            for i in range(count)
        ]

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        pass


class B0V2Scale2RunnerMockTests(unittest.TestCase):
    def test_open_d027_checks_31770_samples_for_scale2_and_main_uses_packed(self) -> None:
        plan = _load_scale2_plan()
        d = plan["data"]
        train_ds = _DummyDataset(
            "closed_train", 31770, d["train_manifest_sha256"], d["train_artifact_waveforms_sha256"]
        )
        dev_ds = _DummyDataset(
            "closed_dev", 38777, d["development_manifest_sha256"], d["development_artifact_waveforms_sha256"]
        )
        audio = json.loads((ROOT / "configs/audio.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for sub in ("train", "dev"):
                (tmp_path / sub).mkdir()
                (tmp_path / sub / "metadata.json").write_text("{}", encoding="utf-8")
            args = type(
                "_Args",
                (),
                {
                    "training_seed": 2026,
                    "manifest": None,
                    "parquet_dir": None,
                    "train_artifact": tmp_path / "train",
                    "dev_artifact": tmp_path / "dev",
                    "resume": False,
                    "technical_smoke": True,
                },
            )()

            with (
                patch("scripts.train_b0_full.verify_packed_artifact") as mock_verify,
                patch("scripts.train_b0_full._packed", side_effect=[train_ds, dev_ds]),
                patch("scripts.train_b0_full._select", return_value=([0, 1, 2, 3], [0, 1])) as mock_select,
            ):
                run_data = _open_d027(args, plan, audio, torch.device("cpu"))
                self.assertEqual(run_data.training_seed, 2026)
                self.assertEqual(mock_verify.call_count, 2)
                mock_select.assert_called_once()
                run_data.close()

            # Kiểm tra _open_d027 từ chối nếu train artifact chỉ có 15885 mẫu
            short_train_ds = _DummyDataset(
                "closed_train", 15885, d["train_manifest_sha256"], d["train_artifact_waveforms_sha256"]
            )
            with (
                patch("scripts.train_b0_full.verify_packed_artifact"),
                patch("scripts.train_b0_full._packed", side_effect=[short_train_ds, dev_ds]),
            ):
                with self.assertRaisesRegex(ValueError, "D027"):
                    _open_d027(args, plan, audio, torch.device("cpu"))

    def test_main_dispatches_scale2_to_open_d027_and_writes_scale2_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dummy_data = RunData(
                train=type("_T", (), {"close": lambda self: None, "set_epoch": lambda self, e: None})(),
                dev=type("_D", (), {"close": lambda self: None})(),
                train_indices=[0, 1, 2, 3],
                dev_indices=[0, 1],
                training_seed=2026,
                identity_extra={"training_seed": 2026, "data_seed": 2026, "backend": "packed", "device": "cpu"},
                report_extra={"training_seed": 2026, "data_seed": 2026, "backend": "packed"},
            )

            def fake_run_training(**kwargs: Any) -> dict[str, Any]:
                store = kwargs["store"]
                (store.root / "g-0001").mkdir(parents=True, exist_ok=True)
                (store.root / "LATEST.json").write_text(
                    json.dumps({"latest": {"generation": "g-0001"}}), encoding="utf-8"
                )
                (store.root / "g-0001" / "state.json").write_text(
                    json.dumps({"identity": kwargs["identity"], "controller": {"global_step": 2}}),
                    encoding="utf-8",
                )
                return {"status": "paused", "steps_this_invocation": 2, "controller": {"global_step": 2}}

            with (
                patch("scripts.train_b0_full._open_d027", return_value=dummy_data) as mock_open_d027,
                patch("scripts.train_b0_full.run_training", side_effect=fake_run_training),
                patch("scripts.train_b0_full.GenerationStore.load", return_value={"identity": {}}),
            ):
                rc = main([
                    "--plan", str(ROOT / "configs" / "b0_v2_training_scale2.json"),
                    "--training-seed", "2026",
                    "--run-root", str(tmp_path / "run"),
                    "--report-json", str(tmp_path / "report.json"),
                    "--report-md", str(tmp_path / "report.md"),
                    "--device", "cpu",
                    "--technical-smoke",
                ])
            self.assertEqual(rc, 0)
            mock_open_d027.assert_called_once()
            md_text = (tmp_path / "report.md").read_text(encoding="utf-8")
            self.assertIn("B0 v2 scale2 D036", md_text)
