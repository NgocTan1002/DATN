"""X1S-01 (D036): kiểm thử cấu hình scale2, runner, bootstrap, identity ×1/×2 và profile Kaggle."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from scripts import kaggle_workflow as workflow
from scripts import train_xlsr as runner
from scripts.train_b0 import _set_seed
from tests.test_b0_v2_runner import _write_artifact

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
BASELINE_PATH = ROOT / ".cache" / "x1s01" / "baseline.json"
IDENTITY_EXCLUDED_KEYS = ("source_files", "source_tree_sha256")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def load_bootstrap() -> Any:
    spec = importlib.util.spec_from_file_location(
        "bootstrap_scale2", ROOT / ".kaggle" / "xlsr-d028" / "bootstrap.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaptureIdentity(Exception):
    pass


class Scale2PlanAndRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.plan_x1 = json.loads(
            (ROOT / "configs" / "xlsr_training.json").read_text(encoding="utf-8")
        )
        self.plan_scale2 = json.loads(
            (ROOT / "configs" / "xlsr_training_scale2.json").read_text(encoding="utf-8")
        )

    def test_valid_scale2_plan_accepted_for_three_seeds(self) -> None:
        for seed in (2026, 2027, 2028):
            with self.subTest(seed=seed):
                run = runner.validate_plan(self.plan_scale2, "X1-L8-lr3e-5", seed)
                self.assertEqual(run, {"run_id": "X1-L8-lr3e-5", "layers": 8, "encoder_lr": 3e-5})

    def test_scale2_plan_rejects_single_field_mutations_and_disallowed_runs(self) -> None:
        mutations: list[tuple[str, Any]] = [
            ("train_samples_31769", lambda p: p["data"].__setitem__("train_samples", 31769)),
            ("steps_993", lambda p: p["training"].__setitem__("steps_per_epoch", 993)),
            ("decision_D028", lambda p: p.__setitem__("decision", "D028")),
            (
                "grid_add_X0",
                lambda p: p.__setitem__(
                    "grid", [{"run_id": "X0", "layers": 0, "encoder_lr": None}] + p["grid"]
                ),
            ),
            ("base_decision_wrong", lambda p: p.__setitem__("base_decision", "D035")),
            ("max_optimizer_steps_wrong", lambda p: p["training"].__setitem__("max_optimizer_steps", 9930)),
            ("index_sha_wrong", lambda p: p["data"].__setitem__("train_artifact_index_sha256", "0" * 64)),
        ]
        for label, mutate in mutations:
            with self.subTest(case=label):
                bad = copy.deepcopy(self.plan_scale2)
                mutate(bad)
                with self.assertRaises(ValueError):
                    runner.validate_plan(bad, "X1-L8-lr3e-5", 2026)

        for bad_run_id in ("X0", "X1-L8-lr3e-5-lp7k", "X1-L2-lr1e-5"):
            with self.subTest(run_id=bad_run_id):
                with self.assertRaisesRegex(ValueError, "scale2"):
                    runner.validate_plan(self.plan_scale2, bad_run_id, 2026)

        with self.assertRaisesRegex(ValueError, "Seed"):
            runner.validate_plan(self.plan_scale2, "X1-L8-lr3e-5", 2029)

    def test_x1_plan_still_valid_and_identity_matches_baseline_except_source(self) -> None:
        for run_id in ("X0", "X1-L8-lr3e-5"):
            cfg = runner.validate_plan(self.plan_x1, run_id, 2026)
            self.assertEqual(cfg["run_id"], run_id)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_dir, dev_dir = root / "train", root / "dev"
            _write_artifact(train_dir, split="closed_train", rows=4, manifest_sha256="a" * 64, offset=3)
            _write_artifact(dev_dir, split="closed_dev", rows=4, manifest_sha256="b" * 64, offset=6)
            fixture_plan = copy.deepcopy(self.plan_x1)
            fixture_plan["implementation"]["ready_for_training"] = True
            for prefix, dpath in (("train", train_dir), ("development", dev_dir)):
                meta = json.loads((dpath / "metadata.json").read_text(encoding="utf-8"))
                fixture_plan["data"][prefix + "_samples"] = 4
                fixture_plan["data"][prefix + "_metadata_sha256"] = runner.sha256(dpath / "metadata.json")
                fixture_plan["data"][prefix + "_manifest_sha256"] = meta["source_manifest_sha256"]
                fixture_plan["data"][prefix + "_artifact_waveforms_sha256"] = meta["waveforms_sha256"]
            plan_file = root / "plan_x1.json"
            plan_file.write_text(json.dumps(fixture_plan), encoding="utf-8")

            def fake_backbone(*args: Any, **kwargs: Any) -> Wav2Vec2Model:
                _set_seed(2026)
                cfg = json.loads((ROOT / "configs/xlsr_smoke.json").read_text(encoding="utf-8"))["backbone"]
                cfg.update(num_hidden_layers=9, layerdrop=0.1, apply_spec_augment=True)
                return Wav2Vec2Model(Wav2Vec2Config(**cfg))

            baseline_file = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.is_file() else None
            for run_id in ("X0", "X1-L8-lr3e-5"):
                with self.subTest(run_id=run_id):
                    run_cfg = runner.validate_plan(self.plan_x1, run_id, 2026)
                    captured: list[dict[str, Any]] = []

                    def observe(**kwargs: Any) -> dict[str, Any]:
                        captured.append(copy.deepcopy(kwargs["identity"]))
                        raise CaptureIdentity

                    with (
                        self.assertRaises(CaptureIdentity),
                        patch.object(runner, "validate_plan", return_value=run_cfg),
                        patch.object(runner, "verify_pretrained", return_value={"fixture": "random"}),
                        patch.object(runner, "load_xlsr_backbone", side_effect=fake_backbone),
                        patch.object(runner, "verify_packed_artifact"),
                        patch.object(
                            runner,
                            "_sample_selection_sha256",
                            return_value=self.plan_x1["data"]["sample_selection_sha256"],
                        ),
                        patch.object(runner, "run_training", side_effect=observe),
                    ):
                        runner.main(
                            [
                                "--plan", str(plan_file),
                                "--run-id", run_id,
                                "--training-seed", "2026",
                                "--device", "cpu",
                                "--pretrained-dir", str(root / "no-weights"),
                                "--train-artifact", str(train_dir),
                                "--dev-artifact", str(dev_dir),
                                "--run-root", str(root / run_id),
                                "--report-json", str(root / f"{run_id}.json"),
                            ]
                        )
                    identity = captured[0]
                    fixture_ref = json.loads(
                        (FIXTURES / f"t29_identity_{run_id}_seed2026.json").read_text(encoding="utf-8")
                    )
                    references = [fixture_ref]
                    if baseline_file is not None:
                        references.append(baseline_file["x1_identities_seed2026"][run_id])
                    for ref in references:
                        diffs = sorted(
                            k for k in set(identity) | set(ref) if identity.get(k) != ref.get(k)
                        )
                        self.assertEqual(diffs, sorted(IDENTITY_EXCLUDED_KEYS))
                        stripped = lambda v: {k: x for k, x in v.items() if k not in IDENTITY_EXCLUDED_KEYS}
                        self.assertEqual(canonical(stripped(identity)), canonical(stripped(ref)))

            # Verify scale2 identity carries D036 decision/experiment_id & scale2 checksums
            s2_plan = copy.deepcopy(self.plan_scale2)
            s2_plan["implementation"]["ready_for_training"] = True
            for prefix, dpath in (("train", train_dir), ("development", dev_dir)):
                meta = json.loads((dpath / "metadata.json").read_text(encoding="utf-8"))
                s2_plan["data"][prefix + "_samples"] = 4
                s2_plan["data"][prefix + "_metadata_sha256"] = runner.sha256(dpath / "metadata.json")
                s2_plan["data"][prefix + "_manifest_sha256"] = meta["source_manifest_sha256"]
                s2_plan["data"][prefix + "_artifact_waveforms_sha256"] = meta["waveforms_sha256"]
            s2_plan["data"]["train_artifact_index_sha256"] = runner.sha256(train_dir / "index.csv")
            s2_file = root / "plan_scale2.json"
            s2_file.write_text(json.dumps(s2_plan), encoding="utf-8")
            s2_cfg = runner.validate_plan(self.plan_scale2, "X1-L8-lr3e-5", 2026)
            s2_captured: list[dict[str, Any]] = []

            def observe_s2(**kwargs: Any) -> dict[str, Any]:
                s2_captured.append(copy.deepcopy(kwargs["identity"]))
                raise CaptureIdentity

            with (
                self.assertRaises(CaptureIdentity),
                patch.object(runner, "validate_plan", return_value=s2_cfg),
                patch.object(runner, "verify_pretrained", return_value={"fixture": "random"}),
                patch.object(runner, "load_xlsr_backbone", side_effect=fake_backbone),
                patch.object(runner, "verify_packed_artifact"),
                patch.object(
                    runner,
                    "_sample_selection_sha256",
                    return_value=self.plan_scale2["data"]["sample_selection_sha256"],
                ),
                patch.object(runner, "run_training", side_effect=observe_s2),
            ):
                runner.main(
                    [
                        "--plan", str(s2_file),
                        "--run-id", "X1-L8-lr3e-5",
                        "--training-seed", "2026",
                        "--device", "cpu",
                        "--pretrained-dir", str(root / "no-weights"),
                        "--train-artifact", str(train_dir),
                        "--dev-artifact", str(dev_dir),
                        "--run-root", str(root / "s2"),
                        "--report-json", str(root / "s2.json"),
                    ]
                )
            s2_id = s2_captured[0]
            self.assertEqual(s2_id["plan"]["decision"], "D036")
            self.assertEqual(s2_id["plan"]["experiment_id"], "xlsr-d036-scale2")
            self.assertEqual(
                s2_id["artifacts"]["train"]["index_sha256"],
                runner.sha256(train_dir / "index.csv"),
            )


class Scale2BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_bootstrap()
        self.plan_x1 = json.loads(
            (ROOT / "configs" / "xlsr_training.json").read_text(encoding="utf-8")
        )
        self.plan_scale2 = json.loads(
            (ROOT / "configs" / "xlsr_training_scale2.json").read_text(encoding="utf-8")
        )

    def test_bootstrap_commands_for_scale2_single_pair_and_x1_preserved(self) -> None:
        # x1 command unchanged
        x1_args = argparse.Namespace(
            run_id="X0",
            training_seed="2026",
            data_root=Path("/kaggle/input/doanttnt-t14-packed"),
            dry_run=False,
            resume_manifest_sha256=None,
        )
        x1_cmd = self.module.build_runner_command(
            x1_args,
            plan=self.plan_x1,
            python="/kaggle/working/.xlsr-offline-env/bin/python",
            weights=Path("/kaggle/input/doanttnt-t14-weights"),
            run_root=Path("/kaggle/working/checkpoints/xlsr_X0_seed2026"),
            report=Path("/kaggle/working/reports/xlsr_X0_seed2026.json"),
            started_unix=1000.0,
        )
        self.assertEqual(x1_cmd[x1_cmd.index("--plan") + 1], "configs/xlsr_training.json")
        self.assertEqual(
            x1_cmd[x1_cmd.index("--train-artifact") + 1],
            str(Path("/kaggle/input/doanttnt-t14-packed/train_development_20k_v2")),
        )

        # scale2 single-run (seed 2028)
        scale2_input = Path("/kaggle/input/doanttnt-scale2-packed-v1")
        s2_single = argparse.Namespace(
            plan="configs/xlsr_training_scale2.json",
            train_artifact=scale2_input,
            run_id="X1-L8-lr3e-5",
            training_seed="2028",
            data_root=Path("/kaggle/input/doanttnt-t14-packed"),
            dry_run=False,
            resume_manifest_sha256=None,
        )
        s2_cmd = self.module.build_runner_command(
            s2_single,
            plan=self.plan_scale2,
            python="/kaggle/working/.xlsr-offline-env/bin/python",
            weights=Path("/kaggle/input/doanttnt-t14-weights"),
            run_root=Path("/kaggle/working/checkpoints/xlsr_scale2_X1-L8-lr3e-5_seed2028"),
            report=Path("/kaggle/working/reports/xlsr_scale2_X1-L8-lr3e-5_seed2028.json"),
            started_unix=1000.0,
        )
        self.assertEqual(s2_cmd[s2_cmd.index("--plan") + 1], "configs/xlsr_training_scale2.json")
        self.assertEqual(s2_cmd[s2_cmd.index("--train-artifact") + 1], str(scale2_input))
        self.assertEqual(s2_cmd[s2_cmd.index("--session-steps") + 1], "19860")

        # scale2 pair-run (seed 2026 + 2027)
        s2_pair = argparse.Namespace(
            plan="configs/xlsr_training_scale2.json",
            train_artifact=scale2_input,
            data_root=Path("/kaggle/input/doanttnt-t14-packed"),
            working_root=Path("/kaggle/working"),
            run_id="X1-L8-lr3e-5",
            training_seed="2026",
            dry_run=False,
            resume_manifest_sha256=None,
            expected_step=None,
            second_run_id="X1-L8-lr3e-5",
            second_training_seed="2027",
            second_resume_manifest_sha256=None,
            second_expected_step=None,
        )
        specs = self.module.build_pair_runner_specs(
            s2_pair,
            plan=self.plan_scale2,
            python="/kaggle/working/.xlsr-offline-env/bin/python",
            weights=Path("/kaggle/input/doanttnt-t14-weights"),
            base_env={},
            started_unix=1000.0,
        )
        self.assertEqual(len(specs), 2)
        for spec, expected_seed, expected_gpu in zip(specs, ("2026", "2027"), ("0", "1"), strict=True):
            cmd = spec["command"]
            self.assertEqual(spec["env"]["CUDA_VISIBLE_DEVICES"], expected_gpu)
            self.assertEqual(cmd[cmd.index("--plan") + 1], "configs/xlsr_training_scale2.json")
            self.assertEqual(cmd[cmd.index("--train-artifact") + 1], str(scale2_input))
            self.assertEqual(cmd[cmd.index("--training-seed") + 1], expected_seed)
            self.assertIn(f"xlsr_scale2_X1-L8-lr3e-5_seed{expected_seed}", str(spec["run_root"]))

    def test_bootstrap_stops_when_scale2_file_sha_mismatches_and_logs_timing_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scale2_dir = root / "scale2_input"
            scale2_dir.mkdir()
            fake_contents = {
                "index.csv": b"fake-index-content\n",
                "metadata.json": b'{"artifact_count": 31770}\n',
                "waveforms.f32": b"\x00\x01\x02\x03" * 16,
            }
            fake_plan = copy.deepcopy(self.plan_scale2)
            for fname, payload in fake_contents.items():
                fpath = scale2_dir / fname
                fpath.write_bytes(payload)
            fake_plan["data"]["train_artifact_index_sha256"] = hashlib.sha256(
                fake_contents["index.csv"]
            ).hexdigest()
            fake_plan["data"]["train_metadata_sha256"] = hashlib.sha256(
                fake_contents["metadata.json"]
            ).hexdigest()
            fake_plan["data"]["train_artifact_waveforms_sha256"] = hashlib.sha256(
                fake_contents["waveforms.f32"]
            ).hexdigest()

            plan_path = root / "scale2_plan.json"
            plan_path.write_text(json.dumps(fake_plan), encoding="utf-8")

            # Corrupt each file one by one -> ValueError and runner never called
            for target_name in ("index.csv", "metadata.json", "waveforms.f32"):
                with self.subTest(corrupt_file=target_name):
                    (scale2_dir / target_name).write_bytes(b"corrupted-bytes")
                    with self.assertRaisesRegex(ValueError, f"Checksum dữ liệu scale2 không khớp.*{target_name}"):
                        self.module.verify_scale2_input_checksums(scale2_dir, fake_plan)

                    # Also verify main() stops before venv/pip/runner
                    manifest = root / "offline_manifest.json"
                    manifest.write_text('{"files":{}}', encoding="utf-8")
                    args = argparse.Namespace(
                        data_root=root / "data",
                        working_root=root / "working",
                        plan=str(plan_path),
                        train_artifact=scale2_dir,
                        run_id="X1-L8-lr3e-5",
                        training_seed="2028",
                        dry_run=True,
                        resume_manifest_sha256=None,
                        expected_step=None,
                        second_run_id=None,
                        second_training_seed=None,
                        second_resume_manifest_sha256=None,
                        second_expected_step=None,
                    )
                    with (
                        patch.object(self.module.argparse.ArgumentParser, "parse_args", return_value=args),
                        patch.object(
                            self.module.Path,
                            "rglob",
                            side_effect=lambda pattern: [manifest]
                            if pattern == "offline_manifest.json"
                            else [root / "pytorch_model.bin"],
                        ),
                        patch.object(self.module, "checked") as checked_mock,
                    ):
                        with self.assertRaisesRegex(ValueError, "Checksum dữ liệu scale2"):
                            self.module.main()
                        checked_mock.assert_not_called()

                    # Restore valid bytes
                    (scale2_dir / target_name).write_bytes(fake_contents[target_name])

            # Valid checksums -> main() emits scale2_checksum_end event with hash_seconds
            manifest = root / "offline_manifest.json"
            manifest.write_text('{"files":{}}', encoding="utf-8")
            args = argparse.Namespace(
                data_root=root / "data",
                working_root=root / "working",
                plan=str(plan_path),
                train_artifact=scale2_dir,
                run_id="X1-L8-lr3e-5",
                training_seed="2028",
                dry_run=True,
                resume_manifest_sha256=None,
                expected_step=None,
                second_run_id=None,
                second_training_seed=None,
                second_resume_manifest_sha256=None,
                second_expected_step=None,
            )
            stdout = io.StringIO()
            with (
                redirect_stdout(stdout),
                patch.object(self.module.argparse.ArgumentParser, "parse_args", return_value=args),
                patch.object(
                    self.module.Path,
                    "rglob",
                    side_effect=lambda pattern: [manifest]
                    if pattern == "offline_manifest.json"
                    else [root / "pytorch_model.bin"],
                ),
                patch.object(self.module.venv, "EnvBuilder"),
                patch.object(self.module.subprocess, "check_output", return_value=b"unchanged"),
                patch.object(self.module, "checked"),
            ):
                exit_code = self.module.main()
            self.assertEqual(exit_code, 0)
            events = [json.loads(line) for line in stdout.getvalue().splitlines()]
            s2_events = [e for e in events if e["event"] == "scale2_checksum_end"]
            self.assertEqual(len(s2_events), 1)
            self.assertIn("hash_seconds", s2_events[0])
            self.assertEqual(set(s2_events[0]["files"]), {"index.csv", "metadata.json", "waveforms.f32"})


class Scale2WorkflowAndBaselineProfilesTests(unittest.TestCase):
    def test_scale2_profiles_metadata_and_x1_baseline_byte_parity(self) -> None:
        project_path = ROOT / ".kaggle" / "xlsr-d028" / "project.json"
        project_raw = json.loads(project_path.read_text(encoding="utf-8"))
        config = workflow.load_config(project_path)

        expected_sources = [
            "ngctnnguyn1111/doanttnt-t14-packed",
            "ngctnnguyn1111/doanttnt-xlsr-d028-source",
            "ngctnnguyn1111/doanttnt-t14-weights",
            "ngctnnguyn1111/doanttnt-t14-offline-deps",
            "ngctnnguyn1111/doanttnt-scale2-packed-v1",
        ]

        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            temp_root = Path(td)
            source_dir = temp_root / "source_dataset"
            source_dir.mkdir()
            (source_dir / "source_manifest.json").write_text(
                json.dumps({"bundle_sha256": "a" * 64}), encoding="utf-8"
            )
            patched_cfg = workflow.WorkflowConfig(
                data_dataset_ref=config.data_dataset_ref,
                source_dataset_ref=config.source_dataset_ref,
                source_dataset_title=config.source_dataset_title,
                kernel_ref=config.kernel_ref,
                kernel_title=config.kernel_title,
                workspace_dir=temp_root,
                enable_gpu=config.enable_gpu,
                enable_internet=config.enable_internet,
                default_profile=config.default_profile,
                profiles=config.profiles,
            )

            for name in workflow.XLSR_SCALE2_PROFILES:
                with self.subTest(scale2_profile=name):
                    self.assertIn(name, config.profiles)
                    kdir = workflow.prepare_kernel(patched_cfg, name)
                    meta = json.loads((kdir / "kernel-metadata.json").read_text(encoding="utf-8"))
                    self.assertEqual(meta["dataset_sources"], expected_sources)
                    self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
                    self.assertTrue(meta["is_private"])
                    self.assertTrue(meta["enable_gpu"])
                    self.assertFalse(meta["enable_internet"])
                    cmd = config.profiles[name][0]
                    encoded = re.search(r"b64decode\('([^']+)'\)", cmd[2]).group(1)
                    boot_text = base64.b64decode(encoded).decode("utf-8")
                    compile(boot_text, "bootstrap", "exec")
                    self.assertIn("verify_scale2_input_checksums", boot_text)
                    self.assertEqual(cmd[cmd.index("--plan") + 1], "configs/xlsr_training_scale2.json")
                    self.assertEqual(
                        cmd[cmd.index("--train-artifact") + 1],
                        "/kaggle/input/doanttnt-scale2-packed-v1",
                    )

            # Verify all x1 baseline profiles match byte-for-byte
            self.assertTrue(BASELINE_PATH.is_file(), f"Thiếu baseline: {BASELINE_PATH}")
            baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            pair_profiles = {
                "X0-seed2027-X0-seed2028",
                "X0-seed2027-X0-seed2028-dry-run",
                "X1-L8-lr3e-5-lp7k-seed2026-X1-L8-lr3e-5-lp7k-seed2027",
                "X1-L8-lr3e-5-lp7k-seed2026-X1-L8-lr3e-5-lp7k-seed2027-dry-run",
            }
            for x1_name, expected in baseline["profiles"].items():
                self.assertEqual(canonical(project_raw["profiles"][x1_name]), canonical(expected["raw_commands"]))
                rendered = [list(c) for c in workflow.render_commands(config, x1_name)]
                self.assertEqual(canonical(rendered), canonical(expected["rendered_commands"]))
                if x1_name in pair_profiles:
                    actual_meta = workflow.build_xlsr_pair_kernel_metadata(config, rendered)
                else:
                    kdir = workflow.prepare_kernel(patched_cfg, x1_name)
                    actual_meta = json.loads((kdir / "kernel-metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(canonical(actual_meta), canonical(expected["kernel_metadata"]))


if __name__ == "__main__":
    unittest.main()
