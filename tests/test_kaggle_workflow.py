from __future__ import annotations

import base64
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "kaggle_workflow.py"
SPEC = importlib.util.spec_from_file_location("kaggle_workflow", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Không thể nạp scripts/kaggle_workflow.py")
kaggle_workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kaggle_workflow
SPEC.loader.exec_module(kaggle_workflow)


def config_payload(workspace_dir: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "data_dataset_ref": "owner/vsasv-data",
        "source_dataset_ref": "owner/project-source",
        "source_dataset_title": "Project Source Dataset",
        "kernel_ref": "owner/project-runner",
        "kernel_title": "Project Kaggle Runner",
        "workspace_dir": workspace_dir,
        "enable_gpu": True,
        "enable_internet": False,
        "default_profile": "runtime",
        "profiles": {
            "runtime": [
                [
                    "python",
                    "script.py",
                    "--data",
                    "{data_root}",
                    "--out",
                    "{working_root}/report.json",
                ]
            ]
        },
    }


class KaggleWorkflowTests(unittest.TestCase):
    def write_config(self, root: Path, payload: dict[str, object]) -> Path:
        path = root / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_config_rejects_mixed_owners(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            payload = config_payload(root.relative_to(PROJECT_ROOT).as_posix())
            payload["kernel_ref"] = "other/project-runner"
            path = self.write_config(root, payload)
            with self.assertRaisesRegex(
                kaggle_workflow.WorkflowError, "cùng một owner"
            ):
                kaggle_workflow.load_config(path)

    def test_run_process_forces_utf8_for_kaggle_output(self) -> None:
        completed = CompletedProcess(args=["kaggle"], returncode=0)
        with mock.patch.object(
            kaggle_workflow.subprocess, "run", return_value=completed
        ) as run:
            result = kaggle_workflow.run_process(["kaggle", "--version"])
        self.assertIs(result, completed)
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONUTF8"], "1")
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")

    def test_render_commands_expands_only_supported_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            payload = config_payload(root.relative_to(PROJECT_ROOT).as_posix())
            config = kaggle_workflow.load_config(self.write_config(root, payload))
            commands = kaggle_workflow.render_commands(config, "runtime")
            self.assertEqual(
                commands[0],
                (
                    "python",
                    "script.py",
                    "--data",
                    "__KAGGLE_DATA_ROOT__",
                    "--out",
                    "/kaggle/working/report.json",
                ),
            )

    def test_bundle_source_is_deterministic_for_same_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            payload = config_payload(root.relative_to(PROJECT_ROOT).as_posix())
            config = kaggle_workflow.load_config(self.write_config(root, payload))
            selected = [PROJECT_ROOT / "requirements-kaggle.txt"]
            with mock.patch.object(kaggle_workflow, "source_files", return_value=selected):
                first = kaggle_workflow.bundle_source(config)
                second = kaggle_workflow.bundle_source(config)
            self.assertEqual(first["bundle_sha256"], second["bundle_sha256"])
            bundle = config.workspace_dir / "source_dataset" / "project_source.zip"
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(archive.namelist(), ["requirements-kaggle.txt"])

    def test_prepare_kernel_writes_private_gpu_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            payload = config_payload(root.relative_to(PROJECT_ROOT).as_posix())
            config = kaggle_workflow.load_config(self.write_config(root, payload))
            source_dir = config.workspace_dir / "source_dataset"
            source_dir.mkdir(parents=True)
            (source_dir / "source_manifest.json").write_text(
                json.dumps({"bundle_sha256": "a" * 64}), encoding="utf-8"
            )
            kernel_dir = kaggle_workflow.prepare_kernel(config, "runtime")
            metadata = json.loads(
                (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")
            )
            run_config = json.loads(
                (kernel_dir / "run-config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            self.assertEqual(
                metadata["dataset_sources"],
                ["owner/vsasv-data", "owner/project-source"],
            )
            self.assertEqual(run_config["source_bundle_sha256"], "a" * 64)
            self.assertEqual(run_config["data_dataset_slug"], "vsasv-data")
            runner_source = (kernel_dir / "runner.py").read_text(encoding="utf-8")
            self.assertNotIn(kaggle_workflow.RUN_CONFIG_ASSIGNMENT, runner_source)
            assignment_prefix = 'EMBEDDED_RUN_CONFIG_B64 = "'
            encoded = next(
                line.removeprefix(assignment_prefix).removesuffix('"')
                for line in runner_source.splitlines()
                if line.startswith(assignment_prefix)
            )
            embedded_config = json.loads(
                base64.b64decode(encoded).decode("utf-8")
            )
            self.assertEqual(embedded_config, run_config)

    def test_prepare_kernel_omits_unused_data_dataset(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            payload = config_payload(root.relative_to(PROJECT_ROOT).as_posix())
            payload["profiles"] = {
                "runtime": [
                    [
                        "python",
                        "script.py",
                        "--out",
                        "{working_root}/report.json",
                    ]
                ]
            }
            config = kaggle_workflow.load_config(self.write_config(root, payload))
            source_dir = config.workspace_dir / "source_dataset"
            source_dir.mkdir(parents=True)
            (source_dir / "source_manifest.json").write_text(
                json.dumps({"bundle_sha256": "a" * 64}), encoding="utf-8"
            )
            kernel_dir = kaggle_workflow.prepare_kernel(config, "runtime")
            metadata = json.loads(
                (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["dataset_sources"], ["owner/project-source"]
            )

    def test_missing_source_dataset_is_reported_as_false(self) -> None:
        completed = CompletedProcess(
            args=["kaggle"], returncode=0, stdout="No datasets found\n", stderr=""
        )
        with mock.patch.object(
            kaggle_workflow, "run_process", return_value=completed
        ):
            self.assertFalse(
                kaggle_workflow.kaggle_dataset_exists("owner/project-source")
            )

    def test_benchmark_profile_repeats_one_locked_workload(self) -> None:
        payload = json.loads(
            (PROJECT_ROOT / "configs" / "kaggle_cli.example.json").read_text(
                encoding="utf-8"
            )
        )
        commands = payload["profiles"]["b0-benchmark"]
        self.assertEqual(len(commands), 4)
        benchmark_commands = commands[1:]

        def normalized(command: list[str]) -> list[str]:
            result = list(command)
            for option in ("--checkpoint", "--report-json", "--report-md"):
                index = result.index(option)
                result[index + 1] = f"<{option}>"
            return result

        self.assertEqual(
            [normalized(command) for command in benchmark_commands],
            [normalized(benchmark_commands[0])] * 3,
        )
        self.assertIn("balanced_hash", benchmark_commands[0])
        self.assertIn("32", benchmark_commands[0])

    def test_pilot_profiles_only_change_policy_and_output_paths(self) -> None:
        payload = json.loads(
            (PROJECT_ROOT / "configs" / "kaggle_cli.example.json").read_text(
                encoding="utf-8"
            )
        )
        profile_names = (
            "b0-pilot-none",
            "b0-pilot-peak",
            "b0-pilot-rms-dbfs",
        )
        policies = ("none", "peak", "rms_dbfs")
        training_commands: list[list[str]] = []
        for profile_name, policy in zip(profile_names, policies, strict=True):
            commands = payload["profiles"][profile_name]
            self.assertEqual(len(commands), 2)
            command = commands[1]
            self.assertEqual(command[command.index("--run-mode") + 1], "pilot")
            self.assertEqual(command[command.index("--amplitude-policy") + 1], policy)
            self.assertEqual(command[command.index("--step-budget") + 1], "896")
            self.assertEqual(command[command.index("--batch-size") + 1], "8")
            self.assertEqual(command[command.index("--learning-rate") + 1], "0.001")
            self.assertNotIn("--train-limit", command)
            self.assertNotIn("--development-limit", command)
            training_commands.append(command)

        def normalized(command: list[str]) -> list[str]:
            result = list(command)
            for option in (
                "--amplitude-policy",
                "--checkpoint",
                "--report-json",
                "--report-md",
            ):
                index = result.index(option)
                result[index + 1] = f"<{option}>"
            return result

        self.assertEqual(
            [normalized(command) for command in training_commands],
            [normalized(training_commands[0])] * 3,
        )


if __name__ == "__main__":
    unittest.main()
