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

    def write_prepared_kernel(
        self,
        root: Path,
        *,
        embedded_profile: str,
        run_config_profile: str,
    ) -> Path:
        kernel_dir = root / "kernel"
        kernel_dir.mkdir()
        embedded_config = {
            "schema_version": 1,
            "profile": embedded_profile,
            "commands": [["python", "script.py"]],
        }
        run_config = {
            "schema_version": 1,
            "profile": run_config_profile,
            "commands": [["python", "script.py"]],
        }
        encoded = base64.b64encode(
            json.dumps(embedded_config).encode("utf-8")
        ).decode("ascii")
        runner = kaggle_workflow.RUNNER_TEMPLATE_PATH.read_text(encoding="utf-8")
        runner = runner.replace(
            kaggle_workflow.RUN_CONFIG_ASSIGNMENT,
            f'EMBEDDED_RUN_CONFIG_B64 = "{encoded}"',
        )
        (kernel_dir / "runner.py").write_text(runner, encoding="utf-8")
        (kernel_dir / "run-config.json").write_text(
            json.dumps(run_config), encoding="utf-8"
        )
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": "owner/project-runner",
                    "code_file": "runner.py",
                    "is_private": True,
                }
            ),
            encoding="utf-8",
        )
        return kernel_dir

    def test_verify_prepared_kernel_rejects_embedded_run_config_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            kernel_dir = self.write_prepared_kernel(
                Path(temporary),
                embedded_profile="b0-full-resume-session-10",
                run_config_profile="b0-full-resume-session-11",
            )
            with self.assertRaisesRegex(
                kaggle_workflow.WorkflowError,
                "Cấu hình nhúng trong runner không khớp run-config.json",
            ):
                kaggle_workflow.verify_prepared_kernel(
                    kernel_dir,
                    expected_kernel_ref="owner/project-runner",
                )

    def test_verify_prepared_kernel_accepts_matching_run_config(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            kernel_dir = self.write_prepared_kernel(
                Path(temporary),
                embedded_profile="b0-full-resume-session-11",
                run_config_profile="b0-full-resume-session-11",
            )
            summary = kaggle_workflow.verify_prepared_kernel(
                kernel_dir,
                expected_kernel_ref="owner/project-runner",
            )
            self.assertEqual(summary["profile"], "b0-full-resume-session-11")
            self.assertEqual(len(summary["runner_sha256"]), 64)

    def test_push_prepared_kernel_stops_before_kaggle_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            config = kaggle_workflow.load_config(
                self.write_config(
                    root,
                    config_payload(root.relative_to(PROJECT_ROOT).as_posix()),
                )
            )
            kernel_dir = self.write_prepared_kernel(
                root,
                embedded_profile="b0-full-resume-session-10",
                run_config_profile="b0-full-resume-session-11",
            )
            with mock.patch.object(kaggle_workflow, "run_process") as run:
                with self.assertRaisesRegex(
                    kaggle_workflow.WorkflowError,
                    "Cấu hình nhúng trong runner không khớp run-config.json",
                ):
                    kaggle_workflow.push_prepared_kernel(config, kernel_dir)
            run.assert_not_called()

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
            cfg_path = root / "project.json"
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            config = kaggle_workflow.load_config(cfg_path)
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
            cfg_path = root / "project.json"
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            config = kaggle_workflow.load_config(cfg_path)
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
            cfg_path = root / "project.json"
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            config = kaggle_workflow.load_config(cfg_path)
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
            cfg_path = root / "project.json"
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            config = kaggle_workflow.load_config(cfg_path)
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



class KaggleWarningFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp=tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root=Path(self.temp.name)
        path=self.root/"config.json"
        path.write_text(json.dumps(config_payload(self.root.relative_to(PROJECT_ROOT).as_posix())),encoding="utf-8")
        self.config=kaggle_workflow.load_config(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assert_quiet(self, run: mock.Mock) -> None:
        self.assertGreater(len(run.call_args_list),0)
        for call in run.call_args_list:
            command=call.args[0]
            self.assertEqual(command[:2],["kaggle","-W"])
            self.assertEqual(command.count("-W"),1)

    def test_missing_dataset_uses_no_warn_before_parsing_stdout(self) -> None:
        def cli(command: list[str], **kwargs: object) -> CompletedProcess[str]:
            stdout = "No datasets found\n" if command[1:2] == ["-W"] else "Warning: outdated version\nNo datasets found\n"
            return CompletedProcess(command,0,stdout,"")
        with mock.patch.object(kaggle_workflow.subprocess,"run",side_effect=cli) as run:
            self.assertFalse(kaggle_workflow.kaggle_dataset_exists("owner/project-source"))
        self.assert_quiet(run)

    def test_dataset_exists_parses_matching_json(self) -> None:
        result=CompletedProcess([],0,json.dumps([{"ref":"owner/project-source"}]),"")
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=result) as run:
            self.assertTrue(kaggle_workflow.kaggle_dataset_exists("owner/project-source"))
        self.assert_quiet(run)

    def test_dataset_exists_returns_false_for_other_refs(self) -> None:
        result=CompletedProcess([],0,json.dumps([{"ref":"owner/other"}]),"")
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=result) as run:
            self.assertFalse(kaggle_workflow.kaggle_dataset_exists("owner/project-source"))
        self.assert_quiet(run)

    def test_doctor_calls_version_files_and_dataset_list_with_no_warn(self) -> None:
        files=[{"name":f"{i}.parquet"} for i in range(68)]
        files += [{"name":name} for name in ["development_20k_v2.csv","smoke_train.csv","smoke_dev.csv"]]
        def cli(command: list[str], **kwargs: object) -> CompletedProcess[str]:
            if "--version" in command:stdout="Kaggle CLI 2.2.4"
            elif "files" in command:stdout=json.dumps(files)
            else:stdout="No datasets found\n"
            return CompletedProcess(command,0,stdout,"")
        with mock.patch.object(kaggle_workflow.subprocess,"run",side_effect=cli) as run, \
             mock.patch.object(kaggle_workflow.shutil,"which",return_value="kaggle"):
            kaggle_workflow.doctor(self.config,offline=False)
        self.assert_quiet(run)
        self.assertEqual(run.call_count,3)

    def test_publish_create_calls_list_and_create_with_no_warn(self) -> None:
        result=CompletedProcess([],0,"No datasets found\n","")
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=result) as run, \
             mock.patch.object(kaggle_workflow,"bundle_source",return_value={"bundle_sha256":"a"*64}):
            self.assertEqual(kaggle_workflow.publish_source(self.config,"fixture"),"created")
        self.assert_quiet(run)
        self.assertIn("create",run.call_args.args[0])

    def test_publish_version_calls_list_and_version_with_no_warn(self) -> None:
        result=CompletedProcess([],0,json.dumps([{"ref":"owner/project-source"}]),"")
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=result) as run, \
             mock.patch.object(kaggle_workflow,"bundle_source",return_value={"bundle_sha256":"a"*64}):
            self.assertEqual(kaggle_workflow.publish_source(self.config,"fixture"),"versioned")
        self.assert_quiet(run)
        self.assertIn("version",run.call_args.args[0])

    def test_push_prepared_uses_no_warn(self) -> None:
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=CompletedProcess([],0,"","")) as run, \
             mock.patch.object(kaggle_workflow,"verify_prepared_kernel",return_value={"profile":"fixture"}):
            kaggle_workflow.push_prepared_kernel(self.config,self.root/"prepared")
        self.assert_quiet(run)
        self.assertIn("push",run.call_args.args[0])

    def test_status_uses_no_warn(self) -> None:
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=CompletedProcess([],0,"complete","")) as run:
            kaggle_workflow.kernel_status(self.config)
        self.assert_quiet(run)

    def test_wait_uses_no_warn_for_each_poll(self) -> None:
        with mock.patch.object(kaggle_workflow.subprocess,"run",side_effect=[
                CompletedProcess([],0,"running",""),CompletedProcess([],0,"complete","")]) as run, \
             mock.patch.object(kaggle_workflow.time,"sleep"):
            kaggle_workflow.wait_for_kernel(self.config,timeout_seconds=60,interval_seconds=1)
        self.assert_quiet(run)
        self.assertEqual(run.call_count,2)

    def test_output_uses_no_warn(self) -> None:
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=CompletedProcess([],0,"","")) as run:
            kaggle_workflow.download_output(self.config,self.root/"output",force=True)
        self.assert_quiet(run)
        self.assertEqual(run.call_args.args[0][-1],"--force")

    def test_dataset_files_uses_no_warn_and_keeps_json_parsing(self) -> None:
        values=[{"name":"fixture.csv"}]
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=CompletedProcess([],0,json.dumps(values),"")) as run:
            self.assertEqual(kaggle_workflow.read_dataset_files("owner/project-source"),values)
        self.assert_quiet(run)

    def test_version_command_also_uses_no_warn(self) -> None:
        with mock.patch.object(kaggle_workflow.subprocess,"run",return_value=CompletedProcess([],0,"2.2.4","")) as run:
            kaggle_workflow.run_process(["kaggle","--version"])
        self.assertEqual(run.call_args.args[0],["kaggle","-W","--version"])

    def test_other_commands_unchanged(self) -> None:
        for command in [["git","status","--short"],["python","-V"],["tool","kaggle","status"]]:
            with self.subTest(command=command),mock.patch.object(kaggle_workflow.subprocess,"run") as run:
                kaggle_workflow.run_process(command)
                self.assertEqual(run.call_args.args[0],command)
                self.assertNotIn("-W",run.call_args.args[0])

    def test_existing_no_warn_is_not_duplicated(self) -> None:
        with mock.patch.object(kaggle_workflow.subprocess,"run") as run:
            kaggle_workflow.run_process(["kaggle","-W","quota"])
        self.assert_quiet(run)

    def test_callers_command_list_is_not_mutated(self) -> None:
        command=["kaggle","datasets","list"]
        with mock.patch.object(kaggle_workflow.subprocess,"run"):
            kaggle_workflow.run_process(command)
        self.assertEqual(command,["kaggle","datasets","list"])

    def test_xlsr_pair_profile_and_kernel_metadata_d033(self) -> None:
        base_cmd = ["python", "-c", "pass", "--data-root", "{data_root}", "--working-root", "{working_root}"]
        pair = kaggle_workflow.build_xlsr_pair_profiles(base_cmd)
        self.assertIn("X0-seed2027-X0-seed2028", pair)
        self.assertIn("X0-seed2027-X0-seed2028-dry-run", pair)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            payload = config_payload(root.relative_to(PROJECT_ROOT).as_posix())
            payload["profiles"].update(pair)
            cfg_path = root / "project.json"
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            config = kaggle_workflow.load_config(cfg_path)
            source_dir = config.workspace_dir / "source_dataset"
            source_dir.mkdir(parents=True)
            (source_dir / "source_manifest.json").write_text(
                json.dumps({"bundle_sha256": "b" * 64}), encoding="utf-8"
            )
            kernel_dir = kaggle_workflow.prepare_xlsr_pair_kernel(
                config, "X0-seed2027-X0-seed2028"
            )
            meta = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["is_private"])
            self.assertTrue(meta["enable_gpu"])
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertNotIn("gpu" + "_type", meta)
            single = kaggle_workflow.prepare_kernel(config, config.default_profile)
            single_meta = json.loads((single / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertNotIn("machine_shape", single_meta)
            self.assertFalse(meta["enable_internet"])
            self.assertEqual(len(meta["dataset_sources"]), 4)


if __name__ == "__main__":
    unittest.main()
