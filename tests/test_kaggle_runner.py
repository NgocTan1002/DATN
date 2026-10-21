from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "deploy" / "kaggle" / "runner.py"
SPEC = importlib.util.spec_from_file_location("kaggle_runner", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Không thể nạp deploy/kaggle/runner.py")
kaggle_runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kaggle_runner
SPEC.loader.exec_module(kaggle_runner)


class KaggleRunnerTests(unittest.TestCase):
    def test_cleanup_project_removes_only_expected_working_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            working = Path(temporary)
            project = working / "DoAnTTNT"
            project.mkdir()
            (project / "source.py").write_text("pass\n", encoding="utf-8")
            kaggle_runner.cleanup_project(project, working)
            self.assertFalse(project.exists())

    def test_find_source_payload_supports_nested_kaggle_mount(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            payload = root / "datasets" / "owner" / "source" / "project_source"
            payload.mkdir(parents=True)
            original_root = kaggle_runner.KAGGLE_INPUT_ROOT
            kaggle_runner.KAGGLE_INPUT_ROOT = root
            try:
                found = kaggle_runner.find_source_payload("source")
            finally:
                kaggle_runner.KAGGLE_INPUT_ROOT = original_root
            self.assertEqual(found, payload)

    def test_find_data_root_supports_nested_kaggle_mount(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            data_root = root / "datasets" / "owner" / "vsasv-data"
            data_root.mkdir(parents=True)
            (data_root / "development_20k_v2.csv").write_text(
                "manifest_version\n", encoding="utf-8"
            )
            original_root = kaggle_runner.KAGGLE_INPUT_ROOT
            kaggle_runner.KAGGLE_INPUT_ROOT = root
            try:
                found = kaggle_runner.find_data_root("vsasv-data")
            finally:
                kaggle_runner.KAGGLE_INPUT_ROOT = original_root
            self.assertEqual(found, data_root)

    def test_resolve_runtime_commands_replaces_data_root_token(self) -> None:
        commands = [
            ["python", "script.py", "--manifest", "__KAGGLE_DATA_ROOT__/m.csv"]
        ]
        resolved = kaggle_runner.resolve_runtime_commands(
            commands, Path("/actual/data")
        )
        self.assertEqual(
            resolved,
            [["python", "script.py", "--manifest", "/actual/data/m.csv"]],
        )

    def test_copy_verified_source_tree_checks_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            source_file = source / "script.py"
            source_file.write_text("print('ok')\n", encoding="utf-8")
            manifest = {
                "files": [
                    {
                        "path": "script.py",
                        "sha256": kaggle_runner.sha256_file(source_file),
                    }
                ]
            }
            destination = root / "destination"
            kaggle_runner.copy_verified_source_tree(
                source, destination, manifest
            )
            self.assertEqual(
                (destination / "script.py").read_text(encoding="utf-8"),
                "print('ok')\n",
            )

    def test_copy_verified_source_tree_rejects_tampered_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "script.py").write_text("tampered\n", encoding="utf-8")
            manifest = {"files": [{"path": "script.py", "sha256": "0" * 64}]}
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                kaggle_runner.copy_verified_source_tree(
                    source, root / "destination", manifest
                )


if __name__ == "__main__":
    unittest.main()
