"""Kiểm thử cho scripts/check_test_inventory.py (PREP-01)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import scripts.check_test_inventory as cti


class CheckTestInventoryTests(unittest.TestCase):
    """Kiểm tra băm luồng byte, khóa không parse bảng test và whitelist lệnh Kaggle."""

    def test_stream_sha256_and_size_multi_chunk(self) -> None:
        payload = (b"vsasv-packed-test-bytes-" * 430) + b"tail"
        expected_sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sample.bin"
            target.write_bytes(payload)
            got_sha, got_size = cti.stream_sha256_and_size(target, chunk_size=1024)
        self.assertEqual(got_sha, expected_sha)
        self.assertEqual(got_size, len(payload))

    def test_rejects_text_mode_and_table_parsing_on_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_index = Path(tmp) / "index.csv"
            fake_index.write_bytes(b"col1,col2\n0,1\n")

            for bad_mode in ("r", "rt", "w", "wb", "a", "r+", "rb+"):
                with self.assertRaises(ValueError):
                    cti.open_test_artifact_binary_only(fake_index, bad_mode)

            with cti.open_test_artifact_binary_only(fake_index, "rb") as handle:
                self.assertEqual(handle.read(4), b"col1")

        for test_path in (
            Path("data/manifests/closed_test_full_v1.csv"),
            Path("data/processed/packed/closed_test_full_v1/index.csv"),
        ):
            with self.assertRaises(PermissionError):
                cti.forbid_test_table_parse(test_path)

    def test_source_ast_forbids_csv_or_audio_imports(self) -> None:
        source_path = Path(cti.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        forbidden_modules = {"csv", "pandas", "soundfile", "torchaudio", "scipy", "librosa", "wave"}
        self.assertTrue(imported.isdisjoint(forbidden_modules), f"Import bị cấm: {imported & forbidden_modules}")

        func_nodes = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        check_fn = func_nodes["check_local_test_inventory"]
        read_text_targets: list[str] = []
        for node in ast.walk(check_fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_text":
                if isinstance(node.func.value, ast.Name):
                    read_text_targets.append(node.func.value.id)
        self.assertEqual(
            set(read_text_targets),
            {"metadata_path", "config_v1_path", "config_x1r_path"},
        )

    def test_validate_kaggle_readonly_command_whitelist(self) -> None:
        allowed = [
            ["kaggle", "-W", "quota"],
            ["kaggle", "-W", "datasets", "list", "--mine"],
            ["kaggle", "-W", "datasets", "files", "ngctnnguyn1111/doanttnt-t14-packed"],
        ]
        for cmd in allowed:
            self.assertEqual(cti.validate_kaggle_readonly_command(cmd), cmd)

        disallowed = [
            ["kaggle", "quota"],
            ["kaggle", "-W", "quota", "--format", "json"],
            ["kaggle", "-W", "datasets", "list"],
            ["kaggle", "-W", "datasets", "download", "ngctnnguyn1111/doanttnt-t14-packed"],
            ["kaggle", "-W", "datasets", "create", "-p", "."],
            ["kaggle", "-W", "datasets", "version", "-p", ".", "-m", "v2"],
            ["kaggle", "-W", "datasets", "files", "--help"],
            ["kaggle", "-W", "datasets", "files", "ngctnnguyn1111/doanttnt-t14-packed", "--csv"],
            ["kaggle", "-W", "kernels", "push", "-p", "."],
            ["kaggle", "-W", "kernels", "output", "ngctnnguyn1111/doanttnt-xlsr-d028"],
            ["python", "-c", "print(1)"],
        ]
        for cmd in disallowed:
            with self.assertRaises(ValueError, msg=f"Phải từ chối lệnh: {cmd}"):
                cti.validate_kaggle_readonly_command(cmd)

    def test_kaggle_log_parsers_and_auth_error(self) -> None:
        quota_sample = (
            "Warning: outdated\n"
            "resource  used    remaining  total   refreshAt\n"
            "--------  ------  ---------  ------  -------------------\n"
            "GPU       28.94h  16.06h     45.00h  2026-11-21T00:00:00\n"
            "TPU       0.00h   20.00h     20.00h  2026-11-21T00:00:00\n"
        )
        parsed_quota = cti._parse_kaggle_quota_text(quota_sample)
        self.assertEqual(parsed_quota["gpu"]["remaining_hours"], 16.06)
        self.assertEqual(parsed_quota["gpu"]["total_hours"], 45.0)

        with self.assertRaises(RuntimeError):
            cti._parse_kaggle_quota_text("401 Unauthorized: Authentication required")

        datasets_sample = (
            "ref                                    title                     size  lastUpdated                 downloadCount  voteCount  usabilityRating\n"
            "-------------------------------------  ----------------  -----------  --------------------------  -------------  ---------  ---------------\n"
            "ngctnnguyn1111/doanttnt-t14-packed     DoAnTTNT T14 packed  13757151484  2026-11-03 10:43:49.433000              6          0  0.23529412\n"
        )
        parsed_ds = cti._parse_kaggle_datasets_list_text(datasets_sample)
        self.assertEqual(len(parsed_ds), 1)
        self.assertEqual(parsed_ds[0]["ref"], "ngctnnguyn1111/doanttnt-t14-packed")

        files_sample = (
            "name                               size  creationDate\n"
            "----------------------------  ---------  --------------------------\n"
            "closed_dev_full_v1/index.csv    7684400  2026-11-03 10:45:23.851000\n"
        )
        parsed_files = cti._parse_kaggle_dataset_files_text(files_sample)
        self.assertEqual(parsed_files[0]["name"], "closed_dev_full_v1/index.csv")
        self.assertEqual(parsed_files[0]["size_bytes"], 7684400)

        with tempfile.TemporaryDirectory() as tmp:
            utf16_log = Path(tmp) / "quota_utf16.log"
            utf16_log.write_bytes(quota_sample.encode("utf-16"))
            self.assertEqual(cti._read_log_text(utf16_log), quota_sample)

    def test_check_local_test_inventory_rejects_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            packed = fake_root / "data" / "processed" / "packed" / "closed_test_full_v1"
            manifests = fake_root / "data" / "manifests"
            configs = fake_root / "configs"
            packed.mkdir(parents=True)
            manifests.mkdir(parents=True)
            configs.mkdir(parents=True)

            (manifests / "closed_test_full_v1.csv").write_bytes(b"bad-manifest")
            (packed / "index.csv").write_bytes(b"bad-index")
            (packed / "waveforms.f32").write_bytes(b"\x00" * 16)
            (packed / "metadata.json").write_text(
                json.dumps(
                    {
                        "format_version": "vsasv-packed-v1",
                        "preprocessing_stage": "post_resample_pre_amplitude_pre_crop",
                        "dtype": "float32-le",
                        "target_sample_rate": 16000,
                        "crop_seed": 2026,
                        "split": "closed_test",
                        "artifact_count": 30698,
                        "chunk_shard_count": 162,
                        "total_samples": 4,
                        "source_manifest_sha256": cti.EXPECTED_TEST_CONTEXT["source_manifest_sha256"],
                        "index_sha256": cti.EXPECTED_TEST_CONTEXT["index_sha256"],
                        "waveforms_sha256": cti.EXPECTED_TEST_CONTEXT["waveforms_sha256"],
                    }
                ),
                encoding="utf-8",
            )
            cfg = {"test": {"manifest_sha256": cti.EXPECTED_TEST_CONTEXT["source_manifest_sha256"], "samples": 30698}}
            (configs / "test_evaluation.json").write_text(json.dumps(cfg), encoding="utf-8")
            (configs / "test_evaluation_x1r.json").write_text(json.dumps(cfg), encoding="utf-8")

            with self.assertRaises(ValueError):
                cti.check_local_test_inventory(fake_root)


if __name__ == "__main__":
    unittest.main()
