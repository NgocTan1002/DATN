from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import duckdb

from scripts.smoke_test_development_manifest_loader import parse_args, smoke_split
from src.data import MANIFEST_COLUMNS


class DevelopmentManifestLoaderSmokeTests(unittest.TestCase):
    def test_cli_accepts_locked_amplitude_policy(self) -> None:
        args = parse_args(["--amplitude-policy", "peak"])

        self.assertEqual(args.amplitude_policy, "peak")

    def test_cli_rejects_unknown_policy_in_vietnamese(self) -> None:
        error = io.StringIO()

        with redirect_stderr(error), self.assertRaises(SystemExit):
            parse_args(["--amplitude-policy", "unknown"])

        self.assertIn("Chính sách biên độ không hợp lệ", error.getvalue())

    def test_smoke_split_applies_selected_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            parquet_dir = root / "parquet"
            parquet_dir.mkdir()
            shard_name = "train-00000-of-00432.parquet"
            self._write_parquet(parquet_dir / shard_name)
            manifest = root / "development_20k_v2.csv"
            self._write_manifest(manifest, shard_name)
            output = io.StringIO()

            with redirect_stdout(output):
                smoke_split(
                    manifest,
                    parquet_dir,
                    "closed_dev",
                    2026,
                    "development-20k-v2",
                    amplitude_policy="peak",
                    expected_count=2,
                )

            self.assertIn("policy=peak", output.getvalue())

    def _write_parquet(self, path: Path) -> None:
        target = path.as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples ('
                'file VARCHAR, audio STRUCT("array" DOUBLE[], sampling_rate BIGINT), '
                'label VARCHAR, utt_type VARCHAR)'
            )
            connection.execute(
                "INSERT INTO samples VALUES (?, ?, ?, ?), (?, ?, ?, ?)",
                [
                    "speaker1/bonafide.wav",
                    {"array": [0.1, -0.5, 0.25], "sampling_rate": 16_000},
                    "speaker1",
                    "bonafide",
                    "speaker2/spoof.wav",
                    {"array": [0.2, -0.4, 0.1], "sampling_rate": 16_000},
                    "speaker2",
                    "voice_conversion",
                ],
            )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _write_manifest(self, path: Path, shard_name: str) -> None:
        rows = [
            {
                "manifest_version": "development-20k-v2",
                "shard": shard_name,
                "file": "speaker1/bonafide.wav",
                "speaker_id": "speaker1",
                "split": "closed_dev",
                "binary_label": 0,
                "utt_type": "bonafide",
                "native_sample_rate": 16_000,
                "source_snapshot": "VSASV-HF-public-snapshot-v1",
            },
            {
                "manifest_version": "development-20k-v2",
                "shard": shard_name,
                "file": "speaker2/spoof.wav",
                "speaker_id": "speaker2",
                "split": "closed_dev",
                "binary_label": 1,
                "utt_type": "voice_conversion",
                "native_sample_rate": 16_000,
                "source_snapshot": "VSASV-HF-public-snapshot-v1",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
