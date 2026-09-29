from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import check_leakage, make_splits


class SplitProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "metadata.csv"
        self.rows = self._synthetic_rows()
        self._write_csv(self.metadata, self.rows)
        self.split_dir = self.root / "splits"
        make_splits.generate_splits(self.metadata, self.split_dir, seed=2026)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _synthetic_rows() -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for index in range(12):
            speaker = f"real{index:03d}"
            rows.append(
                {"file": f"{speaker}/bona.wav", "label": speaker, "utt_type": "bonafide"}
            )
        for index in range(12):
            speaker = f"attack{index:03d}"
            rows.extend(
                [
                    {"file": f"{speaker}/bona.wav", "label": speaker, "utt_type": "bonafide"},
                    {"file": f"{speaker}/vc.wav", "label": speaker, "utt_type": "voice_conversion"},
                    {"file": f"{speaker}/adv.wav", "label": speaker, "utt_type": "adversarial_attack"},
                ]
            )
        for index in range(8):
            speaker = f"replay{index:03d}"
            rows.extend(
                [
                    {"file": f"{speaker}/bona.wav", "label": speaker, "utt_type": "bonafide"},
                    {"file": f"{speaker}/replay.wav", "label": speaker, "utt_type": "replay"},
                ]
            )
        return rows

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=make_splits.EXPECTED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_valid_protocol_passes(self) -> None:
        report = check_leakage.check_splits(self.metadata, self.split_dir)
        self.assertTrue(report["passed"], report["issues"])

    def test_generation_is_deterministic(self) -> None:
        second_dir = self.root / "splits-second"
        make_splits.generate_splits(self.metadata, second_dir, seed=2026)
        for filename in make_splits.SPLIT_FILENAMES.values():
            self.assertEqual(
                (self.split_dir / filename).read_bytes(),
                (second_dir / filename).read_bytes(),
            )

    def test_duplicate_train_row_in_dev_is_rejected(self) -> None:
        train_path = self.split_dir / "closed_train.csv"
        dev_path = self.split_dir / "closed_dev.csv"
        dev_rows = self._read_csv(dev_path)
        dev_rows.append(self._read_csv(train_path)[0])
        self._write_csv(dev_path, dev_rows)

        report = check_leakage.check_splits(self.metadata, self.split_dir)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["passed"])
        self.assertIn("speaker_overlap", codes)
        self.assertIn("file_overlap", codes)

    def test_unseen_attack_in_train_is_rejected(self) -> None:
        train_path = self.split_dir / "open_train_vc.csv"
        unseen_path = self.split_dir / "open_unseen_test_adversarial.csv"
        unseen_attack = next(
            row
            for row in self._read_csv(unseen_path)
            if row["utt_type"] == "adversarial_attack"
        )
        train_rows = self._read_csv(train_path)
        train_rows.append(unseen_attack)
        self._write_csv(train_path, train_rows)

        report = check_leakage.check_splits(self.metadata, self.split_dir)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["passed"])
        self.assertIn("attack_constraint", codes)

    def test_cross_partition_audio_hash_is_rejected(self) -> None:
        audio_root = self.root / "audio"
        for index, row in enumerate(self.rows):
            path = audio_root.joinpath(*Path(row["file"]).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"unique-audio-{index}".encode("utf-8"))

        train_row = self._read_csv(self.split_dir / "closed_train.csv")[0]
        dev_row = self._read_csv(self.split_dir / "closed_dev.csv")[0]
        duplicated_content = b"deliberate-duplicate-audio"
        audio_root.joinpath(*Path(train_row["file"]).parts).write_bytes(duplicated_content)
        audio_root.joinpath(*Path(dev_row["file"]).parts).write_bytes(duplicated_content)

        report = check_leakage.check_splits(
            self.metadata, self.split_dir, audio_root=audio_root
        )
        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["passed"])
        self.assertIn("audio_hash_overlap", codes)


if __name__ == "__main__":
    unittest.main()
