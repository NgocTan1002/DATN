"""T23: đối chiếu metadata trước bootstrap, không đọc dữ liệu test."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import paired_bootstrap_compare as compare


class PairedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [{"file": str(i), "speaker_id": f"s{i}", "label": int(i >= 2), "score": score}
                        for i, score in enumerate([.1, .7, .4, .9])]

    def test_identical_scores_have_zero_paired_difference_and_ci(self) -> None:
        value = compare.compare_records(self.records, self.records, {"n_resamples": 40, "seed": 2026, "confidence": .95})
        difference = value["paired_eer_difference"]
        self.assertEqual((difference["estimate"], difference["ci_low"], difference["ci_high"]), (0, 0, 0))
        self.assertEqual(value["system_a"], value["system_b"])

    def test_mismatch_file_order_label_and_speaker_rejected(self) -> None:
        for key in ("file", "label", "speaker_id"):
            values = deepcopy(self.records)
            values[0][key] = 1 if key == "label" else "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "lệch"):
                compare.compare_records(self.records, values, {"n_resamples": 10, "seed": 2026, "confidence": .95})
        with self.assertRaisesRegex(ValueError, "lệch"):
            compare.aligned_records(self.records, self.records[::-1])
        with self.assertRaisesRegex(ValueError, "lệch"):
            compare.aligned_records(self.records, self.records[:-1])

    def test_duplicate_files_rejected(self) -> None:
        directory = compare.ROOT / ".cache/t23/temp"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary) / "scores.json"
            path.write_text(json.dumps({"records": [self.records[0]] * 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trùng"):
                compare.read_scores(path)

    def test_saved_report_and_score_formats(self) -> None:
        directory = compare.ROOT / ".cache/t23/temp"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            for name, value in (("scores", {"records": self.records}),
                                ("report", {"controller": {"best": {"development": {"records": self.records}}}})):
                path = Path(temporary) / (name + ".json")
                path.write_text(json.dumps(value), encoding="utf-8")
                self.assertEqual(compare.read_scores(path), self.records)

    def test_refuses_closed_test_path_before_open(self) -> None:
        with self.assertRaisesRegex(ValueError, "test"):
            compare.read_scores(Path("closed_test_full_v1.csv"))


if __name__ == "__main__":
    unittest.main()
