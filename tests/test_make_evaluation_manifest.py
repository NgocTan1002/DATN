from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_development_manifest import stable_key
from scripts.make_evaluation_manifest import (
    EvalRecord,
    build_manifests,
    load_audit,
    load_chunk_records,
    write_manifest,
)
from scripts.plan_evaluation_shards import ShardGeometry
from src.data.vsasv import MANIFEST_COLUMNS


GEOMETRY = ShardGeometry(first_count=2, first_rows=3, rest_count=2, rest_rows=2)


def rec(file: str, split: str, speaker: str, utt: str, digest: str) -> EvalRecord:
    return EvalRecord(
        file=file,
        split=split,
        speaker_id=speaker,
        utt_type=utt,
        binary_label=0 if utt == "bonafide" else 1,
        native_sample_rate=16_000,
        content_sha256=digest,
        shard="train-00000-of-00004.parquet",
    )


def expected_rows(records: list[EvalRecord]) -> dict[str, dict[str, tuple[str, str]]]:
    result: dict[str, dict[str, tuple[str, str]]] = {
        "closed_dev": {},
        "closed_test": {},
    }
    for item in records:
        result.setdefault(item.split, {})[item.file] = (item.speaker_id, item.utt_type)
    return result


class BuildManifestsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_hashes = {"HT": 0}
        self.train_speakers = {"id_train"}
        self.records = [
            rec("d1.wav", "closed_dev", "id_d", "bonafide", "H1"),
            rec("d2.wav", "closed_dev", "id_d", "bonafide", "HT"),
            rec("d3.wav", "closed_dev", "id_d", "bonafide", "H2"),
            rec("d4.wav", "closed_dev", "id_d", "bonafide", "H2"),
            rec("t1.wav", "closed_test", "id_t", "bonafide", "H1"),
            rec("t2.wav", "closed_test", "id_t", "bonafide", "HT"),
            rec("t3.wav", "closed_test", "id_t", "bonafide", "H5"),
            rec("t4.wav", "closed_test", "id_t", "bonafide", "H5"),
            rec("t5.wav", "closed_test", "id_t", "bonafide", "H6"),
        ]

    def _build(self, records=None, **overrides):
        records = self.records if records is None else records
        arguments = dict(
            records=records,
            split_rows=expected_rows(records),
            train_hashes=self.train_hashes,
            train_speakers=self.train_speakers,
            audit_dev_hashes={},
            seed=2026,
            max_loss_fraction=1.0,
        )
        arguments.update(overrides)
        return build_manifests(**arguments)

    def test_cross_split_and_internal_duplicates_follow_d025(self) -> None:
        result = self._build()
        self.assertEqual(result.status, "ĐẠT")
        dev_rep = min(("d3.wav", "d4.wav"), key=lambda name: stable_key(name, 2026))
        test_rep = min(("t3.wav", "t4.wav"), key=lambda name: stable_key(name, 2026))
        self.assertEqual(
            sorted(item.file for item in result.kept["closed_dev"]),
            sorted(["d1.wav", dev_rep]),
        )
        self.assertEqual(
            sorted(item.file for item in result.kept["closed_test"]),
            sorted([test_rep, "t5.wav"]),
        )
        dev_stats = result.stats["closed_dev"]["dropped"]
        self.assertEqual(dev_stats["train_overlap"], {"bonafide": 1})
        self.assertEqual(dev_stats["internal_duplicate"], {"bonafide": 1})
        test_stats = result.stats["closed_test"]["dropped"]
        self.assertEqual(test_stats["train_overlap"], {"bonafide": 1})
        self.assertEqual(test_stats["dev_overlap"], {"bonafide": 1})
        self.assertEqual(test_stats["internal_duplicate"], {"bonafide": 1})

    def test_kept_sets_have_unique_hashes_and_no_overlap(self) -> None:
        result = self._build()
        dev_hashes = {item.content_sha256 for item in result.kept["closed_dev"]}
        test_hashes = {item.content_sha256 for item in result.kept["closed_test"]}
        self.assertEqual(len(dev_hashes), len(result.kept["closed_dev"]))
        self.assertEqual(len(test_hashes), len(result.kept["closed_test"]))
        self.assertFalse(dev_hashes & test_hashes)
        self.assertNotIn("HT", dev_hashes | test_hashes)

    def test_ledger_has_one_row_per_input_record(self) -> None:
        result = self._build()
        self.assertEqual(len(result.ledger), len(self.records))
        by_file = {row["file"]: row for row in result.ledger}
        self.assertEqual(by_file["d2.wav"]["decision"], "train_overlap")
        self.assertEqual(by_file["d1.wav"]["decision"], "kept")

    def test_loss_above_threshold_stops_without_manifests(self) -> None:
        records = [
            rec(f"d{i}.wav", "closed_dev", "id_d", "bonafide", f"H{i}") for i in range(9)
        ] + [rec("dx.wav", "closed_dev", "id_d", "bonafide", "HT")]
        result = self._build(records=records, max_loss_fraction=0.05)
        self.assertEqual(result.status, "DỪNG")
        self.assertTrue(any("closed_dev" in text and "bonafide" in text for text in result.stop_reasons))

    def test_loss_exactly_at_threshold_passes(self) -> None:
        records = [
            rec(f"d{i}.wav", "closed_dev", "id_d", "bonafide", f"H{i}") for i in range(19)
        ] + [rec("dx.wav", "closed_dev", "id_d", "bonafide", "HT")]
        result = self._build(records=records, max_loss_fraction=0.05)
        self.assertEqual(result.status, "ĐẠT")

    def test_hash_group_with_mixed_labels_stops(self) -> None:
        records = [
            rec("d1.wav", "closed_dev", "id_d", "bonafide", "HX"),
            rec("t1.wav", "closed_test", "id_t", "voice_conversion", "HX"),
        ]
        result = self._build(records=records)
        self.assertEqual(result.status, "DỪNG")
        self.assertTrue(any("nhãn" in text for text in result.stop_reasons))

    def test_label_conflict_with_train_hash_stops(self) -> None:
        records = [rec("d1.wav", "closed_dev", "id_d", "voice_conversion", "HT")]
        result = self._build(records=records)
        self.assertEqual(result.status, "DỪNG")

    def test_missing_records_are_rejected(self) -> None:
        split_rows = expected_rows(self.records)
        split_rows["closed_dev"]["thieu.wav"] = ("id_d", "bonafide")
        with self.assertRaisesRegex(ValueError, "thiếu"):
            self._build(split_rows=split_rows)

    def test_unexpected_records_are_rejected(self) -> None:
        split_rows = expected_rows(self.records)
        del split_rows["closed_dev"]["d1.wav"]
        with self.assertRaisesRegex(ValueError, "thừa"):
            self._build(split_rows=split_rows)

    def test_metadata_mismatch_is_rejected(self) -> None:
        split_rows = expected_rows(self.records)
        split_rows["closed_dev"]["d1.wav"] = ("id_khac", "bonafide")
        with self.assertRaisesRegex(ValueError, "không khớp"):
            self._build(split_rows=split_rows)

    def test_speaker_overlap_between_splits_is_rejected(self) -> None:
        records = [
            rec("d1.wav", "closed_dev", "id_chung", "bonafide", "H1"),
            rec("t1.wav", "closed_test", "id_chung", "bonafide", "H2"),
        ]
        with self.assertRaisesRegex(ValueError, "speaker"):
            self._build(records=records)

    def test_speaker_overlap_with_train_is_rejected(self) -> None:
        records = [rec("d1.wav", "closed_dev", "id_train", "bonafide", "H1")]
        with self.assertRaisesRegex(ValueError, "speaker"):
            self._build(records=records)

    def test_audit_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "audit"):
            self._build(audit_dev_hashes={"d1.wav": "KHAC"})

    def test_audit_hash_match_is_counted(self) -> None:
        result = self._build(audit_dev_hashes={"d1.wav": "H1", "d3.wav": "H2"})
        self.assertEqual(result.stats["audit_crosscheck"], {"compared": 2, "mismatched": 0})


class IoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_chunk(self, index: int, rows: list[dict[str, str]], done: bool = True) -> None:
        chunk = self.root / "chunks"
        chunk.mkdir(exist_ok=True)
        columns = [
            "file", "split", "speaker_id", "utt_type", "binary_label",
            "native_sample_rate", "offset_samples", "num_samples",
            "content_sha256", "waveform_sha256",
        ]
        with (chunk / f"shard-{index:05d}.index.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({**{c: "0" for c in columns}, **row})
        if done:
            (chunk / f"shard-{index:05d}.done.json").write_text("{}", encoding="utf-8")

    def test_load_chunk_records_reads_all_requested_shards(self) -> None:
        row = {"file": "a.wav", "split": "closed_dev", "speaker_id": "id1",
               "utt_type": "bonafide", "binary_label": "0",
               "native_sample_rate": "16000", "content_sha256": "HA"}
        self._write_chunk(1, [row])
        self._write_chunk(2, [{**row, "file": "b.wav", "content_sha256": "HB"}])
        records = load_chunk_records(self.root / "chunks", [1, 2], GEOMETRY)
        self.assertEqual([r.file for r in records], ["a.wav", "b.wav"])
        self.assertEqual(records[0].shard, "train-00001-of-00004.parquet")
        self.assertEqual(records[1].binary_label, 0)

    def test_load_chunk_records_requires_done_marker(self) -> None:
        self._write_chunk(1, [], done=False)
        with self.assertRaisesRegex(ValueError, "chưa hoàn tất"):
            load_chunk_records(self.root / "chunks", [1], GEOMETRY)

    def test_load_audit_returns_train_and_dev_views(self) -> None:
        rows = [
            {"file": "tr1.wav", "speaker_id": "id_tr", "split": "closed_train",
             "binary_label": 0, "waveform_sha256": "H1"},
            {"file": "dv1.wav", "speaker_id": "id_dv", "split": "closed_dev",
             "binary_label": 1, "waveform_sha256": "H2"},
        ]
        path = self.root / "audit.json"
        path.write_text(json.dumps({"row_hashes": rows}), encoding="utf-8")
        audit = load_audit(path)
        self.assertEqual(audit.train_hashes, {"H1": 0})
        self.assertEqual(audit.train_speakers, {"id_tr"})
        self.assertEqual(audit.dev_hashes, {"dv1.wav": "H2"})

    def test_write_manifest_uses_v2_columns_and_sorted_rows(self) -> None:
        records = [
            rec("b.wav", "closed_dev", "id_d", "bonafide", "H2"),
            rec("a.wav", "closed_dev", "id_d", "voice_conversion", "H1"),
        ]
        target = self.root / "m" / "closed_dev_full_v1.csv"
        digest = write_manifest(target, records, "closed-dev-full-v1")
        with target.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames or ()), MANIFEST_COLUMNS)
            rows = list(reader)
        self.assertEqual([row["file"] for row in rows], ["a.wav", "b.wav"])
        self.assertEqual(rows[0]["binary_label"], "1")
        self.assertEqual(rows[0]["manifest_version"], "closed-dev-full-v1")
        self.assertEqual(rows[0]["source_snapshot"], "VSASV-HF-public-snapshot-v1")
        self.assertEqual(digest, hashlib.sha256(target.read_bytes()).hexdigest())

    def test_write_manifest_refuses_to_overwrite(self) -> None:
        target = self.root / "x.csv"
        target.write_text("da co", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            write_manifest(target, [rec("a.wav", "closed_dev", "id", "bonafide", "H")], "v")


if __name__ == "__main__":
    unittest.main()
