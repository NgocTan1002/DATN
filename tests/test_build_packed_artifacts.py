from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.build_packed_artifacts import (
    DEFAULT_TARGETS,
    SCALE2_KAGGLE_DATASET_ID,
    SCALE2_KAGGLE_DATASET_TITLE,
    build_one,
    build_train_chunks,
    compare_prefix_with_v1_packed,
    parse_args,
    prepare_scale2_kaggle_metadata,
    render_markdown,
    shard_index,
    smoke_load_packed_artifact,
    train_wanted_by_shard,
    verify_manifest_shards_against_tree,
)
from scripts.ingest_evaluation_shards import TreeEntry
from src.data import MANIFEST_COLUMNS
from src.data.eval_ingest import chunk_paths


def _name(index: int) -> str:
    return f"train-{index:05d}-of-00432.parquet"


class BuildPackedArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.parquet_dir = self.root / "parquet"
        self.parquet_dir.mkdir()
        self.chunk_dir = self.root / "chunks"
        self.rows = [
            (0, "spk1/a.wav", "spk1", "bonafide", "closed_train"),
            (0, "spk2/b.wav", "spk2", "voice_conversion", "closed_train"),
            (1, "spk1/c.wav", "spk1", "replay", "closed_train"),
            (1, "spk3/d.wav", "spk3", "bonafide", "closed_dev"),
        ]
        self.tree: dict[int, TreeEntry] = {}
        for index in (0, 1):
            path = self.parquet_dir / _name(index)
            self._write_parquet(path, [row for row in self.rows if row[0] == index])
            self.tree[index] = TreeEntry(
                size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for index, file_name, speaker, utt, split in self.rows:
                writer.writerow(
                    {
                        "manifest_version": "development-20k-v2",
                        "shard": _name(index),
                        "file": file_name,
                        "speaker_id": speaker,
                        "split": split,
                        "binary_label": 0 if utt == "bonafide" else 1,
                        "utt_type": utt,
                        "native_sample_rate": 16_000,
                        "source_snapshot": "VSASV-HF-public-snapshot-v1",
                    }
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_parquet(self, path: Path, rows: list[tuple[int, str, str, str, str]]) -> None:
        target = path.as_posix().replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                'CREATE TABLE samples (file VARCHAR, audio STRUCT("array" DOUBLE[], '
                "sampling_rate BIGINT), label VARCHAR, utt_type VARCHAR)"
            )
            for position, (_index, file_name, speaker, utt, _split) in enumerate(rows):
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    [file_name, {"array": [0.01 * (position + 1)] * 32, "sampling_rate": 16_000}, speaker, utt],
                )
            connection.execute(f"COPY samples TO '{target}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def _build(self, tree: dict[int, TreeEntry] | None = None) -> dict[str, int]:
        return build_train_chunks(
            wanted_by_shard=train_wanted_by_shard(self.manifest),
            parquet_dir=self.parquet_dir,
            chunk_dir=self.chunk_dir,
            tree=tree or self.tree,
            shard_name=_name,
        )

    def test_shard_index_parses_official_names(self) -> None:
        self.assertEqual(shard_index("train-00431-of-00432.parquet"), 431)
        with self.assertRaises(ValueError):
            shard_index("data/train-00001-of-00432.parquet")

    def test_train_wanted_only_contains_train_rows_grouped_by_shard(self) -> None:
        wanted = train_wanted_by_shard(self.manifest)
        self.assertEqual(sorted(wanted), [0, 1])
        self.assertEqual(sorted(wanted[0]), ["spk1/a.wav", "spk2/b.wav"])
        self.assertEqual(sorted(wanted[1]), ["spk1/c.wav"])

    def test_build_train_chunks_keeps_parquet_and_skips_done(self) -> None:
        self.assertEqual(self._build(), {"ingested": 2, "already_done": 0})
        for index in (0, 1):
            self.assertTrue(chunk_paths(self.chunk_dir, index).done.is_file())
            self.assertTrue((self.parquet_dir / _name(index)).is_file())
        self.assertEqual(self._build(), {"ingested": 0, "already_done": 2})

    def test_build_train_chunks_rejects_wrong_sha256(self) -> None:
        tree = dict(self.tree)
        tree[1] = TreeEntry(size=tree[1].size, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._build(tree)
        self.assertFalse(chunk_paths(self.chunk_dir, 1).done.exists())
        self.assertTrue((self.parquet_dir / _name(1)).is_file())

    def test_markdown_uses_vietnamese_number_format(self) -> None:
        text = render_markdown(
            {
                "generated_at": "2026-11-02",
                "train_chunks": {"ingested": 1, "already_done": 0},
                "artifacts": [
                    {
                        "name": "dev-full", "split": "closed_dev", "manifest": "m.csv",
                        "artifact_count": 38777, "chunk_shard_count": 186,
                        "waveform_bytes": 3 * 1024**3, "build_seconds": 12.25,
                        "verify_seconds": 3.5, "waveforms_sha256": "a", "index_sha256": "b",
                        "metadata_sha256": "c", "source_manifest_sha256": "d",
                    }
                ],
            }
        )
        self.assertIn("| 38.777 | 186 | 3,00 |", text)

    def test_scale2_preserves_legacy_defaults_and_rejects_closed_test(self) -> None:
        args = parse_args([])
        self.assertEqual(args.targets, list(DEFAULT_TARGETS))
        self.assertIsNone(args.report_json)

        test_dir = self.root / "closed_test_full_v1"
        test_dir.mkdir()
        (test_dir / "metadata.json").write_text(json.dumps({"split": "closed_test"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "closed_test"):
            compare_prefix_with_v1_packed(scale2_dir=test_dir, v1_dir=test_dir, prefix_count=1)
        with self.assertRaisesRegex(ValueError, "closed_test"):
            prepare_scale2_kaggle_metadata(
                artifact_dir=test_dir,
                output_meta_path=self.root / "meta.json",
            )
        test_manifest = self.root / "closed_test_full_v1.csv"
        test_manifest.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "closed_test"):
            verify_manifest_shards_against_tree(
                manifest_csv=test_manifest,
                parquet_dir=self.parquet_dir,
                tree=self.tree,
                expected_split="closed_test",
            )

    def test_scale2_build_bit_compare_metadata_smoke_load_and_kaggle_meta(self) -> None:
        v1_manifest = self.root / "development_20k_v2.csv"
        v1_rows = [r for r in self.rows if r[4] == "closed_train"]
        with v1_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for index, file_name, speaker, utt, split in v1_rows:
                writer.writerow(
                    {
                        "manifest_version": "development-20k-v2",
                        "shard": _name(index),
                        "file": file_name,
                        "speaker_id": speaker,
                        "split": split,
                        "binary_label": 0 if utt == "bonafide" else 1,
                        "utt_type": utt,
                        "native_sample_rate": 16_000,
                        "source_snapshot": "VSASV-HF-public-snapshot-v1",
                    }
                )

        # Thêm shard 2 với 1 mẫu train mới cho scale2 (3 mẫu ×1 + 1 mẫu mới = 4)
        extra_row = (2, "spk5/e.wav", "spk5", "adversarial_attack", "closed_train")
        p2 = self.parquet_dir / _name(2)
        self._write_parquet(p2, [extra_row])
        self.tree[2] = TreeEntry(
            size=p2.stat().st_size,
            sha256=hashlib.sha256(p2.read_bytes()).hexdigest(),
        )
        scale2_manifest = self.root / "development_scale2_v1.csv"
        with scale2_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for index, file_name, speaker, utt, split in [*v1_rows, extra_row]:
                writer.writerow(
                    {
                        "manifest_version": "development-scale2-v1",
                        "shard": _name(index),
                        "file": file_name,
                        "speaker_id": speaker,
                        "split": split,
                        "binary_label": 0 if utt == "bonafide" else 1,
                        "utt_type": utt,
                        "native_sample_rate": 16_000,
                        "source_snapshot": "VSASV-HF-public-snapshot-v1",
                    }
                )

        shard_check = verify_manifest_shards_against_tree(
            manifest_csv=scale2_manifest, parquet_dir=self.parquet_dir, tree=self.tree
        )
        self.assertEqual(shard_check["checked_shards"], 3)
        self.assertTrue(shard_check["all_matched"])

        v1_chunk_dir = self.root / "v1_chunks"
        build_train_chunks(
            wanted_by_shard=train_wanted_by_shard(v1_manifest),
            parquet_dir=self.parquet_dir,
            chunk_dir=v1_chunk_dir,
            tree=self.tree,
            shard_name=_name,
        )
        v1_dir = self.root / "packed" / "train_development_20k_v2"
        build_one(
            name="train",
            manifest=v1_manifest,
            split="closed_train",
            chunk_dir=v1_chunk_dir,
            output_dir=v1_dir,
        )

        s2_chunk_dir = self.root / "s2_chunks"
        build_train_chunks(
            wanted_by_shard=train_wanted_by_shard(scale2_manifest),
            parquet_dir=self.parquet_dir,
            chunk_dir=s2_chunk_dir,
            tree=self.tree,
            shard_name=_name,
        )
        s2_dir = self.root / "packed" / "train_development_scale2_v1"
        art = build_one(
            name="train-scale2",
            manifest=scale2_manifest,
            split="closed_train",
            chunk_dir=s2_chunk_dir,
            output_dir=s2_dir,
        )
        meta = json.loads((s2_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["manifest_version"], "development-scale2-v1")
        self.assertEqual(meta["source_manifest"], "development_scale2_v1.csv")
        self.assertEqual(meta["artifact_count"], 4)
        self.assertEqual(meta["chunk_shard_count"], 3)
        self.assertEqual(meta["crop_seed"], 2026)
        self.assertEqual(art["index_sha256"], meta["index_sha256"])
        self.assertEqual(art["waveforms_sha256"], meta["waveforms_sha256"])

        bit_res = compare_prefix_with_v1_packed(
            scale2_dir=s2_dir, v1_dir=v1_dir, prefix_count=3
        )
        self.assertEqual(bit_res["matched_waveform_byte_slices"], 3)
        self.assertTrue(bit_res["passed"])

        audio_cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "audio.json").read_text(encoding="utf-8"))
        smoke = smoke_load_packed_artifact(
            artifact_dir=s2_dir,
            manifest_csv=scale2_manifest,
            audio_config=audio_cfg,
            sample_count=4,
            seed=2026,
            v1_dir=v1_dir,
        )
        self.assertTrue(smoke["passed"])
        self.assertEqual(smoke["sample_count"], 4)
        self.assertEqual(smoke["v1_prefix_indices_checked"], 3)

        kmeta_path = self.root / "kaggle" / "dataset-metadata.json"
        kinfo = prepare_scale2_kaggle_metadata(artifact_dir=s2_dir, output_meta_path=kmeta_path)
        kjson = json.loads(kmeta_path.read_text(encoding="utf-8"))
        self.assertEqual(kjson["id"], SCALE2_KAGGLE_DATASET_ID)
        self.assertEqual(kjson["title"], SCALE2_KAGGLE_DATASET_TITLE)
        self.assertTrue(kjson["isPrivate"])
        self.assertEqual(len(kinfo["files"]), 3)

        # Kiểm phát hiện lệch byte waveforms.f32
        broken_dir = self.root / "packed" / "broken_s2"
        shutil.copytree(s2_dir, broken_dir)
        wf_path = broken_dir / "waveforms.f32"
        wf_bytes = bytearray(wf_path.read_bytes())
        wf_bytes[0] ^= 0x01
        wf_path.write_bytes(bytes(wf_bytes))
        with self.assertRaisesRegex(ValueError, "Lệch byte waveform"):
            compare_prefix_with_v1_packed(scale2_dir=broken_dir, v1_dir=v1_dir, prefix_count=3)

        # Kiểm phát hiện lệch trường index.csv
        shutil.copytree(s2_dir, self.root / "packed" / "broken_idx")
        broken_idx = self.root / "packed" / "broken_idx"
        idx_text = (broken_idx / "index.csv").read_text(encoding="utf-8")
        (broken_idx / "index.csv").write_text(idx_text.replace("spk1/a.wav", "spk1/x.wav"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Lệch index"):
            compare_prefix_with_v1_packed(scale2_dir=broken_idx, v1_dir=v1_dir, prefix_count=3)


if __name__ == "__main__":
    unittest.main()
