"""Kiểm thử tải shard thích ứng, khử trùng và lập/audit manifest scale2 (không gọi mạng)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_scale2_manifest import (  # noqa: E402
    REPOSITORY,
    REVISION,
    SCALE2_MANIFEST_VERSION,
    audit_scale2_manifest_independently,
    download_and_verify_shard,
    public_hf_downloader,
    run_scale2_build,
)
from scripts.ingest_evaluation_shards import TreeEntry  # noqa: E402
from scripts.make_development_manifest import MANIFEST_COLUMNS, UTT_TYPES, sha256_file  # noqa: E402
from scripts.plan_evaluation_shards import ShardGeometry, shard_filename  # noqa: E402
from scripts.plan_scale2_shards import LEDGER_COLUMNS, order_digest  # noqa: E402
from src.data.content_hash import waveform_sha256  # noqa: E402


def _make_wave(freq: float, length: int = 160, amp: float = 0.4) -> list[float]:
    return [amp * math.sin(2.0 * math.pi * freq * (i / 16000.0)) for i in range(length)]


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet_shard(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            'CREATE TABLE tmp_shard (file VARCHAR, label VARCHAR, utt_type VARCHAR, audio STRUCT("array" DOUBLE[], sampling_rate BIGINT))'
        )
        for rec in records:
            connection.execute(
                """INSERT INTO tmp_shard VALUES (?, ?, ?, {'array': ?::DOUBLE[], 'sampling_rate': ?::BIGINT})""",
                [
                    rec["file"],
                    rec["speaker_id"],
                    rec["utt_type"],
                    rec["waveform"],
                    rec["sample_rate"],
                ],
            )
        connection.execute(f"COPY tmp_shard TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        connection.close()


def _create_fixture_env(
    root: Path,
    *,
    make_initial_shard_all_dev_overlap: bool = False,
    make_internal_dup_pair: bool = False,
) -> dict[str, Any]:
    """Tạo fixture nhỏ 4 shard (mỗi shard 4 mẫu = 16 mẫu) hoàn toàn khép kín trong tmp."""
    geometry = ShardGeometry(first_count=4, first_rows=4, rest_count=0, rest_rows=0)
    data_dir = root / "parquet_data"
    remote_dir = root / "remote_repo"
    incoming_dir = root / "incoming"
    cache_dir = root / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    remote_dir.mkdir(parents=True, exist_ok=True)

    # Xác định thứ tự remote shard từ 1..3 theo SHA256("2026|shard")
    remote_ordered = sorted(
        (1, 2, 3),
        key=lambda idx: (
            order_digest(shard_filename(geometry, idx), 2026),
            shard_filename(geometry, idx),
        ),
    )

    dev_wave = _make_wave(210.0)
    dev_hash = waveform_sha256(dev_wave, 16000)
    test_wave = _make_wave(220.0)
    test_hash = waveform_sha256(test_wave, 16000)
    dup_wave = _make_wave(315.0)

    # Shard 0 (cục bộ ban đầu): 4 mẫu train hiện tại (1 mẫu mỗi utt_type)
    # -> 4 dòng gốc mỗi shard, nhưng chia đều: shard 0 chỉ có 1 bonafide + 1 vc + 1 ap + 1 replay.
    shard_records: dict[int, list[dict[str, Any]]] = {}
    shard_records[0] = [
        {"file": f"spk0/{u}_cur.wav", "speaker_id": "spk0", "utt_type": u, "waveform": _make_wave(100.0 + idx * 10), "sample_rate": 16000}
        for idx, u in enumerate(UTT_TYPES)
    ]

    # Remote #1 (theo SHA256("2026|shard")):
    r1, r2, r3 = remote_ordered
    shard_records[r1] = [
        {
            "file": "spk1/bon_r1.wav",
            "speaker_id": "spk1",
            "utt_type": "bonafide",
            # Nếu test adaptive_next_shard, làm bonafide ở r1 trùng dev_hash -> thiếu bonafide sau r1
            "waveform": list(dev_wave) if make_initial_shard_all_dev_overlap else _make_wave(410.0),
            "sample_rate": 16000,
        },
        {
            "file": "spk1/vc_r1.wav",
            "speaker_id": "spk1",
            "utt_type": "voice_conversion",
            "waveform": _make_wave(420.0),
            "sample_rate": 16000,
        },
        {
            "file": "spk1/ap_r1.wav",
            "speaker_id": "spk1",
            "utt_type": "adversarial_attack",
            "waveform": _make_wave(430.0),
            "sample_rate": 16000,
        },
        {
            "file": "spk1/rep_r1.wav",
            "speaker_id": "spk1",
            "utt_type": "replay",
            "waveform": _make_wave(440.0),
            "sample_rate": 16000,
        },
    ]

    dup_files = ["spk2/bon_dup_a.wav", "spk2/bon_dup_b.wav"]
    expected_dup_winner = min(dup_files, key=lambda n: (order_digest(n, 2026), n))
    expected_dup_loser = max(dup_files, key=lambda n: (order_digest(n, 2026), n))
    shard_records[r2] = [
        {
            "file": dup_files[0],
            "speaker_id": "spk2",
            "utt_type": "bonafide",
            "waveform": list(dup_wave) if make_internal_dup_pair else _make_wave(510.0),
            "sample_rate": 16000,
        },
        {
            "file": dup_files[1],
            "speaker_id": "spk2",
            "utt_type": "bonafide",
            "waveform": list(dup_wave) if make_internal_dup_pair else _make_wave(515.0),
            "sample_rate": 16000,
        },
        {
            "file": "spk2/vc_r2.wav",
            "speaker_id": "spk2",
            "utt_type": "voice_conversion",
            "waveform": _make_wave(520.0),
            "sample_rate": 16000,
        },
        {
            "file": "spk2/testov_r2.wav",
            "speaker_id": "spk2",
            "utt_type": "bonafide",
            "waveform": list(test_wave),
            "sample_rate": 16000,
        },
    ]
    shard_records[r3] = [
        {"file": f"spk3/{u}_r3.wav", "speaker_id": "spk3", "utt_type": u, "waveform": _make_wave(610.0 + idx * 10), "sample_rate": 16000}
        for idx, u in enumerate(UTT_TYPES)
    ]

    tree_entries: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, object]] = []
    closed_train_rows: list[dict[str, object]] = []
    for idx in range(4):
        s_name = shard_filename(geometry, idx)
        dest = (data_dir if idx == 0 else remote_dir) / s_name
        _write_parquet_shard(dest, shard_records[idx])
        tree_entries.append(
            {
                "path": f"data/{s_name}",
                "size": dest.stat().st_size,
                "type": "file",
                "sha256": sha256_file(dest),
            }
        )
        for rec in shard_records[idx]:
            metadata_rows.append({"file": rec["file"], "label": rec["speaker_id"], "utt_type": rec["utt_type"]})
            closed_train_rows.append({"file": rec["file"], "label": rec["speaker_id"], "utt_type": rec["utt_type"]})

    metadata_path = root / "vsasv_metadata.csv"
    split_path = root / "closed_train.csv"
    _write_csv(metadata_path, ("file", "label", "utt_type"), metadata_rows)
    _write_csv(split_path, ("file", "label", "utt_type"), closed_train_rows)

    current_manifest_path = root / "development_20k_v2.csv"
    current_manifest_rows = [
        {
            "manifest_version": "development-20k-v2",
            "shard": shard_filename(geometry, 0),
            "file": rec["file"],
            "speaker_id": rec["speaker_id"],
            "split": "closed_train",
            "binary_label": 0 if rec["utt_type"] == "bonafide" else 1,
            "utt_type": rec["utt_type"],
            "native_sample_rate": 16000,
            "source_snapshot": "VSASV-HF-public-snapshot-v1",
        }
        for rec in shard_records[0]
    ]
    _write_csv(current_manifest_path, MANIFEST_COLUMNS, current_manifest_rows)

    audit_json_path = root / "audit_v2.json"
    audit_json_path.write_text(
        json.dumps(
            {
                "row_hashes": [
                    {
                        "split": "closed_train",
                        "file": rec["file"],
                        "binary_label": 0 if rec["utt_type"] == "bonafide" else 1,
                        "waveform_sha256": waveform_sha256(rec["waveform"], 16000),
                    }
                    for rec in shard_records[0]
                ]
            }
        ),
        encoding="utf-8",
    )

    prefix_train_hash_0 = waveform_sha256(shard_records[0][0]["waveform"], 16000)
    prefix_train_hash_1 = waveform_sha256(shard_records[0][1]["waveform"], 16000)
    eval_report_path = root / "closed_eval_full.json"
    eval_report_path.write_text(json.dumps({"ledger_sha256": "unused", "ledger_rows": 4}), encoding="utf-8")
    ledger_path = root / "content_hash_ledger.csv"
    _write_csv(
        ledger_path,
        LEDGER_COLUMNS,
        [
            {"file": "d/0.wav", "split": "closed_dev", "speaker_id": "spk_d", "utt_type": "bonafide", "content_sha256": dev_hash, "decision": "kept"},
            {"file": "d/train_ov.wav", "split": "closed_dev", "speaker_id": "spk_d2", "utt_type": "bonafide", "content_sha256": prefix_train_hash_0, "decision": "train_overlap"},
            {"file": "t/0.wav", "split": "closed_test", "speaker_id": "spk_t", "utt_type": "bonafide", "content_sha256": test_hash, "decision": "kept"},
            {"file": "t/train_ov.wav", "split": "closed_test", "speaker_id": "spk_t2", "utt_type": "voice_conversion", "content_sha256": prefix_train_hash_1, "decision": "train_overlap"},
        ],
    )
    tree_path = root / "vsasv_shard_tree.json"
    tree_path.write_text(json.dumps(tree_entries), encoding="utf-8")

    baseline_hashed_path = root / "baseline_hashed.json"
    baseline_hashed_path.write_text(
        json.dumps({"shard_fingerprint": [{"shard_index": 0}]}),
        encoding="utf-8",
    )

    dev_manifest_path = root / "closed_dev_full_v1.csv"
    _write_csv(
        dev_manifest_path,
        MANIFEST_COLUMNS,
        [
            {
                "manifest_version": "closed-dev-full-v1",
                "shard": "train-00000-of-00004.parquet",
                "file": "d/0.wav",
                "speaker_id": "spk_d",
                "split": "closed_dev",
                "binary_label": 0,
                "utt_type": "bonafide",
                "native_sample_rate": 16000,
                "source_snapshot": "VSASV-HF-public-snapshot-v1",
            }
        ],
    )

    downloaded_calls: list[str] = []

    def fake_downloader(shard_name: str, destination: Path) -> Path:
        downloaded_calls.append(shard_name)
        src = remote_dir / shard_name
        destination.mkdir(parents=True, exist_ok=True)
        dst = destination / shard_name
        dst.write_bytes(src.read_bytes())
        return dst

    return {
        "geometry": geometry,
        "remote_ordered": remote_ordered,
        "data_dir": data_dir,
        "incoming_dir": incoming_dir,
        "cache_dir": cache_dir,
        "metadata_path": metadata_path,
        "split_path": split_path,
        "current_manifest_path": current_manifest_path,
        "audit_json_path": audit_json_path,
        "eval_report_path": eval_report_path,
        "ledger_path": ledger_path,
        "tree_path": tree_path,
        "baseline_hashed_json": baseline_hashed_path,
        "dev_manifest_path": dev_manifest_path,
        "output_manifest_path": root / "development_scale2_v1.csv",
        "download_json_path": root / "scale2_download.json",
        "download_md_path": root / "scale2_download.md",
        "manifest_json_path": root / "development_manifest_scale2_v1.json",
        "manifest_md_path": root / "development_manifest_scale2_v1.md",
        "audit_output_json_path": root / "development_manifest_scale2_v1_content_audit.json",
        "downloader": fake_downloader,
        "downloaded_calls": downloaded_calls,
        "dup_winner": expected_dup_winner,
        "dup_loser": expected_dup_loser,
        "shard_records": shard_records,
        "dev_hash": dev_hash,
        "test_hash": test_hash,
    }


class BuildScale2ManifestTests(unittest.TestCase):
    def test_download_verification_and_adaptive_next_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _create_fixture_env(Path(tmp), make_initial_shard_all_dev_overlap=True)
            result = run_scale2_build(
                parquet_dir=env["data_dir"],
                incoming_dir=env["incoming_dir"],
                metadata_path=env["metadata_path"],
                closed_train_split_path=env["split_path"],
                current_manifest_path=env["current_manifest_path"],
                audit_json_path=env["audit_json_path"],
                eval_report_json_path=env["eval_report_path"],
                ledger_csv_path=env["ledger_path"],
                tree_json_path=env["tree_path"],
                baseline_candidates_csv=None,
                baseline_hashed_json=env["baseline_hashed_json"],
                dev_manifest_path=env["dev_manifest_path"],
                cache_dir=env["cache_dir"],
                output_manifest_path=env["output_manifest_path"],
                download_json_path=env["download_json_path"],
                download_md_path=env["download_md_path"],
                manifest_json_path=env["manifest_json_path"],
                manifest_md_path=env["manifest_md_path"],
                audit_output_json_path=env["audit_output_json_path"],
                geometry=env["geometry"],
                expected_baseline_shards=None,
                min_free_bytes=1,
                verify_ledger_sha=False,
                max_shards_ceiling=10,
                max_bytes_ceiling=10_000_000_000,
                initial_plan_shards=1,
                downloader=env["downloader"],
            )

            # Lô đầu (1 shard) bị dev_overlap ở bonafide nên phải tự tải tiếp shard kế theo thứ tự SHA256("2026|shard")
            r1, r2, _r3 = env["remote_ordered"]
            self.assertEqual(
                env["downloaded_calls"],
                [shard_filename(env["geometry"], r1), shard_filename(env["geometry"], r2)],
            )
            self.assertTrue(result["download_report"]["summary"]["downloaded_beyond_35"])
            self.assertEqual(result["download_report"]["summary"]["adaptive_extra_shards_count"], 1)
            self.assertEqual(result["manifest_report"]["manifest"]["samples"], 8)
            self.assertEqual(result["audit_report"]["checks"]["duplicate_groups_total"], 0)
            self.assertEqual(result["audit_report"]["checks"]["dev_full_hash_overlaps"], 0)
            self.assertEqual(result["audit_report"]["checks"]["test_full_hash_overlaps"], 0)
            self.assertEqual(result["audit_report"]["checks"]["new_rows_all_ledger_dev_overlaps"], 0)
            self.assertEqual(result["audit_report"]["checks"]["new_rows_all_ledger_test_overlaps"], 0)

            # Chạy lại lần 2 phải bỏ qua tải mạng (reused_verified)
            env["downloaded_calls"].clear()
            result2 = run_scale2_build(
                parquet_dir=env["data_dir"],
                incoming_dir=env["incoming_dir"],
                metadata_path=env["metadata_path"],
                closed_train_split_path=env["split_path"],
                current_manifest_path=env["current_manifest_path"],
                audit_json_path=env["audit_json_path"],
                eval_report_json_path=env["eval_report_path"],
                ledger_csv_path=env["ledger_path"],
                tree_json_path=env["tree_path"],
                baseline_candidates_csv=None,
                baseline_hashed_json=env["baseline_hashed_json"],
                dev_manifest_path=env["dev_manifest_path"],
                cache_dir=env["cache_dir"],
                output_manifest_path=env["output_manifest_path"],
                download_json_path=env["download_json_path"],
                download_md_path=env["download_md_path"],
                manifest_json_path=env["manifest_json_path"],
                manifest_md_path=env["manifest_md_path"],
                audit_output_json_path=env["audit_output_json_path"],
                geometry=env["geometry"],
                expected_baseline_shards=None,
                min_free_bytes=1,
                verify_ledger_sha=False,
                max_shards_ceiling=10,
                max_bytes_ceiling=10_000_000_000,
                initial_plan_shards=1,
                downloader=env["downloader"],
            )
            self.assertEqual(env["downloaded_calls"], [])
            self.assertEqual(result2["download_report"]["summary"]["reused_verified_count"], 2)
            self.assertIn("initial_download_and_verify_seconds", result2["manifest_report"]["timing_seconds"])

    def test_stops_at_shard_ceiling_and_does_not_write_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _create_fixture_env(Path(tmp), make_initial_shard_all_dev_overlap=True)
            with self.assertRaises(RuntimeError):
                run_scale2_build(
                    parquet_dir=env["data_dir"],
                    incoming_dir=env["incoming_dir"],
                    metadata_path=env["metadata_path"],
                    closed_train_split_path=env["split_path"],
                    current_manifest_path=env["current_manifest_path"],
                    audit_json_path=env["audit_json_path"],
                    eval_report_json_path=env["eval_report_path"],
                    ledger_csv_path=env["ledger_path"],
                    tree_json_path=env["tree_path"],
                    baseline_candidates_csv=None,
                    baseline_hashed_json=env["baseline_hashed_json"],
                    dev_manifest_path=env["dev_manifest_path"],
                    cache_dir=env["cache_dir"],
                    output_manifest_path=env["output_manifest_path"],
                    download_json_path=env["download_json_path"],
                    download_md_path=env["download_md_path"],
                    manifest_json_path=env["manifest_json_path"],
                    manifest_md_path=env["manifest_md_path"],
                    audit_output_json_path=env["audit_output_json_path"],
                    geometry=env["geometry"],
                    expected_baseline_shards=None,
                    min_free_bytes=1,
                    verify_ledger_sha=False,
                    max_shards_ceiling=1,  # Trần = 1 -> dừng ngay trước khi tải shard thứ 2
                    max_bytes_ceiling=10_000_000_000,
                    initial_plan_shards=1,
                    downloader=env["downloader"],
                )
            self.assertFalse(env["output_manifest_path"].exists())
            dl_report = json.loads(env["download_json_path"].read_text(encoding="utf-8"))
            self.assertEqual(dl_report["status"], "STOPPED_CEILING_EXCEEDED")
            self.assertTrue(dl_report["summary"]["ceiling_exceeded"])

    def test_dedup_smallest_order_hash_and_manifest_prefix_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _create_fixture_env(
                Path(tmp),
                make_initial_shard_all_dev_overlap=True,
                make_internal_dup_pair=True,
            )
            run_scale2_build(
                parquet_dir=env["data_dir"],
                incoming_dir=env["incoming_dir"],
                metadata_path=env["metadata_path"],
                closed_train_split_path=env["split_path"],
                current_manifest_path=env["current_manifest_path"],
                audit_json_path=env["audit_json_path"],
                eval_report_json_path=env["eval_report_path"],
                ledger_csv_path=env["ledger_path"],
                tree_json_path=env["tree_path"],
                baseline_candidates_csv=None,
                baseline_hashed_json=env["baseline_hashed_json"],
                dev_manifest_path=env["dev_manifest_path"],
                cache_dir=env["cache_dir"],
                output_manifest_path=env["output_manifest_path"],
                download_json_path=env["download_json_path"],
                download_md_path=env["download_md_path"],
                manifest_json_path=env["manifest_json_path"],
                manifest_md_path=env["manifest_md_path"],
                audit_output_json_path=env["audit_output_json_path"],
                geometry=env["geometry"],
                expected_baseline_shards=None,
                min_free_bytes=1,
                verify_ledger_sha=False,
                max_shards_ceiling=10,
                max_bytes_ceiling=10_000_000_000,
                initial_plan_shards=2,
                downloader=env["downloader"],
            )
            with env["output_manifest_path"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            # 4 dòng đầu giữ nguyên thứ tự train ×1
            with env["current_manifest_path"].open("r", encoding="utf-8", newline="") as handle:
                old_rows = list(csv.DictReader(handle))
            self.assertEqual([r["file"] for r in rows[:4]], [r["file"] for r in old_rows])
            # Nhóm trùng nội bộ bon_dup_a/bon_dup_b giữ đúng file có SHA256("2026|file") nhỏ hơn
            chosen_files = {r["file"] for r in rows[4:]}
            self.assertIn(env["dup_winner"], chosen_files)
            self.assertNotIn(env["dup_loser"], chosen_files)

    def test_independent_audit_detects_duplicate_and_eval_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry = ShardGeometry(first_count=1, first_rows=4, rest_count=0, rest_rows=0)
            data_dir = root / "data"
            dup_wave = _make_wave(150.0)
            recs = [
                {"file": "spk0/a.wav", "speaker_id": "spk0", "utt_type": "bonafide", "waveform": dup_wave, "sample_rate": 16000},
                {"file": "spk0/b.wav", "speaker_id": "spk0", "utt_type": "voice_conversion", "waveform": dup_wave, "sample_rate": 16000},
                {"file": "spk0/c.wav", "speaker_id": "spk0", "utt_type": "adversarial_attack", "waveform": _make_wave(160.0), "sample_rate": 16000},
                {"file": "spk0/d.wav", "speaker_id": "spk0", "utt_type": "replay", "waveform": _make_wave(170.0), "sample_rate": 16000},
            ]
            s_name = shard_filename(geometry, 0)
            _write_parquet_shard(data_dir / s_name, recs)
            manifest_path = root / "bad_manifest.csv"
            _write_csv(
                manifest_path,
                MANIFEST_COLUMNS,
                [
                    {
                        "manifest_version": SCALE2_MANIFEST_VERSION,
                        "shard": s_name,
                        "file": r["file"],
                        "speaker_id": r["speaker_id"],
                        "split": "closed_train",
                        "binary_label": 0 if r["utt_type"] == "bonafide" else 1,
                        "utt_type": r["utt_type"],
                        "native_sample_rate": 16000,
                        "source_snapshot": "VSASV-HF-public-snapshot-v1",
                    }
                    for r in recs
                ],
            )
            split_map = {r["file"]: (r["speaker_id"], r["utt_type"]) for r in recs}
            meta_map = {r["file"]: (i, r["speaker_id"], r["utt_type"]) for i, r in enumerate(recs)}
            prefix = [
                {
                    "shard": s_name,
                    "file": recs[0]["file"],
                    "speaker_id": recs[0]["speaker_id"],
                    "split": "closed_train",
                    "binary_label": "0",
                    "utt_type": recs[0]["utt_type"],
                    "native_sample_rate": "16000",
                }
            ]
            quota = {u: 1 for u in UTT_TYPES}
            # 1) Phát hiện trùng lặp nội bộ
            with self.assertRaisesRegex(ValueError, "nhóm waveform_sha256 lặp"):
                audit_scale2_manifest_independently(
                    manifest_path=manifest_path,
                    parquet_dir=data_dir,
                    expected_prefix_rows=prefix,
                    expected_train_hashes_by_file={},
                    dev_hashes=set(),
                    test_hashes=set(),
                    closed_train_split_rows=split_map,
                    metadata_by_file=meta_map,
                    dev_speakers=set(),
                    expected_quota=quota,
                )

            # 2) Sửa b.wav thành duy nhất, sau đó kiểm tra phát hiện trùng dev-full / test-full
            recs[1]["waveform"] = _make_wave(180.0)
            _write_parquet_shard(data_dir / s_name, recs)
            c_hash = waveform_sha256(recs[2]["waveform"], 16000)
            with self.assertRaisesRegex(ValueError, "dev-full"):
                audit_scale2_manifest_independently(
                    manifest_path=manifest_path,
                    parquet_dir=data_dir,
                    expected_prefix_rows=prefix,
                    expected_train_hashes_by_file={},
                    dev_hashes={c_hash},
                    test_hashes=set(),
                    closed_train_split_rows=split_map,
                    metadata_by_file=meta_map,
                    dev_speakers=set(),
                    expected_quota=quota,
                )
            with self.assertRaisesRegex(ValueError, "test-full"):
                audit_scale2_manifest_independently(
                    manifest_path=manifest_path,
                    parquet_dir=data_dir,
                    expected_prefix_rows=prefix,
                    expected_train_hashes_by_file={},
                    dev_hashes=set(),
                    test_hashes={c_hash},
                    closed_train_split_rows=split_map,
                    metadata_by_file=meta_map,
                    dev_speakers=set(),
                    expected_quota=quota,
                )
            with self.assertRaisesRegex(ValueError, "ledger closed_dev"):
                audit_scale2_manifest_independently(
                    manifest_path=manifest_path,
                    parquet_dir=data_dir,
                    expected_prefix_rows=prefix,
                    expected_train_hashes_by_file={},
                    dev_hashes=set(),
                    test_hashes=set(),
                    all_ledger_dev_hashes={c_hash},
                    all_ledger_test_hashes=set(),
                    closed_train_split_rows=split_map,
                    metadata_by_file=meta_map,
                    dev_speakers=set(),
                    expected_quota=quota,
                )
            with self.assertRaisesRegex(ValueError, "ledger closed_test"):
                audit_scale2_manifest_independently(
                    manifest_path=manifest_path,
                    parquet_dir=data_dir,
                    expected_prefix_rows=prefix,
                    expected_train_hashes_by_file={},
                    dev_hashes=set(),
                    test_hashes=set(),
                    all_ledger_dev_hashes=set(),
                    all_ledger_test_hashes={c_hash},
                    closed_train_split_rows=split_map,
                    metadata_by_file=meta_map,
                    dev_speakers=set(),
                    expected_quota=quota,
                )

    def test_download_auth_error_aborts_immediately_without_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = 0

            def auth_fail_downloader(_shard_name: str, _dest: Path) -> Path:
                nonlocal calls
                calls += 1
                raise RuntimeError("401 Unauthorized: Cannot access gated repo. Please log in.")

            with self.assertRaises(PermissionError):
                download_and_verify_shard(
                    order_position=1,
                    shard_index=0,
                    shard_name="train-00000-of-00001.parquet",
                    entry=TreeEntry(size=100, sha256="0" * 64),
                    expected_row_count=1,
                    expected_metadata_rows={"a.wav": ("spk0", "bonafide")},
                    data_dir=root / "data",
                    incoming_dir=root / "incoming",
                    downloader=auth_fail_downloader,
                    download_log_path=root / "dl.jsonl",
                    max_attempts=3,
                )
            self.assertEqual(calls, 1)

            env = _create_fixture_env(root / "env_auth", make_initial_shard_all_dev_overlap=True)
            env_calls = 0

            def env_auth_fail_downloader(_shard_name: str, _dest: Path) -> Path:
                nonlocal env_calls
                env_calls += 1
                raise RuntimeError("401 Client Error: Cannot access gated repo. Please log in.")

            with self.assertRaises(PermissionError):
                run_scale2_build(
                    parquet_dir=env["data_dir"],
                    incoming_dir=env["incoming_dir"],
                    metadata_path=env["metadata_path"],
                    closed_train_split_path=env["split_path"],
                    current_manifest_path=env["current_manifest_path"],
                    audit_json_path=env["audit_json_path"],
                    eval_report_json_path=env["eval_report_path"],
                    ledger_csv_path=env["ledger_path"],
                    tree_json_path=env["tree_path"],
                    baseline_candidates_csv=None,
                    baseline_hashed_json=env["baseline_hashed_json"],
                    dev_manifest_path=env["dev_manifest_path"],
                    cache_dir=env["cache_dir"],
                    output_manifest_path=env["output_manifest_path"],
                    download_json_path=env["download_json_path"],
                    download_md_path=env["download_md_path"],
                    manifest_json_path=env["manifest_json_path"],
                    manifest_md_path=env["manifest_md_path"],
                    audit_output_json_path=env["audit_output_json_path"],
                    geometry=env["geometry"],
                    expected_baseline_shards=None,
                    min_free_bytes=1,
                    verify_ledger_sha=False,
                    max_shards_ceiling=10,
                    max_bytes_ceiling=10_000_000_000,
                    initial_plan_shards=1,
                    downloader=env_auth_fail_downloader,
                )
            self.assertEqual(env_calls, 1)
            self.assertFalse(env["output_manifest_path"].exists())
            dl_report = json.loads(env["download_json_path"].read_text(encoding="utf-8"))
            self.assertEqual(dl_report["status"], "STOPPED_HF_AUTH_REQUIRED")
            self.assertEqual(dl_report["summary"]["total_shards_acquired"], 0)
            self.assertEqual(len(dl_report["failed_events"]), 1)
            self.assertEqual(dl_report["failed_events"][0]["status"], "failed_auth")

    def test_downloader_omits_token_argument_for_default_auth(self) -> None:
        captured: dict[str, Any] = {}

        def fake_hub_download(*args: Any, **kwargs: Any) -> str:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return str(Path(kwargs["local_dir"]) / "data" / "train-00000-of-00432.parquet")

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "incoming"
            with patch("huggingface_hub.hf_hub_download", side_effect=fake_hub_download):
                out = public_hf_downloader("train-00000-of-00432.parquet", dest)

        self.assertEqual(captured["args"], (REPOSITORY, "data/train-00000-of-00432.parquet"))
        self.assertNotIn("token", captured["kwargs"])
        self.assertEqual(captured["kwargs"]["repo_type"], "dataset")
        self.assertEqual(captured["kwargs"]["revision"], REVISION)
        self.assertEqual(captured["kwargs"]["local_dir"], str(dest))
        self.assertEqual(out.name, "train-00000-of-00432.parquet")


if __name__ == "__main__":
    unittest.main()
