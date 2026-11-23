"""Kiểm thử lập kế hoạch shard và khử trùng cho scale2 (D036)."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_evaluation_shards import TreeEntry  # noqa: E402
from scripts.make_development_manifest import MANIFEST_COLUMNS  # noqa: E402
from scripts.plan_evaluation_shards import ShardGeometry, shard_filename  # noqa: E402
from scripts.plan_scale2_shards import (  # noqa: E402
    LEDGER_COLUMNS,
    MAX_BYTES_CEILING_GB,
    MAX_SHARDS_CEILING,
    deduplicate_and_select_candidates,
    load_and_verify_hash_sources,
    order_digest,
    plan_remote_shards,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class PlanScale2ShardsTests(unittest.TestCase):
    def test_order_digest_matches_d036_definition(self) -> None:
        self.assertEqual(
            order_digest("id00001/00001.wav", 2026),
            hashlib.sha256(b"2026|id00001/00001.wav").hexdigest(),
        )
        self.assertEqual(
            order_digest("train-00012-of-00432.parquet", 2026),
            hashlib.sha256(b"2026|train-00012-of-00432.parquet").hexdigest(),
        )
        shards = [
            shard_filename(ShardGeometry(3, 2, 2, 1), idx) for idx in range(5)
        ]
        ordered = sorted(shards, key=lambda name: (order_digest(name, 2026), name))
        self.assertEqual(
            ordered,
            sorted(
                shards,
                key=lambda name: (
                    hashlib.sha256(f"2026|{name}".encode("utf-8")).hexdigest(),
                    name,
                ),
            ),
        )

    def test_deduplication_filters_splits_and_keeps_smallest_file_hash(self) -> None:
        # Tạo 3 tên file có nội dung giống hệt nhau để kiểm tra giữ file có SHA256("2026|file") nhỏ nhất
        dup_files = ["spk/dup_a.wav", "spk/dup_b.wav", "spk/dup_c.wav"]
        expected_rep = min(dup_files, key=lambda name: (order_digest(name, 2026), name))
        dup_hash = _sha256_bytes(b"internal-dup")
        train_hash = _sha256_bytes(b"overlap-train")
        dev_hash = _sha256_bytes(b"overlap-dev")
        test_hash = _sha256_bytes(b"overlap-test")
        vc_clean_1 = _sha256_bytes(b"vc-1")
        vc_clean_2 = _sha256_bytes(b"vc-2")
        ap_clean = _sha256_bytes(b"ap-1")
        rep_clean_1 = _sha256_bytes(b"rep-1")
        rep_clean_2 = _sha256_bytes(b"rep-2")

        candidates = [
            {
                "file": name,
                "speaker_id": "spk",
                "utt_type": "bonafide",
                "binary_label": 0,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 16000,
                "content_sha256": dup_hash,
            }
            for name in dup_files
        ] + [
            {
                "file": "spk/tr_ov.wav",
                "speaker_id": "spk",
                "utt_type": "bonafide",
                "binary_label": 0,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 16000,
                "content_sha256": train_hash,
            },
            {
                "file": "spk/dev_ov.wav",
                "speaker_id": "spk",
                "utt_type": "bonafide",
                "binary_label": 0,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 16000,
                "content_sha256": dev_hash,
            },
            {
                "file": "spk/test_ov.wav",
                "speaker_id": "spk",
                "utt_type": "bonafide",
                "binary_label": 0,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 16000,
                "content_sha256": test_hash,
            },
            {
                "file": "spk/vc1.wav",
                "speaker_id": "spk",
                "utt_type": "voice_conversion",
                "binary_label": 1,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 40000,
                "content_sha256": vc_clean_1,
            },
            {
                "file": "spk/vc2.wav",
                "speaker_id": "spk",
                "utt_type": "voice_conversion",
                "binary_label": 1,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 40000,
                "content_sha256": vc_clean_2,
            },
            {
                "file": "spk/ap1.wav",
                "speaker_id": "spk",
                "utt_type": "adversarial_attack",
                "binary_label": 1,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 16000,
                "content_sha256": ap_clean,
            },
            {
                "file": "spk/rep1.wav",
                "speaker_id": "spk",
                "utt_type": "replay",
                "binary_label": 1,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 16000,
                "content_sha256": rep_clean_1,
            },
            {
                "file": "spk/rep2.wav",
                "speaker_id": "spk",
                "utt_type": "replay",
                "binary_label": 1,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 16000,
                "content_sha256": rep_clean_2,
            },
        ]

        result = deduplicate_and_select_candidates(
            candidates,
            train_hashes={train_hash: 0},
            dev_hashes={dev_hash},
            test_hashes={test_hash},
            additional_needed={
                "bonafide": 3,
                "voice_conversion": 2,
                "adversarial_attack": 2,
                "replay": 1,
            },
            seed=2026,
        )

        self.assertEqual(
            result["clean_counts_by_utt_type"],
            {
                "bonafide": 1,
                "voice_conversion": 2,
                "adversarial_attack": 1,
                "replay": 2,
            },
        )
        self.assertEqual(
            result["shortfalls_by_utt_type"],
            {
                "bonafide": 2,
                "voice_conversion": 0,
                "adversarial_attack": 1,
                "replay": 0,
            },
        )
        self.assertEqual(result["surplus_local_by_utt_type"]["replay"], 1)
        self.assertEqual(result["dropped_by_primary_reason"]["train_overlap"]["bonafide"], 1)
        self.assertEqual(result["dropped_by_primary_reason"]["dev_overlap"]["bonafide"], 1)
        self.assertEqual(result["dropped_by_primary_reason"]["test_overlap"]["bonafide"], 1)
        self.assertEqual(result["dropped_by_primary_reason"]["internal_duplicate"]["bonafide"], 2)

        row_map = {r["file"]: r for r in result["candidate_rows"]}
        self.assertEqual(row_map[expected_rep]["selection_status"], "selected_local")
        for other in dup_files:
            if other != expected_rep:
                self.assertEqual(row_map[other]["drop_reason"], "internal_duplicate")
                self.assertEqual(row_map[other]["kept_representative_file"], expected_rep)

        # Kiểm tra từ chối nếu content_sha256 trộn nhãn nhị phân
        bad_candidates = list(candidates) + [
            {
                "file": "spk/mixed.wav",
                "speaker_id": "spk",
                "utt_type": "voice_conversion",
                "binary_label": 1,
                "shard": "train-00000-of-00006.parquet",
                "native_sample_rate": 40000,
                "content_sha256": train_hash,  # train_hash mang nhãn 0
            }
        ]
        with self.assertRaises(ValueError):
            deduplicate_and_select_candidates(
                bad_candidates,
                train_hashes={train_hash: 0},
                dev_hashes=set(),
                test_hashes=set(),
                additional_needed={
                    "bonafide": 1,
                    "voice_conversion": 1,
                    "adversarial_attack": 1,
                    "replay": 1,
                },
            )

    def test_remote_shard_plan_stops_at_first_satisfying_shard_and_reserve(self) -> None:
        geometry = ShardGeometry(first_count=6, first_rows=10, rest_count=0, rest_rows=0)
        tree = {
            idx: TreeEntry(size=1_000 * (idx + 1), sha256=f"{idx:064x}")
            for idx in range(6)
        }
        local_indices = {0}
        ordered_missing = sorted(
            range(1, 6),
            key=lambda idx: (
                order_digest(shard_filename(geometry, idx), 2026),
                shard_filename(geometry, idx),
            ),
        )
        # Tạo số lượng mẫu closed_train sao cho:
        # - shard thứ 1, 2 đạt bonafide=2, ap=1; chưa đủ ap=2
        # - shard thứ 3 đạt đủ bonafide=3, ap=2 (cơ sở dừng tại #3)
        # - shard thứ 4 cần thêm cho mức dự phòng (ví dụ bonafide target = 4)
        closed_train_by_shard = {
            ordered_missing[0]: {"bonafide": 1, "voice_conversion": 0, "adversarial_attack": 1, "replay": 0},
            ordered_missing[1]: {"bonafide": 2, "voice_conversion": 0, "adversarial_attack": 0, "replay": 0},
            ordered_missing[2]: {"bonafide": 0, "voice_conversion": 1, "adversarial_attack": 1, "replay": 0},
            ordered_missing[3]: {"bonafide": 2, "voice_conversion": 0, "adversarial_attack": 1, "replay": 0},
            ordered_missing[4]: {"bonafide": 5, "voice_conversion": 5, "adversarial_attack": 5, "replay": 1},
        }
        plan = plan_remote_shards(
            geometry=geometry,
            tree=tree,
            local_shard_indices=local_indices,
            closed_train_unselected_by_shard=closed_train_by_shard,
            shortfalls={
                "bonafide": 3,
                "voice_conversion": 0,
                "adversarial_attack": 2,
                "replay": 0,
            },
            observed_drop_rate_by_utt={
                "bonafide": 0.25,  # ceil(3 / 0.75) = 4 -> cần thêm shard thứ 4
                "voice_conversion": 0.0,
                "adversarial_attack": 0.0,
                "replay": 0.0,
            },
            observed_drop_rate_overall=0.20,  # bonafide=4, ap=ceil(2/0.8)=3 -> cần đến shard thứ 4
            seed=2026,
            max_shards=120,
            max_bytes=10_000_000_000,
        )

        base = plan["base_plan_pre_dedup"]
        self.assertTrue(base["satisfied"])
        self.assertEqual(base["shard_count"], 3)
        self.assertEqual(
            [item["shard_index"] for item in base["shards"]],
            ordered_missing[:3],
        )
        self.assertFalse(plan["any_exceeds_ceiling"])

        res_utt = plan["reserve_estimate_by_utt_type_rate"]["plan"]
        self.assertEqual(res_utt["targets_by_utt_type"]["bonafide"], 4)
        self.assertEqual(res_utt["shard_count"], 4)

        res_all = plan["reserve_estimate_overall_rate"]["plan"]
        self.assertEqual(res_all["targets_by_utt_type"]["adversarial_attack"], 3)
        self.assertEqual(res_all["shard_count"], 4)

    def test_ceiling_warning_triggered_on_shard_and_byte_limits(self) -> None:
        # 1) Kiểm tra vượt trần 120 shard
        geometry = ShardGeometry(first_count=125, first_rows=1, rest_count=0, rest_rows=0)
        tree = {
            idx: TreeEntry(size=1024, sha256=f"{idx:064x}") for idx in range(125)
        }
        by_shard = {
            idx: {"bonafide": 1, "voice_conversion": 0, "adversarial_attack": 0, "replay": 0}
            for idx in range(125)
        }
        plan_many = plan_remote_shards(
            geometry=geometry,
            tree=tree,
            local_shard_indices=set(),
            closed_train_unselected_by_shard=by_shard,
            shortfalls={
                "bonafide": 122,
                "voice_conversion": 0,
                "adversarial_attack": 0,
                "replay": 0,
            },
            observed_drop_rate_by_utt={u: 0.0 for u in ("bonafide", "voice_conversion", "adversarial_attack", "replay")},
            observed_drop_rate_overall=0.0,
            max_shards=MAX_SHARDS_CEILING,
            max_bytes=MAX_BYTES_CEILING_GB,
        )
        self.assertTrue(plan_many["base_plan_pre_dedup"]["exceeds_shard_ceiling"])
        self.assertTrue(plan_many["base_plan_pre_dedup"]["exceeds_ceiling"])
        self.assertTrue(plan_many["any_exceeds_ceiling"])

        # 2) Kiểm tra vượt trần 10 GB dù số shard <= 120
        geometry_big = ShardGeometry(first_count=5, first_rows=1, rest_count=0, rest_rows=0)
        tree_big = {
            idx: TreeEntry(size=3_500_000_000, sha256=f"{idx:064x}") for idx in range(5)
        }
        by_shard_big = {
            idx: {"bonafide": 1, "voice_conversion": 0, "adversarial_attack": 0, "replay": 0}
            for idx in range(5)
        }
        plan_big = plan_remote_shards(
            geometry=geometry_big,
            tree=tree_big,
            local_shard_indices=set(),
            closed_train_unselected_by_shard=by_shard_big,
            shortfalls={
                "bonafide": 3,
                "voice_conversion": 0,
                "adversarial_attack": 0,
                "replay": 0,
            },
            observed_drop_rate_by_utt={u: 0.0 for u in ("bonafide", "voice_conversion", "adversarial_attack", "replay")},
            observed_drop_rate_overall=0.0,
            max_shards=MAX_SHARDS_CEILING,
            max_bytes=MAX_BYTES_CEILING_GB,
        )
        self.assertFalse(plan_big["base_plan_pre_dedup"]["exceeds_shard_ceiling"])
        self.assertTrue(plan_big["base_plan_pre_dedup"]["exceeds_bytes_ceiling_gb"])
        self.assertTrue(plan_big["any_exceeds_ceiling"])

    def test_no_closed_test_metadata_or_network_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "dev20k.csv"
            audit_path = root / "audit.json"
            eval_report_path = root / "eval_full.json"
            ledger_a = root / "ledger_a.csv"
            ledger_b = root / "ledger_b.csv"

            train_hash = _sha256_bytes(b"tr-0")
            dev_hash = _sha256_bytes(b"dev-0")
            test_hash = _sha256_bytes(b"test-0")

            _write_csv(
                manifest_path,
                MANIFEST_COLUMNS,
                [
                    {
                        "manifest_version": "development-20k-v2",
                        "shard": "train-00000-of-00004.parquet",
                        "file": "tr/0.wav",
                        "speaker_id": "tr",
                        "split": "closed_train",
                        "binary_label": 0,
                        "utt_type": "bonafide",
                        "native_sample_rate": 16000,
                        "source_snapshot": "VSASV-HF-public-snapshot-v1",
                    }
                ],
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "row_hashes": [
                            {
                                "split": "closed_train",
                                "file": "tr/0.wav",
                                "binary_label": 0,
                                "waveform_sha256": train_hash,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            eval_report_path.write_text(
                json.dumps({"ledger_sha256": "unused", "ledger_rows": 2}),
                encoding="utf-8",
            )

            # Hai ledger giống nhau về content_sha256 của closed_test nhưng khác speaker_id/utt_type giả
            _write_csv(
                ledger_a,
                LEDGER_COLUMNS,
                [
                    {
                        "file": "dev/0.wav",
                        "split": "closed_dev",
                        "speaker_id": "d0",
                        "utt_type": "bonafide",
                        "content_sha256": dev_hash,
                        "decision": "kept",
                    },
                    {
                        "file": "test/0.wav",
                        "split": "closed_test",
                        "speaker_id": "t_spk_A",
                        "utt_type": "bonafide",
                        "content_sha256": test_hash,
                        "decision": "kept",
                    },
                ],
            )
            _write_csv(
                ledger_b,
                LEDGER_COLUMNS,
                [
                    {
                        "file": "dev/0.wav",
                        "split": "closed_dev",
                        "speaker_id": "d0",
                        "utt_type": "bonafide",
                        "content_sha256": dev_hash,
                        "decision": "kept",
                    },
                    {
                        "file": "test/renamed.wav",
                        "split": "closed_test",
                        "speaker_id": "t_spk_B",
                        "utt_type": "replay",
                        "content_sha256": test_hash,
                        "decision": "internal_duplicate",
                    },
                ],
            )

            src_a = load_and_verify_hash_sources(
                audit_json_path=audit_path,
                eval_report_json_path=eval_report_path,
                ledger_csv_path=ledger_a,
                current_manifest_csv_path=manifest_path,
                verify_ledger_sha=False,
            )
            src_b = load_and_verify_hash_sources(
                audit_json_path=audit_path,
                eval_report_json_path=eval_report_path,
                ledger_csv_path=ledger_b,
                current_manifest_csv_path=manifest_path,
                verify_ledger_sha=False,
            )
            # Chỉ content_sha256 của closed_test được trích ra (kể cả dòng không kept)
            self.assertEqual(src_a["test_hashes"], {test_hash})
            self.assertEqual(src_b["test_hashes"], {test_hash})
            self.assertEqual(src_a["dev_hashes"], {dev_hash})

        # Đảm bảo mã nguồn script không import thư viện mạng và không trỏ vào closed_test.csv
        script_path = PROJECT_ROOT / "scripts" / "plan_scale2_shards.py"
        script_text = script_path.read_text(encoding="utf-8")
        tree_ast = ast.parse(script_text)
        forbidden_modules = {"urllib", "requests", "httpx", "http", "socket", "huggingface_hub", "kaggle"}
        imported: set[str] = set()
        for node in ast.walk(tree_ast):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported.isdisjoint(forbidden_modules), f"Import mạng bị cấm: {imported & forbidden_modules}")
        self.assertNotIn("closed_test.csv", script_text)
        self.assertNotIn("closed_test_full_v1.csv", script_text)


if __name__ == "__main__":
    unittest.main()
