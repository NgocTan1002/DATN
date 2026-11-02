#!/usr/bin/env python3
"""Tạo evaluation manifest content-disjoint cho dev-full và test-full theo D025.

Đầu vào là sổ hash do `src.data.eval_ingest` ghi lúc nạp shard (không đọc lại audio),
hash của train manifest từ báo cáo audit D017 và các split gốc `closed_dev`/`closed_test`.
Script không đọc score hay metric và không bù mẫu. Nếu gặp điều kiện dừng của D025 thì
chỉ ghi báo cáo, không ghi manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_development_manifest import stable_key  # noqa: E402
from scripts.plan_evaluation_shards import (  # noqa: E402
    OFFICIAL_GEOMETRY,
    ShardGeometry,
    shard_filename,
)
from src.data.eval_ingest import chunk_paths  # noqa: E402
from src.data.vsasv import MANIFEST_COLUMNS  # noqa: E402


SOURCE_SNAPSHOT = "VSASV-HF-public-snapshot-v1"
EVAL_SPLITS = ("closed_dev", "closed_test")
MANIFEST_VERSIONS = {
    "closed_dev": "closed-dev-full-v1",
    "closed_test": "closed-test-full-v1",
}
DEFAULT_SEED = 2026
DEFAULT_MAX_LOSS_FRACTION = 0.05
DEFAULT_CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "eval_chunks"
DEFAULT_PLAN = PROJECT_ROOT / "reports" / "evaluation_shard_plan.json"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "development_manifest_20k_v2_content_audit.json"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_LEDGER = DEFAULT_CHUNK_DIR / "content_hash_ledger.csv"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "closed_eval_full_v1.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "closed_eval_full_v1.md"
LEDGER_COLUMNS = ("file", "split", "speaker_id", "utt_type", "content_sha256", "decision")


@dataclass(frozen=True)
class EvalRecord:
    """Một mẫu dev/test đã nạp cùng hash nội dung D017."""

    file: str
    split: str
    speaker_id: str
    utt_type: str
    binary_label: int
    native_sample_rate: int
    content_sha256: str
    shard: str


@dataclass(frozen=True)
class AuditHashes:
    """Hash từ báo cáo audit D017 của development-20k-v2."""

    train_hashes: dict[str, int]
    train_speakers: set[str]
    dev_hashes: dict[str, str]


@dataclass
class ManifestBuild:
    status: str
    stop_reasons: list[str]
    kept: dict[str, list[EvalRecord]]
    ledger: list[dict[str, str]]
    stats: dict[str, Any]


def load_chunk_records(
    chunk_dir: Path, shard_indices: Sequence[int], geometry: ShardGeometry
) -> list[EvalRecord]:
    """Đọc chỉ mục của các chunk đã hoàn tất, đối chiếu với done marker.

    Chỉ mục phải đúng checksum, số dòng và shard ghi trong marker; nhãn nhị phân
    phải khớp `utt_type`. Băm lại waveform là việc của `verify_chunk`.
    """

    records: list[EvalRecord] = []
    for index in shard_indices:
        paths = chunk_paths(chunk_dir, index)
        if not paths.done.is_file() or not paths.index.is_file():
            raise ValueError(f"Chunk shard {index} chưa hoàn tất; thiếu done marker.")
        shard = shard_filename(geometry, index)
        marker = json.loads(paths.done.read_text(encoding="utf-8"))
        if marker.get("shard_index") != index or marker.get("shard") != shard:
            raise ValueError(f"Done marker của chunk {index} ghi shard khác.")
        if _sha256_file(paths.index) != marker.get("index_sha256"):
            raise ValueError(f"Chỉ mục chunk {index} sai checksum so với done marker.")
        with paths.index.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != marker.get("records"):
            raise ValueError(
                f"Chỉ mục chunk {index} có {len(rows)} dòng, done marker ghi "
                f"{marker.get('records')} số dòng."
            )
        for row in rows:
            expected_label = 0 if row["utt_type"] == "bonafide" else 1
            if int(row["binary_label"]) != expected_label:
                raise ValueError(
                    f"Nhãn nhị phân của {row['file']} không khớp utt_type {row['utt_type']}."
                )
            records.append(
                EvalRecord(
                    file=row["file"],
                    split=row["split"],
                    speaker_id=row["speaker_id"],
                    utt_type=row["utt_type"],
                    binary_label=int(row["binary_label"]),
                    native_sample_rate=int(row["native_sample_rate"]),
                    content_sha256=row["content_sha256"],
                    shard=shard,
                )
            )
    return records


def load_audit(path: Path) -> AuditHashes:
    rows = json.loads(path.read_text(encoding="utf-8"))["row_hashes"]
    train_hashes: dict[str, int] = {}
    train_speakers: set[str] = set()
    dev_hashes: dict[str, str] = {}
    for row in rows:
        if row["split"] == "closed_train":
            train_hashes[str(row["waveform_sha256"])] = int(row["binary_label"])
            train_speakers.add(str(row["speaker_id"]))
        elif row["split"] == "closed_dev":
            dev_hashes[str(row["file"])] = str(row["waveform_sha256"])
    return AuditHashes(train_hashes, train_speakers, dev_hashes)


def load_split_rows(splits_dir: Path) -> dict[str, dict[str, tuple[str, str]]]:
    result: dict[str, dict[str, tuple[str, str]]] = {}
    for split in EVAL_SPLITS:
        with (splits_dir / f"{split}.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            result[split] = {
                row["file"]: (row["label"], row["utt_type"])
                for row in csv.DictReader(handle)
            }
    return result


def _by_utt(records: Sequence[EvalRecord]) -> dict[str, int]:
    return dict(sorted(Counter(item.utt_type for item in records).items()))


def _speakers_by_utt(records: Sequence[EvalRecord]) -> dict[str, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    for item in records:
        groups[item.utt_type].add(item.speaker_id)
    return {key: len(groups[key]) for key in sorted(groups)}


def build_manifests(
    *,
    records: Sequence[EvalRecord],
    split_rows: Mapping[str, Mapping[str, tuple[str, str]]],
    train_hashes: Mapping[str, int],
    train_speakers: set[str],
    audit_dev_hashes: Mapping[str, str],
    seed: int = DEFAULT_SEED,
    max_loss_fraction: float = DEFAULT_MAX_LOSS_FRACTION,
) -> ManifestBuild:
    """Áp quy tắc khử trùng D025; dừng thay vì tự điều chỉnh khi vượt ngưỡng."""

    by_split: dict[str, list[EvalRecord]] = {split: [] for split in EVAL_SPLITS}
    seen_files: set[str] = set()
    for item in records:
        if item.split not in by_split:
            raise ValueError(f"Split {item.split} không thuộc dev/test cần đánh giá.")
        if item.file in seen_files:
            raise ValueError(f"Sổ hash lặp file {item.file}.")
        seen_files.add(item.file)
        by_split[item.split].append(item)

    for split in EVAL_SPLITS:
        expected = split_rows[split]
        present = {item.file for item in by_split[split]}
        missing = sorted(set(expected) - present)
        extra = sorted(present - set(expected))
        if missing:
            raise ValueError(
                f"{split} thiếu {len(missing)} mẫu trong sổ hash, ví dụ {missing[0]}."
            )
        if extra:
            raise ValueError(
                f"{split} thừa {len(extra)} mẫu ngoài split gốc, ví dụ {extra[0]}."
            )
        for item in by_split[split]:
            if (item.speaker_id, item.utt_type) != expected[item.file]:
                raise ValueError(
                    f"Metadata sổ hash không khớp split gốc cho {item.file}."
                )

    speakers = {split: {item.speaker_id for item in by_split[split]} for split in EVAL_SPLITS}
    pairs = (
        ("closed_dev", "closed_test", speakers["closed_dev"] & speakers["closed_test"]),
        ("train manifest", "closed_dev", train_speakers & speakers["closed_dev"]),
        ("train manifest", "closed_test", train_speakers & speakers["closed_test"]),
    )
    for left, right, overlap in pairs:
        if overlap:
            raise ValueError(
                f"Vi phạm speaker-disjoint giữa {left} và {right}: {sorted(overlap)[0]}."
            )

    dev_by_file = {item.file: item for item in by_split["closed_dev"]}
    mismatched = [
        name
        for name, digest in audit_dev_hashes.items()
        if name not in dev_by_file or dev_by_file[name].content_sha256 != digest
    ]
    if mismatched:
        raise ValueError(
            f"Hash không khớp báo cáo audit D017 cho {len(mismatched)} file dev, "
            f"ví dụ {sorted(mismatched)[0]}."
        )

    stop_reasons: list[str] = []
    labels_by_hash: dict[str, set[int]] = defaultdict(set)
    for digest, label in train_hashes.items():
        labels_by_hash[digest].add(label)
    for item in records:
        labels_by_hash[item.content_sha256].add(item.binary_label)
    mixed = sorted(digest for digest, labels in labels_by_hash.items() if len(labels) > 1)
    if mixed:
        stop_reasons.append(
            f"{len(mixed)} nhóm hash trộn hai nhãn nhị phân, ví dụ {mixed[0]}."
        )

    dev_hash_set = {item.content_sha256 for item in by_split["closed_dev"]}
    kept: dict[str, list[EvalRecord]] = {}
    ledger: list[dict[str, str]] = []
    dropped_counts: dict[str, dict[str, Counter[str]]] = {
        split: defaultdict(Counter) for split in EVAL_SPLITS
    }
    for split in EVAL_SPLITS:
        survivors: list[EvalRecord] = []
        decisions: dict[str, str] = {}
        for item in by_split[split]:
            if item.content_sha256 in train_hashes:
                decisions[item.file] = "train_overlap"
            elif split == "closed_test" and item.content_sha256 in dev_hash_set:
                decisions[item.file] = "dev_overlap"
            else:
                survivors.append(item)
        groups: dict[str, list[EvalRecord]] = defaultdict(list)
        for item in survivors:
            groups[item.content_sha256].append(item)
        kept_files: set[str] = set()
        for members in groups.values():
            ordered = sorted(members, key=lambda row: stable_key(row.file, seed))
            kept_files.add(ordered[0].file)
            for removed in ordered[1:]:
                decisions[removed.file] = "internal_duplicate"
        kept[split] = sorted(
            (item for item in by_split[split] if item.file in kept_files),
            key=lambda row: row.file,
        )
        for item in by_split[split]:
            decision = "kept" if item.file in kept_files else decisions[item.file]
            if decision != "kept":
                dropped_counts[split][decision][item.utt_type] += 1
            ledger.append(
                {
                    "file": item.file,
                    "split": split,
                    "speaker_id": item.speaker_id,
                    "utt_type": item.utt_type,
                    "content_sha256": item.content_sha256,
                    "decision": decision,
                }
            )

    stats: dict[str, Any] = {}
    for split in EVAL_SPLITS:
        original = _by_utt(by_split[split])
        kept_by_utt = _by_utt(kept[split])
        loss: dict[str, float] = {}
        for utt_type, count in original.items():
            lost = count - kept_by_utt.get(utt_type, 0)
            loss[utt_type] = lost / count
            if lost / count > max_loss_fraction:
                stop_reasons.append(
                    f"{split}/{utt_type} mất {lost}/{count} mẫu "
                    f"({100 * lost / count:.2f}%), vượt ngưỡng "
                    f"{100 * max_loss_fraction:.2f}%."
                )
        stats[split] = {
            "original_rows": len(by_split[split]),
            "original_by_utt_type": original,
            "kept_rows": len(kept[split]),
            "kept_by_utt_type": kept_by_utt,
            "dropped": {
                reason: dict(sorted(counter.items()))
                for reason, counter in sorted(dropped_counts[split].items())
            },
            "loss_fraction_by_utt_type": loss,
            "speakers_original_by_utt_type": _speakers_by_utt(by_split[split]),
            "speakers_kept_by_utt_type": _speakers_by_utt(kept[split]),
        }
    kept_dev_hashes = {item.content_sha256 for item in kept["closed_dev"]}
    kept_test_hashes = {item.content_sha256 for item in kept["closed_test"]}
    stats["checks"] = {
        "unique_hashes_dev": len(kept_dev_hashes) == len(kept["closed_dev"]),
        "unique_hashes_test": len(kept_test_hashes) == len(kept["closed_test"]),
        "dev_test_hash_overlap": len(kept_dev_hashes & kept_test_hashes),
        "train_dev_hash_overlap": len(kept_dev_hashes & set(train_hashes)),
        "train_test_hash_overlap": len(kept_test_hashes & set(train_hashes)),
        "speaker_disjoint": True,
    }
    stats["audit_crosscheck"] = {"compared": len(audit_dev_hashes), "mismatched": 0}
    return ManifestBuild(
        status="DỪNG" if stop_reasons else "ĐẠT",
        stop_reasons=stop_reasons,
        kept=kept,
        ledger=ledger,
        stats=stats,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, records: Sequence[EvalRecord], version: str) -> str:
    """Ghi manifest theo schema v2, sắp xếp theo file; không ghi đè file có sẵn."""

    if path.exists():
        raise FileExistsError(f"Manifest đã tồn tại; không ghi đè: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in sorted(records, key=lambda row: row.file):
            writer.writerow(
                {
                    "manifest_version": version,
                    "shard": item.shard,
                    "file": item.file,
                    "speaker_id": item.speaker_id,
                    "split": item.split,
                    "binary_label": item.binary_label,
                    "utt_type": item.utt_type,
                    "native_sample_rate": item.native_sample_rate,
                    "source_snapshot": SOURCE_SNAPSHOT,
                }
            )
    temporary.replace(path)
    return _sha256_file(path)


def write_ledger(path: Path, ledger: Sequence[Mapping[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(ledger, key=lambda row: (row["split"], row["file"])))
    return _sha256_file(path)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluation manifest dev-full/test-full (D025)",
        "",
        f"- Trạng thái: **{report['status']}**",
        f"- Nguồn: {report['source_snapshot']}; seed {report['seed']}; ngưỡng mất mẫu {100 * report['max_loss_fraction']:.1f}% mỗi (split, utt_type).",
        f"- Shard đã nạp: {report['chunk_shards']}; mẫu trong sổ hash: {report['ledger_rows']}.",
    ]
    for reason in report["stop_reasons"]:
        lines.append(f"- **Lý do dừng:** {reason}")
    for split in EVAL_SPLITS:
        info = report["stats"][split]
        lines += [
            "",
            f"## {split}",
            "",
            f"- Dòng gốc {info['original_rows']} → giữ {info['kept_rows']}.",
            "",
            "| utt_type | Gốc | Giữ | Mất (%) | Speaker gốc | Speaker giữ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for utt_type, count in info["original_by_utt_type"].items():
            kept = info["kept_by_utt_type"].get(utt_type, 0)
            lines.append(
                f"| {utt_type} | {count} | {kept} | "
                f"{100 * info['loss_fraction_by_utt_type'][utt_type]:.2f} | "
                f"{info['speakers_original_by_utt_type'].get(utt_type, 0)} | "
                f"{info['speakers_kept_by_utt_type'].get(utt_type, 0)} |"
            )
        for reason, counts in info["dropped"].items():
            lines.append(f"- Loại do `{reason}`: {counts}")
    checks = report["stats"]["checks"]
    lines += [
        "",
        "## Kiểm tra",
        "",
        f"- Hash duy nhất trong dev/test: {checks['unique_hashes_dev']}/{checks['unique_hashes_test']}; trùng dev–test: {checks['dev_test_hash_overlap']}; train–dev: {checks['train_dev_hash_overlap']}; train–test: {checks['train_test_hash_overlap']}.",
        f"- Đối chiếu audit D017 cho dev: {report['stats']['audit_crosscheck']}.",
        f"- Speaker-disjoint train manifest/dev/test: {checks['speaker_disjoint']}.",
    ]
    if report.get("manifests"):
        lines += ["", "## Manifest"]
        for name, info in report["manifests"].items():
            lines.append(f"- `{info['path']}`: {info['rows']} dòng, SHA-256 `{info['sha256']}`")
    lines += [
        "",
        "## Giới hạn",
        "",
        "- Không bù mẫu; số mẫu mất là hệ quả khử trùng, không phải chọn lọc theo metric.",
        "- Hash tính lúc nạp shard; audit độc lập 100% cần tải lại shard ở revision ghim.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-loss-fraction", type=float, default=DEFAULT_MAX_LOSS_FRACTION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    shard_indices = list(plan["required_union"]["shards"])
    records = load_chunk_records(args.chunk_dir, shard_indices, OFFICIAL_GEOMETRY)
    audit = load_audit(args.audit)
    result = build_manifests(
        records=records,
        split_rows=load_split_rows(args.splits_dir),
        train_hashes=audit.train_hashes,
        train_speakers=audit.train_speakers,
        audit_dev_hashes=audit.dev_hashes,
        seed=args.seed,
        max_loss_fraction=args.max_loss_fraction,
    )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "D025",
        "scope": "evaluation_manifest_construction_not_scientific_result",
        "source_snapshot": SOURCE_SNAPSHOT,
        "seed": args.seed,
        "max_loss_fraction": args.max_loss_fraction,
        "status": result.status,
        "stop_reasons": result.stop_reasons,
        "chunk_shards": len(shard_indices),
        "ledger_rows": len(result.ledger),
        "stats": result.stats,
        "manifests": {},
    }
    if result.status == "ĐẠT":
        for split in EVAL_SPLITS:
            path = args.manifest_dir / f"{split}_full_v1.csv"
            digest = write_manifest(path, result.kept[split], MANIFEST_VERSIONS[split])
            report["manifests"][split] = {
                "path": path.relative_to(PROJECT_ROOT).as_posix()
                if path.is_relative_to(PROJECT_ROOT)
                else str(path),
                "rows": len(result.kept[split]),
                "sha256": digest,
            }
        report["ledger_sha256"] = write_ledger(args.ledger, result.ledger)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_report.write_text(render_markdown(report), encoding="utf-8")
    kept = {split: len(result.kept[split]) for split in EVAL_SPLITS}
    print(f"{result.status}: giữ {kept}; lý do dừng: {result.stop_reasons or 'không'}.")
    return 0 if result.status == "ĐẠT" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
