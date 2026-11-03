#!/usr/bin/env python3
"""Lắp packed artifact cho train / dev-full / test-full từ chunk từng shard (T07).

Dev-full và test-full dùng chunk đã nạp trong `data/processed/eval_chunks/`. Train
(15.885 mẫu của `development-20k-v2`) được nạp thành chunk từ Parquet cục bộ bằng
cùng `src.data.eval_ingest`, không xóa Parquet, rồi lắp bằng cùng một hàm. Mỗi artifact
được băm lại toàn bộ sau khi ghi. Test-full chỉ được đóng gói và băm lại; không có
mô hình hay metric nào chạy trên test trong script này.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_evaluation_shards import TreeEntry, load_tree  # noqa: E402
from scripts.plan_evaluation_shards import OFFICIAL_GEOMETRY  # noqa: E402
from src.data import (  # noqa: E402
    MANIFEST_COLUMNS,
    assemble_packed_from_chunks,
    verify_packed_artifact,
)
from src.data.eval_ingest import WantedRecord, process_shard  # noqa: E402
from src.data.packed import _SHARD_NAME  # noqa: E402


DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "vsasv_parquet" / "data"
DEFAULT_EVAL_CHUNKS = PROJECT_ROOT / "data" / "processed" / "eval_chunks"
DEFAULT_TRAIN_CHUNKS = PROJECT_ROOT / "data" / "processed" / "train_chunks"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "packed"
DEFAULT_TREE_CACHE = PROJECT_ROOT / "reports" / "vsasv_shard_tree.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "packed_artifacts_v1.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "packed_artifacts_v1.md"
MANIFESTS = PROJECT_ROOT / "data" / "manifests"

# Tên artifact → (manifest, split, nguồn chunk).
TARGETS: dict[str, tuple[str, str, str]] = {
    "train": ("development_20k_v2.csv", "closed_train", "train"),
    "dev-full": ("closed_dev_full_v1.csv", "closed_dev", "eval"),
    "test-full": ("closed_test_full_v1.csv", "closed_test", "eval"),
}
ARTIFACT_DIRS = {
    "train": "train_development_20k_v2",
    "dev-full": "closed_dev_full_v1",
    "test-full": "closed_test_full_v1",
}


def shard_index(shard: str) -> int:
    match = _SHARD_NAME.match(shard)
    if match is None:
        raise ValueError(f"Tên shard không hợp lệ: {shard}")
    return int(match.group(1))


def train_wanted_by_shard(
    manifest_csv: Path, split: str = "closed_train"
) -> dict[int, dict[str, WantedRecord]]:
    """Nhóm các dòng train của manifest theo shard để nạp chunk."""

    wanted: dict[int, dict[str, WantedRecord]] = {}
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError("Schema manifest train không hợp lệ.")
        for row in reader:
            if row["split"] != split:
                continue
            bucket = wanted.setdefault(shard_index(row["shard"]), {})
            if row["file"] in bucket:
                raise ValueError(f"Manifest train lặp file {row['file']}.")
            bucket[row["file"]] = WantedRecord(
                file=row["file"],
                split=row["split"],
                speaker_id=row["speaker_id"],
                utt_type=row["utt_type"],
            )
    if not wanted:
        raise ValueError(f"Manifest không có dòng {split}.")
    return wanted


def build_train_chunks(
    *,
    wanted_by_shard: Mapping[int, Mapping[str, WantedRecord]],
    parquet_dir: Path,
    chunk_dir: Path,
    tree: Mapping[int, TreeEntry],
    shard_name: Callable[[int], str],
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Nạp chunk train từ Parquet cục bộ; không bao giờ xóa Parquet."""

    say = progress or (lambda _message: None)
    counts = {"ingested": 0, "already_done": 0}
    for index in sorted(wanted_by_shard):
        if index not in tree:
            raise ValueError(f"Thiếu thông tin tree Hugging Face cho shard {index}.")
        path = parquet_dir / shard_name(index)
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu shard Parquet cục bộ cho train: {path}")
        result = process_shard(
            shard_path=path,
            shard_index=index,
            wanted=wanted_by_shard[index],
            chunk_dir=chunk_dir,
            delete_after=False,
            expected_size=tree[index].size,
            expected_sha256=tree[index].sha256,
        )
        counts[str(result["status"])] += 1
        say(f"Chunk train shard {index}: {result['status']} ({result['records']} mẫu).")
    return counts


def build_one(
    *,
    name: str,
    manifest: Path,
    split: str,
    chunk_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    summary = assemble_packed_from_chunks(manifest, chunk_dir, output_dir, split=split)  # type: ignore[arg-type]
    built_seconds = time.perf_counter() - started
    started = time.perf_counter()
    verification = verify_packed_artifact(output_dir, allow_test=split == "closed_test")
    verify_seconds = time.perf_counter() - started
    if verification.waveforms_sha256 != summary.waveforms_sha256:
        raise ValueError(f"Băm lại artifact {name} không khớp lúc ghi.")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "name": name,
        "split": split,
        "artifact_dir": output_dir.relative_to(PROJECT_ROOT).as_posix()
        if output_dir.is_relative_to(PROJECT_ROOT)
        else str(output_dir),
        "manifest": manifest.name,
        "manifest_version": metadata["manifest_version"],
        "source_manifest_sha256": summary.source_manifest_sha256,
        "artifact_count": summary.artifact_count,
        "total_samples": summary.total_samples,
        "waveform_bytes": summary.total_samples * 4,
        "chunk_shard_count": metadata["chunk_shard_count"],
        "index_sha256": summary.index_sha256,
        "waveforms_sha256": summary.waveforms_sha256,
        "metadata_sha256": _sha256(output_dir / "metadata.json"),
        "build_seconds": round(built_seconds, 3),
        "verify_seconds": round(verify_seconds, 3),
        "verified_rows": verification.artifact_count,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vn_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _vn_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Packed artifact train / dev-full / test-full (T07)",
        "",
        f"- Ngày tạo: {report['generated_at']}",
        "- Định dạng: `vsasv-packed-v1`, float32 sau resample 16 kHz, trước amplitude/crop.",
        "- Nguồn: chunk từng shard (`src/data/eval_ingest.py`); train nạp từ Parquet cục bộ,"
        " dev/test từ `data/processed/eval_chunks/`.",
        f"- Chunk train mới nạp/đã có: {report['train_chunks']}.",
        "",
        "| Artifact | Split | Manifest | Mẫu | Shard | GiB | Build (s) | Băm lại (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["artifacts"]:
        lines.append(
            f"| {item['name']} | {item['split']} | `{item['manifest']}` | "
            f"{_vn_int(item['artifact_count'])} | {item['chunk_shard_count']} | "
            f"{_vn_float(item['waveform_bytes'] / 1024**3, 2)} | "
            f"{_vn_float(item['build_seconds'], 1)} | {_vn_float(item['verify_seconds'], 1)} |"
        )
    lines += ["", "## Checksum", ""]
    for item in report["artifacts"]:
        lines.append(
            f"- {item['name']}: waveforms `{item['waveforms_sha256']}`, index "
            f"`{item['index_sha256']}`, metadata `{item['metadata_sha256']}`, manifest "
            f"`{item['source_manifest_sha256']}`."
        )
    lines += [
        "",
        "## Giới hạn",
        "",
        "- Băm lại chỉ chứng minh artifact khớp chunk; đối chiếu với Parquet nằm ở báo cáo"
        " parity (T08).",
        "- Test-full chỉ được đóng gói và băm lại; chưa chạy mô hình hay metric trên test.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS), default=list(TARGETS))
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--eval-chunks", type=Path, default=DEFAULT_EVAL_CHUNKS)
    parser.add_argument("--train-chunks", type=Path, default=DEFAULT_TRAIN_CHUNKS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tree-cache", type=Path, default=DEFAULT_TREE_CACHE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--reserve-gib", type=float, default=15.0)
    return parser.parse_args()


def _estimated_bytes(targets: Sequence[str], eval_chunks: Path) -> int:
    # Ước tính trên: dev+test ≤ tổng chunk eval; train ≤ 15.885 × 400 KB cho chunk + packed.
    total = 0
    if {"dev-full", "test-full"} & set(targets):
        total += sum(path.stat().st_size for path in eval_chunks.glob("*.f32"))
    if "train" in targets:
        total += 2 * 15_885 * 400_000
    return total


def main() -> int:
    args = parse_args()
    needed = _estimated_bytes(args.targets, args.eval_chunks) + round(args.reserve_gib * 1024**3)
    args.output_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output_root).free
    if free < needed:
        raise ValueError(f"Không đủ dung lượng trống: còn {free} byte, cần khoảng {needed} byte.")
    for name in args.targets:
        target = args.output_root / ARTIFACT_DIRS[name]
        if target.exists():
            raise FileExistsError(f"Artifact {name} đã tồn tại; không ghi đè: {target}")

    train_chunks: dict[str, int] = {}
    if "train" in args.targets:
        tree = load_tree(args.tree_cache, OFFICIAL_GEOMETRY)
        from scripts.plan_evaluation_shards import shard_filename

        train_chunks = build_train_chunks(
            wanted_by_shard=train_wanted_by_shard(MANIFESTS / TARGETS["train"][0]),
            parquet_dir=args.parquet_dir,
            chunk_dir=args.train_chunks,
            tree=tree,
            shard_name=lambda index: shard_filename(OFFICIAL_GEOMETRY, index),
            progress=lambda message: print(message, flush=True),
        )

    artifacts = []
    for name in args.targets:
        manifest_name, split, source = TARGETS[name]
        chunk_dir = args.train_chunks if source == "train" else args.eval_chunks
        print(f"Lắp artifact {name} ({split}) ...", flush=True)
        item = build_one(
            name=name,
            manifest=MANIFESTS / manifest_name,
            split=split,
            chunk_dir=chunk_dir,
            output_dir=args.output_root / ARTIFACT_DIRS[name],
        )
        print(
            f"  {item['artifact_count']} mẫu, {item['waveform_bytes'] / 1024**3:.2f} GiB, "
            f"build {item['build_seconds']} s, băm lại {item['verify_seconds']} s.",
            flush=True,
        )
        artifacts.append(item)

    report = {
        "report_version": 1,
        "task": "T07",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_chunks": train_chunks,
        "artifacts": artifacts,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"ĐẠT: {len(artifacts)} artifact đã lắp và băm lại.")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        raise SystemExit(1) from error
