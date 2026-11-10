"""T23: CI ghép cặp theo người nói trên hai file score đã lưu."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_model_registry import CONFIG, atomic_json, load_json, reject_test_path, require, sha256_file, utc
from src.metrics.bootstrap import paired_speaker_bootstrap, speaker_bootstrap


def read_scores(path: Path) -> list[dict[str, Any]]:
    reject_test_path(path)
    value = load_json(path)
    records = value["records"] if "records" in value else value["controller"]["best"]["development"]["records"]
    require(bool(records) and len({r["file"] for r in records}) == len(records), "File score rỗng hoặc có file trùng.")
    return records


def aligned_records(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> None:
    require(len(a) == len(b) and bool(a), "Hai file score lệch số mẫu.")
    for index, (left, right) in enumerate(zip(a, b, strict=True)):
        require(all(left[key] == right[key] for key in ("file", "label", "speaker_id")),
                f"Hai file score lệch mẫu/thứ tự/nhãn/speaker tại dòng{index+1}.")


def compare_records(a: list[dict[str, Any]], b: list[dict[str, Any]],
                    parameters: dict[str, Any]) -> dict[str, Any]:
    aligned_records(a, b)
    labels = [r["label"] for r in a]
    speakers = [r["speaker_id"] for r in a]
    left, right = [r["score"] for r in a], [r["score"] for r in b]
    return {"samples": len(a), "speakers": len(set(speakers)), "difference_direction": "EER(a)-EER(b)",
            "paired_eer_difference": asdict(paired_speaker_bootstrap(labels, left, right, speakers, **parameters)),
            "system_a": asdict(speaker_bootstrap(labels, left, speakers, **parameters)),
            "system_b": asdict(speaker_bootstrap(labels, right, speakers, **parameters))}


def compare_files(a: Path, b: Path, config: dict[str, Any]) -> dict[str, Any]:
    began = time.perf_counter()
    result = compare_records(read_scores(a), read_scores(b), config["bootstrap"])
    return {"decision": "D032", "scope": "dev-full, không phải test", "use_test": False,
            "created_utc": utc(), "parameters": config["bootstrap"],
            "inputs": {"a": str(a), "b": str(b), "a_sha256": sha256_file(a), "b_sha256": sha256_file(b)},
            "shards": config["dev"]["shards"] if result["samples"] == config["dev"]["samples"] else None,
            **result, "seconds": time.perf_counter() - began}


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    diff = report["paired_eer_difference"]
    lines = ["# Bootstrap ghép cặp — dev-full, không phải test", "", "Quyết định: D032.",
             f"Mẫu: {report['samples']}; người nói: {report['speakers']}; shard: {report['shards']}.",
             "Hiệu EER(a)-EER(b), đơn vị điểm phần trăm:",
             f"{100*diff['estimate']:.6f}; CI95% [{100*diff['ci_low']:.6f}; {100*diff['ci_high']:.6f}].", ""]
    for name in ("system_a", "system_b"):
        eer = report[name]["eer"]
        lines.append(f"{name}: EER{100*eer['estimate']:.6f}%; CI95% [{100*eer['ci_low']:.6f}; {100*eer['ci_high']:.6f}].")
    lines += ["", "Bootstrap cụm người nói; không phải SD qua seed. Không chọn mô hình/threshold bằng kết quả này."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare_files(args.a, args.b, load_json(args.config))
        atomic_json(args.output, report)
        write_markdown(args.output.with_suffix(".md"), report)
        print("Đã tính bootstrap ghép cặp trên score dev đã lưu.")
        return 0
    except Exception as exc:
        print("LỖI bootstrap: " + str(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
