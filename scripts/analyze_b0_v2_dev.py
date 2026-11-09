"""R1: phân tích score B0 v2 đã lưu trên dev-full; không đọc audio/checkpoint/test."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from html import escape
import inspect
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.metrics.bootstrap import confusion_at_threshold, per_utt_type_eer, speaker_bootstrap
from src.metrics.evaluation import evaluate_development_scores

SEEDS = (2026, 2027, 2028)
ATTACKS = ("voice_conversion", "adversarial_attack", "replay")
NAMES = {"bonafide": "Giọng thật", "voice_conversion": "Chuyển đổi giọng nói",
         "adversarial_attack": "Nhiễu đối kháng", "replay": "Phát lại"}
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
SCOPE = "dev-full, không phải test"


def allowed_input(path: Path, filename: str) -> Path:
    """Từ chối tên/symlink tới split test trước khi mở file."""
    resolved = path.resolve()
    if (path.name != filename or resolved.name != filename
            or re.search(r"(^|[/\\_\W])test($|[/\\_\W])", str(resolved).lower())):
        raise ValueError("Chỉ nhận report B0 nối dài và manifest dev-full; từ chối test.")
    return resolved


def reject_test_metadata(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if ((key == "use_test" and item is not False)
                    or (key == "test" and item is not None)
                    or (key.endswith("split") and "test" in str(item).lower())):
                raise ValueError("Đầu vào có metadata split test; từ chối phân tích.")
            reject_test_metadata(item)
    elif isinstance(value, list):
        for item in value:
            reject_test_metadata(item)


def mean_std(values: Sequence[float]) -> dict[str, float | int]:
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Cần ít nhất hai giá trị hữu hạn để tính độ lệch chuẩn mẫu.")
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values), "ddof": 1}


def analyze(report_paths: dict[int, Path], manifest_path: Path, *,
            n_resamples: int = 1000) -> dict[str, Any]:
    """Chỉ bốn đầu vào dữ liệu; băm đúng byte được phân tích."""
    if set(report_paths) != set(SEEDS):
        raise ValueError("Phải có đúng ba report seed 2026, 2027, 2028.")
    paths = {s: allowed_input(report_paths[s], f"b0_v2_ext_seed{s}.json") for s in SEEDS}
    manifest_path = allowed_input(manifest_path, "closed_dev_full_v1.csv")
    raw_manifest = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw_manifest.decode("utf-8-sig"))))
    required = {"file", "speaker_id", "utt_type", "binary_label", "split", "shard"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest rỗng hoặc thiếu cột bắt buộc.")
    if any(r["split"] != "closed_dev" for r in rows):
        raise ValueError("Manifest phải chỉ chứa closed_dev; từ chối test/split khác.")
    if len({r["file"] for r in rows}) != len(rows):
        raise ValueError("Manifest chứa file lặp.")
    inputs = [{"path": str(manifest_path), "sha256": manifest_hash, "bytes": len(raw_manifest)}]
    seeds: list[dict[str, Any]] = []
    train_counts: set[int] = set()
    for seed in SEEDS:
        print(f"Phân tích seed {seed}: {len(rows)} mẫu {SCOPE}.", flush=True)
        raw = paths[seed].read_bytes()
        report = json.loads(raw)
        reject_test_metadata(report)
        if (report["training_seed"] != seed or report["data_seed"] != 2026
                or report["development_samples"] != len(rows)
                or report["development_manifest_sha256"] != manifest_hash
                or report["artifacts"]["development"]["source_manifest_sha256"] != manifest_hash):
            raise ValueError(f"Seed {seed}: seed/số mẫu/checksum manifest không khớp.")
        train_counts.add(report["train_samples"])
        controller = report["controller"]
        best = controller["best"]
        dev = best["development"]
        records = dev["records"]
        if len(records) != len(rows) or dev["sample_count"] != len(rows):
            raise ValueError(f"Seed {seed}: số record không khớp manifest.")
        for index, (record, row) in enumerate(zip(records, rows, strict=True)):
            label = int(row["binary_label"])
            if (row["utt_type"] not in ("bonafide", *ATTACKS)
                    or label != int(row["utt_type"] != "bonafide")
                    or (record["file"], record["speaker_id"], record["utt_type"], record["label"])
                    != (row["file"], row["speaker_id"], row["utt_type"], label)):
                raise ValueError(f"Seed {seed}: thứ tự/nhãn/metadata lệch manifest tại dòng {index+1}.")
        labels = [r["label"] for r in records]
        scores = [float(r["score"]) for r in records]
        types = [r["utt_type"] for r in records]
        speakers = [r["speaker_id"] for r in records]
        if labels != dev["labels"] or scores != dev["scores"]:
            raise ValueError(f"Seed {seed}: records lệch mảng labels/scores.")
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("Score phải là xác suất hữu hạn trong [0, 1].")
        recomputed = asdict(evaluate_development_scores(labels, scores, loss=dev["loss"], utt_types=types))
        metrics = ("eer", "roc_auc", "macro_f1", "eer_threshold", "decision_threshold")
        for key in metrics:
            if not math.isclose(recomputed[key], dev[key], rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Seed {seed}: metric {key} tính lại lệch report; dừng.")
        history = controller["history"]
        if (not history or len(history) != controller["completed_epochs"]
                or best["epoch"] != min(history, key=lambda h: h["eer"])["epoch"]):
            raise ValueError(f"Seed {seed}: history/best không nhất quán.")
        for row in history:
            if not all(math.isfinite(float(row[k])) for k in ("eer", "loss")):
                raise ValueError("History chứa metric không hữu hạn.")
        by_attack = per_utt_type_eer(labels, scores, types)
        if set(by_attack) != set(ATTACKS):
            raise ValueError("Thiếu kiểu tấn công trong dev-full.")
        threshold = dev["decision_threshold"]
        confusion = confusion_at_threshold(labels, scores, threshold=threshold)
        errors = {"bonafide": {"samples": confusion["tn"]+confusion["fp"],
                  "errors": confusion["fp"], "error_rate": confusion["fpr"],
                  "definition": "bonafide bị nhận thành spoof"}}
        for attack in ATTACKS:
            selected = [i for i, t in enumerate(types) if t in ("bonafide", attack)]
            cm = confusion_at_threshold([labels[i] for i in selected],
                                        [scores[i] for i in selected], threshold=threshold)
            errors[attack] = {"samples": cm["tp"]+cm["fn"], "errors": cm["fn"],
                              "error_rate": cm["fnr"], "definition": "spoof bị nhận thành bonafide"}
            by_attack[attack]["interpretation"] = (
                "không đủ để kết luận" if attack == "replay" else "mô tả trên dev-full")
        quantiles = {}
        for label, name in ((0, "bonafide"), (1, "spoof")):
            values = np.quantile([s for y, s in zip(labels, scores) if y == label],
                                 np.array(PERCENTILES)/100, method="linear")
            quantiles[name] = {str(p): float(v) for p, v in zip(PERCENTILES, values)}
        bootstrap = asdict(speaker_bootstrap(labels, scores, speakers,
                          n_resamples=n_resamples, seed=2026, confidence=.95))
        seeds.append({"training_seed": seed, "status": report["status"],
            "global_step": controller["global_step"], "completed_epochs": controller["completed_epochs"],
            "bad_evaluations": controller["bad_evaluations"],
            "best": {"epoch": best["epoch"], "global_step": best["global_step"],
                     **{k: recomputed[k] for k in metrics}, "loss": dev["loss"]},
            "per_attack_eer": by_attack, "speaker_bootstrap": bootstrap,
            "confusion": confusion, "errors_by_utt_type": errors,
            "score_percentiles": quantiles, "history": history,
            "source_tree_sha256": report["source_tree_sha256"]})
        inputs.append({"path": str(paths[seed]), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    if len(train_counts) != 1:
        raise ValueError("Ba report có số mẫu train khác nhau.")
    return {"report_version": 1, "evaluation_scope": SCOPE, "use_test": False, "inputs": inputs,
        "data": {"training_samples": next(iter(train_counts)),
            "training_shards": 57, "training_shards_source": "Khai báo D027/D028; không đọc lại train.",
            "development_samples": len(rows), "development_shards": len({r["shard"] for r in rows}),
            "development_speakers": len({r["speaker_id"] for r in rows})},
        "bootstrap_parameters": {"n_resamples": n_resamples, "seed": 2026, "confidence": .95,
            "unit": "speaker_id", "method": "percentile; lấy cụm speaker có hoàn lại",
            "interpretation": "CI có điều kiện trên checkpoint đã chọn bằng chính dev-full; không phải CI test hoặc CI qua seed."},
        "score_quantile_method": "numpy.quantile(method=linear)",
        "loss_note": "Loss lấy nguyên report/history; không tính lại BCE từ sigmoid score đã bão hòa.",
        "per_attack_note": "EER mỗi nhóm dùng bonafide chung và threshold EER riêng của nhóm; confusion dùng threshold tổng của chính seed.",
        "seeds": seeds,
        "aggregate": {k: mean_std([s["best"][k] for s in seeds]) for k in ("eer", "roc_auc", "macro_f1")},
        "per_attack_aggregate": {k: mean_std([s["per_attack_eer"][k]["eer"] for s in seeds]) for k in ATTACKS}}


def runtime_probe() -> dict[str, Any]:
    """Tensor tổng hợp; không mở thêm file dữ liệu/cấu hình/checkpoint."""
    import torch
    import torchaudio
    from src.models.lfcc_lcnn import LightCNNBaseline, build_lfcc_transform, lfcc_features
    recipe = {"sample_rate": 16000, "n_filter": 80, "n_lfcc": 60, "f_min": 0.,
              "f_max": 8000., "dct_type": 2, "norm": "ortho", "log_lf": False,
              "spectrum": {"n_fft": 512, "win_length": 400, "hop_length": 160,
                           "power": 2., "center": True, "pad_mode": "reflect"}}
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(2026)
    waveform = torch.randn(1, 64000, generator=generator)*.05
    lfcc = build_lfcc_transform(recipe)
    model = LightCNNBaseline().eval()
    shapes: dict[str, list[int]] = {}
    handles = [module.register_forward_hook(
        lambda _m, _i, output, key=name: shapes.__setitem__(key, list(output.shape)))
        for name, module in model.named_modules()
        if name in ("feature_extractor", "spatial_pool", "classifier")
        or name.startswith("feature_extractor.")]
    with torch.no_grad():
        features = lfcc_features(lfcc, waveform)
        logits = model(features.unsqueeze(1))
    for handle in handles:
        handle.remove()
    defaults = {name: p.default for name, p in inspect.signature(torchaudio.functional.resample).parameters.items()
                if p.default is not inspect.Parameter.empty}
    return {"torch_version": str(torch.__version__), "torchaudio_version": str(torchaudio.__version__),
            "resample_defaults": defaults, "lfcc_recipe": recipe,
            "waveform_shape": list(waveform.shape), "lfcc_shape": list(features.shape),
            "logit_shape": list(logits.shape), "module_shapes": shapes,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "finite_forward": bool(torch.isfinite(logits).all()),
            "evidence": "Forward tổng hợp CPU; không audio thật/huấn luyện; tham số LFCC khai báo tường minh, không đọc thêm cấu hình."}


def fmt(value: float, percent: bool = False, digits: int = 4) -> str:
    return f"{value*(100 if percent else 1):.{digits}f}".replace(".", ",")


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Phân tích B0 v2 — dev-full, không phải test", "",
             "EER trong JSON là tỉ lệ; bảng đổi sang %. SD qua seed dùng mẫu số n−1, khác CI theo người nói.", "",
             "## Phạm vi và đầu vào", "", json.dumps(result["data"], ensure_ascii=False), "",
             "| Đầu vào | SHA-256 |", "|---|---|"]
    for item in result["inputs"]:
        lines.append(f"| {Path(item['path']).name} | {item['sha256']} |")
    lines += ["", "## Kết quả tổng và CI theo người nói", "",
              "| Seed | EER (%) | ROC-AUC | Macro-F1 | Threshold | CI 95% EER (%) | CI 95% AUC | Hợp lệ |",
              "|---|---:|---:|---:|---:|---|---|---:|"]
    for seed in result["seeds"]:
        b, ci = seed["best"], seed["speaker_bootstrap"]
        lines.append(f"| {seed['training_seed']} | {fmt(b['eer'],True)} | {fmt(b['roc_auc'])} | {fmt(b['macro_f1'])} | {fmt(b['decision_threshold'],digits=9)} | "
                     f"[{fmt(ci['eer']['ci_low'],True)}; {fmt(ci['eer']['ci_high'],True)}] | "
                     f"[{fmt(ci['roc_auc']['ci_low'])}; {fmt(ci['roc_auc']['ci_high'])}] | {ci['valid_resamples']}/{ci['requested_resamples']} |")
    lines += ["", f"Bootstrap: {result['bootstrap_parameters']['n_resamples']} lần, seed 2026, confidence 0,95.",
              result["bootstrap_parameters"]["interpretation"], "", "### Trung bình ± độ lệch chuẩn mẫu", ""]
    for metric, stats in result["aggregate"].items():
        lines.append(f"- {metric}: {fmt(stats['mean'],metric=='eer')} ± {fmt(stats['sample_std'],metric=='eer')}.")
    lines += ["", "## EER theo kiểu tấn công", "", result["per_attack_note"], "",
              "| Nhóm | Bonafide | Spoof | 2026 (%) | 2027 (%) | 2028 (%) | Trung bình ± SD (điểm %) |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for attack in ATTACKS:
        group = result["seeds"][0]["per_attack_eer"][attack];stats = result["per_attack_aggregate"][attack]
        values = " | ".join(fmt(s["per_attack_eer"][attack]["eer"],True) for s in result["seeds"])
        lines.append(f"| {NAMES[attack]} | {group['bonafide_samples']} | {group['spoof_samples']} | {values} | {fmt(stats['mean'],True)} ± {fmt(stats['sample_std'],True)} |")
    replay = result["seeds"][0]["per_attack_eer"]["replay"]["spoof_samples"]
    lines += ["", f"**Replay: không đủ để kết luận** ({replay} mẫu spoof).",
              "", "## Ma trận nhầm lẫn và lỗi theo kiểu phát ngôn", "",
              "Hàng là nhãn thật; cột là nhãn dự đoán. Score ≥ threshold tổng của seed là spoof.", "",
              "| Seed | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|"]
    for seed in result["seeds"]:
        cm = seed["confusion"]
        lines.append(f"| {seed['training_seed']} | {cm['tn']} | {cm['fp']} | {cm['fn']} | {cm['tp']} |")
    lines += ["", "| Seed | Loại | Số mẫu | Số lỗi | Tỉ lệ lỗi (%) |", "|---|---|---:|---:|---:|"]
    for seed in result["seeds"]:
        for name, errors in seed["errors_by_utt_type"].items():
            lines.append(f"| {seed['training_seed']} | {NAMES[name]} | {errors['samples']} | {errors['errors']} | {fmt(errors['error_rate'],True)} |")
    lines += ["", "## Phân vị score", "", "Phân vị tuyến tính; không trung bình hóa threshold.", "",
              "| Seed | Nhóm | P1 | P5 | P25 | P50 | P75 | P95 | P99 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in result["seeds"]:
        for name, quantiles in seed["score_percentiles"].items():
            values = " | ".join(f"{quantiles[str(p)]:.9g}" for p in PERCENTILES)
            lines.append(f"| {seed['training_seed']} | {name} | {values} |")
    lines += ["", "## History dev-full", "", result["loss_note"], "",
              "| Seed | Epoch | Bước | EER (%) | Loss |", "|---|---:|---:|---:|---:|"]
    for seed in result["seeds"]:
        for row in seed["history"]:
            lines.append(f"| {seed['training_seed']} | {row['epoch']} | {row['global_step']} | {fmt(row['eer'],True)} | {fmt(row['loss'],digits=6)} |")
    if "technical_checks" in result:
        lines += ["", "## Kiểm kỹ thuật cục bộ (tensor tổng hợp)", "",
                  "Lệnh: .venv/Scripts/python.exe scripts/analyze_b0_v2_dev.py (runtime_probe).", "",
                  "```json", json.dumps(result["technical_checks"], indent=2, ensure_ascii=False), "```"]
    return "\n".join(lines)+"\n"


def svg_text(x: float, y: float, text: str, extra: str = "") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" {extra}>{escape(text)}</text>'


def svg_frame(title: str, description: str, ymax: float) -> list[str]:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="160mm" height="94.545mm" viewBox="0 0 1100 650" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
             '<style>text{font-family:"Times New Roman",serif;fill:#172b4d;font-size:27px}.small{font-size:25px}.title{font-size:32px}</style>',
             '<rect width="1100" height="650" fill="white"/>',
             svg_text(550,40,title,'text-anchor="middle" font-weight="bold" class="title"'),
             svg_text(550,72,SCOPE,'text-anchor="middle" class="small"')]
    for tick in range(6):
        value = ymax*tick/5;y = 480-360*tick/5
        parts += [f'<path d="M 110 {y:.2f} H 1020" stroke="#d9e2ec" fill="none"/>',
                  svg_text(96,y+6,fmt(value,digits=1),'text-anchor="end" class="small"')]
    parts += ['<path d="M 110 110 V 480 H 1020" stroke="#334e68" fill="none"/>',
              '<text transform="translate(32 300) rotate(-90)" text-anchor="middle">EER (%)</text>']
    return parts


def epoch_svg(result: dict[str, Any]) -> str:
    seeds = result["seeds"]
    ymax = max(2.,math.ceil(max(r["eer"]*100 for s in seeds for r in s["history"])/2)*2)
    parts = svg_frame("B0 v2 — EER dev-full theo epoch","Ba seed; khoanh epoch best, dấu chéo ở điểm dừng.",ymax)
    colors = ("#1d4ed8","#b45309","#047857")
    epochs = sorted({r["epoch"] for s in seeds for r in s["history"]})
    def x(epoch: int) -> float:
        return 160+(epoch-epochs[0])*800/max(1,epochs[-1]-epochs[0])
    def y(eer: float) -> float:
        return 480-eer*100/ymax*360
    for epoch in epochs:
        parts.append(svg_text(x(epoch),510,str(epoch),'text-anchor="middle"'))
    for index,(seed,color) in enumerate(zip(seeds,colors)):
        points = " ".join(f"{x(r['epoch']):.2f},{y(r['eer']):.2f}" for r in seed["history"])
        dash = ("none","9 4","3 4")[index]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
        for row in seed["history"]:
            px,py = x(row["epoch"]),y(row["eer"])
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{color}"/>')
            if row["epoch"] == seed["best"]["epoch"]:
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="10" fill="none" stroke="{color}" stroke-width="2"/>')
            if row is seed["history"][-1]:
                parts.append(f'<path d="M {px-7:.2f} {py-7:.2f} l 14 14 m -14 0 l 14 -14" stroke="{color}" stroke-width="2"/>')
        lx = 180+index*290
        parts += [f'<path d="M {lx} 562 h 40" stroke="{color}" stroke-width="3"/>',
                  svg_text(lx+50,568,f"Seed {seed['training_seed']}")]
    parts += [svg_text(560,540,"Epoch",'text-anchor="middle"'),
              svg_text(550,605,"Vòng tròn: best theo EER; dấu ×: kết thúc dừng sớm.",'text-anchor="middle" class="small"'),
              svg_text(550,632,"Nguồn: reports/b0_v2_dev_analysis.json · Không dùng tập test.",'text-anchor="middle" class="small"'),"</svg>"]
    return "\n".join(parts)


def attacks_svg(result: dict[str, Any]) -> str:
    stats = result["per_attack_aggregate"]
    ymax = max(2.,math.ceil(max((a["mean"]+a["sample_std"])*100 for a in stats.values())/2)*2)
    parts = svg_frame("B0 v2 — EER theo kiểu tấn công","Trung bình ba seed và độ lệch chuẩn mẫu; replay ít mẫu.",ymax)
    for index,attack in enumerate(ATTACKS):
        a = stats[attack];px = 270+index*295
        py = 480-a["mean"]*100/ymax*360
        high = 480-(a["mean"]+a["sample_std"])*100/ymax*360
        low = 480-max(0,a["mean"]-a["sample_std"])*100/ymax*360
        color = "#d97706" if attack=="replay" else "#356d99"
        parts += [f'<rect x="{px-65}" y="{py:.2f}" width="130" height="{480-py:.2f}" fill="{color}"/>',
                  f'<path d="M {px} {high:.2f} V {low:.2f} M {px-14} {high:.2f} H {px+14} M {px-14} {low:.2f} H {px+14}" stroke="#172b4d" stroke-width="2" fill="none"/>',
                  svg_text(px,high-14,f"{fmt(a['mean'],True,2)} ± {fmt(a['sample_std'],True,2)}",'text-anchor="middle"'),
                  svg_text(px,512,NAMES[attack]+("*" if attack=="replay" else ""),'text-anchor="middle"')]
        count = result["seeds"][0]["per_attack_eer"][attack]["spoof_samples"]
        parts.append(svg_text(px,537,f"{count:,} mẫu spoof".replace(",","."),'text-anchor="middle" class="small"'))
    replay = result["seeds"][0]["per_attack_eer"]["replay"]["spoof_samples"]
    parts += [svg_text(550,572,"Thanh: trung bình 3 seed; vạch: ± độ lệch chuẩn mẫu (n−1).",'text-anchor="middle" class="small"'),
              svg_text(550,605,f"* Replay chỉ {replay} mẫu spoof: không đủ để kết luận.",'text-anchor="middle" class="small"'),
              svg_text(550,632,"Nguồn: reports/b0_v2_dev_analysis.json · Bonafide chung cho từng nhóm.",'text-anchor="middle" class="small"'),"</svg>"]
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = analyze({s: ROOT/f"reports/b0_v2_ext_seed{s}.json" for s in SEEDS},
                     ROOT/"data/manifests/closed_dev_full_v1.csv")
    result["technical_checks"] = runtime_probe()
    (ROOT/"reports/b0_v2_dev_analysis.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    (ROOT/"reports/b0_v2_dev_analysis.md").write_text(markdown(result),encoding="utf-8",newline="\n")
    for name,content in (("hinh_4_1_b0_v2_eer_theo_epoch.svg",epoch_svg(result)),
                         ("hinh_4_2_b0_v2_eer_theo_kieu_tan_cong.svg",attacks_svg(result))):
        (ROOT/"docs/figures"/name).write_text(content+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":"passed","evaluation_scope":SCOPE,"aggregate":result["aggregate"]},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError,KeyError,OSError,TypeError) as error:
        print(f"LỖI: {error}",file=sys.stderr)
        raise SystemExit(1) from error
