"""R2: đo phổ dev-only và vẽ hình từ report, không chấm mô hình."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_shortcut_dev import (GROUPS, CONDITIONS, BMODELS, XMODEL,
    MANIFEST_SHA, SUBSET_SHA, apply_condition, file_list_sha, open_dev_artifact,
    read_dev_manifest, sha256_file, subset_indices, require)

BANDS = ((6000, 7000), (7000, 7500), (7500, 7800), (7800, 7950), (7950, 8000))
LABELS = dict(zip((*GROUPS, "bonafide_roundtrip"),
    ("Bonafide", "Chuyển đổi giọng (VC)", "Nhiễu đối kháng (AP)", "Phát lại", "Bonafide round-trip")))
FREQUENCIES = np.fft.rfftfreq(64000, 1 / 16000)
WINDOW = np.hanning(64000)


def band_masks(frequencies: np.ndarray) -> list[np.ndarray]:
    return [(frequencies >= lo) & ((frequencies <= hi) if i == 4 else (frequencies < hi))
            for i, (lo, hi) in enumerate(BANDS)]


def spectrum(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Công suất một phía, nhân đôi bin nội; chuẩn hóa tổng của từng mẫu."""
    require(waveform.shape == (64000,) and bool(np.isfinite(waveform).all()),
            "Cần waveform hữu hạn, đúng 64.000 mẫu.")
    power = np.abs(np.fft.rfft(waveform.astype(np.float64) * WINDOW)) ** 2
    power[1:-1] *= 2
    require(float(power.sum()) > 0, "Không đo phổ waveform im lặng hoàn toàn.")
    power /= power.sum()
    db = lambda x: 10 * np.log10(np.maximum(x, 1e-300))
    bands = np.array([db(power[mask].sum()) for mask in band_masks(FREQUENCIES)])
    # 80 ô 50 Hz: [4000,4050), ..., [7950,8000], vẽ tại tâm ô.
    bins = np.array([db(power[(FREQUENCIES >= lo) &
        ((FREQUENCIES <= lo + 50) if lo == 7950 else (FREQUENCIES < lo + 50))].sum())
        for lo in range(4000, 8000, 50)])
    return bands, bins


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def spectrum_table(report: dict[str, Any]) -> str:
    lines = ["| Nhóm | n | 6–7 kHz | 7–7,5 kHz | 7,5–7,8 kHz | 7,8–7,95 kHz | 7,95–8 kHz |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group, result in report["groups"].items():
        cells = [f"{fmt(b['median_db'])} [{fmt(b['p25_db'])}; {fmt(b['p75_db'])}]" for b in result["bands"]]
        lines.append(f"| {LABELS[group]} | {result['samples']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablation_table(probe: dict[str, Any]) -> str:
    lines = ["| Mô hình | Điều kiện | EER tổng (%) | EER VC (%) | EER AP (%) | EER replay (%) | FPR bonafide (%) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for model in (*BMODELS, XMODEL):
        for condition in CONDITIONS:
            r = probe["results"][model]["conditions"][condition]
            values = [r["overall_eer"], *[r["per_attack_eer"][g]["eer"] for g in GROUPS[1:]], r["groups"]["bonafide"]["fpr"]]
            lines.append(f"| {model} | {condition} | " + " | ".join(fmt(v * 100, 4) for v in values) + " |")
    return "\n".join(lines)


def svg_text(x: float, y: float, value: str, extra: str = "") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" {extra}>{escape(value)}</text>'


def svg_marker(index: int, color: str) -> str:
    shapes = ('<circle cx="0" cy="0" r="7"/>',
              '<rect x="-7" y="-7" width="14" height="14"/>',
              '<polygon points="0,-9 8,7 -8,7"/>',
              '<polygon points="0,-9 9,0 0,9 -9,0"/>')
    return f'<g fill="{color}" stroke="{color}">{shapes[index]}</g>'


def chart(series: list[tuple[str, list[float]]], title: str, labels: list[str],
          ymin: float, ymax: float, ylabel: str, footer: str, *, discrete: bool = False) -> str:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="160mm" height="110mm" viewBox="0 0 1100 760" role="img">',
        f'<title>{escape(title)}</title>', '<rect width="1100" height="760" fill="white"/>',
        '<style>text{font-family:"Times New Roman",serif;font-size:25px;fill:#172b4d}</style>',
        svg_text(550, 36, title, 'text-anchor="middle" font-weight="bold"'),
        svg_text(550, 70, "dev-full, không phải test", 'text-anchor="middle"')]
    for tick in range(6):
        y = 480 - 360 * tick / 5
        parts += [f'<path d="M110 {y} H1040" stroke="#d9e2ec"/>',
                  svg_text(96, y + 7, fmt(ymin + (ymax-ymin)*tick/5, 0), 'text-anchor="end"')]
    parts += [f'<text transform="translate(30 300) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>']
    colors = ("#1d4ed8", "#b45309", "#047857", "#9333ea", "#111827")
    dashes = ("", "14 7", "3 6", "18 5 3 5", "9 4")
    for i, (name, values) in enumerate(series):
        points = [(110 + 930*j/(len(values)-1), 480-360*(v-ymin)/(ymax-ymin)) for j,v in enumerate(values)]
        if discrete:
            # Dịch nhẹ ngang trong mỗi nhóm để các mô hình gần nhau không che marker.
            offset = (i - (len(series)-1)/2) * 12
            parts.append(f'<g class="data-series" aria-label="{escape(name)}">')
            for x, y in points:
                parts.append(f'<g class="data-point" transform="translate({x+offset:.2f} {y:.2f})">{svg_marker(i, colors[i])}</g>')
            parts.append('</g>')
        else:
            coords = " ".join(f"{x:.2f},{y:.2f}" for x,y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[i]}" stroke-width="3" stroke-dasharray="{dashes[i]}"/>')
        ly = 590 + i*29
        if discrete:
            parts.append(f'<g transform="translate(145 {ly-8})">{svg_marker(i, colors[i])}</g>')
        else:
            parts.append(f'<path d="M110 {ly-8} H180" stroke="{colors[i]}" stroke-width="3" stroke-dasharray="{dashes[i]}"/>')
        parts.append(svg_text(195, ly, name))
    for i, label in enumerate(labels):
        parts.append(svg_text(110+930*i/(len(labels)-1), 520, label, 'text-anchor="middle"'))
    parts += [svg_text(550, 553, footer, 'text-anchor="middle"'), '</svg>']
    return "\n".join(parts)


def spectrum_svg(report: dict[str, Any]) -> str:
    series = [(LABELS[g], r["mean_log_power_db"]) for g,r in report["groups"].items()]
    low = min(min(v) for _,v in series)
    high = max(max(v) for _,v in series)
    return chart(series, "Phổ dải cao theo nhóm", ["4", "5", "6", "7", "8 kHz"],
                 float(np.floor(low/10)*10), float(np.ceil(high/10)*10+10), "Công suất tương đối (dB/ô 50 Hz)", "3.120 mẫu gốc/169 shard; trung bình theo mẫu trong miền dB")


def ablation_svg(probe: dict[str, Any]) -> str:
    series = [(m, [100*probe["results"][m]["conditions"][c]["overall_eer"] for c in CONDITIONS]) for m in (*BMODELS, XMODEL)]
    return chart(series, "EER tổng theo điều kiện band-limit", ["Gốc", "Round-trip", "Lọc 7 kHz", "Lọc 4 kHz"],
                 0, 60, "EER (%)", "B0: 38.777 mẫu/186 shard; XLS-R: tập con 3.120 mẫu/169 shard", discrete=True)


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(8)
    rows = read_dev_manifest(ROOT / "data/manifests/closed_dev_full_v1.csv")
    selected = subset_indices(rows)
    require(file_list_sha([rows[i]["file"] for i in selected]) == SUBSET_SHA, "SHA-256 tập con không khớp; dừng R2.")
    artifact = ROOT / "data/processed/packed/closed_dev_full_v1"
    ds = open_dev_artifact(artifact)
    require(len(ds) == len(rows) == 38777, "Số mẫu artifact lệch dev-full.")
    for record, row in zip(ds.records, rows):
        require((record.file, record.speaker_id, record.utt_type, record.binary_label) ==
                (row["file"], row["speaker_id"], row["utt_type"], int(row["binary_label"])), "Packed lệch manifest.")
    probe_path = ROOT / "reports/shortcut_probe_dev.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    inputs = {"manifest_sha256": MANIFEST_SHA, "subset_files_sha256": SUBSET_SHA,
              "probe_report_sha256": sha256_file(probe_path)}
    for name in ("metadata.json", "index.csv"):
        inputs[name + "_sha256"] = sha256_file(artifact / name)
    require(inputs['metadata.json_sha256'] == probe['identity']['inputs']['metadata_sha256'] and
            inputs['index.csv_sha256'] == probe['identity']['inputs']['index_sha256'], "Packed khác T19b.")
    bands: dict[str, list[np.ndarray]] = {g: [] for g in LABELS}
    grids: dict[str, list[np.ndarray]] = {g: [] for g in LABELS}
    raw_digest = hashlib.sha256()
    for position, index in enumerate(selected):
        record = ds.records[index]
        raw = np.asarray(ds._waveforms[record.offset_samples:record.offset_samples+record.num_samples], dtype='<f4').tobytes()
        require(hashlib.sha256(raw).hexdigest() == record.waveform_sha256, "Checksum waveform tập con lệch.")
        raw_digest.update(raw)
        waveform = ds[index]["waveform"]
        group = rows[index]["utt_type"]
        variants = [(group, waveform)]
        if group == "bonafide":
            variants.append(("bonafide_roundtrip", apply_condition(waveform, "roundtrip_40k", group)))
        for key, value in variants:
            b, grid = spectrum(value.numpy())
            bands[key].append(b)
            grids[key].append(grid)
        if position % 500 == 0:
            print(f"Đã đo {position+1}/{len(selected)} mẫu", flush=True)
    inputs['selected_raw_waveforms_concatenated_sha256'] = raw_digest.hexdigest()
    groups = {}
    for group in LABELS:
        array = np.stack(bands[group])
        groups[group] = {"samples": len(array), "shards": len({rows[i]['shard'] for i in selected if rows[i]['utt_type'] == ('bonafide' if group == 'bonafide_roundtrip' else group)}),
            "bands": [{"lower_hz": lo, "upper_hz": hi, "samples": len(array),
                       "median_db": float(np.median(array[:,j])), "p25_db": float(np.percentile(array[:,j],25)),
                       "p75_db": float(np.percentile(array[:,j],75))} for j,(lo,hi) in enumerate(BANDS)],
            "mean_log_power_db": np.stack(grids[group]).mean(axis=0).tolist()}
    report = {"task": "R2", "use_test": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "data": {"samples": len(selected), "shards": len({rows[i]['shard'] for i in selected}), "dev_samples": len(rows), "dev_shards": len({r['shard'] for r in rows})},
        "parameters": {"sample_rate":16000, "target_samples":64000, "data_seed":2026, "amplitude_policy":"peak", "crop":"center/repeat-then-trim", "window":"symmetric Hann, numpy.hanning(64000)", "fft":"float64 rfft; one-sided power, double interior bins", "band_bounds":"lower inclusive, upper exclusive; final includes 8000 Hz", "normalization":"per-sample power / total one-sided windowed power", "floor_ratio":1e-300, "grid_centers_hz":list(range(4025,8000,50)), "grid":"sum relative power per 50 Hz bin then 10log10; arithmetic mean over samples in dB", "roundtrip":probe['filters']['roundtrip_40k'], "torch_version":torch.__version__, "numpy_version":np.__version__},
        "groups": groups, "elapsed_seconds":time.monotonic()-started}
    out = ROOT / "reports/highband_spectrum_dev"
    out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    out.with_suffix('.md').write_text('# R2 — Phổ dải cao trên dev-full\n\nDev-full, không phải test. 3.120 mẫu/169 shard; nhóm round-trip dùng lại 1.000 bonafide, không thêm mẫu độc lập.\n\nTrung vị [P25; P75], dB tương đối so với tổng công suất từng mẫu. Đây là số đo mô tả phổ, không kết luận shortcut.\n\n'+spectrum_table(report)+'\n\nTham số và SHA-256:\n\n```json\n'+json.dumps({k:report[k] for k in ('inputs','data','parameters','elapsed_seconds')},ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    # Điểm dừng theo bản giao: chưa vẽ/cập nhật chương nếu phổ mâu thuẫn.
    require(all(groups['voice_conversion']['bands'][j]['median_db'] < groups['bonafide']['bands'][j]['median_db'] for j in (3,4)), "Phổ VC không thấp hơn bonafide sát 8 kHz; dừng, chờ kiểm.")
    (ROOT/'docs/figures/hinh_4_3_pho_dai_cao_theo_nhom.svg').write_text(spectrum_svg(report),encoding='utf-8')
    (ROOT/'docs/figures/hinh_4_4_band_limit_ablation.svg').write_text(ablation_svg(probe),encoding='utf-8')
    print(spectrum_table(report))
    print(f"Hoàn tất R2 trong {report['elapsed_seconds']:.3f} giây; SHA-256 tập con và manifest khớp.")


if __name__ == '__main__':
    main()
