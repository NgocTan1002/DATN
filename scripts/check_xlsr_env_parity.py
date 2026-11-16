#!/usr/bin/env python3
"""Kiểm tương đương môi trường (parity) trên dev-full cho XLS-R theo D033 mục 4."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts import train_xlsr as runner
from src.metrics.bootstrap import fast_eer

# Ngưỡng tương đương môi trường đăng ký trước theo D033 mục 4 (không nhận từ dòng lệnh):
# 1. |ΔEER| <= 0,02 điểm % (tức 0.02 trên thang phần trăm [0, 100])
# 2. Số mẫu đổi quyết định tại decision_threshold của report tham chiếu <= 10
# 3. max |Δ score| <= 0,05 theo đơn vị score đã lưu
MAX_ABS_DELTA_EER_PCT_POINTS: float = 0.02
MAX_DECISION_FLIPS: int = 10
MAX_ABS_DELTA_SCORE: float = 0.05


def resolve_checkpoint_files(checkpoint_path: Path, generation: str | None = None) -> tuple[Path, Path | None, str | None]:
    """Xác định đường dẫn checkpoint.pt và state.json từ file, thư mục generation hoặc run_root."""
    resolved = checkpoint_path.resolve()
    if resolved.is_file():
        state_path = resolved.parent / 'state.json'
        gen_name = resolved.parent.name if resolved.parent.name.startswith('g-') else generation
        return resolved, (state_path if state_path.is_file() else None), gen_name
    if resolved.is_dir():
        if (resolved / 'checkpoint.pt').is_file():
            state_path = resolved / 'state.json'
            gen_name = resolved.name if resolved.name.startswith('g-') else generation
            return resolved / 'checkpoint.pt', (state_path if state_path.is_file() else None), gen_name
        latest_json = resolved / 'LATEST.json'
        if latest_json.is_file():
            pointer = json.loads(latest_json.read_text(encoding='utf-8'))
            target_gen = generation
            if target_gen is None:
                best_ref = pointer.get('best')
                latest_ref = pointer.get('latest')
                target_gen = (best_ref or latest_ref or {}).get('generation')
            if not isinstance(target_gen, str) or not target_gen.startswith('g-'):
                raise ValueError('Không xác định được generation hợp lệ từ LATEST.json.')
            gen_dir = resolved / target_gen
            ckpt = gen_dir / 'checkpoint.pt'
            state = gen_dir / 'state.json'
            if not ckpt.is_file():
                raise ValueError(f'Không tìm thấy checkpoint.pt tại {ckpt}.')
            return ckpt, (state if state.is_file() else None), target_gen
    raise ValueError(f'Đường dẫn checkpoint không hợp lệ: {checkpoint_path}')


def load_trainable_into_model(model: torch.nn.Module, saved_model: Mapping[str, Any]) -> None:
    parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if set(saved_model) != set(parameters):
        raise ValueError('Danh sách tham số trainable không khớp checkpoint.')
    with torch.no_grad():
        for name, param in parameters.items():
            tensor = saved_model[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f'Tham số checkpoint không phải Tensor: {name}')
            if tensor.shape != param.shape or tensor.dtype != param.dtype:
                raise ValueError(f'Tensor trainable không tương thích checkpoint: {name}')
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Tensor trainable chứa NaN/Inf: {name}')
            param.copy_(tensor.to(param.device))


def compare_parity_scores(
    reference_development: Mapping[str, Any],
    evaluated_records: Sequence[Mapping[str, Any]],
    *,
    hardware: Mapping[str, Any],
    reference_generation: str | None = None,
    evaluated_generation: str | None = None,
) -> dict[str, Any]:
    """Đối chiếu score dev-full mới với report tham chiếu theo đúng thứ tự mẫu và ba ngưỡng D033 mục 4."""
    ref_records = reference_development.get('records')
    if not isinstance(ref_records, list) or not ref_records:
        raise ValueError('Report tham chiếu thiếu controller.best.development.records hợp lệ.')
    raw_ref_scores = reference_development.get('scores')
    if raw_ref_scores is None:
        ref_scores = [float(item['score']) for item in ref_records]
    else:
        if not isinstance(raw_ref_scores, list):
            raise ValueError('controller.best.development.scores phải là danh sách.')
        ref_scores = [float(val) for val in raw_ref_scores]
    if len(ref_records) != len(ref_scores) or len(evaluated_records) != len(ref_records):
        raise ValueError(
            f'Số lượng mẫu không khớp: tham chiếu={len(ref_records)}, scores={len(ref_scores)}, đánh giá mới={len(evaluated_records)}.'
        )
    if 'decision_threshold' not in reference_development:
        raise ValueError('Report tham chiếu thiếu decision_threshold.')
    decision_threshold = float(reference_development['decision_threshold'])
    if not math.isfinite(decision_threshold):
        raise ValueError('decision_threshold trong report tham chiếu không hữu hạn.')

    labels: list[int] = []
    new_scores: list[float] = []
    abs_deltas: list[float] = []
    decision_flips = 0
    for idx, (ref_item, cur_item, ref_score) in enumerate(zip(ref_records, evaluated_records, ref_scores, strict=True)):
        ref_file = ref_item.get('file')
        cur_file = cur_item.get('file')
        if not isinstance(ref_file, str) or not ref_file or ref_file != cur_file:
            raise ValueError(
                f'Thứ tự mẫu lệch report tham chiếu tại vị trí {idx}: {cur_file!r} != {ref_file!r}.'
            )
        ref_label = int(ref_item['label'])
        cur_label = int(cur_item['label'])
        if ref_label != cur_label:
            raise ValueError(f'Nhãn mẫu lệch report tham chiếu tại vị trí {idx} ({ref_file}).')
        if 'score' in ref_item and abs(float(ref_item['score']) - ref_score) > 1e-12:
            raise ValueError(f'Score trong records và scores của report tham chiếu không khớp tại vị trí {idx}.')
        cur_score = float(cur_item['score'])
        if not math.isfinite(cur_score) or not math.isfinite(ref_score):
            raise ValueError(f'Score không hữu hạn tại vị trí {idx} ({ref_file}).')
        labels.append(cur_label)
        new_scores.append(cur_score)
        abs_deltas.append(abs(cur_score - ref_score))
        ref_decision = ref_score >= decision_threshold
        cur_decision = cur_score >= decision_threshold
        if ref_decision != cur_decision:
            decision_flips += 1

    ref_eer, ref_eer_threshold = fast_eer(labels, ref_scores)
    new_eer, new_eer_threshold = fast_eer(labels, new_scores)
    ref_eer_pct = ref_eer * 100.0
    new_eer_pct = new_eer * 100.0
    abs_delta_eer_pct_points = abs(new_eer_pct - ref_eer_pct)
    max_abs_delta_score = max(abs_deltas)

    # Trích D033 mục 4: đạt khi đồng thời |ΔEER| <= 0,02 điểm %, đổi quyết định <= 10, max |Δ score| <= 0,05.
    checks = {
        'abs_delta_eer_pct_points_ok': abs_delta_eer_pct_points <= MAX_ABS_DELTA_EER_PCT_POINTS + 1e-12,
        'decision_flips_ok': decision_flips <= MAX_DECISION_FLIPS,
        'max_abs_delta_score_ok': max_abs_delta_score <= MAX_ABS_DELTA_SCORE + 1e-12,
    }
    passed = all(checks.values())
    return {
        'decision': 'D033',
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'use_test': False,
        'passed': passed,
        'reference_generation': reference_generation,
        'evaluated_generation': evaluated_generation,
        'sample_count': len(labels),
        'decision_threshold': decision_threshold,
        'reference_eer': ref_eer,
        'reference_eer_pct': ref_eer_pct,
        'reference_eer_threshold': ref_eer_threshold,
        'new_eer': new_eer,
        'new_eer_pct': new_eer_pct,
        'new_eer_threshold': new_eer_threshold,
        'abs_delta_eer': abs(new_eer - ref_eer),
        'abs_delta_eer_pct_points': abs_delta_eer_pct_points,
        'decision_flips': decision_flips,
        'max_abs_delta_score': max_abs_delta_score,
        'thresholds_d033_item4': {
            'max_abs_delta_eer_pct_points': MAX_ABS_DELTA_EER_PCT_POINTS,
            'max_decision_flips': MAX_DECISION_FLIPS,
            'max_abs_delta_score': MAX_ABS_DELTA_SCORE,
        },
        'checks': checks,
        'hardware': dict(hardware),
    }


def render_parity_markdown(report: Mapping[str, Any]) -> str:
    status_text = 'ĐẠT' if report.get('passed') else 'KHÔNG ĐẠT'
    hw = report.get('hardware') or {}
    checks = report.get('checks') or {}
    lines = [
        '# Báo cáo kiểm tương đương môi trường XLS-R (D033 mục 4)',
        '',
        f'- **Kết luận:** **{status_text}**',
        f'- **Thời điểm kiểm (UTC):** `{report.get("checked_utc")}`',
        f'- **Số mẫu dev-full:** `{report.get("sample_count")}` (`use_test = {report.get("use_test")}`)',
        f'- **Generation tham chiếu:** `{report.get("reference_generation")}`',
        f'- **Generation đánh giá:** `{report.get("evaluated_generation")}`',
        f'- **GPU đánh giá:** `{hw.get("gpu_status")}` (`device = {hw.get("device")}`, `torch = {hw.get("torch_version")}`, `cuda = {hw.get("cuda_version")}`)',
        '',
        '## Kết quả so với ngưỡng D033 mục 4',
        '',
        '| Tiêu chí | Giá trị đo được | Ngưỡng D033 mục 4 | Kết quả |',
        '|---|---:|---:|---|',
        f'| `|ΔEER|` (điểm %) | `{report.get("abs_delta_eer_pct_points"):.6f}` | `<= {MAX_ABS_DELTA_EER_PCT_POINTS:.2f}` | `{"ĐẠT" if checks.get("abs_delta_eer_pct_points_ok") else "KHÔNG ĐẠT"}` |',
        f'| Số mẫu đổi quyết định tại `decision_threshold` (`{report.get("decision_threshold")}`) | `{report.get("decision_flips")}` | `<= {MAX_DECISION_FLIPS}` | `{"ĐẠT" if checks.get("decision_flips_ok") else "KHÔNG ĐẠT"}` |',
        f'| `max |Δ score|` | `{report.get("max_abs_delta_score"):.6f}` | `<= {MAX_ABS_DELTA_SCORE:.2f}` | `{"ĐẠT" if checks.get("max_abs_delta_score_ok") else "KHÔNG ĐẠT"}` |',
        '',
        '## Chi tiết EER',
        '',
        f'- **EER tham chiếu:** `{report.get("reference_eer_pct"):.6f}%` (tỷ lệ `{report.get("reference_eer"):.8f}`)',
        f'- **EER mới:** `{report.get("new_eer_pct"):.6f}%` (tỷ lệ `{report.get("new_eer"):.8f}`)',
        '',
    ]
    return '\n'.join(lines)


def write_parity_reports(report: Mapping[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
        newline='\n',
    )
    output_md.write_text(
        render_parity_markdown(report),
        encoding='utf-8',
        newline='\n',
    )


def evaluate_checkpoint_for_parity(
    *,
    plan_path: Path,
    checkpoint_path: Path,
    reference_report_path: Path,
    dev_artifact_path: Path | None,
    pretrained_dir: Path,
    device_str: str,
    expected_gpu: str | None = None,
    generation: str | None = None,
) -> dict[str, Any]:
    for candidate_path in (checkpoint_path, reference_report_path, pretrained_dir, dev_artifact_path):
        if candidate_path is not None and 'test' in candidate_path.name.lower() and 'contest' not in candidate_path.name.lower():
            raise ValueError(f'Từ chối đường dẫn có dấu hiệu test: {candidate_path}')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('data', {}).get('use_test') is not False:
        raise ValueError('Cấu hình plan không hợp lệ: use_test phải là false.')
    device = torch.device(device_str)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hardware = runner.collect_hardware_info(device)
    runner.verify_expected_and_resume_hardware(
        hardware,
        expected_gpu=expected_gpu,
        resume=False,
        run_root=None,
    )
    ckpt_file, state_file, eval_gen = resolve_checkpoint_files(checkpoint_path, generation=generation)
    payload = torch.load(ckpt_file, map_location='cpu', weights_only=True)
    if payload.get('checkpoint_version') != 4:
        raise ValueError('Checkpoint kiểm tương đương phải có checkpoint_version=4.')
    if state_file is not None:
        state_obj = json.loads(state_file.read_text(encoding='utf-8'))
        if payload.get('controller') != state_obj:
            raise ValueError('Controller trong checkpoint.pt không khớp state.json.')
    identity = payload.get('identity')
    if not isinstance(identity, dict):
        raise ValueError('Checkpoint thiếu khối identity.')
    run_id = str(identity['run_id'])
    seed = int(identity['training_seed'])
    run = runner.validate_plan(plan, run_id, seed)
    runner.verify_pretrained(pretrained_dir, plan)

    reference_report = json.loads(reference_report_path.read_text(encoding='utf-8'))
    if reference_report.get('use_test') is not False:
        raise ValueError('Report tham chiếu không hợp lệ: use_test phải là false.')
    ref_best = (reference_report.get('controller') or {}).get('best')
    if not isinstance(ref_best, dict) or not isinstance(ref_best.get('development'), dict):
        raise ValueError('Report tham chiếu thiếu controller.best.development.')
    ref_gen = ref_best.get('generation')
    if generation is not None and ref_gen is not None and generation != ref_gen:
        raise ValueError(f'Generation yêu cầu ({generation}) khác generation best tham chiếu ({ref_gen}).')

    dev_dir = dev_artifact_path or (ROOT / plan['data']['development_artifact'])
    dev = runner.open_artifact(dev_dir, plan, training=False, seed=seed)
    try:
        backbone = runner.load_xlsr_backbone(pretrained_dir)
        model = runner.build_model(backbone, run, seed).to(device)
        load_trainable_into_model(model, payload['model'])
        evaluation = runner.evaluate(model, dev, device, batch_size=16)
    finally:
        dev.close()

    return compare_parity_scores(
        ref_best['development'],
        evaluation['records'],
        hardware=hardware,
        reference_generation=ref_gen,
        evaluated_generation=eval_gen,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/xlsr_training.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--generation', type=str)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--dev-artifact', type=Path)
    parser.add_argument('--pretrained-dir', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--expected-gpu', type=str)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_checkpoint_for_parity(
        plan_path=args.plan,
        checkpoint_path=args.checkpoint,
        reference_report_path=args.reference_report,
        dev_artifact_path=args.dev_artifact,
        pretrained_dir=args.pretrained_dir,
        device_str=args.device,
        expected_gpu=args.expected_gpu,
        generation=args.generation,
    )
    write_parity_reports(report, args.output_json, args.output_md)
    print(json.dumps({'passed': report['passed'], 'output_json': str(args.output_json), 'output_md': str(args.output_md)}, ensure_ascii=False))
    if not report['passed']:
        raise ValueError('Kiểm tương đương môi trường không đạt ngưỡng D033 mục 4.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f'LỖI: {error}', file=sys.stderr)
        raise SystemExit(1) from error
