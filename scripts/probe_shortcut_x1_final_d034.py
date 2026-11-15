"""T22b: tiếp tục runner T22 nguyên trạng với tiêu chí tái tạo D034."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scripts import probe_shortcut_x1_final as runner

shared = runner.shared
THRESHOLD = runner.THRESHOLD
RUNNER_SHA = 'f4660226bef34a036d9be4b9d9c2eaef51d78b813676b75be49a1ef60c0c9fd2'
IDENTITY_SHA = '21feaf6da7933555369620996b2571104e9d368fb2bbdacdd22679d6b2dca12b'
ORIGINAL_CHECK = runner.check_logit_reproduction
RESUME_REASON = 'D034: sửa tiêu chí tái tạo cho score lưu float32; tiếp tục từ giai đoạn đang dừng'


def check_logit_reproduction_d034(
    raw_logits: Sequence[float], reference: Sequence[float], threshold: float = THRESHOLD,
    *, observed_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Giữ kiểm quyết định gốc, cộng độ phân giải float32 vào biên logit."""
    values = np.asarray(reference, dtype=np.float64)
    shared.stable_logit(values)
    shared.require(bool((values.astype(np.float32).astype(np.float64) == values).all()),
                   'Score tham chiếu không biểu diễn chính xác bằng float32; dừng D034.')
    result = ORIGINAL_CHECK(raw_logits, reference, threshold, observed_scores=observed_scores)
    probabilities = values.astype(np.float32)
    below = np.nextafter(probabilities, np.float32(0), dtype=np.float32).astype(np.float64)
    above = np.nextafter(probabilities, np.float32(1), dtype=np.float32).astype(np.float64)
    old = shared.stable_logit(probabilities.astype(np.float64))
    gaps = np.maximum(np.abs(shared.stable_logit(above) - old),
                      np.abs(old - shared.stable_logit(below)))
    differences = np.abs(np.asarray(raw_logits, dtype=np.float64) - old)
    excess = differences - (.05 + gaps)
    result.update(
        passed=bool((excess <= 0).all() and result['flips_outside_0_05_margin'] == 0),
        criterion='D034',
        max_float32_gap_logit=float(gaps.max()),
        max_excess_over_tolerance=float(excess.max()),
        samples_over_0_05_without_gap=int((differences > .05).sum()),
    )
    return result


def reproduction_rule() -> dict[str, str]:
    return {
        'decision': 'D034',
        'wrapper': 'scripts/probe_shortcut_x1_final_d034.py',
        'wrapper_sha256': shared.sha256_file(Path(__file__)),
        'runner_unchanged_sha256': RUNNER_SHA,
        'identity_unchanged_sha256': IDENTITY_SHA,
        'note': 'identity.logit_convention ghi ngưỡng 0,05 theo đặc tả gốc; tiêu chí áp dụng là D034',
    }


def annotate_report() -> None:
    report = shared.read_json(runner.REPORT_JSON)
    rule = reproduction_rule()
    report['reproduction_rule'] = rule
    shared.atomic_json(runner.REPORT_JSON, report)
    with runner.REPORT_MD.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write('\nreproduction_rule: ' + shared.canonical(rule).decode('utf-8') + '\n')


def main() -> int:
    shared.require(shared.sha256_file(Path(runner.__file__)) == RUNNER_SHA,
                   'SHA runner T22 đã đổi; từ chối chạy wrapper D034.')
    shared.require(shared.sha256_file(runner.CACHE / 'identity.json') == IDENTITY_SHA,
                   'SHA identity T22 đã đổi; từ chối chạy wrapper D034.')
    original_check, original_argv = runner.check_logit_reproduction, sys.argv
    runner.check_logit_reproduction = check_logit_reproduction_d034
    sys.argv = [str(Path(runner.__file__)), '--resume', '--resume-reason', RESUME_REASON]
    try:
        code = runner.main()
    finally:
        runner.check_logit_reproduction = original_check
        sys.argv = original_argv
    annotate_report()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
