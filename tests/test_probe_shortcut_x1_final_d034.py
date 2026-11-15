"""D034 trên score thật đã lưu và dữ liệu tổng hợp; không chấm mô hình."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import probe_shortcut_x1_final as original
from scripts import probe_shortcut_x1_final_d034 as wrapper


class ReproductionD034Tests(unittest.TestCase):
    def test_real_original_blocks_old_fails_d034_passes(self) -> None:
        identity = original.shared.read_json(original.CACHE / 'identity.json')
        self.assertEqual(original.shared.sha256_file(original.REFERENCE_REPORT),
                         identity['models'][original.MODEL]['reference_report_sha256'])
        reference = original.shared.read_json(original.REFERENCE_REPORT)
        saved = {r['file']: r['score'] for r in reference['controller']['best']['development']['records']}
        files = sorted((original.CACHE / 'blocks' / original.MODEL / 'original').glob('*.json'))
        self.assertEqual(len(files), 16)
        store = original.shared.BlockStore(original.CACHE / 'blocks', identity)
        records: list[dict] = []
        for path in files:
            block = original.shared.read_json(path)
            value = store.load(original.MODEL, 'original', block['start'], [r['file'] for r in block['records']])
            self.assertIsNotNone(value)
            records.extend(value['records'])
        self.assertEqual(len(records), 3120)
        raw = [r['raw_logit'] for r in records]
        reference_scores = [saved[r['file']] for r in records]
        observed = [r['score'] for r in records]
        old = original.check_logit_reproduction(raw, reference_scores, observed_scores=observed)
        self.assertFalse(old['passed'])
        self.assertEqual(old['max_abs_logit_difference'], 0.06893052922133514)
        for count in (200, 3120):
            with self.subTest(count=count):
                result = wrapper.check_logit_reproduction_d034(raw[:count], reference_scores[:count], observed_scores=observed[:count])
                self.assertTrue(result['passed'])
                self.assertLess(result['max_excess_over_tolerance'], 0)
                self.assertEqual(result['decision_flips'], 0)
                self.assertEqual(result['flips_outside_0_05_margin'], 0)
                self.assertEqual(result['criterion'], 'D034')
                for key, value in old.items():
                    if key != 'passed' and count == 3120:
                        self.assertEqual(result[key], value)

    def test_central_score_logit_difference_006_is_rejected(self) -> None:
        value = wrapper.check_logit_reproduction_d034([.06], [.5])
        self.assertFalse(value['passed'])
        self.assertGreater(value['max_excess_over_tolerance'], 0)

    def test_saturated_float32_score_logit_difference_006_is_accepted(self) -> None:
        score = float(np.float32(.9999990))
        value = wrapper.check_logit_reproduction_d034([float(original.shared.stable_logit([score])[0]) + .06], [score])
        self.assertTrue(value['passed'])
        self.assertEqual(value['samples_over_0_05_without_gap'], 1)
        p = np.float32(score)
        neighbors = [float(np.nextafter(p, np.float32(0))), float(np.nextafter(p, np.float32(1)))]
        expected = max(abs(float(original.shared.stable_logit([n])[0] - original.shared.stable_logit([score])[0])) for n in neighbors)
        self.assertEqual(value['max_float32_gap_logit'], expected)

    def test_observed_flip_outside_margin_rejected_even_with_small_raw_difference(self) -> None:
        logit = float(original.shared.stable_logit([.75])[0])
        result = wrapper.check_logit_reproduction_d034([logit + .001], [.75], .5, observed_scores=[.49])
        self.assertFalse(result['passed'])
        self.assertLess(result['max_excess_over_tolerance'], 0)
        self.assertEqual(result['flips_outside_0_05_margin'], 1)

    def test_float64_only_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'không biểu diễn chính xác bằng float32'):
            wrapper.check_logit_reproduction_d034([0.], [.1])

    def test_invalid_logits_and_shapes_are_rejected(self) -> None:
        for raw, reference in (([], []), ([float('nan')], [.5]), ([0., 1.], [.5])):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                wrapper.check_logit_reproduction_d034(raw, reference)

    def test_flip_inside_original_margin_remains_accepted(self) -> None:
        boundary = float(original.shared.stable_logit([original.THRESHOLD])[0])
        score = float(np.float32(1 / (1 + np.exp(-(boundary - .01)))))
        result = wrapper.check_logit_reproduction_d034([boundary + .01], [score])
        self.assertTrue(result['passed'])
        self.assertEqual(result['decision_flips'], 1)
        self.assertEqual(result['flips_outside_0_05_margin'], 0)


class WrapperTests(unittest.TestCase):
    def test_main_preserves_runner_and_adds_only_rule_for_every_exit_code(self) -> None:
        before = Path(original.__file__).read_bytes()
        identity = (original.CACHE / 'identity.json').read_bytes()
        prior_argv, prior_check = sys.argv, original.check_logit_reproduction
        for code in (0, 1, 2, 3):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'identity.json').write_bytes(identity)
                report = {'status': 'giả lập', 'reproduction': {'first200': {'passed': True}}, 'unchanged': [1, 2]}
                report_json, report_md = root / 'report.json', root / 'report.md'
                report_json.write_text(json.dumps(report), encoding='utf-8')
                report_md.write_text('Nội dung gốc\n', encoding='utf-8')

                def fake_main() -> int:
                    self.assertIs(original.check_logit_reproduction, wrapper.check_logit_reproduction_d034)
                    self.assertEqual(sys.argv, [str(Path(original.__file__)), '--resume', '--resume-reason', wrapper.RESUME_REASON])
                    return code

                with patch.object(original, 'CACHE', root), patch.object(original, 'REPORT_JSON', report_json), patch.object(original, 'REPORT_MD', report_md), patch.object(original, 'main', side_effect=fake_main):
                    self.assertEqual(wrapper.main(), code)
                actual = json.loads(report_json.read_text(encoding='utf-8'))
                rule = actual.pop('reproduction_rule')
                self.assertEqual(actual, report)
                self.assertEqual(rule, wrapper.reproduction_rule())
                self.assertEqual(set(rule), {'decision', 'wrapper', 'wrapper_sha256', 'runner_unchanged_sha256', 'identity_unchanged_sha256', 'note'})
                self.assertEqual(report_md.read_text(encoding='utf-8'), 'Nội dung gốc\n\nreproduction_rule: ' + original.shared.canonical(rule).decode('utf-8') + '\n')
                self.assertFalse(list(root.glob('*.tmp')))
                self.assertIs(sys.argv, prior_argv)
                self.assertIs(original.check_logit_reproduction, prior_check)
                self.assertEqual(Path(original.__file__).read_bytes(), before)
                self.assertEqual(original.shared.sha256_file(Path(original.__file__)), wrapper.RUNNER_SHA)


if __name__ == '__main__':
    unittest.main()
