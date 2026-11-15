from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np

from scripts import analyze_highband_spectrum as a


class HighbandSpectrumTests(unittest.TestCase):
    def test_sines(self) -> None:
        t = np.arange(64000)/16000
        bands, _ = a.spectrum(np.sin(2*np.pi*7900*t))
        self.assertEqual(int(np.argmax(bands)), 3)
        self.assertAlmostEqual(bands[3], 0, places=6)
        low, _ = a.spectrum(np.sin(2*np.pi*3000*t))
        self.assertTrue(bool((low < -100).all()))

    def test_boundaries(self) -> None:
        f = np.array([5999,6000,6999,7000,7499,7500,7799,7800,7949,7950,8000,8001])
        masks = np.stack(a.band_masks(f))
        np.testing.assert_array_equal(masks.sum(axis=0), ((f>=6000)&(f<=8000)).astype(int))
        for value, index in ((6000,0),(7000,1),(7500,2),(7800,3),(7950,4),(8000,4)):
            self.assertTrue(masks[index, np.where(f==value)[0][0]])

    def test_subset_checksum_and_determinism(self) -> None:
        rows = a.read_dev_manifest(a.ROOT/'data/manifests/closed_dev_full_v1.csv')
        selected = a.subset_indices(rows)
        self.assertEqual(selected, a.subset_indices(rows))
        self.assertEqual(a.file_list_sha([rows[i]['file'] for i in selected]), a.SUBSET_SHA)
        reverse = list(reversed(rows))
        self.assertEqual([rows[i]['file'] for i in selected], [reverse[i]['file'] for i in a.subset_indices(reverse)])

    def test_reject_test_before_open(self) -> None:
        for function in (a.read_dev_manifest, a.open_dev_artifact):
            with self.assertRaisesRegex(ValueError, 'test'):
                function(Path('closed_test_missing'))

    def test_reject_test_split(self) -> None:
        with patch('scripts.probe_shortcut_dev.read_json', return_value={'split':'closed_test'}):
            with self.assertRaisesRegex(ValueError, 'closed_dev'):
                a.open_dev_artifact(Path('dev_fixture'))

    def test_invalid_waveforms(self) -> None:
        for x in (np.zeros(64000),np.zeros(64),np.full(64000,np.nan)):
            with self.assertRaises(ValueError):
                a.spectrum(x)

    def test_svg_models_and_groups(self) -> None:
        report = {'groups':{g:{'mean_log_power_db':[-40.0]*80} for g in a.LABELS}}
        svg = a.spectrum_svg(report)
        ET.fromstring(svg)
        for label in a.LABELS.values():
            self.assertIn(label, svg)
        probe = json.loads((a.ROOT/'reports/shortcut_probe_dev.json').read_text(encoding='utf-8'))
        svg = a.ablation_svg(probe)
        ET.fromstring(svg)
        for model in (*a.BMODELS,a.XMODEL):
            self.assertIn(model,svg)
        self.assertIn('3.120',svg)
        self.assertIn('38.777',svg)

    def test_ablation_has_discrete_markers_without_connectors(self) -> None:
        probe = json.loads((a.ROOT/'reports/shortcut_probe_dev.json').read_text(encoding='utf-8'))
        root = ET.fromstring(a.ablation_svg(probe))
        ns = {'s':'http://www.w3.org/2000/svg'}
        self.assertEqual(root.findall('.//s:polyline',ns), [])
        series = root.findall('.//s:g[@class="data-series"]',ns)
        self.assertEqual(len(series),4)
        shapes = []
        for group, model in zip(series, (*a.BMODELS,a.XMODEL)):
            points = group.findall('s:g[@class="data-point"]',ns)
            self.assertEqual(len(points),4)
            self.assertEqual(group.findall('.//s:path',ns), [])
            shapes.append(ET.tostring(points[0][0],encoding='unicode'))
            for point, condition in zip(points,a.CONDITIONS):
                y = float(point.attrib['transform'].split()[1].rstrip(')'))
                eer = probe['results'][model]['conditions'][condition]['overall_eer']
                self.assertAlmostEqual(y,480-360*(eer*100)/60,places=2)
        self.assertEqual(len(set(shapes)),4)

    def test_spectrum_axis_fits_actual_data(self) -> None:
        report = json.loads((a.ROOT/'reports/highband_spectrum_dev.json').read_text(encoding='utf-8'))
        root = ET.fromstring(a.spectrum_svg(report))
        ns = {'s':'http://www.w3.org/2000/svg'}
        ticks = [int(t.text) for t in root.findall('s:text',ns) if t.attrib.get('x') == '96.00']
        self.assertEqual(ticks,[-80,-68,-56,-44,-32,-20])
        for line in root.findall('s:polyline',ns):
            self.assertTrue(all(120 <= float(p.split(',')[1]) <= 480 for p in line.attrib['points'].split()))


if __name__ == '__main__':
    unittest.main()
