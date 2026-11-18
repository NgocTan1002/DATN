"""Kiểm thử đơn vị cho scripts/audit_shortcut_features.py dùng fixture tổng hợp nhỏ."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from scripts import audit_shortcut_features as sc


class ShortcutFeatureAuditTests(unittest.TestCase):
    def test_frame_energy_and_leading_trailing_silence_on_full(self) -> None:
        # 1,6 giây = 25.600 mẫu: 0,20s im lặng đầu + 1,00s sin 1kHz + 0,40s im lặng cuối
        n_lead = 3200  # 0.20s
        n_mid = 16000  # 1.00s
        n_tail = 6400  # 0.40s
        t = np.arange(n_mid, dtype=np.float32) / sc.SAMPLE_RATE
        mid = 0.95 * np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        wave = np.concatenate([
            np.zeros(n_lead, dtype=np.float32),
            mid,
            np.zeros(n_tail, dtype=np.float32),
        ])

        energies = sc.compute_frame_energies_db(wave)
        expected_frames = (wave.size - sc.FRAME_LEN) // sc.FRAME_HOP + 1
        self.assertEqual(energies.shape, (expected_frames,))

        feats = sc.compute_region_features(wave)
        # Khung đầu bắt đầu chứa phần sin ở start = 3200 - 400 + 160 -> khoảng 18 khung (0.18s) lặng hoàn toàn
        self.assertGreaterEqual(feats["leading_silence_s"], 0.17)
        self.assertLessEqual(feats["leading_silence_s"], 0.20)
        # Khung cuối hoàn toàn lặng từ mẫu 19200 -> ~0.38s - 0.40s
        self.assertGreaterEqual(feats["trailing_silence_s"], 0.37)
        self.assertLessEqual(feats["trailing_silence_s"], 0.40)
        self.assertGreater(feats["silence_ratio"], 0.3)
        self.assertLess(feats["noise_floor_db"], -40.0)

    def test_crop_marks_padded_tail_as_silent_even_with_repeat_then_trim(self) -> None:
        # Mẫu ngắn 32.000 mẫu (2 giây), toàn bộ là sin 1 kHz biên độ đều (không có khung lặng tự nhiên)
        n = 32000
        t = np.arange(n, dtype=np.float32) / sc.SAMPLE_RATE
        raw = 0.8 * np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        feats = sc.extract_sample_features(raw, num_samples=n)

        self.assertTrue(feats["is_padded"])
        self.assertAlmostEqual(feats["duration_s"], 2.0, places=6)
        self.assertAlmostEqual(feats["pad_fraction"], 0.5, places=6)
        # Vùng full không có khung lặng
        self.assertEqual(feats["full_silence_ratio"], 0.0)
        self.assertEqual(feats["full_leading_silence_s"], 0.0)
        self.assertEqual(feats["full_trailing_silence_s"], 0.0)
        # Vùng crop lặp 32.000->64.000 nhưng phần pad >= 32.000 phải được tính là lặng
        self.assertGreater(feats["crop_silence_ratio"], 0.48)
        self.assertLess(feats["crop_silence_ratio"], 0.52)
        self.assertEqual(feats["crop_leading_silence_s"], 0.0)
        self.assertGreater(feats["crop_trailing_silence_s"], 1.95)

    def test_train_threshold_applied_verbatim_to_dev_and_direction_selection(self) -> None:
        # Trên train: bonafide quanh 1.0, spoof quanh 0.2 (lower_is_spoof)
        train_vals = np.array([0.9, 1.0, 1.1, 1.2, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        train_y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        # Trên dev: bonafide dịch xuống 0.65 (nếu chọn lại threshold trên dev sẽ đổi, nhưng áp threshold train thì có lỗi)
        dev_vals = np.array([0.65, 0.70, 0.95, 1.05, 0.15, 0.25, 0.35, 0.45], dtype=np.float64)
        dev_y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

        res, dev_scores = sc.evaluate_single_feature(train_vals, train_y, dev_vals, dev_y)
        self.assertEqual(res["direction"], "lower_is_spoof")
        self.assertEqual(res["direction_sign"], -1)
        self.assertEqual(res["train_eer"], 0.0)
        self.assertTrue(np.array_equal(dev_scores, -dev_vals))
        # Threshold train là khoảng -0.65 (tức raw ~ 0.65): mẫu dev 0.65 -> -0.65 >= -0.65 là FP hoặc gần biên
        # Kiểm chứng exact: áp nguyên train_threshold_signed lên dev_scores khớp confusion_at_threshold
        expected_fpr = sum((y == 0) and (s >= res["train_threshold_signed"]) for y, s in zip(dev_y, dev_scores)) / 4.0
        self.assertEqual(res["dev_fpr_bonafide"], expected_fpr)

    def test_severity_classification_boundaries_10_and_25_percent(self) -> None:
        self.assertEqual(sc.classify_severity(0.0), "mạnh")
        self.assertEqual(sc.classify_severity(0.10), "mạnh")
        self.assertEqual(sc.classify_severity(0.100001), "trung bình")
        self.assertEqual(sc.classify_severity(0.25), "trung bình")
        self.assertEqual(sc.classify_severity(0.250001), "yếu")
        self.assertEqual(sc.classify_severity(0.50), "yếu")
        with self.assertRaises(ValueError):
            sc.classify_severity(-0.01)

    def test_rejects_closed_test_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "dữ liệu test"):
            sc.reject_test_path(Path("data/manifests/closed_test_full_v1.csv"))


if __name__ == "__main__":
    unittest.main()
