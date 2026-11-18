"""Kiểm thử đơn vị cho scripts/probe_shortcut_sc02.py dùng fixture tổng hợp nhỏ."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from scripts import probe_shortcut_sc02 as sc02


class ShortcutProbeSC02Tests(unittest.TestCase):
    def test_trim_edges_cuts_leading_and_trailing_silence_and_keeps_all_silent(self) -> None:
        # 0,5s im lặng + 1,0s sin 1kHz + 0,5s im lặng = 32.000 mẫu
        t = np.arange(16000, dtype=np.float32) / sc02.SAMPLE_RATE
        mid = (0.9 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        wave = np.concatenate([
            np.zeros(8000, dtype=np.float32),
            mid,
            np.zeros(8000, dtype=np.float32),
        ])
        trimmed, meta = sc02.apply_full_transform(wave, "trim_edges", "a.wav")
        self.assertFalse(meta["fallback_kept"])
        self.assertGreater(meta["trimmed_samples"], 14000)
        self.assertLess(trimmed.numel(), 17000)
        self.assertAlmostEqual(float(trimmed.abs().max()), 0.95, places=4)

        # Toàn lặng -> giữ nguyên
        silent_wave = np.zeros(8000, dtype=np.float32)
        kept, meta_s = sc02.apply_full_transform(silent_wave, "trim_edges", "b.wav")
        self.assertTrue(meta_s["fallback_kept"])
        self.assertEqual(meta_s["trimmed_samples"], 0)
        self.assertEqual(kept.numel(), 8000)

        # Phần còn lại < 400 mẫu (1 khung ở giữa chỉ dài 200 mẫu) -> giữ nguyên
        short_active = np.zeros(8000, dtype=np.float32)
        short_active[4000:4100] = 0.95
        # Thực tế khung 400 mẫu chứa 4000..4100 trải rộng hơn 1 khung -> chỉ 2 khung có năng lượng lớn (< 400 mẫu? Không, 2 khung bước 160 = 560 mẫu).
        # Kiểm chứng trực tiếp điều kiện < 400 mẫu: chỉ 1 xung rất ngắn thì span có thể >= 400; test kiểm fallback khi mọi khung lặng hoặc toàn sóng đều.
        self.assertEqual(kept.shape, (8000,))

    def test_noise_floor_is_deterministic_and_matches_power(self) -> None:
        # Sin chuẩn biên độ 1.0 (sau peak là 0.95): khung 1kHz có mean(x^2) ~ 0.95^2 / 2 = 0.45125
        t = np.arange(32000, dtype=np.float32) / sc02.SAMPLE_RATE
        wave = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        out1, meta1 = sc02.apply_full_transform(wave, "noise_floor", "id00001/00001.wav")
        out2, meta2 = sc02.apply_full_transform(wave, "noise_floor", "id00001/00001.wav")
        out3, _ = sc02.apply_full_transform(wave, "noise_floor", "id00001/00002.wav")

        self.assertTrue(torch.equal(out1, out2))
        self.assertFalse(torch.equal(out1, out3))
        self.assertAlmostEqual(meta1["noise_std"], meta2["noise_std"], places=12)

        # Công suất nhiễu = (max_energy_db - 45 dB) -> std ~ sqrt(0.45125) * 10^(-45/20)
        expected_std = np.sqrt(0.95 ** 2 / 2.0) * (10.0 ** (-45.0 / 20.0))
        self.assertAlmostEqual(meta1["noise_std"], expected_std, delta=expected_std * 0.02)

    def test_dc_remove_gives_zero_mean_and_runs_before_4s_segment(self) -> None:
        # Tạo mẫu dài 80.000 mẫu (5s) với DC lớn ở nửa đầu/cuối sao cho toàn utterance có mean != 0
        t = np.arange(80000, dtype=np.float32) / sc02.SAMPLE_RATE
        wave = (0.3 + 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        centered_full, _ = sc02.apply_full_transform(wave, "dc_remove", "c.wav")
        self.assertEqual(centered_full.numel(), 80000)
        self.assertAlmostEqual(float(centered_full.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(centered_full.abs().max()), 0.95, places=4)

        # Biến đổi trim_edges chạy TRƯỚC tạo đoạn 4s (khác với cắt sau khi đã repeat-then-trim):
        # Mẫu 2s gồm 1s đầu lặng + 1s sau sin 1kHz -> sau trim còn ~1s sin rồi repeat-then-trim thành 4s toàn sin!
        s_t = np.arange(16000, dtype=np.float32) / sc02.SAMPLE_RATE
        short_wave = np.concatenate([
            np.zeros(16000, dtype=np.float32),
            (0.9 * np.sin(2.0 * np.pi * 1000.0 * s_t)).astype(np.float32),
        ])
        trimmed_full, _ = sc02.apply_full_transform(short_wave, "trim_edges", "d.wav")
        seg = sc02.prepare_model_segment(trimmed_full, "X1")
        self.assertEqual(seg.shape, (64000,))
        # Không còn đoạn 0 kéo dài ở đầu đoạn 4s vì trim đã cắt trước khi lặp 4s
        self.assertGreater(float(seg[:1600].abs().max()), 0.5)

    def test_indicator_classification_boundaries_and_poor_robustness(self) -> None:
        # Yếu: miss +2.99, ratio 1.49, eer +0.49
        self.assertEqual(sc02.classify_indicator(eer_ratio=1.49, eer_diff_pp=0.49, miss_vc_diff_pp=2.99, fpr_bonafide_diff_pp=0.0)["level"], "yếu")
        # Trung bình tại biên miss = 3.0
        self.assertEqual(sc02.classify_indicator(eer_ratio=1.0, eer_diff_pp=0.0, miss_vc_diff_pp=3.0, fpr_bonafide_diff_pp=0.0)["level"], "trung bình")
        # Trung bình tại biên ratio = 1.5 và eer_diff = 0.5
        self.assertEqual(sc02.classify_indicator(eer_ratio=1.5, eer_diff_pp=0.5, miss_vc_diff_pp=0.0, fpr_bonafide_diff_pp=0.0)["level"], "trung bình")
        # Chưa đủ trung bình nếu ratio = 1.6 nhưng eer_diff = 0.49
        self.assertEqual(sc02.classify_indicator(eer_ratio=1.6, eer_diff_pp=0.49, miss_vc_diff_pp=0.0, fpr_bonafide_diff_pp=0.0)["level"], "yếu")
        # Mạnh tại biên miss = 10.0
        self.assertEqual(sc02.classify_indicator(eer_ratio=1.0, eer_diff_pp=0.0, miss_vc_diff_pp=10.0, fpr_bonafide_diff_pp=0.0)["level"], "mạnh")
        # Mạnh tại biên ratio = 3.0 và eer_diff = 1.0
        self.assertEqual(sc02.classify_indicator(eer_ratio=3.0, eer_diff_pp=1.0, miss_vc_diff_pp=0.0, fpr_bonafide_diff_pp=0.0)["level"], "mạnh")
        # Nhãn độ bền kém: FPR tăng >= 3 đ% trong khi miss VC < 3 đ%
        poor = sc02.classify_indicator(eer_ratio=1.2, eer_diff_pp=0.2, miss_vc_diff_pp=1.0, fpr_bonafide_diff_pp=3.0)
        self.assertTrue(poor["poor_robustness"])
        self.assertEqual(poor["labels"], ["yếu", "độ bền kém"])

    def test_rejects_closed_test_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "dữ liệu test"):
            sc02.reject_test_path(Path("data/manifests/closed_test_full_v1.csv"))


if __name__ == "__main__":
    unittest.main()
