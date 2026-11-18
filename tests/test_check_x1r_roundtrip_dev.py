"""Kiểm thử đơn vị cho check_x1r_roundtrip_dev.py dùng fixture tổng hợp, không cần checkpoint thật."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import check_x1r_roundtrip_dev as runner
from scripts import probe_shortcut_dev as shared


class CheckX1RRoundtripTests(unittest.TestCase):
    def test_transform_order_applies_roundtrip_before_lowpass(self) -> None:
        # Tạo tín hiệu sin 7.500 Hz (bị bộ lọc lowpass_7k suy giảm ~97,8 dB) + 1.000 Hz.
        t = torch.arange(64000, dtype=torch.float32) / 16000.0
        waveform = 0.5 * torch.sin(2.0 * torch.pi * 1000.0 * t) + 0.5 * torch.sin(2.0 * torch.pi * 7500.0 * t)
        waveform = shared.apply_amplitude_policy(waveform, "peak").waveform

        out_orig = runner.transform_x1r_waveform(waveform, "x1r_original")
        out_rt = runner.transform_x1r_waveform(waveform, "x1r_roundtrip_40k")

        expected_orig = shared.apply_condition(waveform, "lowpass_7k", "bonafide")
        expected_rt = shared.apply_condition(
            shared.apply_condition(waveform, "roundtrip_40k", "bonafide"),
            "lowpass_7k",
            "bonafide",
        )
        reverse_order = shared.apply_condition(
            shared.apply_condition(waveform, "lowpass_7k", "bonafide"),
            "roundtrip_40k",
            "bonafide",
        )
        self.assertTrue(torch.equal(out_orig, expected_orig))
        self.assertTrue(torch.equal(out_rt, expected_rt))
        self.assertFalse(torch.allclose(out_rt, reverse_order, atol=1e-5))

        # Kiểm tra dải 7.500 Hz bị suy giảm mạnh ở cả x1r_original và x1r_roundtrip_40k.
        freqs = torch.fft.rfftfreq(64000, d=1.0 / 16000.0)
        idx_1k = int(torch.argmin(torch.abs(freqs - 1000.0)))
        idx_7500 = int(torch.argmin(torch.abs(freqs - 7500.0)))
        for tensor in (out_orig, out_rt):
            spec = torch.abs(torch.fft.rfft(tensor))
            ratio_db = 20.0 * float(torch.log10(spec[idx_7500] / spec[idx_1k]))
            self.assertLess(ratio_db, -60.0)

    def test_fpr_threshold_0_5_pp_boundary_and_flips(self) -> None:
        files = [f"f_{i:04d}.wav" for i in range(1000)]
        # 13 lỗi trong original (FPR = 1.3%)
        orig_records = [
            {"file": f, "score": 0.9 if i < 13 else 0.01, "raw_logit": 2.2 if i < 13 else -4.6}
            for i, f in enumerate(files)
        ]
        # 18 lỗi trong roundtrip (FPR = 1.8%, hiệu = +0.5 điểm % exact, biên IEEE-754) -> đạt
        rt_pass = [
            {"file": f, "score": 0.9 if i < 18 else 0.01, "raw_logit": 2.2 if i < 18 else -4.6}
            for i, f in enumerate(files)
        ]
        res_pass = runner.evaluate_roundtrip_criterion(orig_records, rt_pass, threshold=0.5)
        self.assertTrue(res_pass["passed"])
        self.assertEqual(res_pass["status_label"], "đạt")
        self.assertAlmostEqual(res_pass["fpr_increase_percentage_points"], 0.5, places=12)
        self.assertEqual(res_pass["decision_flips_vs_original"], 5)
        self.assertEqual(res_pass["flips_bonafide_to_spoof"], 5)
        self.assertEqual(res_pass["flips_spoof_to_bonafide"], 0)

        # 19 lỗi trong roundtrip (FPR = 1.9%, hiệu = +0.6 điểm %) -> không đạt
        rt_fail = [
            {"file": f, "score": 0.9 if i < 19 else 0.01, "raw_logit": 2.2 if i < 19 else -4.6}
            for i, f in enumerate(files)
        ]
        res_fail = runner.evaluate_roundtrip_criterion(orig_records, rt_fail, threshold=0.5)
        self.assertFalse(res_fail["passed"])
        self.assertEqual(res_fail["status_label"], "không đạt")
        self.assertAlmostEqual(res_fail["fpr_increase_percentage_points"], 0.6, places=12)
        self.assertEqual(res_fail["decision_flips_vs_original"], 6)

    def test_reproduction_failure_stops_seed_without_running_roundtrip_and_cache_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = root / "ckpt.pt"
            ckpt_path.write_bytes(b"fake ckpt")
            ckpt_sha = shared.sha256_file(ckpt_path)

            files = ["bonafide_0.wav", "bonafide_1.wav"]
            rows = [
                {"file": files[0], "speaker_id": "s0", "utt_type": "bonafide", "binary_label": "0", "shard": "sh0"},
                {"file": files[1], "speaker_id": "s1", "utt_type": "bonafide", "binary_label": "0", "shard": "sh1"},
            ]
            ref_score = float(np.float32(0.1))
            ref_logit = float(shared.stable_logit([ref_score])[0])
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "controller": {
                            "best": {
                                "generation": "g-" + "a" * 32,
                                "epoch": 6,
                                "development": {
                                    "decision_threshold": 0.5,
                                    "records": [
                                        {
                                            "file": f,
                                            "speaker_id": r["speaker_id"],
                                            "utt_type": "bonafide",
                                            "label": 0,
                                            "score": ref_score,
                                        }
                                        for f, r in zip(files, rows)
                                    ],
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            entry = {
                "model_id": "X1-L8-lr3e-5-lp7k-seed2026",
                "training_seed": 2026,
                "generation": "g-" + "a" * 32,
                "best_epoch": 6,
                "decision_threshold": 0.5,
                "dev_eer": 0.01,
                "checkpoint": "ckpt.pt",
                "checkpoint_sha256": ckpt_sha,
                "source_report": "report.json",
                "source_report_sha256": shared.sha256_file(report_path),
            }
            shared_identity = {"version": 1, "task": "REG-X1R-roundtrip-dev", "bonafide_samples": 2}

            # 1) Trường hợp tái tạo KHÔNG đạt: lệch logit 0.5 -> dừng seed, không chạy roundtrip
            called_conditions: list[str] = []

            def fake_predict_fail(model, dataset, indices, condition, *, batch_size=2):
                called_conditions.append(condition)
                bad_logit = ref_logit + 0.5
                bad_score = float(1.0 / (1.0 + np.exp(-bad_logit)))
                return [{"file": files[i], "score": bad_score, "raw_logit": bad_logit} for i in indices]

            with patch.object(runner, "load_x1r_model", return_value=object()), patch.object(
                runner, "predict_condition", side_effect=fake_predict_fail
            ):
                failed = runner.run_seed_check(
                    entry, rows, [0, 1], object(), shared_identity, cache_dir=cache_dir, root=root
                )
            self.assertFalse(failed["reproduction"]["passed"])
            self.assertIsNone(failed["comparison"])
            self.assertEqual(called_conditions, ["x1r_original"])

            # 2) Trường hợp tái tạo ĐẠT -> chạy cả 2 điều kiện, lưu cache, lần 2 bỏ qua
            called_conditions.clear()

            def fake_predict_ok(model, dataset, indices, condition, *, batch_size=2):
                called_conditions.append(condition)
                return [{"file": files[i], "score": ref_score, "raw_logit": ref_logit} for i in indices]

            with patch.object(runner, "load_x1r_model", return_value=object()), patch.object(
                runner, "predict_condition", side_effect=fake_predict_ok
            ):
                ok_res = runner.run_seed_check(
                    entry, rows, [0, 1], object(), shared_identity, cache_dir=cache_dir, root=root
                )
            self.assertTrue(ok_res["reproduction"]["passed"])
            self.assertTrue(ok_res["comparison"]["passed"])
            self.assertFalse(ok_res["reused_from_cache"])
            self.assertEqual(called_conditions, ["x1r_original", "x1r_roundtrip_40k"])

            # Lần chạy lại cùng identity -> bỏ qua predict và load model
            called_conditions.clear()
            with patch.object(runner, "load_x1r_model", side_effect=AssertionError("Không được nạp lại model")), patch.object(
                runner, "predict_condition", side_effect=AssertionError("Không được chạy lại predict")
            ):
                reused = runner.run_seed_check(
                    entry, rows, [0, 1], object(), shared_identity, cache_dir=cache_dir, root=root
                )
            self.assertTrue(reused["reused_from_cache"])
            self.assertEqual(reused["comparison"], ok_res["comparison"])

            # Đổi identity -> phải chạy lại
            changed_identity = deepcopy(shared_identity)
            changed_identity["bonafide_samples"] = 999
            with patch.object(runner, "load_x1r_model", return_value=object()), patch.object(
                runner, "predict_condition", side_effect=fake_predict_ok
            ):
                rerun = runner.run_seed_check(
                    entry, rows, [0, 1], object(), changed_identity, cache_dir=cache_dir, root=root
                )
            self.assertFalse(rerun["reused_from_cache"])
            self.assertEqual(called_conditions, ["x1r_original", "x1r_roundtrip_40k"])

    def test_rejects_closed_test_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "dữ liệu test"):
            runner.reject_test_path(Path("data/manifests/closed_test_full_v1.csv"))


if __name__ == "__main__":
    unittest.main()
