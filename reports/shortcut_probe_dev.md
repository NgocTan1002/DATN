# T19b — Phép thử shortcut trên dev-full

Trạng thái: complete
Chỉ đánh giá CPU; không train/test/Kaggle.

B0: 38.777 mẫu/186 shard mỗi seed; XLS-R: 3.120 mẫu/169 shard. Threshold cố định từ checkpoint.
Tập con SHA-256: 869d8f87c8ffa6c52efc6bc619048dc85dd1eda6d9148fffeefc71ec7d802dfe
Giai đoạn đã xong: i, ii, iii, iv, v, vi

## Tái tạo original

{"X1-L8-lr1e-5-seed2026_first200": {"passed": true, "samples": 200, "max_abs_logit_difference": 5.521537333308402e-05, "median_abs_logit_difference": 1.802566632846947e-06, "decision_flips": 0, "flips_outside_0_05_margin": 0}, "X1-L8-lr1e-5-seed2026": {"passed": true, "samples": 3120, "max_abs_logit_difference": 0.0010702102824158288, "median_abs_logit_difference": 0.0, "decision_flips": 0, "flips_outside_0_05_margin": 0}, "B0-v2-seed2026": {"passed": true, "samples": 38777, "max_abs_probability_difference": 0.0, "decision_flips": 0}, "B0-v2-seed2027": {"passed": true, "samples": 38777, "max_abs_probability_difference": 0.0, "decision_flips": 0}, "B0-v2-seed2028": {"passed": true, "samples": 38777, "max_abs_probability_difference": 0.0, "decision_flips": 0}}

## Kết quả

| Mô hình | Điều kiện | EER tổng % | EER VC % | EER AP % | EER replay % | FPR % | Bỏ lọt VC % | AP % | Replay % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-v2-seed2026 | original | 8.225567 | 11.507386 | 2.678256 | 5.511978 | 8.225567 | 16.436791 | 0.073451 | 2.500000 |
| B0-v2-seed2026 | roundtrip_40k | 11.667020 | 16.140202 | 4.508515 | 8.333333 | 15.892870 | 16.436791 | 0.073451 | 2.500000 |
| B0-v2-seed2026 | lowpass_7k | 24.526923 | 15.977669 | 30.379478 | 16.666667 | 5.335312 | 35.052640 | 94.083082 | 36.666667 |
| B0-v2-seed2026 | lowpass_4k | 51.539024 | 23.016041 | 84.403929 | 36.666667 | 0.155466 | 78.772545 | 100.000000 | 100.000000 |

B0-v2-seed2026: {"lowpass_4k_is_robustness_only": true, "roundtrip_40k": {"fpr_increase_percentage_points": 7.667302664122677, "strong_threshold_hit": false}, "lowpass_7k": {"eer_ratios": {"voice_conversion": 1.388470804509217, "adversarial_attack": 11.343007915567282}, "strong_threshold_hit": false}, "available_indicator_level": "weak", "complete_for_shortcut_indicators": true, "interpretation": "Chỉ báo theo ngưỡng khai báo trước, không kết luận có/không shortcut."}

| B0-v2-seed2027 | original | 8.263624 | 11.165289 | 2.515723 | 11.235955 | 8.260900 | 16.322533 | 0.138742 | 15.000000 |
| B0-v2-seed2027 | roundtrip_40k | 13.810607 | 18.493393 | 4.766180 | 17.164865 | 19.970320 | 16.322533 | 0.138742 | 15.000000 |
| B0-v2-seed2027 | lowpass_7k | 17.355664 | 20.274186 | 14.811674 | 25.192566 | 16.083669 | 27.381050 | 11.458418 | 34.166667 |
| B0-v2-seed2027 | lowpass_4k | 38.966946 | 19.146332 | 47.898474 | 18.366193 | 0.897463 | 64.155717 | 99.910226 | 78.333333 |

B0-v2-seed2027: {"lowpass_4k_is_robustness_only": true, "roundtrip_40k": {"fpr_increase_percentage_points": 11.70941982898735, "strong_threshold_hit": true}, "lowpass_7k": {"eer_ratios": {"voice_conversion": 1.8158227848101265, "adversarial_attack": 5.8876404494382015}, "strong_threshold_hit": false}, "available_indicator_level": "strong", "complete_for_shortcut_indicators": true, "interpretation": "Chỉ báo theo ngưỡng khai báo trước, không kết luận có/không shortcut."}

| B0-v2-seed2028 | original | 7.928768 | 11.474741 | 1.689382 | 10.833333 | 7.928768 | 15.767567 | 0.032645 | 14.166667 |
| B0-v2-seed2028 | roundtrip_40k | 12.373684 | 17.136598 | 3.607280 | 16.373401 | 18.380326 | 15.767567 | 0.032645 | 14.166667 |
| B0-v2-seed2028 | lowpass_7k | 14.407537 | 20.329715 | 3.865451 | 22.599110 | 17.398064 | 23.855382 | 0.212193 | 25.833333 |
| B0-v2-seed2028 | lowpass_4k | 50.766730 | 22.712176 | 85.277075 | 37.500000 | 0.127199 | 71.974210 | 100.000000 | 100.000000 |

B0-v2-seed2028: {"lowpass_4k_is_robustness_only": true, "roundtrip_40k": {"fpr_increase_percentage_points": 10.451558193767225, "strong_threshold_hit": true}, "lowpass_7k": {"eer_ratios": {"voice_conversion": 1.771692745376956, "adversarial_attack": 2.288085681795759}, "strong_threshold_hit": false}, "available_indicator_level": "strong", "complete_for_shortcut_indicators": true, "interpretation": "Chỉ báo theo ngưỡng khai báo trước, không kết luận có/không shortcut."}

| X1-L8-lr1e-5-seed2026 | original | 1.462264 | 1.300000 | 0.200000 | 5.000000 | 1.000000 | 1.400000 | 0.100000 | 25.833333 |
| X1-L8-lr1e-5-seed2026 | roundtrip_40k | 1.886792 | 1.400000 | 0.200000 | 5.500000 | 1.500000 | 1.400000 | 0.100000 | 25.833333 |
| X1-L8-lr1e-5-seed2026 | lowpass_7k | 1.933962 | 1.600000 | 0.400000 | 7.300000 | 1.800000 | 1.300000 | 0.200000 | 23.333333 |
| X1-L8-lr1e-5-seed2026 | lowpass_4k | 2.300000 | 1.300000 | 1.500000 | 6.800000 | 2.500000 | 0.900000 | 0.900000 | 22.500000 |

X1-L8-lr1e-5-seed2026: {"lowpass_4k_is_robustness_only": true, "roundtrip_40k": {"fpr_increase_percentage_points": 0.49999999999999994, "strong_threshold_hit": false}, "lowpass_7k": {"eer_ratios": {"voice_conversion": 1.2307692307692308, "adversarial_attack": 2.0}, "strong_threshold_hit": false}, "available_indicator_level": "weak", "complete_for_shortcut_indicators": true, "interpretation": "Chỉ báo theo ngưỡng khai báo trước, không kết luận có/không shortcut."}


## Bộ lọc và số đo

{
  "original": {
    "operation": "identity",
    "repeak": false
  },
  "roundtrip_40k": {
    "operation": "torchaudio.functional.resample",
    "sample_rates": [
      16000,
      40000,
      16000
    ],
    "lowpass_filter_width": 6,
    "rolloff": 0.99,
    "resampling_method": "sinc_interp_hann",
    "beta": null,
    "only_utt_type": "bonafide",
    "repeak": true
  },
  "lowpass_7k": {
    "operation": "linear_phase_windowed_sinc",
    "cutoff_hz": 7000,
    "taps": 513,
    "window": "symmetric_Hann",
    "sample_rate": 16000,
    "fft_length": 65536,
    "delay_compensation_samples": 256,
    "boundary": "zero_extension",
    "coefficient_float32_sha256": "973b82a36db8dab4f802de2102c5feed2bfb8cf6bb5846472c07d9984ab4c977",
    "response_db": {
      "3000": 2.470312563578589e-07,
      "4000": 3.8034457100841807e-07,
      "6000": 1.2904362614004098e-05,
      "7000": -6.020603458515398,
      "7500": -97.84141174703096
    },
    "repeak": true
  },
  "lowpass_4k": {
    "operation": "linear_phase_windowed_sinc",
    "cutoff_hz": 4000,
    "taps": 513,
    "window": "symmetric_Hann",
    "sample_rate": 16000,
    "fft_length": 65536,
    "delay_compensation_samples": 256,
    "boundary": "zero_extension",
    "coefficient_float32_sha256": "72a4f352a618490a0d5f21e59d890d6841236aca8d2321b457f9c98953a7d276",
    "response_db": {
      "3000": 1.3137632006137321e-05,
      "4000": -6.020600430998958,
      "6000": -133.55627605637756,
      "7000": -142.4836433831936,
      "7500": -143.0980157160566
    },
    "repeak": true
  }
}

FIR Hann513tap, pha tuyến tính, tích chập FFT tuyến tính/bù trễ256mẫu, zero-extension. Sau biến đổi áp lại peak.
Đáp ứng dB tại3/4/6/7/7,5kHz nằm ở khối bộ lọc và JSON. Tham số roundtrip là mặc định torchaudio của pipeline.
P5/P50/P95 xác suất và logit từng nhóm, checksum đầu vào/checkpoint/pretrained, threshold, số khối và thời gian có trong JSON/blocks.
Số lần ngắt: 0; số lần resume: 0
{
  "invocation_wall_seconds": 26678.176038700003,
  "rehash_and_setup_seconds_this_invocation": 31.017407100000128,
  "recorded_active_stage_seconds": {
    "i": 323.5908092999962,
    "ii": 5171.4482441999935,
    "iii": 1726.7175876000038,
    "iv": 5396.550686799994,
    "v": 8718.887953899997,
    "vi": 5233.339585900001
  },
  "note": "Tổng thời gian pha ghi đến event cuối; không tính downtime/đuôi chưa commit khi mất điện.",
  "deadline_hours_this_invocation": 11.0,
  "seconds_left": 12921.828000000009
}

Lowpass4k chỉ kiểm độ bền. Không kết luận có hoặc không shortcut.
