# T22 — Phép thử shortcut cho X1 cuối

Trạng thái: complete
CPU/dev-full, không train/test/Kaggle. Tập con 3.120 mẫu/169 shard; artifact 38.777 mẫu/186 shard.
Threshold X1 cuối cố định: 0.00015767214536590907
Tập con SHA-256: 869d8f87c8ffa6c52efc6bc619048dc85dd1eda6d9148fffeefc71ec7d802dfe

## Tái tạo logit

{"first200":{"criterion":"D034","decision_flips":0,"flips_outside_0_05_margin":0,"max_abs_logit_difference":6.100777134498969e-05,"max_excess_over_tolerance":-0.04993907297666951,"max_float32_gap_logit":1.5980278691429817e-06,"median_abs_logit_difference":1.8437987625574692e-06,"observed_logit_source":"block.raw_logit","passed":true,"reference_logit_source":"stable_logit(saved_score)","samples":200,"samples_over_0_05_without_gap":0},"full_subset":{"criterion":"D034","decision_flips":0,"flips_outside_0_05_margin":0,"max_abs_logit_difference":0.06893052922133514,"max_excess_over_tolerance":-0.04943696077261457,"max_float32_gap_logit":0.0741080317584153,"median_abs_logit_difference":0.008288345390345953,"observed_logit_source":"block.raw_logit","passed":true,"reference_logit_source":"stable_logit(saved_score)","samples":3120,"samples_over_0_05_without_gap":126}}

## Kết quả đặt cạnh LR1e-5 (mỗi checkpoint có threshold riêng đã khóa)

| Mô hình | Điều kiện | EER tổng % | VC % | AP % | Replay % | FPR % | Bỏ lọt VC % | AP % | Replay % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| X1-L8-lr1e-5-seed2026 | lowpass_4k | 2.300000 | 1.300000 | 1.500000 | 6.800000 | 2.500000 | 0.900000 | 0.900000 | 22.500000 |
| X1-L8-lr1e-5-seed2026 | lowpass_7k | 1.933962 | 1.600000 | 0.400000 | 7.300000 | 1.800000 | 1.300000 | 0.200000 | 23.333333 |
| X1-L8-lr1e-5-seed2026 | original | 1.462264 | 1.300000 | 0.200000 | 5.000000 | 1.000000 | 1.400000 | 0.100000 | 25.833333 |
| X1-L8-lr1e-5-seed2026 | roundtrip_40k | 1.886792 | 1.400000 | 0.200000 | 5.500000 | 1.500000 | 1.400000 | 0.100000 | 25.833333 |

X1-L8-lr1e-5-seed2026: {"available_indicator_level":"weak","complete_for_shortcut_indicators":true,"interpretation":"Chỉ báo theo ngưỡng khai báo trước, không kết luận có/không shortcut.","lowpass_4k_is_robustness_only":true,"lowpass_7k":{"eer_ratios":{"adversarial_attack":2.0,"voice_conversion":1.2307692307692308},"strong_threshold_hit":false},"roundtrip_40k":{"fpr_increase_percentage_points":0.49999999999999994,"strong_threshold_hit":false}}

| X1-L8-lr3e-5-seed2026 | original | 1.320755 | 1.200000 | 0.100000 | 4.166667 | 1.200000 | 1.100000 | 0.000000 | 18.333333 |
| X1-L8-lr3e-5-seed2026 | roundtrip_40k | 1.415094 | 1.300000 | 0.200000 | 4.166667 | 1.300000 | 1.100000 | 0.000000 | 18.333333 |
| X1-L8-lr3e-5-seed2026 | lowpass_7k | 1.698113 | 1.400000 | 0.300000 | 5.833333 | 1.700000 | 1.000000 | 0.100000 | 20.833333 |
| X1-L8-lr3e-5-seed2026 | lowpass_4k | 2.000000 | 1.400000 | 1.100000 | 7.200000 | 2.800000 | 0.600000 | 0.200000 | 18.333333 |

X1-L8-lr3e-5-seed2026: {"available_indicator_level":"weak","complete_for_shortcut_indicators":true,"interpretation":"Chỉ báo theo ngưỡng khai báo trước, không kết luận có/không shortcut.","lowpass_4k_is_robustness_only":true,"lowpass_7k":{"eer_ratios":{"adversarial_attack":3.0,"voice_conversion":1.1666666666666667},"strong_threshold_hit":false},"roundtrip_40k":{"fpr_increase_percentage_points":0.09999999999999992,"strong_threshold_hit":false}}

## Bộ lọc, checksum và thời gian

{"lowpass_4k":{"boundary":"zero_extension","coefficient_float32_sha256":"72a4f352a618490a0d5f21e59d890d6841236aca8d2321b457f9c98953a7d276","cutoff_hz":4000,"delay_compensation_samples":256,"fft_length":65536,"operation":"linear_phase_windowed_sinc","repeak":true,"response_db":{"3000":1.3137632006137321e-05,"4000":-6.020600430998958,"6000":-133.55627605637756,"7000":-142.4836433831936,"7500":-143.0980157160566},"sample_rate":16000,"taps":513,"window":"symmetric_Hann"},"lowpass_7k":{"boundary":"zero_extension","coefficient_float32_sha256":"973b82a36db8dab4f802de2102c5feed2bfb8cf6bb5846472c07d9984ab4c977","cutoff_hz":7000,"delay_compensation_samples":256,"fft_length":65536,"operation":"linear_phase_windowed_sinc","repeak":true,"response_db":{"3000":2.470312563578589e-07,"4000":3.8034457100841807e-07,"6000":1.2904362614004098e-05,"7000":-6.020603458515398,"7500":-97.84141174703096},"sample_rate":16000,"taps":513,"window":"symmetric_Hann"},"original":{"operation":"identity","repeak":false},"roundtrip_40k":{"beta":null,"lowpass_filter_width":6,"only_utt_type":"bonafide","operation":"torchaudio.functional.resample","repeak":true,"resampling_method":"sinc_interp_hann","rolloff":0.99,"sample_rates":[16000,40000,16000]}}

{"deadline_hours_this_invocation":8,"invocation_wall_seconds":10673.4164927,"recorded_active_stage_seconds":{"i":367.73946010000145,"ii":3515.7301357000033,"iii":1467.9316749999998,"iv":4513.284759300001,"v":4658.9900404},"rehash_and_setup_seconds_this_invocation":29.695554599999014,"seconds_left":18126.577999999998}

Ngắt: 0; resume: 1.
JSON chứa P5/P50/P95 xác suất và logit theo nhóm, SHA đầu vào/checkpoint/pretrained và danh sách file.
Chỉ báo khai báo trước; lowpass4k chỉ đo độ bền. Không kết luận có/không shortcut.

reproduction_rule: {"decision":"D034","identity_unchanged_sha256":"21feaf6da7933555369620996b2571104e9d368fb2bbdacdd22679d6b2dca12b","note":"identity.logit_convention ghi ngưỡng 0,05 theo đặc tả gốc; tiêu chí áp dụng là D034","runner_unchanged_sha256":"f4660226bef34a036d9be4b9d9c2eaef51d78b813676b75be49a1ef60c0c9fd2","wrapper":"scripts/probe_shortcut_x1_final_d034.py","wrapper_sha256":"9db342d4415bd1205e7fed458ba6c92fa89b77f0335c191137ff717dd6da747d"}
