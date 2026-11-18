# SC-01 — Thống kê lối tắt theo lớp trên train và dev-full

- **Cập nhật UTC:** 2026-11-17T07:24:42.164596+00:00
- **Mức phân loại chỉ báo:** **mạnh** (ngưỡng khóa: mạnh ≤ 10%, trung bình > 10% và ≤ 25%, yếu > 25%)
- **Phép đạt EER dev thấp nhất:** `bonafide_vs_vc` / `logistic_full` (vùng `full`) — EER dev = **8.7626%** (CI 95% theo người nói: [4.1763%; 17.5234%], 172 speaker)
- **Dữ liệu train (`closed_train`):** 15,885 mẫu / 57 shard
- **Dữ liệu dev (`closed_dev`):** 38,777 mẫu / 186 shard
- **Thời gian chạy:** 1563.33 giây (CPU, chỉ đọc)

## Bảng chéo sample rate gốc × lớp

| Split | Lớp | n | 16 kHz | 40 kHz |
|---|---|---:|---:|---:|
| train | bonafide | 7,134 | 7,134 | 0 |
| train | voice_conversion | 4,348 | 0 | 4,348 |
| train | adversarial_attack | 4,348 | 4,348 | 0 |
| train | replay | 55 | 55 | 0 |
| dev | bonafide | 14,151 | 14,151 | 0 |
| dev | voice_conversion | 12,253 | 0 | 12,253 |
| dev | adversarial_attack | 12,253 | 12,253 | 0 |
| dev | replay | 120 | 120 | 0 |

## Thống kê mô tả trên train

| Lớp | n | Tỷ lệ pad | Đặc trưng | P5 | P25 | Trung vị | P75 | P95 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| bonafide | 7,134 | 40.57% | `duration_s` | 1.938 | 2.84 | 4.628 | 8.44 | 20.84 |
| bonafide | 7,134 | 40.57% | `pad_fraction` | 0 | 0 | 0 | 0.29 | 0.5156 |
| bonafide | 7,134 | 40.57% | `full_silence_ratio` | 0 | 0 | 0 | 0.07608 | 0.3602 |
| bonafide | 7,134 | 40.57% | `full_leading_silence_s` | 0 | 0 | 0 | 0 | 0.5 |
| bonafide | 7,134 | 40.57% | `full_trailing_silence_s` | 0 | 0 | 0 | 0 | 0.49 |
| bonafide | 7,134 | 40.57% | `full_noise_floor_db` | -55.01 | -38.07 | -27.95 | -21.57 | -14.57 |
| bonafide | 7,134 | 40.57% | `full_dc_offset` | 1.006e-06 | 8.757e-06 | 2.576e-05 | 6.971e-05 | 0.0004692 |
| bonafide | 7,134 | 40.57% | `full_lowband_ratio_db` | -39.19 | -30.36 | -24.06 | -17.4 | -9.606 |
| bonafide | 7,134 | 40.57% | `crop_silence_ratio` | 0 | 0 | 0.08543 | 0.3643 | 0.6206 |
| bonafide | 7,134 | 40.57% | `crop_leading_silence_s` | 0 | 0 | 0 | 0 | 0.39 |
| bonafide | 7,134 | 40.57% | `crop_trailing_silence_s` | 0 | 0 | 0 | 1.14 | 2.34 |
| bonafide | 7,134 | 40.57% | `crop_noise_floor_db` | -54.42 | -36.57 | -26.57 | -20.71 | -14.14 |
| bonafide | 7,134 | 40.57% | `crop_dc_offset` | 2.267e-06 | 1.441e-05 | 3.675e-05 | 8.158e-05 | 0.0004844 |
| bonafide | 7,134 | 40.57% | `crop_lowband_ratio_db` | -39.27 | -30.77 | -24.37 | -17.66 | -9.629 |
| voice_conversion | 4,348 | 56.07% | `duration_s` | 1.86 | 2.96 | 3.76 | 4.96 | 8.18 |
| voice_conversion | 4,348 | 56.07% | `pad_fraction` | 0 | 0 | 0.06 | 0.26 | 0.535 |
| voice_conversion | 4,348 | 56.07% | `full_silence_ratio` | 0.006225 | 0.1237 | 0.2648 | 0.4081 | 0.5753 |
| voice_conversion | 4,348 | 56.07% | `full_leading_silence_s` | 0 | 0 | 0.22 | 0.65 | 1.096 |
| voice_conversion | 4,348 | 56.07% | `full_trailing_silence_s` | 0 | 0.02 | 0.36 | 0.66 | 1.1 |
| voice_conversion | 4,348 | 56.07% | `full_noise_floor_db` | -66.1 | -56.43 | -46.4 | -40.71 | -32.63 |
| voice_conversion | 4,348 | 56.07% | `full_dc_offset` | 4.405e-05 | 0.0003222 | 0.0005353 | 0.0009917 | 0.001623 |
| voice_conversion | 4,348 | 56.07% | `full_lowband_ratio_db` | -41.04 | -33.86 | -29.98 | -23.74 | -10.83 |
| voice_conversion | 4,348 | 56.07% | `crop_silence_ratio` | 0 | 0.1508 | 0.3392 | 0.5251 | 0.6834 |
| voice_conversion | 4,348 | 56.07% | `crop_leading_silence_s` | 0 | 0 | 0.07 | 0.45 | 0.93 |
| voice_conversion | 4,348 | 56.07% | `crop_trailing_silence_s` | 0 | 0.07 | 0.68 | 1.452 | 2.3 |
| voice_conversion | 4,348 | 56.07% | `crop_noise_floor_db` | -64.62 | -54.9 | -43.75 | -38.79 | -24.45 |
| voice_conversion | 4,348 | 56.07% | `crop_dc_offset` | 4.79e-05 | 0.0003171 | 0.0005574 | 0.001043 | 0.001775 |
| voice_conversion | 4,348 | 56.07% | `crop_lowband_ratio_db` | -40.93 | -34.14 | -30.08 | -23.96 | -10.88 |
| adversarial_attack | 4,348 | 47.52% | `duration_s` | 2.42 | 3.3 | 4.08 | 5.145 | 10.02 |
| adversarial_attack | 4,348 | 47.52% | `pad_fraction` | 0 | 0 | 0 | 0.175 | 0.395 |
| adversarial_attack | 4,348 | 47.52% | `full_silence_ratio` | 0 | 0 | 0 | 0 | 0 |
| adversarial_attack | 4,348 | 47.52% | `full_leading_silence_s` | 0 | 0 | 0 | 0 | 0 |
| adversarial_attack | 4,348 | 47.52% | `full_trailing_silence_s` | 0 | 0 | 0 | 0 | 0 |
| adversarial_attack | 4,348 | 47.52% | `full_noise_floor_db` | -28.92 | -26.32 | -24.3 | -22.44 | -19.37 |
| adversarial_attack | 4,348 | 47.52% | `full_dc_offset` | 6.9e-05 | 0.0003765 | 0.0007186 | 0.001062 | 0.001841 |
| adversarial_attack | 4,348 | 47.52% | `full_lowband_ratio_db` | -32.02 | -27.4 | -24.55 | -18.47 | -11.38 |
| adversarial_attack | 4,348 | 47.52% | `crop_silence_ratio` | 0 | 0 | 0 | 0.1765 | 0.397 |
| adversarial_attack | 4,348 | 47.52% | `crop_leading_silence_s` | 0 | 0 | 0 | 0 | 0 |
| adversarial_attack | 4,348 | 47.52% | `crop_trailing_silence_s` | 0 | 0 | 0 | 0.68 | 1.56 |
| adversarial_attack | 4,348 | 47.52% | `crop_noise_floor_db` | -28.82 | -26.09 | -24.05 | -22.19 | -18.54 |
| adversarial_attack | 4,348 | 47.52% | `crop_dc_offset` | 8.12e-05 | 0.0004191 | 0.0007608 | 0.001169 | 0.001977 |
| adversarial_attack | 4,348 | 47.52% | `crop_lowband_ratio_db` | -32.33 | -27.8 | -24.63 | -18.63 | -11.45 |
| replay | 55 | 74.55% | `duration_s` | 2.029 | 2.688 | 3.136 | 4 | 5.03 |
| replay | 55 | 74.55% | `pad_fraction` | 0 | 0.012 | 0.216 | 0.328 | 0.4928 |
| replay | 55 | 74.55% | `full_silence_ratio` | 0.02075 | 0.05631 | 0.1517 | 0.2174 | 0.3366 |
| replay | 55 | 74.55% | `full_leading_silence_s` | 0.037 | 0.045 | 0.06 | 0.08 | 0.165 |
| replay | 55 | 74.55% | `full_trailing_silence_s` | 0 | 0 | 0.02 | 0.165 | 0.376 |
| replay | 55 | 74.55% | `full_noise_floor_db` | -44.33 | -42.87 | -41.38 | -38.88 | -35.48 |
| replay | 55 | 74.55% | `full_dc_offset` | 2.934e-06 | 1.512e-05 | 8.481e-05 | 0.0004102 | 0.001438 |
| replay | 55 | 74.55% | `full_lowband_ratio_db` | -32.79 | -29.34 | -27.63 | -26.12 | -21.58 |
| replay | 55 | 74.55% | `crop_silence_ratio` | 0.008543 | 0.1608 | 0.3241 | 0.4447 | 0.6156 |
| replay | 55 | 74.55% | `crop_leading_silence_s` | 0 | 0.04 | 0.05 | 0.075 | 0.165 |
| replay | 55 | 74.55% | `crop_trailing_silence_s` | 0 | 0.26 | 0.93 | 1.36 | 1.979 |
| replay | 55 | 74.55% | `crop_noise_floor_db` | -44.2 | -43.03 | -41.32 | -38.84 | -34.12 |
| replay | 55 | 74.55% | `crop_dc_offset` | 4.279e-06 | 2.264e-05 | 9.787e-05 | 0.0004658 | 0.001405 |
| replay | 55 | 74.55% | `crop_lowband_ratio_db` | -31.94 | -28.98 | -27.69 | -25.72 | -22.81 |

## Thống kê mô tả trên dev

| Lớp | n | Tỷ lệ pad | Đặc trưng | P5 | P25 | Trung vị | P75 | P95 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| bonafide | 14,151 | 41.11% | `duration_s` | 1.64 | 2.84 | 4.44 | 8.44 | 22.04 |
| bonafide | 14,151 | 41.11% | `pad_fraction` | 0 | 0 | 0 | 0.29 | 0.59 |
| bonafide | 14,151 | 41.11% | `full_silence_ratio` | 0 | 0 | 0 | 0.03451 | 0.188 |
| bonafide | 14,151 | 41.11% | `full_leading_silence_s` | 0 | 0 | 0 | 0 | 0.17 |
| bonafide | 14,151 | 41.11% | `full_trailing_silence_s` | 0 | 0 | 0 | 0 | 0.05 |
| bonafide | 14,151 | 41.11% | `full_noise_floor_db` | -49.09 | -35.27 | -26.93 | -20.86 | -13.58 |
| bonafide | 14,151 | 41.11% | `full_dc_offset` | 1.623e-06 | 1.114e-05 | 2.948e-05 | 7.462e-05 | 0.0003497 |
| bonafide | 14,151 | 41.11% | `full_lowband_ratio_db` | -37.49 | -30.39 | -22.82 | -15.83 | -7.443 |
| bonafide | 14,151 | 41.11% | `crop_silence_ratio` | 0 | 0 | 0.06533 | 0.2915 | 0.5879 |
| bonafide | 14,151 | 41.11% | `crop_leading_silence_s` | 0 | 0 | 0 | 0 | 0.05 |
| bonafide | 14,151 | 41.11% | `crop_trailing_silence_s` | 0 | 0 | 0 | 1.14 | 2.34 |
| bonafide | 14,151 | 41.11% | `crop_noise_floor_db` | -48.08 | -34.01 | -25.82 | -19.92 | -13.1 |
| bonafide | 14,151 | 41.11% | `crop_dc_offset` | 2.797e-06 | 1.661e-05 | 4.135e-05 | 9.38e-05 | 0.0004171 |
| bonafide | 14,151 | 41.11% | `crop_lowband_ratio_db` | -37.72 | -30.5 | -23.02 | -15.86 | -7.31 |
| voice_conversion | 12,253 | 47.39% | `duration_s` | 2.32 | 3.26 | 4.12 | 5.2 | 7.14 |
| voice_conversion | 12,253 | 47.39% | `pad_fraction` | 0 | 0 | 0 | 0.185 | 0.42 |
| voice_conversion | 12,253 | 47.39% | `full_silence_ratio` | 0.02249 | 0.3047 | 0.4005 | 0.4909 | 0.6202 |
| voice_conversion | 12,253 | 47.39% | `full_leading_silence_s` | 0 | 0.33 | 0.66 | 0.84 | 1.13 |
| voice_conversion | 12,253 | 47.39% | `full_trailing_silence_s` | 0 | 0.42 | 0.77 | 0.98 | 1.36 |
| voice_conversion | 12,253 | 47.39% | `full_noise_floor_db` | -63.38 | -59.07 | -54.44 | -49.2 | -32.79 |
| voice_conversion | 12,253 | 47.39% | `full_dc_offset` | 0.0001401 | 0.0005259 | 0.0008766 | 0.001511 | 0.002155 |
| voice_conversion | 12,253 | 47.39% | `full_lowband_ratio_db` | -39.36 | -29.47 | -20.79 | -16.3 | -12.37 |
| voice_conversion | 12,253 | 47.39% | `crop_silence_ratio` | 0.02261 | 0.2035 | 0.3744 | 0.5578 | 0.702 |
| voice_conversion | 12,253 | 47.39% | `crop_leading_silence_s` | 0 | 0 | 0.35 | 0.66 | 0.96 |
| voice_conversion | 12,253 | 47.39% | `crop_trailing_silence_s` | 0 | 0.2 | 0.76 | 1.47 | 2.22 |
| voice_conversion | 12,253 | 47.39% | `crop_noise_floor_db` | -62.97 | -58.31 | -52.56 | -45.95 | -28.41 |
| voice_conversion | 12,253 | 47.39% | `crop_dc_offset` | 0.000131 | 0.00052 | 0.0009662 | 0.001535 | 0.002441 |
| voice_conversion | 12,253 | 47.39% | `crop_lowband_ratio_db` | -39.15 | -29.49 | -20.86 | -16.34 | -12.42 |
| adversarial_attack | 12,253 | 47.39% | `duration_s` | 2.32 | 3.26 | 4.12 | 5.2 | 7.14 |
| adversarial_attack | 12,253 | 47.39% | `pad_fraction` | 0 | 0 | 0 | 0.185 | 0.42 |
| adversarial_attack | 12,253 | 47.39% | `full_silence_ratio` | 0 | 0 | 0 | 0 | 0 |
| adversarial_attack | 12,253 | 47.39% | `full_leading_silence_s` | 0 | 0 | 0 | 0 | 0 |
| adversarial_attack | 12,253 | 47.39% | `full_trailing_silence_s` | 0 | 0 | 0 | 0 | 0 |
| adversarial_attack | 12,253 | 47.39% | `full_noise_floor_db` | -28.67 | -26.63 | -25.04 | -23.37 | -20.58 |
| adversarial_attack | 12,253 | 47.39% | `full_dc_offset` | 8.29e-05 | 0.0003655 | 0.0007247 | 0.001264 | 0.001937 |
| adversarial_attack | 12,253 | 47.39% | `full_lowband_ratio_db` | -33.89 | -26.87 | -20.31 | -16.17 | -12.45 |
| adversarial_attack | 12,253 | 47.39% | `crop_silence_ratio` | 0 | 0 | 0 | 0.1809 | 0.4171 |
| adversarial_attack | 12,253 | 47.39% | `crop_leading_silence_s` | 0 | 0 | 0 | 0 | 0 |
| adversarial_attack | 12,253 | 47.39% | `crop_trailing_silence_s` | 0 | 0 | 0 | 0.72 | 1.66 |
| adversarial_attack | 12,253 | 47.39% | `crop_noise_floor_db` | -28.39 | -26.39 | -24.81 | -23.14 | -20.21 |
| adversarial_attack | 12,253 | 47.39% | `crop_dc_offset` | 7.936e-05 | 0.0003906 | 0.0008242 | 0.001337 | 0.002313 |
| adversarial_attack | 12,253 | 47.39% | `crop_lowband_ratio_db` | -33.97 | -26.94 | -20.43 | -16.24 | -12.5 |
| replay | 120 | 89.17% | `duration_s` | 1.664 | 2.496 | 2.944 | 3.52 | 4.227 |
| replay | 120 | 89.17% | `pad_fraction` | 0 | 0.12 | 0.264 | 0.376 | 0.584 |
| replay | 120 | 89.17% | `full_silence_ratio` | 0.0157 | 0.04513 | 0.1011 | 0.1991 | 0.3498 |
| replay | 120 | 89.17% | `full_leading_silence_s` | 0.04 | 0.05 | 0.06 | 0.09 | 0.2 |
| replay | 120 | 89.17% | `full_trailing_silence_s` | 0 | 0 | 0.01 | 0.105 | 0.3005 |
| replay | 120 | 89.17% | `full_noise_floor_db` | -44.6 | -42.33 | -39.99 | -32.92 | -28.7 |
| replay | 120 | 89.17% | `full_dc_offset` | 1.066e-05 | 6.217e-05 | 0.0002839 | 0.001279 | 0.003081 |
| replay | 120 | 89.17% | `full_lowband_ratio_db` | -35.09 | -32.73 | -30.55 | -26.84 | -23.29 |
| replay | 120 | 89.17% | `crop_silence_ratio` | 0.01721 | 0.2004 | 0.358 | 0.5025 | 0.6609 |
| replay | 120 | 89.17% | `crop_leading_silence_s` | 0.0095 | 0.05 | 0.06 | 0.09 | 0.181 |
| replay | 120 | 89.17% | `crop_trailing_silence_s` | 0 | 0.5725 | 1.155 | 1.625 | 2.321 |
| replay | 120 | 89.17% | `crop_noise_floor_db` | -45.18 | -42.74 | -40.33 | -34.78 | -30.47 |
| replay | 120 | 89.17% | `crop_dc_offset` | 1.358e-05 | 6.546e-05 | 0.0003353 | 0.001312 | 0.003095 |
| replay | 120 | 89.17% | `crop_lowband_ratio_db` | -35.02 | -32.37 | -30.14 | -26.51 | -23.47 |

## EER đơn đặc trưng (ngưỡng chọn trên train, áp nguyên lên dev)

| Đặc trưng | Phép | Hướng (train) | EER train % | EER dev % | FPR bonafide dev % | Miss spoof dev % |
|---|---|---|---:|---:|---:|---:|
| `duration_s` | bonafide_vs_spoof | lower_is_spoof | 44.7435 | 46.0774 | 45.8060 | 50.0893 |
| `pad_fraction` | bonafide_vs_spoof | higher_is_spoof | 44.7112 | 47.0876 | 41.0996 | 52.4040 |
| `full_silence_ratio` | bonafide_vs_spoof | higher_is_spoof | 47.6191 | 45.7147 | 40.0042 | 50.5157 |
| `full_leading_silence_s` | bonafide_vs_spoof | higher_is_spoof | 43.2115 | 38.1528 | 9.7520 | 55.6729 |
| `full_trailing_silence_s` | bonafide_vs_spoof | higher_is_spoof | 42.0471 | 37.1750 | 7.0808 | 54.9825 |
| `full_noise_floor_db` | bonafide_vs_spoof | lower_is_spoof | 47.9394 | 46.1504 | 43.9757 | 48.2620 |
| `full_dc_offset` | bonafide_vs_spoof | higher_is_spoof | 11.7472 | 10.2608 | 10.7201 | 9.5955 |
| `full_lowband_ratio_db` | bonafide_vs_spoof | lower_is_spoof | 43.6922 | 54.4749 | 39.6085 | 67.3759 |
| `crop_silence_ratio` | bonafide_vs_spoof | higher_is_spoof | 44.9474 | 41.3109 | 41.8910 | 40.7009 |
| `crop_leading_silence_s` | bonafide_vs_spoof | higher_is_spoof | 44.4179 | 40.2637 | 6.3529 | 63.1203 |
| `crop_trailing_silence_s` | bonafide_vs_spoof | higher_is_spoof | 43.1598 | 41.3576 | 41.4388 | 40.4491 |
| `crop_noise_floor_db` | bonafide_vs_spoof | lower_is_spoof | 47.0844 | 44.5128 | 43.5800 | 45.9068 |
| `crop_dc_offset` | bonafide_vs_spoof | higher_is_spoof | 11.6215 | 11.5325 | 12.1476 | 10.8300 |
| `crop_lowband_ratio_db` | bonafide_vs_spoof | lower_is_spoof | 44.3930 | 54.5686 | 39.9406 | 67.6846 |
| `duration_s` | bonafide_vs_vc | lower_is_spoof | 42.9845 | 46.0870 | 41.2056 | 52.3953 |
| `pad_fraction` | bonafide_vs_vc | higher_is_spoof | 42.4994 | 47.1843 | 41.0996 | 52.6075 |
| `full_silence_ratio` | bonafide_vs_vc | higher_is_spoof | 21.6428 | 11.7277 | 12.6705 | 11.1238 |
| `full_leading_silence_s` | bonafide_vs_vc | higher_is_spoof | 25.1821 | 11.9865 | 9.7520 | 12.2909 |
| `full_trailing_silence_s` | bonafide_vs_vc | higher_is_spoof | 21.1095 | 10.3611 | 7.0808 | 10.7402 |
| `full_noise_floor_db` | bonafide_vs_vc | lower_is_spoof | 21.6651 | 11.9318 | 12.7906 | 11.1156 |
| `full_dc_offset` | bonafide_vs_vc | higher_is_spoof | 10.9706 | 9.6318 | 9.9145 | 9.0672 |
| `full_lowband_ratio_db` | bonafide_vs_vc | lower_is_spoof | 35.4876 | 53.9786 | 33.8704 | 69.7788 |
| `crop_silence_ratio` | bonafide_vs_vc | higher_is_spoof | 35.3927 | 30.9309 | 31.8564 | 28.4012 |
| `crop_leading_silence_s` | bonafide_vs_vc | higher_is_spoof | 31.5354 | 22.4092 | 6.3529 | 27.0464 |
| `crop_trailing_silence_s` | bonafide_vs_vc | higher_is_spoof | 37.5218 | 35.5138 | 35.8420 | 32.2941 |
| `crop_noise_floor_db` | bonafide_vs_vc | lower_is_spoof | 22.6100 | 13.8435 | 15.2074 | 13.0335 |
| `crop_dc_offset` | bonafide_vs_vc | higher_is_spoof | 11.2695 | 10.7974 | 11.7165 | 9.6711 |
| `crop_lowband_ratio_db` | bonafide_vs_vc | lower_is_spoof | 36.2006 | 54.0669 | 34.1177 | 69.9829 |
| `duration_s` | bonafide_vs_ap | lower_is_spoof | 45.2897 | 46.0870 | 46.0250 | 47.3190 |
| `pad_fraction` | bonafide_vs_ap | higher_is_spoof | 46.8951 | 47.1843 | 41.0996 | 52.6075 |
| `full_silence_ratio` | bonafide_vs_ap | lower_is_spoof | 36.2259 | 37.8442 | 59.9816 | 1.4854 |
| `full_leading_silence_s` | bonafide_vs_ap | lower_is_spoof | 45.5605 | 47.5370 | 90.2480 | 0.3999 |
| `full_trailing_silence_s` | bonafide_vs_ap | lower_is_spoof | 45.7386 | 48.3428 | 92.9192 | 0.7100 |
| `full_noise_floor_db` | bonafide_vs_ap | higher_is_spoof | 39.3468 | 44.0110 | 42.6754 | 49.6368 |
| `full_dc_offset` | bonafide_vs_ap | higher_is_spoof | 11.9708 | 10.8968 | 11.1582 | 10.6260 |
| `full_lowband_ratio_db` | bonafide_vs_ap | lower_is_spoof | 48.6123 | 55.1622 | 43.7213 | 66.1797 |
| `crop_silence_ratio` | bonafide_vs_ap | lower_is_spoof | 43.0441 | 44.4111 | 45.9049 | 43.2139 |
| `crop_leading_silence_s` | bonafide_vs_ap | lower_is_spoof | 46.6178 | 48.4188 | 93.6471 | 0.2367 |
| `crop_trailing_silence_s` | bonafide_vs_ap | higher_is_spoof | 49.4797 | 48.6116 | 43.2832 | 53.6522 |
| `crop_noise_floor_db` | bonafide_vs_ap | higher_is_spoof | 42.3045 | 46.8518 | 45.2477 | 52.8932 |
| `crop_dc_offset` | bonafide_vs_ap | higher_is_spoof | 11.6204 | 12.1603 | 12.1405 | 12.1929 |
| `crop_lowband_ratio_db` | bonafide_vs_ap | lower_is_spoof | 49.1870 | 55.3318 | 43.8909 | 66.0981 |

## Kết hợp hồi quy logistic (chuẩn hóa z-score trên train, chấm dev)

| Phép | Vùng | EER train % | EER dev % | FPR bonafide dev % | Miss spoof dev % |
|---|---|---:|---:|---:|---:|
| bonafide_vs_spoof | crop | 14.4555 | 11.7589 | 9.1513 | 13.7781 |
| bonafide_vs_spoof | full | 14.9706 | 11.8932 | 9.2078 | 14.1761 |
| bonafide_vs_vc | crop | 15.7275 | 9.3350 | 8.0984 | 11.3197 |
| bonafide_vs_vc | full | 15.4191 | 8.7626 | 7.3422 | 11.0177 |
| bonafide_vs_ap | crop | 13.1785 | 13.1886 | 12.2182 | 14.0211 |
| bonafide_vs_ap | full | 14.1155 | 13.6783 | 13.3630 | 13.9150 |

## Giới hạn

- Không có cột nguồn corpus hay codec trong metadata cục bộ nên không kiểm được hai lối tắt này.
- Đây là chỉ báo dữ liệu thô, không đo hành vi mô hình hay chứng minh mô hình dùng lối tắt.
- Crop train lúc huấn luyện là ngẫu nhiên theo seed/epoch, nên đặc trưng vùng crop (center crop) chỉ xấp xỉ cái mô hình thấy khi train.
- Kết quả chỉ áp dụng cho VSASV-HF-public-snapshot-v1 và tập con 15.885 mẫu closed_train / 38.777 mẫu closed_dev.
