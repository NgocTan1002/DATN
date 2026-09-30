# Báo cáo amplitude audit VSASV

- **Trạng thái kỹ thuật:** ĐẠT
- **Mẫu đã xử lý:** 2,558/2,558
- **Policy observation:** 7,674
- **Giao thức âm thanh:** phiên bản 3
- **Phạm vi đo:** `full_resampled_utterance`
- **Sample rate sau resample:** 16,000 Hz

## Phân bố mẫu cục bộ

| Split | Nhãn | Loại | Mẫu |
|---|---|---|---:|
| `dev` | `bonafide` | `bonafide` | 128 |
| `dev` | `spoof` | `adversarial_attack` | 511 |
| `dev` | `spoof` | `replay` | 80 |
| `test` | `bonafide` | `bonafide` | 247 |
| `test` | `spoof` | `replay` | 40 |
| `train` | `bonafide` | `bonafide` | 917 |
| `train` | `spoof` | `adversarial_attack` | 302 |
| `train` | `spoof` | `replay` | 145 |
| `train` | `spoof` | `voice_conversion` | 188 |

## Tổng hợp theo policy

| Policy | Mẫu | Input peak median/P95 | Input RMS median/P95 (dBFS) | Gain median/P95 | Near-silence | Peak-limited |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 2,558 | 0.6517 / 1.0000 | -22.07 / -13.29 | 1.0000 / 1.0000 | 0 | 0 |
| `peak` | 2,558 | 0.6517 / 1.0000 | -22.07 / -13.29 | 1.4578 / 3.5392 | 0 | 0 |
| `rms_dbfs` | 2,558 | 0.6517 / 1.0000 | -22.07 / -13.29 | 0.7136 / 1.6266 | 0 | 4 |

## Theo split

| split | policy | Mẫu | Input RMS median (dBFS) | Output RMS median (dBFS) | Gain median | Near-silence | Peak-limited |
|---|---|---:|---:|---:|---:|---:|---:|
| `dev` | `none` | 719 | -22.65 | -22.65 | 1.0000 | 0 | 0 |
| `dev` | `peak` | 719 | -22.65 | -19.23 | 1.3912 | 0 | 0 |
| `dev` | `rms_dbfs` | 719 | -22.65 | -25.00 | 0.7633 | 0 | 0 |
| `test` | `none` | 287 | -17.38 | -17.38 | 1.0000 | 0 | 0 |
| `test` | `peak` | 287 | -17.38 | -16.26 | 1.0338 | 0 | 0 |
| `test` | `rms_dbfs` | 287 | -17.38 | -25.00 | 0.4157 | 0 | 1 |
| `train` | `none` | 1,552 | -22.21 | -22.21 | 1.0000 | 0 | 0 |
| `train` | `peak` | 1,552 | -22.21 | -17.98 | 1.6018 | 0 | 0 |
| `train` | `rms_dbfs` | 1,552 | -22.21 | -25.00 | 0.7251 | 0 | 3 |

## Theo nhãn nhị phân

| binary_label | policy | Mẫu | Input RMS median (dBFS) | Output RMS median (dBFS) | Gain median | Near-silence | Peak-limited |
|---|---|---:|---:|---:|---:|---:|---:|
| `bonafide` | `none` | 1,292 | -21.29 | -21.29 | 1.0000 | 0 | 0 |
| `bonafide` | `peak` | 1,292 | -21.29 | -17.57 | 1.4303 | 0 | 0 |
| `bonafide` | `rms_dbfs` | 1,292 | -21.29 | -25.00 | 0.6526 | 0 | 3 |
| `spoof` | `none` | 1,266 | -22.54 | -22.54 | 1.0000 | 0 | 0 |
| `spoof` | `peak` | 1,266 | -22.54 | -18.90 | 1.4838 | 0 | 0 |
| `spoof` | `rms_dbfs` | 1,266 | -22.54 | -25.00 | 0.7536 | 0 | 1 |

## Theo utt_type

| utt_type | policy | Mẫu | Input RMS median (dBFS) | Output RMS median (dBFS) | Gain median | Near-silence | Peak-limited |
|---|---|---:|---:|---:|---:|---:|---:|
| `adversarial_attack` | `none` | 813 | -23.36 | -23.36 | 1.0000 | 0 | 0 |
| `adversarial_attack` | `peak` | 813 | -23.36 | -19.46 | 1.6022 | 0 | 0 |
| `adversarial_attack` | `rms_dbfs` | 813 | -23.36 | -25.00 | 0.8282 | 0 | 1 |
| `bonafide` | `none` | 1,292 | -21.29 | -21.29 | 1.0000 | 0 | 0 |
| `bonafide` | `peak` | 1,292 | -21.29 | -17.57 | 1.4303 | 0 | 0 |
| `bonafide` | `rms_dbfs` | 1,292 | -21.29 | -25.00 | 0.6526 | 0 | 3 |
| `replay` | `none` | 265 | -16.47 | -16.47 | 1.0000 | 0 | 0 |
| `replay` | `peak` | 265 | -16.47 | -16.52 | 0.9504 | 0 | 0 |
| `replay` | `rms_dbfs` | 265 | -16.47 | -25.00 | 0.3745 | 0 | 0 |
| `voice_conversion` | `none` | 188 | -23.91 | -23.91 | 1.0000 | 0 | 0 |
| `voice_conversion` | `peak` | 188 | -23.91 | -18.18 | 1.9650 | 0 | 0 |
| `voice_conversion` | `rms_dbfs` | 188 | -23.91 | -25.00 | 0.8819 | 0 | 0 |

## Theo sample rate gốc

| native_sample_rate | policy | Mẫu | Input RMS median (dBFS) | Output RMS median (dBFS) | Gain median | Near-silence | Peak-limited |
|---|---|---:|---:|---:|---:|---:|---:|
| `16000` | `none` | 2,370 | -21.90 | -21.90 | 1.0000 | 0 | 0 |
| `16000` | `peak` | 2,370 | -21.90 | -18.31 | 1.4030 | 0 | 0 |
| `16000` | `rms_dbfs` | 2,370 | -21.90 | -25.00 | 0.7000 | 0 | 4 |
| `40000` | `none` | 188 | -23.91 | -23.91 | 1.0000 | 0 | 0 |
| `40000` | `peak` | 188 | -23.91 | -18.18 | 1.9650 | 0 | 0 |
| `40000` | `rms_dbfs` | 188 | -23.91 | -25.00 | 0.8819 | 0 | 0 |

## Cảnh báo

- Năm shard cục bộ không phải mẫu ngẫu nhiên của toàn bộ 432 shard.
- Audit chỉ mô tả shortcut risk; không được dùng closed test để chọn policy.

## Kết luận sử dụng

Báo cáo này dùng để phát hiện chênh lệch biên độ và kiểm tra việc cài đặt. Policy chiến thắng vẫn phải được chọn bằng development EER với B0; không được chọn từ closed test hoặc từ thống kê mô tả này.
