# Báo cáo audio smoke test VSASV

- **Trạng thái kỹ thuật:** ĐẠT
- **Shard cục bộ:** 6/432
- **Utterance:** 3,070
- **Speaker trong 6 shard:** 42
- **Sample rate đích bắt buộc:** 16,000 Hz

## Kiểm tra toàn bộ 6 shard

- Đường dẫn duy nhất: 3,070/3,070
- Thiếu file/label/utt_type: 0
- Waveform thiếu hoặc rỗng: 0
- Sample rate không hợp lệ: 0

## Phân bố sample rate

| Loại | Sample rate | Số mẫu |
|---|---:|---:|
| `adversarial_attack` | 16,000 Hz | 813 |
| `bonafide` | 16,000 Hz | 1,804 |
| `replay` | 16,000 Hz | 265 |
| `voice_conversion` | 40,000 Hz | 188 |

## Thời lượng

| Loại | Min (s) | Median (s) | Mean (s) | P95 (s) | Max (s) | Tổng giờ |
|---|---:|---:|---:|---:|---:|---:|
| `adversarial_attack` | 1.86 | 4.04 | 4.16 | 6.14 | 8.38 | 0.94 |
| `bonafide` | 0.84 | 3.24 | 4.42 | 11.64 | 35.24 | 2.21 |
| `replay` | 1.41 | 3.07 | 3.22 | 4.80 | 6.46 | 0.24 |
| `voice_conversion` | 2.08 | 3.62 | 3.74 | 5.38 | 8.04 | 0.20 |

## Kiểm tra waveform mẫu

Đã đọc và kiểm tra số học 27 waveform được chọn xác định theo từng cặp shard/loại bằng hash(file).

| File | Loại | Hz | Giây | Peak | RMS | Non-finite | Silent |
|---|---|---:|---:|---:|---:|---:|---|
| `id00003/00044.wav` | `bonafide` | 16,000 | 3.24 | 0.4152 | 0.0640 | 0 | Không |
| `id00012/00029.wav` | `bonafide` | 16,000 | 1.24 | 0.5765 | 0.1001 | 0 | Không |
| `id00003/00036.wav` | `bonafide` | 16,000 | 9.24 | 0.4185 | 0.0593 | 0 | Không |
| `id00020/00011.wav` | `bonafide` | 16,000 | 3.24 | 0.5601 | 0.0858 | 0 | Không |
| `id00018/00015.wav` | `bonafide` | 16,000 | 3.24 | 0.5299 | 0.0502 | 0 | Không |
| `id00014/00018.wav` | `bonafide` | 16,000 | 2.84 | 1.0000 | 0.3391 | 0 | Không |
| `id00026/00073.wav` | `bonafide` | 16,000 | 3.24 | 1.0000 | 0.2152 | 0 | Không |
| `id00026/00135.wav` | `bonafide` | 16,000 | 1.64 | 0.7535 | 0.1217 | 0 | Không |
| `id00024/00049.wav` | `bonafide` | 16,000 | 2.44 | 0.5837 | 0.0962 | 0 | Không |
| `id01004/bonafide/00000.wav` | `bonafide` | 16,000 | 5.33 | 0.7287 | 0.0795 | 0 | Không |
| `id01006/voice_conversion/id01006_vc_00041.wav` | `voice_conversion` | 40,000 | 3.98 | 0.3731 | 0.0533 | 0 | Không |
| `id01005/adversarial_attack/00282.wav` | `adversarial_attack` | 16,000 | 2.76 | 0.4568 | 0.0539 | 0 | Không |
| `id01006/adversarial_attack/00290.wav` | `adversarial_attack` | 16,000 | 3.04 | 0.5615 | 0.0693 | 0 | Không |
| `id01006/voice_conversion/id01006_vc_00118.wav` | `voice_conversion` | 40,000 | 3.04 | 0.5010 | 0.0528 | 0 | Không |
| `id01004/bonafide/00001.wav` | `bonafide` | 16,000 | 6.91 | 0.5165 | 0.0654 | 0 | Không |
| `id01005/adversarial_attack/00285.wav` | `adversarial_attack` | 16,000 | 5.20 | 0.6746 | 0.0876 | 0 | Không |
| `id01004/bonafide/00004.wav` | `bonafide` | 16,000 | 2.62 | 0.7169 | 0.1054 | 0 | Không |
| `id01005/voice_conversion/id01005_vc_00008.wav` | `voice_conversion` | 40,000 | 4.16 | 0.3528 | 0.0636 | 0 | Không |
| `id01039/adversarial_attack/20135.wav` | `adversarial_attack` | 16,000 | 4.84 | 0.4173 | 0.0473 | 0 | Không |
| `id01039/adversarial_attack/20043.wav` | `adversarial_attack` | 16,000 | 4.76 | 0.5761 | 0.0561 | 0 | Không |
| `id01039/adversarial_attack/20214.wav` | `adversarial_attack` | 16,000 | 4.16 | 0.8770 | 0.0873 | 0 | Không |
| `id01129/replay/id01129_replay_00004.wav` | `replay` | 16,000 | 3.14 | 0.9993 | 0.1846 | 0 | Không |
| `id01129/bonafide/00029.wav` | `bonafide` | 16,000 | 2.81 | 0.5263 | 0.0624 | 0 | Không |
| `id01126/bonafide/00000.wav` | `bonafide` | 16,000 | 2.69 | 0.3026 | 0.0379 | 0 | Không |
| `id01128/replay/id01128_replay_00016.wav` | `replay` | 16,000 | 3.52 | 0.9997 | 0.2054 | 0 | Không |
| `id01126/bonafide/00032.wav` | `bonafide` | 16,000 | 2.41 | 0.5459 | 0.0695 | 0 | Không |
| `id01124/replay/id01124_replay_00035.wav` | `replay` | 16,000 | 2.56 | 0.7595 | 0.1439 | 0 | Không |

## Cảnh báo

- Sample rate không đồng nhất giữa các loại; đây là shortcut risk. Phải resample toàn bộ waveform về 16000 Hz.
- 6 shard chỉ là một phần nhỏ, không ngẫu nhiên của 432 shard; không suy rộng phân bố cục bộ cho toàn bộ snapshot.

## Quyết định

Có thể tiếp tục xây dựng pipeline và baseline. Mọi waveform phải được resample về mono 16 kHz trong pipeline; kết quả trên 6 shard chỉ dùng cho smoke test, không đại diện toàn bộ snapshot.
