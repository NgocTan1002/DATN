# Báo cáo audio smoke test VSASV

- **Trạng thái kỹ thuật:** ĐẠT
- **Shard cục bộ:** 1/432
- **Utterance:** 512
- **Speaker trong 1 shard:** 7
- **Sample rate đích bắt buộc:** 16,000 Hz

## Kiểm tra toàn bộ 1 shard

- Đường dẫn duy nhất: 512/512
- Thiếu file/label/utt_type: 0
- Waveform thiếu hoặc rỗng: 0
- Sample rate không hợp lệ: 0

## Phân bố sample rate

| Loại | Sample rate | Số mẫu |
|---|---:|---:|
| `bonafide` | 16,000 Hz | 512 |

## Thời lượng

| Loại | Min (s) | Median (s) | Mean (s) | P95 (s) | Max (s) | Tổng giờ |
|---|---:|---:|---:|---:|---:|---:|
| `bonafide` | 0.84 | 2.84 | 4.31 | 12.44 | 34.41 | 0.61 |

## Kiểm tra waveform mẫu

Đã đọc và kiểm tra số học 8 waveform được chọn xác định theo từng cặp shard/loại bằng hash(file).

| File | Loại | Hz | Giây | Peak | RMS | Non-finite | Silent |
|---|---|---:|---:|---:|---:|---:|---|
| `id00026/00073.wav` | `bonafide` | 16,000 | 3.24 | 1.0000 | 0.2152 | 0 | Không |
| `id00026/00135.wav` | `bonafide` | 16,000 | 1.64 | 0.7535 | 0.1217 | 0 | Không |
| `id00024/00049.wav` | `bonafide` | 16,000 | 2.44 | 0.5837 | 0.0962 | 0 | Không |
| `id00024/00044.wav` | `bonafide` | 16,000 | 5.24 | 0.6836 | 0.1043 | 0 | Không |
| `id00021/00042.wav` | `bonafide` | 16,000 | 2.44 | 0.4096 | 0.0718 | 0 | Không |
| `id00024/00033.wav` | `bonafide` | 16,000 | 26.82 | 0.6582 | 0.1006 | 0 | Không |
| `id00024/00018.wav` | `bonafide` | 16,000 | 2.44 | 0.6404 | 0.1246 | 0 | Không |
| `id00026/00062.wav` | `bonafide` | 16,000 | 6.84 | 0.6568 | 0.1040 | 0 | Không |

## Cảnh báo

- 1 shard chỉ là một phần nhỏ, không ngẫu nhiên của 432 shard; không suy rộng phân bố cục bộ cho toàn bộ snapshot.

## Quyết định

Có thể tiếp tục xây dựng pipeline và baseline. Mọi waveform phải được resample về mono 16 kHz trong pipeline; kết quả trên 1 shard chỉ dùng cho smoke test, không đại diện toàn bộ snapshot.
