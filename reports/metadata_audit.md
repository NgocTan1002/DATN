# Báo cáo audit metadata VSASV

- **Trạng thái:** ĐẠT
- **Metadata:** `data/metadata/vsasv_metadata.csv`
- **SHA-256:** `254887ac5a1e357cbe1a8f0ee4bde266c491c9c349505e61c5b3dd2b461e73fb`
- **Thời điểm UTC:** `2026-09-01T10:38:50.840141+00:00`

## Tính toàn vẹn

| Kiểm tra | Kết quả |
|---|---:|
| Tổng số dòng | 220,963 |
| Đường dẫn duy nhất | 220,963 |
| Dòng trùng hoàn toàn | 0 |
| Đường dẫn lặp | 0 |
| Giá trị thiếu | 0 |
| Speaker duy nhất | 1,141 |
| Mẫu/speaker nhỏ nhất | 1 |
| Mẫu/speaker trung vị | 64 |
| Mẫu/speaker lớn nhất | 6,720 |
| Label khớp prefix file | 100.00% |

## Phân bố `utt_type`

| utt_type | Số mẫu | Tỷ lệ | Số speaker |
|---|---:|---:|---:|
| `adversarial_attack` | 60,949 | 27.58% | 144 |
| `bonafide` | 98,305 | 44.49% | 1,141 |
| `replay` | 760 | 0.34% | 19 |
| `voice_conversion` | 60,949 | 27.58% | 144 |

## Phân bố nhị phân

| Nhóm | Số mẫu | Tỷ lệ |
|---|---:|---:|
| `bonafide` | 98,305 | 44.49% |
| `spoof` | 122,658 | 55.51% |
| `unclassified` | 0 | 0.00% |

## Vấn đề phát hiện

Không phát hiện lỗi schema hoặc tính toàn vẹn trong các kiểm tra đã chạy.

## Giới hạn metadata

- Không có `generator_id` hoặc `source_id`.
- Không có nhóm TTS trong snapshot hiện tại.
- Không thể dùng metadata này để tuyên bố tổng quát hóa trên unseen TTS engine.
