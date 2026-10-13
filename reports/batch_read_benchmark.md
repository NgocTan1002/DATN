# Benchmark P0.2 — batch-read theo shard

## Kết luận

- Quyết định: **LOẠI BATCH-READ**.
- Cải thiện throughput end-to-end: **-0.93%**; cổng yêu cầu ít nhất 15.00%.
- Chênh peak RSS: **+5.96 MiB**.
- Hook batch-read trong Dataset: **đã gỡ**.
- Đây là benchmark kỹ thuật trên CPU, không phải kết quả khoa học.

## Giao thức đo

- Manifest: `development-20k-v2`; split `closed_train`.
- Seed/epoch/policy: `2026` / `0` / `none`.
- Workload end-to-end: 24 mẫu, batch 8, worker 0, 3 lần lặp.
- Số shard từng batch theo thứ tự shuffle cố định: [8, 8, 8].
- Đường end-to-end gồm DataLoader, LFCC, LCNN forward, BCE loss, backward và Adam step.
- Không xóa cache hệ điều hành; thứ tự hai biến thể được đảo xen kẽ. Vì vậy không gọi lượt đầu là cold cache.

## Micro-benchmark

| Case | Biến thể | File | Shard | Truy vấn từng lượt | Trung vị batch | Mẫu/giây |
|---|---|---:|---:|---|---:|---:|
| `same_shard` | `per_item` | 8 | 1 | [8, 8, 8] | 16168.77 ms | 0.495 |
| `same_shard` | `batch_read` | 8 | 1 | [1, 1, 1] | 2472.54 ms | 3.235 |
| `dispersed` | `per_item` | 8 | 8 | [8, 8, 8] | 11665.77 ms | 0.686 |
| `dispersed` | `batch_read` | 8 | 8 | [8, 8, 8] | 11664.28 ms | 0.686 |

Batch cùng shard chứng minh ứng viên giảm 8 truy vấn còn 1; batch phân tán giữ 8 truy vấn. Cả hai case dùng cùng file và checksum đầu ra giữa hai đường đọc.

## A/B pipeline end-to-end

| Biến thể | Mẫu/giây | CV throughput | Trung vị batch | First batch | Truy vấn từng lượt | Peak RSS |
|---|---:|---:|---:|---:|---|---:|
| `per_item` | 0.638 | 7.82% | 14004.19 ms | 14004.19 ms | [24, 24, 24] | 575.54 MiB |
| `batch_read` | 0.632 | 5.78% | 14116.43 ms | 14116.43 ms | [24, 24, 24] | 581.50 MiB |

- File, speaker, `utt_type`, nhãn, native sample rate và checksum waveform: **ĐẠT**.
- Checksum danh sách đầu ra: `6c4a068279ab946470f85580d67e45b3588c5b5188e3d10afe841615263d4800`.
- Cổng ổn định (CV ≤ 15.00%): **ĐẠT**.
- Cổng bộ nhớ (tăng ≤ 256.00 MiB): **ĐẠT**.
- Cổng số truy vấn: **ĐẠT**.

## Quyết định và phạm vi

Không dùng __getitems__; giữ D014 và kích hoạt P1 artifact đóng gói.

Không cập nhật `reports/runtime_projection.*`: workload 24 mẫu chỉ dùng để ra quyết định A/B, và cấu hình được giữ sau P0.2 vẫn là đường đọc từng item đã dùng cho dự báo ngày 02/10. P1 tiếp theo là đề xuất artifact dẫn xuất đúng manifest 20k, cache sau decode/resample nhưng trước amplitude policy và crop.
