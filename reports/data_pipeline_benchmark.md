# Benchmark pipeline dữ liệu B0

## Kết quả

- Trạng thái: **ĐẠT**
- Thời điểm: `2026-10-02T14:32:05+07:00`
- Thiết bị: `cpu`
- Seed: `2026`
- Amplitude policy: `none`
- Số mẫu cố định: 24
- Số lần lặp: 3
- Batch size: 8
- DataLoader workers: 0
- Kiểm tra waveform/file/label tái lập: **ĐẠT**

## Khởi tạo Dataset

| Chỉ số | Giá trị |
|---|---:|
| Lần đầu | 2452.851 ms |
| Trung vị các lần sau | 534.238 ms |
| P95 toàn bộ | 2266.575 ms |
| Mẫu local/split | 256/256 |

## Đo trực tiếp từng giai đoạn trên một mẫu

| Giai đoạn | Trung vị | P95 | Tỷ lệ trong tiền xử lý |
|---|---:|---:|---:|
| Đọc/decode Parquet | 925.361 ms | 1386.687 ms | 99.82% |
| Resample | 0.001 ms | 3.076 ms | 0.00% |
| Amplitude policy | 1.476 ms | 2.681 ms | 0.16% |
| Cắt/lặp 64.000 mẫu | 0.148 ms | 0.468 ms | 0.02% |

## Đường train theo batch

| Giai đoạn | Trung vị/batch | P95/batch | Tỷ lệ đường train |
|---|---:|---:|---:|
| DataLoader | 7723.815 ms | 8268.783 ms | 95.31% |
| LFCC | 7.129 ms | 8.126 ms | 0.09% |
| LCNN forward | 96.985 ms | 97.459 ms | 1.20% |
| Loss | 6.428 ms | 11.265 ms | 0.08% |
| Backward | 265.166 ms | 301.615 ms | 3.27% |
| Optimizer | 4.441 ms | 15.680 ms | 0.05% |

DataLoader warm throughput: **1.037 mẫu/giây**.

## Điểm nghẽn

Giai đoạn lớn nhất theo trung vị đường train là **data_loader**, chiếm khoảng **95.31%** tổng thời gian các giai đoạn được đo. Trong phần đọc và tiền xử lý một mẫu, giai đoạn lớn nhất là **read_decode**.

Kết luận này chỉ áp dụng cho cấu hình CPU, 24 mẫu smoke, batch size 8 và `num_workers=0`. Đây là benchmark kỹ thuật, không phải kết quả mô hình.

## Tài nguyên và môi trường

- Python: `3.12.14`
- PyTorch: `2.11.0+cpu`
- CPU: `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`
- Logical CPU: 16
- RSS đầu: 211.40 MiB
- RSS lớn nhất quan sát: 532.41 MiB
- RSS cuối: 532.41 MiB
- Tổng thời gian benchmark: 150.953 giây

## Phạm vi sử dụng

Benchmark dùng cùng smoke manifest, seed và policy với pilot ngày 01/10. Số liệu dùng để chọn bước tối ưu tiếp theo; không ngoại suy thành tốc độ chắc chắn trên toàn bộ 432 shard nếu chưa kiểm tra phân bố shard và cache hệ điều hành.
