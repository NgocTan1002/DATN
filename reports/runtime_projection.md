# Dự báo thời gian chạy B0

## Số đo đầu vào

- Train end-to-end: **0.871 mẫu/giây** từ `batch8_workers0`.
- Đánh giá: **1.636 mẫu/giây** từ 128 mẫu trong 78.237 giây.
- Biên dự phòng kế hoạch: **+20%**.

## Nếu toàn bộ 20.000 hoặc 40.000 mẫu chạy cùng một chế độ

| Số mẫu | Chỉ train | Train + dự phòng | Chỉ đánh giá | Đánh giá + dự phòng |
|---:|---:|---:|---:|---:|
| 20,000 | 6.37 giờ | 7.65 giờ | 3.40 giờ | 4.07 giờ |
| 40,000 | 12.75 giờ | 15.30 giờ | 6.79 giờ | 8.15 giờ |

## Ngân sách development subset đề xuất

Phân bổ theo tỷ lệ closed train/dev hiện tại: **79.43% train / 20.57% dev**.

| Quy mô | Train | Dev | 1 epoch train | 1 lượt dev | Tổng | Tổng + dự phòng |
|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 15,885 | 4,115 | 5.06 giờ | 0.70 giờ | 5.76 giờ | 6.91 giờ |
| 40,000 | 31,770 | 8,230 | 10.13 giờ | 1.40 giờ | 11.52 giờ | 13.83 giờ |

## Toàn bộ closed protocol

| Giai đoạn | Số mẫu | Ước lượng | Có dự phòng |
|---|---:|---:|---:|
| 1 epoch closed train | 150,999 | 48.13 giờ | 57.76 giờ |
| 1 lượt closed dev | 39,116 | 6.64 giờ | 7.97 giờ |
| 1 lượt closed test | 30,848 | 5.24 giờ | 6.29 giờ |
| Tổng | 220,963 | 60.01 giờ | 72.01 giờ |

## Cấu hình DataLoader mặc định

Giữ `batch_size=8`, `num_workers=0` cho chặng B0 kế tiếp. Cấu hình 2 worker nhanh hơn 14,42% nhưng chưa vượt ngưỡng 15% và peak RSS cây tiến trình tăng từ 530,64 MiB lên 1.831,23 MiB. Đo lại sau khi thay cách lưu/đọc Parquet hoặc bổ sung cache; không suy rộng quyết định này sang máy khác.

## Giả định và giới hạn

- Thời gian tăng tuyến tính theo số mẫu và dùng cùng máy, đường đọc Parquet, LFCC và LCNN như benchmark ngày 02/10/2026.
- Train dùng throughput end-to-end gồm DataLoader, LFCC, forward, loss, backward và optimizer.
- Dev/test dùng throughput development của pilot B0; số đo này chỉ có một lần chạy 128 mẫu nên kém chắc chắn hơn benchmark train.
- Ước lượng chưa gồm tải dữ liệu, tạo manifest, kiểm tra leakage, lưu checkpoint hoặc thời gian gián đoạn.
- Biên dự phòng là ngân sách lập kế hoạch, không phải khoảng tin cậy thống kê.

Các con số trên dùng để lập lịch tài nguyên. Chúng không phải kết quả khoa học và không dự báo chắc chắn tốc độ trên đủ 432 shard.
