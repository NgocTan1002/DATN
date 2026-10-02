# So sánh cấu hình DataLoader B0

## Giao thức

- Seed: `2026`
- Mẫu cố định: 16
- Số lần lặp: 3
- Amplitude policy: `none`
- Mỗi cấu hình chạy đủ DataLoader, LFCC, LCNN, loss, backward và optimizer.
- File, label và SHA-256 waveform được đối chiếu với benchmark P0.

## Kết quả

| Cấu hình | Batch | Worker | Mẫu/giây | So với chuẩn | Trung vị/lần | Chờ dữ liệu | Peak RSS cây tiến trình | Tái lập |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch4_workers0 | 4 | 0 | 0.882 | +1.25% | 18104.92 ms | 93.92% | 438.23 MiB | ĐẠT |
| batch8_workers0 | 8 | 0 | 0.871 | +0.00% | 18405.62 ms | 94.72% | 530.64 MiB | ĐẠT |
| batch16_workers0 | 16 | 0 | 0.814 | -6.57% | 20082.94 ms | 95.10% | 837.75 MiB | ĐẠT |
| batch8_workers2 | 8 | 2 | 0.997 | +14.42% | 15765.62 ms | 90.95% | 1831.23 MiB | ĐẠT |

## Quyết định

- Cấu hình chuẩn: `batch8_workers0`.
- Cấu hình nhanh nhất: `batch8_workers2`.
- Mức cải thiện: **14.42%**.
- Ngưỡng chấp nhận: 15.00%.
- Kết quả: **GIỮ CẤU HÌNH CHUẨN**.
- Khuyến nghị: Giữ batch8_workers0 vì chưa có cấu hình vượt ngưỡng cải thiện mà vẫn đạt mọi cổng chất lượng.

## Diễn giải

So sánh này chỉ thay đổi batch size hoặc số worker trên cùng danh sách mẫu. Cấu hình chỉ được chấp nhận khi checksum waveform, nhãn, thứ tự file và tính hữu hạn của loss/gradient đều đạt. Số liệu là benchmark kỹ thuật trên smoke subset, không phải kết quả khoa học của mô hình.
