# Benchmark PoC artifact đóng gói

## Kết luận

- Trạng thái: **ĐẠT**
- Quyết định: **CHẤP NHẬN PoC**
- Phạm vi: benchmark kỹ thuật trên 24 mẫu cố định; không phải kết quả khoa học.
- Artifact: `C:\Users\Admin\Documents\DoAnTTNT\data\processed\packed_manifest_poc_20261013` (6.82 MiB), tạo trong 38.38 giây.
- Kiểm tra toàn vẹn: ĐẠT; 24 waveform được băm lại độc lập.

## Kết quả A/B

| Đường đo | Parquet (mẫu/s) | Packed (mẫu/s) | Cải thiện |
|---|---:|---:|---:|
| Loader-only | 0.653 | 245.871 | 37568.78% |
| B0 LFCC + LCNN | 0.621 | 19.853 | 3097.89% |

Mỗi biến thể chạy 3 lượt, batch 8, worker 0 và luân phiên thứ tự. Fingerprint metadata/waveform khớp tuyệt đối giữa hai đường đọc.

## Cổng quyết định

- Tương đương đầu ra: True.
- Cải thiện B0 tối thiểu 15%: True.
- CV packed không quá 15%: True.
- RSS tăng không quá 256 MiB: True.
- Manifest/split/seed/amplitude/crop không đổi: True.

## Diễn giải

PoC chỉ chứng minh đường đọc artifact cho cùng 24 mẫu và cùng B0. Không dùng kết quả này để suy ra thời gian toàn bộ 20k; cần benchmark lại sau khi tạo artifact đầy đủ.
