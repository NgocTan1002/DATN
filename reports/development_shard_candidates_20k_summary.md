# Kết quả tải shard cho development manifest 20.000 mẫu

- Trạng thái: **ĐÃ TẢI VÀ XÁC MINH**
- Dataset: `VSASV-HF-public-snapshot-v1`
- Revision: `92de668616780ed4f3d7f58a82262f82b56b4557`
- Shard đã xác minh: **68**
- Audio cục bộ: **34.782 mẫu**, 10.580.529.098 byte (9,85 GiB)
- Đợt này đã tải: **62 shard**, 9.964.212.675 byte
- Metadata khớp: **34.782/34.782**
- Manifest hoàn tất: **20.000 mẫu**, dùng **66 shard**
- SHA-256 danh sách CSV: `80fa2001d37d969ac3b007deb7ed80d7322e38895457b543bb8d743b582f41d6`

## Quy tắc chọn và mở rộng

1. Tạo 48 chỉ số phân bố đều bằng `round(k*431/47)`, hợp với 6 shard đã có rồi loại trùng.
2. Khi coverage còn thiếu, lấy 12 shard từ lưới midpoint và xếp ổn định theo SHA-256 của `2026|shard`.
3. Khi audit còn thiếu `closed_dev/voice_conversion`, lấy 4 shard lân cận các shard đã biết chứa nhóm này. Không xem hoặc dùng `closed_test` để chọn.

Mỗi shard chỉ được chuyển khỏi `incoming/` sau khi kích thước, SHA-256, khả năng đọc Parquet và độ khớp metadata đều đạt.

## Ngân sách tải tham chiếu

Từ tốc độ tải thử 9.873.957,76 byte/giây, 9.964.212.675 byte cần tải có ETA khoảng 16,82 phút; cộng dự phòng 20% là 20,18 phút. Đây là ước lượng từ một shard thử, không phải thời lượng đo của toàn bộ đợt tải.

## Kết quả

Coverage đã đủ để tạo `data/manifests/development_20k_v1.csv`. Báo cáo kiểm tra và phân bố nằm tại `reports/development_manifest_20k.json` và `.md`.
