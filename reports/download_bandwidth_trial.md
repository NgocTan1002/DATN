# Báo cáo đo băng thông tải VSASV

- Trạng thái: **ĐẠT**
- Shard: `train-00002-of-00432.parquet`
- Revision: `92de668616780ed4f3d7f58a82262f82b56b4557`
- Dung lượng: **70504633 byte**
- Thời gian: **7.14 giây**
- Tốc độ: **9.417 MiB/s** (78.992 Mb/s)
- SHA-256: `a48e840f19e51a54c0e3d86a7658679b979dc8a147b5ca8bd76d1c069bcdf73e`
- Đối chiếu checksum: **khớp SHA-256 nguồn**
- ETA tuyến tính cho 20.000 mẫu theo byte/mẫu: **0.11 giờ**
- ETA tuyến tính cho 40.000 mẫu theo byte/mẫu: **0.23 giờ**
- Ghi chú mạng: Wi-Fi; tình trạng gián đoạn không được ghi nhận trong lần đo.

ETA dùng dự báo dung lượng mới nhất từ `reports/local_storage_audit.json` và một phép đo tải. Đây là số liệu lập kế hoạch, không phải kết quả khoa học.

## ETA theo phương án shard ứng viên 20.000 mẫu

- Shard ứng viên: **52**.
- Shard đã có: **6**.
- Shard cần tải: **46**.
- Dung lượng thực theo danh sách shard còn thiếu: **7.608.382.672 byte** (7,09 GiB).
- ETA theo tốc độ đã đo: **0,21 giờ** (12,84 phút).
- ETA có dự phòng 20%: **0,26 giờ** (15,41 phút).

ETA này phản ánh lượng byte của `reports/development_shard_candidates_20k.csv`. Danh sách vẫn là phương án tải; coverage 20.000 mẫu train + development chỉ được xác nhận sau khi tải và audit manifest.
