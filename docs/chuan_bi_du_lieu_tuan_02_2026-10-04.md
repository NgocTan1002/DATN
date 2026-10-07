# Chuẩn bị dữ liệu tuần 02 — 04/10/2026

## 1. Mục đích

Tài liệu này chuẩn hóa ba việc trước khi bước vào tuần 02:

1. dùng kết quả kiểm kê cục bộ để lập ngân sách lưu trữ;
2. quy định cách đo băng thông tải một shard;
3. khóa schema nháp cho manifest development subset mà không sửa các split hiện có.

## 2. Kết quả kiểm kê cục bộ

Nguồn số liệu: `reports/local_storage_audit.json`, sinh ngày 04/10/2026 bằng `scripts/audit_local_storage.py`.

| Thuộc tính | Kết quả |
|---|---:|
| Shard hiện có | 5/432 |
| Coverage theo số shard | 1,16% |
| Mẫu cục bộ | 2.558 |
| Khớp metadata | 2.558/2.558 |
| Tổng dung lượng | 545.811.790 byte, tương đương 520,53 MiB |
| Trung bình | 213.374,43 byte/mẫu |
| Dung lượng đĩa còn trống lúc đo | 117,91 GiB |

Dự báo tuyến tính theo byte/mẫu quan sát được:

| Mục tiêu | Dung lượng Parquet dự báo |
|---:|---:|
| 20.000 mẫu | 3,97 GiB, tương đương 4,27 GB |
| 40.000 mẫu | 7,95 GiB, tương đương 8,53 GB |
| 220.963 mẫu | 43,91 GiB, tương đương 47,15 GB |

Năm shard không phải mẫu ngẫu nhiên của 432 shard. Các con số trên chỉ là ngân sách lưu trữ ban đầu, chưa gồm cache, file trung gian, checkpoint và khoảng trống an toàn.

## 3. Giao thức đo băng thông một shard

### 3.1. Điều kiện chọn shard thử

- Shard chưa có cục bộ, thuộc `VSASV-HF-public-snapshot-v1`.
- URL tải trực tiếp và nguồn phát hành phải được ghi lại.
- Nguồn phải cung cấp kích thước; checksum hoặc ETag được ghi nếu có.
- Không chọn shard chỉ vì có kích thước nhỏ nhất.
- File được tải vào `data/raw/vsasv_parquet/incoming/` và chỉ chuyển sang thư mục dữ liệu sau khi kiểm tra hoàn tất.

### 3.2. Trường phải ghi

| Trường | Ý nghĩa |
|---|---|
| `source_url` | URL chính xác của shard |
| `shard` | Tên shard |
| `started_at`, `finished_at` | Thời điểm bắt đầu và kết thúc có múi giờ |
| `elapsed_seconds` | Thời gian monotonic của lần tải |
| `downloaded_bytes` | Số byte thực nhận |
| `bytes_per_second` | `downloaded_bytes / elapsed_seconds` |
| `mib_per_second` | `bytes_per_second / 1024²` |
| `megabits_per_second` | `bytes_per_second × 8 / 10⁶` |
| `sha256` | SHA-256 tính sau tải |
| `source_checksum_or_etag` | Giá trị do nguồn công bố, nếu có |
| `checksum_status` | Khớp, không khớp hoặc nguồn không cung cấp |
| `network_notes` | Wi-Fi/LAN, gián đoạn hoặc giới hạn tốc độ quan sát được |

### 3.3. Tính ETA sau phép đo

Với `B` là số byte mục tiêu và `v` là tốc độ thực đo theo byte/giây:

```text
ETA_seconds = B / v
ETA_hours = ETA_seconds / 3600
```

Tính riêng ETA cho khoảng 4.267.488.585 byte của mốc 20.000 mẫu và 8.534.977.170 byte của mốc 40.000 mẫu. Các giá trị byte mục tiêu phải được lấy lại từ JSON kiểm kê khi chạy, không chép cố định vào script đo.

### 3.4. Cổng chấp nhận

- File tải xong có đúng kích thước.
- SHA-256 được ghi; nếu nguồn có checksum thì phải khớp.
- DuckDB đọc được schema và đếm được hàng.
- Mọi file/label/`utt_type` trong shard khớp metadata.
- Báo cáo phải ghi số byte và thời gian thực; không suy đoán băng thông khi chưa tải.

Đầu ra dự kiến của lần đo: `reports/download_bandwidth_trial.json` và `reports/download_bandwidth_trial.md`.

## 4. Schema nháp cho development manifest

Manifest là lớp giao giữa split đã khóa và audio thực tế có thể truy cập. Manifest không thay thế hoặc sửa `closed_train.csv`, `closed_dev.csv` hay `closed_test.csv`.

| Cột | Kiểu | Nguồn/quy tắc |
|---|---|---|
| `manifest_version` | chuỗi | Phiên bản manifest, ví dụ `dev-v1-draft` |
| `shard` | chuỗi | Tên Parquet chứa waveform |
| `file` | chuỗi | Khóa file từ metadata/split |
| `speaker_id` | chuỗi | Trường `label` gốc trong metadata VSASV |
| `split` | enum | `closed_train`, `closed_dev` hoặc `closed_test` từ split đã khóa |
| `binary_label` | số nguyên | `0` nếu `utt_type=bonafide`, ngược lại `1` |
| `utt_type` | enum | `bonafide`, `voice_conversion`, `adversarial_attack` hoặc `replay` |
| `native_sample_rate` | số nguyên | Đọc từ Parquet để audit; không dùng làm đặc trưng |
| `source_snapshot` | chuỗi | Luôn là `VSASV-HF-public-snapshot-v1` |

### 4.1. Quy tắc tạo

1. Lấy giao của split đã khóa với danh sách audio thực tế có thể truy cập.
2. Không đổi speaker sang split khác để cân bằng dữ liệu.
3. Không dùng closed test để chọn quy mô hoặc phân bố development subset.
4. Khi cần lấy mẫu trong một split/stratum, xếp ổn định theo SHA-256 của `2026|file`.
5. Ghi số mẫu thực tế sau giao; không dùng quy mô danh nghĩa thay cho số hàng manifest.
6. Lưu checksum của toàn file manifest trong báo cáo đi kèm.

### 4.2. Kiểm tra bắt buộc

- `file` duy nhất trong manifest.
- Mỗi speaker chỉ thuộc một split.
- Mọi dòng khớp metadata về speaker và `utt_type`.
- Mọi dòng trỏ tới shard tồn tại và đọc được.
- `binary_label` khớp quy tắc nhị phân đã khóa.
- Có bảng đếm theo split, nhãn, `utt_type`, speaker và shard.

## 5. Phương án xử lý trạng thái khoa học sau D015

Các vị trí hiện dùng `provisional_public_snapshot`:

- `reports/vsasv_snapshot_verification.json` tại trường `scientific_status`;
- `reports/vsasv_snapshot_verification.md` tại dòng “Giá trị khoa học”.

D015 xác nhận đây là bản phát hành công khai chính thức duy nhất hiện còn, nhưng snapshot vẫn không tương đương dữ liệu mô tả trong bài báo gốc.

**Phương án đề xuất:** giữ `scientific_status = provisional_public_snapshot` để bảo toàn giới hạn khoa học; bổ sung một trường riêng như `release_status = official_current_public_release` và giữ tuyên bố `paper_equivalent = false` khi người dùng cho phép sửa báo cáo/script. Cách này phân biệt rõ trạng thái phát hành với mức tương đương khoa học.

Chưa sửa hai báo cáo xác minh hoặc script sinh báo cáo trong phiên này vì D015 yêu cầu người dùng quyết định trước.

## 6. Việc đầu tiên ngày 05/10/2026

Chọn một shard thiếu có URL, kích thước và checksum/ETag từ nguồn phát hành; chạy phép tải thử theo mục 3; sau đó dùng tốc độ thực đo để tính ETA cho mốc 20.000 và 40.000 mẫu.
