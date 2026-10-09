# Báo cáo development manifest 20k

- **Trạng thái:** ĐẠT
- **Phiên bản:** `development-20k-v2`
- **Seed:** `2026`
- **Mục tiêu:** 20,000 mẫu
- **Shard cục bộ:** 68
- **Audio cục bộ:** 34,782 mẫu

## Quota đã khóa

| Split | Tổng | Bonafide | Voice conversion | Adversarial | Replay |
|---|---:|---:|---:|---:|---:|
| `closed_train` | 15,885 | 7,134 | 4,348 | 4,348 | 55 |
| `closed_dev` | 4,115 | 1,524 | 1,289 | 1,289 | 13 |

## Ứng viên audio cục bộ

| Split | Bonafide | Voice conversion | Adversarial | Replay |
|---|---:|---:|---:|---:|
| `closed_train` | 10,562 | 7,173 | 4,887 | 145 |
| `closed_dev` | 2,097 | 2,965 | 1,906 | 80 |

## Khử trùng nội dung trước chọn quota

- File cục bộ đã băm: 34,782
- Nhóm content hash trùng: 76
- File thuộc nhóm trùng: 152
- File bị loại khỏi pool ứng viên: 76
- Quy tắc đại diện: file có `SHA256(2026|file)` nhỏ nhất trong mỗi nhóm.

| Split | Loại khỏi pool | Mất so với lựa chọn v1 | File bù |
|---|---:|---:|---:|
| `closed_train` | 63 | 28 | 28 |
| `closed_dev` | 13 | 10 | 10 |

## Manifest đã tạo

- Đường dẫn: `data/manifests/development_20k_v2.csv`
- SHA-256: `f549423b1fb33665f7e5606cdf56321bc8cd07246a0cdf0b90cd918c0c09e728`
- Số mẫu: 20,000
- Số speaker: 184
- Số shard được dùng: 66
- Speaker overlap train/dev: 0
- File lặp: 0
- Content hash lặp: 0
- Nhãn nhị phân, metadata và đường dẫn shard: ĐẠT

## Giới hạn diễn giải

- Manifest chỉ dùng closed_train và closed_dev; closed_test không tham gia chọn quy mô hoặc phân bố.
- Khử trùng nội dung không chứng minh hoặc sửa quan hệ giữa các speaker ID có cùng waveform.
- Duplicate đi qua closed_test phải được quyết định riêng trước đánh giá cuối.
- Quy mô và quota phản ánh bản phát hành công khai hiện có, không được diễn giải là tái lập dữ liệu trong bài báo gốc.
