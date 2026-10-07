# Báo cáo development manifest 20k

- **Trạng thái:** ĐẠT
- **Phiên bản:** `development-20k-v1`
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

## Manifest đã tạo

- Đường dẫn: `data/manifests/development_20k_v1.csv`
- SHA-256: `63df695e232a424c95ebd869f1d71cae715040ec30428177967cac26c6c5e673`
- Số mẫu: 20,000
- Số speaker: 184
- Số shard được dùng: 66
- Speaker overlap train/dev: 0
- File lặp: 0
- Nhãn nhị phân, metadata và đường dẫn shard: ĐẠT

## Giới hạn diễn giải

- Manifest chỉ dùng closed_train và closed_dev; closed_test không tham gia chọn quy mô hoặc phân bố.
- Quy mô và quota phản ánh bản phát hành công khai hiện có, không được diễn giải là tái lập dữ liệu trong bài báo gốc.
