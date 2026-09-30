# Báo cáo smoke subset VSASV

- **Trạng thái kỹ thuật:** ĐẠT
- **Seed:** `2026`
- **Tổng mẫu:** 448
- **Mục đích:** chỉ kiểm tra code, không dùng làm kết quả khoa học

## Manifest

| Split | Mẫu | Speaker | Bonafide | Spoof | VC | AP | Replay | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `train` | 256 | 24 | 128 | 128 | 43 | 43 | 42 | `d7f60a692ef2c9572293d7dd65f69906cfecc0f8977d349b2a70acc99c43bf55` |
| `dev` | 128 | 4 | 64 | 64 | 0 | 32 | 32 | `ff4173e686733726f6e30faf8aa3fa3cc7037e38dfb3719ad4a483e10ee53dd9` |
| `test` | 64 | 4 | 32 | 32 | 0 | 0 | 32 | `1c7f7179b00584cac8dd3da2a80aa68041f16324c6ee26d85be5fd0fd4174c00` |

## Kiểm tra bắt buộc

- Cân bằng 50/50 trong từng partition: ĐẠT.
- File trùng: 0.
- Mọi file đều tồn tại trong năm shard cục bộ: ĐẠT.
- Speaker overlap train/dev/test: 0.
- Thứ tự chọn xác định bằng SHA-256 và seed `2026`.

## Nguồn cục bộ

| Split | Dòng closed split | Ứng viên cục bộ | Tỷ lệ |
|---|---:|---:|---:|
| `train` | 150,999 | 1,552 | 1.03% |
| `dev` | 39,116 | 719 | 1.84% |
| `test` | 30,848 | 287 | 0.93% |

## Cảnh báo

- Subset chỉ dùng để kiểm tra code và tốc độ, không dùng để báo cáo EER khoa học.
- Phân bố kiểu spoof bị giới hạn bởi năm shard hiện có; test cục bộ chỉ có replay.
