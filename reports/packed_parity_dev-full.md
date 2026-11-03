# Parity packed artifact với đường Parquet (T08)

- Ngày chạy: 2026-11-03T07:24:04+0700; trạng thái: **ĐẠT**.
- So bằng hàm production của `VSASVManifestDataset`; amplitude `peak`, đoạn 64.000 mẫu.
- Crop train so ở các cặp (seed, epoch): [[2026, 0], [2027, 1], [2028, 2]].

| Artifact | Dòng | Dòng so được | Phủ | Shard thiếu Parquet | Lệch metadata | Lệch waveform | Lệch sample eval | Lệch sample train | Đạt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dev-full | 38.777 | 7.035 | 18,14% | 154 | 0 | 0 | 0 | — | có |

## Giới hạn

- Dev-full/test-full chỉ so được dòng thuộc 68 shard còn Parquet cục bộ; phần còn lại dựa vào SHA-256 shard theo tree Hugging Face và checksum waveform ghi lúc nạp.
- Test-full chỉ được đọc để so dữ liệu; không chạy mô hình, metric hay thống kê nhãn.
- Parity chứng minh đầu vào mô hình giống hệt; không nói gì về kết quả huấn luyện.
