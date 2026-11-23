# Parity packed artifact với đường Parquet (T08)

- Ngày chạy: 2026-11-18T20:22:15+0700; trạng thái: **ĐẠT**.
- So bằng hàm production của `VSASVManifestDataset`; amplitude `peak`, đoạn 64.000 mẫu.
- Crop train so ở các cặp (seed, epoch): [[2026, 0], [2027, 1], [2028, 2]].

| Artifact | Dòng | Dòng so được | Phủ | Shard thiếu Parquet | Lệch metadata | Lệch waveform | Lệch sample eval | Lệch sample train | Đạt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train-scale2 | 31.770 | 31.770 | 100,00% | 0 | 0 | 0 | 0 | 0 | có |

## Kiểm chứng content hash D017 (train-scale2)

- Nguồn audit đối chiếu: `reports/development_manifest_scale2_v1_content_audit.json` (SHA `060a7833b0cc35e4390f8c624c20030695e3260ced2e519e368abf74063be861`).
- Kiểm lát float32 trong `waveforms.f32` khớp `index.csv`: `mismatches=0`.
- Băm trực tiếp công thức D017 trên `waveforms.f32` (`native_sample_rate=16000`): `23073/23073` khớp (lệch `0`).
- Băm trực tiếp công thức D017 trên `waveforms.f32` (`native_sample_rate=40000`): `0/8697` khớp (lệch `8697`).
- Giải thích trực tiếp: Artifact vsasv-packed-v1 lưu waveform ở giai đoạn post_resample_pre_amplitude_pre_crop (16.000 Hz, float32-le). Với các mẫu gốc 16.000 Hz (bonafide, adversarial_attack, replay), hàm D017 trên lát float32 khớp trực tiếp khi không sai số biểu diễn; với nhóm voice_conversion (40.000 Hz gốc đã resample xuống 16.000 Hz), không thể băm lại trực tiếp định nghĩa D017 của tín hiệu 40.000 Hz float64 chỉ từ lát 16.000 Hz float32.
- Kiểm chứng bắc cầu qua Parquet gốc (`float64` -> D017 & `float32/16kHz` -> `waveforms.f32`): `31770/31770` khớp audit D017 (`passed=True`).

## Giới hạn

- Dev-full/test-full chỉ so được dòng thuộc 68 shard còn Parquet cục bộ; phần còn lại dựa vào SHA-256 shard theo tree Hugging Face và checksum waveform ghi lúc nạp.
- Test-full chỉ được đọc để so dữ liệu; không chạy mô hình, metric hay thống kê nhãn.
- Parity chứng minh đầu vào mô hình giống hệt; không nói gì về kết quả huấn luyện.
