# Báo cáo đóng gói packed artifact train ×2 (`train_development_scale2_v1`, D036)

- Ngày tạo: `2026-11-18T16:55:18+0700`
- Trạng thái tổng thể: **ĐẠT**
- Định dạng & giai đoạn: `vsasv-packed-v1`, `post_resample_pre_amplitude_pre_crop`, `dtype=float32-le`, `crop_seed=2026`.
- Manifest nguồn: `development_scale2_v1.csv` (`manifest_version=development-scale2-v1`, SHA `6533fced4b3a25ebab9f037a62d34774b44c2fa1f00d17040fd815bd17fa51a3`).
- Số mẫu / tổng số mẫu float32 / số shard: `31.770` / `3.030.945.120` / `89` (`verified_rows=31.770`).
- Kiểm `89` shard Parquet vs `vsasv_shard_tree.json` trước đóng gói: `all_matched=True` (`15.217.312.359` B).
- Thời gian đóng gói & băm lại: build `86,535 s`, verify `48,885 s`.

## 1. Kích thước và SHA-256 3 file artifact

| File | Kích thước (B) | GiB | SHA-256 |
|---|---:|---:|---|
| `index.csv` | 6.280.248 | 0,006 | `7eb18914d7d2ea3b101273d0ab4a4e882a0cac4c8676ae5c018a281086aa8080` |
| `metadata.json` | 964 | 0,000 | `3341f75b45f089c4b6b2f71ca3659233a87c6232d416a247b5886d7e0ecec48d` |
| `waveforms.f32` | 12.123.780.480 | 11,291 | `260d93d3762f20b564400d023d599d4151d8e74be618441aed255675f3e3da5e` |

- Tổng kích thước dự kiến upload Kaggle: `12.130.061.692` B (`12.130 GB` / `11.297 GiB`).

## 2. So bit 15.885 mẫu đầu với ×1 (`train_development_20k_v2`)

- Kết quả: **15885/15885** dòng trùng toàn bộ trường `index.csv` và trùng từng byte `waveforms.f32` (`passed=True`, thời gian `8.677 s`).

## 3. Thử nạp ngẫu nhiên tất định 200 mẫu (`PackedManifestDataset`)

- Số mẫu kiểm: `200` (seed `2026`); shape `[64000]`, dtype `torch.float32`, hữu hạn `all_finite=True`, nhãn & metadata khớp `True`.
- Thời gian đọc: `eval=1.477 s`, `training=0.947 s` (tổng `2.424 s`).
- Số mẫu tiền tố (< 15.885) trùng khớp tuyệt đối với ×1 khi nạp: `eval=80`, `training=80`.

## 4. Parity và kiểm chứng D017

- Báo cáo chi tiết: `reports/packed_parity_train_scale2_v1.json` (trạng thái: **ĐẠT**, tổng thời gian `5850.95 s`).
- Parity Parquet ↔ Artifact (`31.770/31.770`): `raw_mismatches=0`, `metadata_mismatches=0`, `eval_item_mismatches=0`, `train_item_mismatches={'seed2026_epoch0': 0, 'seed2027_epoch1': 0, 'seed2028_epoch2': 0}`.
- Hash D017 trực tiếp từ `waveforms.f32` (nhóm `native_sample_rate=16.000 Hz`): `23073/23073` khớp (nhóm `40.000 Hz` không áp dụng trực tiếp do artifact đã resample xuống `16.000 Hz` `float32`).
- Hash D017 bắc cầu toàn vẹn Parquet (`float64`) ↔ audit D017 & ↔ lát packed (`float32` `16.000 Hz`): `31770/31770` khớp (`0` lệch).

## 5. Chuẩn bị metadata dataset Kaggle (chưa upload)

- File: `.cache/scale2_03/kaggle_dataset/dataset-metadata.json` (SHA `3c18a16c3b021c7ffb4069cae49bb763cb6c9c0fecaf949de23594da8e717867`).
- `id`: `ngctnnguyn1111/doanttnt-scale2-packed-v1` | `title`: `DoAnTTNT scale2 packed v1` | `licenses`: `['CC0-1.0']` | `isPrivate`: `True`.
- Nội dung dự kiến: thư mục `data/processed/packed/train_development_scale2_v1` (3 file, tổng `11.297 GiB`).
