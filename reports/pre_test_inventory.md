# PREP-01 — Kiểm kho test-full, dataset Kaggle và ước tính chi phí chấm test

- Thời điểm tạo (UTC): `2026-11-18T03:03:23.821534+00:00`
- Cam kết dữ liệu test: **Không đọc nhãn, không parse dòng/cột của `index.csv` hay manifest test, không giải mã audio test** (`no_test_labels_or_content_read = True`).
- Cổng test: `test_gate_open = False` (D032) và `False` (D035).

## 1. Kho test-full cục bộ (`closed_test_full_v1`)

- Kết luận: **KHỚP TOÀN BỘ** (`artifact_count = 30698`, `chunk_shard_count = 162`, `total_samples = 3060286845`).
- Kiểm kích thước `waveforms.f32`: `12,241,147,380` byte = `3,060,286,845 × 4` byte (`khớp = True`).

| File | Đường dẫn | Kích thước (byte) | SHA-256 thực đo (luồng byte) | Khớp metadata/config/bối cảnh |
|---|---|---:|---|---|
| Manifest | `data/manifests/closed_test_full_v1.csv` | 4,538,521 | `0a7a70bc8ededdd83e5f6df06580fc954c6f2da848366e37bc265331ca17d266` | True |
| Index | `data/processed/packed/closed_test_full_v1/index.csv` | 5,955,247 | `8f862032b6af42e1d35ff7d9e862308afb781308d13f24178a4bb1409237c78a` | True |
| Waveforms | `data/processed/packed/closed_test_full_v1/waveforms.f32` | 12,241,147,380 | `556bb99de8c39ed8f8568ae36d170907f4ac2959b9c4ab2f1a8d4e48423b32b2` | True |
| Metadata | `data/processed/packed/closed_test_full_v1/metadata.json` | 960 | `7e16da487f4d066f9694acd62b099ab4b70cc9158429b21c6e85c5fd1185ee9b` | Đã đọc JSON hợp lệ |

## 2. Trạng thái Kaggle (chỉ đọc)

- Quota GPU tuần hiện tại: còn **16.06h** / **45.00h** (đã dùng `28.94h`, làm mới lúc `2026-11-21T00:00:00` UTC; log `.cache/prep01/kaggle_quota.log`).
- Tổng số dataset thuộc tài khoản: `10` (log `.cache/prep01/kaggle_datasets_list.log`).
- Kết luận `closed_test_full_v1` trên Kaggle: **CHƯA CÓ**.
- Ghi chú: Lệnh chỉ đọc 'kaggle -W datasets files' chỉ trả tên và kích thước byte (không trả SHA-256). Không tìm thấy thư mục hay file nào tên closed_test_full_v1 trong tài khoản Kaggle.

| Dataset ứng viên (`packed`/`test`/`eval`) | Kích thước nén (byte) | File từ xa | Kích thước file (byte) |
|---|---:|---|---:|
| `ngctnnguyn1111/doanttnt-t14-packed` | 13,757,151,484 | `closed_dev_full_v1/index.csv` | 7,684,400 |
| `ngctnnguyn1111/doanttnt-t14-packed` | 13,757,151,484 | `closed_dev_full_v1/metadata.json` | 957 |
| `ngctnnguyn1111/doanttnt-t14-packed` | 13,757,151,484 | `closed_dev_full_v1/waveforms.f32` | 13,454,011,384 |
| `ngctnnguyn1111/doanttnt-t14-packed` | 13,757,151,484 | `train_development_20k_v2/index.csv` | 3,121,751 |
| `ngctnnguyn1111/doanttnt-t14-packed` | 13,757,151,484 | `train_development_20k_v2/metadata.json` | 958 |
| `ngctnnguyn1111/doanttnt-t14-packed` | 13,757,151,484 | `train_development_20k_v2/waveforms.f32` | 5,792,780,884 |

## 3. Ước tính chi phí chấm test-full

- Hệ số quy đổi số mẫu: `30.698 / 38.777 = 0.791655`.
- **Xấp xỉ cho `roundtrip_40k` (không đọc nhãn test)**: dùng tỷ lệ bonafide trên `dev-full` (`14,151 / 38,777 = 0.364933`, nguồn `reports/shortcut_probe_dev.json`) → ước lượng khoảng `~11,203` mẫu bonafide trong `test-full`.

### 3a. Dự kiến GPU Kaggle Tesla T4 (X0, X1, X1R)

| Họ mô hình | TB 1 lượt dev-full thực đo (s) | Quy đổi 1 lượt test-full (s / h) | Số lượt test-full (3 seed) | GPU-giờ gốc | GPU-giờ (+25% an toàn) |
|---|---:|---:|---:|---:|---:|
| X0 | 1949.25 (30 lượt) | 1543.13 s / 0.4286 h | 3.00 (`original` ×3) | 1.286 | 1.607 |
| X1 (`X1-L8-lr3e-5`) | 1902.52 (21 lượt) | 1506.14 s / 0.4184 h | 10.0948 (xấp xỉ D032) | 4.223 | 5.279 |
| X1R (`X1-L8-lr3e-5-lp7k`) | 2052.20 (26 lượt) | 1624.64 s / 0.4513 h | 3.00 (`lowpass_7k` ×3) | 1.354 | 1.692 |
| **Tổng GPU T4 (gốc D032+D035)** | — | — | **16.0948** | **6.863** | **8.579** |

- **Độ nhạy chi phí lọc lowpass trên X1**: nếu 6 lượt `lowpass_7k`/`lowpass_4k` của X1 cộng thêm thời gian lọc CPU trong loader (như X1R thực đo `2052.20` s/dev-full), tổng chi phí gốc là **7.061 GPU-h** (cộng 25% an toàn = **8.826 GPU-h**).
- **Lưu ý script `evaluate_full_split.py`**: hiện tại hàm `Models.score` chạy qua toàn bộ `30.698` mẫu cả ở điều kiện `roundtrip_40k` (tức `12.0` lượt cho X1 thay vì `10.0948` lượt). Nếu chạy nguyên trạng hiện tại (18 lượt tổng X0+X1+X1R) và tính overhead lowpass cho 6 lượt X1: gốc **7.858 GPU-h**, +25% an toàn = **9.822 GPU-h**.
- **Đối chiếu quota Kaggle hiện tại (`16.06h`)**: cả phương án chuẩn lẫn độ nhạy +25% đều nằm trong quota tuần còn lại (`<= 16.06 h`).
- **Nguồn số đo từng mô hình**:
  - X0: `W/reports/xlsr_t17_x0_seed2026_session1.json` (10 lượt single GPU, TB `1826.38` s); `W/reports/xlsr_t28_x0_pair_seed2027_2028.json` (seed 2027 10 lượt TB `1941.20` s, seed 2028 10 lượt TB `2080.17` s). Tham chiếu T14 batch=16: `0.648860` h.
  - X1: `W/reports/xlsr_t20_seed2026_l8_lr3e-5.json` (4 lượt TB `1774.56` s); `W/reports/xlsr_t21_seeds_2027_2028.json` + `W/reports/xlsr_t24b_resume_run.json` (10 lượt TB `1947.90` s); `W/reports/xlsr_t26_x1_seed2028.json` (7 lượt TB `1910.82` s). Tham chiếu T14 L8 batch=16: `0.657884` h.
  - X1R: `W/reports/xlsr_t30_x1r_pair_seed2026_2027.json` (seed 2026 9 lượt pair TB `1948.09` s, seed 2027 8 lượt pair TB `2149.16` s); `W/reports/xlsr_t32b_x1r_seed2026_resume_run.json` (1 lượt resume `2038.83` s); `W/reports/xlsr_t31_x1r_seed2028.json` (8 lượt single TB `2074.04` s).

### 3b. Dự kiến B0 v2 trên CPU cục bộ

- **Nguồn chính (đúng pipeline 3 seed gộp batch=8, 8 luồng CPU)**: `reports/shortcut_probe_dev.json` (`sessions[0].stages` `ii`–`v` trên `dev-full` 38.777 mẫu):
  - `original` × 3 seed: dev-full `5171.45` s → quy đổi test-full `4094.00` s = **1.137 giờ CPU** (+25% an toàn: **1.422 giờ**).
  - Đủ 4 điều kiện (`original`, `roundtrip_40k` bonafide xấp xỉ, `lowpass_7k`, `lowpass_4k`) × 3 seed: dev-full `21013.60` s → quy đổi test-full `16635.52` s = **4.621 giờ CPU** (+25% an toàn: **5.776 giờ**).
- **Tham chiếu 1 luồng CPU từ training/benchmark**: `reports/b0_v2_ext_seed{2026,2027,2028}.json` (TB `1184.86` s/dev-full → 3 seed `original` test-full `0.782` giờ); `reports/b0_v2_cpu_benchmark_1thread.json` (`1.195` giờ cho 3 seed `original` test-full).

## 4. Việc còn thiếu để đóng băng registry (chỉ đọc, chưa sửa gì)

- Registry chính thức `reports/model_registry_v1.json`: tồn tại = `False` (bản nháp: `9/9` present, `complete=True`, header `.md` ghi `'snapshot chưa đầy đủ'` = `True`).
- Registry chính thức `reports/model_registry_x1r.json`: tồn tại = `False` (bản nháp: `3/3` present, `complete=True`, header `.md` ghi `'snapshot chưa đầy đủ'` = `True`).
- `configs/test_evaluation.json`: `registry_sha256=None`, `test_gate_open=False`, `pre_test_checks={"entrypoint_dev_claude_verified": true, "bootstrap_dev_claude_verified": true, "t22_claude_verified": false, "test_dataset_inventory_checked": false}`.
- `configs/test_evaluation_x1r.json`: `registry_sha256=None`, `test_gate_open=False`.

Danh sách chi tiết cần xử lý trước khi mở cổng:
1. Tạo registry chính thức reports/model_registry_v1.json (hiện chưa tồn tại; chỉ có bản nháp reports/model_registry_v1_draft.json với role='draft'; scripts/build_model_registry.py đang chặn ghi ra đường dẫn chính thức).
2. Tạo registry chính thức X1R reports/model_registry_x1r.json (hiện chưa tồn tại; chỉ có bản nháp reports/model_registry_x1r_draft.json với role='draft').
3. Điền registry_sha256 vào configs/test_evaluation.json (hiện đang null).
4. Điền registry_sha256 vào configs/test_evaluation_x1r.json (hiện đang null).
5. Cập nhật pre_test_checks.t22_claude_verified=true trong configs/test_evaluation.json (T22b đã được Claude kiểm ĐẠT 13/11/2026 nhưng cờ trong config vẫn đang false).
6. Upload dataset closed_test_full_v1 lên Kaggle (hoặc tạo dataset test trên Kaggle), kiểm khớp kích thước/checksum và cập nhật pre_test_checks.test_dataset_inventory_checked=true.
7. Tiêu đề reports/model_registry_v1_draft.md vẫn ghi '# Registry mô hình v1 — snapshot chưa đầy đủ' và 'Không phải registry chính thức; không mở cổng test.' (dù JSON nháp đã complete=True, 9/9 mục present và claude_verified=True).
8. Tiêu đề reports/model_registry_x1r_draft.md vẫn ghi '# Registry phụ mô hình X1R — snapshot chưa đầy đủ' (dù JSON nháp đã complete=True, 3/3 mục present và claude_verified=True).
9. Theo D036 (scale2), cần hoàn tất và đăng ký registry phụ 6 mục trước khi mở cổng test nếu giữ phạm vi đánh giá gồm scale2.
10. Cổng test hiện vẫn đóng (test_gate_open=false trong cả hai config); chỉ người dùng xác nhận mới được mở.
