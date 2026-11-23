# Báo cáo X1S-06B — Tải và kiểm output kernel S3 (X1 scale2, resume seed 2027, version 19)

- Trạng thái: **ĐẠT**
- Xác nhận người dùng: `2026-11-23`, nguyên văn `"hiện tại Kaggle đã COMPLETE"` (tiếp nối chuỗi X1S đã duyệt; chỉ đọc trạng thái, đọc quota và tải output; không ghi lên Kaggle/Drive).
- Tài liệu JSON chi tiết: [xlsr_x1s06b_s3_output.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/reports/xlsr_x1s06b_s3_output.json)

## 1. Thông tin kernel S3 và quá trình tải output
- **Kernel**: `ngctnnguyn1111/doanttnt-xlsr-d028`, **version `19`**, profile `scale2-X1-L8-lr3e-5-seed2027-resume1` (chạy đơn 1×T4).
- **Trạng thái từ xa**: `KernelWorkerStatus.COMPLETE` (`kaggle_cli_workflow.json` `status = complete`, mọi lệnh workflow mã thoát `0`).
- **Thời gian chạy kernel**: push lúc `2026-11-22T20:26:16.432165+07:00`; workflow bắt đầu `2026-11-22T19:23:04+00:00`, kết thúc `2026-11-22T22:19:49+00:00`, thời lượng `10596.867s` (~`2.94h`).
- **Checksum dữ liệu scale2 trên Kaggle (`scale2_checksum_end`)**: khớp `3/3` file (`index.csv` `7eb18914d7d2ea3b101273d0ab4a4e882a0cac4c8676ae5c018a281086aa8080`, `metadata.json` `3341f75b45f089c4b6b2f71ca3659233a87c6232d416a247b5886d7e0ecec48d`, `waveforms.f32` `260d93d3762f20b564400d023d599d4151d8e74be618441aed255675f3e3da5e`).
- **Runner đơn (seed 2027)**: `exit_code = 0`, `elapsed = 10596.409s`.
- **Lệnh tải output có lọc** (lần chạy `1`, không cần phân trang `--page-token`):
  - Lệnh: `kaggle kernels output ngctnnguyn1111/doanttnt-xlsr-d028 -p C:\Users\Admin\Documents\DoAnTTNT\kaggle_outputs\xlsr-d036-scale2-s3-seed2027-v1 --force --file-pattern ^reports/|^checkpoints/|^[^/]+\.log$ --page-size 200`
  - Thời gian (UTC+7): `2026-11-23T09:05:12.402386+07:00` → `2026-11-23T09:10:52.955515+07:00` (`340.552s`), **mã thoát `0`**.
- **Thư mục output**: [xlsr-d036-scale2-s3-seed2027-v1](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-s3-seed2027-v1)
- **Số file và tổng dung lượng**: **`9` file**, tổng **`2,787,891,205` bytes** (~`2.596 GiB` / `2.788 GB`); **`0` file trong `.xlsr-offline-env`** (`has_offline_env = false`); **`0` file/score từ test** (`use_test = false`).

## 2. Kết quả kiểm tra SHA-256 checkpoint & `LATEST.json` (Seed 2027)
Toàn bộ `checkpoint.pt` và `state.json` của cả `latest` và `best` đều đã được tính SHA-256 bằng `hashlib.sha256` và assert khớp `100%` với `LATEST.json` trong [step6_verify.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/step6_verify.py):

| Seed | Vai trò | Generation | File | Kích thước (bytes) | SHA-256 (khớp `LATEST.json`) |
|---:|---|---|---|---:|---|
| 2027 | pointer | — | `LATEST.json` | 538 | `7d01f7a9c5646be073997755d25ce6deaa465846e6d9e5aed3b286bc4cb6f403` |
| 2027 | `latest` | `g-3271a17b6e6143d085cb67a728faa690` | `checkpoint.pt` | 1,261,931,931 | `92f35d4069b92342e2d72e8ff0b09a12f5d216e792699393ed6c2d24c91a0a1f` |
| 2027 | `latest` | `g-3271a17b6e6143d085cb67a728faa690` | `state.json` | 95,924,361 | `38ca30de1288e876602e29bf1ff83485122788c22137248364d9052b4bab74ff` |
| 2027 | `best` | `g-2f64908aa31b4ea09320fa58bfec0277` | `checkpoint.pt` | 1,250,983,067 | `a9ed12b3cdb3805f46836cb2a371e07014fdab835704eacf6956edecb31c4e6f` |
| 2027 | `best` | `g-2f64908aa31b4ea09320fa58bfec0277` | `state.json` | 75,253,548 | `61bf95032c0865d68e097c059782e2cc2784362fa576fcfa51dfa49274b47071` |

## 3. Kết quả chi tiết Seed 2027 (S3) & Kiểm tra bước 7–9
- **Dữ liệu & cấu hình**: `train_samples = 31,770` (`89` shard), `development_samples = 38,777` (`186` shard), `use_test = false`, `identity.plan` khớp `configs/xlsr_training_scale2.json` (`SHA-256 3ea92e5724d04d6f5ec427c113bf1d15e276a6ee99662b7857f737e4eff10b64`).
- **Trạng thái kết thúc**: `status = early_stopped`, `stop_reason = early_stopped`, `global_step = 15,888`, `completed_epochs = 8`, `steps_this_invocation = 2,768`, `bad_evaluations = 2`.
- **Best checkpoint**: `generation = g-2f64908aa31b4ea09320fa58bfec0277` (khớp `LATEST.json`), `epoch = 6`, `global_step = 11,916`, **EER dev = `0.239584%`** (`0.002395841793226671`), `ROC-AUC = 0.999923`, `macro-F1 = 0.997442`, `decision_threshold = 0.057826`.
- **Kiểm tra `resume_input` (Bước 7)**: `passed = true`, `restored_global_step = 13,120`, `manifest_sha256 = 86429f1520d02e7f14931be39e32c8340be34272a4cf7886de9774c771406ec3` (khớp seed 2027 trong X1S-04B và pointer S1 `LATEST.json`).
- **Kiểm tra nối tiếp S1 → S3 (Bước 8)**: Cả `6/6` epoch đầu (epoch 1–6) khớp tuyệt đối với S1 ở EER, ROC-AUC, macro-F1, `decision_threshold` và `loss` (`abs_diff = 0.0`); `global_step` tăng từ `13,120` lên `15,888` (`+2,768` bước).
- **Kiểm lại luật dừng bằng code (Bước 9)**: Áp `patience = 2`, `max_epochs = 10`, cải thiện nghiêm ngặt theo EER lên `history`: cho `best_epoch = 6` (`global_step = 11,916`), `bad_evaluations = 2`, dừng tại epoch `8` với `stop_reason = early_stopped` — **khớp 100% report**.

### Lịch sử từng epoch — Seed 2027 (S1 + S3)
| Epoch | Global step | EER dev | ROC-AUC | Macro-F1 | Decision threshold | Loss | Nguồn / Ghi chú |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1,986 | 0.507594% | 0.999735 | 0.994552 | 0.304515 | 0.021097 | S1 (khớp `diff = 0.0`) |
| 2 | 3,972 | 0.483229% | 0.999739 | 0.994802 | 0.071429 | 0.029199 | S1 (khớp `diff = 0.0`) |
| 3 | 5,958 | 0.452265% | 0.999742 | 0.995107 | 0.000364 | 0.050856 | S1 (khớp `diff = 0.0`) |
| 4 | 7,944 | 0.588809% | 0.999785 | 0.993663 | 0.025856 | 0.050622 | S1 (khớp `diff = 0.0`) |
| 5 | 9,930 | 0.381710% | 0.999818 | 0.995885 | 0.586709 | 0.019252 | S1 (khớp `diff = 0.0`) |
| 6 | 11,916 | 0.239584% | 0.999923 | 0.997442 | 0.057826 | 0.019498 | S1 (khớp `diff = 0.0`); **Best** |
| 7 | 13,902 | 0.367465% | 0.999852 | 0.996024 | 0.001664 | 0.041439 | Mới ở S3 |
| 8 | 15,888 | 0.268009% | 0.999777 | 0.997136 | 0.871721 | 0.022814 | Mới ở S3 |

## 4. Quota GPU Kaggle
- Trước push S3 ([xlsr_x1s06_s3_push.md](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/reports/xlsr_x1s06_s3_push.md)): `used = 5.68h`, `remaining = 39.32h` / `45.00h`.
- Sau khi S3 hoàn tất (kiểm lúc `2026-11-23T09:18:59.744348+07:00`, [quota_after.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/quota_after.json)): **`used = 8.62h`**, **`remaining = 36.38h`**, `total = 45.00h`, `refreshAt = 2026-11-28T00:00:00` (S3 trừ `2.94h`).

## 5. Bảng tổng hợp cả ba seed X1 scale2 (X1-L8-lr3e-5, 31.770 mẫu train / 89 shard)
| Seed | Phiên kết thúc | Status | Completed epochs | Global step | Best epoch (step) | Best generation | EER dev best | ROC-AUC | Macro-F1 | Decision threshold | Thư mục output chứa best |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| 2026 | S2 (v18, resume S1) | `early_stopped` | 8 | 15,888 | 6 (`11,916`) | `g-77f65dc59bdd4c87b939552eada88692` | 0.304556% | 0.999693 | 0.996719 | 0.396790 | `C:\Users\Admin\Documents\DoAnTTNT\kaggle_outputs\xlsr-d036-scale2-pair-s2-v1` |
| 2027 | S3 (v19, resume S1) | `early_stopped` | 8 | 15,888 | 6 (`11,916`) | `g-2f64908aa31b4ea09320fa58bfec0277` | 0.239584% | 0.999923 | 0.997442 | 0.057826 | `C:\Users\Admin\Documents\DoAnTTNT\kaggle_outputs\xlsr-d036-scale2-s3-seed2027-v1` |
| 2028 | S2 (v18, chạy mới) | `early_stopped` | 4 | 7,944 | 2 (`3,972`) | `g-f38317599c1b49aa8a10acbab9723572` | 0.397953% | 0.999696 | 0.995719 | 0.075373 | `C:\Users\Admin\Documents\DoAnTTNT\kaggle_outputs\xlsr-d036-scale2-pair-s2-v1` |

- **Trung bình ± SD mẫu (3 seed 2026/2027/2028) EER dev-full**: **`0.314031%` ± `0.079609` điểm %**.
- **Trạng thái tổng thể X1 scale2**: Đã hoàn tất cả 3/3 seed (`early_stopped` / `max_epochs`, không còn seed nào cần resume); sẵn sàng cho bước lập registry scale2 khi Claude soạn nhiệm vụ tiếp theo.

## 6. Bằng chứng kiểm tra và đường dẫn file
- [git_status_before.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/git_status_before.log)
- [kernel_status.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/kernel_status.log)
- [step3_download_output.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/step3_download_output.py), [kernel_output_download.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/kernel_output_download.log), [step3_download.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/step3_download.json)
- [step4_5_check_files.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/step4_5_check_files.py), [file_list.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/file_list.json)
- [step6_verify.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/step6_verify.py), [verify.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/verify.json)
- [step10_quota.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/step10_quota.py), [quota_after.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s06b/quota_after.json)
