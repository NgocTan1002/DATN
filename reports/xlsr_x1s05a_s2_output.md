# Báo cáo X1S-05A — Tải và kiểm output kernel S2 (X1 scale2, version 18)

- Trạng thái: **ĐẠT**
- Xác nhận người dùng: `2026-11-22`, nguyên văn `"tạo prompt cho Antigrvity thực hiện tải output S2"` (chỉ đọc trạng thái kernel, đọc quota và tải output; không thao tác ghi lên Kaggle/Drive).
- Tài liệu JSON chi tiết: [xlsr_x1s05a_s2_output.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/reports/xlsr_x1s05a_s2_output.json)

## 1. Thông tin kernel S2 và quá trình tải output
- **Kernel**: `ngctnnguyn1111/doanttnt-xlsr-d028`, **version 18**, profile `scale2-X1-L8-lr3e-5-seed2026-resume1-scale2-X1-L8-lr3e-5-seed2028`.
- **Trạng thái từ xa**: `KernelWorkerStatus.COMPLETE` (`kaggle_cli_workflow.json` `status = complete`, mã thoát lệnh workflow = `0`, `pair_gpu_count = 2`).
- **Thời gian chạy kernel**: push lúc `2026-11-21T11:24:05.876081+07:00`; workflow bắt đầu `2026-11-21T04:24:34+00:00`, kết thúc `2026-11-21T10:05:00+00:00`, thời lượng `20418.005s` (~`5.67h`).
- **Checksum dữ liệu scale2 trên Kaggle (`scale2_checksum_end`)**: khớp `3/3` file (`index.csv` `7eb18914d7d2ea3b101273d0ab4a4e882a0cac4c8676ae5c018a281086aa8080`, `metadata.json` `3341f75b45f089c4b6b2f71ca3659233a87c6232d416a247b5886d7e0ecec48d`, `waveforms.f32` `260d93d3762f20b564400d023d599d4151d8e74be618441aed255675f3e3da5e`).
- **Hai runner trên 2×T4**: Slot A (`seed 2026`, `exit_code = 0`, `elapsed = 12436.673s`); Slot B (`seed 2028`, `exit_code = 0`, `elapsed = 20417.384s`).
- **Lệnh tải output có lọc** (lần chạy `1`, không cần phân trang `--page-token`):
  - Lệnh: `kaggle kernels output ngctnnguyn1111/doanttnt-xlsr-d028 -p C:\Users\Admin\Documents\DoAnTTNT\kaggle_outputs\xlsr-d036-scale2-pair-s2-v1 --force --file-pattern ^reports/|^checkpoints/|^[^/]+\.log$ --page-size 200`
  - Thời gian (UTC+7): `2026-11-22T14:55:29.184839+07:00` → `2026-11-22T15:06:02.010778+07:00` (`632.812s`), **mã thoát `0`**.
- **Thư mục output**: [xlsr-d036-scale2-pair-s2-v1](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-s2-v1)
- **Số file và tổng dung lượng**: **`17` file**, tổng **`5,391,779,859` bytes** (~`5.021 GiB` / `5.392 GB`); **`0` file trong `.xlsr-offline-env`** (`has_offline_env = false`); **`0` file/score từ test** (`use_test = false`).

## 2. Kết quả kiểm tra SHA-256 checkpoint & `LATEST.json`
Toàn bộ `checkpoint.pt` và `state.json` của cả `latest` và `best` cho hai seed đều đã được tính SHA-256 bằng `hashlib.sha256` và assert khớp `100%` với `LATEST.json` trong [step6_verify.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/step6_verify.py):

| Seed | Vai trò | Generation | File | Kích thước (bytes) | SHA-256 (khớp `LATEST.json`) |
|---:|---|---|---|---:|---|
| 2026 | pointer | — | `LATEST.json` | 538 | `cf329440250c767f19beb36026953453d8f336870b58fa58335e5383b4bd221b` |
| 2026 | `latest` | `g-bf505c00010e4277a13356bf55a13ea4` | `checkpoint.pt` | 1,261,931,931 | `1938f54ba3fb0ca5ac1d464c692d919f05ad881e23c7b4ec127dae9b596bafe8` |
| 2026 | `latest` | `g-bf505c00010e4277a13356bf55a13ea4` | `state.json` | 93,024,417 | `79b4dc9af1609f4fe10bf984b52fa4fc81819a1fc2ca4e8aaabdc2ed16bceda9` |
| 2026 | `best` | `g-77f65dc59bdd4c87b939552eada88692` | `checkpoint.pt` | 1,250,983,067 | `36e4cb21dee17bb55770409a6c78d07ac9c057eeeaf4f7762d3624d05e8b9905` |
| 2026 | `best` | `g-77f65dc59bdd4c87b939552eada88692` | `state.json` | 72,573,178 | `2cf47d3d22b4024f068887e7ed9920f241e9020180d9fa832721910ece26d22a` |
| 2028 | pointer | — | `LATEST.json` | 538 | `de295afef67e82bb4ce9b565002ba151dd7051f7522ecfb69615c1fd72ac27ad` |
| 2028 | `latest` | `g-cbfeee48bbbc4b58914c0490e9a76706` | `checkpoint.pt` | 1,240,034,331 | `7abe801bf5dae58966539ff4ea227988b89bd55200b3145354c47674c4667a50` |
| 2028 | `latest` | `g-cbfeee48bbbc4b58914c0490e9a76706` | `state.json` | 53,405,226 | `96a97712a0ff8e7c33bb61d0fc2475803c8403df391762ed5fb97aeecd6c41c2` |
| 2028 | `best` | `g-f38317599c1b49aa8a10acbab9723572` | `checkpoint.pt` | 1,229,085,531 | `195a8c7be547b2eff3994d363de9edddfcb485f0c55cbbc2a9e2a58e68d9c235` |
| 2028 | `best` | `g-f38317599c1b49aa8a10acbab9723572` | `state.json` | 31,582,857 | `ae4eec92c4cd872052ff72baa7dbd07aa431318a7abb1281934709bca548151b` |

## 3. Kết quả hai seed (X1-L8-lr3e-5, scale2)
- Dữ liệu thực tế trong cả hai report: **`train_samples = 31,770`** (`89` shard), **`development_samples = 38,777`** (`186` shard), **`use_test = false`**, `identity.plan` khớp `configs/xlsr_training_scale2.json` (`SHA-256 3ea92e5724d04d6f5ec427c113bf1d15e276a6ee99662b7857f737e4eff10b64`).

| Seed | Slot | Status | Stop reason | Global step | Completed epochs | Steps phiên này | Best epoch | Best step | Best generation | Best EER dev | ROC-AUC | Macro-F1 | Decision threshold |
|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2026 | A (resume S1) | `early_stopped` | `early_stopped` | 15,888 | 8 | 4,397 | 6 | 11,916 | `g-77f65dc59bdd4c87b939552eada88692` | 0.304556% | 0.999693 | 0.996719 | 0.396790 |
| 2028 | B (chạy mới) | `early_stopped` | `early_stopped` | 7,944 | 4 | 7,944 | 2 | 3,972 | `g-f38317599c1b49aa8a10acbab9723572` | 0.397953% | 0.999696 | 0.995719 | 0.075373 |

### Lịch sử từng epoch — Seed 2026 (Slot A, resume từ S1)
| Epoch | Global step | EER dev | ROC-AUC | Macro-F1 | Decision threshold | Loss | Ghi chú |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1,986 | 0.548201% | 0.999643 | 0.994108 | 0.026605 | 0.042868 | Khớp S1 (`diff = 0.0`) |
| 2 | 3,972 | 0.402014% | 0.999739 | 0.995691 | 0.239411 | 0.021905 | Khớp S1 (`diff = 0.0`) |
| 3 | 5,958 | 0.402014% | 0.999703 | 0.995691 | 0.432202 | 0.021696 | Khớp S1 (`diff = 0.0`) |
| 4 | 7,944 | 0.320799% | 0.999532 | 0.996552 | 0.074831 | 0.020412 | Khớp S1 (`diff = 0.0`) |
| 5 | 9,930 | 0.431065% | 0.999712 | 0.995330 | 0.060258 | 0.033194 | Khớp S1 (`diff = 0.0`) |
| 6 | 11,916 | 0.304556% | 0.999693 | 0.996719 | 0.396790 | 0.020822 | **Best** |
| 7 | 13,902 | 0.320799% | 0.999813 | 0.996552 | 0.081207 | 0.026026 | Mới ở S2 |
| 8 | 15,888 | 0.431065% | 0.999462 | 0.995330 | 0.460658 | 0.028090 | Mới ở S2 |

- **Kiểm tra nối tiếp S1 → S2 (Seed 2026)**: `resume_input` đạt (`passed = true`, `restored_global_step = 11,491`, `manifest_sha256 = 9012282f20cd4dbe818f845f1cd02b133fee5cf0297fed2b40276675e0d8ccbd` khớp X1S-04B và pointer S1); cả `5/5` epoch đã có ở S1 (epoch 1–5) giữ nguyên chính xác EER, ROC-AUC, macro-F1, `decision_threshold` và `loss` (`abs_diff = 0.0`), và `global_step` tăng từ `11,491` lên `15,888` (`+4,397` bước, hoàn thành thêm epoch 6–8).

### Lịch sử từng epoch — Seed 2028 (Slot B, chạy mới)
| Epoch | Global step | EER dev | ROC-AUC | Macro-F1 | Decision threshold | Loss | Ghi chú |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1,986 | 0.522931% | 0.999723 | 0.994358 | 0.401464 | 0.017139 |  |
| 2 | 3,972 | 0.397953% | 0.999696 | 0.995719 | 0.075373 | 0.022441 | **Best** |
| 3 | 5,958 | 0.560383% | 0.999658 | 0.993969 | 0.001181 | 0.058537 |  |
| 4 | 7,944 | 0.414196% | 0.999818 | 0.995552 | 0.397020 | 0.018983 |  |

## 4. Quota GPU Kaggle
- Trước push S2 (preflight X1S-04B): còn lại `45.00h` / `45.00h`.
- Sau push S2 ~6 phút (X1S-04B): `used = 0.10h`, `remaining = 44.90h`.
- Sau khi S2 hoàn tất (kiểm lúc `2026-11-22T15:22:17.234958+07:00`, [quota_after.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/quota_after.json)): **`used = 5.68h`**, **`remaining = 39.32h`**, `total = 45.00h`, `refreshAt = 2026-11-28T00:00:00`.

## 5. Kết luận chuẩn bị S3 / S4
- **Các seed đã dừng hoàn toàn (`early_stopped`, không cần resume)**:
  - **Seed 2026**: `early_stopped` tại epoch `8` (`global_step = 15,888`, `bad_evaluations = 2`), best epoch `6` (`global_step = 11,916`, generation `g-77f65dc59bdd4c87b939552eada88692`, EER dev `0.304556%`).
  - **Seed 2028**: `early_stopped` tại epoch `4` (`global_step = 7,944`, `bad_evaluations = 2`), best epoch `2` (`global_step = 3,972`, generation `g-f38317599c1b49aa8a10acbab9723572`, EER dev `0.397953%`).
- **Các seed cần resume (`paused`)**:
  - Trong S2 **không có seed nào ở trạng thái `paused`**; do đó **không cần phiên S4 hay dataset resume cho seed 2028**.
  - Trên toàn thí nghiệm X1 scale2 chỉ còn **seed 2027** (đang `paused` từ S1 tại `global_step = 13,120`, `completed_epochs = 6`, best epoch `6` EER dev `0,239584%`) chờ chạy đơn ở phiên **S3**. Dataset resume private `ngctnnguyn1111/doanttnt-xlsr-scale2-s2027-resume1` (`manifest_sha256 = 86429f1520d02e7f14931be39e32c8340be34272a4cf7886de9774c771406ec3`) đã được tạo và kiểm tra sẵn ở X1S-04B.

## 6. Bằng chứng kiểm tra và đường dẫn file
- Trạng thái git trước khi chạy: [git_status_before.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/git_status_before.log)
- Trạng thái kernel S2: [kernel_status.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/kernel_status.log)
- Script & log tải output: [step3_download_output.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/step3_download_output.py), [kernel_output_download.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/kernel_output_download.log), [step3_download.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/step3_download.json)
- Kiểm file & checksum: [step4_5_check_files.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/step4_5_check_files.py), [file_list.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/file_list.json), [step6_verify.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/step6_verify.py), [verify.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/verify.json)
- Quota GPU sau S2: [step9_quota.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/step9_quota.py), [quota_after.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s05a/quota_after.json)
