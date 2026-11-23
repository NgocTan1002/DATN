# Báo cáo X1S-02 — Publish source scale2 và chạy kernel dry-run cặp trên Kaggle

## 1. Thông tin chung và xác nhận
- **Xác nhận người dùng**: ngày `2026-11-19`, nguyên văn `"Cho phép publish và dry-run"` (một lần `publish-source` bundle `9ac6f814...` lên `ngctnnguyn1111/doanttnt-xlsr-d028-source` và một lần `push-prepared-kernel` dry-run cặp `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027-dry-run`).
- **Source dataset**: `ngctnnguyn1111/doanttnt-xlsr-d028-source`, bundle SHA-256 `9ac6f8144c4959809007e42a1df7db15923fd9cea1f6433679e8a81090accac1`; cập nhật lúc `2026-11-18T22:33:33.740000` (UTC), kích thước `288.370` B (`kaggle datasets list` không trả số version nguyên); tải từ xa về [remote_source](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s02/remote_source) xác minh **76/76 file** và `source_manifest.json` khớp cục bộ (`0` sai khác).
- **Kernel**: `ngctnnguyn1111/doanttnt-xlsr-d028`, **version 16**, profile `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027-dry-run`, trạng thái cuối `KernelWorkerStatus.COMPLETE`.
- **Output tải về**: [xlsr-d036-scale2-pair-dry-run-v1](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-dry-run-v1).
- **Chi tiết JSON & bảng lệnh đầy đủ**: [xlsr_x1s02_pair_dry_run.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/reports/xlsr_x1s02_pair_dry_run.json).

## 2. Sửa report X1S-01
- Đã sửa hai tên dataset nguồn trong [xlsr_x1s01_prep.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/reports/xlsr_x1s01_prep.json) và [xlsr_x1s01_prep.md](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/reports/xlsr_x1s01_prep.md): `doanttnt-xlsr-r-300m-weights` → `doanttnt-t14-weights` và `doanttnt-xlsr-d028-offline-deps` → `doanttnt-t14-offline-deps`.
- Đối với lỗi (1): tại Bước 2, cả hai file đã có đủ 64 ký tự `9b4b44cee285b8e325696750cc74244a691702dfdb2c25324f49ee625a52f411`. Diff lưu tại [report_fix.diff](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s02/report_fix.diff).

## 3. Kết quả kiểm tra (a)–(f)
- **(a) `pair_gpu_check` và GPU**: `gpu_count = 2` (`elapsed_seconds = 10.119s`); slot A (`seed 2026`, `CUDA_VISIBLE_DEVICES=0`) và slot B (`seed 2027`, `CUDA_VISIBLE_DEVICES=1`) đều chạy trên `Tesla T4` (`visible_gpu_count = 1` mỗi tiến trình).
- **(b) `scale2_checksum_end`**: thư mục `/kaggle/input/doanttnt-scale2-packed-v1`, **`hash_seconds = 73.155s`**, khớp 3/3 file:
  - `index.csv`: `6.280.248` B, SHA-256 `7eb18914d7d2ea3b101273d0ab4a4e882a0cac4c8676ae5c018a281086aa8080`
  - `metadata.json`: `964` B, SHA-256 `3341f75b45f089c4b6b2f71ca3659233a87c6232d416a247b5886d7e0ecec48d`
  - `waveforms.f32`: `12.123.780.480` B, SHA-256 `260d93d3762f20b564400d023d599d4151d8e74be618441aed255675f3e3da5e`
- **(c) Hai runner dry-run song song**:
  - `seed 2026` ([xlsr_scale2_X1-L8-lr3e-5_seed2026.json](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-dry-run-v1/reports/xlsr_scale2_X1-L8-lr3e-5_seed2026.json)): `runner_end` `exit_code = 0`, `status = dry_run`, `train_samples = 31770`, `development_samples = 38777`, `train_shards = 89`, `development_shards = 186`, `use_test = false`, `plan_sha256 = 3ea92e5724d04d6f5ec427c113bf1d15e276a6ee99662b7857f737e4eff10b64`, `checksums_match = true`, `wall_seconds = 40.101s`.
  - `seed 2027` ([xlsr_scale2_X1-L8-lr3e-5_seed2027.json](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-dry-run-v1/reports/xlsr_scale2_X1-L8-lr3e-5_seed2027.json)): `runner_end` `exit_code = 0`, `status = dry_run`, `train_samples = 31770`, `development_samples = 38777`, `train_shards = 89`, `development_shards = 186`, `use_test = false`, `plan_sha256 = 3ea92e5724d04d6f5ec427c113bf1d15e276a6ee99662b7857f737e4eff10b64`, `checksums_match = true`, `wall_seconds = 40.100s`.
- **(d) Thời gian từng giai đoạn bootstrap** (từ [doanttnt-xlsr-d028.log](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-dry-run-v1/doanttnt-xlsr-d028.log) và [kaggle_cli_workflow.json](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-dry-run-v1/reports/kaggle_cli_workflow.json)):
  - `bootstrap_start`: `0.000s` (`2026-11-18T22:36:12.338053+00:00`)
  - `pair_gpu_check`: `10.119s` (`+10.119s`)
  - `offline_checksum_end`: `14.790s` (`+4.671s`)
  - `scale2_checksum_end`: `87.946s` (`+73.155s`)
  - `venv_pip_end`: `102.480s` (`+14.534s`)
  - `runner_start` (A/B): `102.480s` / `102.481s`
  - `runner_end` (A/B): `146.221s` / `146.226s` (`+43.740s` / `+43.745s`), tổng workflow `146.328s`.
- **(e) Không có checkpoint huấn luyện hay score test**: `0` checkpoint/weight trong output; ngoài `.xlsr-offline-env/` chỉ có log bootstrap, 2 log runner, 2 JSON dry-run và `kaggle_cli_workflow.json`.
- **(f) Quota**: trước `remaining = 16.06h` (`used = 28.94h`), sau `remaining = 16.02h` (`used = 28.98h`), **đã trừ `0.04h`** (`<= 0.3h`).

## 4. Lệnh đã chạy chính (UTC+7 và mã thoát)
1. Preflight (`quota_before`, `kernels status`, `git status`, `verify-prepared-kernel`): `2026-11-19T05:31:26–05:31:30+07:00`, tất cả thoát `0` ([preflight.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s02/preflight.json)).
2. `publish-source --message "X1S-02: bundle 9ac6f814 (D036 scale2)" --confirm-private-upload`: `05:33:09–05:33:17+07:00`, thoát `0` ([publish.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s02/publish.log)).
3. `datasets status` & `datasets download` (`--unzip`): `05:33:17–05:33:24+07:00`, thoát `0` ([source_download.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s02/source_download.log)).
4. `push-prepared-kernel`: `05:35:31–05:35:35+07:00`, thoát `0`, `Kernel version 16` ([push.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s02/push.log)).
5. Theo dõi 6 lần `kernels status` (`05:36:10–05:39:13+07:00`, mỗi 35s) → `COMPLETE` ([kernel_status_polls.jsonl](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s02/kernel_status_polls.jsonl)).
6. `kernels output`, `quota_after`, `git status`: thoát `0` ([step5_check.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s02/step5_check.json)).

## 5. Sai lệch và bước tiếp theo
- **Sai lệch**: (1) `kaggle datasets list/metadata` không trả số version nguyên cho source dataset (đã ghi `lastUpdated` UTC và kiểm SHA-256 76/76 file); (2) lỗi (1) của report X1S-01 đã có đủ 64 ký tự từ trước nên `report_fix.diff` chỉ chứa phần sửa lỗi (2).
- **Bước tiếp theo**: chờ Claude kiểm X1S-02 và xin phép người dùng cho lần push huấn luyện thật cặp `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027`.
