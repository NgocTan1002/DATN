# Báo cáo X1S-03 — Phiên 1 cặp X1-L8-lr3e-5 seed 2026+2027 trên scale2

## 1. Thông tin chung và xác nhận
- **Xác nhận người dùng**: ngày `2026-11-19`, nguyên văn `"Cho phép run thật"` (áp dụng cho đúng một lần `push-prepared-kernel` profile `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027`, phiên 1).
- **Kernel**: `ngctnnguyn1111/doanttnt-xlsr-d028`, **version 17**, đẩy lúc `2026-11-19T09:26:40.560681–09:26:44.718742+07:00` ([push.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/push.log)), kết thúc `KernelWorkerStatus.COMPLETE` (`workflow_elapsed_seconds = 39534.962s`, `2026-11-19T02:27:06+00:00` → `13:26:09+00:00`).
- **Output tải về**: [xlsr-d036-scale2-pair-s1-v1](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-s1-v1) (`exit_code = 0`).
- **Sửa lỗi report X1S-01**: đã kiểm `configs/xlsr_training.json` bằng `python -I` (`9b4b44cee285b8e32d596750cc74244a691702dfdb2c25324f49ee625a52f411`, `h[16:20] = 2d59`, [sha_check.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/sha_check.log)) và sửa bằng script [fix_x1s01_sha.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/fix_x1s01_sha.py), diff lưu tại [report_fix.diff](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/report_fix.diff).
- **Chi tiết JSON & danh sách lệnh đầy đủ**: [xlsr_x1s03_pair_session1.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/reports/xlsr_x1s03_pair_session1.json).

## 2. Kết quả kiểm tra (a)–(f)
- **(a) Phần cứng và checksum scale2**: `pair_gpu_check gpu_count = 2` (`11.971s`); `scale2_checksum_end` (`hash_seconds = 61.504s`) khớp `3/3` file (`index.csv` `7eb18914...`, `metadata.json` `3341f75b...`, `waveforms.f32` `260d93d3...`).
- **(b) Trạng thái hai runner (`exit_code = 0`, `use_test = false`, `identity.plan` khớp `configs/xlsr_training_scale2.json` SHA-256 `3ea92e5724d04d6f5ec427c113bf1d15e276a6ee99662b7857f737e4eff10b64`)**:
  - **Seed 2026** (Slot A, GPU `0`, [xlsr_scale2_X1-L8-lr3e-5_seed2026.json](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-s1-v1/reports/xlsr_scale2_X1-L8-lr3e-5_seed2026.json)):
    - `runner_end exit_code = 0`, `status = paused`, `pause_cause = budget_before_training`.
    - `global_step = 11491`, `completed_epochs = 5` (đang ở bước `1.561/1.986` của epoch 6), `best_epoch = 4` (`global_step = 7944`, EER dev `0,320799%`).
    - EER dev-full theo epoch: ep 1 `0,548201%`, ep 2 `0,402014%`, ep 3 `0,402014%`, ep 4 `0,320799%` (**best**), ep 5 `0,431065%`.
  - **Seed 2027** (Slot B, GPU `1`, [xlsr_scale2_X1-L8-lr3e-5_seed2027.json](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d036-scale2-pair-s1-v1/reports/xlsr_scale2_X1-L8-lr3e-5_seed2027.json)):
    - `runner_end exit_code = 0`, `status = paused`, `pause_cause = budget_before_training`.
    - `global_step = 13120`, `completed_epochs = 6` (đang ở bước `1.204/1.986` của epoch 7), `best_epoch = 6` (`global_step = 11916`, EER dev `0,239584%`).
    - EER dev-full theo epoch: ep 1 `0,507594%`, ep 2 `0,483229%`, ep 3 `0,452265%`, ep 4 `0,588809%`, ep 5 `0,381710%`, ep 6 `0,239584%` (**best**).
- **(c) Checkpoint resume v4 hợp lệ (khớp 100% SHA-256 trong `LATEST.json`)**:
  - **Seed 2026**: `LATEST.json` (`538` B, SHA `dba7078c...`) trỏ `latest = g-f0deb62f783f4f5095a2f44a0702dfe2` (`checkpoint.pt`: `1.245.609.947` B SHA `d5737f3f...`; `state.json`: `62.615.055` B SHA `447e77d8...`, `global_step = 11491`) và `best = g-6d3cd386b2514c7292a0e976f48ad28e` (`checkpoint.pt`: `1.240.034.267` B SHA `a86925f2...`; `state.json`: `52.024.959` B SHA `50fd8391...`, `global_step = 7944`).
  - **Seed 2027**: `LATEST.json` (`538` B, SHA `e0ce78b7...`) trỏ `latest = g-b19748034808412f8a006986895313e6` (`checkpoint.pt`: `1.251.061.211` B SHA `1236c6ae...`; `state.json`: `75.544.432` B SHA `3374d907...`, `global_step = 13120`) và `best = g-2f64908aa31b4ea09320fa58bfec0277` (`checkpoint.pt`: `1.250.983.067` B SHA `a9ed12b3...`; `state.json`: `75.253.548` B SHA `61bf9503...`, `global_step = 11916`).
- **(d) Thời gian trung bình & ước tính số phiên còn lại**:
  - **Seed 2026**: `mean_step_seconds = 2,008739s`, trung bình chấm dev `2.357,621s` (`1.964,700s` → `2.646,700s`). Tối đa tới epoch 10 còn `8.369` bước + `5` lần dev ≈ `7,94h` (dự kiến thực tế sớm hơn nếu early-stop với `patience = 2`) → còn **1 phiên**.
  - **Seed 2027**: `mean_step_seconds = 1,810937s`, trung bình chấm dev `1.722,687s` (`1.700,583s` → `1.733,241s`). Tối đa tới epoch 10 còn `6.740` bước + `4` lần dev ≈ `5,30h` → còn **1 phiên**.
  - Có thể gộp cả hai seed vào **1 phiên resume cặp** tiếp theo (`~8h` thời gian tường).
- **(e) Không có file hay score từ test**: kiểm toàn bộ `16` file ngoài `.xlsr-offline-env/` xác nhận `0` file/score test (`LATEST.json` chỉ chứa chuỗi con `test` của từ `latest`).
- **(f) Quota Kaggle**: trước `16.02h` (`used = 28.98h`) → sau `5.03h` (`used = 39.97h`), **đã trừ `10.99h`** (`<= 11.5h`; làm mới `2026-11-21 00:00 UTC`).

## 3. Lệnh đã chạy chính (UTC+7 và mã thoát)
1. Kiểm SHA & sửa report X1S-01 (`python -I`, `fix_x1s01_sha.py`): `09:22–09:23+07:00`, thoát `0` ([sha_check.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/sha_check.log), [report_fix.diff](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/report_fix.diff)).
2. Chuẩn bị kernel thật & Preflight (`step1_prepare_and_preflight.py`): thoát `0` ([preflight.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/preflight.json)).
3. `push-prepared-kernel` đúng 1 lần (`step3_push_once.py`): `09:26:40–09:26:44+07:00`, thoát `0`, `Kernel version 17` ([push.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/push.log)).
4. Theo dõi trạng thái (`step4_watch.py`, [status_polls.jsonl](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/status_polls.jsonl); kiểm `20:25:18` ở `task-1435` và poll #4 `COMPLETE` lúc `20:41:53+07:00`).
5. `kaggle -W quota --format json` & `kernels status` (`task-1443`): thoát `0`.
6. `kaggle -W kernels output` về `xlsr-d036-scale2-pair-s1-v1` (`21:10:49–22:46:07+07:00`): thoát `0` ([kernel_output_download.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s03/kernel_output_download.log)).

## 4. Sai lệch và bước tiếp theo
- **Sai lệch**:
  1. Ở lượt đầu X1S-03 đã dừng trước Bước 2 do nhìn nhầm ký tự 17–20 (`2d59` nhìn nhầm thành `2569`). Sau chỉ dẫn `X1S-03 (tiếp)`, đã kiểm bằng `python -I` và sửa tự động bằng script `fix_x1s01_sha.py` có assert đạt.
  2. Trong `preflight`, `git_status_after.log` của X1S-02 chụp trước khi tạo `xlsr_x1s02_pair_dry_run.json/.md`; đã loại trừ 2 file report X1S-02 và xác nhận 0 file code/config/test mới.
  3. 3 poll buổi sáng trong `status_polls.jsonl` bị lỗi mạng `WinError 10065` và server Antigravity khởi động lại lúc ~20:24; đã kiểm `RUNNING` + quota lúc `20:25:18` và bật lại watcher nhận `COMPLETE` lúc `20:41:53+07:00`.
- **Bước tiếp theo**: chờ Claude kiểm X1S-03; sau khi quota làm mới lúc `2026-11-21 00:00 UTC` và được người dùng cho phép riêng, chuẩn bị dataset checkpoint resume và chạy **phiên 2 cặp** cho `seed 2026 + seed 2027`.
