# Báo cáo chuẩn bị X1S-01 — X1 cuối trên scale2 (chưa ghi Kaggle)

## 1. Cấu hình, manifest và artifact D036
- File cấu hình riêng: [xlsr_training_scale2.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/configs/xlsr_training_scale2.json) (`5.768` byte, SHA-256 `3ea92e5724d04d6f5ec427c113bf1d15e276a6ee99662b7857f737e4eff10b64`).
- `experiment_id`: `xlsr-d036-scale2`, `decision`: `D036`, `base_decision`: `D028`.
- `grid`: `[{"run_id": "X1-L8-lr3e-5", "layers": 8, "encoder_lr": 3e-05}]`; chỉ cho phép các seed `2026`, `2027`, `2028`.
- `training`: `steps_per_epoch = 1986`, `max_optimizer_steps = 19860`, `max_train_sample_exposures = 317700`.
- `outputs`: `root_template = checkpoints/xlsr_scale2_{run_id}_seed{training_seed}`, `report_json_template = reports/xlsr_scale2_{run_id}_seed{training_seed}.json`.
- Manifest train ×2 trong `W`: [development_scale2_v1.csv](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/data/manifests/development_scale2_v1.csv) (`4.875.923` byte, SHA-256 `6533fced4b3a25ebab9f037a62d34774b44c2fa1f00d17040fd815bd17fa51a3`, `31.770` mẫu, `89` shard).
- Artifact train ×2 (SHA-256 đã khóa trong config và xác minh trên `M`):
  - `index.csv`: `7eb18914d7d2ea3b101273d0ab4a4e882a0cac4c8676ae5c018a281086aa8080`
  - `metadata.json`: `3341f75b45f089c4b6b2f71ca3659233a87c6232d416a247b5886d7e0ecec48d`
  - `waveforms.f32`: `260d93d3762f20b564400d023d599d4151d8e74be618441aed255675f3e3da5e`
- `sample_selection_sha256` (thứ tự toàn bộ `31.770` ID train và `38.777` ID dev-full): `b76b56a3fdfa53ee0a5fa28f98c31c800b9351073b39a2133da32d0086895ac2`.

## 2. Xác minh dry-run cục bộ
- **X1 scale2 seed 2026**: thoát `0`, `wall_seconds = 8.748s`.
  - Log: [dry_run_2026.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s01/dry_run_2026.log)
  - Report: [dry_run_2026_report.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s01/dry_run_2026_report.json)
  - Kết quả: `train_samples = 31770`, `development_samples = 38777`, `train_shards = 89`, `development_shards = 186`, `steps_per_epoch = 1986`, `runs = ["X1-L8-lr3e-5"]`, `sample_selection_sha256 = b76b56a3fdfa53ee0a5fa28f98c31c800b9351073b39a2133da32d0086895ac2`, `plan_sha256 = 3ea92e5724d04d6f5ec427c113bf1d15e276a6ee99662b7857f737e4eff10b64`.
- **Từ chối X0 trên plan scale2**: thoát `1` với thông báo `LỖI: Cấu hình scale2 (D036) chỉ cho phép run X1-L8-lr3e-5.` ([dry_run_x0_rejected.log](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s01/dry_run_x0_rejected.log)).

## 3. Ước lượng thời gian từ T31 và kế hoạch quota Kaggle
- Nguồn tham chiếu T31 ([xlsr_X1-L8-lr3e-5-lp7k_seed2028.json](file:///C:/Users/Admin/Documents/DoAnTTNT/kaggle_outputs/xlsr-d028-X1R-seed2028-v1/reports/xlsr_X1-L8-lr3e-5-lp7k_seed2028.json)): `status = early_stopped`, `stop_epoch = 8`, `global_step = 7944`, `wall_seconds = 32545.149s` (`~9.04h`), `mean_step_seconds = 1.570730s` (epoch 1 ở mức `1.858936s`), eval `38.777` mẫu dev từ `2056.39s` đến `2099.79s`, rehash train `49.455s`, rehash dev `117.152s`.
- **Chiếu cho scale2 (`1.986` bước train/epoch + `38.777` mẫu dev/epoch)**:
  - Train 1 epoch: `3.119,5s` (~`0.87h`) ở mức trung bình `1.571s/step` đến `3.691,8s` (~`1.03h`) ở mức thận trọng `1.859s/step`.
  - Eval dev-full 1 epoch: `~2.075s` đến `2.100s` (~`0.58h`).
  - Tổng 1 epoch: **`1.44h` (~`5.195s`) trung bình** đến **`1.61h` (~`5.792s`) thận trọng**, cộng `~0.08h` khởi tạo & rehash (`12,13 GB` train + dev).
- **Số phiên/run (trần Kaggle 11h)**:
  - Dừng sớm ở `<= 6–7` epoch: hoàn thành trong **1 phiên**.
  - Dừng ở `8–10` epoch: cần **2 phiên** (1 phiên đầu + 1 phiên resume).
- **Quota Kaggle (trước và sau đều giữ nguyên, 0 ghi Kaggle)**:
  - GPU còn `16.06h / 45.00h` (`used = 28.94h`, làm mới lúc `2026-11-21T00:00:00` UTC); TPU `0.00h / 20.00h`.
  - Thứ tự khuyến nghị khi được duyệt:
    1. Upload source bundle và chạy kernel dry-run cặp `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027-dry-run`.
    2. Chạy kernel cặp thật `seed 2026 + seed 2027` (tối đa `~11h` cho phiên 1, nằm gọn trong `16.06h` còn lại).
    3. Resume cặp (nếu chưa xong ở phiên 1) và chạy kernel đơn `scale2-X1-L8-lr3e-5-seed2028` với phần quota còn lại hoặc sau khi làm mới `45.00h` ngày `21/11/2026`.

## 4. Bundle nguồn và kernel cặp dry-run đã chuẩn bị
- Bundle nguồn: [project_source.zip](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.kaggle/xlsr-d028/workflow/source_dataset/project_source.zip) (`76` file, `277.546` byte, `bundle_sha256 = 9ac6f8144c4959809007e42a1df7db15923fd9cea1f6433679e8a81090accac1`). Chi tiết danh sách file và hash lưu trong [step8_bundle_and_kernel.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s01/step8_bundle_and_kernel.json).
- Thư mục kernel chuẩn bị: [workflow/kernel](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.kaggle/xlsr-d028/workflow/kernel) cho profile `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027-dry-run`.
- `verify-prepared-kernel`: thoát `0`, `runner_sha256 = 8f00f7ffe6de2de3e3cdf0d65832324bc15868e76d2f81c3e5eb40d97f370686`, đủ 5 `dataset_sources` (`ngctnnguyn1111/doanttnt-xlsr-d028-source`, `ngctnnguyn1111/doanttnt-t14-weights`, `ngctnnguyn1111/doanttnt-t14-offline-deps`, `ngctnnguyn1111/doanttnt-t14-packed`, `ngctnnguyn1111/doanttnt-scale2-packed-v1`).

## 5. So sánh profile và baseline ×1
- 4 profile scale2 mới thêm vào [project.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.kaggle/xlsr-d028/project.json) (tổng `56` profile):
  - `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027`
  - `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027-dry-run`
  - `scale2-X1-L8-lr3e-5-seed2028`
  - `scale2-X1-L8-lr3e-5-seed2028-dry-run`
- So với [baseline.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/.cache/x1s01/baseline.json): [xlsr_training.json](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/configs/xlsr_training.json) giữ nguyên SHA-256 `9b4b44cee285b8e32d596750cc74244a691702dfdb2c25324f49ee625a52f411`; toàn bộ `52` profile ×1 giữ nguyên từng byte; test suite tăng từ `450` lên `456/456 OK` (thêm `6` test trong [test_xlsr_scale2.py](file:///C:/Users/Admin/Documents/DoAnTTNT-xlsr/tests/test_xlsr_scale2.py), thoát `0` trong `122.276s`).

## 6. Thao tác ghi Kaggle tương lai cần người dùng duyệt
1. `publish-source` để cập nhật `ngctnnguyn1111/doanttnt-xlsr-d028-source` bằng `project_source.zip`.
2. Push kernel dry-run cặp `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027-dry-run`.
3. Push kernel huấn luyện thật cặp `scale2-X1-L8-lr3e-5-seed2026-scale2-X1-L8-lr3e-5-seed2027`.
4. Push resume cặp (nếu dừng ở trần thời gian phiên 1).
5. Push kernel đơn `scale2-X1-L8-lr3e-5-seed2028` (và resume nếu cần).

## 7. Sai lệch so với prompt
1. Bước 6 nêu `--pretrained-dir C:\Users\Admin\Documents\DoAnTTNT\.kaggle\t14\download\weights`, nhưng thư mục đó rỗng trên đĩa; đã thay bằng thư mục pretrained D021 thực tế [wav2vec2-xls-r-300m](file:///C:/Users/Admin/Documents/DoAnTTNT/data/raw/xlsr/wav2vec2-xls-r-300m) (`pytorch_model.bin` SHA-256 `d5e49057...`).
2. Bước 7(e) ghi "đủ 4 dataset nguồn", thực tế cần đủ **5** `dataset_sources` (gồm cả `ngctnnguyn1111/doanttnt-t14-packed` cho `closed_dev_full_v1` và `ngctnnguyn1111/doanttnt-scale2-packed-v1` cho artifact train ×2) đúng theo Bước 5 và phần Bối cảnh.
