# B0S-01 — Chuẩn bị B0 v2 ×2 (scale2, D036) trên CPU

- Trạng thái: **ĐẠT** (chỉ chuẩn bị cục bộ, **chưa chạy** coordinator hoặc huấn luyện dài).
- File cấu hình: `configs/b0_v2_training_scale2.json`
- SHA-256 cấu hình: `a2e4cf4bbbd9b3fb1a73a6b3e279a023f416d20f32b6d75434bc8e9c288d8018`
- Các khóa khác biệt so với `configs/b0_v2_training.json` (15 khóa):
  - `base_decision`
  - `data.sample_selection_sha256`
  - `data.train_artifact`
  - `data.train_artifact_waveforms_sha256`
  - `data.train_manifest`
  - `data.train_manifest_sha256`
  - `data.train_samples`
  - `decision`
  - `experiment_id`
  - `outputs.report_json_template`
  - `outputs.report_md_template`
  - `outputs.root_template`
  - `training.max_optimizer_steps`
  - `training.max_train_sample_exposures`
  - `training.steps_per_epoch`

## Dữ liệu và Checksum
- Train manifest: `data/manifests/development_scale2_v1.csv` (`6533fced4b3a25ebab9f037a62d34774b44c2fa1f00d17040fd815bd17fa51a3`)
- Train artifact: `data/processed/packed/train_development_scale2_v1` (31770 mẫu / 89 shard)
  - `waveforms.f32`: `260d93d3762f20b564400d023d599d4151d8e74be618441aed255675f3e3da5e`
  - `index.csv`: `7eb18914d7d2ea3b101273d0ab4a4e882a0cac4c8676ae5c018a281086aa8080`
  - `metadata.json`: `3341f75b45f089c4b6b2f71ca3659233a87c6232d416a247b5886d7e0ecec48d`
- Dev-full artifact: `data/processed/packed/closed_dev_full_v1` (38777 mẫu / 186 shard)
  - `waveforms.f32`: `579e5e85a020aeab70c8ad912023f5ee4efaef19f8e61af4fd2c923ee692f648`
  - `source_manifest_sha256`: `a79f1bb2595bc46b08ce30b8652234b1858cc26e558e2ccd33632435df1a5360`
- `sample_selection_sha256`: `b76b56a3fdfa53ee0a5fa28f98c31c800b9351073b39a2133da32d0086895ac2`

## Kết quả kiểm tra
- Technical smoke (`.cache/b0s/smoke.log`): status `early_stopped`, `global_step=6`, thời gian `5.401 s` (bao gồm kiểm băm toàn bộ waveforms).
- Toàn suite (`.cache/b0s/suite.log`): `510` tests, `OK=True` trong `297.535 s`.
- 4 file cấu hình bị khóa giữ nguyên SHA-256 trước/sau.

## Dự toán CPU (phép chiếu 1 luồng)
- Nguồn benchmark: `reports/b0_v2_cpu_benchmark_1thread.json` (`seconds_per_step_mean=1.074529`, `seconds_per_dev_sample=0.046712`).
- Mỗi epoch (3.972 bước train + 38.777 mẫu dev-full): `6079.39 s` (`1.689 giờ`).
- Trần 10 epoch / seed: `16.887 giờ/seed` (tổng 3 seed 1 luồng: `50.662 giờ`).
- Lưu ý: Phép chiếu 1 luồng từ benchmark D027; chưa đo khi 3 process chạy song song trên CPU.
- Hệ thống lúc chuẩn bị: RAM `1.905 / 15.692 GiB` khả dụng/tổng; ổ C trống `55.373 GiB`.

## Lệnh khởi chạy coordinator đề xuất (chưa chạy)
```powershell
$env:PYTHONUTF8=1; .\.venv\Scripts\python.exe .cache\b0s\coordinator.py
```
