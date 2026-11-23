# Báo cáo X1S-04A — Chuẩn bị cục bộ resume scale2 (S2 cặp 2026+2028 và gói resume 2027/S3)

## 1. Phạm vi & Preflight
- Chỉ thao tác cục bộ (**0 lệnh ghi Kaggle**, **0 ghi Drive**, **0 sửa code/config/project.json/test**).
- `bundle_sha256`: `9ac6f8144c4959809007e42a1df7db15923fd9cea1f6433679e8a81090accac1`; `source_tree_sha256`: `32713f9d89a4b0d47e7b2dfdf77c18d4a4d553ea50c263ef2a564e4d74845400` (khớp checkpoint nguồn).
- Băm lại `4` file checkpoint + `2` `LATEST.json` trong `xlsr-d036-scale2-pair-s1-v1`: khớp 100% với report X1S-03.
- Cả hai dataset dự kiến (`ngctnnguyn1111/doanttnt-xlsr-scale2-s2026-resume1`, `ngctnnguyn1111/doanttnt-xlsr-scale2-s2027-resume1`) chưa tồn tại trên Kaggle.

## 2. Hai payload resume cục bộ
- **Seed 2026** (`ngctnnguyn1111/doanttnt-xlsr-scale2-s2026-resume1`, title `DoAnTTNT XLSR Scale2 S2026 Resume1`):
  - Thư mục: `C:\Users\Admin\Documents\DoAnTTNT-xlsr\.cache\x1s04a\resume_s2026`
  - `global_step = 11491`, `files = 5` (`2,600,284,766` B payload; tổng `2,600,287,150` B cả manifest + metadata)
  - `manifest_sha256`: `9012282f20cd4dbe818f845f1cd02b133fee5cf0297fed2b40276675e0d8ccbd`
- **Seed 2027** (`ngctnnguyn1111/doanttnt-xlsr-scale2-s2027-resume1`, title `DoAnTTNT XLSR Scale2 S2027 Resume1`):
  - Thư mục: `C:\Users\Admin\Documents\DoAnTTNT-xlsr\.cache\x1s04a\resume_s2027`
  - `global_step = 13120`, `files = 5` (`2,652,842,796` B payload; tổng `2,652,845,180` B cả manifest + metadata)
  - `manifest_sha256`: `86429f1520d02e7f14931be39e32c8340be34272a4cf7886de9774c771406ec3`

## 3. Hai kernel đã chuẩn bị & xác minh (`verify-prepared-kernel` exit 0)
- **S2 (`scale2-X1-L8-lr3e-5-seed2026-resume1-scale2-X1-L8-lr3e-5-seed2028`)**:
  - `kernel-metadata.json`: `703` B, SHA `9bc0ef80d2aeebecbd14424a837f13591ba3a4e57d71da60926eb112e4e11692` (6 nguồn, đúng `1` dataset resume)
  - `run-config.json`: `37224` B, SHA `b87d459f8a327b9cba97bffd3b24f034e86a3eba0b74f18c869885afca786946` (`source_bundle_sha256 = 9ac6f8144c4959809007e42a1df7db15923fd9cea1f6433679e8a81090accac1`)
  - `runner.py`: `61129` B, SHA `4f024f8429d2a78a7df91ef8a295e4f64222f7220471a9c24adbe1b24b702c28` (`bootstrap_exact_match = True`)
  - Lệnh runner `slot_A_seed2026` (`run_root=/kaggle/working/checkpoints/xlsr_scale2_X1-L8-lr3e-5_seed2026`): `/kaggle/working/.xlsr-offline-env/bin/python scripts/train_xlsr.py --plan configs/xlsr_training_scale2.json --run-id X1-L8-lr3e-5 --training-seed 2026 --device cuda:0 --train-artifact \kaggle\input\doanttnt-scale2-packed-v1 --dev-artifact __KAGGLE_DATA_ROOT__\closed_dev_full_v1 --pretrained-dir \kaggle\input\doanttnt-t14-weights --run-root \kaggle\working\checkpoints\xlsr_scale2_X1-L8-lr3e-5_seed2026 --report-json \kaggle\working\reports\xlsr_scale2_X1-L8-lr3e-5_seed2026.json --session-steps 19860 --deadline-unix 1700039600.0 --resume`
  - Lệnh runner `slot_B_seed2028` (`run_root=/kaggle/working/checkpoints/xlsr_scale2_X1-L8-lr3e-5_seed2028`): `/kaggle/working/.xlsr-offline-env/bin/python scripts/train_xlsr.py --plan configs/xlsr_training_scale2.json --run-id X1-L8-lr3e-5 --training-seed 2028 --device cuda:0 --train-artifact \kaggle\input\doanttnt-scale2-packed-v1 --dev-artifact __KAGGLE_DATA_ROOT__\closed_dev_full_v1 --pretrained-dir \kaggle\input\doanttnt-t14-weights --run-root \kaggle\working\checkpoints\xlsr_scale2_X1-L8-lr3e-5_seed2028 --report-json \kaggle\working\reports\xlsr_scale2_X1-L8-lr3e-5_seed2028.json --session-steps 19860 --deadline-unix 1700039600.0`
- **S3 (`scale2-X1-L8-lr3e-5-seed2027-resume1`)**:
  - `kernel-metadata.json`: `703` B, SHA `824354982aeee109b62973546510aca54a70e40d088005cc18d2bd17e3227389` (6 nguồn, đúng `1` dataset resume)
  - `run-config.json`: `37098` B, SHA `fedf835c8144349e084ec73a6826a0ca2b8aa3f50312933cd875e2dc1961908c` (`source_bundle_sha256 = 9ac6f8144c4959809007e42a1df7db15923fd9cea1f6433679e8a81090accac1`)
  - `runner.py`: `61001` B, SHA `2af6ff7faac401db163766d938247ef75d8e678910a2c4f451013f1fb5c111ec` (`bootstrap_exact_match = True`)
  - Lệnh runner `single_seed2027` (`run_root=/kaggle/working/checkpoints/xlsr_scale2_X1-L8-lr3e-5_seed2027`): `/kaggle/working/.xlsr-offline-env/bin/python scripts/train_xlsr.py --plan configs/xlsr_training_scale2.json --run-id X1-L8-lr3e-5 --training-seed 2027 --device cuda:0 --train-artifact \kaggle\input\doanttnt-scale2-packed-v1 --dev-artifact __KAGGLE_DATA_ROOT__\closed_dev_full_v1 --pretrained-dir \kaggle\input\doanttnt-t14-weights --run-root \kaggle\working\checkpoints\xlsr_scale2_X1-L8-lr3e-5_seed2027 --report-json \kaggle\working\reports\xlsr_scale2_X1-L8-lr3e-5_seed2027.json --session-steps 19860 --deadline-unix 1700039600.0 --resume`

## 4. Mô phỏng restore cục bộ (Bước 7)
- **Dương seed 2026**: `passed = True`, `step = 11491`, `LATEST.json` khớp nguồn.
- **Dương seed 2027**: `passed = True`, `step = 13120`, `LATEST.json` khớp nguồn.
- **Phủ định (2 dataset cùng lúc)**: ném `ValueError`: `"Không xác định được duy nhất manifest checkpoint đầu vào."`.

## 5. Ước tính quota cho S2 (Bước 8)
- Slot A (`seed 2026` resume): `<= 8.369` bước + `<= 5` lần dev (`2,0087 s/bước`, `2.357,6 s/dev`) ≈ tối đa **`7.94h`**.
- Slot B (`seed 2028` chạy mới): chạy tới hạn **`11.0h`** của phiên rồi pause (cần S4 để xong).
- **Kỳ vọng S2**: kéo dài khoảng **`11.0h`** do slot B; cần đợi quota làm mới lúc `2026-11-21 00:00 UTC` (hiện còn `5.03h`).

## 6. Lệnh Kaggle dự kiến cho X1S-04B (CHƯA CHẠY)
1. Tạo dataset private seed 2026 (không `--public`): `kaggle -W datasets create -p C:\Users\Admin\Documents\DoAnTTNT-xlsr\.cache\x1s04a\resume_s2026 --dir-mode skip`
2. Kiểm từ xa: `kaggle -W datasets status ngctnnguyn1111/doanttnt-xlsr-scale2-s2026-resume1` & `datasets files`
3. Push S2: `C:\Users\Admin\Documents\DoAnTTNT\.venv\Scripts\python.exe -B scripts/kaggle_workflow.py --config .kaggle/xlsr-d028/project.json push-prepared-kernel --kernel-dir C:\Users\Admin\Documents\DoAnTTNT-xlsr\.kaggle\xlsr-d028\prepared\scale2-X1-L8-lr3e-5-seed2026-resume1-scale2-X1-L8-lr3e-5-seed2028`
4. Tải output có lọc: `kaggle -W kernels output ngctnnguyn1111/doanttnt-xlsr-d028 -p C:\Users\Admin\Documents\DoAnTTNT\kaggle_outputs\xlsr-d036-scale2-pair-s2-v1 --file-pattern "^reports/|^checkpoints/|^[^/]+\.log$" --page-size 200 --force`

## 7. Sai lệch so với prompt
- Trên Windows, `Path('/kaggle/input/...')` đổi `/` thành `\` khi ép kiểu chuỗi; đã chuẩn hóa dấu phân cách khi kiểm tra lệnh runner giải mã.
- Thư mục prepared gốc cho `scale2-X1-L8-lr3e-5-seed2028` chưa tồn tại sẵn nên đã chuẩn bị bằng `prepare_xlsr_pair_kernel` trước khi sao sang S3.
