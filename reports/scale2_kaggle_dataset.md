# Báo cáo SCALE2-04 — Upload dataset Kaggle train ×2 (`READY_VERIFIED`)

- **Trạng thái cuối:** `READY_VERIFIED` (**chờ Claude kiểm**; không push kernel).
- **Dataset Kaggle:** `ngctnnguyn1111/doanttnt-scale2-packed-v1` (`datasetId = 13459408`, `title = "DoAnTTNT scale2 packed v1"`, `isPrivate = true`, `license = CC0-1.0`).
- **URL Kaggle:** `https://www.kaggle.com/datasets/ngctnnguyn1111/doanttnt-scale2-packed-v1`
- **Nguồn artifact cục bộ:** `data/processed/packed/train_development_scale2_v1/` (`31.770` mẫu, `89` shard).
- **Xác nhận số lượng tạo:** chỉ có **đúng 1** dataset mới được tạo thành công (`datasets list --mine` tăng từ `10` lên `11`, 10 dataset cũ giữ nguyên).
- **Đường dẫn trong kernel Kaggle dự kiến:**
  - `/kaggle/input/doanttnt-scale2-packed-v1/index.csv`
  - `/kaggle/input/doanttnt-scale2-packed-v1/metadata.json`
  - `/kaggle/input/doanttnt-scale2-packed-v1/waveforms.f32`

## 1. Lịch sử 3 lần chạy `datasets create` và 3 file lock

| Lần | Lock file | Lệnh | Thời gian (`+07:00`) | Mã thoát | Kết quả / Lý do dừng | Dataset tạo mới |
|---|---|---|---|---:|---|---:|
| Lần 1 (`SCALE2-04`) | `.cache/scale2_04/DATASET_CREATE_FAILED_PATH_20261118.lock` (đổi tên từ `DATASET_CREATE_ONCE.lock`) | `kaggle -W datasets create --path .cache/scale2_04/upload --dir-mode skip --keep-tabular` | `21:26:23` → `21:27:24` | `1` | Lỗi đường dẫn tạm cục bộ `[Errno 2]` (`.../.kaggle/uploads\.cache/scale2_04/upload_index.csv.json`) ([dataset_create.log](file:///c:/Users/Admin/Documents/DoAnTTNT/.cache/scale2_04/dataset_create.log)) | `0` |
| Lần 2 (`SCALE2-04 (tiếp)`) | `.cache/scale2_04/DATASET_CREATE_FAILED_AUTH_20261118.lock` (đổi tên từ `DATASET_CREATE_ONCE_2.lock`) | `kaggle -W datasets create --path "C:\Users\Admin\Documents\DoAnTTNT\.cache\scale2_04\upload" --dir-mode skip --keep-tabular` | `21:48:27` → `21:48:30` (`2,747s`) | `1` | Phiên OAuth hết hạn trước khi gửi byte nào: `Authentication required to call the Kaggle API.` ([dataset_create_2.log](file:///c:/Users/Admin/Documents/DoAnTTNT/.cache/scale2_04/dataset_create_2.log)) | `0` |
| Lần 3 (`SCALE2-04 (tiếp 2)`) | `.cache/scale2_04/DATASET_CREATE_ONCE_3.lock` | `kaggle -W datasets create --path "C:\Users\Admin\Documents\DoAnTTNT\.cache\scale2_04\upload" --dir-mode skip --keep-tabular` | `22:04:33` → `22:22:19` (`1065,96s`) | `0` | **Thành công:** tải đủ `3` file (`12.130.061.692` B), Kaggle tạo dataset riêng tư mới ([dataset_create_3.log](file:///c:/Users/Admin/Documents/DoAnTTNT/.cache/scale2_04/dataset_create_3.log)) | `1` |

## 2. Thời gian upload từng file (Lần 3 — `dataset_create_3.log`)

- Nguồn: `.cache/scale2_04/dataset_create_3_summary.json` và `.cache/scale2_04/dataset_create_3.log`.

| Tệp | Bắt đầu (`+07:00`) | Kết thúc (`+07:00`) | Thời lượng | Ghi nhận CLI |
|---|---|---|---:|---|
| `index.csv` | `22:04:35` (`+2,365s`) | `22:04:40` (`+6,567s`) | `4,202s` | `Upload successful: index.csv (6MB)` |
| `metadata.json` | `22:04:40` (`+6,567s`) | `22:04:42` (`+8,623s`) | `2,056s` | `Upload successful: metadata.json (964B)` |
| `waveforms.f32` | `22:04:42` (`+8,624s`) | `22:22:15` (`+1062,166s`) | `1053,542s` | `Upload successful: waveforms.f32 (11GB)` (`11.3G` trên thanh tqdm) |
| Hoàn tất tạo API | `22:04:33` | `22:22:19` | `1065,960s` | `Your private Dataset is being created.` (`exit=0`) |

## 3. Theo dõi trạng thái `ready` và kiểm tra từ xa

- Theo dõi `kaggle -W datasets status ngctnnguyn1111/doanttnt-scale2-packed-v1` (`.cache/scale2_04/status_polls.json`):
  - Poll 1 (`22:22:47+07:00`): `403 Client Error: Forbidden` (Kaggle đang xử lý tạo bất đồng bộ bộ dữ liệu `12,13 GB`).
  - Poll 2 (`22:25:48+07:00`, cách > 2 phút): `403 Client Error: Forbidden` (kiểm đối chứng dataset cũ cùng lúc trả `ready`, chứng tỏ endpoint bình thường).
  - Poll 3 (`22:30:25+07:00`, cách > 4 phút): **`ready`** (mã thoát `0`).
- Kiểm tra file từ xa (`kaggle -W datasets files ngctnnguyn1111/doanttnt-scale2-packed-v1 --format json`, không tải payload về; `.cache/scale2_04/remote_verify.json`):

| Tệp từ xa | Kích thước từ xa (byte) | Kích thước cục bộ (byte) | Khớp từng byte | `creationDate` (UTC) |
|---|---:|---:|:---:|---|
| `index.csv` | `6.280.248` | `6.280.248` | Có | `2026-11-18T15:22:37.259000` |
| `metadata.json` | `964` | `964` | Có | `2026-11-18T15:22:36.820000` |
| `waveforms.f32` | `12.123.780.480` | `12.123.780.480` | Có | `2026-11-18T15:27:13.674000` |
| **Tổng (`3` tệp cấp gốc)** | **`12.130.061.692`** | **`12.130.061.692`** | **Có** | — |

- Kiểm tra metadata máy chủ (`kaggle -W datasets metadata ...`, `.cache/scale2_04/server_metadata/dataset-metadata.json`): `datasetId = 13459408`, `isPrivate = true`, `licenses = [{"name": "CC0-1.0"}]`.
- Kiểm tra `datasets list --mine --format json`: trước upload `10` dataset (`.cache/scale2_04/datasets_before_3.log`) → sau upload `11` dataset (`.cache/scale2_04/datasets_after.log`), chỉ thêm đúng `ngctnnguyn1111/doanttnt-scale2-packed-v1`.

## 4. Quota Kaggle trước và sau upload

| Thời điểm | GPU đã dùng | GPU còn lại | GPU tổng | TPU còn lại | Làm mới |
|---|---:|---:|---:|---:|---|
| Trước lần 3 (`22:02:04+07:00`, `quota_before_3.log`) | `28.94h` | `16.06h` | `45.00h` | `20.00h / 20.00h` | `2026-11-21T00:00:00` |
| Sau upload (`22:32:30+07:00`, `quota_after.log`) | `28.94h` | `16.06h` | `45.00h` | `20.00h / 20.00h` | `2026-11-21T00:00:00` |

## 5. Gỡ hardlink và xác minh artifact gốc sau khi gỡ

- Đã gỡ `3` hardlink trong `.cache/scale2_04/upload/` (`index.csv`, `metadata.json`, `waveforms.f32`; còn lại `dataset-metadata.json`) lúc `22:54:39+07:00` (`.cache/scale2_04/post_unlink_verify.json`).
- Xác minh 3 file gốc trong `data/processed/packed/train_development_scale2_v1/` đều giữ nguyên `st_nlink = 1`, `st_ino`, kích thước và SHA-256:
  - `index.csv`: `6.280.248` byte, `st_ino = 6192449487790201`, SHA-256 `7eb18914d7d2ea3b101273d0ab4a4e882a0cac4c8676ae5c018a281086aa8080`.
  - `metadata.json`: `964` byte, `st_ino = 3659174697720356`, SHA-256 `3341f75b45f089c4b6b2f71ca3659233a87c6232d416a247b5886d7e0ecec48d`.
  - `waveforms.f32`: `12.123.780.480` byte, `st_ino = 2251799813784224`, SHA-256 `260d93d3762f20b564400d023d599d4151d8e74be618441aed255675f3e3da5e`.
