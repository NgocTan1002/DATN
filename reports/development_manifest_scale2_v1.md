# Báo cáo development manifest scale2 (`development-scale2-v1`, D036)

- Thời điểm tạo (UTC): `2026-11-18T08:34:55.781206+00:00`
- Trạng thái: **ĐẠT**
- Đường dẫn manifest: `data/manifests/development_scale2_v1.csv`
- SHA-256 manifest: `6533fced4b3a25ebab9f037a62d34774b44c2fa1f00d17040fd815bd17fa51a3`
- Số mẫu: `31,770` (`15,885` train ×1 giữ nguyên phần đầu + `15,885` mới).
- Số speaker: `225` (tăng từ `153` của train ×1); số shard được dùng: `89`.
- Quy tắc thứ tự dòng: Dòng 1..15.885 giữ nguyên thứ tự gốc của closed_train trong development_20k_v2.csv (để bảo toàn source_index và _crop_seed); dòng 15.886..31.770 là 15.885 mẫu mới xếp theo utt_type (bonafide, voice_conversion, adversarial_attack, replay) và SHA256('2026|file') tăng dần.

## 1. Thống kê ứng viên, khử trùng và số chọn theo `utt_type`

| `utt_type` | Đã có (×1) | Phần thêm cần | Tổng ứng viên thô (cục bộ+tải) | Loại `train` | Loại `dev` | Loại `test` | Loại nội bộ | Ứng viên sạch | Số mới chọn | Tổng trong `scale2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bonafide` | 7,134 | 7,133 | 9,424 | 60 | 45 | 0 | 70 | 9,249 | 7,133 | 14,267 |
| `voice_conversion` | 4,348 | 4,349 | 6,157 | 0 | 0 | 0 | 0 | 6,157 | 4,349 | 8,697 |
| `adversarial_attack` | 4,348 | 4,349 | 4,403 | 0 | 0 | 0 | 0 | 4,403 | 4,349 | 8,697 |
| `replay` | 55 | 54 | 105 | 0 | 0 | 0 | 0 | 105 | 54 | 109 |
| **Tổng** | **15,885** | **15,885** | **20,089** | **60** | **45** | **0** | **70** | **19,914** | **15,885** | **31,770** |

## 2. Kết quả kiểm chứng độc lập từ Parquet (Bước 6)

- Audit file: `reports/development_manifest_scale2_v1_content_audit.json` (thời gian băm lại `31.770` waveform: `1527.91 s`).
- Đã băm lại: `31,770/31,770`; content hash duy nhất: `31,770`.
- Trùng nội bộ: `0`; trùng `dev-full` (`kept`): `0`; trùng `test-full` (`kept`): `0`.
- Nhóm mẫu mới (`15.885` dòng mới) trùng toàn bộ ledger `closed_dev` / `closed_test`: `0` / `0`.
- Giữ nguyên trọn `15.885` mẫu train ×1 và khớp audit cũ: `True` (lệch hash: `0`).
- Speaker overlap với `closed_dev`: `0`; metadata & sample rate khớp: `True`.
