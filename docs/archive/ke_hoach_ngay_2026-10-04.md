# Kế hoạch công việc ngày 04/10/2026

## 1. Mục tiêu hôm nay

Hôm nay là ngày cuối của tuần 01. Mục tiêu chính là **kiểm chứng và chốt tuần**, sau đó bàn giao trạng thái sạch, rõ ràng cho tuần 02 bắt đầu ngày 05/10/2026.

Ba kết quả cần đạt:

1. Xác nhận code hiện tại vẫn vượt toàn bộ cổng kỹ thuật tuần 01.
2. Tạo báo cáo chốt tuần và cập nhật checkpoint vận hành bằng kết quả thực chạy hôm nay.
3. Chuẩn bị phép đo dữ liệu đầu tiên của tuần 02 mà không thay đổi metadata, split hoặc giao thức đã khóa.

## 2. Trạng thái đầu ngày

- Toàn bộ P0, P1 và P2 của tuần 01 đã được đánh dấu hoàn thành.
- G0 đã hoàn thành; G2 đã đạt về kỹ thuật trên smoke subset.
- Lần kiểm chứng gần nhất ngày 02/10 ghi nhận 49/49 kiểm thử đạt.
- Pipeline B0 đã chạy qua Parquet → waveform → LFCC → LCNN → loss → backward → optimizer → checkpoint.
- Điểm nghẽn hiện tại là đọc/decode waveform từ Parquet.
- Cấu hình kỹ thuật đang giữ: `batch_size=8`, `num_workers=0`.
- Git hiện chỉ ghi nhận `docs/decisions.md` có thay đổi chưa commit: bổ sung D015 về trạng thái bản phát hành công khai VSASV.
- `reports/vsasv_snapshot_verification.*` vẫn dùng trạng thái `provisional_public_snapshot`; chưa thay đổi cách diễn giải này khi chưa có quyết định của người dùng.

## 3. P0 — Phải hoàn thành để chốt tuần 01

### 3.1. Kiểm chứng repository hiện tại

- [x] Chạy toàn bộ unit test bằng `python -m unittest discover -s tests -v`.
- [x] Chạy `python scripts/verify_environment.py` để xác nhận môi trường khóa vẫn đúng.
- [x] Chạy `python scripts/check_leakage.py` để kiểm tra lại split ở mức metadata.
- [x] Chạy `python scripts/smoke_test_dataset_loader.py` trên 5 shard cục bộ.
- [x] Ghi đúng số kiểm thử, thời gian, số shard và số mẫu thực tế từ kết quả mới.

Nếu một kiểm tra thất bại, dừng công việc mở rộng và xử lý lỗi thuộc tuần 01 trước khi bàn giao sang tuần 02.

### 3.2. Chốt tài liệu tuần 01

- [x] Tạo `docs/plans/weekly/2026-W40.md`.
- [x] Tổng hợp mục tiêu, đầu ra, kiểm thử, thay đổi kỹ thuật, quyết định và giới hạn diễn giải.
- [x] Ghi rõ smoke subset chỉ chứng minh pipeline hoạt động, chưa phải kết quả khoa học.
- [x] Đối chiếu README, kế hoạch tuần 01, decision log và báo cáo ngày 02/10 để loại số liệu mâu thuẫn.
- [x] Cập nhật `docs/current_state.md` bằng kết quả kiểm chứng thực tế hôm nay.

### 3.3. Kiểm tra phạm vi Git và bàn giao

- [x] Xem lại diff của D015; bảo đảm không có thay đổi ngoài nội dung xác nhận trạng thái phát hành VSASV.
- [x] Kiểm tra `git diff --check` và `git status`.
- [x] Xác nhận không có audio, checkpoint, log lớn hoặc dữ liệu trung gian được đưa vào Git.
- [x] Không tự commit hoặc push; chỉ báo danh sách thay đổi còn lại.

## 4. P1 — Chuẩn bị tuần 02 nếu P0 đạt

### 4.1. Chuẩn hóa phép đo 5 shard hiện có

- [x] Thiết kế hoặc bổ sung script kiểm kê có thể chạy lại cho 5 shard.
- [x] Đầu ra tối thiểu: tên shard, byte, số hàng/mẫu, byte/mẫu, coverage metadata và tổng dung lượng.
- [x] Ghi dung lượng đĩa còn trống tại thời điểm đo.
- [x] Sinh báo cáo JSON để máy đọc và Markdown để xem nhanh.

Đây là số đo cục bộ cho `VSASV-HF-public-snapshot-v1`; không suy rộng thành dung lượng chắc chắn của toàn bộ 432 shard nếu chưa có mẫu tải kiểm chứng.

### 4.2. Chuẩn bị phép đo băng thông

- [x] Chọn tiêu chí cho một shard tải thử: URL nguồn rõ ràng, kích thước biết trước và có checksum.
- [x] Quy định cần ghi: thời điểm bắt đầu/kết thúc, số byte, tốc độ trung bình, checksum và số lần thử.
- [x] Chuẩn bị công thức ETA cho mục tiêu 20.000 và 40.000 mẫu.
- [x] Chưa tải toàn bộ dữ liệu trong hôm nay.

## 5. P2 — Chỉ làm khi P0 và P1 hoàn tất

- [x] Rà các vị trí đang dùng `provisional_public_snapshot` và chuẩn bị phương án diễn giải sau D015.
- [x] Không sửa `reports/vsasv_snapshot_verification.*` cho đến khi người dùng chốt thay đổi trạng thái khoa học.
- [x] Phác thảo schema manifest 20.000 mẫu gồm `shard`, `file`, `speaker`, `split`, `label`, `utt_type` và checksum phiên bản.
- [x] Không tạo manifest bằng cách sửa `closed_*.csv` hoặc `open_*.csv`; manifest phải là lớp giao với audio có thể truy cập.

## 6. Thứ tự thực hiện theo khối thời gian

| Khối | Thời lượng dự kiến | Công việc | Đầu ra |
|---|---:|---|---|
| 1 | 20 phút | Rà `current_state`, D015 và Git | Phạm vi làm việc được xác nhận |
| 2 | 60–90 phút | Unit test, môi trường, leakage và loader smoke | Bộ kết quả kiểm chứng ngày 04/10 |
| 3 | 45–60 phút | Viết tổng kết tuần 01 và cập nhật checkpoint | Báo cáo tuần, `current_state` mới |
| 4 | 60–90 phút | Kiểm kê 5 shard và chuẩn hóa báo cáo | JSON/Markdown dung lượng và coverage |
| 5 | 30 phút | Chuẩn bị giao thức đo băng thông, rà diff/status | Bàn giao rõ cho ngày 05/10 |

Nếu thời gian hạn chế, hoàn thành khối 1–3 trước. Khối 4–5 có thể chuyển sang sáng 05/10 mà không làm tuần 01 bị xem là chưa đạt.

## 7. Cổng hoàn thành hôm nay

- Toàn bộ kiểm tra tuần 01 chạy lại thành công trên code hiện tại.
- Báo cáo tuần ghi đúng số shard, số mẫu và phạm vi smoke.
- `docs/current_state.md` phản ánh kết quả ngày 04/10 và việc đầu tiên của tuần 02.
- Không có thay đổi ngoài ý muốn trong metadata, split, `configs/audio.json` hoặc seed `2026`.
- Có kế hoạch đo dung lượng/băng thông đủ rõ để bắt đầu tuần 02.
- D015 được giữ riêng và không làm thay đổi báo cáo khoa học khi chưa có quyết định.

## 8. Việc không làm hôm nay

- Không huấn luyện B0 đầy đủ hoặc báo cáo EER từ smoke subset.
- Không chọn `none`, `peak` hay `rms_dbfs` làm policy chiến thắng.
- Không dùng closed test để chọn mô hình, threshold, policy hoặc quy mô dữ liệu.
- Không sửa `data/metadata/`, `data/splits/closed_*.csv`, `data/splits/open_*.csv`, seed `2026` hoặc `configs/audio.json`.
- Không triển khai XLS-R, AASIST hoặc demo.
- Không tải toàn bộ 432 shard.
- Không tự commit hoặc push.

## 9. Mẫu chốt cuối ngày

```text
Tuần 01: ĐẠT / CHƯA ĐẠT
Unit test:
Môi trường:
Leakage check:
Dataset loader smoke:
Số shard và số mẫu thực tế:
Tài liệu đã cập nhật:
Thay đổi Git còn lại:
Blocker:
Việc đầu tiên ngày 05/10:
```

## 10. Kết quả thực hiện

- **Tuần 01:** ĐẠT.
- **Unit test:** 52/52 đạt trong 5,749 giây ở lần kiểm tra cuối.
- **Môi trường:** ĐẠT; CPU-only, PyTorch/TorchAudio 2.11.0, DuckDB 1.5.5.
- **Leakage check:** ĐẠT; không phát hiện rò rỉ speaker, file hoặc vi phạm giao thức.
- **Dataset Loader smoke:** ĐẠT trên ba policy và ba split; smoke batch tái lập với seed `2026`.
- **Dữ liệu thực tế:** 5 shard, 2.558 mẫu, 545.811.790 byte; metadata coverage 100%.
- **Dung lượng đĩa còn trống:** 117,91 GiB tại thời điểm đo.
- **Đầu ra mới:** script + 3 kiểm thử kiểm kê storage, báo cáo JSON/Markdown, tổng kết tuần 01 và tài liệu chuẩn bị tuần 02.
- **D015:** đã rà; giữ nguyên `provisional_public_snapshot` trong báo cáo xác minh và chờ người dùng quyết định trước khi đổi cách biểu diễn.
- **Git:** không commit hoặc push; không đưa audio/checkpoint/log lớn vào Git.
- **Việc đầu tiên ngày 05/10:** chọn một shard thiếu có URL và checksum/ETag, đo băng thông tải thực tế rồi tính ETA 20.000/40.000 mẫu.
