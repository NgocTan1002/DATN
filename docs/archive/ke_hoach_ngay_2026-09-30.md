# Kế hoạch công việc ngày 30/09/2026

## 1. Kết quả cần đạt cuối ngày

Hôm nay được xem là hoàn thành khi có đủ hai đầu ra:

1. Amplitude audit chạy trên toàn bộ 2.558 mẫu cục bộ và sinh báo cáo JSON/Markdown về peak, RMS, gain, near-silence và peak limiting theo nhãn/kiểu mẫu.
2. Smoke subset cố định, cân bằng `bonafide`/`spoof`, kế thừa tính speaker-disjoint của `closed_train/dev/test` và được DataLoader đọc lặp lại ổn định với seed `2026`.

Mục tiêu phụ là đóng gói sạch phần amplitude policy đã hoàn thành ngày 29/09. Không triển khai LFCC + LCNN trước khi hai đầu ra trên đạt.

## 2. Trạng thái đầu ngày lúc 06:58

### Đã xác minh

- [x] 35/35 unit test đạt trong 4,33 giây.
- [x] Dataset-loader smoke test đạt cho cả `none`, `peak`, `rms_dbfs` trên train/dev/test.
- [x] Batch đầu ra có shape `(2, 64000)`, có đủ nhãn thật/giả và bao gồm audio gốc 40 kHz ở train.
- [x] Năm shard cục bộ có tổng cộng 2.558 mẫu thuộc closed split: train 1.552, dev 719, test 287.
- [x] `git diff --check` không phát hiện lỗi whitespace.

### Đang dở cần đóng gói

- Mã amplitude policy, tích hợp Dataset Loader, kiểm thử và cập nhật tài liệu đang là thay đổi chưa commit.
- Hai tệp đề cương đã được chuyển vào `baocao/`; tệp `1012KH-DHTL_0001.pdf` đang hiện là đã xóa nhưng chưa có bản thay thế trong `baocao/`. Cần xác nhận ý định trước khi commit.
- Chưa có script/báo cáo amplitude audit.
- Chưa có manifest smoke subset và kiểm thử tính cân bằng/tái lập.

## 3. Thứ tự ưu tiên

### P0 — Phải xong hôm nay

- [x] Rà soát diff amplitude policy; hai tệp DOCX trong `baocao/` khớp chính xác bản gốc. Việc xóa `1012KH-DHTL_0001.pdf` vẫn chờ chủ dự án xác nhận.
- [x] Cài script amplitude audit, không thay đổi waveform nguồn.
- [x] Sinh `reports/amplitude_audit.json` và `reports/amplitude_audit.md` từ 2.558 mẫu cục bộ.
- [x] Kiểm tra báo cáo theo `split`, nhãn, `utt_type`, sample rate gốc và policy.
- [x] Thêm kiểm thử cho phép tính/tổng hợp audit và chạy lại toàn bộ test.

### P1 — Nên xong hôm nay

- [x] Tạo smoke manifest cố định với seed `2026`:
  - train: 128 thật + 128 giả;
  - dev: 64 thật + 64 giả;
  - test: 32 thật + 32 giả.
- [x] Với train, phân bổ mẫu giả qua `voice_conversion`, `adversarial_attack`, `replay` khi dữ liệu cục bộ cho phép.
- [x] Xác minh không trùng file, không trùng speaker giữa ba partition và mọi dòng đều tồn tại trong năm shard cục bộ.
- [x] Chạy DataLoader hai lần để xác nhận thứ tự/batch tái lập với cùng seed.
- [x] Cập nhật README và kế hoạch tuần bằng số liệu thực tế.

### P2 — Chỉ làm nếu P0 và P1 đã đạt

- [x] Phác thảo `configs/lfcc_lcnn.json` và giao diện đầu vào/đầu ra của B0.
- [x] Không cài mô hình hoặc bắt đầu pilot trước khi báo cáo audit và smoke manifest được khóa.

## 4. Lịch thực hiện đề xuất

| Thời gian | Khối công việc | Đầu ra kiểm chứng |
|---|---|---|
| 07:00–07:30 | Dọn đầu việc và rà soát thay đổi | Quyết định rõ cho ba tệp báo cáo; diff amplitude policy sẵn sàng |
| 07:30–10:00 | Xây amplitude audit | Script, unit test và schema báo cáo |
| 10:15–11:30 | Chạy audit 2.558 mẫu, đọc kết quả | JSON/Markdown; số lượng mẫu khớp 1.552/719/287 |
| 13:30–15:30 | Tạo smoke subset cân bằng | Ba manifest cố định, tổng 448 mẫu |
| 15:45–16:30 | Kiểm tra DataLoader và tính tái lập | Batch hợp lệ; cùng seed cho cùng kết quả |
| 16:30–17:15 | Chạy regression test và cập nhật tài liệu | Toàn bộ test đạt; README/kế hoạch tuần đồng bộ |
| 17:15–17:30 | Chốt ngày | Ghi vướng mắc và việc đầu tiên ngày 01/10 |

Nghỉ ngắn 10–15 phút giữa các khối; không kéo dài một lỗi quá 45 phút mà không ghi lại nguyên nhân và chuyển sang phương án nhỏ hơn.

## 5. Cổng chất lượng

Amplitude audit chỉ được đánh dấu hoàn thành khi:

- tổng số mẫu đúng 2.558 và không có waveform NaN/Inf;
- thống kê không trộn train/dev/test;
- báo cáo có số lượng near-silence và peak-limited;
- không dùng closed test để chọn policy chiến thắng;
- kết quả đủ để phát hiện chênh lệch biên độ theo nhãn/`utt_type`, chưa diễn giải thành kết luận mô hình.

Smoke subset chỉ được đánh dấu hoàn thành khi:

- cân bằng 50/50 theo nhãn trong từng partition;
- không có file trùng và không có speaker overlap giữa train/dev/test;
- chỉ tham chiếu file đang có cục bộ;
- sinh lại với seed `2026` cho checksum giống nhau;
- subset được ghi rõ là công cụ kiểm tra code, không dùng để báo cáo kết quả khoa học.

## 6. Việc không làm hôm nay

- Không tải toàn bộ 432 shard.
- Không chọn `none`, `peak` hoặc `rms_dbfs` dựa trên closed test hay thống kê audit.
- Không huấn luyện B0/XLS-R và không chạy pilot nhiều epoch.
- Không sửa split protocol đã khóa.

## 7. Mẫu chốt cuối ngày

```text
Đã hoàn thành:
Minh chứng/file đầu ra:
Kiểm thử cuối ngày:
Số liệu audit đáng chú ý:
Vướng mắc còn lại:
Quyết định mới:
Việc đầu tiên ngày 01/10:
```

## 8. Kết quả thực hiện

- **Đã hoàn thành:** amplitude audit, smoke subset, kiểm tra DataLoader tái lập, cập nhật tài liệu và bản nháp giao diện LFCC của B0.
- **Minh chứng:** `reports/amplitude_audit.*`, `reports/smoke_subset_summary.*`, `data/splits/smoke_*.csv`, `configs/lfcc_lcnn.json`.
- **Kiểm thử cuối ngày:** 44/44 unit test đạt; `git diff --check` đạt; dataset-loader smoke test đạt cho ba policy và ba smoke partition.
- **Số liệu audit đáng chú ý:** 2.558 waveform, 7.674 policy observation, 0 near-silence, 4 trường hợp RMS peak-limited. RMS đầu vào median: bonafide -21,29 dBFS, adversarial attack -23,36 dBFS, voice conversion -23,91 dBFS, replay -16,47 dBFS.
- **Smoke subset:** train 256, dev 128, test 64; cân bằng 50/50 từng partition; 0 file trùng; 0 speaker overlap; checksum không đổi sau khi sinh lại.
- **Vướng mắc còn lại:** `1012KH-DHTL_0001.pdf` đang bị xóa khỏi thư mục gốc và chưa có bản trong `baocao/`; chưa thay đổi trạng thái này khi chưa có xác nhận.
- **Quyết định mới:** giữ audit ở vai trò phát hiện shortcut; không chọn policy từ thống kê mô tả hoặc closed test.
- **Việc đầu tiên ngày 01/10:** cài LCNN tối thiểu theo tensor đầu vào `(batch, 1, 60, 401)`, sau đó chạy forward/loss/backward một bước.
