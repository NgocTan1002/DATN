# Kế hoạch công việc ngày 01/10/2026

## 1. Mục tiêu trọng tâm

Hôm nay chuyển dự án từ trạng thái **đã khóa giao diện LFCC** sang trạng thái **pipeline LFCC + LCNN chạy được một bước tối ưu hoàn chỉnh**.

Ngày làm việc được xem là đạt khi có đủ ba kết quả:

1. LCNN tối thiểu nhận tensor `(batch, 1, 60, 401)` và trả một raw spoof logit cho mỗi mẫu.
2. Một batch từ smoke subset đi qua `waveform -> LFCC -> LCNN -> BCEWithLogitsLoss -> backward -> optimizer step` với toàn bộ giá trị hữu hạn.
3. Toàn bộ kiểm thử cũ và mới đạt; trạng thái, số liệu chạy thử và phần còn thiếu được ghi lại rõ ràng.

Checkpoint thử là mục tiêu nên hoàn thành. Pilot 1–3 epoch chỉ thực hiện nếu ba kết quả trên đã đạt ổn định.

## 2. Trạng thái đầu ngày lúc 11:27

### Đã hoàn thành và đã kiểm chứng

- [x] Metadata audit trên 220.963 mẫu và 1.141 speaker.
- [x] Tám split speaker-disjoint đã khóa với seed `2026`.
- [x] Leakage checker đạt ở mức metadata.
- [x] Dataset Loader trả waveform `float32`, mono, 16 kHz, dài 64.000 mẫu.
- [x] Ba amplitude policy `none`, `peak`, `rms_dbfs` đã được cài và audit trên 2.558 waveform cục bộ.
- [x] Smoke subset 448 mẫu cân bằng nhãn, tách biệt speaker và tái lập bằng seed `2026`.
- [x] LFCC sinh tensor `(batch, 60, 401)` hữu hạn; đầu vào LCNN đã khóa là `(batch, 1, 60, 401)`.
- [x] 44/44 kiểm thử hiện tại đạt ngày 01/10/2026, thời gian chạy 5,652 giây.

### Các khoảng trống tại mốc đầu ngày

Các mục sau là trạng thái lúc 11:27 và đã được xử lý trong kết quả cuối ngày ở mục 10:

- Chưa có lớp LCNN.
- Chưa có forward pass sinh logit `(batch,)`.
- Chưa có loss, backward hoặc optimizer step.
- Chưa có checkpoint B0 thử nghiệm.
- Chưa đo thời gian và bộ nhớ cho bước tối ưu B0.

### Trạng thái Git cần lưu ý

- Nhánh `main` đang cùng mốc với `origin/main`.
- Có hai tài liệu chưa được Git theo dõi: `docs/kien_thuc_can_nam_vung_de_bao_ve_do_an.md` và `docs/plans/daily/2026-09-30.md`.
- Không gộp hai tài liệu này vào commit kỹ thuật trước khi rà soát nội dung và mục đích lưu trữ.

## 3. Thứ tự công việc

### P0 — Phải hoàn thành hôm nay

- [x] Khóa kiến trúc LCNN tối thiểu đủ cho smoke test; ghi rõ đầu vào, các khối chính và đầu ra.
- [x] Cài mô hình trong `src/models/` với kiểm tra shape rõ ràng.
- [x] Thêm kiểm thử cho:
  - logit có shape `(batch,)`;
  - logit hữu hạn;
  - batch size 1 và batch size lớn hơn 1 đều chạy;
  - gradient tồn tại và hữu hạn sau backward.
- [x] Chạy một batch thật từ `smoke_train.csv` qua LFCC và LCNN.
- [x] Tính `BCEWithLogitsLoss`, chạy backward và một bước Adam với learning rate `0.001`.
- [x] Xác nhận tham số mô hình thay đổi sau optimizer step và không xuất hiện NaN/Inf.
- [x] Chạy lại toàn bộ bộ kiểm thử.

### P1 — Nên hoàn thành hôm nay

- [x] Tạo một script smoke train có thể chạy lại bằng một lệnh và seed `2026`.
- [x] Lưu checkpoint thử gồm model state, optimizer state, step, seed và cấu hình B0.
- [x] Nạp checkpoint vào mô hình mới và xác minh output khớp khi ở chế độ đánh giá trên cùng input.
- [x] Ghi thời gian forward/backward, thời gian toàn bước và mức dùng RAM/VRAM nếu có.
- [x] Cập nhật `README.md`, `configs/lfcc_lcnn.json` và kế hoạch tuần theo trạng thái thực tế.

### P2 — Chỉ làm khi P0 và P1 đã đạt

- [x] Chạy pilot kỹ thuật một epoch trên `smoke_train`, theo dõi loss train và development.
- [x] Xác nhận loss có thể tính ổn định qua 32 batch và checkpoint nạp lại được cả model/optimizer state.
- [x] Ghi kết quả dưới nhãn **smoke test kỹ thuật**, không diễn giải thành kết quả khoa học.

## 4. Lịch thực hiện từ 11:30

| Thời gian | Khối công việc | Đầu ra kiểm chứng |
|---|---|---|
| 11:30–12:00 | Khóa phạm vi LCNN và viết tiêu chí test trước khi cài | Sơ đồ tensor, đầu ra và danh sách test rõ ràng |
| 12:00–13:00 | Nghỉ trưa | Giữ khối chiều liên tục, tránh kéo dài lỗi từ buổi sáng |
| 13:00–14:30 | Cài LCNN tối thiểu và unit test | Forward trên tensor giả đạt; logit `(batch,)` hữu hạn |
| 14:30–14:45 | Nghỉ ngắn | — |
| 14:45–15:45 | Nối Dataset Loader, LFCC, LCNN và loss | Một batch smoke thật chạy xuyên suốt |
| 15:45–16:00 | Nghỉ ngắn | — |
| 16:00–16:45 | Backward, optimizer step và kiểm tra gradient | Loss/gradient hữu hạn; tham số được cập nhật |
| 16:45–17:15 | Checkpoint, nạp lại và đo thời gian/bộ nhớ | Checkpoint khôi phục đúng hoặc lỗi được ghi cụ thể |
| 17:15–17:45 | Regression test và cập nhật tài liệu | Toàn bộ test đạt; README/kế hoạch tuần đồng bộ |
| Sau 17:45 | Chốt ngày; chỉ bắt đầu pilot nếu mọi cổng trên đạt | Nhật ký cuối ngày và việc đầu tiên cho 02/10 |

Nếu một lỗi kéo dài quá 45 phút, ghi lại input, shape, stack trace và giả thuyết nguyên nhân; thu nhỏ về tensor giả trước khi quay lại DataLoader.

## 5. Trình tự kỹ thuật bắt buộc

```text
Kiểm tra LCNN bằng tensor giả
        ↓
Kiểm tra LFCC + LCNN bằng waveform giả
        ↓
Chạy một batch thật từ smoke_train
        ↓
Tính loss và backward
        ↓
Optimizer step
        ↓
Lưu/nạp checkpoint
        ↓
Chạy toàn bộ regression test
```

Không bỏ qua bước tensor giả. Việc tách ba tầng kiểm tra giúp xác định lỗi thuộc mô hình, đặc trưng LFCC hay dữ liệu.

## 6. Cổng chất lượng

LCNN chỉ được đánh dấu hoàn thành khi:

- đầu vào đúng `(batch, 1, 60, 401)`;
- đầu ra đúng `(batch,)`, kể cả khi batch size bằng 1;
- logit, loss và toàn bộ gradient được kiểm tra đều hữu hạn;
- mô hình không phụ thuộc vào kích thước batch cố định;
- nhãn tuân theo `0 = bonafide`, `1 = spoof`;
- không áp dụng sigmoid trước `BCEWithLogitsLoss`.

Checkpoint chỉ được đánh dấu hoàn thành khi:

- lưu đủ model state, optimizer state, step, seed và cấu hình;
- nạp được trên môi trường hiện tại;
- cùng model state và cùng input cho output nhất quán ở chế độ đánh giá;
- đường dẫn checkpoint thử không được đưa tệp trọng số lớn vào Git.

## 7. Việc không làm hôm nay

- Không tải toàn bộ 432 shard.
- Không triển khai XLS-R, AASIST, open-set evaluation hoặc demo.
- Không chọn amplitude policy chiến thắng từ smoke subset, thống kê audit hoặc closed test.
- Không báo cáo EER từ smoke subset như kết quả khoa học.
- Không chỉnh sửa các split đã khóa.
- Không tối ưu kiến trúc hoặc hyperparameter trước khi một bước tối ưu cơ bản chạy ổn định.

## 8. Thứ tự cắt giảm khi thiếu thời gian

Nếu thời gian không đủ, cắt công việc theo thứ tự:

1. Bỏ pilot nhiều epoch.
2. Hoãn đo bộ nhớ chi tiết, nhưng vẫn ghi thời gian chạy.
3. Hoãn tinh chỉnh kiến trúc LCNN.
4. Giữ nguyên các việc bắt buộc: forward, loss, backward, optimizer step và regression test.

## 9. Mẫu chốt cuối ngày

```text
Đã hoàn thành:
Minh chứng/file đầu ra:
Kiểm thử cuối ngày:
Shape input/output:
Loss smoke test:
Thời gian và bộ nhớ:
Checkpoint:
Vướng mắc:
Quyết định mới:
Việc đầu tiên ngày 02/10:
```

## 10. Kết quả thực hiện

- **LCNN:** bốn block convolution dùng Max-Feature-Map, ba tầng pooling, adaptive average pooling và head một raw logit; tổng cộng 498.113 tham số trainable.
- **Pipeline:** batch waveform `(8, 64000)` tạo LFCC `(8, 60, 401)`, LCNN input `(8, 1, 60, 401)` và logit `(8,)`.
- **Một bước tối ưu:** loss, logit và gradient đều hữu hạn; Adam làm thay đổi tham số mô hình.
- **Pilot kỹ thuật:** một epoch, 32 bước, đủ 256 mẫu train; loss đầu `0,807307`, loss cuối `0,328532`, loss trung bình `0,769690`.
- **Development:** đủ 128 mẫu; BCE loss trung bình `0,962665`, hữu hạn.
- **Checkpoint:** dung lượng 5.995.301 byte; nạp lại model/optimizer thành công; sai khác logit lớn nhất sau khôi phục bằng `0`.
- **Tài nguyên CPU:** train `259,039` giây; development `78,237` giây; tổng `339,946` giây; RSS lớn nhất quan sát `625,03 MiB`; VRAM `0 MiB`.
- **Kiểm thử cuối ngày:** 49/49 kiểm thử đạt trong `4,099` giây; `git diff --check` không phát hiện lỗi whitespace.
- **Minh chứng:** `reports/b0_smoke_training.json`, `reports/b0_smoke_training.md` và checkpoint cục bộ `checkpoints/b0_lfcc_lcnn_smoke.pt`.
- **Giới hạn:** pilot dùng amplitude policy `none` và smoke subset cục bộ; không phải kết quả khoa học và không được dùng để chọn policy.
- **Việc đầu tiên ngày 02/10:** đo điểm nghẽn đọc Parquet/DataLoader trước khi mở rộng development subset lên 20.000–40.000 mẫu.
