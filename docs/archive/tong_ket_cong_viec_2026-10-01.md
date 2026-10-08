# Tổng kết công việc ngày 01/10/2026

## 1. Kết quả tổng thể

Toàn bộ công việc P0, P1 và P2 trong kế hoạch ngày 01/10 đã hoàn thành. Dự án đã chuyển từ trạng thái “khóa giao diện LFCC” sang trạng thái “pipeline B0 tối thiểu chạy xuyên suốt và có thể khôi phục”.

Chuỗi đã được kiểm chứng trên dữ liệu thật:

```text
VSASV Parquet
  → waveform 16 kHz, 64.000 mẫu
  → LFCC (batch, 60, 401)
  → LCNN (batch, 1, 60, 401)
  → raw spoof logit (batch,)
  → BCEWithLogitsLoss
  → backward
  → Adam optimizer step
  → checkpoint save/load
```

Pilot kỹ thuật một epoch hoàn tất 32 bước optimizer trên 256 mẫu train, đánh giá 128 mẫu development và sinh báo cáo có cấu trúc. Toàn bộ 49/49 kiểm thử tự động đạt.

## 2. LCNN tối thiểu đã triển khai

Mô hình mới nằm tại `src/models/lfcc_lcnn.py`. Kiến trúc gồm:

- bốn block convolution với số kênh đầu ra `32 → 64 → 128 → 128`;
- Max-Feature-Map sau mỗi convolution;
- pooling sau ba block đầu;
- adaptive average pooling về `1 × 1`;
- fully connected Max-Feature-Map 64 chiều;
- dropout `0,2` và một raw logit đầu ra.

Mô hình có 498.113 tham số trainable. Forward kiểm tra chặt tensor đầu vào `(batch, 1, 60, 401)`, từ chối shape sai và dữ liệu chứa NaN/Inf. Đầu ra dùng shape `(batch,)` ổn định kể cả khi batch size bằng 1.

### Ý nghĩa

Max-Feature-Map giữ đặc trưng cốt lõi của họ Light CNN: mỗi cặp feature map cạnh tranh trực tiếp và chỉ giữ phản hồi mạnh hơn. Adaptive pooling tách classifier khỏi kích thước không gian trung gian và làm kiến trúc nhỏ đủ để kiểm tra trên CPU. Việc trả raw logit giúp dùng `BCEWithLogitsLoss` đúng cách và tránh áp dụng sigmoid hai lần.

Đây là kiến trúc B0 cho cổng kỹ thuật và smoke pilot. Kết quả hôm nay chưa chứng minh đây là topology LCNN tối ưu cho thí nghiệm khoa học cuối.

## 3. Cấu hình B0 được nâng phiên bản

`configs/lfcc_lcnn.json` được nâng từ phiên bản 1 lên phiên bản 2. Ngoài giao diện LFCC cũ, cấu hình mới ghi rõ:

- topology LCNN và vị trí pooling;
- hidden size, dropout và activation;
- đầu ra là raw logit;
- kết quả smoke validation ngày 01/10;
- số tham số, số bước train, số mẫu development và báo cáo bằng chứng.

Trạng thái cấu hình chuyển từ `draft_interface_locked_implementation_pending` sang `smoke_validation_passed`.

### Ý nghĩa

Kiến trúc không còn tồn tại dưới dạng giả định trong mã nguồn. Cấu hình trở thành hợp đồng có phiên bản giữa LFCC, LCNN, script huấn luyện, checkpoint và tài liệu; nhờ đó các lần chạy sau có thể phát hiện sớm khi shape hoặc topology bị thay đổi ngoài ý muốn.

## 4. Script smoke train có thể tái lập

Script `scripts/smoke_train_lfcc_lcnn.py` được bổ sung để chạy pipeline bằng một lệnh. Script thực hiện:

1. đọc cấu hình B0 và đặt seed `2026`;
2. tạo train/dev DataLoader từ các smoke manifest;
3. sinh LFCC và kiểm tra tensor hữu hạn;
4. chạy LCNN, loss, backward và Adam;
5. kiểm tra toàn bộ gradient hữu hạn và tham số thực sự thay đổi;
6. tính development loss;
7. lưu model state, optimizer state, cấu hình, seed và tiến độ;
8. nạp checkpoint vào model/optimizer mới và so sánh logit;
9. đo thời gian, RSS và peak VRAM;
10. sinh báo cáo JSON và Markdown.

Lệnh chạy một bước:

```powershell
python scripts/smoke_train_lfcc_lcnn.py
```

Lệnh pilot một epoch:

```powershell
python scripts/smoke_train_lfcc_lcnn.py --epochs 1 --max-steps 0
```

### Ý nghĩa

Một lệnh duy nhất thay thế chuỗi thao tác thủ công khó tái lập. Các kiểm tra hữu hạn, cập nhật tham số và khôi phục checkpoint biến những lỗi âm thầm như gradient NaN, optimizer không cập nhật hoặc checkpoint thiếu state thành lỗi rõ ràng ngay trong smoke run.

## 5. Kiểm thử mới

`tests/test_lcnn.py` bổ sung năm kiểm thử:

- Max-Feature-Map chọn đúng giá trị lớn hơn trong mỗi cặp;
- từ chối số feature lẻ;
- forward trả đúng một logit hữu hạn cho mỗi mẫu với batch size 1 và 3;
- từ chối tensor sai shape hoặc chứa NaN;
- loss, backward, gradient và optimizer step đều hoạt động.

`src/models/__init__.py` xuất các thành phần LCNN, builder LFCC/LCNN và hàm đếm tham số để script và test dùng chung một implementation.

### Kết quả hồi quy

```text
Ran 49 tests in 4.099s
OK
```

Trước thay đổi có 44 test; năm test mới nâng tổng số lên 49 mà không làm hỏng các kiểm tra dữ liệu, split, amplitude, LFCC hoặc EER hiện có.

## 6. Kết quả pilot kỹ thuật

Pilot chạy trên CPU, amplitude policy `none`, batch size 8 và learning rate `0,001`.

| Thuộc tính | Kết quả |
|---|---:|
| Train subset | 256 mẫu |
| Development subset | 128 mẫu |
| Epoch | 1 |
| Optimizer step | 32 |
| Tham số trainable | 498.113 |
| Train loss đầu | 0,807307 |
| Train loss cuối | 0,328532 |
| Train loss trung bình | 0,769690 |
| Development BCE loss | 0,962665 |
| Gradient hữu hạn | Đạt |
| Tham số thay đổi | Đạt |
| Sai khác logit sau khôi phục | 0 |

Loss có dao động, với giá trị lớn nhất `1,666013`. Điều này không phải lỗi trong smoke test: toàn bộ loss và gradient vẫn hữu hạn, đủ 32 optimizer step đã hoàn tất và checkpoint khôi phục chính xác.

### Phạm vi diễn giải

Sự giảm của train loss chứng minh optimizer có thể học trên smoke subset. Không được dùng các số loss này để kết luận mô hình tổng quát tốt, so sánh kiến trúc hoặc chọn amplitude policy vì:

- chỉ chạy một epoch;
- chỉ có 5/432 shard cục bộ;
- smoke subset được thiết kế để kiểm tra code;
- policy `none` được chọn cho kiểm tra kỹ thuật, không qua ablation development EER;
- chưa chạy đánh giá khoa học trên closed test.

## 7. Checkpoint và tài nguyên

Checkpoint chính được lưu cục bộ tại `checkpoints/b0_lfcc_lcnn_smoke.pt` và bị loại khỏi Git theo `.gitignore`.

| Thuộc tính | Kết quả |
|---|---:|
| Dung lượng checkpoint | 5.995.301 byte |
| SHA-256 | `0a6fbea4bd8b430bc0ea2af9fa0d7a6b24be40a9f7599ec1f05a0a2b841f9dc0` |
| Thời gian train | 259,039 giây |
| Thời gian development | 78,237 giây |
| Tổng thời gian | 339,946 giây |
| RSS đầu | 212,85 MiB |
| RSS lớn nhất quan sát | 625,03 MiB |
| RSS cuối | 621,71 MiB |
| Peak VRAM | 0 MiB |

### Ý nghĩa

Checkpoint khoảng 5,72 MiB cho thấy B0 smoke đủ nhỏ để lặp nhanh và lưu thường xuyên. Logit khớp chính xác sau khi nạp lại chứng minh model state được bảo toàn. Optimizer state cũng được nạp thành công, tạo nền cho resume training.

Tổng thời gian gần 5 phút 40 giây cho 384 lượt mẫu train/dev cho thấy tốc độ đọc Parquet lười và xử lý CPU là điểm cần đo kỹ trước khi mở rộng development subset lên 20.000–40.000 mẫu. RSS quan sát khoảng 625 MiB hiện chưa phải giới hạn bộ nhớ, nhưng cần đo lại khi tăng worker, prefetch hoặc batch size.

## 8. Tài liệu và quản lý phạm vi

Các tài liệu sau đã được cập nhật:

| Tệp | Thay đổi | Ý nghĩa |
|---|---|---|
| `README.md` | Trạng thái B0, lệnh chạy và giới hạn diễn giải | Người mới có thể tái chạy đúng quy trình |
| `docs/decisions.md` | Thêm quyết định D013 về LCNN smoke | Ghi rõ lý do chọn topology và giới hạn khoa học |
| `docs/plans/daily/2026-10-01.md` | Đánh dấu toàn bộ P0/P1/P2 và ghi kết quả | Kế hoạch ngày có bằng chứng hoàn thành |
| `docs/plans/weekly/2026-W40.md` | Hoàn tất checklist pipeline/checkpoint/pilot | Mục tiêu kỹ thuật tuần 01 hoàn thành sớm |
| `reports/b0_smoke_training.json` | Kết quả có cấu trúc | Dùng cho kiểm tra tự động và tổng hợp sau này |
| `reports/b0_smoke_training.md` | Báo cáo đọc nhanh | Dùng làm bằng chứng khi rà soát hoặc bảo vệ |

Các nguyên tắc phạm vi vẫn được giữ: không tải toàn bộ dataset, không triển khai XLS-R, không sửa split, không dùng test để chọn cấu hình và không chọn amplitude policy từ smoke pilot.

## 9. Mức độ hoàn thành so với kế hoạch tuần

Sau lần chạy này:

- P0 tuần: 8/8 hoàn thành;
- P1 tuần: 4/4 hoàn thành;
- P2 tuần: 3/3 hoàn thành;
- tiêu chí nghiệm thu kỹ thuật cuối tuần: 9/9 đạt.

Mục tiêu kỹ thuật tuần 01 đã hoàn thành trước lịch. Công việc còn lại của tuần là rà soát cuối tuần và chuẩn bị backlog tuần 02, không còn cổng kỹ thuật B0 bị thiếu.

## 10. Việc tiếp theo

Ưu tiên đầu tiên cho ngày 02/10 là đo chi tiết điểm nghẽn của Dataset Loader và DuckDB/Parquet, sau đó thử các phương án cache/index, worker và batch size trên cùng smoke subset. Chỉ sau khi có số đo tải ổn định mới mở rộng development subset lên 20.000–40.000 mẫu.

Các bước nghiên cứu tiếp theo vẫn giữ nguyên thứ tự: hoàn thiện B0 trên dữ liệu lớn hơn, chạy ablation ba amplitude policy bằng closed development EER, khóa policy trước 25/10, rồi mới triển khai XLS-R đóng băng và XLS-R fine-tune một phần.
