# Tổng kết công việc ngày 30/09/2026

## 1. Tổng quan

Trong ngày 30/09/2026, dự án đã hoàn thiện phần kiểm soát biên độ, kiểm tra toàn bộ audio cục bộ, tạo tập dữ liệu smoke test có thể tái lập và khóa giao diện đầu vào cho baseline LFCC + LCNN. Các thay đổi đưa dự án từ trạng thái “Dataset Loader đã đọc được audio” sang trạng thái “pipeline dữ liệu đã có kiểm soát, có báo cáo định lượng và sẵn sàng để cài mô hình cơ sở”.

Các kết quả chính:

- Cài đặt ba chính sách biên độ `none`, `peak`, `rms_dbfs` theo giao thức âm thanh phiên bản 3.
- Chạy amplitude audit trên đủ 2.558 waveform cục bộ, tương ứng 7.674 lượt đánh giá policy.
- Tạo smoke subset cố định gồm 448 mẫu, cân bằng thật/giả và không rò rỉ speaker giữa train/dev/test.
- Xác minh Dataset Loader tạo batch 16 kHz, dài 64.000 mẫu và tái lập với seed `2026`.
- Khóa cấu hình đặc trưng LFCC và giao diện tensor đầu vào cho LCNN.
- Nâng tổng số kiểm thử tự động từ 35 lên 44; toàn bộ 44/44 kiểm thử đạt.
- Cập nhật README, giao thức âm thanh, kế hoạch tuần và nhật ký công việc ngày.
- Sắp xếp lại tài liệu đề cương trong thư mục `baocao/` và xóa tài liệu hướng dẫn PDF không còn sử dụng.

## 2. Thay đổi về xử lý biên độ

### Nội dung đã thực hiện

Ba policy biên độ được cài tại `src/data/amplitude.py` và tích hợp trực tiếp vào `VSASVParquetDataset`:

| Policy | Cách xử lý |
|---|---|
| `none` | Giữ nguyên biên độ sau resample, gain bằng 1 |
| `peak` | Đưa peak của waveform về 0,95 |
| `rms_dbfs` | Đưa RMS về -25 dBFS, đồng thời giới hạn gain để peak không vượt 0,95 |

Các quy tắc an toàn đi kèm:

- Policy được áp dụng sau resample 16 kHz và trước khi cắt/lặp thành đoạn 4 giây.
- Waveform rỗng, sai số chiều, chứa NaN hoặc vô cực bị từ chối.
- Waveform có RMS thấp hơn -50 dBFS được đánh dấu `near_silence` và không bị khuếch đại.
- Không hard clip waveform.
- Khi RMS normalization có nguy cơ làm peak vượt 0,95, gain được giảm và đánh dấu `peak_limited`.
- Mỗi mẫu trả thêm peak, RMS, RMS dBFS, gain, `near_silence` và `peak_limited` để phục vụ audit.

### Ý nghĩa đối với đồ án

Biên độ có thể trở thành shortcut: mô hình có thể dự đoán thật/giả dựa vào độ lớn âm thanh thay vì học dấu hiệu giả mạo. Việc định nghĩa và cài đặt ba policy giúp:

- so sánh các phương án tiền xử lý một cách công bằng;
- ngăn việc chọn normalization bằng cảm tính;
- bảo đảm B0, XLS-R và demo sau này có thể dùng cùng một quy tắc;
- truy vết chính xác biến đổi đã áp dụng cho từng waveform;
- giảm nguy cơ clipping hoặc khuếch đại nhiễu ở đoạn gần im lặng.

Policy chiến thắng **chưa được chọn**. Quyết định này chỉ được thực hiện bằng development EER trong pilot B0, không dựa trên closed test hoặc thống kê mô tả.

## 3. Amplitude audit trên dữ liệu cục bộ

### Phạm vi và kết quả kỹ thuật

Script `scripts/audit_amplitude.py` đã xử lý toàn bộ năm shard cục bộ:

| Thuộc tính | Kết quả |
|---|---:|
| Waveform đã xử lý | 2.558/2.558 |
| Policy trên mỗi waveform | 3 |
| Tổng observation | 7.674 |
| Waveform NaN/Inf | 0 |
| Sai metadata split/Parquet | 0 |
| Waveform gần im lặng | 0 |
| Trường hợp RMS bị giới hạn bởi peak | 4 |

Phân bố 2.558 mẫu cục bộ:

| Split | Số mẫu |
|---|---:|
| Train | 1.552 |
| Development | 719 |
| Test | 287 |

RMS đầu vào median sau khi toàn bộ audio được đưa về 16 kHz:

| Loại utterance | RMS median |
|---|---:|
| Bonafide | -21,29 dBFS |
| Adversarial attack | -23,36 dBFS |
| Voice conversion | -23,91 dBFS |
| Replay | -16,47 dBFS |

Hành vi tổng quát của các policy:

| Policy | Gain median | Gain P95 | Kết quả chính |
|---|---:|---:|---|
| `none` | 1,0000 | 1,0000 | Giữ nguyên phân bố biên độ |
| `peak` | 1,4578 | 3,5392 | Peak đầu ra xấp xỉ 0,95 cho mọi mẫu |
| `rms_dbfs` | 0,7136 | 1,6266 | RMS đầu ra chủ yếu xấp xỉ -25 dBFS; 4 mẫu bị peak-limit |

Báo cáo đầy đủ được lưu tại:

- `reports/amplitude_audit.json`: dữ liệu có cấu trúc cho kiểm tra tự động và tái lập.
- `reports/amplitude_audit.md`: bảng tổng hợp theo split, nhãn, `utt_type`, sample rate gốc và policy.

### Ý nghĩa đối với đồ án

Replay trong năm shard cục bộ có RMS cao hơn đáng kể so với bonafide, VC và adversarial attack. Đây là bằng chứng cho thấy mô hình có thể khai thác biên độ như một shortcut. Phát hiện này có các tác dụng:

- xác nhận amplitude ablation là cần thiết;
- cung cấp cơ sở định lượng cho phần phân tích shortcut trong báo cáo;
- cảnh báo không diễn giải kết quả baseline khi chưa kiểm soát biên độ;
- giúp thiết kế ba pilot B0 công bằng ở giai đoạn chọn policy.

Đây chỉ là bằng chứng cục bộ trên 5/432 shard, không đại diện cho toàn bộ VSASV và không được dùng để kết luận policy nào tốt nhất.

## 4. Smoke subset cân bằng và tái lập

### Nội dung đã thực hiện

Script `scripts/make_smoke_subset.py` tạo ba manifest bằng thứ tự SHA-256 cố định với seed `2026`:

| Split | Tổng | Bonafide | Spoof | VC | AP | Replay |
|---|---:|---:|---:|---:|---:|---:|
| Train | 256 | 128 | 128 | 43 | 43 | 42 |
| Development | 128 | 64 | 64 | 0 | 32 | 32 |
| Test | 64 | 32 | 32 | 0 | 0 | 32 |
| **Tổng** | **448** | **224** | **224** | **43** | **75** | **106** |

Các điều kiện đã được xác minh:

- cân bằng 50/50 giữa bonafide và spoof trong từng partition;
- không có file trùng;
- không có speaker overlap giữa train, development và test;
- mọi file đều tồn tại trong năm shard cục bộ;
- sinh lại với cùng seed cho checksum giống nhau;
- train bao phủ cả voice conversion, adversarial attack và replay;
- subset được đánh dấu rõ là chỉ dùng để kiểm tra code.

Các manifest được lưu tại:

- `data/splits/smoke_train.csv`;
- `data/splits/smoke_dev.csv`;
- `data/splits/smoke_test.csv`.

### Ý nghĩa đối với đồ án

Smoke subset cung cấp một tập nhỏ nhưng có cấu trúc để kiểm tra pipeline nhanh trước khi chạy trên dữ liệu lớn. Nó giúp:

- phát hiện sớm lỗi đọc dữ liệu, batching, nhãn và tensor shape;
- rút ngắn thời gian lặp khi phát triển LFCC + LCNN;
- kiểm tra forward/backward và checkpoint mà chưa cần tải toàn bộ 432 shard;
- tránh việc dùng một vài mẫu chọn thủ công, khó tái lập;
- giữ train/dev/test tách biệt theo speaker ngay cả trong kiểm tra kỹ thuật.

Smoke subset không đại diện cho phân bố đầy đủ của VSASV và không được dùng để báo cáo EER, accuracy hoặc kết luận khoa học.

## 5. Kiểm tra Dataset Loader và tính tái lập

`scripts/smoke_test_dataset_loader.py` đã được mở rộng để kiểm tra:

- cả ba policy `none`, `peak`, `rms_dbfs` trên train/dev/test;
- waveform đầu ra luôn là `float32`, hữu hạn, mono, 16 kHz và dài 64.000 mẫu;
- batch chứa được cả nhãn bonafide và spoof;
- audio voice conversion gốc 40 kHz được resample đúng về 16 kHz;
- các trường audit biên độ đều hữu hạn;
- ba smoke manifest có đủ 256/128/64 mẫu cục bộ;
- hai DataLoader độc lập với cùng seed tạo cùng thứ tự file, nhãn và waveform ở batch đầu.

### Ý nghĩa đối với đồ án

Kiểm tra này chứng minh dữ liệu đi vào mô hình có hình dạng và quy ước nhất quán. Điều đó giảm nguy cơ kết quả mô hình bị sai do tiền xử lý không đồng nhất, crop ngẫu nhiên không kiểm soát hoặc khác biệt sample rate giữa các loại tấn công.

## 6. Khóa giao diện LFCC + LCNN

Cấu hình `configs/lfcc_lcnn.json` đã được tạo cho baseline B0. Các thông số chính:

| Thành phần | Giá trị |
|---|---|
| Waveform đầu vào | `(batch, 64000)` |
| Sample rate | 16 kHz |
| Độ dài | 4 giây |
| Số LFCC | 60 |
| Số linear filter | 80 |
| FFT | 512 mẫu |
| Window | 400 mẫu |
| Hop | 160 mẫu |
| LFCC đầu ra | `(batch, 60, 401)` |
| Tensor vào LCNN | `(batch, 1, 60, 401)` |
| Tensor đầu ra dự kiến | Một raw spoof logit cho mỗi utterance |
| Loss | `BCEWithLogitsLoss` |
| Optimizer smoke test | Adam, learning rate 0,001 |

Kiểm thử tự động đã khởi tạo `torchaudio.transforms.LFCC`, đưa batch waveform qua transform và xác nhận tensor `(2, 60, 401)` hữu hạn.

### Ý nghĩa đối với đồ án

Việc khóa giao diện tensor trước khi cài LCNN giúp:

- loại bỏ mơ hồ về chiều tensor giữa Dataset Loader, LFCC và LCNN;
- phát hiện sớm lỗi cấu hình FFT/window/hop;
- tạo hợp đồng rõ ràng cho phần triển khai mô hình tiếp theo;
- bảo đảm một batch từ smoke subset có thể được đưa vào baseline mà không phải thay đổi pipeline dữ liệu.

Hiện mới khóa phần LFCC và giao diện. Kiến trúc LCNN, forward pass và vòng tối ưu chưa được triển khai.

## 7. Kiểm thử và tài liệu

### Kiểm thử tự động

Tổng số kiểm thử tăng từ 35 lên 44. Toàn bộ 44/44 kiểm thử đạt, bao phủ:

- các phép biến đổi biên độ;
- waveform gần im lặng và waveform không hữu hạn;
- giới hạn gain để tránh clipping;
- tích hợp policy vào Dataset Loader;
- tổng hợp amplitude audit;
- lựa chọn smoke subset cân bằng và xác định;
- phát hiện file hoặc speaker overlap;
- shape và tính hữu hạn của LFCC;
- các kiểm thử cũ về split, segmentation, EER và Dataset Loader.

### Tài liệu được cập nhật

- `README.md`: trạng thái mới, cách chạy audit và cách tạo smoke subset.
- `docs/audio_protocol.md`: kết quả audit và cảnh báo shortcut biên độ.
- `docs/ke_hoach_tuan_01_2026-09-28_2026-10-04.md`: cập nhật các đầu việc đã hoàn thành.
- `docs/ke_hoach_ngay_2026-09-30.md`: kế hoạch và kết quả thực hiện trong ngày.

### Quản lý tài liệu và Git

- Hai tệp đề cương được chuyển nguyên vẹn vào `baocao/`.
- Tệp `1012KH-DHTL_0001.pdf` không còn sử dụng đã được xóa bằng một commit riêng.
- Repository hiện sạch và đồng bộ với `origin/main`.

Các commit đã tạo:

| Commit | Nội dung |
|---|---|
| `83a9c90` | `feat(data): add amplitude auditing and deterministic smoke datasets` |
| `1a56809` | `feat(model): define and validate LFCC-LCNN input interface` |
| `4a2dedb` | `docs(project): update implementation status and daily plan` |
| `688ef15` | `chore(docs): move thesis proposal files into report directory` |
| `5384abf` | `chore(docs): remove obsolete project guideline PDF` |

## 8. Tác động tổng thể đối với đồ án

Các thay đổi hôm nay mang lại bốn giá trị chính:

1. **Tăng độ tin cậy khoa học:** các shortcut do biên độ và sample rate được nhận diện, kiểm soát và đưa vào kế hoạch ablation.
2. **Tăng khả năng tái lập:** split, smoke subset, crop, policy và LFCC đều có seed hoặc cấu hình được khóa; báo cáo có checksum và dữ liệu JSON.
3. **Giảm rủi ro kỹ thuật:** lỗi dữ liệu, shape, NaN/Inf, clipping và speaker leakage được chặn trước khi bắt đầu huấn luyện.
4. **Tăng tốc phát triển:** smoke subset 448 mẫu cho phép kiểm tra model/training/checkpoint nhanh mà không cần tải hoặc quét toàn bộ dataset.

Sau ngày 30/09, phần dữ liệu tối thiểu cho baseline đã sẵn sàng. Trọng tâm tiếp theo có thể chuyển từ “đảm bảo dữ liệu đúng” sang “chứng minh một vòng huấn luyện LFCC + LCNN chạy xuyên suốt”.

## 9. Giới hạn hiện tại

- Audio cục bộ mới gồm 5/432 shard và không phải mẫu ngẫu nhiên.
- Test cục bộ trong smoke subset chỉ có replay ở phía spoof do giới hạn dữ liệu đang có.
- Amplitude audit là thống kê mô tả, không phải kết quả đánh giá mô hình.
- Chưa được chọn amplitude policy chiến thắng.
- LCNN chưa được cài đặt.
- Chưa có forward/loss/backward, optimizer step hoặc checkpoint thử.
- Chưa đo thời gian huấn luyện, RAM hoặc VRAM của B0.

## 10. Công việc tiếp theo

Thứ tự thực hiện đề xuất cho ngày 01/10/2026:

1. Cài LCNN tối thiểu nhận tensor `(batch, 1, 60, 401)`.
2. Chạy một batch smoke qua LFCC và LCNN; xác nhận logit có shape `(batch,)` và hữu hạn.
3. Tính `BCEWithLogitsLoss`, chạy backward và một optimizer step.
4. Lưu rồi nạp lại checkpoint; xác nhận output trước/sau khôi phục thống nhất.
5. Ghi thời gian chạy, RAM/VRAM, loss và lỗi kỹ thuật.
6. Chỉ khi vòng kỹ thuật trên đạt mới cân nhắc pilot 1–3 epoch trên smoke subset.

Không tải toàn bộ dataset, không triển khai XLS-R và không chọn amplitude policy trước khi baseline B0 chạy ổn định.
