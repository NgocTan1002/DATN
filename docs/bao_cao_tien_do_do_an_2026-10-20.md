# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN TỐT NGHIỆP

## 1. Thông tin chung

- **Tên đề tài:** Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần.
- **Thời điểm báo cáo:** Ngày 20/10/2026.
- **Bài toán chính:** Phân loại một đoạn tiếng nói thành hai nhóm: tiếng nói thật (`bonafide`) hoặc tiếng nói giả mạo (`spoof`).
- **Mô hình đối chứng:** LFCC + LCNN.
- **Mô hình nghiên cứu chính:** XLS-R đóng băng và XLS-R tinh chỉnh từng phần.

## 2. Mục đích của báo cáo

Báo cáo này trình bày những công việc đã thực hiện, kết quả đã kiểm chứng, ý nghĩa của từng hạng mục và kế hoạch tiếp theo của đồ án. Báo cáo cũng nêu rõ những phần chưa hoàn thành để tránh nhầm lẫn giữa việc pipeline đã chạy được và việc đã có kết quả nghiên cứu cuối cùng.

## 3. Tóm tắt tiến độ hiện tại

Đến ngày 20/10/2026, đồ án đã hoàn thành phần nền tảng gồm kiểm tra dữ liệu, tạo tập train/development không rò rỉ nội dung, chuẩn hóa âm thanh, xây dựng bộ đọc dữ liệu, triển khai mô hình LFCC + LCNN, xây dựng các chỉ số đánh giá, checkpoint có thể tiếp tục chính xác và quy trình chạy GPU trên Kaggle.

Đồ án cũng đã chạy benchmark ba lần trên Kaggle để đo thời gian thực tế. Từ kết quả này, cấu hình chung cho ba thí nghiệm amplitude đã được khóa ở 896 bước huấn luyện. Ba cấu hình `none`, `peak` và `rms_dbfs` đã sẵn sàng để chạy.

Hiện tại, dự án đang ở giai đoạn bắt đầu thu kết quả khoa học. Ba thí nghiệm amplitude chưa được chạy, do đó chưa chọn được cách chuẩn hóa biên độ tốt nhất. Mô hình B0 đầy đủ, XLS-R và đánh giá test cuối cũng chưa được thực hiện.

## 4. Mục tiêu và phạm vi nghiên cứu đã thống nhất

Đề tài chỉ giải quyết bài toán phát hiện tiếng nói thật hoặc giả mạo. Ba loại tấn công trong dữ liệu gồm `voice_conversion`, `adversarial_attack` và `replay` đều được gộp vào lớp giả mạo. Thông tin loại tấn công chỉ được sử dụng để thống kê và phân tích lỗi, không phải đầu ra của mô hình.

Giao thức chính là `closed_train/dev/test`. Người nói giữa các tập không được trùng nhau. Seed cố định là `2026`. Tập test không được dùng để chọn mô hình, chọn threshold, chọn cách chuẩn hóa biên độ hoặc điều chỉnh ngân sách huấn luyện.

Việc thu hẹp phạm vi giúp đề tài tập trung vào một câu hỏi rõ ràng: XLS-R và kỹ thuật tinh chỉnh từng phần có cải thiện khả năng phát hiện tiếng nói giả mạo tiếng Việt so với baseline LFCC + LCNN hay không.

## 5. Những công việc đã hoàn thành

### 5.1. Kiểm tra metadata và cấu trúc dữ liệu

Metadata VSASV đã được kiểm tra về số lượng, schema, nhãn, đường dẫn, dữ liệu thiếu và dòng trùng. Metadata hiện có 220.963 mẫu và 1.141 người nói.

Tám tập dữ liệu theo giao thức closed/open đã được tạo bằng seed `2026`. Giao thức chính `closed_train/dev/test` được chia theo người nói để một người không xuất hiện ở cả tập huấn luyện và tập đánh giá.

**Ý nghĩa:** bước này giúp giảm nguy cơ mô hình ghi nhớ giọng của người nói thay vì học dấu hiệu của tiếng nói giả mạo. Quy tắc chia cố định cũng giúp kết quả có thể được tái tạo.

### 5.2. Kiểm kê dữ liệu âm thanh thực tế

Máy hiện có 68 shard, chứa 34.782 waveform với tổng dung lượng khoảng 9,85 GiB. Các shard đã được kiểm tra kích thước, mã SHA-256, khả năng đọc Parquet và độ khớp với metadata.

Dữ liệu hiện có chưa phải toàn bộ dữ liệu được mô tả trong bài báo VSASV gốc. Vì vậy, báo cáo cuối cần ghi rõ số shard và số mẫu thực tế được sử dụng, đồng thời không tuyên bố tái lập kết quả chính thức của bài báo.

**Ý nghĩa:** đồ án biết chính xác mình đang sử dụng phần dữ liệu nào, tránh báo cáo quá phạm vi bằng chứng thực tế.

### 5.3. Chuẩn hóa âm thanh

Toàn bộ âm thanh được chuyển về mono 16 kHz. Mỗi mẫu đầu vào có độ dài 4 giây, tương đương 64.000 điểm lấy mẫu.

Khi huấn luyện, hệ thống chọn ngẫu nhiên một đoạn 4 giây theo seed và epoch. Khi đánh giá, hệ thống lấy đoạn ở giữa để kết quả ổn định. Tệp ngắn hơn 4 giây được lặp lại rồi cắt đủ độ dài.

Ba cách xử lý biên độ đã được định nghĩa:

- `none`: giữ nguyên biên độ;
- `peak`: đưa peak về mức 0,95;
- `rms_dbfs`: đưa RMS về -25 dBFS nhưng vẫn giới hạn peak ở 0,95.

Các đoạn gần im lặng không bị khuếch đại. Cả ba cách xử lý đã vượt qua kiểm tra kỹ thuật trên train và development.

**Ý nghĩa:** việc đưa audio về cùng sample rate giúp tránh trường hợp mô hình phân biệt thật/giả bằng đặc điểm kỹ thuật của file. Thử nghiệm ba cách xử lý biên độ giúp kiểm tra xem biên độ có đang trở thành một “đường tắt” cho mô hình hay không.

### 5.4. Phát hiện và xử lý dữ liệu trùng nội dung

Phiên bản development manifest đầu tiên gồm 20.000 mẫu. Sau khi kiểm tra waveform, dự án phát hiện 91 nhóm với 182 file bonafide trùng nội dung trong phần audio cục bộ.

Manifest v1 giữ nhiều file của 38 nhóm trùng và có 17 nhóm xuất hiện ở cả train và development. Đây là rò rỉ dữ liệu, dù tên file và người nói không trùng nhau.

Để xử lý, dự án đã tạo `development-20k-v2`. Mỗi waveform được băm dựa trên sample rate và toàn bộ nội dung. Với mỗi nhóm trùng, chỉ một đại diện xác định được giữ lại. Manifest v2 vẫn có đúng 20.000 mẫu, gồm:

- 15.885 mẫu train;
- 4.115 mẫu development;
- 184 người nói;
- 66 shard.

Một lượt audit độc lập đã đọc lại 20.000/20.000 waveform và xác nhận có 20.000 content hash duy nhất. Không còn nội dung trùng trong cùng tập hoặc đi qua train và development.

**Ý nghĩa:** đây là cổng quan trọng để kết quả development có giá trị. Nếu cùng một nội dung xuất hiện ở train và development, mô hình có thể ghi nhớ waveform và tạo ra EER thấp giả tạo.

### 5.5. Xây dựng bộ đọc dữ liệu

`VSASVManifestDataset` đã được xây dựng để đọc trực tiếp manifest v2 và các shard Parquet. Loader kiểm tra schema, phiên bản manifest, nhãn, người nói, shard, sample rate và metadata trước khi trả dữ liệu cho mô hình.

Mỗi mẫu trả về waveform `float32` dài 64.000 điểm cùng nhãn, tên file, người nói, loại tấn công, sample rate và thông tin biên độ.

Loader đã được kiểm tra trên đủ 15.885 mẫu train và 4.115 mẫu development. Cả ba amplitude policy đều hoạt động ổn định và có thể tái tạo bằng seed `2026`.

**Ý nghĩa:** loader là cầu nối giữa dữ liệu và mô hình. Việc kiểm tra chặt ngay tại loader giúp lỗi dữ liệu dừng sớm, thay vì âm thầm làm sai kết quả huấn luyện.

### 5.6. Đo và tối ưu hiệu năng pipeline

Các benchmark cho thấy phần đọc và giải mã Parquet chiếm phần lớn thời gian. LFCC và LCNN không phải nút thắt chính.

Dự án đã thử các cấu hình batch size và số worker khác nhau. Kết quả hiện tại giữ `batch_size=8` và `num_workers=0`. Cấu hình hai worker có nhanh hơn trên một phép đo nhỏ nhưng không vượt ngưỡng cải thiện 15% và sử dụng nhiều RAM hơn.

Phương án đọc nhiều mẫu cùng shard trong một query cũng đã được thử. Nó có hiệu quả khi cả batch nằm cùng shard, nhưng thứ tự shuffle thực tế làm mỗi batch trải trên nhiều shard. Kết quả end-to-end giảm 0,93%, vì vậy phương án này đã được gỡ khỏi production.

Một proof-of-concept packed artifact trên 24 mẫu cho thấy khả năng tăng tốc lớn. Tuy nhiên, kết quả này chưa được ngoại suy lên 20.000 mẫu và chưa được dùng làm đường dữ liệu chính.

**Ý nghĩa:** dự án chọn cấu hình bằng số liệu thực tế và loại các phương án không đạt điều kiện, thay vì giữ tối ưu hóa chỉ vì nó có vẻ hợp lý về lý thuyết.

### 5.7. Triển khai baseline LFCC + LCNN

Baseline B0 đã được triển khai bằng LFCC 60 hệ số và LCNN có bốn block convolution với Max-Feature-Map. Mô hình có 498.113 tham số có thể huấn luyện và tạo ra một score cho lớp spoof.

Pipeline đã chạy qua các bước:

1. đọc waveform;
2. tạo đặc trưng LFCC;
3. chạy forward qua LCNN;
4. tính loss;
5. lan truyền ngược;
6. cập nhật optimizer;
7. đánh giá development;
8. lưu và nạp checkpoint.

**Ý nghĩa:** baseline B0 là mốc so sánh bắt buộc. Sau này chỉ có thể đánh giá lợi ích của XLS-R khi cả hai mô hình được chạy trên cùng dữ liệu và giao thức.

### 5.8. Xây dựng các chỉ số đánh giá

Hệ thống đã triển khai ba chỉ số:

- **EER:** tỷ lệ lỗi tại điểm false acceptance và false rejection cân bằng;
- **ROC-AUC:** khả năng xếp hạng mẫu thật và giả bằng score;
- **Macro-F1:** chất lượng quyết định nhị phân khi hai lớp được coi trọng như nhau.

Threshold được chọn tại điểm EER trên development. Threshold này phải được giữ nguyên khi tính macro-F1 trên test và khi dùng cho demo. Test không được dùng để tính lại threshold.

**Ý nghĩa:** quy tắc này ngăn hệ thống điều chỉnh theo test và bảo vệ tính khách quan của kết quả cuối.

### 5.9. Hoàn thiện checkpoint và khả năng tiếp tục huấn luyện

Entrypoint production `scripts/train_b0.py` đã tích hợp train, development evaluation, metric, report và checkpoint trong một đường chạy.

Checkpoint lưu:

- tham số mô hình;
- trạng thái optimizer;
- số bước đã chạy;
- epoch và vị trí batch tiếp theo;
- trạng thái sinh số ngẫu nhiên của Python, NumPy, PyTorch và CUDA;
- mã kiểm tra của manifest và cấu hình;
- seed, amplitude policy, batch size, learning rate và step budget.

Khi chạy liên tục và khi dừng rồi tiếp tục bằng process mới, 12/12 tensor cuối của mô hình khớp hoàn toàn trên smoke dữ liệu thật.

**Ý nghĩa:** Kaggle có thể ngắt phiên. Exact resume giúp quá trình huấn luyện tiếp tục đúng vị trí mà không lặp hoặc bỏ batch, đồng thời tránh thay đổi quỹ đạo ngẫu nhiên của mô hình.

### 5.10. Xây dựng quy trình Kaggle CLI

Kaggle đã được chọn làm môi trường GPU chính. Dữ liệu và mã nguồn được lưu trong dataset riêng tư; internet của kernel mặc định bị tắt.

Workflow CLI đã hỗ trợ:

- kiểm tra tài khoản và môi trường;
- đóng gói mã nguồn;
- tạo phiên bản source dataset riêng tư;
- chuẩn bị và đẩy kernel;
- theo dõi trạng thái;
- tải report và checkpoint về máy.

Mỗi source bundle có SHA-256 và danh sách mã kiểm tra từng file. Runner sẽ từ chối chạy nếu source thực tế không khớp cấu hình đã ghi.

Kaggle kernel version 7 đã xác nhận checkpoint production và CUDA RNG có thể lưu/khôi phục. Snapshot mã nguồn mới nhất gồm 48 file đã được publish riêng tư.

**Ý nghĩa:** quy trình này giảm thao tác thủ công trên notebook và giúp mỗi thí nghiệm có thể truy ngược về source, cấu hình, dữ liệu và runtime đã sử dụng.

### 5.11. Benchmark Kaggle và khóa ngân sách pilot

Để đo thời gian đại diện, dự án tạo một tập technical benchmark cân bằng gồm:

- 256 mẫu train, cân bằng 128 bonafide và 128 spoof;
- 128 mẫu development, cân bằng 64 bonafide và 64 spoof;
- mẫu được chọn xác định bằng SHA-256 và trải trên nhiều người nói, nhiều shard.

Kaggle kernel version 8 đã chạy ba lần độc lập trên Tesla T4. Cả ba lần đều hoàn thành 32/32 bước với cùng dữ liệu và cấu hình.

Dự án sử dụng thời gian chậm nhất trong ba lần để tránh ước lượng quá lạc quan. Ngân sách được giới hạn ở 20 GPU-giờ cho ba pilot và một lượt dự phòng. Kết quả cấu hình chung được khóa như sau:

| Thành phần | Giá trị đã khóa |
|---|---:|
| Batch size | 8 |
| Worker | 0 |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Số optimizer step | 896 |
| Development mỗi run | 4.115 mẫu |
| Early stopping trong pilot | Không sử dụng |

Thời gian dự kiến là khoảng 4,785 giờ cho mỗi run. Tổng bốn run-equivalent được dự báo khoảng 19,139 GPU-giờ.

Ba profile `b0-pilot-none`, `b0-pilot-peak` và `b0-pilot-rms-dbfs` đã được tạo. Kiểm thử xác nhận ba profile chỉ khác amplitude policy và tên file đầu ra.

**Ý nghĩa:** ba amplitude policy sẽ nhận cùng dữ liệu và cùng lượng huấn luyện. Nhờ đó, chênh lệch development EER có thể được xem là ảnh hưởng của policy, thay vì do run này được huấn luyện nhiều hơn run khác.

### 5.12. Kiểm thử và tài liệu

Hệ thống kiểm thử hiện bao phủ dữ liệu, amplitude, chia đoạn, split, manifest, duplicate content, loader, LFCC, LCNN, metric, checkpoint, exact resume, packed artifact và Kaggle workflow.

Cổng gần nhất đạt 147/147 unit test trong 112,924 giây. `compileall`, kiểm tra môi trường CPU, kiểm tra báo cáo benchmark và `git diff --check` đều đạt.

Ngoài code, dự án đã duy trì nhật ký quyết định, trạng thái hiện tại, kế hoạch ngày/tuần và hướng dẫn sử dụng lại.

**Ý nghĩa:** kiểm thử giúp bảo vệ cả code và giao thức nghiên cứu. Ví dụ, kiểm thử có thể phát hiện ba profile pilot dùng step budget khác nhau hoặc checkpoint được nạp bằng cấu hình không tương thích.

## 6. Kết quả nổi bật và ý nghĩa

| Kết quả | Ý nghĩa đối với đồ án |
|---|---|
| Metadata và split đã được kiểm tra | Dữ liệu có cơ sở để huấn luyện và đánh giá |
| Manifest v2 không trùng nội dung train–development | Giảm nguy cơ EER thấp giả tạo do rò rỉ waveform |
| Audio được chuẩn hóa về 16 kHz và 4 giây | Giảm nguy cơ mô hình học sample rate hoặc độ dài file |
| Baseline B0 chạy xuyên suốt | Sẵn sàng thu kết quả baseline khoa học |
| Metric và threshold contract đã khóa | Test không tham gia quá trình chọn mô hình |
| Exact resume đạt trên CPU và CUDA | Có thể tiếp tục run dài bị gián đoạn |
| Kaggle CLI hoạt động | Thí nghiệm GPU có thể tái lập và kiểm chứng |
| Step budget 896 đã khóa | Ba amplitude pilot có thể được so sánh công bằng |
| 147/147 kiểm thử đạt | Code và giao thức hiện tại ổn định ở mức kiểm thử |

## 7. Khó khăn gặp phải và cách xử lý

| Khó khăn | Cách xử lý | Kết quả |
|---|---|---|
| Dữ liệu có waveform trùng dù file và speaker khác nhau | Băm nội dung waveform và tạo manifest v2 | Không còn duplicate train–development trong manifest |
| Sample rate khác nhau giữa các nhóm | Resample toàn bộ về 16 kHz | Đầu vào mô hình thống nhất |
| I/O Parquet chậm | Benchmark batch, worker, batch-read và packed artifact | Đã khóa cấu hình hiện tại và có phương án dự phòng |
| Kaggle mount dataset theo cấu trúc lồng | Sửa runner để tự tìm data root | Kernel production chạy thành công |
| Phiên GPU có thể bị ngắt | Xây dựng checkpoint và exact resume | Có thể tiếp tục đúng batch/RNG |
| Subset lấy các hàng đầu chỉ có bonafide | Chọn cân bằng bằng SHA-256 | Workload benchmark đại diện hơn |
| Chưa biết số bước pilot phù hợp | Chạy ba benchmark Kaggle | Khóa chung 896 bước |

## 8. Những giới hạn hiện tại

Đồ án chưa có kết quả để kết luận amplitude policy nào tốt nhất. Metric trong smoke hoặc technical benchmark chỉ chứng minh pipeline hoạt động; không phải kết quả khoa học của mô hình.

B0 chưa được huấn luyện đầy đủ bằng policy đã chọn. XLS-R đóng băng và XLS-R tinh chỉnh từng phần chưa được triển khai/chạy trong giai đoạn kết quả chính.

Đánh giá closed test chưa được thực hiện. Trong dữ liệu cục bộ đã phát hiện bốn nhóm waveform trùng đi qua `closed_train` và `closed_test`. Cần quyết định riêng về evaluation manifest content-disjoint trước khi đánh giá cuối.

Packed artifact mới được kiểm tra trên 24 mẫu. Chưa có bằng chứng rằng mức tăng tốc này được giữ nguyên khi mở rộng lên 20.000 mẫu.

Dữ liệu hiện có chỉ là phần của snapshot công khai. Kết quả cuối phải ghi đúng số shard và số mẫu, không so sánh trực tiếp với bài báo nếu dữ liệu và protocol không tương đương.

## 9. Công việc tiếp theo

### Giai đoạn 1 — Chạy ba pilot amplitude

1. Chạy profile `b0-pilot-none` đủ 896 bước.
2. Kiểm tra đủ 4.115 mẫu development, checkpoint và report.
3. Chạy `b0-pilot-peak` với cùng cấu hình.
4. Chạy `b0-pilot-rms-dbfs` với cùng cấu hình.

### Giai đoạn 2 — Chọn amplitude policy

Policy được chọn theo thứ tự:

1. development EER thấp nhất;
2. nếu EER bằng nhau, development loss thấp hơn;
3. nếu vẫn bằng nhau, chọn `none`.

Kết quả lựa chọn phải được ghi thành quyết định mới. Không dùng closed test để chọn policy.

### Giai đoạn 3 — Huấn luyện baseline B0 đầy đủ

Sau khi policy được khóa, B0 sẽ được huấn luyện đầy đủ. Early stopping chỉ được sử dụng ở giai đoạn này, với checkpoint tốt nhất được chọn bằng development EER.

### Giai đoạn 4 — Triển khai và chạy XLS-R

- X0: giữ nguyên phần trích xuất đặc trưng XLS-R và chỉ huấn luyện classifier.
- X1: tinh chỉnh một phần XLS-R theo cấu hình được khóa.

Hai mô hình phải dùng cùng dữ liệu, preprocessing và metric với baseline để so sánh công bằng.

### Giai đoạn 5 — Đánh giá cuối và hoàn thiện sản phẩm

- Khóa evaluation manifest cho test.
- Đánh giá B0, X0 và X1 trên test bằng threshold development.
- Phân tích lỗi theo nhãn và `utt_type`.
- Xây dựng demo xử lý file âm thanh dài.
- Hoàn thiện báo cáo, hình vẽ, bảng kết quả và slide bảo vệ.

## 10. Đánh giá chung

Đồ án đã hoàn thành phần nền móng quan trọng: dữ liệu development đã được làm sạch theo content hash, giao thức đã khóa, pipeline có kiểm thử, baseline kỹ thuật hoạt động, checkpoint có thể tiếp tục chính xác và hạ tầng Kaggle đã sẵn sàng.

Ý nghĩa lớn nhất của giai đoạn vừa qua là giảm nguy cơ thu được một kết quả đẹp nhưng sai do rò rỉ dữ liệu, shortcut hoặc cách đánh giá không công bằng. Các bước đã làm chưa tạo ra kết luận cuối về hiệu năng mô hình, nhưng đã tạo điều kiện để các kết quả tiếp theo đáng tin cậy và có thể tái lập.

Giai đoạn tiếp theo là chạy ba pilot amplitude để thu kết quả khoa học đầu tiên, sau đó hoàn thiện baseline và triển khai XLS-R. Đây là phần trực tiếp trả lời câu hỏi nghiên cứu của đề tài.

## 11. Tài liệu và báo cáo liên quan

- Trạng thái mới nhất: [current_state.md](current_state.md).
- Nhật ký quyết định: [decisions.md](decisions.md).
- Phân tích chi tiết toàn dự án: [phan_tich_tong_the_tien_do_du_an.md](phan_tich_tong_the_tien_do_du_an.md).
- Báo cáo metadata: [metadata_audit.md](../reports/metadata_audit.md).
- Báo cáo split: [split_summary.md](../reports/split_summary.md).
- Kiểm tra leakage: [leakage_check.md](../reports/leakage_check.md).
- Báo cáo manifest v2: [development_manifest_20k_v2.md](../reports/development_manifest_20k_v2.md).
- Audit nội dung manifest v2: [development_manifest_20k_v2_content_audit.md](../reports/development_manifest_20k_v2_content_audit.md).
- Báo cáo benchmark pipeline: [data_pipeline_benchmark.md](../reports/data_pipeline_benchmark.md).
- Báo cáo packed artifact: [packed_artifact_benchmark.md](../reports/packed_artifact_benchmark.md).
- Báo cáo benchmark Kaggle B0: [b0_benchmark_summary.md](../reports/b0_benchmark_summary.md).
- Kế hoạch tuần hiện tại: [2026-W43.md](plans/weekly/2026-W43.md).

