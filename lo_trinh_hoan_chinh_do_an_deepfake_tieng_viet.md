# Lộ trình hoàn thành đồ án phát hiện tiếng nói giả mạo tiếng Việt

## 1. Quyết định phạm vi

**Tên đề tài chính thức (đã chốt ngày 28/09/2026):**

> Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần

**Bài toán duy nhất:** nhận một đoạn tiếng nói tiếng Việt và phân loại thành `bonafide` (thật) hoặc `spoof` (giả mạo).

**Mô hình chính:** XLS-R 300M kết hợp bộ phân loại nhị phân.

**Mô hình cơ sở:** LFCC kết hợp LCNN.

**Thời gian thực hiện chính thức:** 14 tuần, từ 28/09/2026 đến 03/01/2027.

**Cập nhật lộ trình:** 25/09/2026.

Đề tài không xây dựng mô hình nhận diện VC, AP hay replay. Trường `utt_type` chỉ dùng để thống kê dữ liệu và phân tích lỗi của bộ phân loại thật/giả. AASIST, fine-tune toàn bộ XLS-R và các bộ dữ liệu ngoài miền đều là phần tùy chọn.

## 2. Trạng thái hiện tại

Các công việc chuẩn bị trước giai đoạn huấn luyện đã hoàn thành:

- Đề cương đã được gửi và được đánh giá hoàn thành.
- Audit metadata 220.963 mẫu và 1.141 người nói.
- Xác minh 5 shard Parquet gồm 2.558 waveform.
- Tạo tám file split theo người nói với seed `2026`.
- Chạy leakage checker, không phát hiện trùng người nói hoặc file giữa các tập học.
- Khóa giao thức âm thanh mono 16 kHz, đoạn 4 giây.
- Hoàn thiện phép tính EER và kiểm thử tự động.
- Chạy audio smoke test; không phát hiện waveform rỗng, không hữu hạn hoặc im lặng hoàn toàn.
- Toàn bộ 18 kiểm thử hiện có đã đạt.

**Bước tiếp theo:** xây dựng bộ nạp dữ liệu cho `closed_train/dev/test`, sau đó chạy mô hình cơ sở LFCC + LCNN trên tập nhỏ trước khi mở rộng dữ liệu.

## 3. Sản phẩm bắt buộc

Đồ án được xem là hoàn thành khi có đủ các sản phẩm sau:

1. Quy trình dữ liệu tái lập được từ metadata đến train, development và test.
2. Các tập dữ liệu tách biệt theo người nói và báo cáo kiểm tra rò rỉ.
3. Mô hình cơ sở LFCC + LCNN hoạt động ổn định.
4. Mô hình XLS-R đóng băng kết hợp bộ phân loại nhị phân.
5. Mô hình XLS-R fine-tune một phần làm kết quả chính.
6. Bảng so sánh EER, ROC-AUC và macro-F1 trên cùng tập kiểm thử.
7. Phân tích lỗi theo nhãn thật/giả và theo nhóm dữ liệu VC, AP, replay.
8. Chương trình minh họa nhận tệp âm thanh và trả về xác suất thật/giả.
9. Báo cáo đồ án, slide, mã nguồn, cấu hình và hướng dẫn chạy.

## 4. Phạm vi thực hiện

### 4.1. Phần bắt buộc

- Dữ liệu chính: `VSASV-HF-public-snapshot-v1`.
- Nhãn đầu ra: `0 = bonafide`, `1 = spoof`.
- VC, AP và replay được gộp thành nhãn `spoof` khi huấn luyện mô hình chính.
- Chia dữ liệu theo người nói, không chia ngẫu nhiên theo từng file.
- Chuẩn hóa mọi waveform về mono 16 kHz.
- Độ dài đầu vào ban đầu: 4 giây.
- Mô hình cơ sở: LFCC + LCNN.
- Mô hình chính: XLS-R 300M + bộ phân loại nhị phân.
- So sánh XLS-R đóng băng với XLS-R fine-tune một phần.
- Đánh giá bằng EER, ROC-AUC và macro-F1.

### 4.2. Phần tùy chọn

Chỉ thực hiện sau khi toàn bộ phần bắt buộc đã hoàn thành:

- Fine-tune toàn bộ XLS-R.
- Dùng AASIST làm mô hình đối chứng bổ sung.
- Thử tăng cường dữ liệu âm thanh đơn giản.
- Đánh giá thêm trên các split open-set đã tạo.
- Đánh giá trên một bộ dữ liệu khác nếu có quyền truy cập phù hợp.

### 4.3. Ngoài phạm vi

- Phân loại kiểu tấn công VC, AP hoặc replay.
- Nhận dạng danh tính người nói.
- Tự huấn luyện foundation model từ đầu.
- Xây dựng mô hình tạo hoặc chuyển đổi giọng nói.
- Phát hiện deepfake video.
- Cam kết phát hiện mọi loại tiếng nói giả mạo.
- Dùng kết quả mô hình như bằng chứng pháp lý.

### 4.4. Vị trí đóng góp của đồ án

Đồ án không tuyên bố đề xuất một kiến trúc học sâu mới. Đóng góp chính nằm ở ba điểm có thể kiểm chứng:

1. Áp dụng và đánh giá có hệ thống LFCC + LCNN và XLS-R cho phát hiện tiếng nói giả mạo tiếng Việt.
2. Xây dựng giao thức speaker-disjoint, kiểm tra rò rỉ và quy trình tiền xử lý thống nhất trên `VSASV-HF-public-snapshot-v1`.
3. Công bố minh bạch các vấn đề chất lượng của snapshot như chênh lệch với bài báo, thiếu provenance và nguy cơ shortcut sample rate hoặc biên độ.

So sánh đóng băng và fine-tune một phần XLS-R là nội dung thực nghiệm chính, không được mô tả như một phương pháp hoàn toàn mới.

## 5. Câu hỏi nghiên cứu

### RQ1

Mô hình XLS-R kết hợp bộ phân loại nhị phân có cải thiện EER so với mô hình cơ sở LFCC + LCNN trên cùng giao thức dữ liệu hay không?

### RQ2

Fine-tune một phần XLS-R có cải thiện kết quả so với việc đóng băng toàn bộ bộ mã hóa hay không, và chi phí tính toán tăng bao nhiêu?

### Giả thuyết

XLS-R fine-tune một phần sẽ đạt EER thấp hơn XLS-R đóng băng và LFCC + LCNN, trong khi vẫn có thể huấn luyện bằng tài nguyên GPU khả dụng cho đồ án.

## 6. Dữ liệu đã xác minh

| Thuộc tính | Giá trị |
|---|---:|
| Tổng số mẫu metadata | 220.963 |
| Số người nói | 1.141 |
| Bonafide | 98.305 |
| Voice conversion | 60.949 |
| Adversarial attack | 60.949 |
| Replay | 760 |
| Dòng trùng | 0 |
| Giá trị thiếu | 0 |

Ảnh chụp công khai không trùng hoàn toàn với dữ liệu được mô tả trong bài báo VSASV. Vì vậy, báo cáo phải ghi rõ tên `VSASV-HF-public-snapshot-v1` và không tuyên bố tái lập giao thức chính thức của bài báo.

Năm shard cục bộ có 2.558 waveform:

| Nhóm | Số mẫu | Sample rate quan sát được |
|---|---:|---:|
| Bonafide | 1.292 | 16 kHz |
| Voice conversion | 188 | 40 kHz |
| Adversarial attack | 813 | 16 kHz |
| Replay | 265 | 16 kHz |

Năm shard không phải mẫu ngẫu nhiên. Chúng chỉ được dùng để kiểm tra pipeline và chạy smoke test, không dùng để đưa ra kết luận khoa học cuối cùng. Việc chuẩn hóa toàn bộ audio về 16 kHz là bắt buộc để hạn chế mô hình học sample rate thay cho dấu hiệu giả mạo.

## 7. Giao thức dữ liệu sau khi thu hẹp đề tài

### 7.1. Giao thức chính

Sử dụng ba file sau:

- `data/splits/closed_train.csv`
- `data/splits/closed_dev.csv`
- `data/splits/closed_test.csv`

| Split | Mẫu | Người nói | Bonafide | Spoof |
|---|---:|---:|---:|---:|
| Train | 150.999 | 798 | 67.809 | 83.190 |
| Development | 39.116 | 172 | 14.490 | 24.626 |
| Test | 30.848 | 171 | 16.006 | 14.842 |

Ba tập không trùng người nói hoặc file. Train chứa các nhóm giả mạo nhưng đầu ra mô hình vẫn chỉ là `bonafide` hoặc `spoof`.

### 7.2. Vai trò của các split open-set

Năm split `open_*` đã tạo vẫn được giữ để tái lập và có thể dùng cho phân tích bổ sung. Chúng không còn là giao thức chính và không được làm tăng khối lượng bắt buộc của đồ án.

### 7.3. Ba mức dữ liệu

1. **Smoke test:** 2.558 mẫu hiện có, dùng để kiểm tra code và một vòng huấn luyện.
2. **Development subset:** 20.000-40.000 mẫu từ nhiều người nói và nhiều shard, dùng để phát triển mô hình.
3. **Final experiment:** dùng phần dữ liệu lớn nhất có thể truy cập, giữ nguyên split theo người nói và ghi chính xác số mẫu thực tế.

Không bắt buộc tải đủ 432 shard trước khi viết code. Tuy nhiên, không dùng năm shard hiện tại làm kết quả cuối vì chúng không đại diện ngẫu nhiên cho toàn bộ snapshot.

### 7.4. Mốc khóa dữ liệu cuối tuần 3

Chậm nhất ngày 18/10/2026 phải chốt phạm vi audio dùng cho thí nghiệm chính. Trước mốc này cần:

1. Đo dung lượng 5 shard hiện có và ước lượng dung lượng phần dữ liệu dự kiến tải.
2. Kiểm tra dung lượng ổ đĩa, băng thông tải và thời gian đọc dữ liệu.
3. Chọn một trong ba phương án: toàn bộ snapshot truy cập được, một tập shard đại diện, hoặc development subset 20.000-40.000 mẫu.
4. Lưu manifest cố định gồm tên shard, file, speaker, split, nhãn nhị phân và `utt_type`.
5. Ghi số mẫu thực tế của train, development và test sau khi giao với audio đã có.
6. Chạy lại kiểm tra coverage và speaker leakage trên manifest đã khóa.

Sau ngày 18/10, không thay đổi tập audio giữa các mô hình. Nếu buộc phải thay đổi, tạo phiên bản manifest mới và chạy lại toàn bộ baseline. Kết quả cuối phải ghi chính xác số shard, số mẫu và dung lượng đã sử dụng.

## 8. Tiền xử lý âm thanh

Quy trình chung cho mọi mô hình:

```text
Audio bytes
  -> giải mã waveform
  -> chuyển mono
  -> resample 16 kHz
  -> kiểm tra giá trị hữu hạn và mức năng lượng
  -> áp dụng chính sách biên độ đã khóa
  -> cắt hoặc đệm thành 4 giây
  -> đưa vào bộ trích xuất đặc trưng hoặc XLS-R
```

Quy tắc cắt và đệm:

- Train: random crop.
- Development và test: center crop.
- File ngắn: repeat-then-trim.
- Chính sách biên độ phải áp dụng giống nhau cho mọi nhãn và mọi split.
- Không dùng sample rate gốc làm đầu vào huấn luyện chính.

Tuần 3 chỉ triển khai ba cấu hình và kiểm tra phân bố peak/RMS; chưa thể chọn bằng EER vì B0 chưa được huấn luyện. Trong tuần 4, chạy ba pilot B0 ngắn trên cùng train subset, development subset, seed và số bước huấn luyện. So sánh development EER rồi khóa một chính sách trước ngày 25/10/2026 để dùng cho lần huấn luyện B0 đầy đủ, toàn bộ mô hình XLS-R, test và demo. Không dùng test để lựa chọn. Cấu hình RMS phải có ngưỡng năng lượng tối thiểu để không khuếch đại đoạn gần im lặng. Random gain augmentation chỉ là kiểm tra bổ sung và phải áp dụng độc lập với nhãn.

## 9. Các mô hình cần thực hiện

### B0 LFCC kết hợp LCNN

**Vai trò:** mô hình cơ sở bắt buộc.

**Đầu vào:** LFCC được tính từ waveform mono 16 kHz.

**Đầu ra:** một logit cho bài toán thật/giả.

**Mục tiêu:** xác minh toàn bộ pipeline huấn luyện, metric và checkpoint trước khi dùng mô hình lớn.

### X0 XLS-R đóng băng

**Vai trò:** cấu hình đầu tiên của mô hình chính.

```text
Waveform 16 kHz
  -> XLS-R 300M đóng băng
  -> mean và standard-deviation pooling
  -> MLP hoặc Linear classifier
  -> logit thật/giả
```

Chỉ tầng pooling và classifier được cập nhật. Cấu hình này có thể dùng để kiểm tra embedding và giảm chi phí GPU.

### X1 XLS-R fine-tune một phần

**Vai trò:** mô hình chính của đồ án.

- Giữ đóng băng bộ trích xuất đặc trưng đầu vào.
- Mở một số tầng Transformer cuối.
- Huấn luyện với learning rate nhỏ hơn tầng phân loại.
- Chọn epoch và cấu hình trên development, không dùng test để điều chỉnh.

Số tầng được mở sẽ được chốt sau pilot; điểm khởi đầu là bốn tầng cuối.

### X2 XLS-R fine-tune toàn bộ

**Vai trò:** tùy chọn.

Chỉ chạy khi B0, X0 và X1 đã hoàn chỉnh, dữ liệu đủ lớn và GPU đáp ứng. Không hoàn thành X2 không làm đồ án thiếu nội dung bắt buộc.

### B1 AASIST

**Vai trò:** đối chứng tùy chọn.

AASIST chỉ được thực hiện nếu giảng viên yêu cầu thêm mô hình so sánh hoặc tiến độ còn dư. Nó không phải một nhánh nghiên cứu riêng.

## 10. Huấn luyện và đánh giá

### 10.1. Thiết lập chung

- Cố định seed cho mỗi run.
- Dùng cùng split và tiền xử lý cho mọi mô hình.
- Cân bằng bonafide và spoof bằng sampler hoặc trọng số loss nếu cần.
- Lưu cấu hình, checkpoint tốt nhất, lịch sử loss, thời gian chạy và peak VRAM.
- Chọn mô hình tốt nhất theo EER trên development.
- Chỉ đánh giá test sau khi đã khóa cấu hình.

### 10.2. Chỉ số

- **Chính:** EER.
- **Bổ sung:** ROC-AUC và macro-F1.
- Accuracy chỉ dùng tham khảo, không dùng làm kết luận chính.

Quy ước hiện tại:

```text
0 = bonafide
1 = spoof
score cao hơn = bằng chứng spoof mạnh hơn
```

### 10.3. Bảng so sánh bắt buộc

| Mô hình | Trạng thái encoder | EER | ROC-AUC | Macro-F1 | Thời gian train | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| LFCC + LCNN | Không áp dụng | | | | | |
| XLS-R + classifier | Đóng băng | | | | | |
| XLS-R + classifier | Fine-tune một phần | | | | | |

Kết quả theo VC, AP và replay được đặt ở bảng phân tích lỗi. Không diễn giải các bảng đó như kết quả của mô hình nhận diện kiểu tấn công.

### 10.4. Kiểm tra shortcut bắt buộc

Do mẫu VC cục bộ có sample rate khác các nhóm còn lại, cần chạy một phép kiểm tra band-limit hoặc so sánh trước và sau chuẩn hóa 16 kHz. Mục đích là xác minh mô hình không dựa chủ yếu vào băng thông. Đây là kiểm tra tính hợp lệ của dữ liệu, không phải phương pháp nghiên cứu thứ hai.

Ngoài sample rate, cần so sánh phân bố peak và RMS giữa bonafide và spoof. Nếu hai lớp có chênh lệch rõ, báo cáo kết quả của chính sách biên độ đã chọn và một ablation tối thiểu để xác định mức độ mô hình phụ thuộc vào gain hoặc loudness.

### 10.5. Chiến lược suy luận cho demo

Demo nhận WAV hoặc MP3 có độ dài bất kỳ và dùng đúng tiền xử lý đã khóa khi huấn luyện.

- File ngắn hơn hoặc bằng 4 giây: dùng cùng quy tắc repeat-then-trim hoặc center crop như evaluation.
- File dài hơn 4 giây: chạy cửa sổ 4 giây với bước nhảy 2 giây; luôn thêm cửa sổ cuối kết thúc tại cuối tệp.
- Tính xác suất spoof cho từng cửa sổ.
- Xác suất toàn tệp là trung bình xác suất các cửa sổ.
- Hiển thị thêm cửa sổ có xác suất cao nhất và timeline để hỗ trợ quan sát; không dùng giá trị cực đại làm kết luận chính.
- Threshold thật/giả lấy từ development và không điều chỉnh bằng tệp người dùng.

Chiến lược này phải được cài cùng module inference trước tuần 13 để demo không dùng một đường tiền xử lý khác với thí nghiệm.

## 11. Lộ trình 14 tuần

| Tuần | Thời gian | Công việc chính | Đầu ra và cổng hoàn thành |
|---:|---|---|---|
| 1 | 28/09-04/10 | Xác nhận tên đề tài, phạm vi nhị phân và mô hình chính; khóa giao thức closed-set | **Đã hoàn thành:** đề cương được đánh giá hoàn thành; tài liệu dự án đã cập nhật thống nhất |
| 2 | 05/10-11/10 | Hoàn thiện tập dữ liệu phát triển; đo dung lượng và băng thông; lập danh sách shard dự kiến | Development subset có thống kê; có ước lượng dung lượng và phương án tải audio |
| 3 | 12/10-18/10 | Hoàn thiện dataset loader, khóa manifest audio; triển khai none/peak/RMS và thống kê biên độ | Chốt số shard, số mẫu và dung lượng; ba chính sách chạy đúng; chưa chọn bằng EER |
| 4 | 19/10-25/10 | Cài LFCC + LCNN; chạy ba pilot B0 có cùng ngân sách để chọn chính sách biên độ | Khóa chính sách biên độ bằng development EER; loss giảm và checkpoint lưu được |
| 5 | 26/10-01/11 | Huấn luyện, đánh giá và sửa lỗi mô hình cơ sở | Có EER, ROC-AUC và macro-F1 của LFCC + LCNN |
| 6 | 02/11-08/11 | Cài XLS-R đóng băng, pooling và classifier | Trích xuất embedding và chạy một epoch thành công |
| 7 | 09/11-15/11 | Huấn luyện XLS-R đóng băng trên development subset | Có checkpoint và bảng so sánh ban đầu với LCNN |
| 8 | 16/11-22/11 | Mở các tầng cuối của XLS-R; chạy pilot fine-tune một phần | Chọn được batch size, learning rate và số tầng mở ban đầu |
| 9 | 23/11-29/11 | Fine-tune XLS-R trên tập phát triển lớn hơn | Có run ổn định, log và checkpoint tốt nhất |
| 10 | 30/11-06/12 | Chốt cấu hình chính và chạy lại với seed cố định | X1 được khóa trước khi đánh giá test |
| 11 | 07/12-13/12 | Đánh giá ba mô hình trên closed test; đo thời gian và VRAM | Bảng kết quả chính hoàn chỉnh |
| 12 | 14/12-20/12 | Phân tích lỗi, band-limit và loudness; chạy X2 hoặc AASIST nếu còn thời gian | Bảng phân tích theo nhóm và phần giới hạn nghiên cứu |
| 13 | 21/12-27/12 | Hoàn thiện sliding-window inference và demo; viết kết quả và thảo luận | Demo xử lý được tệp dài bất kỳ và bản thảo báo cáo đầy đủ |
| 14 | 28/12-03/01 | Hoàn thiện báo cáo, README, mã nguồn và slide; chạy kiểm tra tái lập | Bộ bàn giao sẵn sàng nộp và bảo vệ |

## 12. Cổng kiểm soát tiến độ

### G0 Dữ liệu sẵn sàng

- Metadata audit đạt.
- Split không rò rỉ.
- Audio smoke test đạt.
- Metric EER có kiểm thử.

**Trạng thái:** đã hoàn thành.

### G1 Dữ liệu thí nghiệm được khóa

- Chốt danh sách shard và số mẫu thực tế trước ngày 18/10/2026.
- Manifest khớp audio truy cập được và giữ speaker-disjoint.
- Có thống kê dung lượng, nhãn và người nói cho từng split.

### G2 Pipeline huấn luyện hoạt động

- Dataset loader tạo batch ổn định.
- Một batch chạy qua mô hình và loss hữu hạn.
- Có thể lưu và nạp checkpoint.

### G3 Mô hình cơ sở hoàn thành

- Ba pilot B0 cho `none`, peak và RMS dùng cùng dữ liệu, seed và ngân sách huấn luyện.
- Chính sách biên độ được khóa trước ngày 25/10/2026 bằng development EER.
- LFCC + LCNN chạy hết train/dev.
- Có EER, ROC-AUC và macro-F1.
- Có file cấu hình và log.

### G4 XLS-R đóng băng hoàn thành

- Feature encoder không cập nhật trọng số.
- Classifier hội tụ trên development.
- Kết quả được so sánh công bằng với B0.

### G5 XLS-R fine-tune một phần hoàn thành

- Cấu hình số tầng mở được ghi rõ.
- Có checkpoint và log tài nguyên.
- Không dùng test để chọn cấu hình.

### G6 Đánh giá cuối hoàn thành

- Bảng kết quả chính đủ ba cấu hình.
- Có phân tích lỗi và kiểm tra shortcut.
- Kết luận không vượt quá phạm vi snapshot công khai.

### G7 Bàn giao hoàn thành

- Demo chạy được với một tệp âm thanh.
- README đủ hướng dẫn.
- Báo cáo và slide thống nhất số liệu.

## 13. Thứ tự cắt giảm khi chậm tiến độ

Nếu thiếu thời gian hoặc GPU, bỏ theo thứ tự sau:

1. Bỏ AASIST.
2. Bỏ fine-tune toàn bộ XLS-R.
3. Bỏ đánh giá dataset ngoài miền và các split open-set.
4. Bỏ augmentation nâng cao.
5. Giảm số run hoặc kích thước dữ liệu, nhưng vẫn giữ speaker-disjoint.

Không được bỏ:

- Kiểm tra dữ liệu và rò rỉ.
- Chuẩn hóa audio 16 kHz.
- LFCC + LCNN.
- XLS-R đóng băng và fine-tune một phần.
- EER, ROC-AUC và macro-F1.
- Báo cáo giới hạn của snapshot.

## 14. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Không tải đủ VSASV | Dùng subset có kiểm soát, ghi đúng số shard và số mẫu; không tuyên bố dùng toàn bộ VSASV |
| Năm shard hiện tại không đại diện | Mở rộng mẫu từ nhiều shard và người nói trước khi chạy kết quả cuối |
| GPU hết bộ nhớ | Đóng băng XLS-R, dùng mixed precision, batch nhỏ và gradient accumulation |
| Fine-tune bị quá khớp | Giảm số tầng mở, learning rate và số epoch; chọn checkpoint theo development EER |
| Kết quả cao bất thường | Kiểm tra rò rỉ, sample rate, duration, codec và cân bằng nhãn |
| Replay quá ít | Chỉ báo cáo như nhóm phân tích; không dùng accuracy tổng để kết luận |
| Tác giả không cung cấp bản đầy đủ | Tiếp tục với snapshot đã version hóa và nêu rõ giới hạn |
| Tiến độ chậm | Giữ B0, X0 và X1; loại bỏ X2, B1 và các thí nghiệm mở rộng |

## 15. Cấu trúc báo cáo dự kiến

1. **Giới thiệu:** bài toán, mục tiêu, phạm vi và đóng góp.
2. **Cơ sở lý thuyết:** tiếng nói giả mạo, LFCC, LCNN, mô hình tiền huấn luyện và XLS-R.
3. **Công trình liên quan:** các hướng phát hiện tiếng nói giả mạo và fine-tune mô hình tiếng nói.
4. **Dữ liệu và tiền xử lý:** VSASV snapshot, kiểm kê, split theo người nói và chuẩn hóa 16 kHz.
5. **Phương pháp:** LFCC + LCNN, XLS-R đóng băng và XLS-R fine-tune một phần.
6. **Thực nghiệm:** cấu hình huấn luyện, metric và phần cứng.
7. **Kết quả và thảo luận:** bảng chính, phân tích lỗi, shortcut và giới hạn.
8. **Chương trình minh họa:** luồng suy luận và hướng dẫn sử dụng.
9. **Kết luận:** trả lời hai câu hỏi nghiên cứu và nêu hướng phát triển.

## 16. Công việc ưu tiên từ 25/09 đến 04/10/2026

1. Tên đề tài đã chốt; tiếp tục xác nhận phạm vi nhị phân và ba cấu hình B0/X0/X1 với giảng viên.
2. Chuyển cấu hình thí nghiệm chính sang `closed_train/dev/test`.
3. Đo dung lượng năm shard và kiểm tra dung lượng ổ đĩa để chuẩn bị quyết định tải dữ liệu.
4. Xây dựng dataset loader đọc audio Parquet và trả waveform mono 16 kHz dài 4 giây.
5. Cài đặt lựa chọn chính sách biên độ thống nhất: none, peak hoặc RMS.
6. Tạo cấu hình `lfcc_lcnn` và một smoke subset cân bằng thật/giả.
7. Cài LFCC, tạo LCNN nhỏ, chạy forward/backward và lưu checkpoint thử nghiệm.
8. Ghi lại thời gian chạy, RAM/VRAM và lỗi phát sinh để chuẩn bị mở rộng dữ liệu.

## 17. Definition of Done

```text
Dữ liệu được kiểm tra và chia theo người nói
        +
LFCC + LCNN có kết quả tái lập
        +
XLS-R đóng băng có kết quả tái lập
        +
XLS-R fine-tune một phần có kết quả chính
        +
EER, ROC-AUC, macro-F1 và phân tích lỗi
        +
Demo, báo cáo, slide và hướng dẫn chạy
        =
Đồ án hoàn thành
```

Fine-tune toàn bộ XLS-R, AASIST, external test và open-set analysis là phần nâng cao. Việc không thực hiện các mục này không ảnh hưởng đến mức hoàn thành cốt lõi nếu B0, X0 và X1 được đánh giá đúng giao thức.
