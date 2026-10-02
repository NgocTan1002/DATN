# Kiến thức cần nắm vững để hiểu sâu và bảo vệ đồ án

## 1. Mục đích của tài liệu

Tài liệu này là bản đồ kiến thức phục vụ ba mục tiêu:

1. Hiểu đúng bản chất khoa học và kỹ thuật của đồ án.
2. Có thể giải thích mạch lạc từ dữ liệu, tiền xử lý, mô hình đến đánh giá.
3. Có thể trả lời câu hỏi phản biện khi thuyết trình và bảo vệ.

Tên đề tài hiện hành:

> **Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần**

Phạm vi hiện hành là phân loại nhị phân `bonafide`/`spoof` với ba cấu hình bắt buộc:

- **B0:** LFCC + LCNN.
- **X0:** XLS-R đóng băng + bộ phân loại.
- **X1:** XLS-R fine-tune một phần + bộ phân loại.

Các hướng cũ như channel consistency, nhận diện loại tấn công, AASIST, fine-tune toàn bộ XLS-R và open-set evaluation không phải phần bắt buộc. Không dùng tài liệu lịch sử để mở rộng lại phạm vi đã chốt.

## 2. Tiêu chuẩn “nắm vững”

Với mỗi nội dung, cần đạt đủ bốn mức:

- **Mức 1 — Định nghĩa:** nói được khái niệm là gì.
- **Mức 2 — Bản chất:** giải thích được vì sao nó hoạt động.
- **Mức 3 — Liên hệ đồ án:** chỉ ra nó xuất hiện ở đâu trong pipeline, mã nguồn hoặc báo cáo.
- **Mức 4 — Phản biện:** giải thích được vì sao chọn phương án này, giới hạn là gì và phương án thay thế có trade-off gì.

Không nên chỉ học thuộc tên mô hình hoặc thông số. Khi bảo vệ, mỗi quyết định cần trả lời được chuỗi câu hỏi:

```text
Vấn đề là gì?
→ Nếu không xử lý thì sai ở đâu?
→ Đồ án xử lý bằng cách nào?
→ Bằng chứng nào chứng minh cách xử lý đang hoạt động?
→ Giới hạn còn lại là gì?
```

## 3. Câu chuyện trung tâm của đồ án

### 3.1. Phải trình bày được trong một câu

Đồ án xây dựng và đánh giá một hệ thống phân loại tiếng nói tiếng Việt thành thật hoặc giả mạo, đồng thời so sánh mô hình cơ sở LFCC + LCNN với XLS-R đóng băng và XLS-R fine-tune một phần trên cùng một giao thức dữ liệu tách biệt theo người nói.

### 3.2. Phải trình bày được trong 60 giây

Deepfake tiếng nói có thể được tạo bằng voice conversion, tấn công đối kháng hoặc phát lại. Đồ án dùng VSASV làm dữ liệu chính, ánh xạ mọi loại giả mạo về nhãn `spoof` và giữ `utt_type` để phân tích lỗi. Dữ liệu được chia theo speaker để tránh mô hình ghi nhớ danh tính người nói. Tất cả waveform được đưa về mono 16 kHz, dài 4 giây và dùng cùng giao thức biên độ. LFCC + LCNN tạo mốc cơ sở; XLS-R đóng băng kiểm tra giá trị của biểu diễn tiền huấn luyện; XLS-R fine-tune một phần kiểm tra liệu thích nghi các tầng cuối có cải thiện EER với chi phí tính toán chấp nhận được hay không. EER là metric chính, còn ROC-AUC và macro-F1 là metric bổ sung. Mọi lựa chọn mô hình và threshold được thực hiện trên development; test chỉ dùng sau khi cấu hình đã khóa.

### 3.3. Câu hỏi nghiên cứu phải thuộc lòng

**RQ1:** XLS-R kết hợp bộ phân loại nhị phân có cải thiện EER so với LFCC + LCNN trên cùng giao thức dữ liệu hay không?

**RQ2:** Fine-tune một phần XLS-R có cải thiện kết quả so với đóng băng toàn bộ encoder hay không, và chi phí tính toán tăng bao nhiêu?

**Giả thuyết:** X1 có EER thấp hơn X0 và B0 vì các tầng cuối của XLS-R được thích nghi với dấu vết giả mạo, trong khi số tầng được mở có giới hạn để giảm chi phí và nguy cơ overfit.

### 3.4. Đóng góp phải mô tả đúng mức

Đồ án không tuyên bố tạo kiến trúc học sâu mới. Đóng góp nằm ở:

1. Áp dụng và so sánh có hệ thống LFCC + LCNN, XLS-R đóng băng và XLS-R fine-tune một phần cho tiếng Việt.
2. Xây dựng pipeline dữ liệu tái lập, speaker-disjoint, có kiểm tra rò rỉ và tiền xử lý thống nhất.
3. Phân tích minh bạch các shortcut và hạn chế của snapshot VSASV như sample rate, biên độ, thiếu provenance và mất cân bằng replay.

## 4. Bài toán phát hiện tiếng nói giả mạo

### 4.1. Các khái niệm phải phân biệt

- **Bonafide:** tiếng nói thật, được thu từ người nói thật trong điều kiện dữ liệu cho phép.
- **Spoof:** tiếng nói giả mạo hoặc tín hiệu đã bị thao túng để đánh lừa hệ thống.
- **Voice conversion — VC:** biến đổi giọng nguồn thành đặc trưng giọng của người khác trong khi giữ nội dung tương đối giống.
- **Adversarial attack — AP:** tín hiệu bị tối ưu hoặc chỉnh sửa nhằm làm sai hành vi của mô hình/hệ thống.
- **Replay:** phát lại một bản ghi qua loa rồi thu lại; chứa dấu vết thiết bị và môi trường.

Đầu ra của đồ án chỉ có:

```text
0 = bonafide
1 = spoof
```

VC, AP và replay không phải ba lớp đầu ra. Chúng chỉ là nhóm con của `spoof` dùng để thống kê và phân tích lỗi.

### 4.2. Mô hình thực sự học gì?

Mô hình không “hiểu thật giả” theo nghĩa con người. Nó tìm tương quan giữa nhãn và dấu vết trong tín hiệu, ví dụ:

- phổ tần và chuyển tiếp âm vị bất thường;
- dấu vết của vocoder hoặc thuật toán chuyển đổi giọng;
- cấu trúc năng lượng, pha hoặc nhiễu nền;
- mất tự nhiên ở nhịp thở, khoảng dừng hoặc phát âm;
- dấu vết thiết bị và phòng trong replay.

Mô hình cũng có thể học nhầm:

- sample rate hoặc băng thông;
- độ lớn âm thanh;
- độ dài file;
- codec hoặc định dạng;
- danh tính speaker;
- shard hoặc corpus nguồn.

Hiểu được nguy cơ học shortcut là chìa khóa để giải thích vì sao phần dữ liệu và giao thức quan trọng ngang với kiến trúc mô hình.

## 5. Dữ liệu VSASV và phạm vi tuyên bố

### 5.1. Các con số cần nhớ

| Thuộc tính | Giá trị |
|---|---:|
| Tổng metadata | 220.963 mẫu |
| Speaker | 1.141 |
| Bonafide | 98.305 |
| Voice conversion | 60.949 |
| Adversarial attack | 60.949 |
| Replay | 760 |

Giao thức closed chính:

| Split | Mẫu | Speaker | Bonafide | Spoof |
|---|---:|---:|---:|---:|
| Train | 150.999 | 798 | 67.809 | 83.190 |
| Development | 39.116 | 172 | 14.490 | 24.626 |
| Test | 30.848 | 171 | 16.006 | 14.842 |

### 5.2. Snapshot và provenance

Dữ liệu đang dùng phải được gọi đúng là `VSASV-HF-public-snapshot-v1`. Snapshot công khai không trùng hoàn toàn với mô tả trong bài báo gốc và không có đầy đủ provenance như `generator_id` hoặc `source_id`.

Hệ quả khi báo cáo:

- được nói “đánh giá trên snapshot VSASV công khai đang sử dụng”;
- không được nói “tái lập chính xác giao thức chính thức của bài báo VSASV” nếu chưa có bằng chứng;
- nếu chỉ dùng VSASV, có thể nói “unseen spoofing attacks” trong phân tích bổ sung;
- không được nói “unseen TTS engines” vì metadata không chứng minh được engine TTS.

### 5.3. Mất cân bằng dữ liệu

Replay chỉ có 760 mẫu, nhỏ hơn rất nhiều so với VC và AP. Accuracy tổng có thể vẫn cao dù mô hình thất bại trên replay. Vì vậy cần:

- báo cáo metric tổng;
- phân tích riêng theo `utt_type`;
- xem confusion matrix hoặc EER theo nhóm khi đủ dữ liệu;
- không để nhóm lớn che khuất nhóm nhỏ.

## 6. Chia dữ liệu và chống rò rỉ

### 6.1. Vì sao phải chia theo speaker?

Nếu một speaker xuất hiện ở cả train và test, mô hình có thể ghi nhớ giọng, thiết bị hoặc điều kiện thu của speaker thay vì học dấu vết giả mạo. Khi đó kết quả test cao nhưng không phản ánh khả năng làm việc với người nói mới.

Điều kiện bắt buộc:

```text
speaker(train) ∩ speaker(dev)  = ∅
speaker(train) ∩ speaker(test) = ∅
speaker(dev)   ∩ speaker(test) = ∅
```

### 6.2. Vì sao không chia ngẫu nhiên theo file?

Chia theo file dễ đưa nhiều câu nói của cùng speaker vào các partition khác nhau. Đây là leakage ở cấp danh tính. File khác tên không có nghĩa là độc lập về speaker hoặc nội dung.

### 6.3. Vai trò của train, development và test

- **Train:** cập nhật trọng số mô hình.
- **Development:** chọn epoch, hyperparameter, amplitude policy, threshold và checkpoint tốt nhất.
- **Test:** chỉ đánh giá cuối sau khi mọi quyết định đã khóa.

Nếu xem test nhiều lần rồi điều chỉnh mô hình, test đã trở thành development và kết quả bị lạc quan.

### 6.4. Cách split được tạo

- Seed: `2026`.
- Tỷ lệ speaker theo từng stratum: 70/15/15.
- Speaker được xếp ổn định bằng SHA-256 của seed, stratum và speaker ID.
- Làm tròn bằng largest remainder.
- Ba stratum: `real_only`, `vc_adversarial`, `replay`.

Phân tầng theo stratum giúp nhóm hiếm không biến mất khỏi một partition.

### 6.5. Các loại leakage cần nhận biết

- **Speaker leakage:** cùng speaker ở train và test.
- **File leakage:** cùng đường dẫn xuất hiện nhiều partition.
- **Content leakage:** hai file khác tên nhưng cùng nội dung âm thanh.
- **Near-duplicate leakage:** file được nén/resample từ cùng nguồn xuất hiện ở hai partition.
- **Preprocessing leakage:** tính normalization hoặc statistics bằng cả test.
- **Model-selection leakage:** dùng test để chọn hyperparameter hoặc threshold.

Hiện dự án đã kiểm tra metadata, file và speaker. Kiểm tra hash nội dung cần được chạy đầy đủ khi phạm vi audio cuối được khóa.

## 7. Nền tảng xử lý tín hiệu âm thanh

### 7.1. Waveform, sample rate và Nyquist

Waveform là chuỗi giá trị biên độ theo thời gian. Sample rate (f_s) là số mẫu mỗi giây.

- 16 kHz nghĩa là 16.000 mẫu/giây.
- Đoạn 4 giây có (16.000 \times 4 = 64.000) mẫu.
- Theo định lý Nyquist, sample rate 16 kHz biểu diễn được tần số đến khoảng 8 kHz.

Cần hiểu rằng resample 40 kHz xuống 16 kHz không chỉ đổi số lượng mẫu; nó phải dùng low-pass/band-limited filtering để tránh aliasing.

### 7.2. Vì sao phải resample toàn bộ về 16 kHz?

Trong năm shard cục bộ, VC ở 40 kHz trong khi các nhóm khác chủ yếu ở 16 kHz. Nếu đưa sample rate gốc vào mô hình, mô hình có thể nhận ra VC từ băng thông thay vì dấu vết deepfake.

Chuẩn hóa 16 kHz:

- loại bỏ shortcut trực tiếp từ sample rate;
- phù hợp đầu vào XLS-R;
- giúp LFCC của các mẫu có cùng miền tần số;
- bảo đảm so sánh mô hình công bằng.

### 7.3. Peak, RMS và dBFS

Với waveform (x_1, x_2, ..., x_N):

```text
Peak = max(|x_i|)
RMS  = sqrt((1/N) × Σ x_i²)
RMS_dBFS = 20 × log10(max(RMS, ε))
```

- Peak đo giá trị tức thời lớn nhất.
- RMS gần với năng lượng/mức lớn trung bình.
- dBFS dùng mốc 0 dBFS là biên độ số cực đại; phần lớn tín hiệu có giá trị âm.
- dB là thang logarit; tăng 6 dB xấp xỉ gấp đôi biên độ.

### 7.4. Ba amplitude policy

**Peak normalization:**

```text
gain = 0,95 / input_peak
```

**RMS normalization:**

```text
desired_gain = 10^((-25 - input_rms_dbfs) / 20)
peak_safe_gain = 0,95 / input_peak
applied_gain = min(desired_gain, peak_safe_gain)
```

Waveform dưới -50 dBFS không được khuếch đại để tránh biến nhiễu rất nhỏ thành tín hiệu lớn.

### 7.5. Chia đoạn 4 giây

- Train: random crop để tăng đa dạng vị trí và giảm ghi nhớ một đoạn cố định.
- Development/test: center crop để kết quả xác định và so sánh được.
- File ngắn: repeat-then-trim để đạt đúng 64.000 mẫu mà không thêm khoảng im lặng dài.

Cần giải thích được trade-off: repeat có thể tạo biên nối nhân tạo, nhưng đơn giản, xác định và tránh attention mask ở baseline. Các phương án padding/masking hoặc multi-crop có thể là ablation sau.

## 8. LFCC và biểu diễn phổ

### 8.1. Từ waveform đến phổ

Tín hiệu được chia thành các cửa sổ ngắn. Mỗi cửa sổ đi qua FFT để chuyển từ miền thời gian sang miền tần số. Chuỗi phổ theo thời gian tạo thành spectrogram.

Các khái niệm cần hiểu:

- **Window length 400 mẫu:** 25 ms ở 16 kHz.
- **Hop length 160 mẫu:** 10 ms.
- **FFT 512:** số điểm FFT, quyết định lưới tần số của phổ.
- **Window Hann:** giảm spectral leakage ở biên cửa sổ.

### 8.2. LFCC là gì?

LFCC là Linear-Frequency Cepstral Coefficients:

```text
Waveform
→ STFT/power spectrum
→ linear-frequency filter bank
→ log năng lượng
→ DCT
→ hệ số LFCC
```

DCT làm giảm tương quan giữa các năng lượng filter và nén thông tin thành các hệ số cepstral.

### 8.3. Vì sao dùng LFCC thay vì chỉ waveform hoặc MFCC?

- LFCC cung cấp đặc trưng gọn và dễ huấn luyện cho baseline nhỏ.
- Filter bank tuyến tính giữ độ phân giải tương đối đều ở miền tần số cao.
- Dấu vết giả mạo có thể xuất hiện ở tần số cao; mel scale của MFCC nén vùng này mạnh hơn.
- LFCC là baseline phổ biến và dễ so sánh trong anti-spoofing.

Không được khẳng định LFCC chắc chắn tốt hơn MFCC nếu chưa có thực nghiệm. Đây là lựa chọn baseline hợp lý, còn hiệu quả phải được đo.

### 8.4. Cấu hình hiện tại phải nhớ

```text
Input waveform:       (batch, 64000)
LFCC output:          (batch, 60, 401)
LCNN input:           (batch, 1, 60, 401)
```

- 60 hệ số LFCC.
- 80 linear filters.
- FFT 512, window 400, hop 160.
- 401 frame do cấu hình `center=true` trên đoạn 64.000 mẫu.

## 9. LCNN và mô hình cơ sở B0

### 9.1. CNN học gì trên LFCC?

LCNN xem LFCC như một ảnh hai chiều:

- trục dọc: hệ số LFCC/tần số;
- trục ngang: thời gian;
- convolution học pattern cục bộ theo thời gian và tần số.

Các tầng sâu hơn kết hợp pattern đơn giản thành dấu vết phức tạp hơn.

### 9.2. LCNN là gì?

LCNN là Light Convolutional Neural Network. Đặc điểm thường gặp là Max-Feature-Map — MFM:

```text
MFM(a, b) = max(a, b)
```

Các channel được ghép cặp và lấy giá trị lớn hơn. MFM vừa tạo phi tuyến, vừa giảm số channel và có thể giúp giữ đặc trưng mạnh hơn. Cần phân biệt MFM với max pooling:

- MFM cạnh tranh giữa channel tại cùng vị trí.
- Max pooling thường giảm kích thước không gian/thời gian.

### 9.3. Vai trò của B0

B0 không chỉ là mô hình yếu để so điểm. Nó có ba vai trò:

1. Chứng minh pipeline dữ liệu, loss, metric và checkpoint chạy xuyên suốt.
2. Cung cấp mốc để biết XLS-R có thực sự tạo giá trị thêm hay không.
3. Chạy nhanh để thử amplitude policy và phát hiện lỗi thí nghiệm.

Nếu X1 chỉ tốt hơn B0 rất ít nhưng tốn tài nguyên lớn, cần thảo luận trade-off thay vì chỉ nói mô hình lớn tốt hơn.

## 10. XLS-R, đóng băng và fine-tune một phần

### 10.1. XLS-R là gì?

XLS-R là mô hình biểu diễn tiếng nói tự giám sát đa ngôn ngữ thuộc họ wav2vec 2.0. Ở mức khái niệm:

```text
Raw waveform
→ convolutional feature encoder
→ chuỗi latent representation
→ Transformer context network
→ embedding theo thời gian
```

Mô hình tiền huấn luyện trên lượng lớn tiếng nói không gán nhãn, nhờ đó đã học cấu trúc âm thanh và ngữ âm có thể chuyển sang tác vụ phát hiện giả mạo.

### 10.2. X0 — đóng băng XLS-R

- Không cập nhật trọng số encoder.
- Chỉ huấn luyện pooling và classifier.
- Nhanh hơn, ít VRAM hơn, giảm overfit.
- Hạn chế: embedding tiền huấn luyện không được thích nghi trực tiếp với dấu vết spoof.

Luồng khái niệm:

```text
Waveform
→ frozen XLS-R
→ embedding theo thời gian
→ mean/std pooling
→ classifier
→ spoof logit
```

### 10.3. X1 — fine-tune một phần

- Giữ đóng băng feature encoder đầu vào.
- Mở một số tầng Transformer cuối; bốn tầng cuối là điểm bắt đầu, chưa phải kết luận cố định.
- Dùng learning rate nhỏ cho encoder và có thể lớn hơn cho classifier.
- Cho phép đặc trưng tầng cao thích nghi với anti-spoofing.

Trade-off:

- có thể giảm EER;
- tốn VRAM và thời gian hơn;
- dễ overfit hơn nếu dữ liệu nhỏ;
- nhạy với learning rate, số tầng mở và regularization.

### 10.4. Phải giải thích được so sánh B0/X0/X1

| Cấu hình | Câu hỏi mà nó trả lời |
|---|---|
| B0 | Đặc trưng thủ công + mạng nhỏ đạt mức nào? |
| X0 | Biểu diễn XLS-R tiền huấn luyện có ích khi không cập nhật encoder không? |
| X1 | Thích nghi một phần encoder có cải thiện đủ để biện minh chi phí tăng không? |

## 11. Huấn luyện mô hình

### 11.1. Logit, sigmoid và xác suất

Mô hình trả một logit (z), chưa phải xác suất:

```text
p(spoof) = sigmoid(z) = 1 / (1 + exp(-z))
```

- Logit dương lớn → bằng chứng spoof mạnh.
- Logit âm lớn về trị tuyệt đối → bằng chứng bonafide mạnh.
- Threshold không bắt buộc bằng 0,5; threshold vận hành phải chọn trên development.

### 11.2. Binary cross-entropy

Với nhãn (y \in \{0,1\}):

```text
L = -[y × log(p) + (1-y) × log(1-p)]
```

`BCEWithLogitsLoss` kết hợp sigmoid và BCE theo cách ổn định số học hơn việc tự gọi sigmoid rồi tính log.

### 11.3. Một bước tối ưu gồm gì?

1. Đọc batch.
2. Forward qua feature extractor và classifier.
3. Tính loss.
4. Xóa gradient cũ.
5. Backward để tính gradient.
6. Optimizer cập nhật tham số.
7. Kiểm tra loss, logit và gradient hữu hạn.

### 11.4. Epoch, batch và learning rate

- **Batch:** nhóm mẫu xử lý trong một lần forward/backward.
- **Epoch:** một lượt qua tập train.
- **Learning rate:** độ lớn bước cập nhật trọng số.
- **Optimizer:** quy tắc biến gradient thành cập nhật; Adam là điểm khởi đầu cho smoke test.

Cần hiểu batch size ảnh hưởng VRAM, nhiễu gradient và thời gian; learning rate quá lớn gây mất ổn định, quá nhỏ làm học chậm.

### 11.5. Overfitting và early stopping

Dấu hiệu overfit:

- train loss tiếp tục giảm nhưng development EER tăng;
- kết quả tốt trên nhóm phổ biến nhưng kém ở replay;
- mô hình nhạy với codec, gain hoặc speaker.

Checkpoint tốt nhất phải chọn bằng development EER. Test không tham gia early stopping.

## 12. Metric đánh giá

### 12.1. Confusion matrix theo quy ước của đồ án

Với `1 = spoof`, score cao hơn nghĩa là spoof:

- **TP:** spoof được dự đoán spoof.
- **TN:** bonafide được dự đoán bonafide.
- **FP:** bonafide bị báo nhầm là spoof.
- **FN:** spoof bị bỏ sót và dự đoán bonafide.

Trong trình bày nên dùng FP/FN hoặc mô tả bằng lời để tránh nhầm tên FAR/FRR giữa các tài liệu sinh trắc học.

### 12.2. EER

Với threshold (t):

```text
FPR(t) = số bonafide có score ≥ t / tổng bonafide
FNR(t) = số spoof có score < t / tổng spoof
```

EER là điểm vận hành mà FPR và FNR bằng nhau hoặc gần nhau sau nội suy. EER càng thấp càng tốt.

Phải hiểu:

- EER đánh giá khả năng tách hai phân bố score mà không khóa trước một threshold cố định.
- EER không phải accuracy.
- Threshold tại EER dùng để mô tả điểm cân bằng lỗi, không nhất thiết là threshold tốt nhất cho mọi ứng dụng.
- Nếu chi phí bỏ sót spoof khác chi phí báo nhầm bonafide, hệ thống thực tế có thể chọn threshold khác.

### 12.3. ROC-AUC

ROC vẽ TPR theo FPR khi thay đổi threshold. AUC đo xác suất một mẫu spoof ngẫu nhiên có score cao hơn một mẫu bonafide ngẫu nhiên.

- AUC gần 1: phân tách tốt.
- AUC gần 0,5: gần ngẫu nhiên.
- AUC không cho biết trực tiếp threshold triển khai.

### 12.4. Macro-F1

F1 kết hợp precision và recall. Macro-F1 tính F1 riêng cho từng lớp rồi trung bình, vì vậy mỗi lớp có trọng số ngang nhau hơn so với accuracy.

### 12.5. Vì sao không dùng accuracy làm metric chính?

Accuracy phụ thuộc threshold và dễ bị chi phối bởi mất cân bằng lớp/nhóm. Một mô hình có thể có accuracy cao nhưng bỏ sót phần lớn replay hoặc tạo nhiều false alarm ở bonafide.

## 13. Thiết kế thực nghiệm công bằng

### 13.1. Nguyên tắc one-variable-at-a-time

Khi so B0, X0 và X1, các yếu tố sau phải giữ giống nhau nếu có thể:

- train/dev/test manifest;
- nhãn và tiền xử lý;
- amplitude policy đã khóa;
- seed hoặc danh sách seed;
- quy tắc chọn checkpoint;
- metric và cách tính;
- ngân sách dữ liệu.

Chỉ thay đổi thành phần mô hình cần nghiên cứu. Nếu nhiều yếu tố thay đổi cùng lúc, không thể biết nguyên nhân tạo ra khác biệt.

### 13.2. Amplitude ablation

Chạy ba pilot B0 với:

- cùng train subset;
- cùng development subset;
- cùng seed;
- cùng số bước/epoch;
- chỉ thay `none`, `peak`, `rms_dbfs`.

Chọn policy bằng development EER. Sau khi khóa policy, dùng nó cho B0 đầy đủ, X0, X1, test và demo.

### 13.3. Band-limit/sample-rate ablation

Mục tiêu là kiểm tra mô hình có dựa vào băng thông hay không. Có thể so kết quả trước/sau một biến đổi band-limit được kiểm soát. Nếu hiệu năng giảm mạnh theo nhóm VC, cần thảo luận shortcut sample rate.

### 13.4. Báo cáo chi phí tính toán

RQ2 không chỉ hỏi EER. Phải ghi:

- thời gian huấn luyện;
- peak RAM/VRAM;
- số tham số trainable;
- số tầng XLS-R được mở;
- batch size và gradient accumulation;
- thời gian suy luận nếu có.

### 13.5. Nhiều seed

Nếu ngân sách cho phép, chạy nhiều seed và báo cáo mean ± standard deviation. Một run duy nhất có thể may mắn hoặc không may do khởi tạo và thứ tự batch.

## 14. Shortcut, bias và tính hợp lệ

### 14.1. Shortcut là gì?

Shortcut là đặc trưng tương quan với nhãn trong dataset nhưng không phải bản chất cần học. Ví dụ VC 40 kHz còn nhóm khác 16 kHz.

Mô hình dùng shortcut có thể đạt điểm test cao nếu test giữ cùng bias, nhưng thất bại khi triển khai ngoài miền.

### 14.2. Bằng chứng shortcut hiện có

Amplitude audit trên 2.558 waveform cho RMS median:

| Nhóm | RMS median |
|---|---:|
| Bonafide | -21,29 dBFS |
| AP | -23,36 dBFS |
| VC | -23,91 dBFS |
| Replay | -16,47 dBFS |

Replay cục bộ lớn hơn rõ rệt. Đây là lý do phải có amplitude ablation, không phải bằng chứng rằng RMS normalization chắc chắn tốt nhất.

### 14.3. Internal validity và external validity

- **Internal validity:** kết quả có thực sự do mô hình/phương pháp đang so sánh hay do leakage, shortcut, tuning test?
- **External validity:** kết quả có tổng quát ra dữ liệu, thiết bị, codec hoặc attack khác không?

Speaker-disjoint split và preprocessing chung tăng internal validity. External validity vẫn bị giới hạn vì chỉ dùng một snapshot và số shard cuối chưa khóa.

## 15. Demo và suy luận file dài

### 15.1. Quy trình dự kiến

```text
WAV/MP3
→ decode, mono, 16 kHz
→ amplitude policy đã khóa
→ cửa sổ 4 giây, hop 2 giây
→ score từng cửa sổ
→ trung bình xác suất cửa sổ
→ threshold lấy từ development
→ kết quả và timeline
```

Luôn thêm cửa sổ cuối kết thúc tại cuối file nếu bước nhảy trước chưa phủ hết.

### 15.2. Vì sao dùng mean thay vì max?

- Mean phản ánh toàn bộ file và ít nhạy với một cửa sổ bất thường.
- Max hữu ích để chỉ ra đoạn đáng ngờ nhưng dễ tạo false alarm.
- Vì vậy mean là score chính; max/timeline chỉ hỗ trợ giải thích.

### 15.3. Giới hạn khi trình diễn

- Xác suất mô hình không phải bằng chứng pháp lý.
- File ngoài miền huấn luyện có thể làm score không đáng tin.
- “90% spoof” không tự động nghĩa là xác suất thống kê đã được calibration tốt.
- Demo phải dùng đúng preprocessing và threshold của thí nghiệm, không có pipeline riêng.

## 16. Cấu trúc mã nguồn phải biết chỉ dẫn

| Thành phần | Vị trí | Vai trò |
|---|---|---|
| Metadata | `data/metadata/vsasv_metadata.csv` | Nguồn tạo split |
| Split chính | `data/splits/closed_*.csv` | Train/dev/test speaker-disjoint |
| Smoke subset | `data/splits/smoke_*.csv` | Kiểm tra code nhanh |
| Giao thức audio | `configs/audio.json` | Quy tắc tiền xử lý đã khóa |
| Cấu hình B0 | `configs/lfcc_lcnn.json` | Hợp đồng waveform/LFCC/LCNN |
| Dataset Loader | `src/data/vsasv.py` | Đọc Parquet và tiền xử lý |
| Amplitude policy | `src/data/amplitude.py` | `none`, `peak`, `rms_dbfs` |
| Metric EER | `src/metrics/eer.py` | Tính EER và threshold |
| Leakage checker | `scripts/check_leakage.py` | Kiểm tra split |
| Amplitude audit | `scripts/audit_amplitude.py` | Thống kê shortcut biên độ |
| Smoke generator | `scripts/make_smoke_subset.py` | Tạo subset xác định |
| Báo cáo audit | `reports/amplitude_audit.md` | Bằng chứng biên độ |
| Nhật ký quyết định | `docs/decisions.md` | Lý do và ảnh hưởng quyết định |

Khi bảo vệ, cần có khả năng mở đúng file để chứng minh một tuyên bố thay vì chỉ nói “em đã kiểm tra”.

## 17. Trạng thái hiện tại và phần chưa làm

### Đã có bằng chứng hoàn thành

- Metadata audit.
- Split speaker-disjoint và leakage checker.
- Audio smoke test trên 5 shard.
- Dataset Loader mono 16 kHz, 64.000 mẫu.
- Ba amplitude policy và audit 2.558 waveform.
- Smoke subset 448 mẫu cân bằng và tái lập.
- Metric EER có unit test.
- Cấu hình LFCC và kiểm tra shape `(batch, 60, 401)`.
- LCNN tối thiểu, một epoch smoke pilot và checkpoint B0 có thể khôi phục.
- 49/49 unit test đạt tại mốc 01/10/2026.

### Chưa được phép nói là đã hoàn thành

- Kết quả EER của B0.
- XLS-R đóng băng.
- XLS-R fine-tune một phần.
- Lựa chọn amplitude policy.
- Thí nghiệm cuối và demo.

Phải phân biệt rõ “đã khóa giao diện” với “đã huấn luyện mô hình”.

## 18. Cách đọc và diễn giải kết quả sau này

### Trường hợp X1 tốt hơn X0 và B0

Kết luận hợp lý: trên giao thức và dữ liệu đã khóa, fine-tune một phần giúp giảm EER so với encoder đóng băng và baseline LFCC.

Không được suy rộng thành: X1 luôn tốt hơn trên mọi ngôn ngữ, attack hoặc môi trường.

### Trường hợp X1 không tốt hơn X0

Các nguyên nhân cần xem xét:

- dữ liệu chưa đủ lớn;
- learning rate không phù hợp;
- mở quá nhiều hoặc quá ít tầng;
- overfit;
- baseline frozen đã đủ mạnh;
- dữ liệu chứa shortcut hoặc noise;
- variance giữa seed.

Kết quả âm vẫn có giá trị nếu giao thức đúng và phân tích rõ trade-off.

### Trường hợp metric tổng tốt nhưng replay kém

Không được chỉ trình bày metric tổng. Phải nêu mất cân bằng replay và kết quả theo nhóm. Đây là ví dụ vì sao macro-F1 và phân tích `utt_type` cần thiết.

### Trường hợp normalization cải thiện development nhưng giảm test

Có thể có distribution shift hoặc pilot dev chưa đại diện. Vẫn phải giữ policy đã khóa trước khi xem test; không đổi policy sau khi biết test vì như vậy là test leakage.

## 19. Những câu tuyệt đối không nên nói

- “Mô hình phát hiện được mọi deepfake.”
- “Đây là bằng chứng pháp lý cho file thật hay giả.”
- “Đồ án nhận diện được loại tấn công.”
- “Đồ án tổng quát trên unseen TTS engines.”
- “Năm shard đại diện cho toàn bộ VSASV.”
- “Accuracy cao chứng minh mô hình tốt.”
- “XLS-R hiểu nội dung và biết âm thanh là giả.”
- “RMS normalization chắc chắn tốt nhất” trước khi có development EER.
- “Đã hoàn thành LFCC + LCNN” khi mới chỉ khóa LFCC và tensor interface.
- “Fine-tune một phần là kiến trúc mới do đồ án đề xuất.”

Nên dùng cách nói có điều kiện:

> Trên snapshot, split, preprocessing và ngân sách thí nghiệm đã khóa, mô hình đạt kết quả X; khả năng suy rộng ngoài phạm vi này cần được đánh giá thêm.

## 20. Bộ câu hỏi phản biện cần tự luyện

### 20.1. Phạm vi và đóng góp

1. Tại sao chọn bài toán nhị phân thay vì phân loại VC/AP/replay?
2. Đóng góp của đồ án là gì nếu không tạo kiến trúc mới?
3. Vì sao mô hình chính là XLS-R còn LFCC + LCNN là baseline?
4. Fine-tune một phần khác fine-tune toàn bộ như thế nào?
5. Tại sao không dùng AASIST làm mô hình chính?

### 20.2. Dữ liệu

6. Vì sao phải chia theo speaker?
7. Nếu cùng speaker ở train và test thì kết quả bị sai lệch ra sao?
8. Tại sao dùng closed protocol làm chính?
9. Open split còn vai trò gì?
10. Replay quá ít ảnh hưởng thế nào đến đánh giá?
11. Vì sao không thể tuyên bố unseen TTS engine?
12. Snapshot công khai khác dữ liệu bài báo có hệ quả gì?

### 20.3. Tiền xử lý

13. Vì sao chọn 16 kHz và 4 giây?
14. Resample khác thay đổi header sample rate như thế nào?
15. Vì sao train random crop nhưng test center crop?
16. Vì sao file ngắn dùng repeat-then-trim?
17. Peak và RMS khác nhau thế nào?
18. Vì sao không khuếch đại near-silence?
19. Tại sao amplitude policy phải áp dụng giống nhau cho mọi lớp?
20. Shortcut sample rate và amplitude nguy hiểm ra sao?

### 20.4. Mô hình

21. LFCC được tạo như thế nào?
22. Tại sao LFCC phù hợp làm baseline anti-spoofing?
23. LCNN khác CNN thông thường ở điểm nào?
24. Max-Feature-Map khác max pooling ra sao?
25. XLS-R đã học gì trong pretraining?
26. Frozen XLS-R có ưu/nhược điểm gì?
27. Vì sao mở các tầng Transformer cuối thay vì tầng đầu?
28. Vì sao encoder và classifier có thể cần learning rate khác nhau?

### 20.5. Huấn luyện và đánh giá

29. Logit khác xác suất thế nào?
30. Vì sao dùng `BCEWithLogitsLoss`?
31. EER là gì và được tính như thế nào?
32. Vì sao EER phù hợp hơn accuracy?
33. ROC-AUC và macro-F1 bổ sung thông tin gì?
34. Threshold được chọn ở đâu và vì sao?
35. Tại sao không được dùng test để early stopping?
36. Làm sao bảo đảm so sánh B0/X0/X1 công bằng?
37. Chi phí tính toán nào cần báo cáo cho RQ2?

### 20.6. Kết quả và giới hạn

38. Nếu X1 kém X0 thì đồ án có thất bại không?
39. Nếu kết quả replay rất kém nhưng EER tổng tốt thì kết luận gì?
40. Làm sao biết mô hình không chỉ học sample rate?
41. Vì sao amplitude audit không đủ để chọn policy?
42. Kết quả trên 5 shard có ý nghĩa gì và không có ý nghĩa gì?
43. Mô hình có dùng được làm bằng chứng pháp lý không?
44. Nếu có thêm thời gian/GPU, thí nghiệm tiếp theo là gì?

Mỗi câu cần chuẩn bị câu trả lời ngắn 20–30 giây và câu trả lời sâu 1–2 phút có dẫn chứng từ dự án.

## 21. Khung slide bảo vệ đề xuất

| Slide | Nội dung | Thông điệp chính |
|---|---|---|
| 1 | Bối cảnh và vấn đề | Deepfake tiếng nói cần detector đáng tin cậy |
| 2 | Mục tiêu, RQ1, RQ2 | So sánh B0, X0, X1 |
| 3 | Dữ liệu và giới hạn snapshot | 220.963 mẫu, 1.141 speaker, provenance hạn chế |
| 4 | Split và chống leakage | Speaker-disjoint, train/dev/test rõ vai trò |
| 5 | Tiền xử lý | Mono, 16 kHz, 4 giây, amplitude policy |
| 6 | B0 LFCC + LCNN | Baseline và pipeline đặc trưng |
| 7 | X0/X1 XLS-R | Frozen so với partial fine-tuning |
| 8 | Thiết kế thí nghiệm | Công bằng, seed, checkpoint theo dev EER |
| 9 | Metric | EER chính; ROC-AUC, macro-F1 bổ sung |
| 10 | Kết quả | Bảng B0/X0/X1 và chi phí |
| 11 | Phân tích lỗi/shortcut | VC/AP/replay, sample rate, biên độ |
| 12 | Demo, giới hạn, kết luận | Giá trị và phạm vi sử dụng |

Trong mỗi slide, ưu tiên một thông điệp và một bằng chứng. Không đưa quá nhiều mã nguồn hoặc bảng số dày đặc lên slide.

## 22. Checklist tự đánh giá trước khi bảo vệ

### Phạm vi

- [ ] Nói đúng tên đề tài và ba cấu hình B0/X0/X1.
- [ ] Giải thích được bài toán nhị phân và vai trò của `utt_type`.
- [ ] Nêu đúng đóng góp, không tuyên bố kiến trúc mới.
- [ ] Nêu được RQ1, RQ2 và giả thuyết.

### Dữ liệu

- [ ] Nhớ các số liệu chính của VSASV.
- [ ] Giải thích được speaker-disjoint và các dạng leakage.
- [ ] Phân biệt train/dev/test.
- [ ] Trình bày đúng giới hạn snapshot và replay.

### Tín hiệu âm thanh

- [ ] Giải thích sample rate, Nyquist, resampling và aliasing.
- [ ] Tính được 4 giây × 16 kHz = 64.000 mẫu.
- [ ] Giải thích peak, RMS, dBFS và ba amplitude policy.
- [ ] Giải thích random crop, center crop và repeat-then-trim.

### Mô hình

- [ ] Vẽ được luồng waveform → LFCC → LCNN → logit.
- [ ] Giải thích được STFT, filter bank, log và DCT trong LFCC.
- [ ] Giải thích CNN, MFM và vai trò baseline.
- [ ] Vẽ được luồng waveform → XLS-R → pooling → classifier.
- [ ] So sánh frozen và partial fine-tuning bằng cả hiệu năng lẫn chi phí.

### Huấn luyện và metric

- [ ] Giải thích logit, sigmoid và BCEWithLogitsLoss.
- [ ] Mô tả được một bước forward/backward/optimizer.
- [ ] Tính và diễn giải được confusion matrix, EER, ROC-AUC, macro-F1.
- [ ] Giải thích vì sao threshold và checkpoint chọn trên dev.
- [ ] Nêu được cách bảo đảm so sánh công bằng.

### Phản biện

- [ ] Nêu được ít nhất năm shortcut/bias có thể xảy ra.
- [ ] Giải thích đúng ý nghĩa và giới hạn của amplitude audit.
- [ ] Không suy rộng kết quả từ năm shard.
- [ ] Có câu trả lời nếu X1 không tốt hơn X0.
- [ ] Có thể chỉ vào file/report cụ thể làm bằng chứng.

## 23. Lộ trình học đề xuất

### Buổi 1 — Câu chuyện nghiên cứu

- Học mục 3–6.
- Tự nói bài trình bày 60 giây không nhìn tài liệu.
- Trả lời câu hỏi 1–12.

### Buổi 2 — Xử lý tín hiệu

- Học mục 7–8.
- Tự tính số mẫu, thời lượng window/hop và giải thích LFCC bằng sơ đồ.
- Trả lời câu hỏi 13–22.

### Buổi 3 — Mô hình và huấn luyện

- Học mục 9–11.
- Vẽ B0, X0, X1 từ trí nhớ.
- Trả lời câu hỏi 23–30.

### Buổi 4 — Metric và thí nghiệm

- Học mục 12–14.
- Tự tạo một confusion matrix nhỏ và tính FPR/FNR.
- Trả lời câu hỏi 31–37.

### Buổi 5 — Kết quả, giới hạn và demo

- Học mục 15–19.
- Luyện cách diễn giải ba kịch bản kết quả tốt/xấu/trái kỳ vọng.
- Trả lời câu hỏi 38–44.

### Buổi 6 — Thực hành với repository

- Chạy test và các script audit.
- Mở từng file ở mục 16 và giải thích vai trò.
- Tự lần theo một mẫu từ CSV → Parquet → waveform → LFCC.

### Buổi 7 — Bảo vệ thử

- Trình bày 10–15 phút theo khung slide.
- Nhờ người khác hỏi ngẫu nhiên 15 câu phản biện.
- Ghi lại câu chưa trả lời rõ và bổ sung bằng bằng chứng từ repository.

## 24. Tài liệu trong dự án nên đọc theo thứ tự

1. `README.md` — trạng thái và cách chạy.
2. `docs/decisions.md` — phạm vi và lý do các quyết định.
3. `docs/split_protocol.md` — dữ liệu và chống leakage.
4. `docs/audio_protocol.md` — tiền xử lý và shortcut.
5. `reports/metadata_audit.md` — chất lượng metadata.
6. `reports/split_summary.md` — số liệu split.
7. `reports/leakage_check.md` — bằng chứng không rò rỉ metadata.
8. `reports/amplitude_audit.md` — bằng chứng về biên độ.
9. `configs/lfcc_lcnn.json` — giao diện baseline.
10. `lo_trinh_hoan_chinh_do_an_deepfake_tieng_viet.md` — toàn bộ lộ trình nghiên cứu.

Tài liệu `docs/tong_quan_cach_thuc_hoat_dong_do_an.md` chỉ có giá trị lịch sử. Nếu nội dung mâu thuẫn, ưu tiên `docs/decisions.md`, lộ trình hiện hành và các giao thức đã khóa.

## 25. Kết luận cần ghi nhớ

Bản chất của đồ án không phải chỉ là “đưa audio vào mô hình rồi lấy accuracy”. Một kết quả có giá trị cần đồng thời thỏa mãn:

```text
Dữ liệu đúng và không rò rỉ
→ tiền xử lý nhất quán, không tạo shortcut
→ mô hình và baseline có vai trò rõ ràng
→ thí nghiệm công bằng
→ metric phù hợp
→ test không bị dùng để lựa chọn
→ kết luận nằm trong phạm vi bằng chứng
```

Nếu nắm vững được chuỗi lập luận này, các công thức, lựa chọn kỹ thuật và kết quả thực nghiệm sẽ liên kết thành một câu chuyện nghiên cứu nhất quán thay vì những phần rời rạc.
