# Lộ trình hoàn chỉnh đồ án phát hiện giọng nói deepfake tiếng Việt

## 0. Thông tin đề tài

**Tên đề tài khuyến nghị:**

> Phát hiện giả mạo giọng nói tiếng Việt có khả năng tổng quát hóa trên người nói, loại tấn công và điều kiện kênh truyền chưa biết

**Tên tiếng Anh:**

> Generalizable Vietnamese Speech Deepfake Detection under Unseen Speakers, Spoofing Attacks, and Channel Transformations

**Bài toán chính:** phân loại một đoạn âm thanh thành `bonafide` (giọng thật) hoặc `spoof` (giọng giả/giả mạo).

**Đóng góp trọng tâm:** xây dựng giao thức đánh giá chống rò rỉ theo người nói, huấn luyện mô hình bền vững trước biến đổi kênh truyền, và đánh giá trên loại tấn công/corpus chưa xuất hiện trong lúc huấn luyện.

**Thời lượng đề xuất:** 14–16 tuần.

**Phần cứng tối thiểu:** GPU 8 GB VRAM; 12–16 GB VRAM thuận lợi hơn. Có thể dùng Google Colab/Kaggle hoặc máy chủ của trường.

---

## 1. Kết quả cần đạt ở cuối đồ án

Đồ án được xem là hoàn thành khi có đủ các sản phẩm sau:

1. Một pipeline dữ liệu tái lập được từ metadata đến train/dev/test.
2. Script kiểm tra rò rỉ người nói và file giữa các tập.
3. Ít nhất hai baseline hoạt động:
   - LFCC/log-Mel + LCNN hoặc ResNet nhỏ.
   - XLS-R đóng băng + classifier.
4. Một phương pháp đề xuất có tính nghiên cứu:
   - Channel augmentation.
   - Consistency learning giữa âm thanh gốc và âm thanh đã biến đổi.
5. Đánh giá riêng:
   - Người nói chưa thấy.
   - Attack type đã thấy.
   - Attack type chưa thấy.
   - Kênh truyền chưa thấy.
   - Corpus ngoài miền nếu truy cập được SEA-Spoof.
6. Báo cáo EER, Generalization Gap, Worst-group EER và ablation study.
7. Demo nhận một file âm thanh và trả về:
   - Xác suất deepfake.
   - Mức không chắc chắn.
   - Timeline các đoạn đáng ngờ.
8. Báo cáo tốt nghiệp, slide và hướng dẫn tái lập kết quả.

---

## 2. Phạm vi chính thức

### 2.1. Phạm vi bắt buộc

- Tiếng Việt.
- Phân loại `bonafide` và `spoof`.
- VSASV là bộ dữ liệu chính.
- Chia tập không trùng người nói.
- Đánh giá `voice_conversion`, `adversarial_attack` và `replay`.
- Không yêu cầu nhận diện danh tính người nói.
- Không yêu cầu xác định chính xác công nghệ nào đã tạo giọng.

### 2.2. Phạm vi mở rộng

- SEA-Spoof tiếng Việt làm external test hoặc dữ liệu bổ sung.
- Đánh giá TTS/synthesis source nếu metadata SEA-Spoof đủ chi tiết.
- Fine-tune một phần XLS-R.
- Sharpness-Aware Minimization (SAM).
- Calibration và cơ chế từ chối khi không chắc chắn.
- Phát hiện partially spoofed theo cửa sổ thời gian.

### 2.3. Ngoài phạm vi

- Tự huấn luyện foundation model từ đầu.
- Tạo hệ thống voice cloning.
- Nhận diện người nói.
- Deepfake video hoặc đồng bộ môi–tiếng.
- Tuyên bố phát hiện chính xác mọi giọng AI.
- Dùng kết quả detector như bằng chứng pháp lý.

---

## 3. Câu hỏi nghiên cứu

### RQ1 — Khả năng tổng quát hóa theo người nói

Detector có duy trì hiệu năng khi toàn bộ người nói trong test chưa từng xuất hiện trong train không?

### RQ2 — Khả năng tổng quát hóa theo loại tấn công

Mô hình huấn luyện bằng voice conversion có phát hiện được adversarial attack và replay chưa từng thấy không?

### RQ3 — Biểu diễn self-supervised

XLS-R có tổng quát tốt hơn LFCC/log-Mel trên dữ liệu tiếng Việt không?

### RQ4 — Channel-consistency learning

Huấn luyện nhất quán giữa audio gốc và audio bị nén/thêm nhiễu có giảm EER trên kênh truyền chưa biết không?

### RQ5 — Tổng quát hóa ngoài corpus

Mô hình huấn luyện trên VSASV có hoạt động trên SEA-Spoof tiếng Việt mà không fine-tune không?

### Giả thuyết chính

> XLS-R kết hợp channel augmentation và consistency loss sẽ giảm EER trung bình và Worst-group EER trên attack/channel chưa biết so với XLS-R chỉ huấn luyện bằng cross-entropy.

---

## 4. Hiện trạng dữ liệu VSASV đã xác minh

File `vsasv_metadata.csv` đã được kiểm tra với kết quả:

| Thuộc tính | Kết quả |
|---|---:|
| Tổng số mẫu | 220.963 |
| Số đường dẫn duy nhất | 220.963 |
| Dòng trùng | 0 |
| Giá trị thiếu | 0 |
| Số speaker ID | 1.141 |
| `label` khớp prefix trong `file` | 100% |
| Bonafide | 98.305 |
| Voice conversion | 60.949 |
| Adversarial attack | 60.949 |
| Replay | 760 |

Kết luận schema:

```text
file      = đường dẫn logic của audio
label     = speaker_id
utt_type  = bonafide / voice_conversion / adversarial_attack / replay
```

Phân bố nhị phân:

```text
bonafide: 98.305 mẫu, 44,49%
spoof:   122.658 mẫu, 55,51%
```

Phân bố người nói theo loại:

| Nhóm | Số người nói |
|---|---:|
| Có bonafide | 1.141 |
| Có voice conversion | 144 |
| Có adversarial attack | 144 |
| Có replay | 19 |

Các điểm cần lưu ý:

- 144 speaker voice conversion trùng hoàn toàn với 144 speaker adversarial.
- 19 speaker replay không trùng nhóm voice conversion/adversarial.
- Voice conversion và adversarial có cùng số mẫu; có khả năng chúng liên quan nguồn, nhưng metadata chưa có `source_id` để xác minh.
- Replay chỉ chiếm 0,34%, không thể đánh giá bằng accuracy tổng.
- Số mẫu trên mỗi speaker mất cân bằng mạnh: nhỏ nhất 1, trung vị 64, lớn nhất 6.720.
- VSASV không có TTS hoặc `generator_id`; không được tuyên bố unseen TTS engine nếu chỉ dùng bộ này.

---

## 5. Chiến lược sử dụng dữ liệu

### 5.1. VSASV là dữ liệu chính

VSASV được dùng để:

- Huấn luyện baseline và phương pháp đề xuất.
- Đánh giá speaker-disjoint.
- Đánh giá seen attack và unseen attack.
- Đánh giá replay như một loại tấn công hiếm ngoài miền.

### 5.2. SEA-Spoof là dữ liệu bổ sung

SEA-Spoof chỉ được đưa vào sau khi pipeline VSASV đã hoàn chỉnh.

Mục đích ưu tiên:

1. External cross-dataset test mà không fine-tune.
2. Sau đó mới thử diverse training bằng một subset SEA-Spoof tiếng Việt.

Không cần tải toàn bộ SEA-Spoof. Mục tiêu ban đầu:

- 5.000 bonafide.
- 5.000–15.000 spoof.
- Cân bằng theo `spoof_type` nếu có thể.

### 5.3. Không trộn dữ liệu một cách mù quáng

Khi hợp nhất nhiều corpus phải thêm trường:

```csv
file,corpus,speaker_id,label,attack_type,generator_id,original_split
```

Mỗi batch cần cân bằng:

- Bonafide/spoof.
- Corpus.
- Attack type.
- Speaker, trong khả năng cho phép.

Nếu toàn bộ bonafide đến từ corpus A và toàn bộ spoof đến từ corpus B, mô hình sẽ học nhận diện corpus thay vì deepfake.

---

## 6. Giao thức chia dữ liệu

### 6.1. Nguyên tắc bắt buộc

1. Không chia ngẫu nhiên theo dòng.
2. Không có `speaker_id` giao nhau giữa train, dev và test.
3. Không dùng test để chọn threshold, epoch hoặc siêu tham số.
4. Mọi phiên bản augmentation của cùng file phải nằm trong cùng split.
5. Nếu sau này có `source_id`, source không được giao nhau giữa các split.
6. Lưu danh sách split thành CSV cố định và version-control.

### 6.2. Phân tầng speaker

Chia speaker thành ba nhóm trước:

```text
S_real_only: chỉ có bonafide
S_vc_adv:    có bonafide + voice_conversion + adversarial_attack
S_replay:    có bonafide + replay
```

Không chia toàn bộ 1.141 speaker một lần vì các nhóm attack có phân bố khác nhau.

### 6.3. Protocol A — Closed-set speaker generalization

Mục tiêu: đo khả năng tổng quát hóa sang người nói mới khi attack type đã biết.

Chia từng speaker stratum theo tỷ lệ:

```text
Train:      70%
Validation: 15%
Test:       15%
```

Train/dev/test đều có các loại attack tương ứng, nhưng speaker hoàn toàn tách biệt.

Đầu ra:

- `closed_train.csv`
- `closed_dev.csv`
- `closed_test.csv`

### 6.4. Protocol B — Open-set attack generalization

Đây là giao thức chính của đồ án.

#### Train

- Bonafide từ speaker train.
- Voice conversion từ speaker train.
- Không có adversarial attack.
- Không có replay.

#### Validation

- Bonafide từ speaker dev.
- Voice conversion từ speaker dev.
- Dùng để chọn epoch và threshold.

#### Seen test

- Bonafide từ speaker test.
- Voice conversion từ speaker test.
- Speaker chưa thấy, attack type đã thấy.

#### Unseen test A

- Bonafide từ cùng nhóm speaker test.
- Adversarial attack từ speaker test.
- Speaker chưa thấy, attack type chưa thấy.

#### Unseen test B

- Bonafide của 19 speaker replay.
- 760 replay samples.
- Speaker và attack type đều chưa thấy.

Đầu ra:

- `open_train_vc.csv`
- `open_dev_vc.csv`
- `open_seen_test_vc.csv`
- `open_unseen_test_adversarial.csv`
- `open_unseen_test_replay.csv`

### 6.5. Protocol C — Reciprocal attack experiment

Đổi vai trò:

```text
Train known attack: adversarial_attack
Unseen attack:      voice_conversion
```

Thí nghiệm này kiểm tra kết quả có phụ thuộc việc chọn attack nào làm train hay không.

### 6.6. Kiểm tra rò rỉ tự động

Script `check_leakage.py` phải kiểm tra:

- Giao nhau speaker giữa mọi cặp split.
- Giao nhau file.
- Giao nhau hash nếu audio đã tải cục bộ.
- Phân bố label/attack/corpus.
- Số mẫu và số speaker của từng split.
- File path tồn tại hoặc row ID truy cập được.

Nếu có bất kỳ overlap nào, script phải trả exit code khác 0.

---

## 7. Chiến lược lưu trữ và tải dữ liệu

### 7.1. Không đưa audio vào Git

Git chỉ lưu:

- Metadata.
- Split CSV.
- Config.
- Code.
- Kết quả tổng hợp và hình nhỏ.

Thêm vào `.gitignore`:

```gitignore
data/raw/
data/processed/
data/embeddings/
checkpoints/
logs/
*.wav
*.flac
*.pt
*.ckpt
```

### 7.2. Ba mức làm việc

#### Mức 1 — Smoke test

- 1.000–2.000 mẫu.
- Chạy trên CPU hoặc GPU nhỏ.
- Mục tiêu: kiểm tra code, không báo cáo kết quả khoa học.

#### Mức 2 — Development subset

- 20.000–40.000 mẫu.
- Cân bằng class, attack type và speaker.
- Dùng để phát triển mô hình và chọn cấu hình.

#### Mức 3 — Final experiment

- Dùng tập lớn hơn hoặc toàn bộ phần dữ liệu thuộc protocol.
- Chỉ chạy sau khi code và metric đã được khóa.

### 7.3. Ưu tiên embedding offline

Nếu ổ đĩa/GPU hạn chế:

1. Stream audio qua XLS-R bằng `eval()` và `torch.no_grad()`.
2. Tính embedding pooled kích thước cố định.
3. Lưu embedding clean.
4. Tạo một số augmentation có seed cố định rồi lưu augmented embedding.
5. Huấn luyện classifier trên embedding.

Không lưu toàn bộ hidden state theo frame nếu không cần, vì dung lượng sẽ tăng mạnh.

### 7.4. Khi nào cần tải audio cục bộ

Audio cục bộ cần thiết khi:

- Fine-tune một phần encoder.
- Tạo augmentation on-the-fly.
- Phân tích lỗi bằng waveform/spectrogram.
- Chạy lặp lại nhiều epoch mà streaming quá chậm.

Nếu tải toàn bộ, nên dùng SSD ngoài hoặc máy chủ có ít nhất 300–500 GB trống.

---

## 8. Tiền xử lý âm thanh

### 8.1. Chuẩn hóa chung

- Mono.
- Sample rate 16 kHz.
- Dùng `float32` trong khoảng hợp lệ.
- Không denoise mạnh.
- Không xử lý hai lớp bằng pipeline khác nhau.

### 8.2. Độ dài đầu vào

- Cửa sổ 4 giây = 64.000 samples tại 16 kHz.
- Train: random crop.
- Validation/test: sliding window 4 giây, hop 2 giây.
- File ngắn: pad và dùng attention mask.
- File dài: không chỉ lấy đoạn đầu.

### 8.3. Im lặng

- Đo tỷ lệ im lặng và duration.
- Không loại im lặng hoàn toàn trước khi phân tích, vì nó có thể là dấu hiệu corpus.
- Sau audit mới đặt ngưỡng VAD/energy chung cho cả hai lớp.

### 8.4. Augmentation dùng khi train

Áp dụng cho cả bonafide và spoof:

- MP3: 32/64/96/128 kbps.
- Additive noise: SNR 5/10/20 dB.
- Room impulse response.
- Random gain.
- Resample 16 kHz → 8 kHz → 16 kHz.
- Speed perturbation nhẹ: 0,9/1,0/1,1.
- Clipping nhẹ.

### 8.5. Unseen channel transformations

Giữ một số phép biến đổi chỉ cho test:

- Opus.
- Telephone band-pass 300–3400 Hz.
- Dynamic range compression.
- Kết hợp noise + codec.
- Một RIR family không dùng trong train.

Không được dùng unseen transformation để chọn siêu tham số.

---

## 9. Các mô hình cần thực hiện

### B0 — LFCC/log-Mel + LCNN

Mục đích:

- Baseline nhẹ.
- Kiểm tra pipeline end-to-end.
- So sánh đặc trưng phổ truyền thống với SSL embedding.

Cấu hình gợi ý:

- 60–80 Mel bins hoặc LFCC tương đương.
- LCNN/ResNet18 nhỏ.
- Weighted cross-entropy.
- Early stopping theo dev EER.

### B1 — AASIST

Mục đích:

- Baseline anti-spoofing chuẩn.
- So sánh với phương pháp phổ biến trong tài liệu.

Nếu thiếu thời gian/GPU, có thể dùng checkpoint hoặc cấu hình nhỏ hơn, nhưng phải ghi rõ.

### B2 — Frozen XLS-R + classifier

Kiến trúc:

```text
Audio 16 kHz
    ↓
XLS-R 300M đóng băng
    ↓
Attentive statistics pooling
    ↓
Linear → GELU → Dropout
    ↓
Linear(2)
```

Đây là baseline chính để phát triển phương pháp đề xuất.

### M1 — XLS-R + channel augmentation

Giống B2 nhưng huấn luyện với audio clean và các phép biến đổi train-time.

### M2 — XLS-R + channel consistency

Với audio gốc `x`, bản biến đổi `T(x)` và nhãn `y`:

```text
p_clean = f(x)
p_aug   = f(T(x))
```

Loss:

```text
L_total = L_cls(p_clean, y)
        + L_cls(p_aug, y)
        + lambda_cons * L_cons(p_clean, p_aug)
```

Lựa chọn ban đầu:

- `L_cls`: weighted cross-entropy.
- `L_cons`: Jensen–Shannon divergence hoặc MSE giữa logits.
- `lambda_cons`: thử 0,1; 0,5; 1,0 trên validation.

### M3 — Partial fine-tuning

Sau khi M2 ổn định:

- Mở 2–4 block cuối của XLS-R.
- Learning rate encoder nhỏ hơn classifier khoảng 10–30 lần.
- Dùng gradient accumulation và AMP.
- So sánh frozen và partial fine-tune.

### M4 — Tùy chọn

- SAM.
- Supervised contrastive loss.
- Domain-adversarial corpus classifier.
- Calibration head.

Chỉ thực hiện M4 nếu B0–M2 và ablation đã hoàn chỉnh.

---

## 10. Thiết lập huấn luyện

### Cấu hình mặc định

```text
Python:             3.10+
Framework:          PyTorch
Optimizer:          AdamW
LR classifier:      3e-4, điều chỉnh trên dev
LR encoder:         1e-5 khi partial fine-tune
Weight decay:       1e-4 hoặc 1e-2
Epoch tối đa:       30
Early stopping:     patience 5 theo dev EER
Scheduler:          cosine + warmup 5–10%
Gradient clipping:  1.0
Precision:          AMP FP16/BF16
Seeds cuối:         42, 123, 2026
```

### Theo VRAM

| VRAM | Thiết lập phù hợp |
|---:|---|
| 4–6 GB | B0 hoặc embedding offline |
| 8 GB | XLS-R frozen, batch 2–4, audio 3–4 giây |
| 12 GB | Frozen hoặc mở 1–2 block cuối |
| 16 GB | AASIST thuận lợi, partial fine-tune XLS-R |
| 24 GB+ | Có thể thử full fine-tune, không bắt buộc |

### Cân bằng dữ liệu

Không chỉ dùng class-balanced sampler. Cần cân bằng theo nhiều cấp:

1. Bonafide/spoof.
2. Attack type.
3. Speaker.
4. Corpus nếu dùng SEA-Spoof.

Replay không được để mất trong batch do tỷ lệ quá nhỏ.

---

## 11. Ma trận thí nghiệm bắt buộc

| ID | Mô hình | Augmentation | Consistency | Protocol |
|---|---|---:|---:|---|
| B0 | LFCC/log-Mel + LCNN | Không | Không | Closed + Open |
| B0-A | LFCC/log-Mel + LCNN | Có | Không | Open |
| B1 | AASIST | Theo recipe | Không | Closed + Open |
| B2 | XLS-R frozen | Không | Không | Closed + Open |
| M1 | XLS-R frozen | Có | Không | Open |
| M2 | XLS-R frozen | Có | Có | Open |
| M3 | XLS-R partial FT | Có | Có | Open |

### Ablation bắt buộc

1. M2 không có consistency loss.
2. M2 không có codec augmentation.
3. M2 không có noise/RIR augmentation.
4. Mean pooling so với attentive statistics pooling.
5. Frozen encoder so với partial fine-tune.
6. `lambda_cons` khác nhau.
7. Train VC → test adversarial và chiều ngược lại.

### Cross-dataset experiments

| ID | Train | Test | Mục tiêu |
|---|---|---|---|
| X1 | VSASV | VSASV unseen | Cross-attack nội bộ |
| X2 | VSASV | SEA-Spoof vi | Cross-corpus không fine-tune |
| X3 | VSASV + SEA-Spoof train subset | VSASV + SEA-Spoof test | Diverse training |

X2 quan trọng hơn X3. Phải biết mô hình gốc chuyển miền kém đến mức nào trước khi thêm dữ liệu ngoài.

---

## 12. Chỉ số đánh giá

### 12.1. Chỉ số chính

#### Equal Error Rate

EER càng thấp càng tốt. Báo cáo riêng:

- `EER_seen`
- `EER_adversarial`
- `EER_replay`
- `EER_unseen_channel`
- `EER_external_corpus`

#### Generalization Gap

```text
Generalization Gap = EER_unseen - EER_seen
```

#### Worst-group EER

```text
Worst-group EER = max(EER của từng attack/channel/corpus)
```

### 12.2. Chỉ số phụ

- ROC-AUC.
- Macro-F1 theo attack type.
- False Acceptance Rate.
- False Rejection Rate.
- minDCF hoặc actDCF.
- Expected Calibration Error.
- Brier score.
- Latency trên một phút audio.
- Peak VRAM.
- Số tham số được huấn luyện.

Accuracy chỉ dùng mô tả, không dùng làm kết luận chính.

### 12.3. Threshold

- Chọn threshold trên validation.
- Khóa threshold trước khi chạy test.
- Không chọn threshold riêng cho từng test set.
- Báo cáo cả threshold-independent metric và fixed-threshold metric.

### 12.4. Độ tin cậy thống kê

Mô hình cuối cần:

- Chạy ba seed.
- Báo cáo mean ± standard deviation.
- Nếu có thể, bootstrap confidence interval theo speaker thay vì theo file.

---

## 13. Phân tích lỗi

Chọn ít nhất 100 false positive và 100 false negative để phân tích:

- Giọng thật thu studio quá sạch.
- Giọng thật bị nén mạnh.
- Spoof chất lượng cao.
- Audio quá ngắn.
- Nhiều khoảng im lặng.
- Noise hoặc nhạc nền lớn.
- Speaker có rất ít mẫu.
- Voice conversion/adversarial có khả năng cùng source.
- Model học codec/corpus thay vì dấu vết giả mạo.

Hình cần tạo:

- Spectrogram các mẫu tiêu biểu.
- UMAP embedding tô màu theo label.
- UMAP tô màu theo attack type.
- UMAP tô màu theo corpus.
- EER theo attack/channel.
- Reliability diagram.
- Confusion matrix tại threshold cố định.

UMAP/t-SNE chỉ là minh họa, không dùng thay thế metric.

---

## 14. Demo cuối

### 14.1. Luồng suy luận

```text
Upload WAV/MP3
    ↓
Resample 16 kHz, mono
    ↓
Cửa sổ 4 giây, hop 2 giây
    ↓
Detector từng cửa sổ
    ↓
Tổng hợp điểm toàn file
    ↓
Kết quả + timeline + uncertainty
```

### 14.2. Ba vùng kết luận

```text
p < tau_low:                có khả năng bonafide
tau_low <= p <= tau_high:  không chắc chắn
p > tau_high:               có khả năng spoof
```

`tau_low` và `tau_high` được chọn trên validation theo yêu cầu FAR/FRR, không đặt tùy ý.

### 14.3. Chức năng tối thiểu

- Upload file.
- Nghe lại audio.
- Hiển thị waveform.
- Hiển thị xác suất.
- Timeline cửa sổ đáng ngờ.
- Hiển thị cảnh báo giới hạn sử dụng.
- Xuất JSON kết quả.

### 14.4. Công nghệ giao diện

Ưu tiên Streamlit hoặc Gradio để giảm thời gian frontend. Demo không được chiếm thời gian của thí nghiệm chính.

---

## 15. Cấu trúc thư mục dự án

```text
DoAnTTNT/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── b0_lcnn.yaml
│   ├── b1_aasist.yaml
│   ├── b2_xlsr_frozen.yaml
│   ├── m1_xlsr_aug.yaml
│   └── m2_xlsr_consistency.yaml
├── data/
│   ├── metadata/
│   │   ├── vsasv_metadata.csv
│   │   └── sea_spoof_vi_metadata.csv
│   ├── splits/
│   │   ├── closed/
│   │   └── open/
│   ├── raw/               # gitignored
│   ├── processed/         # gitignored
│   └── embeddings/        # gitignored
├── scripts/
│   ├── audit_metadata.py
│   ├── make_speaker_splits.py
│   ├── check_leakage.py
│   ├── fetch_audio_subset.py
│   ├── extract_embeddings.py
│   └── evaluate_all.py
├── src/
│   ├── data/
│   │   ├── datasets.py
│   │   ├── samplers.py
│   │   └── augmentations.py
│   ├── models/
│   │   ├── lcnn.py
│   │   ├── aasist_wrapper.py
│   │   └── xlsr_detector.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   └── inference.py
├── tests/
│   ├── test_metadata.py
│   ├── test_splits.py
│   ├── test_augmentations.py
│   └── test_metrics.py
├── checkpoints/           # gitignored
├── logs/                  # gitignored
├── reports/
│   ├── figures/
│   ├── tables/
│   └── error_analysis/
├── app/
│   └── demo.py
└── docs/
    ├── experiment_log.md
    └── decisions.md
```

---

## 16. Môi trường và thư viện

Danh sách khuyến nghị:

```text
torch
torchaudio
transformers
datasets
huggingface_hub
numpy
pandas
scikit-learn
scipy
soundfile
librosa
audiomentations
pyarrow
pyyaml
tqdm
matplotlib
seaborn
umap-learn
tensorboard hoặc wandb
streamlit hoặc gradio
pytest
```

Mỗi run phải lưu:

- Git commit.
- Config YAML.
- Seed.
- Split version.
- Model checkpoint.
- Dev/test metrics.
- Thời gian chạy.
- Peak VRAM.

---

## 17. Lộ trình 16 tuần

### Tuần 1 — Chốt phạm vi và tổ chức dự án

**Công việc:**

- Chốt tên đề tài với giảng viên.
- Tạo cấu trúc thư mục.
- Tạo Git repository.
- Viết `.gitignore`.
- Chốt RQ1–RQ5 và phạm vi bắt buộc/mở rộng.

**Đầu ra:**

- README đầu tiên.
- Lộ trình được duyệt.
- Repository sạch.

**Cổng kiểm tra G1:** giảng viên đồng ý đề tài tập trung unseen speaker/attack/channel, không yêu cầu tự huấn luyện foundation model.

### Tuần 2 — Audit metadata và giấy phép

**Công việc:**

- Đưa metadata vào `data/metadata`.
- Viết `audit_metadata.py`.
- Lưu thống kê class, speaker và attack.
- Đọc điều kiện sử dụng VSASV/SEA-Spoof.
- Gửi yêu cầu truy cập SEA-Spoof nếu cần.

**Đầu ra:**

- `metadata_audit.json`.
- Bảng thống kê dữ liệu.
- Danh sách rủi ro dữ liệu.

### Tuần 3 — Tạo split chống rò rỉ

**Công việc:**

- Phân nhóm `S_real_only`, `S_vc_adv`, `S_replay`.
- Tạo closed-set split.
- Tạo open-set split.
- Viết leakage checker.
- Tạo unit test cho split.

**Đầu ra:**

- Các file split CSV.
- Báo cáo không overlap speaker/file.

**Cổng kiểm tra G2:** mọi split không trùng speaker; phân bố attack được ghi lại.

### Tuần 4 — Pipeline audio và smoke test

**Công việc:**

- Load audio từ Hugging Face/local.
- Resample, crop, pad.
- Kiểm tra waveform và spectrogram.
- Chạy thử 1.000–2.000 mẫu.
- Đo tốc độ DataLoader.

**Đầu ra:**

- DataLoader ổn định.
- Bộ hình kiểm tra audio.
- Báo cáo throughput.

### Tuần 5 — Metric và baseline B0

**Công việc:**

- Cài EER.
- Kiểm thử EER bằng dữ liệu giả lập.
- Huấn luyện LFCC/log-Mel + LCNN.
- Chạy closed-set smoke experiment.

**Đầu ra:**

- B0 checkpoint.
- B0 dev/test metrics.
- Unit test metrics.

### Tuần 6 — Hoàn thiện B0 và attack-wise evaluation

**Công việc:**

- Chạy B0 trên open protocol.
- Báo cáo seen/adversarial/replay riêng.
- Kiểm tra mất cân bằng replay.
- Thử balanced sampler.

**Đầu ra:**

- Bảng B0 đầy đủ.
- Nhận xét lỗi ban đầu.

**Cổng kiểm tra G3:** pipeline train → score → EER chạy tự động end-to-end.

### Tuần 7 — XLS-R feature extraction

**Công việc:**

- Load XLS-R 300M.
- Đóng băng encoder.
- Trích xuất embedding.
- So sánh mean pooling và statistics pooling.
- Cache embedding development subset.

**Đầu ra:**

- Embedding cache.
- Báo cáo thời gian/VRAM/dung lượng.

### Tuần 8 — Baseline B2

**Công việc:**

- Huấn luyện classifier trên XLS-R embedding.
- Chạy closed/open protocols.
- So sánh B0 và B2.
- Khóa cấu hình baseline chính.

**Đầu ra:**

- B2 checkpoint.
- Bảng B0/B2.

### Tuần 9 — Augmentation pipeline

**Công việc:**

- Cài MP3/noise/RIR/resampling/gain.
- Kiểm tra augmentation áp dụng cho cả hai lớp.
- Nghe và vẽ mẫu augmented.
- Tạo unseen channel test.

**Đầu ra:**

- `augmentations.py` có unit test.
- Danh sách train/unseen transformations.

### Tuần 10 — M1 và consistency loss

**Công việc:**

- Chạy XLS-R + augmentation.
- Cài consistency loss.
- Thử `lambda_cons` trên dev.
- Không xem test trong quá trình chọn tham số.

**Đầu ra:**

- M1 metrics.
- M2 prototype.

### Tuần 11 — M2 chính thức

**Công việc:**

- Chạy M2 với config đã khóa.
- Chạy ba seed.
- Đánh giá seen/unseen attacks và channels.
- Tính Generalization Gap và Worst-group EER.

**Đầu ra:**

- M2 checkpoints.
- Bảng kết quả chính.

**Cổng kiểm tra G4:** phương pháp đề xuất đã được đánh giá công bằng với cùng split/baseline.

### Tuần 12 — Ablation study

**Công việc:**

- Bỏ consistency.
- Bỏ codec augmentation.
- Bỏ noise/RIR.
- So sánh pooling.
- Chạy reciprocal VC/adversarial.

**Đầu ra:**

- Bảng ablation.
- Kết luận thành phần nào có tác dụng.

### Tuần 13 — Cross-dataset evaluation

**Công việc:**

- Hoàn thiện SEA-Spoof metadata và subset nếu đã được cấp quyền.
- Chạy mô hình VSASV trên SEA-Spoof mà không fine-tune.
- Báo cáo cross-corpus EER.
- Nếu không có SEA-Spoof, dùng tuần này cho calibration và phân tích replay sâu hơn.

**Đầu ra:**

- X2 metrics hoặc báo cáo thay thế.

### Tuần 14 — Phân tích lỗi và calibration

**Công việc:**

- Lấy false positive/false negative.
- Vẽ spectrogram/UMAP/reliability diagram.
- Hiệu chỉnh confidence.
- Định nghĩa vùng không chắc chắn.

**Đầu ra:**

- Thư mục error analysis.
- Bộ hình dùng trong báo cáo.

### Tuần 15 — Demo và viết báo cáo

**Công việc:**

- Xây Streamlit/Gradio demo.
- Đo latency và VRAM.
- Viết chương dữ liệu, phương pháp, thí nghiệm.
- Sinh bảng và hình từ script, không nhập tay.

**Đầu ra:**

- Demo chạy được.
- Bản nháp báo cáo đầy đủ.

### Tuần 16 — Tái lập, sửa lỗi và bảo vệ

**Công việc:**

- Chạy lại một thí nghiệm từ config sạch.
- Kiểm tra README.
- Hoàn thiện slide.
- Chuẩn bị câu hỏi phản biện.
- Đóng băng release cuối.

**Đầu ra:**

- Source code final.
- Report final.
- Slide final.
- Demo final.
- Bảng kết quả đã xác minh.

**Cổng kiểm tra G5:** một người khác có thể chạy inference/evaluation theo README.

---

## 18. Kế hoạch rút gọn 12 tuần

Nếu chỉ có 12 tuần:

1. Gộp tuần 1–2.
2. Gộp tuần 5–6.
3. Không thực hiện AASIST nếu B0 và XLS-R đã đủ.
4. Không fine-tune toàn bộ encoder.
5. Chỉ chạy M2 frozen.
6. SEA-Spoof chỉ làm external test, không diverse training.
7. Không thêm SAM/contrastive/domain adversarial.

Không được bỏ:

- Speaker-disjoint split.
- Leakage checker.
- B0 và B2.
- Open-set attack test.
- Ablation consistency/augmentation.
- EER theo attack type.

---

## 19. Tiêu chí dừng và ưu tiên

Thứ tự ưu tiên:

```text
Đúng split
→ Đúng metric
→ Baseline tái lập
→ Phương pháp chính
→ Ablation
→ External test
→ Demo
→ Các mở rộng khác
```

Khi thiếu thời gian, bỏ theo thứ tự:

1. Full fine-tuning.
2. SAM.
3. Contrastive loss.
4. Diverse training nhiều corpus.
5. Giao diện nâng cao.

Không giảm chất lượng split/metric để đổi lấy thêm mô hình.

---

## 20. Rủi ro và phương án xử lý

| Rủi ro | Phương án |
|---|---|
| Audio quá lớn | Development subset, streaming, embedding offline, SSD ngoài |
| SEA-Spoof chưa được duyệt | Hoàn thành VSASV; thay X2 bằng calibration/error analysis |
| GPU hết VRAM | Frozen XLS-R, batch nhỏ, AMP, gradient accumulation |
| Replay quá ít | Dùng làm unseen test; attack-wise metric; không dựa accuracy |
| Speaker mất cân bằng | Speaker-balanced sampler hoặc giới hạn mẫu/speaker |
| VC và adversarial cùng nguồn | Speaker-disjoint; kiểm tra duration/fingerprint; ghi rõ hạn chế |
| Kết quả quá cao | Audit codec/duration/source leakage; chạy external test |
| Kết quả unseen thấp | Đây là kết quả nghiên cứu; phân tích lỗi và domain shift |
| Augmentation làm giảm seen | Điều chỉnh xác suất/lambda trên dev, báo cáo trade-off |
| Fine-tune không ổn định | Quay lại frozen baseline; giảm LR; mở ít block hơn |
| Demo chậm | Dùng model frozen/quantized; xử lý theo cửa sổ; cache processor |

---

## 21. Đạo đức và giấy phép

- Chỉ dùng dữ liệu theo license nghiên cứu.
- Không phân phối lại audio gated.
- Không đưa audio hoặc token Hugging Face vào GitHub.
- Không dùng dữ liệu để clone hoặc mạo danh giọng nói.
- Không công bố thông tin nhận dạng speaker nếu license không cho phép.
- Ghi rõ detector có false positive/false negative.
- Demo phải có cảnh báo kết quả chỉ mang tính hỗ trợ.
- Báo cáo fairness theo nhóm nếu metadata giới tính/vùng miền hợp lệ; không tự suy đoán các thuộc tính này từ giọng nói.

---

## 22. Cấu trúc báo cáo tốt nghiệp

### Chương 1 — Giới thiệu

- Bối cảnh audio deepfake.
- Vấn đề tổng quát hóa.
- Khoảng trống.
- Mục tiêu và đóng góp.

### Chương 2 — Cơ sở lý thuyết

- TTS, VC, replay và adversarial attack.
- Spectrogram/LFCC.
- Self-supervised speech representation.
- Domain generalization.
- EER, DCF và calibration.

### Chương 3 — Công trình liên quan

- ASVspoof/VSASV/SEA-Spoof.
- LCNN/AASIST.
- XLS-R/WavLM.
- Channel augmentation và consistency learning.

### Chương 4 — Dữ liệu và giao thức

- Thống kê VSASV.
- Metadata audit.
- Speaker strata.
- Closed/open protocols.
- Leakage prevention.
- SEA-Spoof external test.

### Chương 5 — Phương pháp

- Tiền xử lý.
- Baselines.
- Kiến trúc XLS-R detector.
- Augmentation.
- Consistency loss.

### Chương 6 — Thực nghiệm

- Phần cứng/phần mềm.
- Config.
- Metrics.
- Ma trận thí nghiệm.
- Ablation.

### Chương 7 — Kết quả và thảo luận

- Seen/unseen.
- Cross-corpus.
- Error analysis.
- Calibration.
- Hạn chế.

### Chương 8 — Demo và triển khai

- Kiến trúc suy luận.
- Timeline detection.
- Latency/VRAM.

### Chương 9 — Kết luận

- Trả lời từng RQ.
- Đóng góp.
- Hướng phát triển.

---

## 23. Bảng kết quả dự kiến trong báo cáo

### Bảng chính

| Model | Seen EER ↓ | Adv EER ↓ | Replay EER ↓ | Channel EER ↓ | Worst EER ↓ |
|---|---:|---:|---:|---:|---:|
| B0 | | | | | |
| B1 | | | | | |
| B2 | | | | | |
| M1 | | | | | |
| M2 | | | | | |
| M3 | | | | | |

### Bảng cross-dataset

| Train | VSASV seen | VSASV unseen | SEA-Spoof vi |
|---|---:|---:|---:|
| VSASV | | | |
| SEA-Spoof subset | | | |
| VSASV + SEA-Spoof | | | |

### Bảng ablation

| Cấu hình | Consistency | Codec aug | Noise/RIR | Pooling | Unseen EER |
|---|---:|---:|---:|---|---:|
| Full M2 | ✓ | ✓ | ✓ | Attentive stats | |
| Không consistency | ✗ | ✓ | ✓ | Attentive stats | |
| Không codec | ✓ | ✗ | ✓ | Attentive stats | |
| Không noise/RIR | ✓ | ✓ | ✗ | Attentive stats | |
| Mean pooling | ✓ | ✓ | ✓ | Mean | |

---

## 24. Checklist trước khi chạy thí nghiệm cuối

### Dữ liệu

- [ ] Metadata không thiếu giá trị.
- [ ] Không trùng file.
- [ ] Speaker-disjoint được xác nhận bằng script.
- [ ] Phân bố class/attack/corpus đã lưu.
- [ ] Split CSV đã khóa.
- [ ] Test không được dùng để chọn tham số.

### Mô hình

- [ ] Seed cố định.
- [ ] Config được lưu.
- [ ] AMP hoạt động.
- [ ] Checkpoint theo dev EER.
- [ ] Resume training hoạt động.

### Metric

- [ ] EER unit test đúng.
- [ ] Quy ước score thống nhất: điểm cao là spoof hay bonafide.
- [ ] Threshold lấy từ dev.
- [ ] Metric theo attack type.
- [ ] Metric theo corpus.

### Báo cáo

- [ ] Mọi bảng sinh từ file kết quả.
- [ ] Không nhập số thủ công.
- [ ] Có ba seed cho mô hình cuối.
- [ ] Có ablation.
- [ ] Có error analysis.
- [ ] Ghi rõ giới hạn dữ liệu.

---

## 25. Checklist bàn giao cuối

- [ ] `README.md` hướng dẫn cài đặt.
- [ ] `requirements.txt` hoặc environment file.
- [ ] Code train/inference/evaluation.
- [ ] Metadata và split CSV hợp lệ.
- [ ] Không có audio gated trong repository.
- [ ] Không có access token.
- [ ] Checkpoint hoặc link checkpoint theo license.
- [ ] Config của mọi bảng kết quả.
- [ ] Báo cáo PDF.
- [ ] Slide.
- [ ] Demo.
- [ ] Video demo dự phòng.
- [ ] Danh sách hạn chế và hướng phát triển.

---

## 26. Công việc cần làm ngay trong 72 giờ tới

### Ngày 1

1. Tạo cấu trúc thư mục dự án.
2. Đưa `vsasv_metadata.csv` vào `data/metadata` hoặc ghi rõ đường dẫn ngoài Git.
3. Tạo `.gitignore`.
4. Viết `audit_metadata.py` để tái tạo các thống kê đã có.

### Ngày 2

1. Viết `make_speaker_splits.py`.
2. Sinh Protocol A và Protocol B.
3. Viết `check_leakage.py`.
4. Kiểm tra không overlap speaker/file.

### Ngày 3

1. Load thử 100–1.000 audio.
2. Kiểm tra sample rate, duration, amplitude.
3. Vẽ spectrogram của từng `utt_type`.
4. Đo dung lượng/throughput.
5. Chốt development subset đầu tiên.

Sau 72 giờ phải có:

```text
metadata audit
speaker split
leakage report
audio loader smoke test
```

---

## 27. Quy tắc ra quyết định trong quá trình làm

1. Nếu kết quả cao bất thường, kiểm tra leakage trước khi vui mừng.
2. Nếu unseen EER cao, không chỉnh tham số trên test; phân tích domain shift.
3. Nếu GPU yếu, đóng băng encoder thay vì giảm chất lượng split.
4. Nếu SEA-Spoof bị chặn, hoàn thành VSASV trước.
5. Nếu một kỹ thuật mới không có ablation, không tuyên bố nó tạo cải thiện.
6. Nếu chỉ dùng VSASV, dùng cụm từ “unseen spoofing attacks”, không dùng “unseen TTS engines”.
7. Nếu thêm nhiều corpus, luôn báo cáo kết quả riêng từng corpus.
8. Chất lượng giao thức và phân tích quan trọng hơn số lượng mô hình.

---

## 28. Phiên bản trình bày ngắn với giảng viên

> Đồ án xây dựng hệ thống phát hiện giả mạo giọng nói tiếng Việt tập trung vào khả năng tổng quát hóa. Em sử dụng VSASV làm dữ liệu chính với hơn 220 nghìn mẫu và 1.141 người nói. Dữ liệu được chia theo speaker ID để không trùng người nói giữa train và test. Mô hình được huấn luyện bằng một loại tấn công đã biết và kiểm thử trên adversarial attack, replay cùng các biến đổi kênh truyền chưa xuất hiện trong train. Baseline gồm LFCC-LCNN, AASIST và XLS-R đóng băng. Phương pháp đề xuất kết hợp channel augmentation với consistency learning giữa âm thanh gốc và âm thanh bị biến đổi. Kết quả được đánh giá bằng EER, Generalization Gap, Worst-group EER, ablation study và external test trên SEA-Spoof tiếng Việt nếu được cấp quyền.

---

## 29. Definition of Done

Đồ án hoàn thành khi đồng thời thỏa mãn:

```text
[Dữ liệu]
Speaker-disjoint + leakage checker + split cố định

[Mô hình]
B0 + B2 + M2 chạy tái lập

[Thực nghiệm]
Seen + unseen attack + unseen channel + ablation

[Đánh giá]
EER + Generalization Gap + Worst-group EER + 3 seeds

[Phân tích]
Error analysis + calibration hoặc uncertainty

[Sản phẩm]
Code + config + report + slide + demo + README
```

External SEA-Spoof và partial fine-tuning là phần nâng cao; việc chưa thực hiện chúng không làm đồ án thất bại nếu toàn bộ phần bắt buộc đã hoàn chỉnh và được đánh giá đúng giao thức.
