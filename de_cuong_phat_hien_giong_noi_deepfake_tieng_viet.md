# Đề cương chi tiết: Phát hiện giọng nói deepfake tiếng Việt có khả năng tổng quát hóa

**Tên đề tài tiếng Việt:** Phát hiện giọng nói deepfake tiếng Việt bền vững trước công nghệ tổng hợp chưa biết và biến đổi kênh truyền bằng biểu diễn tiếng nói tự giám sát

**Tên đề tài tiếng Anh:** *Generalizable Vietnamese Audio Deepfake Detection under Unseen Synthesis Methods and Channel Transformations using Self-Supervised Speech Representations*

**Loại bài toán:** Phân loại âm thanh nhị phân — giọng thật (*bonafide*) hoặc giọng giả (*spoof/deepfake*)

**Phạm vi khuyến nghị:** Đồ án tốt nghiệp 12–14 tuần, một sinh viên, một GPU 8–16 GB VRAM hoặc Google Colab/Kaggle

---

## 1. Tóm tắt đề tài

Các hệ thống phát hiện giọng nói deepfake thường đạt kết quả tốt khi dữ liệu huấn luyện và kiểm thử được tạo bởi cùng công nghệ Text-to-Speech (TTS) hoặc Voice Conversion (VC). Tuy nhiên, hiệu năng có thể suy giảm mạnh khi hệ thống gặp người nói mới, công nghệ tạo giọng chưa xuất hiện trong quá trình huấn luyện, hoặc âm thanh đã bị nén, thêm nhiễu và truyền qua các kênh thực tế.

Đề tài xây dựng một hệ thống phát hiện deepfake tiếng Việt tập trung vào khả năng **tổng quát hóa trên tấn công chưa biết**. Hệ thống sử dụng mô hình tiếng nói tự giám sát đa ngôn ngữ XLS-R làm bộ trích xuất đặc trưng, kết hợp một đầu phân loại nhẹ. Phương pháp đề xuất sử dụng hai phiên bản của cùng một mẫu âm thanh — bản gốc và bản đã biến đổi kênh truyền — rồi tối ưu đồng thời loss phân loại và loss nhất quán. Mục tiêu là buộc mô hình nhận diện dấu vết tổng hợp ổn định, thay vì ghi nhớ người nói, TTS engine, codec hoặc điều kiện thu âm cụ thể.

Hệ thống được đánh giá bằng giao thức tách riêng người nói và công nghệ tạo giọng giữa các tập dữ liệu. Chỉ số chính là Equal Error Rate (EER) trên các tấn công đã biết và chưa biết; các chỉ số bổ sung gồm minDCF, ROC-AUC, tỷ lệ chấp nhận nhầm deepfake, Generalization Gap và Worst-group EER.

---

## 2. Vấn đề nghiên cứu

Một detector thông thường có thể học các dấu hiệu không mong muốn như:

- Danh tính hoặc chất giọng của người nói trong tập huấn luyện.
- Dấu vết riêng của một TTS engine cụ thể.
- Codec chỉ xuất hiện ở lớp giả.
- Độ dài file, độ lớn âm thanh hoặc nhiễu nền khác nhau giữa hai lớp.
- Câu đọc hoặc bản ghi gốc xuất hiện ở cả train và test.

Khi đó, kết quả kiểm thử cao không phản ánh khả năng phát hiện deepfake trong thực tế. Khoảng trống mà đề tài xử lý là:

> Làm thế nào để detector học những đặc trưng có tính khái quát của âm thanh tổng hợp, đồng thời giảm phụ thuộc vào người nói, công nghệ tạo giọng và kênh truyền đã gặp trong quá trình huấn luyện?

VSASV 2025 đã thiết kế một điều kiện đánh giá phù hợp với vấn đề này: dữ liệu huấn luyện chỉ chứa replay attack, trong khi tập kiểm thử chứa thêm TTS, VC và adversarial attack chưa xuất hiện lúc huấn luyện; người nói giữa các tập cũng được tách riêng [1].

---

## 3. Mục tiêu và phạm vi

### 3.1. Mục tiêu chính

1. Xây dựng giao thức dữ liệu tiếng Việt đánh giá riêng tấn công đã biết và chưa biết.
2. Cài đặt ít nhất hai baseline: một mô hình dựa trên đặc trưng phổ và một mô hình dựa trên biểu diễn self-supervised.
3. Đề xuất phương pháp huấn luyện nhất quán dưới biến đổi kênh truyền.
4. Đánh giá khả năng tổng quát hóa theo TTS/VC engine, người nói và điều kiện âm thanh.
5. Xây dựng demo trả về xác suất deepfake, độ tin cậy và đoạn thời gian đáng ngờ.

### 3.2. Ngoài phạm vi

- Không xác minh danh tính người nói.
- Không truy tìm chính xác TTS engine đã tạo ra âm thanh.
- Không tuyên bố phát hiện được mọi loại deepfake.
- Không tự huấn luyện một foundation model từ đầu.
- Không sử dụng accuracy làm chỉ số duy nhất.
- Không tập trung vào deepfake video hoặc đồng bộ môi–tiếng.

---

## 4. Câu hỏi và giả thuyết nghiên cứu

### RQ1 — Mức suy giảm trên tấn công chưa biết

Hiệu năng của detector giảm bao nhiêu khi TTS/VC engine trong tập kiểm thử không xuất hiện trong tập huấn luyện?

### RQ2 — Lợi ích của biểu diễn self-supervised

Đặc trưng XLS-R có tổng quát tốt hơn LFCC/log-Mel trên deepfake tiếng Việt hay không?

### RQ3 — Lợi ích của consistency learning

Huấn luyện nhất quán giữa âm thanh gốc và âm thanh bị biến đổi có giảm EER trên codec/noise/reverberation chưa biết hay không?

### RQ4 — Đánh đổi giữa seen và unseen

Cải thiện trên tấn công chưa biết có làm suy giảm đáng kể kết quả trên tấn công đã biết không?

### Giả thuyết chính

> XLS-R kết hợp channel-consistency learning sẽ giảm EER trung bình và Worst-group EER trên unseen attacks so với XLS-R chỉ huấn luyện bằng cross-entropy, trong khi không làm EER trên seen attacks tăng quá lớn.

---

## 5. Kiến trúc hệ thống

```text
Tệp âm thanh đầu vào
        │
        ├── Chuẩn hóa mono, 16 kHz
        ├── Cắt thành cửa sổ 4 giây, bước nhảy 2 giây
        ├── Loại đoạn gần như im lặng
        │
        ▼
XLS-R 300M đóng băng hoặc fine-tune một phần
        │
        ▼
Attentive statistics pooling
        │
        ▼
MLP classifier
        │
        ├── Xác suất bonafide
        ├── Xác suất deepfake
        ├── Độ không chắc chắn
        └── Timeline các cửa sổ đáng ngờ
```

Khi huấn luyện, mỗi mẫu tạo thành hai nhánh:

```text
                         ┌── x: âm thanh gốc ────────┐
Một mẫu và nhãn y ───────┤                           ├── Classification loss
                         └── T(x): bản biến đổi ─────┘
                                      │
                                      └─────────────── Consistency loss
```

---

## 6. Dữ liệu

### 6.1. Nguồn dữ liệu ưu tiên

#### VSASV

VSASV là bộ dữ liệu tiếng Việt dành cho spoofing-aware speaker verification và spoof detection. VSASV 2025 bao gồm giọng thật cùng các dạng replay, VC, TTS và adversarial attack; thiết kế challenge nhấn mạnh khả năng tổng quát hóa trên người nói và kiểu tấn công chưa biết [1][2].

Bản VSASV công khai trên Hugging Face có dung lượng tải xuống khoảng 69 GB và tác giả lưu ý dữ liệu công khai có thể chưa đầy đủ như mô tả trong bài báo [3]. Do đó, trước khi chốt quy mô thí nghiệm cần kiểm tra:

- Khả năng truy cập file âm thanh và metadata.
- Giấy phép sử dụng.
- Trường `speaker_id`, `attack_type`, `generator_id` hoặc thông tin tương đương.
- Dung lượng sau giải nén và tốc độ đọc dữ liệu.

#### SEA-Spoof

SEA-Spoof chứa hơn 300 giờ giọng thật và giả của nhiều ngôn ngữ Nam Á/Đông Nam Á, bao gồm tiếng Việt. Bộ dữ liệu được xây dựng từ nhiều hệ thống tổng hợp mã nguồn mở và thương mại, thích hợp để bổ sung thử nghiệm cross-source hoặc cross-language [4]. Chỉ sử dụng nếu dữ liệu và giấy phép có thể truy cập hợp lệ.

#### ASVspoof 2021/5

ASVspoof có baseline, giao thức và mã tính EER/minDCF công khai [5][6]. Có thể dùng để:

- Kiểm tra code pipeline trước khi có dữ liệu tiếng Việt.
- Tiền huấn luyện hoặc so sánh cross-dataset.
- Đối chiếu cách tính chỉ số.

Không dùng kết quả ASVspoof thay thế cho đánh giá cuối cùng trên tiếng Việt.

### 6.2. Quy mô tối thiểu khả thi

Không bắt buộc sử dụng toàn bộ dữ liệu. Quy mô gợi ý:

- Train: 30–50 giờ.
- Validation: 5–10 giờ.
- Seen test: 5–10 giờ.
- Unseen test: 10–20 giờ.
- Mỗi người nói có đủ mẫu nhưng không xuất hiện ở nhiều split.
- Mỗi lớp có nhiều điều kiện thu âm và codec.

Nếu lớp giả áp đảo, sử dụng weighted sampler hoặc class-weighted loss; không nhân bản file theo cách làm rò rỉ source utterance.

### 6.3. Metadata bắt buộc

Tạo một file `metadata.csv` với tối thiểu các cột:

| Cột | Ý nghĩa |
|---|---|
| `file_id` | ID duy nhất |
| `path` | Đường dẫn file |
| `label` | `bonafide` hoặc `spoof` |
| `speaker_id` | ID người nói |
| `source_id` | ID câu/bản ghi gốc trước khi tạo giả |
| `attack_type` | `tts`, `vc`, `replay`, `adversarial` hoặc `bonafide` |
| `generator_id` | TTS/VC engine; để `none` cho bonafide |
| `corpus` | Nguồn dữ liệu |
| `codec` | WAV/MP3/AAC/Opus... |
| `condition` | clean/noise/reverb/telephone... |
| `duration` | Độ dài giây |
| `split` | train/dev/seen_test/unseen_test |

Nếu dữ liệu không cung cấp `generator_id`, không được tự khẳng định giao thức unseen-engine. Khi đó, chuyển mục tiêu thành **unseen attack type** hoặc **unseen corpus/channel**, tùy metadata thực tế.

---

## 7. Giao thức chia dữ liệu chống rò rỉ

### 7.1. Giao thức chính

| Tập | Người nói | Generator | Mục đích |
|---|---|---|---|
| Train | Nhóm A | G1, G2, G3 | Huấn luyện |
| Validation | Nhóm B | G1, G2, G3 | Chọn mô hình và threshold |
| Seen test | Nhóm C | G1, G2, G3 | Đánh giá tấn công đã biết |
| Unseen test | Nhóm D | G4, G5, G6 | Đánh giá tổng quát hóa |

### 7.2. Kiểm tra rò rỉ bắt buộc

Viết script tự động và làm chương trình dừng nếu vi phạm:

1. `speaker_id` không giao nhau giữa train, validation và test.
2. `generator_id` của unseen test không xuất hiện trong train/validation.
3. `source_id` không giao nhau giữa các split.
4. Hash của file âm thanh không trùng giữa các split.
5. Bản gốc và các bản augmentation của nó nằm trong cùng một split.
6. Không dùng test set để chọn threshold, epoch hoặc siêu tham số.

### 7.3. Leave-one-generator-out

Nếu có ít nhất bốn generator, chạy thêm thí nghiệm:

- Mỗi lần giữ lại một generator làm unseen test.
- Huấn luyện trên các generator còn lại.
- Báo cáo trung bình, độ lệch chuẩn và generator khó nhất.

Giao thức này đáng tin cậy hơn một lần chia cố định vì kết quả không phụ thuộc vào việc vô tình chọn một generator quá dễ.

---

## 8. Tiền xử lý và tăng cường dữ liệu

### 8.1. Tiền xử lý

- Đọc audio bằng `torchaudio` hoặc `soundfile`.
- Chuyển về một kênh.
- Resample về 16 kHz.
- Cắt hoặc pad thành 4 giây, tương đương 64.000 samples.
- Khi train: random crop; khi test: sliding window 4 giây, hop 2 giây.
- Loại cửa sổ có tỷ lệ im lặng quá cao bằng energy threshold hoặc VAD.
- Không thực hiện denoise mạnh vì có thể xóa cả dấu vết deepfake.
- Không chuẩn hóa hai lớp bằng các quy trình khác nhau.

### 8.2. Channel augmentation

Mỗi augmentation phải được áp dụng ngẫu nhiên cho **cả bonafide và spoof**:

- MP3: 32/64/96/128 kbps.
- Opus: 16/32/64 kbps.
- Resampling: 16 kHz → 8 kHz → 16 kHz.
- Telephone band-pass khoảng 300–3400 Hz.
- Additive noise ở SNR 5/10/20 dB.
- Room impulse response.
- Random gain.
- Clipping nhẹ.
- Speed perturbation 0,9×/1,0×/1,1×.

Chia augmentation thành hai nhóm:

- **Seen transformations:** dùng khi train, ví dụ MP3 và noise.
- **Unseen transformations:** chỉ dùng khi test, ví dụ Opus và telephone band-pass.

Nhờ đó có thể đánh giá mô hình học bất biến thật hay chỉ ghi nhớ loại augmentation.

---

## 9. Các mô hình

### B0 — LFCC + LCNN

Baseline nhẹ dựa trên đặc trưng phổ:

- Đầu vào: LFCC hoặc log-Mel spectrogram.
- Backbone: LCNN hoặc ResNet18 nhỏ.
- Loss: weighted cross-entropy.
- Vai trò: xác định mức hiệu năng của phương pháp truyền thống.

### B1 — AASIST

AASIST xử lý đặc trưng thời gian–phổ bằng graph attention và là baseline phổ biến cho anti-spoofing. Mã chính thức cho biết khoảng 16 GB VRAM khi huấn luyện batch size 24; có thể giảm batch xuống 4–8 trên GPU nhỏ hơn [7].

### B2 — Frozen XLS-R + classifier

- Encoder: `facebook/wav2vec2-xls-r-300m`.
- XLS-R là mô hình 300 triệu tham số, đa ngôn ngữ và nhận audio 16 kHz [8].
- Đóng băng toàn bộ encoder ở thí nghiệm đầu tiên.
- Lấy hidden states theo thời gian.
- Attentive statistics pooling: ghép weighted mean và weighted standard deviation.
- Classifier: `Linear → GELU → Dropout → Linear(2)`.

### M — Phương pháp đề xuất

Giữ kiến trúc B2, bổ sung channel-consistency learning và có thể fine-tune 2–4 block cuối của XLS-R ở giai đoạn sau.

---

## 10. Hàm mất mát đề xuất

Với mẫu gốc `x`, nhãn `y` và phép biến đổi ngẫu nhiên `T`, mô hình tạo:

- `p = f(x)`
- `p_aug = f(T(x))`

Loss tổng:

```text
L_total = L_cls(x, y)
        + L_cls(T(x), y)
        + λ_cons · L_cons(p, p_aug)
```

Trong đó:

- `L_cls`: weighted cross-entropy hoặc focal loss.
- `L_cons`: Jensen–Shannon divergence giữa hai phân phối đầu ra; có thể bắt đầu bằng MSE giữa logits để triển khai đơn giản.
- `λ_cons`: thử `{0.1, 0.5, 1.0}` và chọn trên validation set.

Biến thể nâng cao nếu còn thời gian:

```text
L_total = L_cls + λ_cons L_cons + λ_ctr L_supervised_contrastive
```

Không thêm contrastive loss trước khi pipeline chính và các baseline hoạt động ổn định.

---

## 11. Thiết lập huấn luyện khuyến nghị

### Cấu hình mặc định

- Framework: PyTorch + Transformers + torchaudio.
- Seed: chạy tối thiểu `42`, `123`, `2026` cho mô hình cuối.
- Optimizer cho classifier: AdamW.
- Learning rate classifier: bắt đầu `3e-4`.
- Learning rate XLS-R khi fine-tune một phần: bắt đầu `1e-5`.
- Weight decay: `1e-4` hoặc `1e-2`, chọn trên validation.
- Batch size thực: 4–8.
- Gradient accumulation: 2–8 bước.
- Epoch: tối đa 30.
- Early stopping: patience 5 theo validation EER.
- Scheduler: cosine decay với warm-up 5–10% tổng số bước.
- Precision: AMP FP16/BF16 nếu GPU hỗ trợ [9].
- Gradient clipping: 1,0.

### Theo mức GPU

| VRAM | Cách triển khai |
|---|---|
| 4–6 GB | B0; hoặc trích xuất embedding XLS-R offline từng file |
| 8 GB | XLS-R đóng băng, batch 2–4, audio 3–4 giây, AMP |
| 12 GB | XLS-R đóng băng hoặc fine-tune 1–2 block cuối |
| 16 GB | AASIST thuận lợi; fine-tune một phần XLS-R |
| 24 GB+ | Có thể thử fine-tune toàn bộ XLS-R 300M, nhưng không cần thiết |

### Chế độ tiết kiệm GPU

1. Chạy XLS-R bằng `eval()` và `torch.no_grad()`.
2. Pool hidden states thành vector kích thước cố định.
3. Lưu embedding của bản clean và các bản augmented.
4. Huấn luyện classifier cùng consistency loss trên embedding.
5. Chỉ sau khi có kết quả ổn định mới fine-tune một phần encoder.

---

## 12. Ma trận thí nghiệm

| ID | Encoder/đặc trưng | Augmentation | Consistency | Mục tiêu |
|---|---|---:|---:|---|
| B0 | LFCC + LCNN | Không | Không | Baseline phổ |
| B0-A | LFCC + LCNN | Có | Không | Ảnh hưởng augmentation |
| B1 | AASIST | Theo cấu hình gốc | Không | Baseline anti-spoofing |
| B2 | XLS-R frozen | Không | Không | Baseline SSL |
| M1 | XLS-R frozen | Có | Không | SSL + augmentation |
| M2 | XLS-R frozen | Có | Có | Phương pháp chính |
| M3 | XLS-R partial fine-tune | Có | Có | Phương pháp đầy đủ |

### Ablation study bắt buộc

1. M2 không có MP3/Opus augmentation.
2. M2 không có noise/reverb augmentation.
3. M2 đặt `λ_cons = 0`.
4. So sánh mean pooling và attentive statistics pooling.
5. So sánh frozen encoder và fine-tune 2 block cuối.
6. Nếu dùng SAM, so sánh cùng cấu hình có/không có SAM. Sharpness-Aware Minimization đã được báo cáo là cải thiện độ ổn định trên các tập deepfake chưa biết [10].

---

## 13. Chỉ số đánh giá

### 13.1. Chỉ số chính

#### Equal Error Rate

EER là điểm mà False Acceptance Rate bằng False Rejection Rate. EER càng thấp càng tốt. Đây cũng là chỉ số chính của VSASV [2].

Báo cáo tối thiểu:

- `EER_seen`
- `EER_unseen_generator`
- `EER_unseen_channel`
- `EER_replay`
- `EER_TTS`
- `EER_VC`
- EER theo từng generator

#### Generalization Gap

```text
Generalization Gap = EER_unseen − EER_seen
```

Gap càng nhỏ càng tốt, nhưng phải báo cáo cùng EER tuyệt đối để tránh diễn giải sai.

#### Worst-group EER

```text
Worst-group EER = max(EER của từng generator/condition)
```

Chỉ số này phản ánh nhóm tấn công mà hệ thống xử lý kém nhất.

### 13.2. Chỉ số bổ sung

- minDCF hoặc actDCF theo giao thức ASVspoof 5 [6].
- ROC-AUC.
- Precision, recall và F1 tại threshold chọn trên validation.
- FAR của lớp deepfake tại một hoặc nhiều operating point.
- Expected Calibration Error hoặc Brier score.
- Thời gian suy luận trên một phút audio.
- Số tham số được huấn luyện và peak GPU memory.

Không chọn threshold trên test set. Ngoài EER, mọi chỉ số phụ thuộc threshold phải sử dụng threshold đã cố định từ validation set.

---

## 14. Suy luận và demo

### 14.1. Suy luận theo cửa sổ

- Window: 4 giây.
- Hop: 2 giây.
- Mỗi cửa sổ trả về xác suất deepfake.
- Điểm toàn file: trung bình của top-k cửa sổ hoặc attention-weighted average.
- Với file ngắn hơn 4 giây: pad hoặc dùng độ dài thật cùng attention mask.

### 14.2. Ba vùng kết luận

Threshold được hiệu chỉnh trên validation set, ví dụ:

- `p < τ_low`: có khả năng là giọng thật.
- `τ_low ≤ p ≤ τ_high`: không chắc chắn.
- `p > τ_high`: có khả năng là deepfake.

Các giá trị `τ_low` và `τ_high` phải được chọn theo yêu cầu FAR/FRR, không đặt tùy ý.

### 14.3. Giao diện tối thiểu

- Upload WAV/MP3.
- Hiển thị waveform.
- Xác suất toàn file.
- Timeline xác suất theo cửa sổ.
- Cảnh báo “kết quả không phải bằng chứng pháp lý”.
- Hiển thị điều kiện ngoài phân phối nếu mô hình không chắc chắn.

Demo chỉ là sản phẩm minh họa; kết quả nghiên cứu nằm ở giao thức và thí nghiệm.

---

## 15. Phân tích lỗi

Chọn tối thiểu 100 mẫu dự đoán sai và phân loại nguyên nhân:

- Giọng thật thu trong studio quá sạch.
- Giọng thật bị codec mạnh.
- Deepfake chất lượng cao.
- Đoạn quá ngắn.
- Nhiều khoảng im lặng.
- Nhạc hoặc tiếng nền lớn.
- Giọng vùng miền hoặc chất giọng ít có trong train.
- Partially spoofed: chỉ một đoạn ngắn bị thay thế.
- Model dựa vào codec/nhiễu thay vì dấu vết tổng hợp.

Trực quan hóa bằng:

- Spectrogram của các trường hợp tiêu biểu.
- UMAP/t-SNE embedding, tô màu theo label, generator và codec.
- Biểu đồ EER theo generator/condition.
- Reliability diagram cho calibration.

Không dùng t-SNE làm bằng chứng duy nhất về khả năng phân tách; đây chỉ là công cụ mô tả.

---

## 16. Tiêu chí hoàn thành

### Mức tối thiểu đạt yêu cầu

- Có pipeline dữ liệu tái lập được.
- Có script kiểm tra rò rỉ.
- Chạy được B0 và B2.
- Có seen test và unseen test đúng giao thức.
- Báo cáo EER, Generalization Gap và kết quả theo nhóm.
- Có ít nhất một ablation study.
- Có demo hoặc script suy luận một file.

### Mức tốt

- Chạy đủ B0, B1, B2, M1 và M2.
- M2 cải thiện EER trung bình hoặc Worst-group EER trên unseen test một cách ổn định qua ba seed.
- Có đánh giá unseen codec và unseen generator.
- Có calibration và vùng “không chắc chắn”.
- Có phân tích lỗi định tính và định lượng.

### Mức xuất sắc

- Leave-one-generator-out trên nhiều generator.
- Fine-tune một phần XLS-R hoặc thêm SAM.
- Phát hiện partially spoofed theo timeline.
- Công bố code, split protocol và metadata đã ẩn thông tin nhạy cảm.
- Viết báo cáo theo cấu trúc một bài nghiên cứu có thể gửi workshop sinh viên/VLSP.

Kết quả âm không đồng nghĩa đề tài thất bại. Nếu consistency learning không cải thiện, một phân tích nghiêm túc chỉ ra điều kiện thất bại và nguyên nhân rò rỉ/tổng quát hóa vẫn là đóng góp có giá trị.

---

## 17. Cấu trúc mã nguồn đề xuất

```text
vietnamese-deepfake-detection/
├── README.md
├── requirements.txt
├── configs/
│   ├── lfcc_lcnn.yaml
│   ├── xlsr_frozen.yaml
│   └── xlsr_consistency.yaml
├── data/
│   ├── metadata.csv
│   └── splits/
│       ├── train.csv
│       ├── dev.csv
│       ├── seen_test.csv
│       └── unseen_test.csv
├── scripts/
│   ├── prepare_metadata.py
│   ├── make_splits.py
│   ├── check_leakage.py
│   ├── extract_embeddings.py
│   └── evaluate.py
├── src/
│   ├── datasets.py
│   ├── augmentations.py
│   ├── features.py
│   ├── models/
│   │   ├── lcnn.py
│   │   └── xlsr_detector.py
│   ├── losses.py
│   ├── train.py
│   └── inference.py
├── tests/
│   ├── test_splits.py
│   ├── test_augmentations.py
│   └── test_metrics.py
├── app/
│   └── demo.py
└── reports/
    ├── figures/
    └── experiment_table.csv
```

Mọi run phải lưu:

- Git commit hoặc phiên bản code.
- File cấu hình.
- Seed.
- Danh sách split.
- Validation/test metrics.
- Checkpoint tốt nhất.
- Peak VRAM và thời gian huấn luyện.

---

## 18. Kế hoạch 14 tuần

| Tuần | Công việc | Đầu ra |
|---:|---|---|
| 1 | Chốt phạm vi, đọc VSASV/ASVspoof/AASIST/XLS-R | Đề cương và bảng tài liệu |
| 2 | Tải dữ liệu, kiểm tra license, metadata và dung lượng | Báo cáo dữ liệu khả dụng |
| 3 | Chuẩn hóa metadata, thiết kế split | `metadata.csv`, các file split |
| 4 | Viết leakage checker và metric EER | Unit tests chạy thành công |
| 5 | Cài B0 LFCC + LCNN | Kết quả baseline B0 |
| 6 | Cài XLS-R frozen + pooling + classifier | Kết quả B2 |
| 7 | Xây augmentation pipeline | Kiểm thử nghe/đo các bản biến đổi |
| 8 | Chạy M1: XLS-R + augmentation | Seen/unseen metrics M1 |
| 9 | Cài consistency loss, chạy M2 | Seen/unseen metrics M2 |
| 10 | Ablation study | Bảng so sánh thành phần |
| 11 | Leave-one-generator-out hoặc unseen codec | Bảng generalization đầy đủ |
| 12 | Calibration và phân tích lỗi | Reliability/error figures |
| 13 | Xây demo, đo latency và VRAM | Demo và benchmark tài nguyên |
| 14 | Hoàn thiện báo cáo, slide và tái lập kết quả | Bản nộp cuối |

### Kế hoạch dự phòng

Nếu dữ liệu hoặc GPU bị chậm, bỏ theo thứ tự:

1. Bỏ fine-tune toàn bộ XLS-R.
2. Bỏ SAM/contrastive loss.
3. Giảm leave-one-generator-out xuống một unseen split cố định.
4. Giảm dữ liệu nhưng giữ nguyên nguyên tắc tách speaker/generator/source.

Không bỏ leakage checker, unseen test và baseline.

---

## 19. Rủi ro và cách giảm thiểu

| Rủi ro | Cách xử lý |
|---|---|
| Không truy cập đủ VSASV | Dùng subset công khai; xin quyền từ tác giả; chuyển sang unseen corpus/channel với tuyên bố phạm vi rõ ràng |
| Thiếu `generator_id` | Không tuyên bố unseen-generator; đánh giá theo attack type hoặc corpus |
| GPU hết bộ nhớ | Frozen encoder, AMP, batch nhỏ, gradient accumulation, embedding offline |
| Dữ liệu quá lớn | Subset theo speaker/generator có kiểm soát; không chọn ngẫu nhiên từng file |
| Mô hình học codec | Áp dụng cùng augmentation cho cả thật và giả; đánh giá codec chéo |
| Kết quả quá cao bất thường | Kiểm tra speaker/source/hash leakage và phân bố duration/codec |
| Kết quả unseen rất thấp | Phân tích theo generator, thử augmentation/consistency và calibration; không chỉnh trên test |
| Nhãn không cân bằng | Weighted loss, sampler và báo cáo theo nhóm |
| Demo bị hiểu là kết luận tuyệt đối | Hiển thị xác suất, vùng không chắc chắn và cảnh báo giới hạn |

---

## 20. Đạo đức và sử dụng có trách nhiệm

- Chỉ sử dụng dữ liệu có quyền nghiên cứu phù hợp.
- Không công bố dữ liệu giọng nói cá nhân nếu giấy phép không cho phép.
- Nếu tự tạo deepfake, chỉ dùng giọng có sự đồng ý hoặc dữ liệu được cấp phép.
- Không xây chức năng giả mạo danh tính hoặc phát hành model tạo giọng trong phạm vi đề tài.
- Trình bày detector như công cụ hỗ trợ sàng lọc, không phải bằng chứng pháp lý.
- Báo cáo false positive trên giọng vùng miền, giới tính và điều kiện thu âm nếu metadata cho phép.

---

## 21. Cấu trúc báo cáo tốt nghiệp

1. **Giới thiệu:** bối cảnh, vấn đề, khoảng trống và đóng góp.
2. **Cơ sở lý thuyết:** TTS, VC, vocoder, anti-spoofing, SSL speech model, domain generalization.
3. **Công trình liên quan:** LFCC/LCNN, AASIST, XLS-R, consistency learning, unseen attack evaluation.
4. **Dữ liệu và giao thức:** nguồn dữ liệu, metadata, split và leakage prevention.
5. **Phương pháp:** kiến trúc, augmentation, loss và quy trình huấn luyện.
6. **Thực nghiệm:** baseline, ma trận thí nghiệm, metric và cấu hình phần cứng.
7. **Kết quả và phân tích:** seen/unseen, ablation, calibration, error analysis.
8. **Demo và triển khai:** suy luận theo cửa sổ, latency và giới hạn.
9. **Kết luận:** câu trả lời cho từng RQ, hạn chế và hướng phát triển.

---

## 22. Công việc cần làm trong 7 ngày đầu

### Ngày 1

- Đọc bài VSASV 2025.
- Chốt Task 2: Vietnamese Spoof Detection.
- Tạo bảng thuật ngữ: bonafide, spoof, TTS, VC, replay, EER.

### Ngày 2

- Kiểm tra khả năng tải VSASV.
- Xem metadata có `speaker_id`, `attack_type`, `generator_id`, `source_id` hay không.
- Ước lượng dung lượng ổ đĩa.

### Ngày 3

- Chọn subset thử nghiệm khoảng 1–2 giờ.
- Viết script load, resample và kiểm tra duration/sample rate.

### Ngày 4

- Định nghĩa schema `metadata.csv`.
- Viết `make_splits.py` theo speaker và generator.

### Ngày 5

- Viết `check_leakage.py`.
- Kiểm tra speaker/source/hash overlap.

### Ngày 6

- Cài metric EER.
- Tạo một classifier LFCC/log-Mel nhỏ để kiểm tra pipeline end-to-end.

### Ngày 7

- Viết báo cáo một trang gồm dữ liệu thực tế có thể dùng, GPU có sẵn, giao thức chia và baseline đầu tiên.
- Chỉ sau bước này mới chốt phiên bản cuối của tên đề tài với giảng viên.

---

## 23. Sản phẩm bàn giao

1. Mã nguồn và file môi trường.
2. Metadata và split protocol có thể tái lập.
3. Leakage checker.
4. Checkpoint baseline và mô hình đề xuất.
5. Bảng kết quả seen/unseen và ablation.
6. Biểu đồ EER, calibration, lỗi theo generator/codec.
7. Demo phân tích một file và timeline đoạn đáng ngờ.
8. Báo cáo đồ án, slide và hướng dẫn chạy.

---

## 24. Tài liệu tham khảo ban đầu

[1] Dat, P. T., Vu, H. L., & Trang, N. T. T. (2025). [The Vietnamese Spoofing-aware Speaker Verification Challenge 2025: Summary and Results](https://aclanthology.org/2025.vlsp-1.10/). VLSP 2025.

[2] [VLSP 2025 Challenge on Vietnamese Spoofing-Aware Speaker Verification](https://vlsp.org.vn/vlsp2025/eval/vsasv).

[3] [VSASV Dataset — Hugging Face](https://huggingface.co/datasets/hustep-lab/VSASV-Dataset).

[4] Wu, J. et al. (2025). [SEA-Spoof: Bridging The Gap in Multilingual Audio Deepfake Detection for South-East Asian](https://arxiv.org/abs/2509.19865).

[5] [ASVspoof 2021: datasets, protocols and baselines](https://www.asvspoof.org/index2021.html).

[6] [ASVspoof 5 Evaluation Plan](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf).

[7] Jung, J. et al. [AASIST official implementation](https://github.com/clovaai/aasist).

[8] Meta AI. [Wav2Vec2 XLS-R 300M model card](https://huggingface.co/facebook/wav2vec2-xls-r-300m).

[9] PyTorch. [Automatic Mixed Precision documentation](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

[10] Huang, W. et al. (2025). [From Sharpness to Better Generalization for Speech Deepfake Detection](https://www.isca-archive.org/interspeech_2025/huang25e_interspeech.html). Interspeech 2025.

---

## 25. Phiên bản phát biểu ngắn với giảng viên

> Em muốn nghiên cứu khả năng tổng quát hóa của hệ thống phát hiện giọng nói deepfake tiếng Việt. Thay vì chia dữ liệu ngẫu nhiên, em sẽ tách riêng người nói và công nghệ TTS/VC giữa train và test. Baseline gồm LFCC-LCNN, AASIST và XLS-R đóng băng. Phương pháp đề xuất dùng consistency learning giữa âm thanh gốc và phiên bản bị nén hoặc thêm nhiễu để giảm phụ thuộc vào kênh truyền. Hệ thống được đánh giá chủ yếu bằng EER trên seen và unseen attacks, Generalization Gap, Worst-group EER và ablation study. Sản phẩm cuối gồm mô hình, giao thức dữ liệu chống rò rỉ, báo cáo thực nghiệm và demo hiển thị xác suất cùng đoạn thời gian đáng ngờ.
