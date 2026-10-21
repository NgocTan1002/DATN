# Phân tích tổng thể tiến độ và ý nghĩa các hạng mục của đồ án

> Cập nhật đến ngày 20/10/2026. Tài liệu này giải thích toàn bộ những gì dự án đã thực hiện, ý nghĩa của từng hạng mục trong chuỗi nghiên cứu và phần việc còn lại để hoàn thành đồ án. Trạng thái vận hành mới nhất vẫn được quản lý tại [current_state.md](current_state.md); lịch sử quyết định chính thức nằm tại [decisions.md](decisions.md).

## 1. Kết luận điều hành

Dự án đã hoàn thành phần nền móng khó và có rủi ro cao nhất: khóa phạm vi bài toán, kiểm tra dữ liệu, tạo giao thức speaker-disjoint, phát hiện và xử lý rò rỉ nội dung trong tập development, chuẩn hóa audio, xây dựng loader, metric, baseline LFCC + LCNN, checkpoint có thể tiếp tục chính xác và toàn bộ quy trình Kaggle CLI riêng tư. Benchmark đại diện trên Kaggle đã cho phép khóa ngân sách chung 896 optimizer step cho ba pilot amplitude.

Điều này có nghĩa dự án đã chuyển từ giai đoạn **xây dựng và kiểm chứng hạ tầng nghiên cứu** sang giai đoạn **thực hiện thí nghiệm khoa học**. Ba pilot `none`, `peak`, `rms_dbfs` chưa chạy, amplitude policy chưa được chọn, B0 đầy đủ chưa hoàn thành, XLS-R chưa được huấn luyện và chưa có kết quả test cuối. Vì vậy hiện tại có thể khẳng định pipeline đủ điều kiện để bắt đầu thí nghiệm có kiểm soát, nhưng chưa thể khẳng định mô hình nào tốt nhất hoặc hiệu năng cuối của hệ thống.

## 2. Mục tiêu cuối cùng và đường đi của toàn bộ dự án

Tên đề tài đã khóa là **“Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần”**. Bài toán đầu ra là phân loại nhị phân:

- `0 = bonafide`: tiếng nói thật;
- `1 = spoof`: tiếng nói giả mạo;
- `voice_conversion`, `adversarial_attack` và `replay` đều được gộp vào lớp `spoof`;
- `utt_type` chỉ dùng để thống kê và phân tích lỗi, không phải nhãn dự đoán.

Chuỗi hoàn chỉnh của đồ án được thiết kế như sau:

```text
Dữ liệu VSASV
  → audit metadata và audio
  → split speaker-disjoint
  → loại rò rỉ nội dung trong development
  → chuẩn hóa mono 16 kHz / đoạn 4 giây / amplitude policy
  → DataLoader và LFCC
  → baseline LCNN B0
  → chọn amplitude policy trên closed_dev
  → huấn luyện B0 đầy đủ
  → XLS-R đóng băng X0
  → XLS-R fine-tune từng phần X1
  → khóa mô hình và threshold bằng closed_dev
  → đánh giá closed_test đúng một lần theo giao thức
  → phân tích lỗi, demo, báo cáo và bảo vệ
```

Ý nghĩa của cấu trúc này là mỗi kết quả ở cuối chuỗi chỉ đáng tin khi tất cả cổng trước đó đã đạt. Một mô hình có EER thấp nhưng dùng split rò rỉ, sample rate không đồng nhất hoặc threshold lấy từ test sẽ không còn giá trị khoa học.

## 3. Phân tích theo từng giai đoạn đã thực hiện

### 3.1. Khóa phạm vi và câu hỏi nghiên cứu

**Đã làm được**

- Chọn VSASV làm corpus chính cho phiên bản đầu.
- Thu hẹp bài toán về phát hiện nhị phân bonafide/spoof.
- Chọn LFCC + LCNN làm baseline, XLS-R đóng băng làm đối chiếu và XLS-R fine-tune từng phần làm mô hình chính.
- Giới hạn đúng tuyên bố khoa học: không tuyên bố tái lập official VSASV và không gọi dữ liệu là “unseen TTS engines” khi metadata không cung cấp bằng chứng đó.
- Loại channel consistency, nhận diện kiểu tấn công và dataset ngoài miền khỏi phần bắt buộc.

**Ý nghĩa trong toàn dự án**

Đây là bước biến một ý tưởng rộng thành một đề tài có thể hoàn thành và bảo vệ. Việc khóa đầu ra nhị phân giúp toàn bộ data loader, loss, metric, mô hình và demo dùng chung một hợp đồng. Việc giới hạn tuyên bố ngăn báo cáo cuối vượt quá bằng chứng thực tế của dữ liệu.

**Bằng chứng**

- Các quyết định D001–D003, D008, D010 và D011 trong [decisions.md](decisions.md).
- Phạm vi hiện hành trong [README.md](../README.md).

### 3.2. Audit metadata và xây dựng giao thức split

**Đã làm được**

- Audit metadata VSASV với 220.963 hàng và 1.141 speaker; kiểm tra schema, giá trị thiếu, dòng trùng và tính nhất quán nhãn/đường dẫn.
- Tạo tám split closed/open bằng seed `2026` và cách sắp xếp ổn định dựa trên SHA-256.
- Khóa giao thức chính `closed_train/dev/test`, speaker-disjoint.
- Xây dựng leakage checker cho speaker và file path.

**Ý nghĩa trong toàn dự án**

Split speaker-disjoint ngăn cùng một người nói xuất hiện ở cả train và evaluation. Nếu không có cổng này, mô hình có thể nhận diện đặc trưng người nói thay vì dấu hiệu giả mạo. Seed và quy tắc hash ổn định giúp tái tạo đúng split ở máy khác hoặc trong một phiên làm việc khác.

**Bằng chứng**

- D006 trong [decisions.md](decisions.md).
- [split_summary.md](../reports/split_summary.md).
- [leakage_check.md](../reports/leakage_check.md).
- Giao thức chi tiết tại [split_protocol.md](split_protocol.md).

### 3.3. Kiểm kê audio thực tế và xác định giới hạn của snapshot

**Đã làm được**

- Kiểm kê 68 shard cục bộ, 34.782 waveform và khoảng 9,85 GiB; toàn bộ file quan sát được khớp metadata.
- Kiểm tra kích thước, SHA-256, khả năng đọc Parquet, sample rate và waveform mẫu.
- Ghi nhận phiên bản phát hành hiện có không tương đương đầy đủ với số liệu trong bài báo VSASV gốc.
- Xác nhận từ tác giả rằng bản Hugging Face là phiên bản chính thức còn được phát hành và phần dữ liệu đã mất không thể khôi phục.

**Ý nghĩa trong toàn dự án**

Dự án biết chính xác đang huấn luyện trên dữ liệu nào, có bao nhiêu shard và bao nhiêu mẫu thật sự truy cập được. Điều này ngăn việc báo cáo sai rằng mô hình đã dùng toàn bộ corpus gốc. Đây cũng là cơ sở để lựa chọn development subset 20.000 mẫu phù hợp với tài nguyên hiện có.

**Bằng chứng**

- D015 trong [decisions.md](decisions.md).
- [local_storage_audit.md](../reports/local_storage_audit.md).
- [audio_smoke_test.md](../reports/audio_smoke_test.md).
- [vsasv_snapshot_verification.md](../reports/vsasv_snapshot_verification.md).

### 3.4. Khóa giao thức tiền xử lý audio

**Đã làm được**

- Chuẩn hóa waveform về mono 16 kHz.
- Dùng đoạn 4 giây, tương đương 64.000 mẫu.
- Train dùng random crop xác định; evaluation dùng center crop; waveform ngắn dùng repeat-then-trim.
- Khóa ba ứng viên amplitude policy:
  - `none`;
  - `peak` với đích 0,95;
  - `rms_dbfs` với đích -25 dBFS.
- Không khuếch đại đoạn gần im lặng dưới -50 dBFS; RMS gain bị giới hạn để peak không vượt 0,95; không hard clipping.
- Cả ba policy đã vượt smoke trên train và development.

**Ý nghĩa trong toàn dự án**

Audio nguồn có sample rate khác nhau giữa các nhóm. Nếu không resample thống nhất, mô hình có thể học sample rate thay vì học dấu hiệu spoof. Tương tự, biên độ có thể trở thành shortcut. Ba policy được định nghĩa trước và áp dụng giống nhau cho mọi nhãn giúp phép so sánh sau này công bằng.

**Bằng chứng**

- D007, D009 và D012 trong [decisions.md](decisions.md).
- Cấu hình khóa tại [audio.json](../configs/audio.json).
- Mô tả tại [audio_protocol.md](audio_protocol.md).
- [amplitude_audit.md](../reports/amplitude_audit.md).

### 3.5. Phát hiện rò rỉ nội dung và thay manifest v1 bằng v2

**Đã làm được**

- Tạo `development-20k-v1` gồm 15.885 train và 4.115 development.
- Audit waveform phát hiện 91 nhóm/182 file bonafide trùng nội dung trong 34.782 waveform cục bộ.
- Phát hiện manifest v1 giữ nhiều file của 38 nhóm trùng và có 17 nhóm đi qua train/development.
- Thiết kế quy tắc D017: băm sample rate cùng toàn bộ waveform, chọn một đại diện ổn định cho mỗi content hash, quản lý hash trên toàn train + development và không đổi quota.
- Tạo `development-20k-v2` với đúng 20.000 mẫu, 184 speaker và 66 shard.
- Audit độc lập đọc lại 20.000/20.000 waveform, xác nhận 20.000 content hash duy nhất và không còn duplicate trong hoặc xuyên train/development.

**Ý nghĩa trong toàn dự án**

Đây là một trong các đóng góp quan trọng nhất về tính đúng đắn. Speaker và file path disjoint chưa đủ nếu hai file khác tên chứa cùng waveform. Manifest v2 loại một nguồn rò rỉ trực tiếp có thể làm EER thấp giả tạo. Vì vậy các pilot và mô hình về sau có cơ sở khoa học tốt hơn manifest v1.

**Bằng chứng**

- D016 và D017 trong [decisions.md](decisions.md).
- [development_manifest_20k_v2.md](../reports/development_manifest_20k_v2.md).
- [development_manifest_20k_v2_content_audit.md](../reports/development_manifest_20k_v2_content_audit.md).

### 3.6. Xây dựng DataLoader production

**Đã làm được**

- Xây dựng loader đọc lười waveform từ Parquet.
- Xây dựng `VSASVManifestDataset` đọc trực tiếp manifest v2 qua cột `shard`, không quét toàn bộ Parquet để dựng chỉ mục khi khởi tạo.
- Kiểm tra schema, manifest version, source snapshot, file trùng, speaker-disjoint, nhãn, shard và metadata/sample rate khi đọc.
- Batch trả đúng waveform `float32` 64.000 mẫu cùng label, file, speaker, `utt_type`, sample rate và các trường audit biên độ.
- Crop train tái lập theo seed/epoch; evaluation center crop xác định.
- Loader manifest đã smoke trên đủ 15.885/4.115 hàng và cả ba amplitude policy.

**Ý nghĩa trong toàn dự án**

Loader là cầu nối giữa giao thức nghiên cứu và mô hình. Các kiểm tra ở loader khiến lỗi dữ liệu dừng sớm thay vì âm thầm đi vào training. Tính xác định của crop và thứ tự batch là điều kiện để resume chính xác và so sánh công bằng giữa ba policy.

**Bằng chứng**

- Mã nguồn trong [src/data](../src/data/).
- [smoke_test_development_manifest_loader.py](../scripts/smoke_test_development_manifest_loader.py).
- [manifest_loader_benchmark.md](../reports/manifest_loader_benchmark.md).

### 3.7. Đo hiệu năng và xác định nút thắt I/O

**Đã làm được**

- Đo tách thời gian DataLoader, LFCC, model, loss/backward và optimizer.
- Xác định đọc/decode Parquet là nút thắt chính; LFCC và LCNN không phải phần chiếm thời gian lớn nhất.
- So sánh batch size 4/8/16 và worker 0/2; giữ batch 8, worker 0 vì cải thiện của worker phụ không đạt cổng 15% và tăng RAM đáng kể.
- Thử batch-read theo shard. Micro-benchmark giảm số query khi batch cùng shard, nhưng thứ tự shuffle thực tế làm mỗi batch trải trên nhiều shard và throughput end-to-end giảm 0,93%; hook production đã được gỡ.
- Xây dựng proof-of-concept packed artifact trên 24 mẫu. Kết quả cho thấy tiềm năng tăng tốc lớn, nhưng chưa được ngoại suy lên 20.000 mẫu.

**Ý nghĩa trong toàn dự án**

Dự án không tối ưu dựa trên cảm tính. Hai hướng không đạt cổng đã được loại bằng số đo; packed artifact được giữ ở trạng thái ứng viên thay vì đưa thẳng vào production. Điều này bảo vệ tính ổn định của pipeline và chỉ tiêu tài nguyên.

**Bằng chứng**

- D014 trong [decisions.md](decisions.md).
- [data_pipeline_benchmark.md](../reports/data_pipeline_benchmark.md).
- [data_pipeline_optimization.md](../reports/data_pipeline_optimization.md).
- [batch_read_benchmark.md](../reports/batch_read_benchmark.md).
- [packed_artifact_benchmark.md](../reports/packed_artifact_benchmark.md).

### 3.8. Xây dựng baseline LFCC + LCNN

**Đã làm được**

- Triển khai LFCC 60 hệ số.
- Triển khai LCNN tối thiểu gồm bốn convolution block có Max-Feature-Map, pooling, adaptive average pooling và một raw spoof logit.
- Mô hình có 498.113 tham số trainable và dùng `BCEWithLogitsLoss`.
- Kiểm tra forward, loss, backward, gradient, optimizer step, development evaluation và checkpoint restore trên CPU.

**Ý nghĩa trong toàn dự án**

B0 là mốc đối chứng bắt buộc. Nó xác nhận toàn bộ đường dữ liệu–đặc trưng–mô hình–loss hoạt động trước khi đầu tư vào XLS-R. Về sau, X0/X1 chỉ có ý nghĩa khi được so sánh với một baseline B0 chạy trên cùng giao thức.

**Bằng chứng**

- D013 trong [decisions.md](decisions.md).
- Cấu hình [lfcc_lcnn.json](../configs/lfcc_lcnn.json).
- Mô hình [lfcc_lcnn.py](../src/models/lfcc_lcnn.py).
- [b0_smoke_training.md](../reports/b0_smoke_training.md).

### 3.9. Khóa metric, threshold và hợp đồng đánh giá

**Đã làm được**

- Triển khai EER, ROC-AUC và macro-F1.
- Khóa quy ước score cao hơn biểu thị xác suất/bằng chứng spoof.
- Development tính threshold tại điểm EER.
- Khi đánh giá test hoặc inference, macro-F1 phải dùng threshold đã khóa từ development; không tính lại threshold từ test.
- Report lưu score, label, loss, metric, số mẫu và phân bố `utt_type`.

**Ý nghĩa trong toàn dự án**

Metric contract ngăn rò rỉ thông tin từ test vào lựa chọn threshold. EER là metric chọn chính; ROC-AUC cho biết khả năng xếp hạng score; macro-F1 phản ánh chất lượng quyết định nhị phân tại threshold đã khóa. Ba góc nhìn này bổ sung cho nhau nhưng không thay thế quy tắc chọn model trên development.

**Bằng chứng**

- D019 trong [decisions.md](decisions.md).
- Mã nguồn [src/metrics](../src/metrics/).
- Kiểm thử [test_metrics.py](../tests/test_metrics.py).

### 3.10. Hoàn thiện entrypoint B0 production và exact resume

**Đã làm được**

- Xây dựng [train_b0.py](../scripts/train_b0.py) dùng chung cho `technical_smoke`, `pilot` và `full_training`.
- Tích hợp train → development → metric → report → checkpoint trong một đường chạy.
- `ResumableBatchSampler` lưu epoch và cursor của batch tiếp theo.
- Checkpoint lưu model, optimizer, global step, sampler, Python/NumPy/PyTorch/CUDA RNG và identity của run.
- Checkpoint identity version 3 khóa manifest/config/seed/run mode/policy/batch/optimizer/learning rate/step budget và exact sample selection cho technical benchmark.
- Chạy liên tục và chạy ngắt–resume cho kết quả 12/12 tensor model trùng tuyệt đối trên smoke dữ liệu thật.
- DataLoader dùng generator riêng để việc tạo iterator không làm lệch RNG của dropout.

**Ý nghĩa trong toàn dự án**

Exact resume giúp một run dài bị ngắt có thể tiếp tục đúng vị trí, không bỏ/lặp batch và không đổi quỹ đạo ngẫu nhiên. Đây là điều kiện quan trọng khi dùng Kaggle có giới hạn phiên. Identity check ngăn vô tình resume checkpoint bằng manifest, policy hoặc ngân sách khác.

**Bằng chứng**

- [training/checkpoint.py](../src/training/checkpoint.py).
- [training/sampler.py](../src/training/sampler.py).
- [b0_production_local_resumed.md](../reports/b0_production_local_resumed.md).
- Các kiểm thử [test_training_state.py](../tests/test_training_state.py) và [test_train_b0_entrypoint.py](../tests/test_train_b0_entrypoint.py).

### 3.11. Xây dựng hạ tầng Kaggle CLI riêng tư

**Đã làm được**

- Chọn Kaggle làm GPU chính; máy cá nhân dùng cho code, unit test và smoke nhỏ.
- Xây dựng workflow CLI cho doctor, đóng gói source, publish private dataset, chuẩn bị/push kernel, theo dõi và tải output.
- Source bundle có manifest và SHA-256 từng file; runner từ chối source không khớp.
- Data dataset và source dataset đều để private; internet kernel mặc định tắt.
- Runner hỗ trợ layout mount trực tiếp/lồng nhau và source ZIP/thư mục.
- Kernel runtime xác nhận Python, PyTorch CUDA, cuDNN và Tesla T4 hoạt động.
- Kernel version 7 xác nhận checkpoint production và CUDA RNG resume hoạt động thực tế.
- Snapshot nguồn cuối 48 file, SHA-256 `12d91121abac99367724d63f336951a36cd2a47075e79ed4c6d670b3c3d060db`, đã được publish riêng tư.

**Ý nghĩa trong toàn dự án**

Hạ tầng này biến thí nghiệm từ thao tác notebook thủ công thành quy trình có thể tái lập từ script của repository. Source, config, manifest, runtime và output đều có dấu vết kiểm chứng, giúp so sánh ba pilot và các mô hình sau này đáng tin hơn.

**Bằng chứng**

- D018 trong [decisions.md](decisions.md).
- [kaggle_cli_workflow.py](../scripts/kaggle_workflow.py).
- Runner [deploy/kaggle/runner.py](../deploy/kaggle/runner.py).
- [kaggle_cli_workflow.md](guides/kaggle_cli_workflow.md).
- [kaggle_runtime.json](../reports/kaggle_runtime.json).
- [b0_production_kaggle_workflow.json](../reports/b0_production_kaggle_workflow.json).

### 3.12. Benchmark đại diện và khóa ngân sách ba pilot

**Đã làm được**

- Phát hiện cách lấy các hàng đầu manifest tạo subset chỉ có bonafide và không đại diện.
- Bổ sung `balanced_hash`: chọn mẫu xác định, cân bằng nhãn và trải trên nhiều speaker/shard.
- Kernel version 8 chạy ba repeat độc lập trên cùng workload 256 train/128 development, batch 8, 32 step và policy `none`.
- Cả ba repeat hoàn thành trên Tesla T4 với cùng identity và selection SHA-256.
- Dùng thời gian xấu nhất trong ba repeat để chiếu đủ 4.115 development sample.
- Dành 20 GPU-giờ cho ba pilot cộng một run-equivalent dự phòng, sau đó làm tròn xuống bội số 64.
- Khóa D020: batch 8, worker 0, Adam, learning rate 0,001 và 896 optimizer step cho cả ba pilot.
- Tạo ba profile `b0-pilot-none`, `b0-pilot-peak`, `b0-pilot-rms-dbfs`; kiểm thử xác nhận chỉ policy và tên output khác nhau.

**Ý nghĩa trong toàn dự án**

Ngân sách pilot được quyết định bằng timing thực tế thay vì ước lượng từ CPU hoặc smoke một bước. Ba policy nhận cùng lượng tối ưu nên EER development có thể so sánh công bằng. Một run-equivalent dự phòng giúp xử lý gián đoạn mà không làm mất kế hoạch trước hạn 25/10.

**Bằng chứng**

- D020 trong [decisions.md](decisions.md).
- [b0_benchmark_summary.md](../reports/b0_benchmark_summary.md).
- Ba báo cáo [repeat 1](../reports/b0_benchmark_repeat_1.md), [repeat 2](../reports/b0_benchmark_repeat_2.md), [repeat 3](../reports/b0_benchmark_repeat_3.md).
- [kaggle_cli.example.json](../configs/kaggle_cli.example.json).

### 3.13. Kiểm thử, tài liệu và khả năng bàn giao

**Đã làm được**

- Xây dựng kiểm thử cho amplitude, segment, split, loader, manifest, duplicate content, LFCC/LCNN, metric, checkpoint, exact resume, packed artifact, Kaggle runner/workflow và benchmark summary.
- Cổng gần nhất đạt 147/147 unit test trong 112,924 giây.
- `compileall`, kiểm tra môi trường CPU, kiểm tra báo cáo tổng hợp và `git diff --check` đều đạt.
- Chuẩn hóa kế hoạch ngày/tuần, current state, decision log và hướng dẫn dùng lại.

**Ý nghĩa trong toàn dự án**

Kiểm thử không chỉ bảo vệ code mà còn bảo vệ giao thức nghiên cứu. Ví dụ, test ngăn đổi nhãn, đổi threshold contract, resume sai identity hoặc tạo ba profile pilot có ngân sách khác nhau. Tài liệu bàn giao giúp phiên làm việc sau tiếp tục đúng trạng thái thay vì dựa vào trí nhớ.

**Bằng chứng**

- Thư mục [tests](../tests/).
- [Kế hoạch tuần hiện tại](plans/weekly/2026-W43.md).
- [Kết quả ngày 20/10](plans/daily/2026-10-20.md).

## 4. Ý nghĩa tổng hợp của những việc đã hoàn thành

### 4.1. Về tính đúng đắn khoa học

Dự án đã xử lý ba nguồn sai lệch lớn trước khi báo metric:

1. **Speaker leakage:** giải quyết bằng split speaker-disjoint.
2. **Content leakage:** phát hiện waveform trùng và tạo manifest v2 không trùng train/development.
3. **Audio shortcut:** resample về 16 kHz và đưa amplitude policy vào thí nghiệm đối chứng có kiểm soát.

Nhờ đó, metric development sau này có cơ sở phản ánh khả năng phát hiện spoof tốt hơn thay vì ghi nhớ speaker, file trùng hoặc sample rate.

### 4.2. Về khả năng tái lập

Seed, split, manifest, config, checksum, selection hash, source bundle, runtime, checkpoint identity và RNG state đều được ghi lại. Một run có thể được kiểm tra đã dùng đúng dữ liệu/cấu hình hay chưa, và khi bị gián đoạn có thể tiếp tục chính xác.

### 4.3. Về tính công bằng của thí nghiệm

Ba amplitude policy sẽ dùng cùng manifest, seed, sampler, batch, optimizer, learning rate, step budget và full development. Chỉ policy thay đổi. Thiết kế này cho phép quy kết chênh lệch metric cho amplitude policy thay vì cho ngân sách hay dữ liệu khác nhau.

### 4.4. Về khả năng vận hành

Pipeline đã chạy trên CPU và Kaggle CUDA; source và data được quản lý riêng tư; checkpoint/resume đã được kiểm chứng. Nút thắt I/O đã được đo, phản ánh vào step budget và có phương án packed artifact dự phòng nếu được phê duyệt.

### 4.5. Về giá trị của đồ án khi bảo vệ

Ngoài mô hình, dự án có một chuỗi bằng chứng rõ ràng về cách bảo vệ độ tin cậy của kết quả: audit dữ liệu, chống leakage, giao thức audio, metric contract, exact resume và thực nghiệm công bằng. Đây là phần giải thích quan trọng khi hội đồng hỏi vì sao kết quả đáng tin và có thể tái lập.

## 5. Ma trận trạng thái toàn dự án

| Hạng mục | Trạng thái | Ý nghĩa hiện tại |
|---|---|---|
| Phạm vi đề tài và nhãn nhị phân | Đã khóa | Toàn bộ code và báo cáo có cùng mục tiêu |
| Metadata audit | Đạt | Metadata đủ điều kiện tạo split và manifest |
| Split speaker-disjoint | Đạt | Giảm nguy cơ mô hình ghi nhớ speaker |
| Giao thức audio 16 kHz/4 giây | Đã khóa | Loại sample-rate shortcut và thống nhất input |
| Development manifest v2 | Đạt | 20.000 mẫu, không duplicate train–development |
| DataLoader production | Đạt cổng kỹ thuật | Sẵn sàng cho thí nghiệm B0 |
| Metric và threshold contract | Đã khóa | Có quy tắc đánh giá không dùng test để chọn |
| Baseline LFCC + LCNN | Đạt cổng kỹ thuật | Forward/train/eval/checkpoint hoạt động |
| Exact resume CPU/CUDA | Đạt | Run dài có thể tiếp tục an toàn |
| Kaggle CLI riêng tư | Đạt | Có môi trường GPU tái lập |
| Benchmark và step budget | Đã khóa | Ba pilot cùng dùng 896 step |
| Pilot `none` | Chưa chạy | Bước tiếp theo trên đường găng |
| Pilot `peak` | Chưa chạy | Chờ pilot `none` hoàn tất |
| Pilot `rms_dbfs` | Chưa chạy | Chờ hai pilot trước |
| Chọn amplitude policy | Chưa thực hiện | Phải dựa trên development EER/loss |
| B0 full training | Chưa thực hiện | Chỉ chạy sau khi khóa policy |
| XLS-R đóng băng X0 | Chưa thực hiện | Cấu hình đối chiếu học sâu |
| XLS-R fine-tune từng phần X1 | Chưa thực hiện | Mô hình chính của đề tài |
| Evaluation manifest cho test | Chờ xác nhận | Cần xử lý overlap nội dung trước test cuối |
| Đánh giá closed_test | Chưa thực hiện | Chỉ làm sau khi khóa model và threshold |
| Phân tích lỗi | Chưa thực hiện | Cần theo `utt_type`, score và nhóm khó |
| Demo | Chưa thực hiện | Cần inference cửa sổ 4 giây/bước 2 giây |
| Báo cáo/slide bảo vệ cuối | Chưa hoàn thành | Phụ thuộc kết quả B0/X0/X1 và test |

## 6. Những điều chưa được phép kết luận

Tại thời điểm này, dự án **chưa có bằng chứng** để kết luận các nội dung sau:

- amplitude policy nào tốt nhất;
- EER cuối của baseline B0;
- XLS-R tốt hơn LFCC + LCNN;
- fine-tune từng phần tốt hơn XLS-R đóng băng;
- hiệu năng trên closed test;
- khả năng tổng quát hóa tới unseen TTS engines;
- kết quả tái lập official VSASV;
- packed artifact sẽ giữ mức tăng tốc PoC khi mở rộng lên 20.000 mẫu.

Các metric từ smoke hoặc benchmark kỹ thuật chỉ chứng minh pipeline hoạt động và cung cấp timing; chúng không phải kết quả khoa học cuối.

## 7. Rủi ro và phần còn mở

### 7.1. Duplicate giữa train và test

Trong 68 shard cục bộ đã thấy bốn nhóm waveform trùng đi qua `closed_train`–`closed_test`; ba file phía train thuộc manifest v2. Dự án chưa có toàn bộ test audio nên chưa thể tuyên bố đây là danh sách đầy đủ. Trước đánh giá cuối cần người dùng xác nhận phương án tạo evaluation manifest content-disjoint, không bù mẫu và không đọc metric test trong quá trình tạo.

### 7.2. Nút thắt Parquet

Benchmark Kaggle tiếp tục cho thấy loader chiếm gần toàn bộ thời gian. Ngân sách 896 step đã tính theo đường Parquet hiện hành. Packed artifact đầy đủ có thể rút ngắn đáng kể nhưng chưa được phép tạo; nếu chuyển production phải audit 20.000/20.000 waveform và benchmark lại.

### 7.3. Dữ liệu không phải toàn bộ corpus gốc

Audio cục bộ chỉ là phần có thể truy cập của snapshot công khai. Báo cáo cuối phải ghi số shard và số mẫu thực dùng, đồng thời tránh so sánh trực tiếp với số liệu bài báo khi protocol/dữ liệu không tương đương.

### 7.4. Hạn thời gian chọn amplitude policy

Ba pilot cần hoàn tất và policy phải được khóa trước 25/10/2026. Với dự báo khoảng 4,785 giờ mỗi run, cần chạy tuần tự và kiểm tra report ngay sau mỗi pilot để còn thời gian resume/chạy lại khi có lỗi.

## 8. Đường găng từ hiện tại đến khi hoàn thành đồ án

### Bước 1 — Chạy ba pilot amplitude

1. Chạy `b0-pilot-none` đủ 896 step và 4.115 development sample.
2. Kiểm tra report/checkpoint/identity trước khi chạy tiếp.
3. Lặp lại với `b0-pilot-peak` và `b0-pilot-rms-dbfs`.
4. Không thay manifest, seed, sampler, batch, optimizer, learning rate hoặc step budget giữa ba run.

**Đầu ra:** ba report hợp lệ và so sánh được.

### Bước 2 — Khóa amplitude policy

1. Xếp hạng theo development EER thấp nhất.
2. Nếu EER bằng nhau về mặt số học, dùng development loss thấp hơn.
3. Nếu vẫn bằng nhau, chọn `none`.
4. Ghi quyết định mới trong `docs/decisions.md`.

**Đầu ra:** một amplitude policy duy nhất cho các thí nghiệm tiếp theo.

### Bước 3 — Huấn luyện B0 đầy đủ

Chạy baseline với policy đã khóa. Early stopping chỉ được dùng ở giai đoạn này, checkpoint tốt nhất được chọn bằng development EER. Threshold được khóa từ development.

**Đầu ra:** baseline khoa học B0 để so sánh với XLS-R.

### Bước 4 — Triển khai và chạy X0/X1

- X0: XLS-R đóng băng, huấn luyện classifier phía trên.
- X1: fine-tune từng phần theo cấu hình được ghi rõ.
- Giữ cùng dữ liệu, preprocessing, metric và giao thức lựa chọn model.

**Đầu ra:** bằng chứng trả lời câu hỏi trung tâm của đề tài về giá trị của XLS-R và fine-tuning từng phần.

### Bước 5 — Khóa evaluation protocol và đánh giá test

Giải quyết evaluation manifest content-disjoint trước, sau đó chỉ đánh giá các cấu hình đã khóa. Threshold lấy từ development; không tối ưu lại trên test.

**Đầu ra:** bảng kết quả cuối B0/X0/X1 trên giao thức hợp lệ.

### Bước 6 — Phân tích lỗi và demo

- Phân tích theo bonafide/spoof và từng `utt_type`.
- Kiểm tra lỗi bất thường theo sample rate, biên độ, độ dài và speaker.
- Demo xử lý file dài bằng cửa sổ 4 giây, bước 2 giây và trung bình xác suất cửa sổ.

**Đầu ra:** giải thích mô hình sai ở đâu và sản phẩm minh họa có thể sử dụng.

### Bước 7 — Hoàn thiện báo cáo và bảo vệ

Tổng hợp phương pháp, giao thức, ablation amplitude, kết quả B0/X0/X1, phân tích lỗi, giới hạn dữ liệu và hướng phát triển. Mọi bảng phải dẫn từ report thực tế và ghi số shard/mẫu dùng.

## 9. Cách diễn giải mức độ hoàn thành hiện tại

Cách diễn giải đúng là:

> Dự án đã hoàn thành nền tảng dữ liệu, giao thức, baseline kỹ thuật, metric, resume và hạ tầng GPU tái lập. Ngân sách ba pilot đã được khóa bằng benchmark Kaggle. Dự án đang ở ngay trước giai đoạn thu kết quả khoa học đầu tiên; chưa có amplitude policy chiến thắng, kết quả B0 đầy đủ, kết quả XLS-R hoặc đánh giá test cuối.

Không nên diễn giải là “đồ án gần hoàn thành chỉ còn chạy mô hình”, vì các bước còn lại tạo ra kết quả khoa học chính, phân tích lỗi, demo và nội dung báo cáo bảo vệ. Tuy nhiên, rủi ro nền tảng đã giảm đáng kể: dữ liệu development đã sạch theo content hash, pipeline có kiểm thử, GPU workflow hoạt động và phép so sánh pilot đã được khóa công bằng.

## 10. Chỉ mục bằng chứng chính

| Nhóm bằng chứng | Tài liệu |
|---|---|
| Trạng thái mới nhất | [current_state.md](current_state.md) |
| Nhật ký quyết định | [decisions.md](decisions.md) |
| Metadata và split | [metadata_audit.md](../reports/metadata_audit.md), [split_summary.md](../reports/split_summary.md), [leakage_check.md](../reports/leakage_check.md) |
| Audio và amplitude | [audio_smoke_test.md](../reports/audio_smoke_test.md), [amplitude_audit.md](../reports/amplitude_audit.md) |
| Manifest v2 | [development_manifest_20k_v2.md](../reports/development_manifest_20k_v2.md), [content audit](../reports/development_manifest_20k_v2_content_audit.md) |
| Pipeline và I/O | [data_pipeline_benchmark.md](../reports/data_pipeline_benchmark.md), [batch_read_benchmark.md](../reports/batch_read_benchmark.md), [packed artifact PoC](../reports/packed_artifact_benchmark.md) |
| B0 kỹ thuật | [b0_smoke_training.md](../reports/b0_smoke_training.md), [production resume](../reports/b0_production_local_resumed.md) |
| Kaggle/CUDA | [kaggle_runtime.json](../reports/kaggle_runtime.json), [production workflow](../reports/b0_production_kaggle_workflow.json) |
| Benchmark ngân sách | [b0_benchmark_summary.md](../reports/b0_benchmark_summary.md) |
| Kế hoạch hiện hành | [2026-W43.md](plans/weekly/2026-W43.md), [2026-10-20.md](plans/daily/2026-10-20.md) |

