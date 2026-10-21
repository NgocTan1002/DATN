# Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần

Dự án xây dựng bộ phân loại nhị phân cho tiếng nói tiếng Việt: `bonafide` hoặc `spoof`. LFCC + LCNN được dùng làm mô hình cơ sở; XLS-R đóng băng và XLS-R fine-tune một phần là các cấu hình chính cần so sánh.

## Trạng thái

Dự án đã hoàn tất đường găng dữ liệu cho development subset: metadata audit đạt, tám split theo speaker được khóa với seed `2026`, leakage checker theo speaker/file path đạt và 68 shard cục bộ chứa 34.782 mẫu đã được đối chiếu kích thước, SHA-256, Parquet và metadata. Snapshot nguồn có 91 nhóm/182 file waveform bonafide trùng, nhưng development manifest `development-20k-v2` đã khử trùng nội dung trước chọn quota và giữ đúng 15.885 mẫu train cùng 4.115 mẫu development. Audit độc lập đã băm lại 20.000/20.000 waveform và xác nhận 20.000 content hash duy nhất, không còn nhóm trùng trong cùng split hoặc đi qua train/development. Hồ sơ manifest v2 được khóa lại ngày 15/10/2026 với đúng SHA-256 `f549423b1fb33665f7e5606cdf56321bc8cd07246a0cdf0b90cd918c0c09e728`; loader manifest đã smoke thành công cả ba policy `none`, `peak`, `rms_dbfs` trên train/development. Benchmark A/B ngày 10/10 giữ `batch_size=8`, `num_workers=0`; I/O Parquet vẫn là nút thắt. Ngày 19/10/2026, entrypoint B0 production đã tích hợp metric, report và exact resume, đồng thời vượt smoke CPU/Kaggle CUDA. Ngày 20/10/2026, benchmark Kaggle ba lần đã khóa ngân sách chung 896 optimizer step cho ba pilot. Cổng gần nhất đạt 147/147 unit test; policy biên độ chiến thắng vẫn chỉ được chọn bằng closed development EER sau ba pilot.

Phân tích từ đầu đến cuối về các hạng mục đã hoàn thành, ý nghĩa của chúng, phần còn thiếu và đường găng tới khi bảo vệ nằm tại [docs/phan_tich_tong_the_tien_do_du_an.md](docs/phan_tich_tong_the_tien_do_du_an.md).

Bản báo cáo tiến độ có văn phong ngắn gọn, dễ trình bày với giảng viên nằm tại [docs/bao_cao_tien_do_do_an_2026-10-20.md](docs/bao_cao_tien_do_do_an_2026-10-20.md).

## Phạm vi phiên bản đầu

- Bài toán chính: phân loại nhị phân `bonafide` và `spoof`.
- Dữ liệu chính: VSASV, gồm `bonafide`, `voice_conversion`, `adversarial_attack` và `replay`.
- Giao thức bắt buộc: `closed_train/dev/test`, speaker-disjoint.
- VC, AP và replay được gộp thành nhãn `spoof` khi huấn luyện.
- `utt_type` chỉ dùng để thống kê và phân tích lỗi, không phải nhãn đầu ra.
- AASIST, open-set evaluation và dataset ngoài miền là phần tùy chọn.

Các quyết định phạm vi được ghi tại [docs/decisions.md](docs/decisions.md).

## Dữ liệu hiện có

Metadata VSASV được lưu tại `data/metadata/vsasv_metadata.csv` và giữ nguyên nội dung từ bản nhận ban đầu.

| Thuộc tính | Giá trị kỳ vọng |
|---|---:|
| Số mẫu | 220.963 |
| Số đường dẫn duy nhất | 220.963 |
| Số speaker | 1.141 |
| Bonafide | 98.305 |
| Voice conversion | 60.949 |
| Adversarial attack | 60.949 |
| Replay | 760 |
| Dòng trùng | 0 |
| Giá trị thiếu | 0 |

Nguồn và giấy phép được lưu trong [data/metadata/VSASV_DATASET_CARD.md](data/metadata/VSASV_DATASET_CARD.md). VSASV sử dụng giấy phép CC BY-NC 4.0.

## Chạy metadata audit

Yêu cầu Python 3.10 trở lên; script chỉ sử dụng thư viện chuẩn.

```powershell
python scripts/audit_metadata.py
```

Đầu ra:

- `reports/metadata_audit.json`: dữ liệu có cấu trúc để kiểm thử và tái lập.
- `reports/metadata_audit.md`: báo cáo đọc nhanh.

Script trả mã lỗi khác 0 nếu schema sai, có dữ liệu thiếu/trùng, kiểu mẫu không hợp lệ hoặc `label` không khớp thư mục đầu tiên trong `file`.

## Cài đặt môi trường

Tạo môi trường dự án và cài đúng các phiên bản đã khóa:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python scripts/verify_environment.py
```

Hướng dẫn chi tiết nằm tại [docs/cai_dat_moi_truong.md](docs/cai_dat_moi_truong.md).

## Chạy audio smoke test

Kiểm tra schema, duration, sample rate và waveform mẫu trong các shard cục bộ:

```powershell
python scripts/audio_smoke_test.py
```

Đầu ra:

- `reports/audio_smoke_test.json`: kết quả có cấu trúc.
- `reports/audio_smoke_test.md`: báo cáo đọc nhanh.

Giao thức tiền xử lý được khóa tại `configs/audio.json` và [docs/audio_protocol.md](docs/audio_protocol.md).

## Metric và evaluation contract

Metric EER, ROC-AUC và macro-F1 nằm trong `src/metrics/`. Quy ước nhãn là `1 = spoof`, `0 = bonafide`; score cao hơn biểu thị bằng chứng spoof mạnh hơn. `evaluate_development_scores` khóa decision threshold tại điểm EER trên development. `evaluate_scores_at_threshold` tiếp tục tính EER và ROC-AUC trực tiếp từ score nhưng dùng threshold development đã cung cấp để tính macro-F1; không được tính lại threshold từ test.

Chạy toàn bộ kiểm thử bằng:

```powershell
python -m unittest discover -s tests -v
```

## Dataset loader

`VSASVParquetDataset` đọc lười waveform từ các shard Parquet, giao dữ liệu cục bộ với `closed_train/dev/test`, kiểm tra metadata và trả về waveform `float32` dài 64.000 mẫu. Audio 40 kHz được resample về 16 kHz; policy biên độ được áp dụng sau resample và trước chia đoạn. Train dùng random crop xác định theo seed/epoch, còn development và test dùng center crop.

`VSASVManifestDataset` đọc trực tiếp một partition `closed_train` hoặc `closed_dev` từ development manifest v2. Loader dùng cột `shard` để định vị waveform nên không cần quét toàn bộ các shard để dựng chỉ mục khi khởi tạo. Nó kiểm tra schema, phiên bản manifest, source snapshot, file trùng, speaker-disjoint, nhãn nhị phân, shard tồn tại và đối chiếu metadata/sample rate với Parquet khi waveform được đọc.

Chạy kiểm tra trên các shard hiện có:

```powershell
python scripts/smoke_test_dataset_loader.py
```

Mỗi sample trả về `waveform`, nhãn nhị phân `label`, `file`, `speaker_id`, `utt_type`, sample rate gốc/đích và các trường audit biên độ như peak, RMS, gain, `near_silence` và `peak_limited`. Smoke test chạy cả ba policy trên train/dev/test và bao gồm mẫu VC 40 kHz khi có.

Kiểm tra riêng đường đọc manifest 20.000 mẫu trên train/development:

```powershell
python scripts/smoke_test_development_manifest_loader.py --amplitude-policy none
python scripts/smoke_test_development_manifest_loader.py --amplitude-policy peak
python scripts/smoke_test_development_manifest_loader.py --amplitude-policy rms_dbfs
```

Mỗi lệnh xác nhận đúng số mẫu 15.885/4.115, coverage cục bộ 100%, waveform 64.000 mẫu, đủ hai nhãn, policy trả về đúng yêu cầu và batch tái lập với seed `2026`. Đây là kiểm tra kỹ thuật, không phải phép đo EER, căn cứ chọn policy hoặc benchmark tối ưu pipeline.

## Amplitude audit và smoke subset

Chạy audit trên toàn bộ waveform cục bộ sau resample 16 kHz, trước chia đoạn:

```powershell
python scripts/audit_amplitude.py
```

Kết quả được ghi tại `reports/amplitude_audit.json` và `reports/amplitude_audit.md`. Báo cáo tổng hợp theo split, nhãn nhị phân, `utt_type`, sample rate gốc và policy. Thống kê này chỉ dùng để phát hiện shortcut risk, không dùng để chọn policy.

Kiểm kê dung lượng, số mẫu và coverage metadata của các shard cục bộ:

```powershell
python scripts/audit_local_storage.py
```

Kết quả nằm tại `reports/local_storage_audit.*`. Số đo ngày 07/10/2026 ghi nhận 68 shard, 34.782 mẫu, 9,85 GiB, metadata coverage 100% và 98,05 GiB dung lượng đĩa còn trống. Các shard cục bộ không phải mẫu ngẫu nhiên của toàn bộ snapshot nên dự báo tuyến tính chỉ dùng để lập kế hoạch.

Tạo lại ba smoke manifest cân bằng, cố định bằng seed `2026`:

```powershell
python scripts/make_smoke_subset.py
python scripts/smoke_test_dataset_loader.py
```

Đầu ra gồm `data/splits/smoke_train.csv` (256 mẫu), `smoke_dev.csv` (128 mẫu), `smoke_test.csv` (64 mẫu) và báo cáo `reports/smoke_subset_summary.*`. Các subset này chỉ dùng để kiểm tra code, không dùng để báo cáo EER khoa học.

Tạo lại development manifest 20.000 mẫu từ `closed_train` và `closed_dev`:

```powershell
python scripts/make_development_manifest.py --target-total 20000 --seed 2026
python scripts/audit_development_manifest_content.py
python scripts/smoke_test_development_manifest_loader.py
```

Đầu ra là `data/manifests/development_20k_v2.csv` cùng báo cáo `reports/development_manifest_20k_v2.*`. Script băm toàn bộ pool cục bộ, chỉ giữ một đại diện xác định cho mỗi content hash, giữ speaker trong split gốc, phân bổ theo phân bố đầy đủ của closed train/dev và dừng nếu audio sau khử trùng không đủ bất kỳ quota nào. Audit tiếp theo đọc và băm lại độc lập đúng 20.000 waveform, lưu hash từng dòng trong `reports/development_manifest_20k_v2_content_audit.json`. `closed_test` không tham gia chọn quy mô hoặc phân bố manifest; bốn nhóm duplicate `closed_train`–`closed_test` phải được xử lý bằng quyết định riêng trước đánh giá cuối. Manifest v1 được giữ để truy vết nhưng không dùng cho kết quả khoa học.

## Baseline B0 LFCC + LCNN

LCNN tối thiểu dùng bốn block convolution với Max-Feature-Map, adaptive average pooling và một head sinh raw spoof logit. Mô hình nhận tensor `(batch, 1, 60, 401)`, có 498.113 tham số trainable và dùng `BCEWithLogitsLoss` theo quy ước `0 = bonafide`, `1 = spoof`.

Chạy smoke pilot một bước theo cấu hình mặc định:

```powershell
python scripts/smoke_train_lfcc_lcnn.py
```

Chạy pilot kỹ thuật đủ một epoch trên smoke subset:

```powershell
python scripts/smoke_train_lfcc_lcnn.py --epochs 1 --max-steps 0
```

Script kiểm tra LFCC/logit/loss/gradient hữu hạn, xác nhận optimizer làm thay đổi tham số, đánh giá development loss, lưu rồi nạp checkpoint và ghi số liệu thời gian/RAM vào `reports/b0_smoke_training.*`. Checkpoint nằm trong `checkpoints/` và không được đưa vào Git. Kết quả smoke chỉ chứng minh pipeline hoạt động, không dùng để báo cáo EER hoặc chọn amplitude policy.

Tiếp tục kỹ thuật từ checkpoint trong một process mới:

```powershell
python scripts/smoke_train_lfcc_lcnn.py `
  --resume-from checkpoints/b0_lfcc_lcnn_smoke.pt `
  --checkpoint checkpoints/b0_lfcc_lcnn_resumed.pt
```

Đường resume của smoke script kiểm tra cấu hình, seed, amplitude policy, batch size và learning rate trước khi nạp model/optimizer, rồi ghi số bước trước/sau cùng thời gian chờ DataLoader, thời gian model, throughput và peak VRAM.

Entry point production là `scripts/train_b0.py`. Script dùng chung một đường train → full development → EER/ROC-AUC/macro-F1 → report → checkpoint cho ba chế độ `technical_smoke`, `pilot` và `full_training`. `ResumableBatchSampler` lưu epoch và cursor của batch kế tiếp; checkpoint lưu model, optimizer, global step, epoch, batch cursor, Python/NumPy/PyTorch/CUDA RNG cùng checksum manifest/config, run mode và step budget. Checkpoint không tương thích bị từ chối trước khi resume. DataLoader dùng generator riêng để việc tạo iterator không làm lệch RNG của dropout; workload fixture và smoke dữ liệu thật đều xác nhận chạy liên tục với chạy ngắt–resume cho tham số cuối giống từng tensor.

Ví dụ smoke production hai bước trên manifest v2:

```powershell
python scripts/train_b0.py --run-mode technical_smoke `
  --manifest data/manifests/development_20k_v2.csv `
  --parquet-dir data/raw/vsasv_parquet/data `
  --amplitude-policy none --step-budget 2 --batch-size 2 `
  --train-limit 4 --development-limit 2 --device cpu `
  --checkpoint checkpoints/b0_production_local.pt `
  --report-json reports/b0_production_local.json `
  --report-md reports/b0_production_local.md
```

`--train-limit` và `--development-limit` chỉ hợp lệ trong `technical_smoke`. Chế độ pilot/full luôn đọc toàn bộ development đã khóa, không early stopping và giữ `num_workers=0` để resume đúng cursor. Kết quả smoke chỉ xác nhận pipeline hoạt động, không dùng để chọn amplitude policy.

Kaggle là GPU chính theo D018. Không cài `requirements.txt` hoặc `requirements-lock.txt` trên Kaggle vì hai file khóa PyTorch CPU; dùng `requirements-kaggle.txt` và làm theo [hướng dẫn kiểm chứng Kaggle GPU](docs/guides/kaggle_gpu_smoke.md). Quy trình đóng gói source, cập nhật private dataset, gửi kernel, theo dõi và tải output hoàn toàn bằng CLI nằm tại [hướng dẫn Kaggle CLI](docs/guides/kaggle_cli_workflow.md).

Benchmark đại diện ngày 20/10/2026 đã khóa ngân sách ba pilot theo D020: batch 8, worker 0, Adam, learning rate 0,001 và 896 optimizer step, sau đó đánh giá đủ 4.115 mẫu development. Báo cáo timing và phép chiếu nằm tại `reports/b0_benchmark_summary.*`. Ba profile CLI sẵn sàng là `b0-pilot-none`, `b0-pilot-peak` và `b0-pilot-rms-dbfs`; mỗi profile tạo một private kernel run riêng và không dùng `closed_test`.

## Benchmark hiệu năng pipeline

Benchmark P0 trên 24 mẫu cố định, ba lần lặp, batch size 8 và `num_workers=0` xác định DataLoader chiếm 95,31% thời gian đường train. Riêng truy vấn đọc/decode từng waveform từ Parquet có trung vị 925,361 ms và chiếm 99,82% thời gian đọc/tiền xử lý; LFCC và LCNN không phải điểm nghẽn chính.

Chạy lại benchmark chi tiết:

```powershell
python scripts/benchmark_data_pipeline.py
```

So sánh batch size và worker:

```powershell
python scripts/benchmark_dataloader_options.py
```

Trên cùng 16 mẫu và ba lần lặp, batch size 4 chỉ nhanh hơn 1,25%, batch size 16 chậm hơn 6,57%, còn hai worker nhanh hơn 14,42% nhưng peak RSS cây tiến trình tăng từ khoảng 531 MiB lên 1.831 MiB. Do mức tăng tốc không vượt ngưỡng chấp nhận 15% và chi phí bộ nhớ lớn, cấu hình kỹ thuật hiện giữ `batch_size=8`, `num_workers=0`. Báo cáo đầy đủ nằm tại `reports/data_pipeline_benchmark.*` và `reports/data_pipeline_optimization.*`.

So sánh trực tiếp loader split cũ và loader manifest v2 trên cùng danh sách file:

```powershell
python scripts/benchmark_manifest_loaders.py
```

Benchmark ngày 10/10 dùng 16 file thuộc 14 shard, seed `2026`, train mode, policy `none`, batch 8, worker 0/2 và ba lần lặp. File, speaker, `utt_type`, nhãn và checksum waveform sau tiền xử lý khớp giữa hai loader. Với worker 0, manifest đạt 0,645 mẫu/giây, nhanh hơn split cũ 2,13%; với worker 2, manifest đạt 0,593 mẫu/giây. Trên chính loader manifest, hai worker chậm hơn 8,02% và tăng peak RSS 502,40 MiB, nên D014 tiếp tục giữ worker 0. Manifest giảm trung vị thời gian khởi tạo 32,98% so với split cũ. Báo cáo nằm tại `reports/manifest_loader_benchmark.*`; đây là benchmark DataLoader kỹ thuật trên cache hệ điều hành không bị xóa, không phải throughput train end-to-end hoặc kết quả khoa học.

Đo lại ứng viên batch-read theo shard:

```powershell
python scripts/benchmark_batch_read.py
```

Benchmark P0.2 dùng 24 file cố định, ba lần lặp, seed `2026`, batch 8 và worker 0. Micro-benchmark xác nhận batch cùng shard giảm từ 8 xuống 1 truy vấn, nhưng cả ba batch theo thứ tự shuffle thực tế đều trải trên 8 shard nên ứng viên vẫn dùng 24 truy vấn mỗi lượt. Throughput B0 end-to-end giảm 0,93%, từ 0,638 xuống 0,632 mẫu/giây, không đạt cổng tăng 15%; vì vậy hook `__getitems__` đã được gỡ, D014 tiếp tục giữ nguyên và P1 artifact đóng gói được kích hoạt. Báo cáo đầy đủ nằm tại `reports/batch_read_benchmark.*`; `reports/runtime_projection.*` không được cập nhật từ workload nhỏ này.

Chạy lại proof-of-concept artifact đóng gói trên đúng 24 mẫu đối chứng:

```powershell
python scripts/benchmark_packed_artifact.py
```

Artifact lưu waveform `float32` sau decode/resample 16 kHz nhưng trước amplitude policy và crop, giữ `source_index` để random crop không đổi. PoC ngày 13/10 băm lại độc lập 24/24 waveform, khớp toàn bộ fingerprint và đạt 245,871 mẫu/giây loader-only cùng 19,853 mẫu/giây B0, lần lượt cải thiện 37.568,78% và 3.097,89% so với Parquet trên workload nhỏ. Artifact PoC nằm trong `data/processed/` và bị Git bỏ qua; báo cáo nằm tại `reports/packed_artifact_benchmark.*`. Không ngoại suy các số này lên 20.000 mẫu và chưa tạo artifact đầy đủ khi chưa có xác nhận.

Dự báo lại thời gian từ các báo cáo đã đo:

```powershell
python scripts/estimate_runtime.py
```

Với cấu hình mặc định, development subset 20.000 mẫu cần khoảng 5,76 giờ cho một epoch train và một lượt development; mốc 40.000 mẫu cần 11,52 giờ. Ngân sách có dự phòng 20% lần lượt là 6,91 và 13,83 giờ. Báo cáo chi tiết và các giả định nằm tại `reports/runtime_projection.*`.

## Tạo split và kiểm tra rò rỉ

Giao thức chi tiết được mô tả tại [docs/split_protocol.md](docs/split_protocol.md).

Tạo lại tám split cố định:

```powershell
python scripts/make_splits.py
```

Kiểm tra schema, coverage, speaker/file overlap và ràng buộc attack type:

```powershell
python scripts/check_leakage.py
```

Khi audio đã có cục bộ, bổ sung kiểm tra file tồn tại và SHA-256 nội dung:

```powershell
python scripts/check_leakage.py --audio-root data/raw
```

Đầu ra chính:

- `data/splits/*.csv`: tám tập closed/open theo giao thức.
- `reports/split_summary.md`: phân bổ speaker, mẫu và checksum.
- `reports/leakage_check.md`: kết quả kiểm tra rò rỉ.

Chạy kiểm thử tự động:

```powershell
python -m unittest discover -s tests -v
```

## Cấu trúc dự án

```text
configs/                 Cấu hình thí nghiệm
data/metadata/           Metadata và dataset card
data/splits/             Split đã khóa theo giao thức
data/raw/                Audio gốc, không đưa vào Git
data/processed/          Dữ liệu trung gian, không đưa vào Git
data/embeddings/         Embedding offline, không đưa vào Git
scripts/                 Script audit, tạo split và đánh giá
src/data/                Dataset, sampler, augmentation
src/models/              Các mô hình
tests/                   Kiểm thử tự động
reports/                 Audit, bảng, hình và phân tích lỗi
app/                     Demo cuối
docs/                    Nhật ký quyết định và thí nghiệm
```

## Thứ tự ưu tiên

```text
Đúng split → Đúng metric → LFCC + LCNN → XLS-R đóng băng
→ XLS-R fine-tune một phần → Phân tích lỗi → Demo
```

Tài liệu nền:

- [Đề cương cập nhật](baocao/De_cuong_DATN_Nguyen_Ngoc_Tan_cap_nhat.docx)
- [Lộ trình hoàn chỉnh](lo_trinh_hoan_chinh_do_an_deepfake_tieng_viet.md)
