# Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần

Dự án xây dựng bộ phân loại nhị phân cho tiếng nói tiếng Việt: `bonafide` hoặc `spoof`. LFCC + LCNN được dùng làm mô hình cơ sở; XLS-R đóng băng và XLS-R fine-tune một phần là các cấu hình chính cần so sánh.

## Trạng thái

Dự án đã hoàn tất các cổng dữ liệu trước baseline: metadata audit đạt, tám split theo speaker được khóa với seed `2026`, leakage checker đạt, audio smoke test trên năm shard đạt, metric EER đã có unit test và dataset loader đã tạo được batch cố định từ Parquet. Amplitude audit đã chạy đủ 2.558 mẫu cục bộ cho cả ba policy `none`, `peak`, `rms_dbfs`; smoke subset 448 mẫu đã cân bằng nhãn, tách biệt speaker và tái lập bằng seed `2026`. Baseline B0 tối thiểu đã chạy xuyên suốt từ waveform qua LFCC, LCNN, loss, backward, optimizer và checkpoint trên CPU. Pilot kỹ thuật một epoch đạt 32 bước train, checkpoint khôi phục chính xác và toàn bộ 52 kiểm thử đạt ngày 04/10/2026. Policy biên độ chiến thắng vẫn chỉ được chọn bằng closed development trong ablation tuần 4.

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

## Metric EER

Metric EER nằm trong `src/metrics/eer.py`. Quy ước nhãn là `1 = spoof`, `0 = bonafide`; score cao hơn biểu thị bằng chứng spoof mạnh hơn. Chạy toàn bộ kiểm thử bằng:

```powershell
python -m unittest discover -s tests -v
```

## Dataset loader

`VSASVParquetDataset` đọc lười waveform từ các shard Parquet, giao dữ liệu cục bộ với `closed_train/dev/test`, kiểm tra metadata và trả về waveform `float32` dài 64.000 mẫu. Audio 40 kHz được resample về 16 kHz; policy biên độ được áp dụng sau resample và trước chia đoạn. Train dùng random crop xác định theo seed/epoch, còn development và test dùng center crop.

`VSASVManifestDataset` đọc trực tiếp một partition `closed_train` hoặc `closed_dev` từ development manifest v1. Loader dùng cột `shard` để định vị waveform nên không cần quét toàn bộ các shard để dựng chỉ mục khi khởi tạo. Nó kiểm tra schema, phiên bản manifest, source snapshot, file trùng, speaker-disjoint, nhãn nhị phân, shard tồn tại và đối chiếu metadata/sample rate với Parquet khi waveform được đọc.

Chạy kiểm tra trên các shard hiện có:

```powershell
python scripts/smoke_test_dataset_loader.py
```

Mỗi sample trả về `waveform`, nhãn nhị phân `label`, `file`, `speaker_id`, `utt_type`, sample rate gốc/đích và các trường audit biên độ như peak, RMS, gain, `near_silence` và `peak_limited`. Smoke test chạy cả ba policy trên train/dev/test và bao gồm mẫu VC 40 kHz khi có.

Kiểm tra riêng đường đọc manifest 20.000 mẫu trên train/development:

```powershell
python scripts/smoke_test_development_manifest_loader.py
```

Lệnh này xác nhận đúng số mẫu 15.885/4.115, coverage cục bộ 100%, waveform 64.000 mẫu, đủ hai nhãn và batch tái lập với seed `2026`. Đây là kiểm tra kỹ thuật, không phải phép đo EER hoặc benchmark tối ưu pipeline.

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
```

Đầu ra là `data/manifests/development_20k_v1.csv` cùng báo cáo `reports/development_manifest_20k.*`. Script giữ speaker trong split gốc, phân bổ theo phân bố đầy đủ của closed train/dev, chọn ổn định bằng SHA-256 của `2026|file` và dừng nếu audio cục bộ không đủ bất kỳ quota nào. `closed_test` không tham gia chọn quy mô hoặc phân bố manifest.

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

- [Đề cương cập nhật](De_cuong_DATN_Nguyen_Ngoc_Tan_cap_nhat.docx)
- [Lộ trình hoàn chỉnh](lo_trinh_hoan_chinh_do_an_deepfake_tieng_viet.md)
