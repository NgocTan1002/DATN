# Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần

Dự án xây dựng bộ phân loại nhị phân cho tiếng nói tiếng Việt: `bonafide` hoặc `spoof`. LFCC + LCNN được dùng làm mô hình cơ sở; XLS-R đóng băng và XLS-R fine-tune một phần là các cấu hình chính cần so sánh.

## Trạng thái

Dự án đã hoàn tất các cổng trước baseline: metadata audit đạt, tám split theo speaker được khóa với seed `2026`, leakage checker đạt, audio smoke test trên năm shard đạt, metric EER đã có unit test và dataset loader đã tạo được batch cố định từ Parquet. Giao thức âm thanh phiên bản 3 bắt buộc mono 16 kHz, đoạn 4 giây và đã khóa quy tắc của ba policy biên độ ứng viên; policy chiến thắng vẫn phải chọn trên closed development. Giao thức chính sau khi thu hẹp đề tài là `closed_train/dev/test`. Bước tiếp theo là cài ba chính sách biên độ, tạo smoke subset cân bằng và triển khai LFCC + LCNN trước khi mở rộng số shard.

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

Kiểm tra schema, duration, sample rate và waveform mẫu trong năm shard:

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

`VSASVParquetDataset` đọc lười waveform từ các shard Parquet, giao dữ liệu cục bộ với `closed_train/dev/test`, kiểm tra metadata và trả về waveform `float32` dài 64.000 mẫu. Audio 40 kHz được resample về 16 kHz; train dùng random crop xác định theo seed/epoch, còn development và test dùng center crop.

Chạy kiểm tra trên các shard hiện có:

```powershell
python scripts/smoke_test_dataset_loader.py
```

Mỗi sample trả về `waveform`, nhãn nhị phân `label`, `file`, `speaker_id`, `utt_type`, sample rate gốc và sample rate đích.

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
