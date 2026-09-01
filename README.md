# Phát hiện giọng nói deepfake tiếng Việt có khả năng tổng quát hóa

Dự án xây dựng hệ thống phát hiện giả mạo giọng nói tiếng Việt, tập trung vào khả năng tổng quát hóa trên người nói, kiểu tấn công và biến đổi kênh chưa xuất hiện khi huấn luyện.

## Trạng thái

Dự án đang ở giai đoạn khởi tạo và kiểm định metadata. VSASV là bộ dữ liệu chính. Chưa bắt đầu huấn luyện mô hình trước khi hoàn tất audit dữ liệu, tạo split theo người nói và kiểm tra rò rỉ.

## Phạm vi phiên bản đầu

- Bài toán chính: phân loại nhị phân `bonafide` và `spoof`.
- Dữ liệu chính: VSASV, gồm `bonafide`, `voice_conversion`, `adversarial_attack` và `replay`.
- Giao thức bắt buộc: speaker-disjoint và đánh giá riêng theo từng kiểu tấn công.
- Cách diễn đạt kết quả: “unseen spoofing attacks”.
- Không tuyên bố phát hiện “unseen TTS engines” khi VSASV hiện không có TTS hoặc `generator_id`.
- SEA-Spoof chỉ là dữ liệu mở rộng nếu được cấp quyền truy cập.

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
Đúng split → Đúng metric → Baseline tái lập → Phương pháp chính
→ Ablation → External test → Demo
```

Tài liệu nền:

- [Đề cương chi tiết](de_cuong_phat_hien_giong_noi_deepfake_tieng_viet.md)
- [Lộ trình hoàn chỉnh](lo_trinh_hoan_chinh_do_an_deepfake_tieng_viet.md)
