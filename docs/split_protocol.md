# Giao thức chia dữ liệu VSASV phiên bản 1

## Mục tiêu

Giao thức tạo các tập dữ liệu tái lập được, không rò rỉ người nói giữa train, dev và test. Metadata nguồn giữ nguyên ba trường `file`, `label`, `utt_type`.

## Phiên bản và seed

- Phiên bản split: `1`.
- Seed: `2026`.
- Tỷ lệ speaker trong từng stratum: train 70%, dev 15%, test 15%.
- Cách xếp speaker: sắp xếp theo SHA-256 của `seed`, tên stratum và speaker ID.
- Cách làm tròn: largest remainder; thứ tự phá hòa là train, dev, test.

## Speaker strata

Metadata hiện tại có đúng ba nhóm:

| Stratum | Điều kiện | Số speaker |
|---|---|---:|
| `real_only` | Chỉ có bonafide | 978 |
| `vc_adversarial` | Có bonafide, voice conversion và adversarial attack | 144 |
| `replay` | Có bonafide và replay | 19 |

Mỗi stratum được chia riêng để các attack hiếm không biến mất khỏi split do mất cân bằng.

## Closed protocol

Closed protocol chia speaker của cả ba strata theo tỷ lệ 70/15/15. Tất cả dòng của một speaker phải nằm trong cùng một split.

Từ quyết định D008 ngày 25/09/2026, đây là giao thức chính của đồ án. Các loại `voice_conversion`, `adversarial_attack` và `replay` được ánh xạ về cùng nhãn `spoof` khi huấn luyện. Trường `utt_type` được giữ nguyên để thống kê và phân tích lỗi, không phải đầu ra của mô hình.

Đầu ra:

- `closed_train.csv`
- `closed_dev.csv`
- `closed_test.csv`

Ba file phải phủ đúng toàn bộ metadata nguồn và không trùng speaker hoặc file.

## Open protocol

Open protocol là giao thức phân tích bổ sung. Nó không còn thuộc phần bắt buộc sau khi đề tài được thu hẹp vào bài toán phát hiện nhị phân.

### Train và dev

- Chứa bonafide và voice conversion.
- Không chứa adversarial attack hoặc replay.
- Không chứa bất kỳ speaker replay nào.

### Seen test

- Chứa bonafide và voice conversion của speaker test chưa xuất hiện trong train/dev.

### Unseen adversarial test

- Chứa adversarial attack của nhóm speaker test.
- Dùng lại đúng bonafide pool của seen test để hai phép đánh giá có cùng negative pool.
- Việc dùng chung bonafide giữa hai test variant là có chủ đích. Không có file nào từ hai variant này xuất hiện trong train hoặc dev.

### Unseen replay test

- Giữ toàn bộ 19 speaker replay ngoài train, dev và test adversarial.
- Chứa bonafide và replay của 19 speaker này.

Đầu ra:

- `open_train_vc.csv`
- `open_dev_vc.csv`
- `open_seen_test_vc.csv`
- `open_unseen_test_adversarial.csv`
- `open_unseen_test_replay.csv`

## Cổng chất lượng

`scripts/check_leakage.py` phải đạt trước khi dùng split để huấn luyện. Kiểm tra bắt buộc gồm:

- schema và dòng dữ liệu khớp metadata nguồn;
- không lặp file trong từng split;
- không trùng speaker hoặc file giữa các learning partition;
- attack type đúng vai trò của từng split;
- closed protocol phủ toàn bộ metadata;
- open protocol có đúng cohort và coverage đã định nghĩa;
- nếu cung cấp `--audio-root`, tất cả audio phải tồn tại và không có hash trùng qua partition.

Mọi metric phụ thuộc threshold phải chọn threshold trên dev. Test chỉ dùng cho đánh giá cuối cùng.
