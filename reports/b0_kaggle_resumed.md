# Báo cáo smoke pilot B0 LFCC + LCNN

## Kết quả

- Trạng thái: **ĐẠT**
- Thời điểm: `2026-10-13T08:25:36+00:00`
- Thiết bị: `cuda`
- Seed: `2026`
- Amplitude policy: `none`
- Số tham số trainable: 498,113
- Shape waveform: `(8, 64000)`
- Shape LFCC: `(8, 60, 401)`
- Shape LCNN input: `(8, 1, 60, 401)`
- Shape logit: `(8,)`

## Huấn luyện kỹ thuật

| Thuộc tính | Giá trị |
|---|---:|
| Epoch đã chạy | 0 |
| Bước optimizer | 2 |
| Mẫu đã xử lý | 8 |
| Loss đầu | 1.652401 |
| Loss cuối | 1.652401 |
| Loss trung bình | 1.652401 |
| Loss nhỏ nhất | 1.652401 |
| Loss lớn nhất | 1.652401 |
| Gradient hữu hạn | True |
| Tham số đã thay đổi | True |

## Kiểm tra development

- Số mẫu: 128
- BCE loss trung bình: 0.923001
- Mọi giá trị hữu hạn: True

## Checkpoint và tài nguyên

- Checkpoint: `/kaggle/working/checkpoints/b0_kaggle_resumed.pt`
- Dung lượng: 5,996,399 byte
- SHA-256: `5fae6b228cb16b9a60413e50748bb83087ef373252eea9b2d3b80e4d31fc0368`
- Sai khác logit lớn nhất sau khôi phục: 0.0000000000
- Thời gian train: 9.182 giây
- Chờ DataLoader train: 8.584 giây
- Tính toán model train: 0.597 giây
- Throughput train: 0.871 mẫu/giây
- Thời gian development: 89.449 giây
- Chờ DataLoader development: 89.324 giây
- Tính toán model development: 0.122 giây
- Throughput development: 1.431 mẫu/giây
- Tổng thời gian: 100.788 giây
- RSS đầu: 563.70 MiB
- RSS lớn nhất quan sát: 1902.55 MiB
- RSS cuối: 1488.80 MiB
- Peak VRAM: 289.99 MiB
- Resume từ checkpoint: True
- Bước trước/sau lần chạy: 1 → 2
- Nguồn resume: `/kaggle/working/checkpoints/b0_kaggle_phase1.pt`

## Phạm vi diễn giải

Đây là smoke pilot kỹ thuật trên subset 448 mẫu cục bộ. Kết quả chỉ chứng minh pipeline, gradient và checkpoint hoạt động; không phải kết quả khoa học, không dùng để chọn amplitude policy và không đại diện cho toàn bộ VSASV.
