# Báo cáo smoke pilot B0 LFCC + LCNN

## Kết quả

- Trạng thái: **ĐẠT**
- Thời điểm: `2026-10-13T08:23:52+00:00`
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
| Bước optimizer | 1 |
| Mẫu đã xử lý | 8 |
| Loss đầu | 0.684956 |
| Loss cuối | 0.684956 |
| Loss trung bình | 0.684956 |
| Loss nhỏ nhất | 0.684956 |
| Loss lớn nhất | 0.684956 |
| Gradient hữu hạn | True |
| Tham số đã thay đổi | True |

## Kiểm tra development

- Số mẫu: 128
- BCE loss trung bình: 3.053755
- Mọi giá trị hữu hạn: True

## Checkpoint và tài nguyên

- Checkpoint: `/kaggle/working/checkpoints/b0_kaggle_phase1.pt`
- Dung lượng: 5,996,153 byte
- SHA-256: `b6f4671ed3e14ad1225bbde942f7bfdf10699939dd8d72e9435e099b13c9f80e`
- Sai khác logit lớn nhất sau khôi phục: 0.0000000000
- Thời gian train: 13.661 giây
- Chờ DataLoader train: 11.403 giây
- Tính toán model train: 2.249 giây
- Throughput train: 0.586 mẫu/giây
- Thời gian development: 90.990 giây
- Chờ DataLoader development: 90.868 giây
- Tính toán model development: 0.123 giây
- Throughput development: 1.407 mẫu/giây
- Tổng thời gian: 112.530 giây
- RSS đầu: 562.95 MiB
- RSS lớn nhất quan sát: 1957.21 MiB
- RSS cuối: 1571.29 MiB
- Peak VRAM: 286.46 MiB
- Resume từ checkpoint: False
- Bước trước/sau lần chạy: 0 → 1
- Nguồn resume: `không dùng`

## Phạm vi diễn giải

Đây là smoke pilot kỹ thuật trên subset 448 mẫu cục bộ. Kết quả chỉ chứng minh pipeline, gradient và checkpoint hoạt động; không phải kết quả khoa học, không dùng để chọn amplitude policy và không đại diện cho toàn bộ VSASV.
