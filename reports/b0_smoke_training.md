# Báo cáo smoke pilot B0 LFCC + LCNN

## Kết quả

- Trạng thái: **ĐẠT**
- Thời điểm: `2026-10-01T11:55:02+07:00`
- Thiết bị: `cpu`
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
| Epoch đã chạy | 1 |
| Bước optimizer | 32 |
| Mẫu đã xử lý | 256 |
| Loss đầu | 0.807307 |
| Loss cuối | 0.328532 |
| Loss trung bình | 0.769690 |
| Loss nhỏ nhất | 0.328532 |
| Loss lớn nhất | 1.666013 |
| Gradient hữu hạn | True |
| Tham số đã thay đổi | True |

## Kiểm tra development

- Số mẫu: 128
- BCE loss trung bình: 0.962665
- Mọi giá trị hữu hạn: True

## Checkpoint và tài nguyên

- Checkpoint: `C:\Users\Admin\Documents\DoAnTTNT\checkpoints\b0_lfcc_lcnn_smoke.pt`
- Dung lượng: 5,995,301 byte
- SHA-256: `0a6fbea4bd8b430bc0ea2af9fa0d7a6b24be40a9f7599ec1f05a0a2b841f9dc0`
- Sai khác logit lớn nhất sau khôi phục: 0.0000000000
- Thời gian train: 259.039 giây
- Thời gian development: 78.237 giây
- Tổng thời gian: 339.946 giây
- RSS đầu: 212.85 MiB
- RSS lớn nhất quan sát: 625.03 MiB
- RSS cuối: 621.71 MiB
- Peak VRAM: 0.00 MiB

## Phạm vi diễn giải

Đây là smoke pilot kỹ thuật trên subset 448 mẫu cục bộ. Kết quả chỉ chứng minh pipeline, gradient và checkpoint hoạt động; không phải kết quả khoa học, không dùng để chọn amplitude policy và không đại diện cho toàn bộ VSASV.
