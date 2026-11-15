# Preflight B0 full session 9

## Kết luận

ĐẠT và đã gửi. Checkpoint version 21 ở bước 4.096 đã được đóng gói exact-resume cho phiên 9; restore cục bộ xác nhận đủ model, optimizer, sampler, RNG, controller, history và best epoch 2.

## Gói dự kiến upload

- Đích: private dataset `ngctnnguyn1111/doantnnt-b0-full-resume`.
- 77 generation; 155 tệp checkpoint/pointer/trạng thái.
- Nội dung upload: 156 tệp, 557.835.353 byte, gồm manifest 40.396 byte.
- Manifest SHA-256: `cc9ebf87f898162980f2c24de5c4b80bf04dafa0c5d34af3cbdb345b0ed997d8`.
- Không có audio, log hoặc thông tin đăng nhập.

## Kernel và tài nguyên

- Profile: `b0-full-resume-session-9`, private, GPU bật, internet tắt.
- Input bắt buộc: bước 4.096; source B0 khóa 56 tệp giữ nguyên.
- Ba dataset đạt `ready`; checkpoint dataset version 8 đã được đối chiếu đủ 156 tệp/557.835.353 byte và manifest tải ngược khớp SHA-256.
- Quota GPU đã làm mới: còn 30,00/30,00 giờ, chu kỳ tiếp theo ngày 07/11/2026.

Sau xác nhận cụ thể của người dùng, private kernel version 22 được gửi đúng một lần và quan sát `QUEUED`. Nếu chạy đủ giới hạn, phiên 9 sẽ đi từ bước 4.096 tới 4.608, epoch 3/cursor 636. Full development tiếp theo chỉ ở bước 5.958. Không gửi kernel khác khi version 22 còn `QUEUED`/`RUNNING`.
