# Chuẩn bị B0 full training phiên 7 — ngày 30/10/2026

- Trạng thái: preflight đạt, payload đã upload và đối chiếu lại; private kernel version 20 đã gửi đúng một lần và đang `RUNNING`.
- Mục tiêu: resume bước 3.072, tiếp tục epoch 2 và tối đa dừng ở bước 3.584/cursor 1.598. Không có development evaluation trong phiên này; mốc tiếp theo là bước 3.972.
- Input gồm 57 generation/115 file checkpoint, pointer và state; tổng 391,428,734 byte.
- Gói cục bộ gồm manifest/metadata có 117 file, tổng 391,459,139 byte. Dữ liệu thực upload là 116 file/391,458,890 byte.
- Manifest chuyển phiên SHA-256: `688af75ea12f99df92af5e29b25858534d5cab1e324d358e44ecc30970886369`.
- Source bundle SHA-256: `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`; source tree `a46681120d36511e5d1bc7d5290e5488eecf136bf1ad55a93052afdb0e788925`; cả 56 file source huấn luyện giữ nguyên.
- Restore gói thật và bootstrap CLI nhúng đều đạt; model/optimizer/sampler/RNG/controller/history/best khớp payload gốc.
- Best hiện hành ở bước 1.986 và một history entry epoch 1 được bảo toàn. Threshold hiện hành không thay đổi trong phiên 7.
- Profile `b0-full-resume-session-7` có `--resume`, expected input step 3.072 và output report riêng cho phiên 7.
- Kernel private, internet tắt, gắn data/source/checkpoint dataset. Payload chỉ gồm checkpoint/trạng thái/manifest/metadata; không có audio, log hoặc thông tin đăng nhập.
- Preflight trực tuyến lúc 10:02 UTC+7: latest kernel `COMPLETE`; ba dataset data/source/checkpoint đều private/`ready`; quota GPU còn 17,27/30,00 giờ và làm mới lúc 00:00 ngày 31/10/2026. Cảnh báo Kaggle CLI 2.2.4 cũ hơn 2.2.5 không chặn chạy.
- Sau xác nhận cụ thể của người dùng, payload đã upload lên private checkpoint dataset. Tải lại xác nhận private/`ready`, đúng 116 file/391.458.890 byte; inventory, manifest checkpoint/source và checksum bộ kernel khóa đều khớp.
- Bộ kernel khóa là private, bật GPU, tắt internet và gắn đúng ba dataset. Version 20 được gửi đúng một lần và quan sát `RUNNING` lúc 10:18 UTC+7; quota sau gửi còn 17,26/30,00 giờ. Không sửa production code nên không chạy lại unit suite; cổng 179/179 trước đó giữ nguyên.

## Cổng trước khi gửi

1. Đã xác nhận latest kernel không còn `QUEUED`/`RUNNING`, ba dataset private/`ready` và quota GPU còn 17,27 giờ.
2. Đã nhận xác nhận cụ thể và upload gói mới lên private dataset `ngctnnguyn1111/doantnnt-b0-full-resume`.
3. Đã tải lại manifest/inventory: private/ready, 116 file/391.458.890 byte và checksum khớp.
4. Đã gửi đúng bộ kernel khóa tại `.kaggle/resume/session-7-kernel/`; không dùng lệnh profile chung làm mất checkpoint mount.

Điểm đã kiểm chứng gần nhất vẫn là bước 3.072. Mốc 3.584/cursor 1.598 và 28.669 lượt mẫu là kỳ vọng để đối chiếu, chưa phải kết quả thực tế.
