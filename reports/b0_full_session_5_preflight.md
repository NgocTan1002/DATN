# Chuẩn bị B0 full training phiên 5 — ngày 28/10/2026

- Trạng thái: **ĐẠT và đã kiểm chứng output**. Người dùng báo version 18 `COMPLETE` và đã tải output; kiểm chứng ngày 29/10 xác nhận workflow hoàn tất, resume đúng bước 2.048 → 2.560 và dừng `paused`.
- Mục tiêu: resume bước 2.048, tiếp tục epoch 2 và tối đa dừng ở bước 2.560/cursor 574. Không có development evaluation trong phiên này; mốc tiếp theo là bước 3.972.
- Input gồm 39 generation/79 file checkpoint, pointer và state; tổng 245,056,669 byte.
- Gói cục bộ gồm manifest/metadata có 81 file, tổng 245,077,858 byte. Dữ liệu thực upload là 80 file/245,077,609 byte.
- Manifest chuyển phiên SHA-256: `31bac0c9f339264b9484e653a648908b73be72c166bd675cb0632ae7ff7a48c1`.
- Source bundle SHA-256: `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`; source tree `a46681120d36511e5d1bc7d5290e5488eecf136bf1ad55a93052afdb0e788925`; cả 56 file source huấn luyện giữ nguyên.
- Restore gói thật và bootstrap CLI nhúng đều đạt; model/optimizer/sampler/RNG/controller/history/best khớp payload gốc.
- Best hiện hành ở bước 1.986 và một history entry epoch 1 được bảo toàn. Threshold hiện hành không thay đổi trong phiên 5.
- Profile `b0-full-resume-session-5` có `--resume`, expected input step 2.048 và output report riêng cho phiên 5.
- Kernel private, internet tắt, gắn data/source/checkpoint dataset. Payload chỉ gồm checkpoint/trạng thái/manifest/metadata; không có audio, log hoặc thông tin đăng nhập.
- Kiểm tra Kaggle lúc 13:43 UTC+7: latest kernel `COMPLETE`; checkpoint/source/data dataset đều `ready`; quota GPU còn 20,87/30,00 giờ (đã dùng 9,13 giờ, làm mới ngày 31/10/2026). Không sửa production code nên không chạy lại unit suite; cổng 179/179 trước đó giữ nguyên.
- Sau xác nhận cụ thể, Kaggle nhận gói mới; tải lại xác nhận dataset private/`ready`, đúng 80 tệp/245.077.609 byte, inventory và manifest checkpoint/source khớp hoàn toàn.
- Bộ kernel khóa `.kaggle/resume/session-5-kernel/` được push trực tiếp. Version 18 quan sát `RUNNING`; quota ngay sau gửi vẫn còn 20,87 giờ GPU.

## Kết quả cổng gửi

1. Đã xác nhận latest kernel `COMPLETE`, ba dataset `ready` và quota GPU đủ cho một phiên dự kiến.
2. Người dùng đã xác nhận đúng payload; gói mới đã upload lên private dataset `ngctnnguyn1111/doantnnt-b0-full-resume`.
3. Manifest/inventory đã được tải lại và đối chiếu private/ready/checksum đạt trước khi gửi kernel.
4. Chỉ bộ kernel khóa tại `.kaggle/resume/session-5-kernel/` được dùng; private kernel version 18 đã gửi đúng một lần.

Output `kaggle_outputs/b0-full-session-5-v18/` có 103/103 tệp sạch. Đọc đủ 48 generation; 39 generation nhập từ v17 giữ nguyên từng byte. Latest đã kiểm chứng ở bước 2.560/epoch 2/cursor 574 với 20.477 lượt mẫu; best/history/threshold bước 1.986 không đổi và không có development evaluation mới. Xem `reports/b0_full_session_5_verification.*`.
