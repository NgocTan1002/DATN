# Chuẩn bị B0 full training phiên 4 — ngày 27/10/2026

- Trạng thái: preflight cục bộ/từ xa đạt; private kernel version 17 đã gửi thành công và được quan sát `RUNNING` lúc 14:51 UTC+7. Chưa có output hoặc bước training mới được kiểm chứng.
- Mục tiêu: resume bước 1.536, hoàn tất epoch 1 ở bước 1.986, đánh giá đủ 4.115 development và tối đa dừng ở bước 2.048.
- Input gồm 28 generation/57 file checkpoint, pointer và state; tổng 169,861,762 byte.
- Gói cục bộ gồm manifest/metadata có 59 file, tổng 169,877,078 byte. Metadata tạo dataset không phải file dữ liệu upload.
- Manifest chuyển phiên SHA-256: `b11075f5f1f0524b09f114667d5e7420767288727c0b79bf6586f47eeaf29551`.
- Source bundle SHA-256: `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`; source tree `a46681120d36511e5d1bc7d5290e5488eecf136bf1ad55a93052afdb0e788925`; cả 56 file source huấn luyện giữ nguyên.
- Restore gói thật và bootstrap CLI nhúng đều đạt; model/optimizer/sampler/RNG/controller khớp payload gốc.
- Profile `b0-full-resume-session-4` có `--resume`, expected input step 1.536 và output report riêng cho phiên 4.
- Kernel private, internet tắt, gắn data/source/checkpoint dataset. Payload chỉ gồm checkpoint/trạng thái/manifest/metadata; không có audio, log hoặc thông tin đăng nhập.
- Kiểm tra trực tuyến trước upload: kernel mới nhất `COMPLETE`; checkpoint/source/data dataset đều `ready`; quota còn 24,46/30,00 giờ GPU, refresh theo CLI lúc `2026-10-31T00:00:00`.
- Lượt upload đầu bị bộ duyệt an toàn chặn vì xác nhận trước chỉ bao phủ các payload cũ. Sau khi người dùng xác nhận cụ thể gói bước 1.536, upload hoàn tất; remote private/ready, đủ 58 file/169.876.829 byte và manifest checkpoint/source khớp.
- Không sửa code nên không chạy lại unit suite; cổng 179/179 trước đó giữ nguyên.

## Cổng trước khi gửi

1. Theo dõi version 17 mỗi 20–30 phút; không gửi kernel khác khi trạng thái còn `QUEUED`/`RUNNING`.
2. Khi `COMPLETE`, tải output vào thư mục riêng `kaggle_outputs/b0-full-session-4-v17`.
3. Kiểm chứng workflow, bootstrap input bước 1.536, development đủ 4.115 mẫu tại bước 1.986, best/history/threshold và checkpoint dừng dự kiến tối đa bước 2.048.
4. Nếu `ERROR`/`CANCELLED`, giữ log/output và xác định generation cuối đã commit trước khi chạy lại.

Điểm đã kiểm chứng gần nhất vẫn là bước 1.536. Version 17 đang chạy; các mốc 1.986/2.048 và checkpoint best đầu tiên là kỳ vọng để đối chiếu, chưa phải kết quả thực tế.
