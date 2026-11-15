# Preflight phiên B0 thứ tám

## Kết luận

Preflight cục bộ và từ xa **ĐẠT**; private kernel version 21 đã được gửi đúng một lần và quan sát `RUNNING`. Output Kaggle version 20 đã được kiểm chứng làm input: latest bước 3.584, epoch 2/cursor 1.598, 28.669 lượt mẫu train và 66 generation. Best vẫn là generation `g-c7418c4439ac45e9b051fa77b768709d` tại bước 1.986; chưa có development evaluation mới và test chưa được dùng.

Gói exact-resume chứa 133 tệp pointer/checkpoint/state của 66 generation, tổng 465.837.119 byte. Cộng manifest, phần sẽ xuất hiện trong private checkpoint dataset là 134 tệp/465.871.883 byte. Manifest SHA-256: `ec65e82c8e326f0a2d212fc618d194c2315614bcd1f7dcaf50580218ff8db67f`. Gói cục bộ kể cả metadata Kaggle có 135 tệp/465.872.132 byte; không chứa audio, log hoặc thông tin đăng nhập.

Restore thử đã xác nhận đúng bước 3.584 và đủ 133 tệp; model, optimizer, sampler, RNG, controller, history và best được bảo toàn. Phiên B0 tiếp tục dùng nguyên source khóa 56 tệp, bundle `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1` và source tree `a46681120d36511e5d1bc7d5290e5488eecf136bf1ad55a93052afdb0e788925`. Source làm việc hiện tại có thêm phần D024 nhưng **không được publish** vào chuỗi B0 này.

Bộ kernel khóa tại `.kaggle/resume/session-8-kernel/` dùng profile `b0-full-resume-session-8`, có `--resume`, expected input 3.584, private, GPU bật, internet tắt và gắn đúng data/source/checkpoint dataset. Run config nhúng, bootstrap nhúng và checksum ba tệp kernel đều khớp.

Preflight trực tuyến ghi nhận latest kernel `COMPLETE`; ba dataset đều private/`ready`. Quota GPU còn 15,49/30,00 giờ và làm mới lúc 00:00 ngày 31/10/2026. Cảnh báo Kaggle CLI 2.2.4 cũ hơn 2.2.5 không chặn chạy.

## Hành vi dự kiến của phiên 8

- Sau 388 bước, training chạm cuối epoch 2 tại bước 3.972 và đánh giá đủ 4.115 development.
- Nếu EER epoch 2 không thấp hơn nghiêm ngặt `0,08595800524934383`, D023 dừng sớm tại bước 3.972 và giữ best sớm nhất.
- Nếu EER cải thiện, best/threshold được cập nhật và phiên tiếp tục trong giới hạn 512 bước đến tối đa bước 4.096, tức epoch 3/cursor 124.
- Không có nhánh nào dùng test để chọn checkpoint, threshold hoặc thời điểm dừng.

## Upload và gửi kernel

Sau xác nhận cụ thể của người dùng, payload đã được upload lên private dataset `ngctnnguyn1111/doantnnt-b0-full-resume` thành version 7. Dataset chuyển `ready`; inventory tải từ Kaggle khớp đúng 134 tệp/465.871.883 byte, không thiếu/thừa/sai kích thước. Manifest tải ngược có SHA-256 `ec65e82c8e326f0a2d212fc618d194c2315614bcd1f7dcaf50580218ff8db67f`; source manifest tải ngược vẫn có 56 tệp và bundle SHA-256 đã khóa.

Private kernel version 21 được gửi đúng một lần từ `.kaggle/resume/session-8-kernel/` và quan sát `RUNNING` lúc 15:56 UTC+7. Quota sau gửi còn 15,48/30,00 giờ GPU. Chưa có output hoặc kết quả development epoch 2; cần theo dõi đến `COMPLETE`, sau đó tải output vào thư mục version 21 riêng và kiểm chứng trước mọi bước tiếp theo.
