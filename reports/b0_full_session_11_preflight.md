# Preflight B0 đầy đủ — phiên 11

## Trạng thái

**ĐÃ SỬA VÀ GỬI LẠI.** Version 24 `ERROR` vì runner còn nhúng run config phiên 10. Ngày 02/11/2026, đã bổ sung cổng so khớp runner/run-config, tái tạo runner đúng phiên 11 và gửi private kernel version 25; trạng thái ngay sau gửi là `RUNNING`. Gói checkpoint từ version 23 tại bước 5.120 và private dataset version 10 được dùng lại, không upload lại.

## Gói checkpoint

- Thư mục: `.kaggle/resume/session-11-dataset/`.
- 95 generation, 191 tệp checkpoint/pointer/state với 710.652.723 byte.
- Phần upload gồm 192 tệp/710.702.335 byte; manifest SHA-256 `189c5472cd2faf5cbc588f01284792f640f7c3b3deb2e8ff5beabfced34fe831`.
- Identity SHA-256 `c6bd98ff2fee63af52d0d1d059f6cf788acc8462b140dd26672905b4d1c3e735` giữ nguyên chuỗi D023.
- Không có audio, log hoặc thông tin đăng nhập. Restore thử đạt bước 5.120 và kiểm chứng đủ 191 tệp.
- Bộ 7 kiểm thử chuyển checkpoint đạt 7/7 trong 10,520 giây khi dùng thư mục tạm trong workspace.

## Kernel khóa

- Profile `b0-full-resume-session-11`, private, GPU bật, internet tắt.
- Gắn đúng ba private dataset: dữ liệu development, source D023 và checkpoint resume.
- Bootstrap nhúng khớp `deploy/kaggle/resume_checkpoint.py`. Runner v24 cũ không khớp run config; runner đã sửa có SHA-256 `abab4fe7b3d52a5dc1ac9902572e54d54c12f86c3b5cc04e20fbeb1c7d083030` và nhúng đúng profile phiên 11, manifest phiên 11 cùng expected step 5.120.
- Workflow cục bộ nay có cổng `verify-prepared-kernel`/`push-prepared-kernel`; nếu runner lệch `run-config.json`, lệnh gửi dừng trước khi gọi Kaggle. Nhóm kiểm thử workflow đạt 12/12; toàn bộ suite đạt 191/191 trong 152,672 giây với UTF-8 và thư mục tạm workspace.
- Source B0 vẫn là bộ khóa 56 tệp, bundle SHA-256 `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`. Source D024 hiện hành trong workspace không được đưa vào chuỗi B0.

## Preflight Kaggle

- Kernel version 23 `COMPLETE`; ba dataset đều `ready`.
- Quota GPU còn 26,38/30,00 giờ, làm mới ngày 07/11/2026.
- Kaggle CLI 2.2.4 chỉ phát cảnh báo có 2.2.5; cảnh báo không chặn chạy.

## Kết quả upload và gửi chạy

- Người dùng đã xác nhận cụ thể payload phiên 11.
- Payload được upload thành private checkpoint dataset version 10; remote `ready`.
- Inventory từ xa khớp đúng 192 tệp/710.702.335 byte, không thiếu, không thừa và không lệch kích thước.
- Manifest tải ngược trùng từng byte với bản cục bộ và có SHA-256 `189c5472cd2faf5cbc588f01284792f640f7c3b3deb2e8ff5beabfced34fe831`.
- Private kernel version 24 được gửi đúng một lần và ban đầu quan sát `RUNNING` lúc 21:43 UTC+7. Sau đó kernel chuyển `ERROR`.

## Chẩn đoán version 24

- CUDA đạt trên 2 Tesla T4; lỗi không thuộc GPU.
- Workflow thực tế báo profile `b0-full-resume-session-10`, manifest `f7b8ccbdf7b03402324699a92f86b2018d67e393e0237adc66f25c48e15ebde1` và expected step 4.608.
- Dataset version 10 chứa đúng manifest phiên 11 `189c5472cd2faf5cbc588f01284792f640f7c3b3deb2e8ff5beabfced34fe831` ở bước 5.120, nên cổng an toàn báo checksum không khớp.
- Lệnh training chưa chạy; không có checkpoint hoặc metric mới. Mã runner tải trực tiếp từ v24 trùng SHA-256 với runner cục bộ, vì vậy không phải cache Kaggle.
- Bằng chứng chi tiết: `reports/b0_full_session_11_v24_failure.*`.

## Gửi lại version 25

- Dataset checkpoint version 10 vẫn `ready`; không upload lại payload.
- Kernel được gửi qua cổng xác minh mới, vẫn private/GPU/offline và gắn đúng ba dataset.
- Kaggle xác nhận `Kernel version 25 successfully pushed`; API quan sát `RUNNING`.
- Mã runner kéo ngược từ version 25 trùng SHA-256 với runner cục bộ và giải mã ra đúng profile `b0-full-resume-session-11`, manifest `189c5472cd2faf5cbc588f01284792f640f7c3b3deb2e8ff5beabfced34fe831`, expected step 5.120.

## Kỳ vọng phiên 11

Version 25 đang thực hiện kỳ vọng 5.120 → 5.632, sampler epoch 3/cursor 1.660 và 45.050 lượt mẫu train nếu chạy đủ 512 bước. Điểm đã kiểm chứng hiện vẫn là bước 5.120 cho tới khi output v25 được tải và xác minh.

Theo dõi duy nhất version 25. Khi `COMPLETE`, tải output vào `kaggle_outputs/b0-full-session-11-v25/` rồi kiểm chứng; nếu `ERROR`/`CANCELLED`, giữ output/log và không tự gửi lại.
