# Sự cố B0 đầy đủ — phiên 11 / kernel v24

## Kết luận

Version 24 kết thúc với trạng thái `ERROR` trước khi huấn luyện bắt đầu. GPU hoạt động bình thường, không có optimizer step, checkpoint hoặc development evaluation mới. Điểm hợp lệ gần nhất vẫn là version 23 ở bước 5.120.

## Nguyên nhân gốc

`runner.py` trong bộ kernel phiên 11 vẫn nhúng cấu hình của phiên 10. Runner ưu tiên cấu hình nhúng này thay vì `run-config.json` đi kèm, nên v24 yêu cầu manifest `f7b8ccbdf7b03402324699a92f86b2018d67e393e0237adc66f25c48e15ebde1` và bước 4.608. Dataset checkpoint version 10 thực tế chứa đúng manifest phiên 11 `189c5472cd2faf5cbc588f01284792f640f7c3b3deb2e8ff5beabfced34fe831` ở bước 5.120. Cổng checksum vì vậy dừng an toàn với lỗi `Checksum manifest checkpoint đầu vào không khớp.`

Mã nguồn tải trực tiếp từ v24 có SHA-256 `d57b493bbbe6794a20bc501fecb66d00146ea84c1192f16d42d5ead734481e12`, trùng runner cục bộ. Điều này loại trừ giả thuyết Kaggle dùng cache hoặc nhận nhầm mã nguồn. Dataset đã được xác minh từ xa trước khi gửi và không cần upload lại.

## Bằng chứng thực thi

- Workflow chạy từ 21:43:05 đến 21:43:27 UTC+7, trạng thái `failed`.
- Kiểm tra CUDA trả mã 0 trong 11,065 giây; runtime thấy 2 Tesla T4.
- Bootstrap restore trả mã 1 trong 2,694 giây.
- Chỉ có 3 tệp output, tổng 53.174 byte: log, workflow report và runtime report.
- Không có resume report, training report hoặc thư mục checkpoint output; lệnh training chưa được gọi.

## Bước tiếp theo

Đã hoàn thành ngày 02/11/2026: thêm cổng hồi quy và lệnh gửi an toàn, tái tạo runner đúng phiên 11, đạt 12/12 kiểm thử workflow và toàn suite 191/191 trong 152,672 giây. Runner version 25 kéo ngược từ Kaggle trùng SHA-256 `abab4fe7b3d52a5dc1ac9902572e54d54c12f86c3b5cc04e20fbeb1c7d083030` với bản cục bộ và nhúng đúng checksum/bước 5.120. Private kernel version 25 đã gửi, quan sát `RUNNING`; checkpoint dataset version 10 không được upload lại.
