# Gói source D023 đã xác nhận và upload

- Đích: private dataset `ngctnnguyn1111/doantnnt-project-source` (metadata hiện tại xác nhận isPrivate=true).
- Kernel dự kiến: private `ngctnnguyn1111/doanttnt-kaggle-cli-runner`, profile `b0-full-smoke`, GPU bật, internet tắt.
- ZIP: 184609 byte, 56 file; SHA-256 `b016111a9e4081edfb756465dd7f8654e5796fc395e6522dff6f547a70a82f61`.
- Nội dung: source Python, configs JSON, runner và requirements-kaggle; không có audio, data, checkpoint, log, report, DOCX, .kaggle, .env hoặc file credentials.
- Toàn bộ checksum từng file trong ZIP đã đối chiếu manifest. Snapshot pilot cũ đã giữ riêng cục bộ.
- Bộ duyệt tự động từ chối lần đầu trước thực thi; sau đó người dùng đã xác nhận cụ thể payload/đích, smoke CUDA và cập nhật readiness/gửi phiên đầu nếu đạt. Upload đã thành công, metadata sau upload vẫn private; private kernel v13 COMPLETE và comparator CUDA đã đạt, kiểm tra lại generation tải về cũng khớp.

## File mới so với source pilot đã khóa

- `configs/b0_full_training.json`
- `configs/xlsr_smoke.json`
- `scripts/check_b0_full_smoke.py`
- `scripts/estimate_b0_full_training_budget.py`
- `scripts/smoke_test_xlsr.py`
- `scripts/train_b0_full.py`
- `src/models/xlsr.py`
- `src/training/full_training.py`

## File có sẵn bị đổi

[]

Danh sách 56 file và checksum chi tiết tại `b0_full_source_preflight.json`.

## Kích hoạt và phiên đầu

- Cờ readiness đã mở sau cổng CPU/CUDA; chỉ configs/b0_full_training.json đổi trạng thái triển khai/readiness, mọi Python và trường khoa học D023 giữ nguyên.
- Gói kích hoạt SHA-256 `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1` đã upload theo quyền người dùng xác nhận trước đó.
- Private kernel v14, profile b0-full-first-session, đã gửi thành công và quan sát RUNNING; tối đa 512 optimizer step, fresh init peak, chưa có kết quả phiên để kiểm chứng.
- Quota ngay trước phiên đầu còn 29,97 GPU-giờ. Không dựa quota này để cam kết đủ toàn bộ trần 31,55 giờ gồm dự phòng; kiểm tra lại trước từng phiên resume.

## Kiểm chứng phiên đầu ngày 25/10

Version14 đã COMPLETE, runner paused512 đúng giới hạn; checkpoint/generation đã kiểm chứng. Xem `b0_full_session_1_verification.{json,md}`. Trạng thái RUNNING ở phần gửi phiên là lịch sử, không còn là trạng thái hiện hành. Chưa gửi phiên2.
