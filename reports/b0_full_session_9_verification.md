# Kiểm chứng B0 đầy đủ — phiên 9 / Kaggle version 22

## Kết luận

**ĐẠT.** Output version 22 tiếp tục chính xác chuỗi B0 D023 từ bước 4.096 đến 4.608 và dừng ở trạng thái `paused`. Không có development evaluation mới, không đổi best và không dùng test.

## Bằng chứng chính

- Output `kaggle_outputs/b0-full-session-9-v22/` có 179 tệp, 635.665.317 byte; không có tệp rỗng, HTML hoặc lỗi 502.
- Workflow `complete`; cả ba lệnh trả mã 0. Bootstrap xác minh và khôi phục đúng bước 4.096 từ 155 tệp checkpoint.
- Training CUDA chạy thêm đúng 512 optimizer step đến bước 4.608. Sampler ở epoch 3 theo cách gọi của báo cáo (chỉ số nội bộ 2), cursor 636; tổng 36.858 lượt mẫu train.
- Cả 86 generation đều nạp được. 77 generation nhập từ version 21 giữ nguyên từng byte; 9 generation mới nằm tại các mốc 4.160–4.608.
- Prefix 4.096 loss và batch khớp tuyệt đối version 21. Model/optimizer hữu hạn; 12 trạng thái Adam cùng ở bước 4.608; có đủ hai CUDA RNG state.
- Latest là `g-a4e2c835d3914007a0b01936dbadaa15`, checkpoint SHA-256 `60ba1744ef2065a12cd897fa9023978c4fa34faaf0753173fc322c7b29122974`.
- Best vẫn là `g-4174b60757a44324a9b864a4931e7a18` tại bước 3.972. Development epoch 2 giữ EER 8,33333333%, loss 0,63438826, ROC-AUC 0,97512665, macro-F1 0,91165626 và threshold 0,007008192285.
- `bad_evaluations=0`, chưa có lý do dừng; epoch 3 chưa hoàn tất. Test không được dùng.

## Bước tiếp theo

Phiên 10, nếu được người dùng xác nhận, tiếp tục bước 4.608 → tối đa 5.120. Mốc này vẫn chưa chạy development; biên epoch 3 và full development cuối nằm ở bước 5.958.
