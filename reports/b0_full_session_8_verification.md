# Kiểm chứng B0 full session 8 — Kaggle version 21

## Kết luận

ĐẠT. Version 21 đã resume chính xác từ bước 3.584, hoàn thành 512 bước mới và dừng `paused` tại bước 4.096. Epoch 2 được đánh giá trên đủ 4.115 mẫu development ở bước 3.972.

EER giảm từ `8,59580052%` ở epoch 1 xuống `8,33333333%` ở epoch 2, tốt hơn `0,26246719` điểm phần trăm. Theo quy tắc cải thiện nghiêm ngặt D023, checkpoint epoch 2 trở thành best mới; early stopping không kích hoạt và B0 phải tiếp tục tới đánh giá epoch 3.

## Metric checkpoint được chọn

- Best: epoch 2, bước 3.972, generation `g-4174b60757a44324a9b864a4931e7a18`.
- Development loss: `0,634388257066786`.
- EER: `0,08333333333333333`.
- ROC-AUC: `0,975126649790158`.
- Macro-F1: `0,9116562633142766`.
- Threshold development: `0,0070081922846535845`.

## Toàn vẹn

- Output sạch: 161 tệp, 559.563.541 byte; không có tệp rỗng hoặc HTML/502.
- 77/77 generation đọc được và nhất quán; 66 generation nhập từ version 20 giữ nguyên từng byte.
- Prefix 3.584 loss/batch, model, 12 trạng thái Adam, sampler, controller, history và hai CUDA RNG đều đạt.
- Metric epoch 2 được tính lại từ score/label của đủ 4.115 mẫu và khớp report.
- Test set không được dùng.

## Bước tiếp theo

Phiên 9 phải tiếp tục từ latest bước 4.096 tới tối đa 4.608. Biên epoch 3 ở bước 5.958, còn 1.862 bước; chỉ khi tới biên này mới chạy full development lần thứ ba và chốt best cuối theo EER.
