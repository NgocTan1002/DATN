# Báo cáo preflight exact-resume XLS-R

Ngày kiểm tra: 30/10/2026  
Trạng thái: **chỉ là cổng kỹ thuật, không phải kết quả khoa học**.

## Kết quả

- Ứng viên kỹ thuật `facebook/wav2vec2-xls-r-300m` được ghim ở revision bất biến `1a640f32ac3e39899438a2931f9924c02f080a54` cùng checksum config/preprocessor. Chưa tải hoặc băm trọng số.
- Checkpoint XLS-R ghi atomic, chỉ lưu tham số trainable cùng optimizer, progress, sampler và Python/NumPy/PyTorch/CUDA RNG. Identity khóa nguồn backbone, dữ liệu, tiền xử lý và cấu hình huấn luyện cần giữ khi resume.
- Đối chứng CPU ngắt–resume khớp tuyệt đối chuỗi loss và tham số cuối. Nhóm test mới đạt 4/4 trong 4,501 giây; 12/12 test mô hình XLS-R đạt trong 29,073 giây.
- Toàn bộ suite đạt 183/183 trong 186,276 giây khi `TEMP`/`TMP` trỏ vào thư mục trong workspace. `compileall` và parse cấu hình JSON đạt.

## Phạm vi

Không dùng audio thật, shard thật hoặc test set; không tải pretrained weights, chạy CUDA/VSASV, gửi Kaggle XLS-R hay sửa `configs/audio.json`. Lượt chạy suite đầu với thư mục tạm mặc định bị quyền Windows chặn và được chạy lại hợp lệ trong workspace; đây không phải lỗi kiểm thử sản phẩm.

## Cổng còn mở

Phải quyết định normalization waveform, số tầng X1, head/dropout, optimizer/learning rate, ngân sách/dừng và tích hợp checkpoint vào entrypoint/controller/history/best trước pretrained GPU smoke. Không được xem revision ứng viên là lựa chọn khoa học cuối khi các cổng này chưa đạt.
