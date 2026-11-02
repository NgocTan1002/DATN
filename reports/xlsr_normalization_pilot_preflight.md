# Preflight pilot normalization XLS-R D024

Ngày kiểm tra: 30/10/2026  
Trạng thái: **đặc tả và cổng CPU; chưa phải kết quả pilot khoa học**.

## Thiết kế đã khóa

- So sánh ghép cặp X0 trên đúng 15.885 train/4.115 development của manifest 20.000 mẫu/66 shard, seed `2026`; không dùng test.
- Hai nhánh giữ cùng resample 16 kHz, `peak`, đoạn 4 giây, file/thứ tự sampler, khởi tạo classifier và endpoint. Chỉ khác `pipeline_only` với chuẩn hóa từng đoạn `(x - mean) / sqrt(population_variance + 1e-7)`.
- Chọn development EER thấp hơn; hòa chính xác giữ `pipeline_only`. Threshold của mỗi nhánh được xác định riêng trên development.
- Config SHA-256: `18148e4b272b3c6335cbcb5d60852b159d6b4f1f8a3f4f243679d89f4a7b505a`.

## Kiểm chứng

- 5/5 kiểm thử D024 đạt trong 0,048 giây; kiểm tra công thức, nhánh giữ nguyên, đoạn hằng, dữ liệu đầu vào lỗi và khóa cấu hình development-only.
- 21/21 kiểm thử XLS-R liên quan đạt trong 16,347 giây.
- Toàn suite cuối đạt 188/188 trong 191,255 giây với thư mục tạm trong workspace và `PYTHONIOENCODING=utf-8`.
- `compileall`, parse hai file JSON và `git diff --check` đều đạt; diff chỉ có cảnh báo chuyển LF/CRLF.
- Checksum thực của `configs/audio.json` và manifest v2 khớp cấu hình. Kiểm thử mới không đọc audio/shard thật.

Lượt full suite trước đó thiếu UTF-8 nên ba test chuyển checkpoint B0 cũ lỗi khi in tiếng Việt trên Windows `cp1252`; chạy lại đúng môi trường đạt. Đây là lỗi môi trường chạy kiểm thử, không phải lỗi normalization hoặc checkpoint.

## Cổng còn đóng

`ready_for_gpu_smoke=false` và `ready_for_pilot_training=false`. Chưa tải/băm trọng số, load pretrained offline, smoke CUDA, khóa batch/ngân sách chung hay tích hợp entrypoint/controller/history/best. Không sửa `configs/audio.json`, không upload và không gửi kernel XLS-R.
