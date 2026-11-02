# Kiểm chứng B0 full training phiên 1 — Kaggle v14

- Kết quả kiểm chứng: ĐẠT; workflow complete, hai return code 0.
- Runner paused đúng điểm chia phiên; đã hoàn thành 512/5.958 bước trần, cursor 512/epoch 0, 4.096 lượt mẫu train. Chưa hết epoch đầu, chưa đánh giá development hoặc chọn best.
- Phạm vi: manifest 20.000 mẫu/66 shard (15.885 train, 4.115 development); pool cục bộ 68 shard.
- Runtime training report: 1.956 giờ trên Tesla T4.
- Source SHA-256: `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`.
- Checkpoint latest SHA-256: `71464c4f3ebabfc6f3634c7963ced14d8056dc5ad056e6284ee1883fe46c6872`.
- Cả 10 generation đã đọc lại: bước 0,64,128,192,256,320,384,448,512,512; payload/controller/state nhất quán. Pointer/checkpoint/state checksum khớp report.
- Chuỗi 512 batch khớp sampler seed 2026, 4.096 index không lặp trong prefix epoch. Model và optimizer hữu hạn; 12 optimizer state ở bước 512, có Python/NumPy/PyTorch/2 CUDA RNG state.
- Identity/config/manifest/source tree khớp mã và cấu hình hiện hành; không dùng test, không đổi D023.
- Đây là kiểm chứng artifact tải về; không chạy lại GPU resume hoặc huấn luyện liên tục 512 bước để so sánh trong phiên này. Cổng kỹ thuật exact resume CPU/CUDA đã có từ v13.

## Bước tiếp theo

Chuẩn bị input checkpoint private, giữ đúng source/identity và tiếp tục 512 → 1.024 (tối đa 512 bước ở phiên 2). Giữ toàn bộ generation cục bộ; không gửi lại first-session hoặc dùng threshold/metric pilot thay kết quả full training. Chưa gửi phiên 2 trong phiên kiểm chứng này.

Report nguồn: `b0_peak_full_session_1.json`, workflow/runtime tương ứng; checkpoint/log ở `kaggle_outputs/b0-full-session-1-v14/`, ngoài Git.
