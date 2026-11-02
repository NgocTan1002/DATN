# Kiểm chứng B0 full training phiên 3 — Kaggle v16

- Kết quả ĐẠT: workflow complete, cả 3 lệnh mã 0; log xác nhận khôi phục checkpoint bước 1.024 và kết thúc paused ở bước 1.536.
- GPU resume 1.024 → 1.536, thêm 512 bước, epoch 0/cursor 1.536. Tổng trace có 12.288 lượt mẫu train; phần 1.024 loss/batch từ v15 giữ nguyên, toàn bộ 1.536 batch khớp sampler seed 2026.
- Phạm vi: 15.885 train/4.115 development trên manifest 20.000 mẫu/66 shard, pool cục bộ 68 shard; source 56 file, identity và D023 không đổi.
- Đọc lại 28 generation; 19 generation nhập từ v15 giữ nguyên từng byte. Pointer, checkpoint, state, report và controller khớp; model và 12 Adam state hữu hạn, Adam step 1.536, trọng số thay đổi so với bước 1.024, đủ 2 CUDA RNG state. JSON đi kèm lưu từng hash.
- Chưa hết epoch đầu 1.986 bước, chưa có development evaluation, best hoặc threshold; đây là điểm chia phiên dự kiến.
- Source bundle SHA-256 `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`.
- Latest generation `g-794ca799c6fb4cf6b988a8a1c920dcc4`; checkpoint SHA-256 `3cc00d46d9a4676950eb6bcd3c9a4ae6006ebadcdcc52418f3e44d4eda6d8772`.
- Runtime training report: 1,762 giờ trên Tesla T4; workflow chạy từ 09:53:16 đến 11:39:54 ngày 26/10/2026 (UTC+7).
- Kiểm chứng offline trên output người dùng đã tải, khớp profile/lệnh đã gửi và log; không tải lại, kiểm tra API mới, chạy thêm GPU/training/unit suite hoặc đối chứng liên tục 1.536 bước trong phiên này. Cổng 179/179 và exact resume CPU/CUDA nhỏ từ phiên trước là bằng chứng riêng.

## Bước tiếp theo

Chuẩn bị checkpoint từ v16 cho phiên 4, giữ source/identity D023. Còn 450 bước đến mốc 1.986; phải đánh giá đủ 4.115 development và commit best đầu tiên trước batch epoch 2. Nếu invocation hoàn tất thêm 512 bước thì mốc dự kiến là 2.048; đây là kỳ vọng, chưa phải kết quả. Chưa upload input mới hoặc gửi phiên 4.

Giữ toàn bộ `kaggle_outputs/b0-full-session-3-v16/checkpoints/b0_peak_full_v1/` ngoài Git. Report gốc, bootstrap, workflow và runtime đã lưu trong reports/. Không cần theo dõi v16 nữa; chưa có kết quả B0 cuối hoặc đánh giá test.
