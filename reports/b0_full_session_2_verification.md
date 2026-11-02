# Kiểm chứng B0 full training phiên 2 — Kaggle v15

- Kết quả: ĐẠT; v15 COMPLETE, workflow complete, cả 3 lệnh trả mã 0.
- Bootstrap khôi phục đúng checkpoint bước 512 từ v14; lệnh training dùng resume, hoàn thành thêm 512 bước trên CUDA và paused đúng bước 1.024/cursor 1.024, epoch 0.
- Phạm vi: manifest 20.000 mẫu/66 shard, gồm 15.885 train và 4.115 development; pool cục bộ 68 shard. Tổng trace có 8.192 lượt mẫu train.
- Controller, history và best đúng lịch D023: chưa hết epoch đầu 1.986 bước, chưa đánh giá development hoặc tạo best/threshold.
- 19 generation được đọc lại: 10 generation nhập từ v14 giữ nguyên từng byte; 9 generation mới ở các bước 576,640,704,768,832,896,960,1.024,1.024. Pointer/checkpoint/state hash khớp. Chi tiết từng hash trong JSON đi kèm.
- Prefix 512 loss/batch của v14 giữ nguyên; toàn bộ 1.024 batch khớp sampler seed 2026, không lặp index trong prefix epoch. Model/optimizer hữu hạn, 12 Adam state đều bước 1.024; trọng số đã thay đổi so với bước 512. Có đủ Python/NumPy/PyTorch/2 CUDA RNG state.
- Source/manifest/plan/identity giữ nguyên; 56 file source hiện hành khớp fingerprint checkpoint. Source bundle SHA-256 `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`.
- Latest checkpoint SHA-256 `40812f8331d4b4d615ce91c6af4b72c863a4e17cb946041c42f8e94686c620ca`; generation `g-9de1bc836e934c1ab892bdf6683afaec`.
- Runtime training report: 1.746 giờ trên Tesla T4. Workflow 2026-10-25T06:56:59+00:00 → 2026-10-25T08:42:24+00:00.
- Đây là kiểm chứng output của phiên GPU đã hoàn thành; không chạy đối chứng liên tục 1.024 bước, không chạy thêm training/GPU hoặc unit suite trong phiên kiểm chứng. Cổng exact resume CPU/CUDA nhỏ và 179/179 test trước đó là bằng chứng riêng.

## Bước tiếp theo

Chuẩn bị checkpoint từ v15 và phiên tiếp tục 1.024 → 1.536 cùng source/identity D023. Chưa gửi phiên 3. Không cần theo dõi v15 nữa; giữ toàn bộ `kaggle_outputs/b0-full-session-2-v15/checkpoints/b0_peak_full_v1/` ngoài Git.

Báo cáo gốc: `b0_peak_full_session_2.json`, `b0_full_session_2_resume_input.json`, `b0_full_session_2_kaggle_workflow.json` và `b0_full_session_2_kaggle_runtime.json` trong reports/. Kết quả này chưa phải kết quả B0 cuối hoặc đánh giá test.
