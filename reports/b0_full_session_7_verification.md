# Kiểm chứng B0 full training phiên 7 — Kaggle version 20

## Kết luận

**ĐẠT.** Kaggle API xác nhận version 20 `COMPLETE`; output được tải vào `kaggle_outputs/b0-full-session-7-v20/` và kiểm chứng sạch. GPU exact-resume đúng từ bước 3.072 đến 3.584 rồi dừng `paused` theo giới hạn 512 bước của phiên.

Đây chưa phải kết quả B0 cuối: epoch 2 chưa kết thúc, chưa có development evaluation mới và chưa có quyết định early stopping. Test set không được dùng.

## Output và workflow

- 139 tệp, 467.526.970 byte; không có tệp rỗng, HTML hoặc 502.
- Workflow `complete`; runtime check, bootstrap và training đều trả mã 0.
- Bootstrap kiểm chứng đúng input bước 3.072, 115 tệp checkpoint/pointer/state, manifest chuyển phiên `688af75ea12f99df92af5e29b25858534d5cab1e324d358e44ecc30970886369`.
- Log xác nhận khôi phục bước 3.072 và kết thúc `paused, step=3584`.

## Exact-resume và checkpoint

- Latest: generation `g-e1fbbfcb17644703a8f530c4c7ce0300`, checkpoint SHA-256 `e3133e30ad9e0def016f9020ac61cdaaa0130266e773e3788bdc66dedd587f4f`.
- Best giữ nguyên: generation `g-c7418c4439ac45e9b051fa77b768709d`, epoch 1/bước 1.986.
- 66/66 generation đạt kiểm tra checkpoint version, controller/state, pointer/hash, identity, model, 12 Adam state, sampler và hai CUDA RNG state.
- 57 generation nhập từ output version 19 giữ nguyên từng byte; có 9 generation mới tại các bước 3.136, 3.200, 3.264, 3.328, 3.392, 3.456, 3.520 và hai commit ở bước 3.584.
- Prefix 3.072 loss và batch trace khớp hoàn toàn output version 19.

## Tiến độ thực tế

- Global step: 3.584; sampler epoch index 1, tương ứng epoch 2/cursor 1.598.
- Tổng lượt mẫu train: 28.669.
- Runtime training: 1,769715 giờ trên Tesla T4; PyTorch `2.11.0+cu128`, CUDA 12.8, cuDNN 91900.
- Chưa có evaluation mới. Metric/best/threshold epoch 1 được bảo toàn: loss `0,6779944593`, EER `8,59580052%`, ROC-AUC `0,9702422377`, macro-F1 `0,9088565065`, threshold `0,009600887260`.
- Identity tiếp tục khóa manifest 20.000 mẫu/66 shard, 15.885 train/4.115 development, seed `2026`, policy `peak`; test không dùng.

## Bước kế tiếp

Phiên 8 phải resume từ latest bước 3.584. Sau 388 bước sẽ chạm biên epoch 2 ở bước 3.972 và chạy full development trước bước train tiếp theo:

- nếu EER không giảm nghiêm ngặt so với epoch 1: early stop tại 3.972;
- nếu EER cải thiện: cập nhật best/threshold rồi tiếp tục tối đa đến bước 4.096 trong giới hạn 512 bước của phiên.

Chỉ chuẩn bị/upload checkpoint mới sau khi payload từ output này được đóng gói, restore/bootstrap thử và người dùng xác nhận cụ thể.
