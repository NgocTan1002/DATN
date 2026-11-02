# Kiểm chứng B0 full training phiên 4 — Kaggle v17

- Kết quả **ĐẠT**: workflow hoàn tất, cả 3 lệnh trả mã 0; log xác nhận khôi phục bước 1.536, đánh giá development cuối epoch 1 và dừng an toàn tại bước 2.048.
- Resume GPU 1.536 → 2.048, thêm 512 bước. Sampler ở epoch 2, cursor 62; tổng trace là 2.048 batch / 16.381 lượt mẫu train (epoch đầu có batch cuối 5 mẫu).
- Đã đánh giá đủ 4.115 mẫu `closed_dev` tại bước 1.986: loss `0.677994`, EER `8.595801%`, ROC-AUC `0.970242`, macro-F1 `0.908857`, threshold EER `0.009600887260`. Metric được tính lại từ labels/scores và khớp chính xác.
- Best đầu tiên: epoch 1, bước 1.986, generation `g-c7418c4439ac45e9b051fa77b768709d`. Latest: bước 2.048, generation `g-f0940a29982742ebb7cf7aa4a252bb26`.
- Phạm vi 15.885 train / 4.115 development trên manifest 20.000 mẫu / 66 shard; test set không được dùng. Source 56 file, identity và D023 không đổi.
- Đọc lại đủ 39 generation; 28 generation nhập từ v16 giữ nguyên từng byte. Pointer/checkpoint/state/controller khớp; model và 12 Adam state hữu hạn, Adam step 2.048, đủ 2 CUDA RNG state.
- Kiểm tra EER bất thường đã đạt: speaker overlap 0, file overlap 0, 20.000/20.000 content hash duy nhất, duplicate chéo split 0; metadata sample-rate gốc 16/40 kHz khớp và pipeline chuẩn hóa về 16 kHz, 4 giây. Audio config và amplitude policy `peak` giữ nguyên. Đây là kết quả development sau epoch đầu, chưa phải kết quả test hay kết luận cuối.
- Lần tải đầu có 2 tệp bị thay bằng trang HTML 502 (332 byte). Đã tải lại toàn bộ và xác minh 85/85 tệp; chỉ đúng 2 tệp đó khác hash, bản tải lại đọc được và không còn HTML. Dùng `kaggle_outputs/b0-full-session-4-v17-redownload` làm output chuẩn; giữ bản đầu để truy vết.
- Runtime training report `3.571` giờ trên `Tesla T4`; workflow `2026-10-27T07:51:40+00:00` → `2026-10-27T11:26:57+00:00`.

## Bước tiếp theo

Chuẩn bị checkpoint latest đã kiểm chứng của v17 cho phiên 5 và tiếp tục bước 2.048 → 2.560. Mốc 2.560 chỉ là kỳ vọng nếu phiên kế tiếp thành công; chưa upload checkpoint mới và chưa gửi phiên 5.

Giữ toàn bộ hai thư mục output và checkpoint ngoài Git. Report gốc, bootstrap, workflow, runtime và báo cáo kiểm chứng đã lưu trong `reports/`. Không cần theo dõi v17 nữa.
