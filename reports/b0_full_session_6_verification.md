# Kiểm chứng B0 full training phiên 6 — Kaggle v19

- Kết quả **ĐẠT**: workflow hoàn tất, cả 3 lệnh trả mã 0; log xác nhận khôi phục bước 2.560 và dừng an toàn `paused` tại bước 3.072.
- Resume GPU 2.560 → 3.072, thêm 512 bước. Sampler ở epoch 2, cursor 1.086; tổng trace là 3.072 batch / 24.573 lượt mẫu train.
- Phiên này chưa đến cuối epoch 2 nên không chạy development mới. History vẫn có đúng một evaluation epoch 1 tại bước 1.986; metric được tính lại và giữ nguyên: loss `0.677994`, EER `8.595801%`, ROC-AUC `0.970242`, macro-F1 `0.908857`, threshold EER `0.009600887260`.
- Best vẫn là epoch 1/bước 1.986, generation `g-c7418c4439ac45e9b051fa77b768709d`. Latest mới là bước 3.072, generation `g-a4b58c79412c435994f4ce0188b168b8`.
- Phạm vi 15.885 train / 4.115 development trên manifest 20.000 mẫu / 66 shard; test set không được dùng. Source 56 tệp, identity và D023 không đổi.
- Đọc lại đủ 57 generation; 48 generation nhập từ v18 giữ nguyên từng byte. Pointer/checkpoint/state/controller khớp; model và 12 Adam state hữu hạn, Adam step 3.072, đủ 2 CUDA RNG state.
- Kiểm tra EER bất thường vẫn đạt: speaker/file overlap 0, 20.000/20.000 content hash duy nhất, duplicate chéo split 0; metadata sample-rate gốc 16/40 kHz khớp và pipeline vẫn chuẩn hóa về 16 kHz, 4 giây, policy `peak`. Đây chưa phải kết quả test hay kết luận B0 cuối.
- Output có 121/121 tệp sạch, không phát hiện HTML/502 và không cần tải lại. Thư mục chuẩn: `kaggle_outputs/b0-full-session-6-v19`.
- Runtime training report `1.785` giờ trên `Tesla T4`; workflow `2026-10-29T08:59:03+00:00` → `2026-10-29T10:46:39+00:00`.

## Bước tiếp theo

Latest bước 3.072 đã đủ điều kiện làm input cho phiên 7 sau khi đóng gói/preflight và có xác nhận riêng cho payload mới. Nếu tiếp tục tối đa 512 bước, mốc kế tiếp dự kiến là 3.584/epoch 2/cursor 1.598; full development epoch 2 chỉ diễn ra ở bước 3.972.

Giữ checkpoint/log lớn ngoài Git. Report gốc, bootstrap, workflow, runtime và báo cáo kiểm chứng đã lưu trong `reports/`. Không cần theo dõi v19 nữa.
