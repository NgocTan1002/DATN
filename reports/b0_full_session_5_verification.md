# Kiểm chứng B0 full training phiên 5 — Kaggle v18

- Kết quả **ĐẠT**: workflow hoàn tất, cả 3 lệnh trả mã 0; log xác nhận khôi phục bước 2.048 và dừng an toàn `paused` tại bước 2.560.
- Resume GPU 2.048 → 2.560, thêm 512 bước. Sampler ở epoch 2, cursor 574; tổng trace là 2.560 batch / 20.477 lượt mẫu train.
- Phiên này chưa đến cuối epoch 2 nên không chạy development mới. History vẫn có đúng một evaluation epoch 1 tại bước 1.986; metric được tính lại và giữ nguyên: loss `0.677994`, EER `8.595801%`, ROC-AUC `0.970242`, macro-F1 `0.908857`, threshold EER `0.009600887260`.
- Best vẫn là epoch 1/bước 1.986, generation `g-c7418c4439ac45e9b051fa77b768709d`. Latest mới là bước 2.560, generation `g-2d1394dcd5d34ed5bd7fe2bdc26af49f`.
- Phạm vi 15.885 train / 4.115 development trên manifest 20.000 mẫu / 66 shard; test set không được dùng. Source 56 tệp, identity và D023 không đổi.
- Đọc lại đủ 48 generation; 39 generation nhập từ v17 giữ nguyên từng byte. Pointer/checkpoint/state/controller khớp; model và 12 Adam state hữu hạn, Adam step 2.560, đủ 2 CUDA RNG state.
- Kiểm tra EER bất thường vẫn đạt: speaker/file overlap 0, 20.000/20.000 content hash duy nhất, duplicate chéo split 0; metadata sample-rate gốc 16/40 kHz khớp và pipeline vẫn chuẩn hóa về 16 kHz, 4 giây, policy `peak`. Đây chưa phải kết quả test hay kết luận B0 cuối.
- Output có 103/103 tệp sạch, không phát hiện HTML/502 và không cần tải lại. Thư mục chuẩn: `kaggle_outputs/b0-full-session-5-v18`.
- Runtime training report `1.802` giờ trên `Tesla T4`; workflow `2026-10-28T06:54:49+00:00` → `2026-10-28T08:43:25+00:00`.

## Bước tiếp theo

Latest bước 2.560 đã đủ điều kiện làm input cho phiên 6 sau khi đóng gói/preflight và có xác nhận riêng cho payload mới. Nếu tiếp tục tối đa 512 bước, mốc kế tiếp dự kiến là 3.072/epoch 2/cursor 1.086; full development epoch 2 chỉ diễn ra ở bước 3.972.

Giữ checkpoint/log lớn ngoài Git. Report gốc, bootstrap, workflow, runtime và báo cáo kiểm chứng đã lưu trong `reports/`. Không cần theo dõi v18 nữa.
