# Dự toán B0 full training

Đây là phép chiếu từ số đo đã lưu, chưa chạy huấn luyện đầy đủ.

- Đặc tả: `b0-peak-development-20k-v2-full-v1`; SHA-256 `a40f5cd28e4a262c213ec6576748cf61351a6e3a2ea0514d9aad4f799d70fe2c`.
- Manifest 20,000 mẫu/66 shard, pool cục bộ 68 shard.
- 1,986 bước/epoch; tail batch 5 mẫu.
- Tối đa 5,958 bước, 47,655 lượt mẫu train và 3 full-development evaluation.
- Dự toán mỗi epoch: 8.764 giờ.
- Trần 3 epoch: 26.291 giờ; cộng 20% dự phòng: 31.550 giờ.
- Một phiên tối đa 512 bước và có thể kèm development: dự toán có dự phòng 4.313 giờ.
- Tối đa 12 phiên theo cách chia này nếu không gián đoạn ngoài dự kiến hoặc dừng sớm.

Chọn thời gian lớn nhất trên benchmark và ba pilot; training/development maximum có thể đến từ các phép đo khác nhau. Startup/checkpoint/resume chưa đo riêng, được dự trù bằng biên dự phòng; số phiên thực có thể tăng khi gián đoạn. Không coi quota lịch sử là quota hiện tại; kiểm tra lại trước mỗi phiên. Không đổi ngân sách giữa run để vừa quota, không tự dùng tài nguyên trả phí.

Nguồn, checksum và số đo chi tiết trong JSON đi kèm. Tái sinh bằng:

```powershell
.\.venv\Scripts\python.exe scripts/estimate_b0_full_training_budget.py
```
