# B0 v2 D028 — kết quả ba seed nối dài

dev-full, không phải test
Mỗi seed dùng threshold riêng đã đóng băng từ dev-full; không gộp threshold giữa các seed.
Ba run nối dài là kết quả B0 v2 chính thức theo D028; ba report trần 3 epoch chỉ là kết quả trung gian.
Độ lệch chuẩn mẫu theo seed dùng mẫu số n−1; đây không phải khoảng tin cậy theo speaker.

Train: 15.885 mẫu/57 shard. Dev-full: 38.777 mẫu/186 shard.
Số shard theo phạm vi artifact đã khóa; script chỉ đọc ba report đầu vào.

| Seed | Status | Best epoch / bước | EER (%) | ROC-AUC | Macro-F1 | Threshold riêng |
|---|---|---:|---:|---:|---:|---:|
| 2026 | early_stopped | 3 / 5958 | 8.225567 | 0.973024162 | 0.912347930 | 0.0099602373313330134 |
| 2027 | early_stopped | 3 / 5958 | 8.263624 | 0.973283656 | 0.911968046 | 0.14591552048539161 |
| 2028 | early_stopped | 3 / 5958 | 7.928768 | 0.976297827 | 0.915472045 | 0.00025503549158831988 |

Trung bình ± độ lệch chuẩn mẫu (n−1), qua ba seed:

- eer: 8.139320 ± 0.183333 (%; độ lệch chuẩn tính bằng điểm %).
- roc_auc: 0.974202 ± 0.001820.
- macro_f1: 0.913263 ± 0.001923.

| Seed | Epoch | Bước | EER dev-full (%) |
|---|---:|---:|---:|
| 2026 | 1 | 1986 | 13.914211 |
| 2026 | 2 | 3972 | 10.811957 |
| 2026 | 3 | 5958 | 8.225567 |
| 2026 | 4 | 7944 | 11.382279 |
| 2026 | 5 | 9930 | 8.978316 |
| 2027 | 1 | 1986 | 13.030881 |
| 2027 | 2 | 3972 | 10.671648 |
| 2027 | 3 | 5958 | 8.263624 |
| 2027 | 4 | 7944 | 9.908227 |
| 2027 | 5 | 9930 | 10.274892 |
| 2028 | 1 | 1986 | 13.956794 |
| 2028 | 2 | 3972 | 13.343621 |
| 2028 | 3 | 5958 | 7.928768 |
| 2028 | 4 | 7944 | 8.202713 |
| 2028 | 5 | 9930 | 10.009746 |

Generation best và SHA-256 report đầu vào:

- Seed 2026: `g-951dc37b31054c3ab33a74816a20423c`.
  Report: `reports/b0_v2_ext_seed2026.json`; SHA-256: `57823a906e5060c9622f765ec880ae75316813ed66c8a67418d2d9bb1c1dfb7b`.
- Seed 2027: `g-1f9266acc9704954850fdafaa2be1b61`.
  Report: `reports/b0_v2_ext_seed2027.json`; SHA-256: `959d60d0581148e8acfee5a380602e5245ccf23b1c90df2202730e59f58630a6`.
- Seed 2028: `g-86bc3dc8b0cd4dbf91b4f84e4bc62c4b`.
  Report: `reports/b0_v2_ext_seed2028.json`; SHA-256: `7d5dea6e456ef5ced93eef0eaee7f37371826d6bb49d75c78144773c6f1d4d5e`.
