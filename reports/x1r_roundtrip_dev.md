# REG-X1R — Kiểm tính hợp lệ round-trip trên dev (D035 mục 6)

Trạng thái: complete (đạt); cập nhật UTC: 2026-11-17T06:32:55.907679+00:00
Dữ liệu: 1000 mẫu bonafide trong tập con T19b (115 shard); artifact dev-full 38777 mẫu/186 shard.
SHA-256 tập con T19b: `869d8f87c8ffa6c52efc6bc619048dc85dd1eda6d9148fffeefc71ec7d802dfe`
SHA-256 1.000 mẫu bonafide: `04639ac5944b9d5e6e304dc2e9d244bb2d50f79b3a59410733d4d395b7e8e758`
Ngưỡng kỳ vọng D035 mục 6: tăng FPR bonafide ≤ 0.5 điểm %.
Tiêu chí tái tạo: D034. Chỉ đánh giá CPU trên dev, không đọc dữ liệu test.

## Kết quả theo seed

| Seed | Epoch best | Threshold | FPR x1r_original % | FPR x1r_roundtrip_40k % | Hiệu (điểm %) | Đổi quyết định | Tái tạo D034 | Kết quả | Thời gian (s) |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 2026 | 8 | 1.3129237507729687e-06 | 0.8000 | 0.8000 | +0.0000 | 0 | đạt (max Δ=0.00122, gap=0.00001) | đạt | 3372.34 |
| 2027 | 6 | 0.0012128221661104007 | 1.0000 | 0.9000 | -0.1000 | 1 | đạt (max Δ=0.00104, gap=0.00000) | đạt | 3336.04 |
| 2028 | 6 | 0.8290327656601152 | 0.9000 | 0.9000 | +0.0000 | 0 | đạt (max Δ=0.00919, gap=0.01227) | đạt | 3407.84 |

## Phân vị logit (P5 / P50 / P95)

| Seed | Điều kiện | P5 | P50 | P95 |
|---|---|---:|---:|---:|
| 2026 | x1r_original | -16.875433 | -16.652748 | -16.441156 |
| 2026 | x1r_roundtrip_40k | -16.873890 | -16.651543 | -16.439870 |
| 2027 | x1r_original | -17.245701 | -17.138379 | -14.223511 |
| 2027 | x1r_roundtrip_40k | -17.245894 | -17.139420 | -14.633005 |
| 2028 | x1r_original | -18.890560 | -18.714677 | -11.361727 |
| 2028 | x1r_roundtrip_40k | -18.890888 | -18.716173 | -11.529480 |
