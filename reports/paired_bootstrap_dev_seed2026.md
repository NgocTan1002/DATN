# T23 — Bootstrap ghép cặp trên dev-full, không phải test

Quyết định: D032.
Trạng thái: complete

| So sánh a − b | Hiệu EER (điểm %) | CI95% hiệu | CI95% EER(a) % | CI95% EER(b) % |
|---|---:|---|---|---|
| B0-v2-2026_minus_X1-2026 | 7.409357 | [4.234108; 11.493715] | [4.960647; 12.725627] | [0.158436; 2.149993] |
| X0-2026_minus_X1-2026 | 3.382604 | [1.228110; 6.153761] | [1.751709; 7.996280] | [0.158436; 2.149993] |

38777mẫu/186shard; bootstrap ghép cặp theo speaker,1000lượt/seed2026/CI95%.
Không chấm lại để làm bootstrap; dùng score best dev đã lưu. CI không phải SD qua seed; không chọn cấu hình/seed.
