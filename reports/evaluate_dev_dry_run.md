# T23 — Kiểm entrypoint trên dev-full, không phải test

Quyết định: D032.
Trạng thái: complete
Cổng test đóng; không có registry chính thức.

| Mô hình | Điều kiện | Mẫu / shard | Tái tạo | Max Δp hoặc Δlogit | Số đảo | EER % |
|---|---|---:|---|---:|---:|---:|
| B0-v2-seed2026 | original | 38777 / 186 | True | 0.0 | 0 | 8.22556709773161 |
| B0-v2-seed2027 | original | 38777 / 186 | True | 0.0 | 0 | 8.263623812230977 |
| B0-v2-seed2028 | original | 38777 / 186 | True | 0.0 | 0 | 7.92876828492686 |
| X0-seed2026 | original | 64 / 47 | True | 8.579447013734587e-05 | 0 | không tính (một lớp) |
| X1-L8-lr3e-5-seed2026 | original | 64 / 47 | True | 5.5230305939346636e-05 | 0 | không tính (một lớp) |
| B0-v2-seed2026 | lowpass_7k | 32 / 1 | kỹ thuật | — | — | không tính (một lớp) |

XLS-R64: 64 mẫu đầu tập con T19b đã khóa; chưa đủ hai lớp nên không dùng làm metric khoa học.
Lowpass7k32mẫu chỉ kiểm đường biến đổi. Threshold giữ nguyên từ registry best dev.
Raw score/raw_logit và thời gian chi tiết ở .cache/t23; SHA đầu vào/checkpoint trong registry nháp.
Report T22 được đọc chỉ trong nhánh kiểm điều kiện test; T23 không sửa hoặc chạy lại T22.
