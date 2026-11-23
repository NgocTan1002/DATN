# B0S-02 — Khởi chạy B0 v2 ×2 (scale2, D036) 3 seed song song trên CPU

- Trạng thái: **ĐANG CHẠY (đã qua kiểm chứng lượt đầu)**
- PID coordinator: `10776` (danh sách PID liên quan: `[6300, 10776]`)
- Thời điểm khởi chạy (UTC+7): `2026-11-20T22:50:21.788564+07:00`
- Preflight: đạt 5/5 (RAM khả dụng `6.9 GiB`, ổ C trống `55.931 GiB`, SHA config `a2e4cf4bbbd9b3fb1a73a6b3e279a023f416d20f32b6d75434bc8e9c288d8018`).

## Kết quả lượt đầu (call 1, 512 bước)
- RAM khả dụng thấp nhất ghi nhận trong lúc theo dõi: `5.809 GiB` (dưới 1,0 GiB: `False`).

| Seed | Exit | Status | Global step | Thời gian call 1 | Giây/bước | Đỉnh Private / RSS | PID call 2 |
|---|---|---|---|---|---|---|---|
| 2026 | 0 | `paused` | 512 (hiện tại 1024) | 631.106 s | 1.2326 s | 1134.0 / 4258.3 MiB | 22660 |
| 2027 | 0 | `paused` | 512 (hiện tại 1024) | 631.127 s | 1.2327 s | 1137.1 / 4311.6 MiB | 12104 |
| 2028 | 0 | `paused` | 512 (hiện tại 1024) | 631.147 s | 1.2327 s | 1162.8 / 4464.0 MiB | 20108 |

## Phép chiếu thời gian (chỉ là ước lượng)
- Phép chiếu ước lượng từ lượt đầu (call 1, 512 bước đã gồm thời gian băm toàn bộ artifact ~12 GB) khi 3 seed chạy song song; các lượt sau không băm lại waveforms nên thực tế có thể nhanh hơn.
- Tốc độ trung bình lượt đầu 3 seed song song: `1.2327 s/bước` (hệ số chậm so với benchmark 1 luồng: `1.147×`).
- Ước lượng mỗi epoch/seed (3.972 bước train + 38.777 mẫu dev-full): **`1.937 giờ/epoch`**.
- Phép chiếu tổng thời gian: 5 epoch ≈ `9.69 giờ`; 7 epoch ≈ `13.56 giờ`; 10 epoch ≈ `19.37 giờ`.

## Lệnh kiểm tra tiến độ cho người dùng
```powershell
Get-Content logs\b0_v2_scale2_execution.jsonl -Tail 10
```
