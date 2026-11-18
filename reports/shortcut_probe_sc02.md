# SC-02 — Kiểm mô hình dựa vào khoảng lặng, nền nhiễu (và DC với B0) trên dev

- **Trạng thái:** complete; cập nhật UTC: 2026-11-17T12:43:49.786404+00:00
- **Dữ liệu:** 2,000 mẫu (1,000 bonafide + 1,000 VC) trên 148 shard từ tập con T19b (`8a753aef4e10cbd197671134250c8f697f51d29ce7364c0da95659c9f8878073`)
- **Nguyên tắc:** biến đổi trên waveform gốc đầy đủ sau peak, trước crop/pad 4 giây; không đọc test, không dùng làm tiêu chí chọn mô hình.

## Kiểm tái tạo trên 200 mẫu đầu (100 bonafide + 100 VC)

| Mô hình | Tiêu chí | Đạt | Chi tiết |
|---|---|---|---|
| B0-v2-seed2026 | B0_abs_prob_diff_1e-4_and_0_flips | đạt | max |Δp|=1.192e-07, flips=0 |
| X1-L8-lr3e-5-seed2026 | D034 | đạt | max |Δlogit|=0.03785, gap=0.04256, flips_out=0 |
| X1-L8-lr3e-5-lp7k-seed2026 | D034 | đạt | max |Δlogit|=0.19421, gap=0.28768, flips_out=0 |

## Kết quả theo mô hình × điều kiện

| Mô hình | Điều kiện | EER % | Miss VC % | FPR % | Tỷ số EER | Δ EER (đ%) | Δ Miss VC (đ%) | Δ FPR (đ%) | Đổi QĐ | Chỉ báo | Thời gian (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| B0-v2-seed2026 | original (lưu) | 12.0000 | 17.8000 | 7.7000 | 1.000 | +0.0000 | +0.0000 | +0.0000 | 0 | — | 0.00 |
| B0-v2-seed2026 | trim_edges | 21.0000 | 42.5000 | 7.6000 | 1.750 | +9.0000 | +24.7000 | -0.1000 | 254 | mạnh | 61.24 |
| B0-v2-seed2026 | noise_floor | 19.9000 | 25.8000 | 13.7000 | 1.658 | +7.9000 | +8.0000 | +6.0000 | 222 | trung bình | 61.20 |
| B0-v2-seed2026 | dc_remove | 12.1000 | 19.5000 | 7.7000 | 1.008 | +0.1000 | +1.7000 | +0.0000 | 25 | yếu | 47.78 |
| X1-L8-lr3e-5-seed2026 | original (lưu) | 1.2000 | 1.1000 | 1.2000 | 1.000 | +0.0000 | +0.0000 | +0.0000 | 0 | — | 0.00 |
| X1-L8-lr3e-5-seed2026 | trim_edges | 1.4000 | 2.3000 | 1.2000 | 1.167 | +0.2000 | +1.2000 | +0.0000 | 12 | yếu | 3391.24 |
| X1-L8-lr3e-5-seed2026 | noise_floor | 2.5000 | 1.5000 | 5.6000 | 2.083 | +1.3000 | +0.4000 | +4.4000 | 50 | trung bình, độ bền kém | 3121.24 |
| X1-L8-lr3e-5-lp7k-seed2026 | original (lưu) | 1.4000 | 1.5000 | 0.8000 | 1.000 | +0.0000 | +0.0000 | +0.0000 | 0 | — | 0.00 |
| X1-L8-lr3e-5-lp7k-seed2026 | trim_edges | 1.6000 | 1.8000 | 0.7000 | 1.143 | +0.2000 | +0.3000 | -0.1000 | 4 | yếu | 3008.22 |
| X1-L8-lr3e-5-lp7k-seed2026 | noise_floor | 2.1000 | 0.8000 | 3.6000 | 1.500 | +0.7000 | -0.7000 | +2.8000 | 41 | trung bình | 3177.18 |

## Mức tổng theo mô hình và chi tiết trim_edges

| Mô hình | Mức tổng | Số mẫu bị cắt (BF / VC) | Giữ nguyên (fallback) | Trung vị bị cắt BF / VC (s, toàn lớp) |
|---|---|---|---:|---|
| B0-v2-seed2026 | **mạnh** | 144 / 922 | 0 | 0.000 / 1.470 |
| X1-L8-lr3e-5-seed2026 | **trung bình** | 144 / 922 | 0 | 0.000 / 1.470 |
| X1-L8-lr3e-5-lp7k-seed2026 | **trung bình** | 144 / 922 | 0 | 0.000 / 1.470 |

## Giới hạn

- Chỉ kiểm tra seed huấn luyện 2026 cho mỗi họ mô hình.
- Chỉ đánh giá 2.000 mẫu (1.000 bonafide + 1.000 VC) thuộc tập con T19b trên closed_dev; không đọc test.
- Cắt khoảng lặng (trim_edges) và cộng nhiễu (noise_floor) cũng làm thay đổi nội dung hợp lệ hoặc biên phân đoạn 4 giây, nên EER tăng không thuần túy do lối tắt.
- noise_floor đồng thời là phép thử độ bền trước nhiễu cộng.
- Kết quả chỉ áp dụng cho VSASV-HF-public-snapshot-v1 và tập 15.885 mẫu closed_train.
