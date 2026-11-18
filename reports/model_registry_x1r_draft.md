# Registry phụ mô hình X1R — snapshot chưa đầy đủ

Quyết định: D035; complete: True; input_condition: lowpass_7k
SHA-256 hệ số bộ lọc 7 kHz: `973b82a36db8dab4f802de2102c5feed2bfb8cf6bb5846472c07d9984ab4c977`
Snapshot UTC: 2026-11-17T06:32:35.190531+00:00
Không phải registry chính thức; không mở cổng test.

| Mục | Trạng thái | Claude kiểm | Generation | Epoch best | Threshold dev | EER dev % | Ghi chú kiểm |
|---|---|---|---|---:|---:|---:|---|
| X1-L8-lr3e-5-lp7k-seed2026 | present | True | g-98c5c2e9fc9e4f3cad966e8c6147add8 | 8 | 1.3129237507729687e-06 | 0.856818 | T30B 9 epoch + T32B resume |
| X1-L8-lr3e-5-lp7k-seed2027 | present | True | g-a543c4754fcb45b487282d145e318f0a | 6 | 0.0012128221661104007 | 0.917729 | T30B |
| X1-L8-lr3e-5-lp7k-seed2028 | present | True | g-cf5a69e95e384ee6b3352668aa948e70 | 6 | 0.8290327656601152 | 0.921790 | T31B |
