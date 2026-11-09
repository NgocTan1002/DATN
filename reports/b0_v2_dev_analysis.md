# Phân tích B0 v2 — dev-full, không phải test

EER trong JSON là tỉ lệ; bảng đổi sang %. SD qua seed dùng mẫu số n−1, khác CI theo người nói.

## Phạm vi và đầu vào

{"training_samples": 15885, "training_shards": 57, "training_shards_source": "Khai báo D027/D028; không đọc lại train.", "development_samples": 38777, "development_shards": 186, "development_speakers": 172}

| Đầu vào | SHA-256 |
|---|---|
| closed_dev_full_v1.csv | a79f1bb2595bc46b08ce30b8652234b1858cc26e558e2ccd33632435df1a5360 |
| b0_v2_ext_seed2026.json | 57823a906e5060c9622f765ec880ae75316813ed66c8a67418d2d9bb1c1dfb7b |
| b0_v2_ext_seed2027.json | 959d60d0581148e8acfee5a380602e5245ccf23b1c90df2202730e59f58630a6 |
| b0_v2_ext_seed2028.json | 7d5dea6e456ef5ced93eef0eaee7f37371826d6bb49d75c78144773c6f1d4d5e |

## Kết quả tổng và CI theo người nói

| Seed | EER (%) | ROC-AUC | Macro-F1 | Threshold | CI 95% EER (%) | CI 95% AUC | Hợp lệ |
|---|---:|---:|---:|---:|---|---|---:|
| 2026 | 8,2256 | 0,9730 | 0,9123 | 0,009960237 | [4,9606; 12,7256] | [0,9483; 0,9887] | 1000/1000 |
| 2027 | 8,2636 | 0,9733 | 0,9120 | 0,145915520 | [5,0465; 12,4747] | [0,9506; 0,9888] | 1000/1000 |
| 2028 | 7,9288 | 0,9763 | 0,9155 | 0,000255035 | [5,3191; 11,4235] | [0,9551; 0,9887] | 1000/1000 |

Bootstrap: 1000 lần, seed 2026, confidence 0,95.
CI có điều kiện trên checkpoint đã chọn bằng chính dev-full; không phải CI test hoặc CI qua seed.

### Trung bình ± độ lệch chuẩn mẫu

- eer: 8,1393 ± 0,1833.
- roc_auc: 0,9742 ± 0,0018.
- macro_f1: 0,9133 ± 0,0019.

## EER theo kiểu tấn công

EER mỗi nhóm dùng bonafide chung và threshold EER riêng của nhóm; confusion dùng threshold tổng của chính seed.

| Nhóm | Bonafide | Spoof | 2026 (%) | 2027 (%) | 2028 (%) | Trung bình ± SD (điểm %) |
|---|---:|---:|---:|---:|---:|---|
| Chuyển đổi giọng nói | 14151 | 12253 | 11,5074 | 11,1653 | 11,4747 | 11,3825 ± 0,1888 |
| Nhiễu đối kháng | 14151 | 12253 | 2,6783 | 2,5157 | 1,6894 | 2,2945 ± 0,5303 |
| Phát lại | 14151 | 120 | 5,5120 | 11,2360 | 10,8333 | 9,1938 ± 3,1949 |

**Replay: không đủ để kết luận** (120 mẫu spoof).

## Ma trận nhầm lẫn và lỗi theo kiểu phát ngôn

Hàng là nhãn thật; cột là nhãn dự đoán. Score ≥ threshold tổng của seed là spoof.

| Seed | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| 2026 | 12987 | 1164 | 2026 | 22600 |
| 2027 | 12982 | 1169 | 2035 | 22591 |
| 2028 | 13029 | 1122 | 1953 | 22673 |

| Seed | Loại | Số mẫu | Số lỗi | Tỉ lệ lỗi (%) |
|---|---|---:|---:|---:|
| 2026 | Giọng thật | 14151 | 1164 | 8,2256 |
| 2026 | Chuyển đổi giọng nói | 12253 | 2014 | 16,4368 |
| 2026 | Nhiễu đối kháng | 12253 | 9 | 0,0735 |
| 2026 | Phát lại | 120 | 3 | 2,5000 |
| 2027 | Giọng thật | 14151 | 1169 | 8,2609 |
| 2027 | Chuyển đổi giọng nói | 12253 | 2000 | 16,3225 |
| 2027 | Nhiễu đối kháng | 12253 | 17 | 0,1387 |
| 2027 | Phát lại | 120 | 18 | 15,0000 |
| 2028 | Giọng thật | 14151 | 1122 | 7,9288 |
| 2028 | Chuyển đổi giọng nói | 12253 | 1932 | 15,7676 |
| 2028 | Nhiễu đối kháng | 12253 | 4 | 0,0326 |
| 2028 | Phát lại | 120 | 17 | 14,1667 |

## Phân vị score

Phân vị tuyến tính; không trung bình hóa threshold.

| Seed | Nhóm | P1 | P5 | P25 | P50 | P75 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026 | bonafide | 3.5074439e-15 | 1.73539866e-12 | 2.05056532e-08 | 2.52073414e-06 | 0.000133661975 | 0.053693248 | 0.924253851 |
| 2026 | spoof | 2.3547675e-05 | 0.00163941956 | 0.500077397 | 0.990355015 | 0.999806374 | 0.999995828 | 0.999999762 |
| 2027 | bonafide | 5.65254699e-17 | 4.16520361e-13 | 7.12721748e-09 | 1.54716383e-06 | 0.000557123683 | 0.451098174 | 0.965590805 |
| 2027 | spoof | 1.78885016e-05 | 0.0276599815 | 0.864621982 | 0.999178886 | 0.999958515 | 0.999999881 | 1 |
| 2028 | bonafide | 5.83664144e-29 | 2.10835683e-23 | 1.23808508e-15 | 2.00789784e-11 | 8.32746423e-08 | 0.00186392042 | 0.678148597 |
| 2028 | spoof | 3.96813429e-10 | 1.3626815e-05 | 0.596098304 | 0.999924362 | 0.999999166 | 1 | 1 |

## History dev-full

Loss lấy nguyên report/history; không tính lại BCE từ sigmoid score đã bão hòa.

| Seed | Epoch | Bước | EER (%) | Loss |
|---|---:|---:|---:|---:|
| 2026 | 1 | 1986 | 13,9142 | 0,477128 |
| 2026 | 2 | 3972 | 10,8120 | 0,359319 |
| 2026 | 3 | 5958 | 8,2256 | 0,689378 |
| 2026 | 4 | 7944 | 11,3823 | 1,942091 |
| 2026 | 5 | 9930 | 8,9783 | 1,369207 |
| 2027 | 1 | 1986 | 13,0309 | 0,341154 |
| 2027 | 2 | 3972 | 10,6716 | 0,701234 |
| 2027 | 3 | 5958 | 8,2636 | 0,436581 |
| 2027 | 4 | 7944 | 9,9082 | 1,263852 |
| 2027 | 5 | 9930 | 10,2749 | 1,209973 |
| 2028 | 1 | 1986 | 13,9568 | 0,422131 |
| 2028 | 2 | 3972 | 13,3436 | 0,621054 |
| 2028 | 3 | 5958 | 7,9288 | 1,167142 |
| 2028 | 4 | 7944 | 8,2027 | 0,371692 |
| 2028 | 5 | 9930 | 10,0097 | 1,419825 |

## Kiểm kỹ thuật cục bộ (tensor tổng hợp)

Lệnh: .venv/Scripts/python.exe scripts/analyze_b0_v2_dev.py (runtime_probe).

```json
{
  "torch_version": "2.11.0+cpu",
  "torchaudio_version": "2.11.0+cpu",
  "resample_defaults": {
    "lowpass_filter_width": 6,
    "rolloff": 0.99,
    "resampling_method": "sinc_interp_hann",
    "beta": null
  },
  "lfcc_recipe": {
    "sample_rate": 16000,
    "n_filter": 80,
    "n_lfcc": 60,
    "f_min": 0.0,
    "f_max": 8000.0,
    "dct_type": 2,
    "norm": "ortho",
    "log_lf": false,
    "spectrum": {
      "n_fft": 512,
      "win_length": 400,
      "hop_length": 160,
      "power": 2.0,
      "center": true,
      "pad_mode": "reflect"
    }
  },
  "waveform_shape": [
    1,
    64000
  ],
  "lfcc_shape": [
    1,
    60,
    401
  ],
  "logit_shape": [
    1
  ],
  "module_shapes": {
    "feature_extractor.0": [
      1,
      64,
      60,
      401
    ],
    "feature_extractor.1": [
      1,
      32,
      60,
      401
    ],
    "feature_extractor.2": [
      1,
      32,
      30,
      200
    ],
    "feature_extractor.3": [
      1,
      128,
      30,
      200
    ],
    "feature_extractor.4": [
      1,
      64,
      30,
      200
    ],
    "feature_extractor.5": [
      1,
      64,
      15,
      100
    ],
    "feature_extractor.6": [
      1,
      256,
      15,
      100
    ],
    "feature_extractor.7": [
      1,
      128,
      15,
      100
    ],
    "feature_extractor.8": [
      1,
      128,
      7,
      50
    ],
    "feature_extractor.9": [
      1,
      256,
      7,
      50
    ],
    "feature_extractor.10": [
      1,
      128,
      7,
      50
    ],
    "feature_extractor": [
      1,
      128,
      7,
      50
    ],
    "spatial_pool": [
      1,
      128,
      1,
      1
    ],
    "classifier": [
      1,
      1
    ]
  },
  "trainable_parameters": 498113,
  "finite_forward": true,
  "evidence": "Forward tổng hợp CPU; không audio thật/huấn luyện; tham số LFCC khai báo tường minh, không đọc thêm cấu hình."
}
```
