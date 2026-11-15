# R2 — Phổ dải cao trên dev-full

Dev-full, không phải test. 3.120 mẫu/169 shard; nhóm round-trip dùng lại 1.000 bonafide, không thêm mẫu độc lập.

Trung vị [P25; P75], dB tương đối so với tổng công suất từng mẫu. Đây là số đo mô tả phổ, không kết luận shortcut.

| Nhóm | n | 6–7 kHz | 7–7,5 kHz | 7,5–7,8 kHz | 7,8–7,95 kHz | 7,95–8 kHz |
|---|---:|---:|---:|---:|---:|---:|
| Bonafide | 1000 | -36,16 [-40,33; -30,97] | -41,80 [-46,76; -36,47] | -47,63 [-52,60; -41,77] | -53,43 [-58,17; -47,79] | -59,02 [-64,08; -53,37] |
| Chuyển đổi giọng (VC) | 1000 | -38,10 [-47,04; -33,63] | -45,52 [-70,80; -40,43] | -54,88 [-80,18; -49,14] | -62,53 [-87,78; -56,08] | -69,99 [-91,58; -63,44] |
| Nhiễu đối kháng (AP) | 1000 | -25,64 [-27,20; -23,96] | -28,61 [-30,26; -26,96] | -33,39 [-35,29; -31,68] | -38,53 [-40,27; -36,85] | -43,32 [-45,12; -41,53] |
| Phát lại | 120 | -27,31 [-34,71; -19,50] | -30,77 [-45,32; -23,22] | -33,72 [-50,53; -26,48] | -37,25 [-60,26; -29,41] | -41,23 [-63,20; -34,42] |
| Bonafide round-trip | 1000 | -37,47 [-41,58; -32,25] | -46,17 [-51,05; -40,84] | -53,94 [-58,89; -48,06] | -60,44 [-65,17; -54,79] | -66,12 [-71,18; -60,47] |

Tham số và SHA-256:

```json
{
  "inputs": {
    "manifest_sha256": "a79f1bb2595bc46b08ce30b8652234b1858cc26e558e2ccd33632435df1a5360",
    "subset_files_sha256": "869d8f87c8ffa6c52efc6bc619048dc85dd1eda6d9148fffeefc71ec7d802dfe",
    "probe_report_sha256": "957136cbbd7bdab09176ddf9b3ee9b4218b8cf35d0a5e3afcd9226a09b25d287",
    "metadata.json_sha256": "5537af3384c6ffd83093a5e4175676ddbef1b5a5b5607c5daa87285e6de13f14",
    "index.csv_sha256": "364f1993332ec889dd7301357a1aa6145d14c65c56bfa3db334668b93f01b8d6",
    "selected_raw_waveforms_concatenated_sha256": "f8f94300dd11a6662cec282b86347474281b19547a7292b4777c1d7619d09943"
  },
  "data": {
    "samples": 3120,
    "shards": 169,
    "dev_samples": 38777,
    "dev_shards": 186
  },
  "parameters": {
    "sample_rate": 16000,
    "target_samples": 64000,
    "data_seed": 2026,
    "amplitude_policy": "peak",
    "crop": "center/repeat-then-trim",
    "window": "symmetric Hann, numpy.hanning(64000)",
    "fft": "float64 rfft; one-sided power, double interior bins",
    "band_bounds": "lower inclusive, upper exclusive; final includes 8000 Hz",
    "normalization": "per-sample power / total one-sided windowed power",
    "floor_ratio": 1e-300,
    "grid_centers_hz": [
      4025,
      4075,
      4125,
      4175,
      4225,
      4275,
      4325,
      4375,
      4425,
      4475,
      4525,
      4575,
      4625,
      4675,
      4725,
      4775,
      4825,
      4875,
      4925,
      4975,
      5025,
      5075,
      5125,
      5175,
      5225,
      5275,
      5325,
      5375,
      5425,
      5475,
      5525,
      5575,
      5625,
      5675,
      5725,
      5775,
      5825,
      5875,
      5925,
      5975,
      6025,
      6075,
      6125,
      6175,
      6225,
      6275,
      6325,
      6375,
      6425,
      6475,
      6525,
      6575,
      6625,
      6675,
      6725,
      6775,
      6825,
      6875,
      6925,
      6975,
      7025,
      7075,
      7125,
      7175,
      7225,
      7275,
      7325,
      7375,
      7425,
      7475,
      7525,
      7575,
      7625,
      7675,
      7725,
      7775,
      7825,
      7875,
      7925,
      7975
    ],
    "grid": "sum relative power per 50 Hz bin then 10log10; arithmetic mean over samples in dB",
    "roundtrip": {
      "beta": null,
      "lowpass_filter_width": 6,
      "only_utt_type": "bonafide",
      "operation": "torchaudio.functional.resample",
      "repeak": true,
      "resampling_method": "sinc_interp_hann",
      "rolloff": 0.99,
      "sample_rates": [
        16000,
        40000,
        16000
      ]
    },
    "torch_version": "2.11.0+cpu",
    "numpy_version": "2.5.3"
  },
  "elapsed_seconds": 71.92199999999957
}
```
