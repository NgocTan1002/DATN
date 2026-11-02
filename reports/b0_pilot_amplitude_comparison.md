# Đối chiếu ba pilot B0 amplitude

- Phạm vi: `VSASV-HF-public-snapshot-v1 / development-20k-v2`; manifest 20.000 mẫu/66 shard, 15.885 train và 4.115 development; pool cục bộ 68 shard.
- Cả ba run: seed 2026, batch 8, worker 0, Adam, LR 0,001, 896 step (7.168 lượt mẫu train), không early stopping/test.
- Source SHA-256: `12d91121abac99367724d63f336951a36cd2a47075e79ed4c6d670b3c3d060db`.
- Manifest SHA-256: `f549423b1fb33665f7e5606cdf56321bc8cd07246a0cdf0b90cd918c0c09e728`.
- Chính sách được chọn theo D019: **peak**, EER thấp nhất; không cần tie-break. Không sửa `configs/audio.json`.

| Policy / kernel | Loss | EER (%) | ROC-AUC | Macro-F1 (%) | Threshold dev EER | Giờ |
|---|---:|---:|---:|---:|---:|---:|
| none / v9 | 0.40108390 | 10.0394 | 0.97149607 | 89.3614 | 0.17854070 | 4.807 |
| peak / v10 | 0.32683539 | 9.1207 | 0.97006598 | 90.3263 | 0.25846705 | 4.946 |
| rms_dbfs / v12 | 0.37693000 | 11.5486 | 0.95694110 | 87.7914 | 0.19868370 | 4.836 |

## Kiểm chứng

- Workflow, source/config/selection và môi trường khớp; identity chỉ khác amplitude policy, lệnh chỉ khác policy/output. Development metadata/thứ tự khớp manifest, metric tính lại sai số dưới 1e-12.
- Băm checkpoint cả ba run khớp report; checkpoint v3 có step/cursor 896, sampler, optimizer và Python/NumPy/PyTorch/2 CUDA RNG state. Không chạy lại GPU resume trong phiên kiểm chứng.
- Manifest speaker/file-disjoint, checksum khớp content audit đã có: 20.000 hash duy nhất, 0 duplicate. Không băm lại waveform hoặc đọc test.

## Ma trận nhầm lẫn tại threshold riêng của mỗi pilot

| Policy | TN | FP | FN | TP | FPR (%) | FNR (%) |
|---|---:|---:|---:|---:|---:|---:|
| none | 1371 | 153 | 261 | 2330 | 10.0394 | 10.0733 |
| peak | 1385 | 139 | 237 | 2354 | 9.1207 | 9.1470 |
| rms_dbfs | 1348 | 176 | 300 | 2291 | 11.5486 | 11.5785 |

## Số dự đoán đúng theo utt_type

| Policy | Bonafide | Voice conversion | Adversarial attack | Replay |
|---|---:|---:|---:|---:|
| none | 1371/1524 | 1030/1289 | 1289/1289 | 11/13 |
| peak | 1385/1524 | 1059/1289 | 1284/1289 | 11/13 |
| rms_dbfs | 1348/1524 | 991/1289 | 1289/1289 | 11/13 |

## Tài nguyên

| Policy | Peak RSS (MiB) | Peak VRAM (MiB) | Loader / (loader + model) (%) |
|---|---:|---:|---:|
| none | 2146.633 | 288.895 | 99.91 |
| peak | 2136.664 | 288.895 | 99.91 |
| rms_dbfs | 2187.020 | 288.895 | 99.91 |

## Giới hạn và diễn giải

- Đây là lựa chọn vận hành theo D019 trên một seed, chưa đủ một epoch; không tuyên bố ý nghĩa thống kê hoặc hiệu năng cuối của baseline.
- Peak có EER/loss tốt nhất nhưng AUC của none cao hơn; không tuyên bố cải thiện mọi metric. Replay chỉ có 13 mẫu development. Rủi ro shortcut từ sample rate/utt_type chưa được loại bỏ hoàn toàn.
- Threshold riêng được chọn trên development. Threshold pilot peak chỉ gắn với checkpoint pilot đó; mô hình full training mới phải chọn lại threshold trên closed_dev, rồi đóng băng trước test/inference.
- Kết quả chỉ áp dụng snapshot công khai hiện dùng, không so trực tiếp với bài báo gốc hoặc tuyên bố official reproduction/unseen TTS.
- Version 11 lỗi mount source trước runner; version 12 chạy lại cùng source hoàn thành, không resume. Không đổi metadata/split/seed/audio config hoặc dùng test.

Bằng chứng chi tiết: `b0_pilot_amplitude_comparison.json`, ba report `b0_pilot_{none,peak,rms_dbfs}.json` và workflow/runtime tương ứng.
