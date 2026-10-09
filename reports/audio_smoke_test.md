# Báo cáo audio smoke test VSASV

- **Trạng thái kỹ thuật:** ĐẠT
- **Shard cục bộ:** 68/432
- **Utterance:** 34,782
- **Speaker trong 68 shard:** 228
- **Sample rate đích bắt buộc:** 16,000 Hz

## Kiểm tra toàn bộ 68 shard

- Đường dẫn duy nhất: 34,782/34,782
- Thiếu file/label/utt_type: 0
- Waveform thiếu hoặc rỗng: 0
- Sample rate không hợp lệ: 0

## Phân bố sample rate

| Loại | Sample rate | Số mẫu |
|---|---:|---:|
| `adversarial_attack` | 16,000 Hz | 7,120 |
| `bonafide` | 16,000 Hz | 16,324 |
| `replay` | 16,000 Hz | 265 |
| `voice_conversion` | 40,000 Hz | 11,073 |

## Thời lượng

| Loại | Min (s) | Median (s) | Mean (s) | P95 (s) | Max (s) | Tổng giờ |
|---|---:|---:|---:|---:|---:|---:|
| `adversarial_attack` | 1.18 | 4.12 | 4.91 | 8.42 | 39.78 | 9.71 |
| `bonafide` | 0.84 | 4.51 | 7.01 | 21.24 | 39.96 | 31.78 |
| `replay` | 1.41 | 3.07 | 3.22 | 4.80 | 6.46 | 0.24 |
| `voice_conversion` | 1.04 | 3.96 | 4.44 | 7.62 | 39.78 | 13.65 |

## Kiểm tra waveform mẫu

Đã đọc và kiểm tra số học 267 waveform được chọn xác định theo từng cặp shard/loại bằng hash(file).

| File | Loại | Hz | Giây | Peak | RMS | Non-finite | Silent |
|---|---|---:|---:|---:|---:|---:|---|
| `id00003/00044.wav` | `bonafide` | 16,000 | 3.24 | 0.4152 | 0.0640 | 0 | Không |
| `id00012/00029.wav` | `bonafide` | 16,000 | 1.24 | 0.5765 | 0.1001 | 0 | Không |
| `id00003/00036.wav` | `bonafide` | 16,000 | 9.24 | 0.4185 | 0.0593 | 0 | Không |
| `id00020/00011.wav` | `bonafide` | 16,000 | 3.24 | 0.5601 | 0.0858 | 0 | Không |
| `id00018/00015.wav` | `bonafide` | 16,000 | 3.24 | 0.5299 | 0.0502 | 0 | Không |
| `id00014/00018.wav` | `bonafide` | 16,000 | 2.84 | 1.0000 | 0.3391 | 0 | Không |
| `id00026/00073.wav` | `bonafide` | 16,000 | 3.24 | 1.0000 | 0.2152 | 0 | Không |
| `id00026/00135.wav` | `bonafide` | 16,000 | 1.64 | 0.7535 | 0.1217 | 0 | Không |
| `id00024/00049.wav` | `bonafide` | 16,000 | 2.44 | 0.5837 | 0.0962 | 0 | Không |
| `id00057/00003.wav` | `bonafide` | 16,000 | 2.04 | 0.7223 | 0.2378 | 0 | Không |
| `id00055/00140.wav` | `bonafide` | 16,000 | 4.44 | 0.3966 | 0.0384 | 0 | Không |
| `id00060/00045.wav` | `bonafide` | 16,000 | 6.44 | 0.6159 | 0.0900 | 0 | Không |
| `id00096/00000.wav` | `bonafide` | 16,000 | 3.24 | 0.3413 | 0.0461 | 0 | Không |
| `id00098/00026.wav` | `bonafide` | 16,000 | 4.04 | 0.6365 | 0.0927 | 0 | Không |
| `id00101/00097.wav` | `bonafide` | 16,000 | 6.44 | 1.0000 | 0.1875 | 0 | Không |
| `id00155/00042.wav` | `bonafide` | 16,000 | 0.84 | 0.3196 | 0.0449 | 0 | Không |
| `id00156/00043.wav` | `bonafide` | 16,000 | 4.44 | 0.6547 | 0.0796 | 0 | Không |
| `id00156/00002.wav` | `bonafide` | 16,000 | 4.04 | 0.5737 | 0.0914 | 0 | Không |
| `id00205/00025.wav` | `bonafide` | 16,000 | 4.44 | 0.8304 | 0.0940 | 0 | Không |
| `id00208/00058.wav` | `bonafide` | 16,000 | 8.84 | 0.8681 | 0.0892 | 0 | Không |
| `id00208/00042.wav` | `bonafide` | 16,000 | 12.44 | 0.7833 | 0.0794 | 0 | Không |
| `id00222/00009.wav` | `bonafide` | 16,000 | 8.04 | 0.5792 | 0.0520 | 0 | Không |
| `id00222/00016.wav` | `bonafide` | 16,000 | 3.64 | 0.2584 | 0.0354 | 0 | Không |
| `id00225/00020.wav` | `bonafide` | 16,000 | 21.64 | 0.3973 | 0.0467 | 0 | Không |
| `id00248/00229.wav` | `bonafide` | 16,000 | 2.84 | 0.9784 | 0.2713 | 0 | Không |
| `id00248/00092.wav` | `bonafide` | 16,000 | 4.04 | 0.7528 | 0.1531 | 0 | Không |
| `id00247/00156.wav` | `bonafide` | 16,000 | 3.44 | 0.2405 | 0.0287 | 0 | Không |
| `id00300/00021.wav` | `bonafide` | 16,000 | 6.04 | 0.8913 | 0.1161 | 0 | Không |
| `id00302/00043.wav` | `bonafide` | 16,000 | 2.44 | 0.3653 | 0.0360 | 0 | Không |
| `id00303/00016.wav` | `bonafide` | 16,000 | 6.44 | 0.6240 | 0.0669 | 0 | Không |
| `id00347/00069.wav` | `bonafide` | 16,000 | 10.44 | 1.0000 | 0.2279 | 0 | Không |
| `id00348/00056.wav` | `bonafide` | 16,000 | 1.24 | 0.4152 | 0.0564 | 0 | Không |
| `id00348/00046.wav` | `bonafide` | 16,000 | 4.04 | 0.3020 | 0.0445 | 0 | Không |
| `id00349/00010.wav` | `bonafide` | 16,000 | 6.44 | 0.9077 | 0.1226 | 0 | Không |
| `id00348/01244.wav` | `bonafide` | 16,000 | 10.75 | 0.6008 | 0.0795 | 0 | Không |
| `id00348/01262.wav` | `bonafide` | 16,000 | 19.89 | 0.8406 | 0.0959 | 0 | Không |
| `id00380/00407.wav` | `bonafide` | 16,000 | 4.84 | 0.4676 | 0.0680 | 0 | Không |
| `id00380/00102.wav` | `bonafide` | 16,000 | 4.44 | 0.7772 | 0.0929 | 0 | Không |
| `id00380/00365.wav` | `bonafide` | 16,000 | 2.04 | 0.4591 | 0.0760 | 0 | Không |
| `id00439/00023.wav` | `bonafide` | 16,000 | 5.24 | 0.9617 | 0.1228 | 0 | Không |
| `id00436/00150.wav` | `bonafide` | 16,000 | 14.44 | 1.0000 | 0.1606 | 0 | Không |
| `id00439/00005.wav` | `bonafide` | 16,000 | 4.84 | 0.7303 | 0.0764 | 0 | Không |
| `id00450/00025.wav` | `bonafide` | 16,000 | 8.44 | 0.9073 | 0.1391 | 0 | Không |
| `id00449/00039.wav` | `bonafide` | 16,000 | 10.84 | 0.8287 | 0.0549 | 0 | Không |
| `id00450/00044.wav` | `bonafide` | 16,000 | 9.24 | 0.8426 | 0.1766 | 0 | Không |
| `id00489/00026.wav` | `bonafide` | 16,000 | 2.84 | 0.4556 | 0.1398 | 0 | Không |
| `id00488/00231.wav` | `bonafide` | 16,000 | 13.64 | 0.6796 | 0.0572 | 0 | Không |
| `id00489/00185.wav` | `bonafide` | 16,000 | 5.64 | 0.4656 | 0.0787 | 0 | Không |
| `id00526/00220.wav` | `bonafide` | 16,000 | 10.44 | 0.4687 | 0.1335 | 0 | Không |
| `id00531/00064.wav` | `bonafide` | 16,000 | 6.84 | 0.5829 | 0.0519 | 0 | Không |
| `id00526/00195.wav` | `bonafide` | 16,000 | 19.64 | 0.4743 | 0.1130 | 0 | Không |
| `id00536/00054.wav` | `bonafide` | 16,000 | 2.04 | 0.3545 | 0.0484 | 0 | Không |
| `id00536/00038.wav` | `bonafide` | 16,000 | 10.04 | 0.4802 | 0.0526 | 0 | Không |
| `id00537/00098.wav` | `bonafide` | 16,000 | 7.24 | 0.9741 | 0.2260 | 0 | Không |
| `id00562/00170.wav` | `bonafide` | 16,000 | 2.84 | 0.7479 | 0.0795 | 0 | Không |
| `id00565/00013.wav` | `bonafide` | 16,000 | 10.84 | 0.6504 | 0.0895 | 0 | Không |
| `id00562/00293.wav` | `bonafide` | 16,000 | 6.84 | 1.0000 | 0.1695 | 0 | Không |
| `id00600/00103.wav` | `bonafide` | 16,000 | 6.84 | 0.9726 | 0.1853 | 0 | Không |
| `id00600/00001.wav` | `bonafide` | 16,000 | 2.84 | 0.6274 | 0.0852 | 0 | Không |
| `id00600/00114.wav` | `bonafide` | 16,000 | 4.84 | 0.8344 | 0.1251 | 0 | Không |
| `id00634/00017.wav` | `bonafide` | 16,000 | 9.24 | 0.7926 | 0.0605 | 0 | Không |
| `id00636/00010.wav` | `bonafide` | 16,000 | 21.64 | 0.6424 | 0.0818 | 0 | Không |
| `id00636/00049.wav` | `bonafide` | 16,000 | 11.24 | 0.5887 | 0.0721 | 0 | Không |
| `id00686/00260.wav` | `bonafide` | 16,000 | 11.24 | 0.9969 | 0.0993 | 0 | Không |
| `id00683/00066.wav` | `bonafide` | 16,000 | 4.84 | 1.0000 | 0.1637 | 0 | Không |
| `id00686/00163.wav` | `bonafide` | 16,000 | 5.24 | 0.6479 | 0.0719 | 0 | Không |
| `id00689/00062.wav` | `bonafide` | 16,000 | 5.24 | 0.4546 | 0.0580 | 0 | Không |
| `id00690/00128.wav` | `bonafide` | 16,000 | 3.24 | 0.9617 | 0.1907 | 0 | Không |
| `id00688/00005.wav` | `bonafide` | 16,000 | 18.04 | 0.5361 | 0.0788 | 0 | Không |
| `id00738/00139.wav` | `bonafide` | 16,000 | 8.44 | 0.6943 | 0.0898 | 0 | Không |
| `id00737/00101.wav` | `bonafide` | 16,000 | 32.44 | 0.4938 | 0.0716 | 0 | Không |
| `id00738/00104.wav` | `bonafide` | 16,000 | 4.84 | 0.8464 | 0.1028 | 0 | Không |
| `id00744/00000.wav` | `bonafide` | 16,000 | 4.84 | 0.1099 | 0.0158 | 0 | Không |
| `id00742/00048.wav` | `bonafide` | 16,000 | 8.04 | 1.0000 | 0.1132 | 0 | Không |
| `id00742/00044.wav` | `bonafide` | 16,000 | 32.84 | 0.9982 | 0.0952 | 0 | Không |
| `id00793/00043.wav` | `bonafide` | 16,000 | 10.44 | 1.0000 | 0.1358 | 0 | Không |
| `id00797/00002.wav` | `bonafide` | 16,000 | 3.64 | 0.7414 | 0.1375 | 0 | Không |
| `id00793/00017.wav` | `bonafide` | 16,000 | 2.04 | 1.0000 | 0.1464 | 0 | Không |
| `id00800/00009.wav` | `bonafide` | 16,000 | 25.24 | 0.5632 | 0.1069 | 0 | Không |
| `id00797/00032.wav` | `bonafide` | 16,000 | 7.24 | 0.7479 | 0.1309 | 0 | Không |
| `id00799/00082.wav` | `bonafide` | 16,000 | 6.34 | 0.5293 | 0.0544 | 0 | Không |
| `id00968/00032.wav` | `bonafide` | 16,000 | 2.44 | 0.1636 | 0.0233 | 0 | Không |
| `id00967/00010.wav` | `bonafide` | 16,000 | 4.04 | 0.4851 | 0.0619 | 0 | Không |
| `id00966/00006.wav` | `bonafide` | 16,000 | 7.24 | 1.0000 | 0.1600 | 0 | Không |
| `id00979/00158.wav` | `bonafide` | 16,000 | 10.04 | 0.7714 | 0.1066 | 0 | Không |
| `id00976/00091.wav` | `bonafide` | 16,000 | 4.84 | 1.0000 | 0.3000 | 0 | Không |
| `id00979/00051.wav` | `bonafide` | 16,000 | 4.04 | 0.7401 | 0.1091 | 0 | Không |
| `id01005/adversarial_attack/00282.wav` | `adversarial_attack` | 16,000 | 2.76 | 0.4568 | 0.0539 | 0 | Không |
| `id01004/bonafide/00000.wav` | `bonafide` | 16,000 | 5.33 | 0.7287 | 0.0795 | 0 | Không |
| `id01006/voice_conversion/id01006_vc_00041.wav` | `voice_conversion` | 40,000 | 3.98 | 0.3731 | 0.0533 | 0 | Không |
| `id01006/adversarial_attack/00290.wav` | `adversarial_attack` | 16,000 | 3.04 | 0.5615 | 0.0693 | 0 | Không |
| `id01004/bonafide/00001.wav` | `bonafide` | 16,000 | 6.91 | 0.5165 | 0.0654 | 0 | Không |
| `id01006/voice_conversion/id01006_vc_00118.wav` | `voice_conversion` | 40,000 | 3.04 | 0.5010 | 0.0528 | 0 | Không |
| `id01005/adversarial_attack/00285.wav` | `adversarial_attack` | 16,000 | 5.20 | 0.6746 | 0.0876 | 0 | Không |
| `id01004/bonafide/00004.wav` | `bonafide` | 16,000 | 2.62 | 0.7169 | 0.1054 | 0 | Không |
| `id01005/voice_conversion/id01005_vc_00008.wav` | `voice_conversion` | 40,000 | 4.16 | 0.3528 | 0.0636 | 0 | Không |
| `id01009/voice_conversion/id01009_vc_00970.wav` | `voice_conversion` | 40,000 | 2.32 | 0.4328 | 0.0683 | 0 | Không |
| `id01009/voice_conversion/id01009_vc_00882.wav` | `voice_conversion` | 40,000 | 5.38 | 0.5647 | 0.0832 | 0 | Không |
| `id01009/voice_conversion/id01009_vc_00756.wav` | `voice_conversion` | 40,000 | 5.42 | 0.4731 | 0.0823 | 0 | Không |
| `id01009/adversarial_attack/03190.wav` | `adversarial_attack` | 16,000 | 6.20 | 0.5958 | 0.0924 | 0 | Không |
| `id01009/adversarial_attack/03081.wav` | `adversarial_attack` | 16,000 | 4.78 | 0.6944 | 0.0998 | 0 | Không |
| `id01009/adversarial_attack/02857.wav` | `adversarial_attack` | 16,000 | 2.90 | 0.4598 | 0.0621 | 0 | Không |
| `id01022/adversarial_attack/04531.wav` | `adversarial_attack` | 16,000 | 5.02 | 0.7710 | 0.0881 | 0 | Không |
| `id01022/voice_conversion/id01022_vc_02831.wav` | `voice_conversion` | 40,000 | 4.84 | 0.6938 | 0.0815 | 0 | Không |
| `id01022/adversarial_attack/04494.wav` | `adversarial_attack` | 16,000 | 3.32 | 0.5271 | 0.0804 | 0 | Không |
| `id01022/voice_conversion/id01022_vc_02853.wav` | `voice_conversion` | 40,000 | 6.96 | 0.3836 | 0.0467 | 0 | Không |
| `id01022/adversarial_attack/04520.wav` | `adversarial_attack` | 16,000 | 5.68 | 0.4642 | 0.0443 | 0 | Không |
| `id01022/voice_conversion/id01022_vc_02655.wav` | `voice_conversion` | 40,000 | 4.30 | 0.7446 | 0.0634 | 0 | Không |
| `id01025/adversarial_attack/08048.wav` | `adversarial_attack` | 16,000 | 1.76 | 0.7368 | 0.1411 | 0 | Không |
| `id01026/bonafide/00000.wav` | `bonafide` | 16,000 | 3.60 | 0.7887 | 0.0743 | 0 | Không |
| `id01026/voice_conversion/id01026_vc_00323.wav` | `voice_conversion` | 40,000 | 3.88 | 0.5077 | 0.0355 | 0 | Không |
| `id01025/adversarial_attack/08070.wav` | `adversarial_attack` | 16,000 | 1.42 | 0.7785 | 0.1556 | 0 | Không |
| `id01026/bonafide/00002.wav` | `bonafide` | 16,000 | 4.07 | 0.3363 | 0.0331 | 0 | Không |
| `id01026/voice_conversion/id01026_vc_00304.wav` | `voice_conversion` | 40,000 | 4.52 | 0.6842 | 0.0679 | 0 | Không |
| `id01025/adversarial_attack/08113.wav` | `adversarial_attack` | 16,000 | 3.12 | 0.7101 | 0.1217 | 0 | Không |
| `id01026/bonafide/00001.wav` | `bonafide` | 16,000 | 4.07 | 0.2874 | 0.0229 | 0 | Không |
| `id01026/voice_conversion/id01026_vc_00058.wav` | `voice_conversion` | 40,000 | 4.52 | 0.9214 | 0.0791 | 0 | Không |
| `id01026/adversarial_attack/10029.wav` | `adversarial_attack` | 16,000 | 6.32 | 0.8351 | 0.0768 | 0 | Không |
| `id01026/adversarial_attack/10069.wav` | `adversarial_attack` | 16,000 | 3.26 | 0.7880 | 0.0582 | 0 | Không |
| `id01026/adversarial_attack/09833.wav` | `adversarial_attack` | 16,000 | 3.80 | 0.8066 | 0.0586 | 0 | Không |
| `id01030/adversarial_attack/11277.wav` | `adversarial_attack` | 16,000 | 3.98 | 0.5083 | 0.0765 | 0 | Không |
| `id01030/voice_conversion/id01030_vc_02952.wav` | `voice_conversion` | 40,000 | 3.96 | 0.7827 | 0.1221 | 0 | Không |
| `id01030/adversarial_attack/11586.wav` | `adversarial_attack` | 16,000 | 3.44 | 0.2753 | 0.0431 | 0 | Không |
| `id01030/voice_conversion/id01030_vc_02970.wav` | `voice_conversion` | 40,000 | 3.64 | 0.2435 | 0.0312 | 0 | Không |
| `id01030/adversarial_attack/11649.wav` | `adversarial_attack` | 16,000 | 5.46 | 0.3937 | 0.0516 | 0 | Không |
| `id01030/voice_conversion/id01030_vc_02946.wav` | `voice_conversion` | 40,000 | 4.24 | 0.4223 | 0.0575 | 0 | Không |
| `id01037/voice_conversion/id01037_vc_01140.wav` | `voice_conversion` | 40,000 | 7.48 | 0.4374 | 0.0713 | 0 | Không |
| `id01037/voice_conversion/id01037_vc_01055.wav` | `voice_conversion` | 40,000 | 5.12 | 0.5674 | 0.0795 | 0 | Không |
| `id01037/voice_conversion/id01037_vc_01085.wav` | `voice_conversion` | 40,000 | 6.14 | 0.3710 | 0.0640 | 0 | Không |
| `id01037/adversarial_attack/17046.wav` | `adversarial_attack` | 16,000 | 3.56 | 0.3093 | 0.0619 | 0 | Không |
| `id01037/adversarial_attack/17395.wav` | `adversarial_attack` | 16,000 | 6.00 | 0.6380 | 0.0917 | 0 | Không |
| `id01037/adversarial_attack/17404.wav` | `adversarial_attack` | 16,000 | 4.30 | 0.2124 | 0.0440 | 0 | Không |
| `id01039/adversarial_attack/18301.wav` | `adversarial_attack` | 16,000 | 3.12 | 0.7975 | 0.0848 | 0 | Không |
| `id01039/adversarial_attack/18413.wav` | `adversarial_attack` | 16,000 | 6.64 | 0.6989 | 0.0755 | 0 | Không |
| `id01039/adversarial_attack/18236.wav` | `adversarial_attack` | 16,000 | 5.06 | 0.9368 | 0.1150 | 0 | Không |
| `id01039/adversarial_attack/18967.wav` | `adversarial_attack` | 16,000 | 2.68 | 0.6427 | 0.0745 | 0 | Không |
| `id01039/adversarial_attack/19092.wav` | `adversarial_attack` | 16,000 | 3.98 | 0.7601 | 0.0610 | 0 | Không |
| `id01039/adversarial_attack/18975.wav` | `adversarial_attack` | 16,000 | 5.60 | 0.6209 | 0.0670 | 0 | Không |
| `id01039/adversarial_attack/20135.wav` | `adversarial_attack` | 16,000 | 4.84 | 0.4173 | 0.0473 | 0 | Không |
| `id01039/adversarial_attack/20043.wav` | `adversarial_attack` | 16,000 | 4.76 | 0.5761 | 0.0561 | 0 | Không |
| `id01039/adversarial_attack/20214.wav` | `adversarial_attack` | 16,000 | 4.16 | 0.8770 | 0.0873 | 0 | Không |
| `id01042/voice_conversion/id01042_vc_02558.wav` | `voice_conversion` | 40,000 | 4.40 | 0.4760 | 0.0707 | 0 | Không |
| `id01042/voice_conversion/id01042_vc_02679.wav` | `voice_conversion` | 40,000 | 3.80 | 0.7616 | 0.0804 | 0 | Không |
| `id01042/voice_conversion/id01042_vc_02361.wav` | `voice_conversion` | 40,000 | 4.40 | 0.6172 | 0.0656 | 0 | Không |
| `id01048/adversarial_attack/24431.wav` | `adversarial_attack` | 16,000 | 2.70 | 0.4259 | 0.0674 | 0 | Không |
| `id01048/voice_conversion/id01048_vc_00253.wav` | `voice_conversion` | 40,000 | 2.68 | 0.4177 | 0.0469 | 0 | Không |
| `id01048/adversarial_attack/24595.wav` | `adversarial_attack` | 16,000 | 8.18 | 0.7201 | 0.0937 | 0 | Không |
| `id01048/voice_conversion/id01048_vc_00229.wav` | `voice_conversion` | 40,000 | 3.74 | 0.4201 | 0.0603 | 0 | Không |
| `id01048/adversarial_attack/24488.wav` | `adversarial_attack` | 16,000 | 2.32 | 0.5226 | 0.0585 | 0 | Không |
| `id01048/voice_conversion/id01048_vc_00250.wav` | `voice_conversion` | 40,000 | 3.18 | 0.5847 | 0.0627 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_01903.wav` | `voice_conversion` | 40,000 | 5.32 | 0.3612 | 0.0551 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_01967.wav` | `voice_conversion` | 40,000 | 2.90 | 0.5555 | 0.0922 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_01997.wav` | `voice_conversion` | 40,000 | 4.08 | 0.4780 | 0.0694 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_02491.wav` | `voice_conversion` | 40,000 | 3.52 | 0.6669 | 0.0964 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_02172.wav` | `voice_conversion` | 40,000 | 3.40 | 0.4722 | 0.0591 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_02622.wav` | `voice_conversion` | 40,000 | 3.04 | 0.6166 | 0.0886 | 0 | Không |
| `id01061/adversarial_attack/25660.wav` | `adversarial_attack` | 16,000 | 4.78 | 0.4147 | 0.0491 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_02793.wav` | `voice_conversion` | 40,000 | 4.60 | 0.6085 | 0.0977 | 0 | Không |
| `id01061/adversarial_attack/25594.wav` | `adversarial_attack` | 16,000 | 2.98 | 0.2968 | 0.0410 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_02955.wav` | `voice_conversion` | 40,000 | 4.52 | 0.9900 | 0.1040 | 0 | Không |
| `id01061/adversarial_attack/25690.wav` | `adversarial_attack` | 16,000 | 2.18 | 0.7464 | 0.0887 | 0 | Không |
| `id01061/voice_conversion/id01061_vc_02820.wav` | `voice_conversion` | 40,000 | 5.52 | 0.4480 | 0.0706 | 0 | Không |
| `id01066/bonafide/00569.wav` | `bonafide` | 16,000 | 3.22 | 0.4937 | 0.0483 | 0 | Không |
| `id01066/bonafide/00142.wav` | `bonafide` | 16,000 | 5.09 | 0.4616 | 0.0673 | 0 | Không |
| `id01066/bonafide/00364.wav` | `bonafide` | 16,000 | 4.61 | 0.6383 | 0.0714 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_00693.wav` | `voice_conversion` | 40,000 | 5.46 | 0.9117 | 0.0739 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_00696.wav` | `voice_conversion` | 40,000 | 3.44 | 0.3613 | 0.0428 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_00730.wav` | `voice_conversion` | 40,000 | 3.88 | 0.6935 | 0.0721 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_01522.wav` | `voice_conversion` | 40,000 | 4.84 | 0.6170 | 0.0522 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_01466.wav` | `voice_conversion` | 40,000 | 4.78 | 0.7962 | 0.0843 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_01441.wav` | `voice_conversion` | 40,000 | 3.70 | 0.3698 | 0.0411 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_01828.wav` | `voice_conversion` | 40,000 | 5.00 | 0.8767 | 0.0754 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_01691.wav` | `voice_conversion` | 40,000 | 2.86 | 0.7550 | 0.0582 | 0 | Không |
| `id01070/voice_conversion/id01070_vc_01583.wav` | `voice_conversion` | 40,000 | 5.06 | 0.2538 | 0.0292 | 0 | Không |
| `id01076/adversarial_attack/32252.wav` | `adversarial_attack` | 16,000 | 4.48 | 0.8595 | 0.0898 | 0 | Không |
| `id01076/bonafide/00003.wav` | `bonafide` | 16,000 | 3.70 | 0.8588 | 0.1019 | 0 | Không |
| `id01076/voice_conversion/id01076_vc_00048.wav` | `voice_conversion` | 40,000 | 4.12 | 0.9900 | 0.0904 | 0 | Không |
| `id01075/adversarial_attack/32183.wav` | `adversarial_attack` | 16,000 | 3.34 | 0.7546 | 0.1130 | 0 | Không |
| `id01076/bonafide/00005.wav` | `bonafide` | 16,000 | 3.62 | 0.6288 | 0.0805 | 0 | Không |
| `id01076/voice_conversion/id01076_vc_00166.wav` | `voice_conversion` | 40,000 | 3.70 | 0.5841 | 0.0851 | 0 | Không |
| `id01075/adversarial_attack/32238.wav` | `adversarial_attack` | 16,000 | 4.04 | 0.7279 | 0.1246 | 0 | Không |
| `id01076/bonafide/00002.wav` | `bonafide` | 16,000 | 3.82 | 0.8306 | 0.0812 | 0 | Không |
| `id01076/voice_conversion/id01076_vc_00078.wav` | `voice_conversion` | 40,000 | 4.48 | 0.6220 | 0.0842 | 0 | Không |
| `id01080/adversarial_attack/33600.wav` | `adversarial_attack` | 16,000 | 5.78 | 0.5191 | 0.0587 | 0 | Không |
| `id01080/adversarial_attack/33643.wav` | `adversarial_attack` | 16,000 | 4.40 | 0.5955 | 0.0836 | 0 | Không |
| `id01080/adversarial_attack/33657.wav` | `adversarial_attack` | 16,000 | 3.48 | 0.7157 | 0.0944 | 0 | Không |
| `id01093/adversarial_attack/36315.wav` | `adversarial_attack` | 16,000 | 5.14 | 0.7492 | 0.0533 | 0 | Không |
| `id01093/bonafide/00004.wav` | `bonafide` | 16,000 | 2.20 | 0.8518 | 0.0866 | 0 | Không |
| `id01093/voice_conversion/id01093_vc_00052.wav` | `voice_conversion` | 40,000 | 4.16 | 0.3392 | 0.0332 | 0 | Không |
| `id01092/adversarial_attack/36220.wav` | `adversarial_attack` | 16,000 | 4.66 | 0.9868 | 0.1259 | 0 | Không |
| `id01093/bonafide/00002.wav` | `bonafide` | 16,000 | 4.25 | 0.7110 | 0.0852 | 0 | Không |
| `id01095/voice_conversion/id01095_vc_00022.wav` | `voice_conversion` | 40,000 | 3.68 | 0.9091 | 0.0939 | 0 | Không |
| `id01093/adversarial_attack/36307.wav` | `adversarial_attack` | 16,000 | 5.20 | 0.3284 | 0.0270 | 0 | Không |
| `id01095/bonafide/00002.wav` | `bonafide` | 16,000 | 3.98 | 0.7085 | 0.0972 | 0 | Không |
| `id01092/voice_conversion/id01092_vc_00179.wav` | `voice_conversion` | 40,000 | 5.24 | 0.9330 | 0.1097 | 0 | Không |
| `id01095/adversarial_attack/36508.wav` | `adversarial_attack` | 16,000 | 2.24 | 0.4756 | 0.0707 | 0 | Không |
| `id01095/voice_conversion/id01095_vc_00785.wav` | `voice_conversion` | 40,000 | 3.12 | 0.9762 | 0.1473 | 0 | Không |
| `id01095/adversarial_attack/36377.wav` | `adversarial_attack` | 16,000 | 3.56 | 1.0000 | 0.0994 | 0 | Không |
| `id01095/voice_conversion/id01095_vc_00805.wav` | `voice_conversion` | 40,000 | 2.80 | 0.9900 | 0.1087 | 0 | Không |
| `id01095/adversarial_attack/36391.wav` | `adversarial_attack` | 16,000 | 3.48 | 0.9904 | 0.1101 | 0 | Không |
| `id01095/voice_conversion/id01095_vc_00723.wav` | `voice_conversion` | 40,000 | 3.48 | 0.9900 | 0.1299 | 0 | Không |
| `id01096/adversarial_attack/37212.wav` | `adversarial_attack` | 16,000 | 5.08 | 0.3822 | 0.0462 | 0 | Không |
| `id01096/voice_conversion/id01096_vc_02486.wav` | `voice_conversion` | 40,000 | 7.14 | 0.7452 | 0.0992 | 0 | Không |
| `id01096/adversarial_attack/37241.wav` | `adversarial_attack` | 16,000 | 3.14 | 0.3238 | 0.0479 | 0 | Không |
| `id01096/voice_conversion/id01096_vc_02168.wav` | `voice_conversion` | 40,000 | 4.48 | 0.4554 | 0.0672 | 0 | Không |
| `id01096/adversarial_attack/37237.wav` | `adversarial_attack` | 16,000 | 3.80 | 0.5274 | 0.0620 | 0 | Không |
| `id01096/voice_conversion/id01096_vc_02271.wav` | `voice_conversion` | 40,000 | 1.68 | 0.4850 | 0.0834 | 0 | Không |
| `id01096/adversarial_attack/38163.wav` | `adversarial_attack` | 16,000 | 3.32 | 0.3303 | 0.0414 | 0 | Không |
| `id01096/adversarial_attack/38048.wav` | `adversarial_attack` | 16,000 | 3.68 | 0.4349 | 0.0443 | 0 | Không |
| `id01096/adversarial_attack/38178.wav` | `adversarial_attack` | 16,000 | 5.40 | 0.3857 | 0.0503 | 0 | Không |
| `id01098/voice_conversion/id01098_vc_02416.wav` | `voice_conversion` | 40,000 | 3.48 | 0.4964 | 0.0775 | 0 | Không |
| `id01098/voice_conversion/id01098_vc_02396.wav` | `voice_conversion` | 40,000 | 1.82 | 0.5781 | 0.0704 | 0 | Không |
| `id01098/voice_conversion/id01098_vc_02323.wav` | `voice_conversion` | 40,000 | 1.90 | 0.5164 | 0.0841 | 0 | Không |
| `id01099/voice_conversion/id01099_vc_00635.wav` | `voice_conversion` | 40,000 | 1.20 | 0.7587 | 0.1118 | 0 | Không |
| `id01099/voice_conversion/id01099_vc_00793.wav` | `voice_conversion` | 40,000 | 5.68 | 0.6086 | 0.0694 | 0 | Không |
| `id01099/voice_conversion/id01099_vc_00948.wav` | `voice_conversion` | 40,000 | 3.40 | 0.5490 | 0.0670 | 0 | Không |
| `id01100/voice_conversion/id01100_vc_00640.wav` | `voice_conversion` | 40,000 | 1.68 | 0.3795 | 0.0720 | 0 | Không |
| `id01100/voice_conversion/id01100_vc_00781.wav` | `voice_conversion` | 40,000 | 4.92 | 0.3359 | 0.0495 | 0 | Không |
| `id01100/voice_conversion/id01100_vc_00981.wav` | `voice_conversion` | 40,000 | 3.86 | 0.3226 | 0.0475 | 0 | Không |
| `id01100/adversarial_attack/44383.wav` | `adversarial_attack` | 16,000 | 2.90 | 0.3418 | 0.0448 | 0 | Không |
| `id01100/voice_conversion/id01100_vc_02505.wav` | `voice_conversion` | 40,000 | 8.12 | 0.6778 | 0.1192 | 0 | Không |
| `id01100/adversarial_attack/44399.wav` | `adversarial_attack` | 16,000 | 3.14 | 0.3995 | 0.0610 | 0 | Không |
| `id01100/voice_conversion/id01100_vc_02158.wav` | `voice_conversion` | 40,000 | 3.86 | 0.4385 | 0.0762 | 0 | Không |
| `id01100/adversarial_attack/44385.wav` | `adversarial_attack` | 16,000 | 3.24 | 0.3815 | 0.0620 | 0 | Không |
| `id01100/voice_conversion/id01100_vc_02441.wav` | `voice_conversion` | 40,000 | 1.36 | 0.4012 | 0.0922 | 0 | Không |
| `id01101/adversarial_attack/47280.wav` | `adversarial_attack` | 16,000 | 2.24 | 0.3438 | 0.0588 | 0 | Không |
| `id01101/voice_conversion/id01101_vc_01405.wav` | `voice_conversion` | 40,000 | 2.98 | 0.5300 | 0.0700 | 0 | Không |
| `id01101/adversarial_attack/47061.wav` | `adversarial_attack` | 16,000 | 3.74 | 0.3721 | 0.0518 | 0 | Không |
| `id01101/voice_conversion/id01101_vc_01251.wav` | `voice_conversion` | 40,000 | 2.46 | 0.3101 | 0.0578 | 0 | Không |
| `id01101/adversarial_attack/47107.wav` | `adversarial_attack` | 16,000 | 4.24 | 0.4570 | 0.0608 | 0 | Không |
| `id01101/voice_conversion/id01101_vc_01362.wav` | `voice_conversion` | 40,000 | 2.48 | 0.6023 | 0.0467 | 0 | Không |
| `id01105/bonafide/00106.wav` | `bonafide` | 16,000 | 3.94 | 0.4700 | 0.0460 | 0 | Không |
| `id01105/voice_conversion/id01105_vc_00308.wav` | `voice_conversion` | 40,000 | 2.02 | 0.4109 | 0.0652 | 0 | Không |
| `id01105/bonafide/00155.wav` | `bonafide` | 16,000 | 3.00 | 0.6019 | 0.0591 | 0 | Không |
| `id01105/voice_conversion/id01105_vc_00043.wav` | `voice_conversion` | 40,000 | 3.30 | 0.3958 | 0.0521 | 0 | Không |
| `id01105/bonafide/00165.wav` | `bonafide` | 16,000 | 4.31 | 0.4033 | 0.0528 | 0 | Không |
| `id01105/voice_conversion/id01105_vc_00164.wav` | `voice_conversion` | 40,000 | 4.92 | 0.4682 | 0.0637 | 0 | Không |
| `id01109/voice_conversion/id01109_vc_00241.wav` | `voice_conversion` | 40,000 | 2.18 | 0.4340 | 0.0574 | 0 | Không |
| `id01109/voice_conversion/id01109_vc_00580.wav` | `voice_conversion` | 40,000 | 2.18 | 0.5340 | 0.0803 | 0 | Không |
| `id01109/voice_conversion/id01109_vc_00563.wav` | `voice_conversion` | 40,000 | 1.76 | 0.3735 | 0.0434 | 0 | Không |
| `id01114/bonafide/00189.wav` | `bonafide` | 16,000 | 2.81 | 0.7597 | 0.1562 | 0 | Không |
| `id01114/bonafide/00137.wav` | `bonafide` | 16,000 | 2.66 | 0.8586 | 0.1538 | 0 | Không |
| `id01114/bonafide/00224.wav` | `bonafide` | 16,000 | 3.66 | 0.7101 | 0.1537 | 0 | Không |
| `id01129/bonafide/00029.wav` | `bonafide` | 16,000 | 2.81 | 0.5263 | 0.0624 | 0 | Không |
| `id01129/replay/id01129_replay_00004.wav` | `replay` | 16,000 | 3.14 | 0.9993 | 0.1846 | 0 | Không |
| `id01126/bonafide/00000.wav` | `bonafide` | 16,000 | 2.69 | 0.3026 | 0.0379 | 0 | Không |
| `id01128/replay/id01128_replay_00016.wav` | `replay` | 16,000 | 3.52 | 0.9997 | 0.2054 | 0 | Không |
| `id01126/bonafide/00032.wav` | `bonafide` | 16,000 | 2.41 | 0.5459 | 0.0695 | 0 | Không |
| `id01124/replay/id01124_replay_00035.wav` | `replay` | 16,000 | 2.56 | 0.7595 | 0.1439 | 0 | Không |
| `id00930/adversarial_attack/56114.wav` | `adversarial_attack` | 16,000 | 8.82 | 0.2947 | 0.0702 | 0 | Không |
| `id00947/voice_conversion/id00947_vc_00107.wav` | `voice_conversion` | 40,000 | 5.62 | 0.3015 | 0.0462 | 0 | Không |
| `id00930/adversarial_attack/56166.wav` | `adversarial_attack` | 16,000 | 27.10 | 0.3454 | 0.0670 | 0 | Không |
| `id00947/voice_conversion/id00947_vc_00076.wav` | `voice_conversion` | 40,000 | 4.02 | 0.4096 | 0.0666 | 0 | Không |
| `id00930/adversarial_attack/56175.wav` | `adversarial_attack` | 16,000 | 30.02 | 0.3688 | 0.0636 | 0 | Không |
| `id00947/voice_conversion/id00947_vc_00145.wav` | `voice_conversion` | 40,000 | 4.02 | 0.3227 | 0.0535 | 0 | Không |
| `id00965/adversarial_attack/58413.wav` | `adversarial_attack` | 16,000 | 8.82 | 0.8295 | 0.1080 | 0 | Không |
| `id00966/voice_conversion/id00966_vc_00108.wav` | `voice_conversion` | 40,000 | 3.86 | 0.6002 | 0.0882 | 0 | Không |
| `id00965/adversarial_attack/58404.wav` | `adversarial_attack` | 16,000 | 30.82 | 0.8973 | 0.1103 | 0 | Không |
| `id00966/voice_conversion/id00966_vc_00097.wav` | `voice_conversion` | 40,000 | 5.04 | 0.6647 | 0.0763 | 0 | Không |
| `id00965/adversarial_attack/58363.wav` | `adversarial_attack` | 16,000 | 31.76 | 0.9685 | 0.1300 | 0 | Không |
| `id00966/voice_conversion/id00966_vc_00060.wav` | `voice_conversion` | 40,000 | 6.14 | 0.6753 | 0.0816 | 0 | Không |
| `id00992/adversarial_attack/60742.wav` | `adversarial_attack` | 16,000 | 14.82 | 0.7189 | 0.1245 | 0 | Không |
| `id00995/voice_conversion/id00995_vc_00034.wav` | `voice_conversion` | 40,000 | 18.02 | 0.4103 | 0.0487 | 0 | Không |
| `id00995/adversarial_attack/60871.wav` | `adversarial_attack` | 16,000 | 16.02 | 0.4952 | 0.0878 | 0 | Không |
| `id00995/voice_conversion/id00995_vc_00110.wav` | `voice_conversion` | 40,000 | 13.62 | 0.5091 | 0.0806 | 0 | Không |
| `id00995/adversarial_attack/60886.wav` | `adversarial_attack` | 16,000 | 9.62 | 0.4886 | 0.0793 | 0 | Không |
| `id00995/voice_conversion/id00995_vc_00064.wav` | `voice_conversion` | 40,000 | 15.22 | 0.3069 | 0.0472 | 0 | Không |

## Cảnh báo

- Sample rate không đồng nhất giữa các loại; đây là shortcut risk. Phải resample toàn bộ waveform về 16000 Hz.
- 68 shard chỉ là một phần nhỏ, không ngẫu nhiên của 432 shard; không suy rộng phân bố cục bộ cho toàn bộ snapshot.

## Quyết định

Có thể tiếp tục xây dựng pipeline và baseline. Mọi waveform phải được resample về mono 16 kHz trong pipeline; kết quả trên 68 shard chỉ dùng cho smoke test, không đại diện toàn bộ snapshot.
