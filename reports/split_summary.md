# Báo cáo tạo split VSASV

- **Seed:** `2026`
- **Metadata:** `data/metadata/vsasv_metadata.csv`
- **SHA-256 metadata:** `254887ac5a1e357cbe1a8f0ee4bde266c491c9c349505e61c5b3dd2b461e73fb`
- **Số mẫu:** 220,963
- **Số speaker:** 1,141
- **Thời điểm UTC:** `2026-09-21T23:14:44.968902+00:00`

## Phân bổ speaker theo strata

| Stratum | Tổng speaker | Train | Dev | Test |
|---|---:|---:|---:|---:|
| `real_only` | 978 | 684 | 147 | 147 |
| `vc_adversarial` | 144 | 101 | 22 | 21 |
| `replay` | 19 | 13 | 3 | 3 |

## Kết quả từng split

| Split | Mẫu | Speaker | Bonafide | Spoof | SHA-256 |
|---|---:|---:|---:|---:|---|
| `closed_train` | 150,999 | 798 | 67,809 | 83,190 | `656f95dbc30bef010fdf04facfbd69fce0a088f7d5bc93d858e9241fee675252` |
| `closed_dev` | 39,116 | 172 | 14,490 | 24,626 | `d93c28c54a2619d901f4ef60b86316f68ca32c997890aa56e205196d35d91a32` |
| `closed_test` | 30,848 | 171 | 16,006 | 14,842 | `304cf9e202d1470d1af8c24d0afae951a9c038129b8cd13344a69fc8e29c8abf` |
| `open_train_vc` | 108,624 | 785 | 67,289 | 41,335 | `c653b2d66cb0acf90429bcedbae3ea361264880d90538420aad76f6a94fdcb62` |
| `open_dev_vc` | 26,623 | 169 | 14,370 | 12,253 | `5a8eba6370e5f3f740b78d42287680862ee6052afddb5ccb1b1341b56492366f` |
| `open_seen_test_vc` | 23,247 | 168 | 15,886 | 7,361 | `12386a559a9b0377021b9bbe004374e9dd56059e0264967cdee52d42f34b4cf9` |
| `open_unseen_test_adversarial` | 23,247 | 168 | 15,886 | 7,361 | `ce780006433703e35153d4684c9a62185c242897df8bc2438aa9da1695a97fd3` |
| `open_unseen_test_replay` | 1,520 | 19 | 760 | 760 | `3df5b9646b3e70d009bca393d3f9c107b7f0142dc37278e8936478a9b974b2f6` |

## Quy ước quan trọng

- Toàn bộ split được xếp theo speaker bằng SHA-256 của seed, stratum và speaker ID.
- Speaker replay được giữ ngoài toàn bộ open train/dev/seen/adversarial.
- Seen VC test và unseen adversarial test dùng chung tập bonafide test. Đây là overlap có chủ đích giữa hai phép đánh giá, không phải rò rỉ train/test.
- CSV giữ nguyên ba cột nguồn: `file,label,utt_type`.
