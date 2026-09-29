# Báo cáo kiểm tra rò rỉ dữ liệu

- **Trạng thái:** ĐẠT
- **Metadata:** `data/metadata/vsasv_metadata.csv`
- **Thư mục split:** `data/splits`
- **Thời điểm UTC:** `2026-09-21T23:16:33.927402+00:00`

## Tóm tắt split

| Split | Mẫu | Speaker | File duy nhất | File lặp |
|---|---:|---:|---:|---:|
| `closed_train` | 150,999 | 798 | 150,999 | 0 |
| `closed_dev` | 39,116 | 172 | 39,116 | 0 |
| `closed_test` | 30,848 | 171 | 30,848 | 0 |
| `open_train_vc` | 108,624 | 785 | 108,624 | 0 |
| `open_dev_vc` | 26,623 | 169 | 26,623 | 0 |
| `open_seen_test_vc` | 23,247 | 168 | 23,247 | 0 |
| `open_unseen_test_adversarial` | 23,247 | 168 | 23,247 | 0 |
| `open_unseen_test_replay` | 1,520 | 19 | 1,520 | 0 |

## Kiểm tra audio

- Không cung cấp --audio-root; chưa kiểm tra tồn tại và hash audio.

## Vấn đề phát hiện

Không phát hiện rò rỉ speaker, file hoặc vi phạm giao thức.
