# Benchmark A/B loader split cũ và manifest v2

## Trạng thái

- Kết quả: **ĐẠT**
- Thời điểm: `2026-10-10T15:35:15+07:00`
- Phạm vi: benchmark kỹ thuật, không phải kết quả khoa học.
- Manifest: `development-20k-v2`
- Seed: `2026`
- Mẫu mỗi lượt: 16
- Số lần lặp: 3
- Batch size: 8
- Worker: [0, 2]
- Policy: `none`
- Train mode/crop epoch: `True` / `0`

## Tương đương đầu ra

- File, speaker, `utt_type`, nhãn và waveform sau tiền xử lý: **ĐẠT**.
- Hai dataset được sắp theo cùng danh sách file trước khi đọc để cùng vị trí nhận cùng crop seed.
- Checksum danh sách file: `7167059f160e12cfe6ac6973d78cff54a54b6b948de4afeb5571b158fa543b56`
- Số shard trong mẫu benchmark: 14

## Khởi tạo dataset

| Loader | Trung vị | P95 | Bản ghi khả dụng | Coverage |
|---|---:|---:|---:|---:|
| Split cũ | 1065.47 ms | 2025.41 ms | 22,767 | 15.08% |
| Manifest v2 | 714.09 ms | 809.18 ms | 15,885 | 100.00% |

Manifest giảm thời gian khởi tạo so với split cũ **32.98%**.

## Đọc batch

| Biến thể | Mẫu/giây | Trung vị/lượt | First batch | Peak RSS |
|---|---:|---:|---:|---:|
| `legacy_workers0` | 0.631 | 25321.23 ms | 14470.31 ms | 344.66 MiB |
| `manifest_workers0` | 0.645 | 25221.05 ms | 14483.75 ms | 345.66 MiB |
| `legacy_workers2` | 0.568 | 28172.58 ms | 27261.03 ms | 865.41 MiB |
| `manifest_workers2` | 0.593 | 26071.69 ms | 25191.73 ms | 848.07 MiB |

## So sánh

- Worker 0: manifest so với split cũ +2.13% throughput; peak RSS chênh +1.00 MiB.
- Worker 2: manifest so với split cũ +4.51% throughput; peak RSS chênh -17.34 MiB.
- Trên loader manifest, worker 2 so với worker 0: -8.02% throughput; peak RSS chênh +502.40 MiB.

## Quyết định

- Trạng thái D014: **GIỮ D014**.
- Ngưỡng cải thiện worker: 15.00%.
- Khuyến nghị: Giữ batch_size=8, num_workers=0 vì worker 2 chưa vượt ngưỡng cải thiện 15% trên loader manifest.

## Cache và giới hạn

- Không xóa cache hệ điều hành; không gọi lượt đầu là cold cache.
- Thứ tự loader được đảo xen kẽ giữa các lần lặp và được lưu đầy đủ trong JSON.
- Kết quả chỉ áp dụng cho máy, 16 file và các shard đang có.
- Hai loader dùng chung logic truy vấn/decode từng waveform; phép đo này không tự chứng minh một layout/cache mới đã tối ưu đường đọc.
