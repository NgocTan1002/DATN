# Kế hoạch shard cho dev-full/test-full (D025)

- Phạm vi: lập kế hoạch tải và dung lượng; không tải shard, không đọc waveform, không có metric.
- Nguồn: `hustep-lab/VSASV-Dataset` revision `92de668616780ed4f3d7f58a82262f82b56b4557`.
- Shard cục bộ: **68** (9,85 GiB).

## Shard theo split

| Split | Dòng gốc | Shard cần | Đã có | Còn thiếu |
|---|---:|---:|---:|---:|
| `closed_dev` | 39.116 | 186 | 32 | 154 |
| `closed_test` | 30.848 | 162 | 28 | 134 |

- Hợp dev + test cần **265** shard; còn thiếu **221**.
- Chỉ dev: 87 shard; chỉ test: 67; cả hai: 67.
- Dung lượng cần tải: **33,67 GiB** (nguồn: `huggingface_tree`); riêng phần chứa test: 20,53 GiB.
- Danh sách shard thiếu: 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 42, 43, 44, 45, 47, 48, 49, 50, 52, 53, 54, 56, 58, 59, 60, 62, 63, 65, 66, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 86, 87, 89, 90, 91, 93, 94, 95, 96, 97, 99, 100, 102, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 124, 125, 126, 127, 129, 130, 132, 133, 134, 135, 136, 137, 140, 142, 143, 144, 145, 146, 149, 150, 151, 152, 153, 154, 155, 158, 159, 160, 161, 162, 163, 164, 168, 169, 170, 172, 185, 187, 188, 189, 190, 191, 192, 194, 195, 196, 197, 198, 199, 200, 214, 225, 226, 227, 239, 240, 241, 242, 243, 244, 245, 246, 249, 251, 252, 264, 267, 268, 269, 270, 271, 272, 273, 277, 278, 279, 280, 281, 282, 283, 290, 291, 295, 296, 297, 298, 299, 300, 301, 302, 304, 316, 317, 320, 322, 323, 370, 371, 372, 373, 374, 375, 396, 398, 399, 400, 401, 402, 404, 405, 406, 407, 408, 409, 410, 411, 412, 414, 415, 416, 417, 418, 419, 420, 421, 426, 427, 429.

## Ngân sách đĩa

- Dung lượng trống hiện tại: **59,17 GiB**; chỉ coi là vừa nếu nhu cầu không vượt 90% số này (dự phòng 10%).
- Packed ước tính (train manifest + dev + test, trước khử trùng): 23,83 GiB với 297990 byte/mẫu; số này rút từ PoC 24 mẫu nên chỉ là tham chiếu thô.
- Giữ Parquet đã tải cùng packed: **57,50 GiB** — KHÔNG vừa chỗ trống (có dự phòng).
- Đóng gói từng shard rồi xóa Parquet (đỉnh = packed + shard lớn nhất): **24,66 GiB** — vừa chỗ trống. Phương án này cần quy trình build tăng dần chưa được triển khai.

## Giới hạn

- Công thức file→shard dựa trên thứ tự dòng của `vsasv_metadata.csv`; phải được kiểm lại trên shard mới tải trước khi dùng cho manifest.
- Số mẫu là số dòng split gốc; khử trùng nội dung D025 sẽ loại thêm mẫu.
- Quyết định nơi lưu, giữ hay xóa Parquet và việc tải cần xác nhận riêng của người dùng.
