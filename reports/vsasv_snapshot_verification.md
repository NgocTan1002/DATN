# Báo cáo xác minh snapshot VSASV công khai

- **Nhất quán kỹ thuật:** KHÔNG ĐẠT
- **Giá trị khoa học:** `provisional_public_snapshot`
- **Tương đương bộ dữ liệu/giao thức bài báo gốc:** KHÔNG
- **Thời điểm UTC:** `2026-10-09T08:40:52.359107+00:00`
- **Tên phiên bản đề xuất:** `VSASV-HF-public-snapshot-v1`

## Kết luận điều hành

Snapshot nguồn chưa vượt toàn bộ kiểm tra kỹ thuật; xem hard issue bên dưới. Trạng thái development manifest được báo riêng và không được suy ra từ trạng thái nguồn. Đối chiếu với các nhóm duplicate nguồn đã xác nhận cho thấy development manifest không giữ nhiều file trong cùng nhóm; cổng độc lập băm lại toàn bộ manifest vẫn phải được đọc từ báo cáo audit tương ứng.

### Hard issue

- Phát hiện waveform trùng giữa nhiều file và đã xác nhận bằng SHA-256.

## Metadata công khai

- File: `data/metadata/vsasv_metadata.csv`
- SHA-256: `254887ac5a1e357cbe1a8f0ee4bde266c491c9c349505e61c5b3dd2b461e73fb`
- Tổng dòng: 220,963
- File logic duy nhất: 220,963
- Speaker: 1,141
- Giá trị thiếu, dòng/file trùng, sai prefix speaker và sai quy tắc tên: 0

### Đối chiếu với bài báo

| Loại | Snapshot công khai | Bài báo | Chênh lệch | Snapshot/Bài báo |
|---|---:|---:|---:|---:|
| Bona fide | 98,305 | 164,374 | -66,069 | 59.81% |
| VC | 60,949 | 100,564 | -39,615 | 60.61% |
| AP | 60,949 | 16,731 | +44,218 | 364.29% |
| Replay | 760 | 57,571 | -56,811 | 1.32% |
| Tổng | 220,963 | 339,240 | -118,277 | 65.13% |

### Quan hệ VC–AP

- VC: 60,949 mẫu; AP: 60,949 mẫu.
- 144/144 speaker tấn công có số VC bằng đúng số AP.
- Đường dẫn VC trùng đường dẫn AP: 0.
- Chỉ 1/144 speaker có cùng tập chỉ số số học trong tên VC và AP.

Kết luận: sự bằng nhau về số lượng là cấu trúc thật của metadata công khai, nhưng không chứng minh VC và AP là cùng file hoặc do script join sai.

## Parquet cục bộ (68 shard)

- Shard: 68/432 (15.74% theo số shard).
- Tổng hàng: 34,782.
- Khớp metadata chính xác: 34,782/34,782.
- File/audio rỗng hoặc sample rate không hợp lệ: 0.
- Nhóm fingerprint audio lặp: 91.
- Nhóm waveform lặp đã xác nhận bằng SHA-256: 91.
- Nhóm đi qua nhiều speaker: 33.
- Nhóm đi qua nhiều closed split: 30.
- Nhóm trộn nhãn nhị phân: 0.
- Development manifest đối chiếu: `data/manifests/development_20k_v2.csv`; SHA-256 `f549423b1fb33665f7e5606cdf56321bc8cd07246a0cdf0b90cd918c0c09e728`.
- Trong development manifest, khi đối chiếu các nhóm duplicate nguồn đã xác nhận: 0 nhóm giữ nhiều file; 0 nhóm đi qua train/development.

### Phân bố theo loại

| Loại | Số mẫu |
|---|---:|
| `bonafide` | 16,324 |
| `voice_conversion` | 11,073 |
| `adversarial_attack` | 7,120 |
| `replay` | 265 |

### Sample rate quan sát được

| Loại | Sample rate | Số mẫu |
|---|---:|---:|
| `adversarial_attack` | 16,000 Hz | 7,120 |
| `bonafide` | 16,000 Hz | 16,324 |
| `replay` | 16,000 Hz | 265 |
| `voice_conversion` | 40,000 Hz | 11,073 |

Cảnh báo: 68 shard cục bộ có thể không đại diện cho toàn bộ snapshot. Các sample rate quan sát được được liệt kê ở bảng trên; pipeline phải resample mọi waveform về cùng một sample rate.

## Tám split

- Trạng thái kiểm tra schema, coverage, file và speaker leakage: ĐẠT.
- Kiểm tra fingerprint audio đã chạy trên toàn bộ 34,782 file cục bộ; chưa bao phủ toàn bộ snapshot vì hiện có 68/432 shard.
- Đây là custom speaker-disjoint protocol của đồ án, không phải official split của bài báo.

## Cảnh báo và giới hạn

- Số mẫu của snapshot công khai không khớp tổng số trong bài báo gốc.
- Mọi speaker tấn công đều có số VC bằng đúng số AP; đây là cấu trúc cần lưu ý, không phải bằng chứng lỗi join.
- Replay trong snapshot công khai rất nhỏ so với bài báo gốc.
- Các loại audio trong 68 shard có sample rate không đồng nhất; phải resample nhất quán trước huấn luyện.
- Parquet cục bộ chỉ là một phần nhỏ và không phải mẫu ngẫu nhiên của 432 shard.
- Kiểm tra fingerprint audio chỉ bao phủ 34,782 file trong 68 shard cục bộ.
- Metadata không có `generator_id`, `source_corpus`, `official_split` hoặc định danh câu nguồn.
- Không thể tự chứng minh nguyên nhân tác giả tạo số VC/AP bằng nhau chỉ từ ba cột metadata.
- Replay công khai quá nhỏ để đại diện đầy đủ cho replay trong bài báo.

## Quyết định sử dụng

Không dùng trực tiếp snapshot nguồn cho kết quả khoa học. Có thể tiếp tục phát triển pipeline và thử nghiệm baseline bằng development manifest chỉ khi manifest đó vượt audit content hash độc lập, đồng thời áp dụng các điều kiện sau:

- Resample mọi waveform về một sample rate được khai báo thống nhất trong pipeline.
- Giữ train/dev/test tách biệt speaker.
- Báo cáo riêng VC, AP và replay; ghi rõ replay là tập con công khai hạn chế.
- Không so sánh trực tiếp metric của custom split với metric trong bài báo gốc.
- Chạy lại xác minh sau mỗi lần tải thêm shard.

Không được dùng báo cáo này để tuyên bố đã tái lập official VSASV hoặc kết quả bài báo gốc.

## Nguồn đối chiếu

- [Bài báo VSASV gốc](https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.pdf)
- [Trang bài báo Interspeech](https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.html)
- [Bản VSASV công khai trên Hugging Face](https://huggingface.co/datasets/hustep-lab/VSASV-Dataset)
