# Báo cáo xác minh snapshot VSASV công khai

- **Nhất quán kỹ thuật:** ĐẠT
- **Giá trị khoa học:** `provisional_public_snapshot`
- **Tương đương bộ dữ liệu/giao thức bài báo gốc:** KHÔNG
- **Thời điểm UTC:** `2026-09-24T09:33:49.556874+00:00`
- **Tên phiên bản đề xuất:** `VSASV-HF-public-snapshot-v1`

## Kết luận điều hành

Metadata, năm Parquet cục bộ và tám split hiện có nhất quán với nhau ở các kiểm tra đã chạy. Không tìm thấy bằng chứng về lỗi join làm nhân đôi VC thành AP. Tuy nhiên, snapshot công khai khác đáng kể so với thống kê bài báo gốc nên chỉ được dùng như một giao thức tùy chỉnh, có phiên bản và có giới hạn rõ ràng.

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

## Năm Parquet cục bộ

- Shard: 5/432 (1.16% theo số shard).
- Tổng hàng: 2,558.
- Khớp metadata chính xác: 2,558/2,558.
- File/audio rỗng hoặc sample rate không hợp lệ: 0.
- Nhóm fingerprint audio lặp: 0.

### Phân bố theo loại

| Loại | Số mẫu |
|---|---:|
| `bonafide` | 1,292 |
| `voice_conversion` | 188 |
| `adversarial_attack` | 813 |
| `replay` | 265 |

### Sample rate quan sát được

| Loại | Sample rate | Số mẫu |
|---|---:|---:|
| `adversarial_attack` | 16,000 Hz | 813 |
| `bonafide` | 16,000 Hz | 1,292 |
| `replay` | 16,000 Hz | 265 |
| `voice_conversion` | 40,000 Hz | 188 |

Cảnh báo: 188 VC trong phần đã tải đều là 40 kHz, trong khi các mẫu cục bộ còn lại là 16 kHz. Vì năm shard được chọn theo vị trí chứ không ngẫu nhiên, không được suy rộng tỷ lệ này cho toàn bộ snapshot. Pipeline phải resample mọi waveform về cùng một sample rate.

## Tám split

- Trạng thái kiểm tra schema, coverage, file và speaker leakage: ĐẠT.
- Kiểm tra hash audio toàn bộ split: chưa thể chạy vì mới có 5/432 shard.
- Đây là custom speaker-disjoint protocol của đồ án, không phải official split của bài báo.

## Cảnh báo và giới hạn

- Số mẫu của snapshot công khai không khớp tổng số trong bài báo gốc.
- Mọi speaker tấn công đều có số VC bằng đúng số AP; đây là cấu trúc cần lưu ý, không phải bằng chứng lỗi join.
- Replay trong snapshot công khai rất nhỏ so với bài báo gốc.
- Các loại audio trong năm shard có sample rate không đồng nhất; phải resample nhất quán trước huấn luyện.
- Parquet cục bộ chỉ là một phần nhỏ và không phải mẫu ngẫu nhiên của 432 shard.
- Kiểm tra fingerprint audio chỉ bao phủ 2.558 file trong năm shard cục bộ.
- Metadata không có `generator_id`, `source_corpus`, `official_split` hoặc định danh câu nguồn.
- Không thể tự chứng minh nguyên nhân tác giả tạo số VC/AP bằng nhau chỉ từ ba cột metadata.
- Replay công khai quá nhỏ để đại diện đầy đủ cho replay trong bài báo.

## Quyết định sử dụng

Có thể tiếp tục phát triển pipeline và thử nghiệm baseline nếu áp dụng các điều kiện sau:

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
