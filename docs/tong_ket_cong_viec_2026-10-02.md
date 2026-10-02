# Tổng kết công việc ngày 02/10/2026

## 1. Trạng thái cuối ngày

**P0, P1 và P2 đều hoàn thành.** Pipeline B0 đã có benchmark theo giai đoạn, phép so sánh DataLoader, dự báo thời gian mở rộng, quyết định cấu hình mặc định và backlog tuần 02 dựa trên số đo.

## 2. Thay đổi đã thực hiện và ý nghĩa

| Thay đổi | Đầu ra | Ý nghĩa |
|---|---|---|
| Benchmark pipeline theo giai đoạn | `scripts/benchmark_data_pipeline.py`, `reports/data_pipeline_benchmark.*` | Xác định đúng điểm nghẽn trước khi tối ưu |
| So sánh batch size và worker | `scripts/benchmark_dataloader_options.py`, `reports/data_pipeline_optimization.*` | Có bằng chứng định lượng để giữ hoặc bỏ cấu hình |
| Dự báo thời gian mở rộng | `scripts/estimate_runtime.py`, `reports/runtime_projection.*` | Chuyển throughput thực đo thành ngân sách 20.000, 40.000 và toàn bộ dữ liệu |
| Khóa quyết định DataLoader | `docs/decisions.md` — D014 | Giữ một cấu hình vận hành rõ ràng và nêu điều kiện phải đo lại |
| Chuẩn bị lịch sử Git | `docs/ke_hoach_commit_2026-10-02.md` | Tách mã mô hình, benchmark và tài liệu thành ba commit dễ xem xét |
| Lập backlog tuần 02 | `docs/ke_hoach_tuan_02_2026-10-05_2026-10-11.md` | Ưu tiên manifest, dung lượng, băng thông và coverage trước khi tải lớn |

### 2.1. Chi tiết theo tệp

| Tệp | Loại thay đổi | Nội dung chi tiết | Ý nghĩa |
|---|---|---|---|
| `scripts/benchmark_data_pipeline.py` | Tạo mới | Cố định tập mẫu và seed; đo cold/warm Dataset init, đọc/decode Parquet, tiền xử lý, DataLoader, LFCC, LCNN forward/backward và RSS; kiểm tra checksum waveform | Cho biết thời gian bị tiêu tốn ở giai đoạn nào và bảo đảm phép đo không làm đổi dữ liệu |
| `reports/data_pipeline_benchmark.json` | Tạo mới | Lưu toàn bộ số đo P0 ở dạng máy đọc được, gồm các lần lặp, p50/P95, throughput, bộ nhớ và kết quả tái lập | Có thể tổng hợp lại số liệu hoặc so sánh benchmark sau này mà không phải chép số bằng tay |
| `reports/data_pipeline_benchmark.md` | Tạo mới | Tóm tắt giao thức, bảng thời gian và kết luận điểm nghẽn Parquet/DataLoader | Giúp đọc nhanh kết quả kỹ thuật và tránh tối ưu nhầm LFCC hoặc LCNN |
| `scripts/benchmark_dataloader_options.py` | Tạo mới | Chạy A/B bốn cấu hình batch/worker trên cùng 16 mẫu và ba lần lặp; đo đủ DataLoader → LFCC → LCNN → loss → backward → optimizer | So sánh công bằng các cấu hình và kiểm tra cả tốc độ, bộ nhớ lẫn tính đúng của đầu ra |
| `reports/data_pipeline_optimization.json` | Tạo mới | Lưu số đo từng cấu hình, mức cải thiện so với chuẩn, peak RSS và quyết định theo ngưỡng 15% | Là bằng chứng có cấu trúc cho quyết định giữ cấu hình chuẩn |
| `reports/data_pipeline_optimization.md` | Tạo mới | Trình bày bảng A/B và lý do không nhận cấu hình hai worker | Làm rõ rằng tốc độ tăng 14,42% chưa bù được mức RAM tăng lên 1.831,23 MiB |
| `scripts/estimate_runtime.py` | Tạo mới | Đọc trực tiếp ba báo cáo nguồn, lấy throughput train/dev, phân bổ subset theo tỷ lệ closed train/dev và sinh dự báo có biên 20% | Dự báo có thể tái tạo và tự cập nhật khi benchmark thay đổi |
| `reports/runtime_projection.json` | Tạo mới | Lưu dự báo cho 20.000, 40.000 và toàn bộ closed protocol cùng giả định, nguồn số liệu và cấu hình mặc định | Hỗ trợ dùng số liệu trong script hoặc báo cáo tiếp theo |
| `reports/runtime_projection.md` | Tạo mới | Trình bày ngân sách thời gian train/dev/test, biên dự phòng và giới hạn suy rộng | Hỗ trợ chọn quy mô development subset phù hợp lịch và tài nguyên |
| `docs/decisions.md` | Cập nhật | Thêm D014: giữ `batch_size=8`, `num_workers=0`, đồng thời ghi điều kiện phải benchmark lại | Ngăn cấu hình tạm thời bị hiểu thành siêu tham số khoa học hoặc áp dụng máy móc trên máy khác |
| `docs/ke_hoach_ngay_2026-10-02.md` | Cập nhật | Đánh dấu hoàn thành P0/P1/P2 và ghi kết quả, số liệu, giả định, đầu ra của từng mức ưu tiên | Kế hoạch ngày trở thành nhật ký thực thi có minh chứng thay vì danh sách việc dự kiến |
| `docs/ke_hoach_tuan_02_2026-10-05_2026-10-11.md` | Tạo mới | Lập backlog theo ngày, cổng hoàn thành, quy tắc chọn 20.000/40.000 mẫu và những việc không làm | Giữ tuần 02 tập trung vào dữ liệu, dung lượng, băng thông, manifest và leakage |
| `docs/ke_hoach_commit_2026-10-02.md` | Tạo mới | Chia thay đổi thành ba commit: mô hình/kiểm thử, benchmark/dự báo, tài liệu/decision log | Giúp lịch sử Git dễ review và có thể hoàn nguyên theo từng mục đích |
| `README.md` | Cập nhật | Bổ sung cách chạy hai benchmark, script dự báo, kết quả chính và cấu hình DataLoader được giữ | Người khác có thể tìm thấy quy trình chạy lại ngay từ tài liệu đầu dự án |
| `docs/tong_ket_cong_viec_2026-10-02.md` | Tạo mới | Tổng hợp trạng thái, thay đổi, ý nghĩa, số đo, quyết định, kiểm thử và việc tiếp theo | Tạo điểm bàn giao cuối ngày duy nhất cho toàn bộ công việc 02/10 |

## 3. Kết quả P0 — Xác định điểm nghẽn

- Giao thức: 24 mẫu cố định, seed `2026`, amplitude policy `none`, ba lần lặp, batch size 8 và `num_workers=0`.
- DataLoader chiếm **95,31%** thời gian đường train.
- Đọc/decode Parquet chiếm **99,82%** thời gian đọc và tiền xử lý mỗi mẫu.
- Trung vị đọc/decode: **925,361 ms/mẫu**; P95: **1.386,687 ms/mẫu**.
- Throughput DataLoader: **1,037 mẫu/giây**.
- RSS lớn nhất quan sát: **532,41 MiB**.

Ý nghĩa: điểm nghẽn nằm ở truy vấn và giải mã waveform từ Parquet. Tối ưu LFCC hoặc LCNN lúc này không giải quyết phần lớn thời gian chờ.

## 4. Kết quả P1 — So sánh cấu hình

| Cấu hình | Throughput | So với chuẩn | Peak RSS | Quyết định |
|---|---:|---:|---:|---|
| Batch 4, worker 0 | 0,882 mẫu/giây | +1,25% | 438,23 MiB | Không giữ |
| Batch 8, worker 0 | 0,871 mẫu/giây | Mốc chuẩn | 530,64 MiB | **Giữ** |
| Batch 16, worker 0 | 0,814 mẫu/giây | -6,57% | 837,75 MiB | Không giữ |
| Batch 8, worker 2 | 0,997 mẫu/giây | +14,42% | 1.831,23 MiB | Không giữ |

Hai worker cho tốc độ cao nhất nhưng chưa vượt ngưỡng chấp nhận 15% và dùng khoảng 3,45 lần peak RSS của cấu hình chuẩn. Tất cả cấu hình đều giữ nguyên file, nhãn, thứ tự và checksum waveform; loss và gradient hữu hạn.

## 5. Kết quả P2 — Dự báo và kế hoạch

### Development subset

Phân bổ theo tỷ lệ closed train/dev hiện tại: 79,43%/20,57%.

| Quy mô | Train | Dev | 1 epoch + 1 lượt dev | Có dự phòng 20% |
|---:|---:|---:|---:|---:|
| 20.000 | 15.885 | 4.115 | 5,76 giờ | **6,91 giờ** |
| 40.000 | 31.770 | 8.230 | 11,52 giờ | **13,83 giờ** |

### Toàn bộ closed protocol

| Giai đoạn | Số mẫu | Ước lượng | Có dự phòng 20% |
|---|---:|---:|---:|
| 1 epoch train | 150.999 | 48,13 giờ | 57,76 giờ |
| 1 lượt dev | 39.116 | 6,64 giờ | 7,97 giờ |
| 1 lượt test | 30.848 | 5,24 giờ | 6,29 giờ |
| Tổng | 220.963 | 60,01 giờ | **72,01 giờ** |

Dự báo giả định thời gian tăng tuyến tính trên cùng máy và layout Parquet. Thời gian chưa gồm tải dữ liệu, tạo manifest, kiểm tra leakage, lưu checkpoint hoặc gián đoạn. Throughput development lấy từ một lần chạy 128 mẫu nên có độ chắc chắn thấp hơn benchmark train.

## 6. Quyết định cuối ngày

- Giữ `batch_size=8`, `num_workers=0` cho chặng B0 tiếp theo.
- Bắt đầu tuần 02 bằng manifest ứng viên 20.000 mẫu.
- Chỉ mở rộng 40.000 mẫu sau khi đo dung lượng, băng thông, coverage và leakage.
- Đo lại worker sau khi thay layout Parquet hoặc cache.
- Không dùng benchmark kỹ thuật này để kết luận chất lượng mô hình hoặc chọn amplitude policy.

## 7. Kiểm tra cuối ngày

- **49/49 unit test đạt** trong 4,566 giây.
- Ba script benchmark/dự báo biên dịch thành công.
- Script dự báo chạy thành công và tái tạo báo cáo JSON/Markdown.
- `git diff --check` đạt; chỉ có cảnh báo quy ước LF/CRLF của Git trên Windows.
- Checkpoint, audio thô và dữ liệu trung gian không được đưa vào danh sách commit.

## 8. Kế hoạch commit đã chuẩn bị

1. `feat(model): implement and validate LFCC-LCNN smoke baseline`
2. `perf(data): benchmark B0 loading and project runtime`
3. `docs(project): record B0 progress and plan week 2`

Chưa tự động stage hoặc tạo commit. Danh sách tệp chi tiết nằm trong `docs/ke_hoach_commit_2026-10-02.md`.

## 9. Việc đầu tiên của chặng tiếp theo

Đo dung lượng thực của 5 shard hiện có và dung lượng đĩa còn trống, sau đó tạo manifest ứng viên 20.000 mẫu có đầy đủ shard, file, speaker, split, nhãn và `utt_type`.
