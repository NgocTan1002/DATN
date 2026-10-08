# Kế hoạch công việc ngày 02/10/2026

## 1. Mục tiêu trọng tâm

Hôm nay chuyển từ trạng thái **pipeline B0 đã chạy đúng** sang trạng thái **biết rõ điểm nghẽn hiệu năng và có quyết định tối ưu dựa trên số đo**.

Ngày làm việc được xem là đạt khi có đủ bốn kết quả:

1. Có benchmark tái lập, tách được thời gian khởi tạo Dataset, đọc waveform, tiền xử lý, LFCC và LCNN.
2. Xác định bằng số liệu thành phần chiếm nhiều thời gian nhất trong smoke pilot.
3. Thử ít nhất một phương án tối ưu an toàn trên cùng dữ liệu và cùng seed; chỉ giữ thay đổi nếu nhanh hơn rõ ràng và không đổi dữ liệu đầu ra.
4. Đóng gói sạch phần B0 ngày 01/10: kiểm thử đạt, tài liệu nhất quán và các thay đổi Git được rà soát theo nhóm.

Không mở rộng development subset lên 20.000–40.000 mẫu trước khi có số đo throughput và ước lượng thời gian chạy.

## 2. Trạng thái đầu ngày lúc 10:26

### Đã hoàn thành và đã kiểm chứng

- [x] Pipeline `Parquet → waveform → LFCC → LCNN → loss → backward → optimizer` đã chạy xuyên suốt.
- [x] Smoke pilot một epoch hoàn tất 32 bước trên 256 mẫu train và 128 mẫu development.
- [x] Checkpoint B0 lưu/nạp thành công; sai khác logit sau khôi phục bằng `0`.
- [x] 49/49 kiểm thử đạt ngày 02/10/2026 trong 8,840 giây.
- [x] Tiêu chí nghiệm thu kỹ thuật tuần 01 đạt 9/9.

### Số liệu hiệu năng hiện có

| Hạng mục | Kết quả ngày 01/10 |
|---|---:|
| Train 256 mẫu, 32 bước | 259,039 giây |
| Development 128 mẫu | 78,237 giây |
| Tổng thời gian | 339,946 giây |
| Throughput gộp xấp xỉ | 1,13 mẫu/giây |
| RSS lớn nhất quan sát | 625,03 MiB |
| Peak VRAM | 0 MiB |

Số liệu hiện tại mới đo tổng thời gian. Chưa đủ bằng chứng để kết luận chậm do DuckDB/Parquet, resample, LFCC hay forward/backward.

### Trạng thái Git cần xử lý

- Nhánh `main` đang cùng mốc với `origin/main`.
- Có 5 tệp đã theo dõi đang sửa và 9 tệp chưa được theo dõi.
- Các thay đổi gồm mã LCNN, script smoke train, kiểm thử, cấu hình, báo cáo và tài liệu.
- Chưa tối ưu hoặc commit thêm trước khi rà soát để tránh trộn thay đổi hiệu năng với mốc B0 đã đạt.

## 3. Thứ tự ưu tiên

### P0 — Phải hoàn thành hôm nay

- [x] Rà soát diff B0 ngày 01/10 và xác nhận không có tệp dữ liệu/checkpoint lớn lọt vào Git.
- [x] Thiết kế benchmark dùng cùng smoke manifest, seed `2026`, amplitude policy `none` và thứ tự mẫu cố định.
- [x] Đo riêng các giai đoạn:
  - khởi tạo Dataset và tạo local index;
  - đọc/decode waveform từ Parquet;
  - resample, amplitude policy và fixed-length segmentation;
  - tạo batch bằng DataLoader;
  - LFCC;
  - LCNN forward;
  - loss, backward và optimizer step.
- [x] Ghi throughput, thời gian trung vị/P95, RSS và cấu hình chạy vào JSON/Markdown.
- [x] Xác định một điểm nghẽn chính và ghi bằng chứng định lượng.
- [x] Chạy lại toàn bộ kiểm thử sau mọi thay đổi được giữ lại.

### P1 — Nên hoàn thành hôm nay

- [x] So sánh batch size `4`, `8`, `16` trên cùng tập mẫu; không suy diễn từ một lần chạy duy nhất.
- [x] So sánh DataLoader `num_workers=0` và một cấu hình worker an toàn trên Windows nếu việc khởi tạo worker không làm sai tính tái lập.
- [x] Thử một phương án tối ưu trực tiếp vào điểm nghẽn lớn nhất, ưu tiên thay đổi nhỏ và có thể hoàn nguyên.
- [x] Chấp nhận phương án tối ưu chỉ khi:
  - waveform, nhãn và file của batch đối chứng không đổi;
  - không xuất hiện NaN/Inf;
  - 49 kiểm thử cũ và kiểm thử mới liên quan đều đạt;
  - throughput cải thiện ổn định qua ít nhất ba lần đo hoặc mức cải thiện đủ lớn để vượt nhiễu đo.
- [x] Cập nhật README và báo cáo hiệu năng bằng kết quả thực tế, kể cả khi không có phương án nào đủ tốt để giữ.

### P2 — Chỉ làm khi P0 và P1 đã đạt

- [x] Ước lượng thời gian train/dev cho subset 20.000, 40.000 và toàn bộ dữ liệu dựa trên throughput sau tối ưu; ghi rõ giả định tuyến tính.
- [x] Đề xuất cấu hình DataLoader mặc định cho giai đoạn B0 tiếp theo.
- [x] Rà soát, chia nhóm và chuẩn bị commit cho:
  1. mã mô hình và kiểm thử;
  2. script/báo cáo benchmark;
  3. tài liệu và nhật ký quyết định.
- [x] Lập backlog tuần 02 dựa trên số đo thay vì tiếp tục mở rộng theo ước lượng cảm tính.

## 4. Lịch thực hiện từ 10:30

| Thời gian | Khối công việc | Đầu ra kiểm chứng |
|---|---|---|
| 10:30–11:00 | Rà soát trạng thái B0 và Git | Danh sách thay đổi sạch; checkpoint vẫn bị loại khỏi Git |
| 11:00–12:00 | Thiết kế và cài benchmark theo từng giai đoạn | Script benchmark, schema JSON và điều kiện so sánh |
| 12:00–13:00 | Nghỉ trưa | — |
| 13:00–14:00 | Chạy baseline lặp lại | Thời gian p50/P95, throughput và RSS của cấu hình hiện tại |
| 14:00–14:15 | Nghỉ ngắn | — |
| 14:15–15:00 | Phân tích điểm nghẽn | Một kết luận chính có tỷ trọng thời gian và bằng chứng |
| 15:00–16:15 | Thử tối ưu và benchmark A/B | Bảng so sánh cùng seed/cùng dữ liệu |
| 16:15–16:30 | Nghỉ ngắn | — |
| 16:30–17:15 | Chạy regression test và kiểm tra tái lập | Toàn bộ test đạt; batch đối chứng không đổi |
| 17:15–17:45 | Cập nhật tài liệu, ước lượng mở rộng và chốt ngày | Báo cáo Markdown/JSON; backlog ngày 03/10 rõ ràng |

Nếu một benchmark đơn lẻ kéo dài quá 15 phút, giảm số mẫu đo nhưng giữ ít nhất ba lần lặp và ghi rõ phạm vi. Không dùng toàn bộ epoch để đo một thay đổi nhỏ nếu benchmark theo batch đã đủ phân biệt.

## 5. Thiết kế benchmark bắt buộc

### Tập đối chứng

- Dùng file từ `data/splits/smoke_train.csv`.
- Cố định seed `2026` và danh sách chỉ số mẫu trước khi đo.
- Dùng amplitude policy `none` để so với pilot ngày 01/10.
- Chạy warm-up trước khi ghi số liệu compute.
- Ghi riêng cold-start và warm-run; không trộn hai loại thời gian.

### Chỉ số cần ghi

| Nhóm | Chỉ số |
|---|---|
| Dataset | thời gian khởi tạo, số mẫu, local coverage |
| I/O | thời gian đọc từng mẫu/batch, p50, p95, mẫu/giây |
| Tiền xử lý | resample, amplitude, segmentation |
| Đặc trưng | thời gian LFCC mỗi batch và mỗi mẫu |
| Mô hình | forward, backward, optimizer mỗi batch |
| Tài nguyên | RSS đầu/cuối/cao nhất, số worker, batch size |
| Tái lập | file, nhãn và checksum waveform của batch đối chứng |

### Quy tắc ra quyết định

```text
Đo baseline
    ↓
Xác định giai đoạn chiếm thời gian lớn nhất
    ↓
Thay đổi đúng một yếu tố
    ↓
Đo lại trên cùng mẫu và seed
    ↓
Kiểm tra đầu ra + regression test
    ↓
Giữ thay đổi hoặc hoàn nguyên có ghi lý do
```

Không thay đồng thời batch size, worker và cách đọc Parquet trong cùng một phép so sánh vì sẽ không biết yếu tố nào tạo ra chênh lệch.

## 6. Cổng chất lượng

Benchmark chỉ được đánh dấu hoàn thành khi:

- cùng input cho cùng waveform, label và file;
- thời gian được đo bằng đồng hồ monotonic;
- có warm-up và ít nhất ba lần lặp cho phép so sánh chính;
- báo cáo ghi phần cứng, Python/PyTorch, batch size và số worker;
- JSON đủ dữ liệu để tổng hợp lại, Markdown đủ ngắn để đọc nhanh;
- không diễn giải throughput smoke như tốc độ chắc chắn trên toàn bộ 432 shard.

Tối ưu chỉ được đưa vào pipeline khi:

- giữ nguyên quy tắc mono, resample 16 kHz, amplitude và đoạn 64.000 mẫu;
- không thay split, crop seed hoặc thứ tự nhãn;
- không tăng RAM vượt mức không phù hợp với kế hoạch dataset lớn;
- cải thiện đo được lớn hơn nhiễu giữa các lần chạy;
- toàn bộ kiểm thử đạt.

## 7. Việc không làm hôm nay

- Không chạy thêm epoch để cải thiện loss.
- Không báo cáo EER, accuracy hoặc kết luận khoa học từ smoke subset.
- Không chọn amplitude policy chiến thắng.
- Không tạo development subset 20.000–40.000 mẫu trước khi có ước lượng thời gian.
- Không tải toàn bộ 432 shard.
- Không triển khai XLS-R, AASIST hoặc demo.
- Không thay kiến trúc LCNN trong khi đang đo pipeline dữ liệu.

## 8. Thứ tự cắt giảm khi thiếu thời gian

1. Bỏ ước lượng toàn bộ 432 shard, giữ ước lượng 20.000/40.000 mẫu.
2. Chỉ so sánh batch size `8` với một cấu hình thay thế tốt nhất.
3. Hoãn thử nhiều worker nếu Windows worker startup làm benchmark quá nhiễu.
4. Giữ nguyên P0: benchmark theo giai đoạn, kết luận điểm nghẽn, kiểm thử và báo cáo.

## 9. Đầu ra cuối ngày dự kiến

- `scripts/benchmark_data_pipeline.py`: benchmark có thể chạy lại.
- `reports/data_pipeline_benchmark.json`: số liệu có cấu trúc.
- `reports/data_pipeline_benchmark.md`: bảng đọc nhanh và kết luận điểm nghẽn.
- `docs/plans/daily/2026-10-02.md`: thay đổi, ý nghĩa, quyết định giữ/bỏ và việc tiếp theo.
- Cập nhật `README.md`, kế hoạch tuần hoặc decision log nếu benchmark dẫn đến thay đổi pipeline.

## 10. Mẫu chốt cuối ngày

```text
Đã hoàn thành:
Minh chứng/file đầu ra:
Baseline throughput:
Điểm nghẽn chính:
Phương án đã thử:
Mức cải thiện:
Tính tái lập:
Kiểm thử cuối ngày:
Quyết định giữ/bỏ:
Ước lượng cho 20.000/40.000 mẫu:
Việc đầu tiên ngày 03/10:
```

## 11. Kết quả P0

- **Rà soát Git:** hai checkpoint `.pt` được `.gitignore` loại đúng quy tắc; không có tệp chưa theo dõi nào từ 1 MiB trở lên; `git diff --check` không phát hiện lỗi whitespace.
- **Giao thức benchmark:** 24 mẫu cố định từ `smoke_train.csv`, seed `2026`, amplitude policy `none`, ba lần lặp, batch size 8, `num_workers=0`.
- **Tính tái lập:** file, label và SHA-256 waveform khớp giữa ba lần đo trực tiếp và DataLoader.
- **Khởi tạo Dataset:** cold-start `2.452,851 ms`; trung vị warm-start `534,238 ms`.
- **Điểm nghẽn trong một mẫu:** đọc/decode Parquet trung vị `925,361 ms`, P95 `1.386,687 ms`, chiếm `99,82%` thời gian đọc và tiền xử lý.
- **Đường train theo batch:** DataLoader trung vị `7.723,815 ms/batch`, chiếm `95,31%`; LFCC `7,129 ms`; LCNN forward `96,985 ms`; backward `265,166 ms`.
- **Throughput DataLoader:** `1,037 mẫu/giây` trên cấu hình benchmark.
- **Tài nguyên:** tổng benchmark `150,953` giây; RSS đầu `211,40 MiB`; RSS lớn nhất quan sát `532,41 MiB`.
- **Kết luận:** điểm nghẽn chính là truy vấn đọc/decode từng waveform bằng DuckDB từ Parquet. P1 nên tác động vào cách đọc dữ liệu hoặc song song hóa DataLoader; chưa nên tối ưu LFCC hay LCNN.
- **Kiểm thử cuối P0:** 49/49 kiểm thử đạt trong `5,865` giây; script benchmark biên dịch được; JSON báo cáo hợp lệ.
- **Minh chứng:** `scripts/benchmark_data_pipeline.py`, `reports/data_pipeline_benchmark.json` và `reports/data_pipeline_benchmark.md`.

## 12. Kết quả P1

- **Giao thức A/B:** 16 mẫu cố định kế thừa benchmark P0, seed `2026`, policy `none`, ba lần lặp; mỗi cấu hình chạy đủ DataLoader, LFCC, LCNN, loss, backward và optimizer.
- **Batch size 4, worker 0:** `0,882 mẫu/giây`, nhanh hơn `1,25%` so với cấu hình chuẩn; peak RSS `438,23 MiB`.
- **Batch size 8, worker 0:** `0,871 mẫu/giây`; đây là cấu hình chuẩn; peak RSS `530,64 MiB`.
- **Batch size 16, worker 0:** `0,814 mẫu/giây`, chậm hơn `6,57%`; peak RSS `837,75 MiB`.
- **Batch size 8, worker 2:** `0,997 mẫu/giây`, nhanh hơn `14,42%`; peak RSS cây tiến trình `1.831,23 MiB`.
- **Tính tái lập:** mọi cấu hình giữ nguyên file, thứ tự, label và SHA-256 waveform; loss và gradient hữu hạn.
- **Quyết định:** giữ `batch_size=8`, `num_workers=0`. Hai worker là phương án nhanh nhất nhưng không vượt ngưỡng cải thiện `15%` và dùng khoảng 3,45 lần peak RSS của cấu hình chuẩn.
- **Ý nghĩa:** batch size không giải quyết điểm nghẽn truy vấn Parquet tuần tự. Song song hóa có tiềm năng nhưng chi phí worker startup và bộ nhớ trên Windows còn quá lớn đối với smoke subset ngắn.
- **Minh chứng:** `scripts/benchmark_dataloader_options.py`, `reports/data_pipeline_optimization.json` và `reports/data_pipeline_optimization.md`.

## 13. Kết quả P2

- **Throughput dùng để dự báo:** train end-to-end `0,871 mẫu/giây`; development `1,636 mẫu/giây`.
- **Development subset 20.000 mẫu:** phân bổ 15.885 train/4.115 dev; một epoch train và một lượt dev khoảng `5,76 giờ`, hoặc `6,91 giờ` với dự phòng 20%.
- **Development subset 40.000 mẫu:** phân bổ 31.770 train/8.230 dev; một epoch train và một lượt dev khoảng `11,52 giờ`, hoặc `13,83 giờ` với dự phòng 20%.
- **Toàn bộ closed protocol:** một epoch trên 150.999 train, một lượt 39.116 dev và một lượt 30.848 test khoảng `60,01 giờ`, hoặc `72,01 giờ` với dự phòng 20%.
- **Giả định:** thời gian tăng tuyến tính trên cùng máy và layout Parquet; chưa gồm tải dữ liệu, tạo manifest, leakage check, checkpoint và gián đoạn. Throughput development mới có một lần đo nên kém chắc chắn hơn train.
- **Cấu hình mặc định:** giữ `batch_size=8`, `num_workers=0`; đo lại sau khi thay layout/cache.
- **Chuẩn bị commit:** ba nhóm độc lập gồm mô hình/kiểm thử, benchmark/dự báo và tài liệu/decision log; chưa stage hoặc tạo commit.
- **Backlog tuần 02:** bắt đầu bằng ứng viên 20.000 mẫu, đo dung lượng và băng thông, kiểm tra coverage/leakage rồi mới quyết định mở rộng 40.000 mẫu.
- **Minh chứng:** `scripts/estimate_runtime.py`, `reports/runtime_projection.json`, `reports/runtime_projection.md`, `docs/plans/daily/2026-10-02.md` và `docs/plans/weekly/2026-W41.md`.
