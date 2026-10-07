# Kế hoạch công việc từ Thứ Hai đến Thứ Tư — 05/10–07/10/2026

> Lập ngày 07/10/2026. Tài liệu này hệ thống lại ba ngày đầu tuần 02 và chuyển các việc chưa làm của Thứ Hai, Thứ Ba thành thứ tự xử lý trong hôm nay. Chỉ đánh dấu hoàn thành khi có đầu ra kiểm chứng trong repository hoặc log thực tế.

## 1. Mục tiêu đến cuối ngày 07/10

Hoàn thành nền tảng dữ liệu cần thiết để tạo development subset: đo được băng thông bằng một shard mới, xác minh shard đó, lập được quy tắc và danh sách ứng viên cho manifest 20.000 mẫu train + development, đồng thời chỉ ra chính xác phần audio còn thiếu phải tải.

Nếu nguồn tải hoặc ánh xạ file–shard chưa giải quyết được trong ngày, đầu ra tối thiểu là một blocker có bằng chứng, phạm vi ảnh hưởng và bước xử lý tiếp theo. Không dùng số liệu suy đoán thay cho kết quả tải thực tế.

## 2. Trạng thái khi bắt đầu hôm nay

| Hạng mục | Trạng thái | Bằng chứng hiện có |
|---|---|---|
| Hệ thống lại kế hoạch tuần 02 | Đã hoàn thành ngày 05/10 | `docs/ke_hoach_tuan_02_2026-10-05_2026-10-11.md` |
| Kiểm kê dữ liệu cục bộ | Đã hoàn thành từ 04/10 | 5/432 shard, 2.558 mẫu, 520,53 MiB; `reports/local_storage_audit.*` |
| Chọn và tải thử một shard mới | Chưa có bằng chứng hoàn thành | Chưa có `reports/download_bandwidth_trial.*`; thư mục dữ liệu vẫn có 5 shard |
| Tính ETA từ băng thông thực đo | Chưa thực hiện | Chưa có số đo tốc độ tải |
| Lập bản đồ file–shard | Chưa thực hiện | Chưa có artifact tương ứng |
| Manifest ứng viên 20.000 mẫu | Chưa thực hiện | Chưa có thư mục `data/manifests/` |
| Audit manifest | Chưa thể thực hiện | Phụ thuộc manifest ứng viên |

Các mục chưa có bằng chứng được xem là việc tồn, không được ghi là đã hoàn thành chỉ vì đã xuất hiện trong kế hoạch tuần.

## 3. Hệ thống công việc theo ngày

### Thứ Hai 05/10 — Chuẩn bị và xác định phép đo

- [x] Đọc checkpoint, kế hoạch tuần và kết quả kiểm kê storage.
- [x] Xác định mục tiêu ban đầu là 20.000 mẫu tổng train + development; mốc 40.000 chỉ được xét sau khi phương án 20.000 khả thi.
- [x] Chọn một shard chưa có cục bộ từ nguồn phát hành chính thức.
- [x] Ghi URL, tên shard, kích thước và checksum/ETag nếu nguồn cung cấp.
- [x] Tạo vị trí tải tạm `data/raw/vsasv_parquet/incoming/` và bảo đảm file tải dở không bị coi là dữ liệu hợp lệ.

**Kết quả:** toàn bộ công việc Thứ Hai đã hoàn thành và được kiểm chứng ngày 07/10; bằng chứng nằm tại `reports/download_bandwidth_trial.*`, `reports/local_storage_audit.*` và `reports/development_shard_candidates_20k_summary.*`.

### Thứ Ba 06/10 — Đo tải và chuẩn bị manifest

- [x] Tải shard thử và ghi thời điểm bắt đầu/kết thúc, số byte, thời gian monotonic và ghi chú mạng.
- [x] Tính SHA-256; đối chiếu kích thước và checksum nguồn nếu có.
- [x] Kiểm tra Parquet đọc được, đếm được hàng và khớp metadata về `file`, speaker, nhãn và `utt_type`.
- [x] Tính tốc độ tải thực và ETA cho các mốc 20.000/40.000 từ số byte trong `reports/local_storage_audit.json`.
- [x] Xác định cách ánh xạ `file` sang `shard`; lập danh sách shard ứng viên phục vụ train/development.
- [x] Chốt quy tắc lấy mẫu ổn định bằng SHA-256 của `2026|file`, giữ mỗi speaker ở đúng closed split gốc.

**Kết quả:** toàn bộ đường găng đã hoàn thành; manifest 20.000 mẫu được khóa tại D016.

### Thứ Tư 07/10 — Hoàn thành phần tồn và tạo đầu ra

Thực hiện theo bốn khối, chỉ chuyển sang khối sau khi đầu ra phụ thuộc đã có:

1. **Khối A — Nguồn và phép tải thử**
   - Hoàn thành các mục còn tồn của Thứ Hai.
   - Tải một shard mới theo giao thức trong `docs/chuan_bi_du_lieu_tuan_02_2026-10-04.md`.
   - Sinh `reports/download_bandwidth_trial.json` và `reports/download_bandwidth_trial.md`.

2. **Khối B — Xác minh và lập ngân sách**
   - Kiểm tra kích thước, SHA-256, schema, số hàng và độ khớp metadata.
   - Chỉ chuyển shard từ `incoming/` sang thư mục dữ liệu sau khi đạt cổng kiểm tra.
   - Tính ETA 20.000/40.000 từ tốc độ thực đo; ghi rõ đây là ước lượng từ một phép đo.

3. **Khối C — Ứng viên manifest 20.000 mẫu**
   - Lấy ứng viên từ `closed_train` và `closed_dev`; không dùng closed test để chọn quy mô hoặc phân bố.
   - Giữ tỷ lệ tham chiếu 15.885 train và 4.115 development nếu nguồn audio cho phép.
   - Ghi đủ `manifest_version`, `shard`, `file`, `speaker_id`, `split`, `binary_label`, `utt_type`, `native_sample_rate`, `source_snapshot`.
   - Phân biệt ba con số: mẫu mục tiêu, mẫu đã xác định được shard và mẫu đã có audio/được kiểm chứng.

4. **Khối D — Audit và bàn giao cuối ngày**
   - Kiểm tra file trùng, speaker leakage, nhãn nhị phân và độ khớp metadata.
   - Thống kê theo split, nhãn, `utt_type`, speaker và shard; ghi checksum của manifest.
   - Cập nhật `docs/current_state.md` bằng kết quả thật, blocker và việc đầu tiên ngày 08/10.
   - Kiểm tra `git diff` và `git status`; không commit audio, cache, checkpoint hay log lớn.

## 4. Thứ tự ưu tiên trong hôm nay

| Mức | Công việc | Điều kiện hoàn thành |
|---|---|---|
| P0.1 | Chọn và tải thử shard mới | Có file hoàn chỉnh hoặc blocker nguồn tải có bằng chứng |
| P0.2 | Xác minh shard và đo băng thông | Kích thước/schema/metadata đạt; có tốc độ và SHA-256 thực đo |
| P0.3 | Lập mapping file–shard | Biết shard nào chứa các file ứng viên hoặc ghi rõ phần chưa ánh xạ được |
| P0.4 | Tạo manifest ứng viên 20.000 | Tái lập được bằng seed 2026; không đổi closed split |
| P0.5 | Audit manifest | Không trùng file/speaker leakage; có bảng coverage và checksum |
| P1 | Chuẩn bị phương án tải tiếp | Có số shard, tổng byte, ETA và khả năng hoàn thành trước 18/10 |

Không bắt đầu tối ưu cache/layout trong hôm nay nếu P0.1–P0.5 chưa đạt. Điểm nghẽn hiện tại là khả năng có đủ audio và manifest hợp lệ.

## 5. Đầu ra phải có

### Mức hoàn thành đầy đủ

- `reports/download_bandwidth_trial.json` và `.md` chứa số đo tải thật.
- Một manifest ứng viên tại `data/manifests/` và báo cáo audit đi kèm tại `reports/`.
- Danh sách shard còn thiếu, tổng dung lượng cần tải và ETA dự kiến.
- Checkpoint dự án cập nhật ngày 07/10.

### Mức hoàn thành tối thiểu khi có blocker

- Báo cáo tải thử hoặc log lỗi có thời điểm, URL/shard và nguyên nhân quan sát được.
- Bản đặc tả quy tắc chọn 20.000 mẫu có thể tái lập.
- Danh sách phần đã xác minh và phần chưa xác minh; không gọi ứng viên là manifest audio sẵn sàng.
- Việc đầu tiên ngày 08/10, người chịu trách nhiệm và điều kiện để gỡ blocker.

Các đường dẫn đầu ra mới ở trên là sản phẩm dự kiến, chưa phải artifact đã tồn tại tại thời điểm lập kế hoạch.

## 6. Cổng kiểm tra trước khi kết thúc ngày

- [ ] Shard mới có đúng kích thước, SHA-256 và đọc được bằng DuckDB.
- [ ] Mọi hàng đã kiểm chứng khớp metadata; `binary_label` tuân theo `0 = bonafide`, `1 = spoof`.
- [ ] Không có file trùng hoặc speaker xuất hiện ở nhiều closed split trong manifest.
- [ ] Báo cáo ghi riêng số shard và số mẫu thực tế đã tải/đã kiểm chứng.
- [ ] ETA được tính từ băng thông thực đo, không ước lượng bằng cảm tính.
- [ ] Test set không tham gia chọn quy mô, phân bố, model, threshold hoặc amplitude policy.
- [ ] Không sửa metadata, các split đã khóa, seed `2026` hay `configs/audio.json`.

Nếu có thay đổi code, chạy unit test tương ứng và toàn bộ test khi hợp lý. Chỉ báo kiểm thử đạt khi lệnh đã thực sự chạy và quan sát kết quả. Không cần chạy lại toàn bộ test chỉ vì tạo tài liệu kế hoạch này.

## 7. Bàn giao sang ngày 08/10

Ngày 08/10 chỉ bắt đầu audit sâu và tải mở rộng khi ngày 07/10 đã có phép đo hợp lệ cùng manifest ứng viên. Nếu chưa đạt, tiếp tục xử lý blocker P0; chưa chuyển sang tối ưu DataLoader hoặc chọn amplitude policy.

Hạn giữ nguyên: khóa manifest audio trước **18/10/2026** và chọn amplitude policy bằng closed development trước **25/10/2026**.
