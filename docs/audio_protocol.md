# Giao thức âm thanh phiên bản 3

## Phạm vi

Giao thức này khóa các biến đổi âm thanh dùng chung cho mọi mô hình. Mục tiêu là hạn chế mô hình khai thác tần số lấy mẫu, độ dài hoặc biên độ tệp như shortcut thay vì học dấu hiệu giả mạo.

## Bằng chứng cục bộ

Audio smoke test trên 5/432 shard đọc được 2.558 waveform, không phát hiện waveform thiếu, rỗng, sample rate không hợp lệ hoặc giá trị không hữu hạn trong các mẫu kiểm tra. Phân bố sample rate không đồng nhất:

| Loại | Sample rate | Số mẫu cục bộ |
|---|---:|---:|
| Bonafide | 16 kHz | 1.292 |
| Voice conversion | 40 kHz | 188 |
| Adversarial attack | 16 kHz | 813 |
| Replay | 16 kHz | 265 |

Năm shard được chọn theo vị trí và không đại diện ngẫu nhiên cho 432 shard. Các con số chỉ dùng để kiểm tra pipeline.

## Tiền xử lý bắt buộc

- Chuyển waveform về một kênh.
- Resample mọi waveform về 16 kHz bằng bộ lọc band-limited sinc trước khi tạo đặc trưng.
- Giải mã PCM về `float32` trong miền `[-1, 1]` trước các biến đổi khác.
- So sánh `none`, peak normalization và RMS normalization trên closed development. Mọi cấu hình phải áp dụng giống nhau cho bonafide, spoof và mọi split.
- Peak normalization dùng đích 0,95. RMS normalization dùng đích -25 dBFS và không khuếch đại tệp có RMS đầu vào thấp hơn -50 dBFS.
- Tuần 3 chỉ triển khai ba chính sách và thống kê peak/RMS. Tuần 4 chạy ba pilot B0 với cùng train subset, development subset, seed và số bước; chọn bằng development EER và khóa trước ngày 25/10/2026.
- Chính sách đã chọn được dùng chung cho lần huấn luyện B0 đầy đủ, XLS-R, test và demo. Không dùng test để chọn.
- Kiểm tra waveform rỗng, sample rate không hợp lệ và giá trị NaN/vô cực trước khi đưa vào mô hình.

## Đặc tả ba chính sách biên độ

Đặc tả ứng viên được khóa ở phiên bản 3; chính sách chiến thắng vẫn chưa được chọn. Peak và RMS được đo trên toàn bộ waveform sau khi resample 16 kHz và trước khi cắt hoặc lặp thành đoạn 4 giây.

### Quy tắc chung

- Waveform rỗng, sample rate không hợp lệ hoặc chứa NaN/vô cực bị từ chối trước khi áp dụng policy.
- Dùng epsilon số học `1e-8` để tránh chia cho 0 và log của 0.
- Waveform có RMS thấp hơn `-50 dBFS` được đánh dấu `near_silence`. `peak` và `rms_dbfs` giữ nguyên waveform này, không khuếch đại.
- Không hard-clip hoặc clamp waveform. Nếu gain RMS có thể làm peak vượt `0,95`, giảm gain vừa đủ để peak không vượt `0,95` và đánh dấu `peak_limited`.
- Policy không được đọc hoặc phụ thuộc vào nhãn, `utt_type`, speaker hay tên split.

### `none`

- Trả nguyên waveform sau resample.
- Gain được ghi là `1,0`.
- Vẫn đo peak/RMS và áp dụng các kiểm tra đầu vào chung.

### `peak`

- Với waveform không gần im lặng: `gain = 0,95 / input_peak`.
- Nhân toàn bộ waveform với cùng một gain; không thay đổi tương quan giữa các mẫu.
- Peak đầu ra kỳ vọng là `0,95` trong sai số số học.

### `rms_dbfs`

- Tính `input_rms = sqrt(mean(waveform^2))` và `input_rms_dbfs = 20 * log10(input_rms)`.
- Với waveform không gần im lặng: `desired_gain = 10^((-25 - input_rms_dbfs) / 20)`.
- Tính `peak_safe_gain = 0,95 / input_peak` và dùng `applied_gain = min(desired_gain, peak_safe_gain)` để tránh clipping mà không cắt ngọn.
- RMS đầu ra đạt gần `-25 dBFS` khi không bị giới hạn bởi peak. Nếu bị giới hạn, báo cáo RMS thực tế và `peak_limited = true`.

### Trường audit bắt buộc

Mỗi policy phải có thể báo cáo: peak trước/sau, RMS trước/sau, RMS dBFS trước/sau, gain đã áp dụng, cờ `near_silence` và cờ `peak_limited`. Các thống kê được tổng hợp theo split, nhãn nhị phân và `utt_type`.

## Chia đoạn

- Độ dài chuẩn: 4 giây, tương đương 64.000 mẫu ở 16 kHz.
- Tập train: random crop với seed thí nghiệm được ghi lại.
- Dev/test: center crop xác định để cùng một tệp luôn cho cùng đầu vào.
- Tệp ngắn hơn 4 giây: lặp waveform rồi cắt đúng 64.000 mẫu.
- Kết quả chính dùng cấu hình trên; đánh giá full-utterance hoặc multi-crop phải được ghi thành thí nghiệm riêng.

## Kiểm soát shortcut

- Không huấn luyện bằng sample rate gốc trộn lẫn.
- Báo cáo riêng VC, AP và replay.
- Thực hiện band-limit ablation để kiểm tra mức độ mô hình dựa vào băng thông cao.
- So sánh phân bố peak/RMS theo nhãn và thực hiện amplitude ablation tối thiểu.
- Mọi thay đổi giao thức phải tăng phiên bản trong `configs/audio.json` và chạy lại smoke test.

## Suy luận tệp dài cho demo

- Tệp dài không quá 4 giây dùng cùng quy tắc evaluation.
- Tệp dài hơn 4 giây được chia thành cửa sổ 4 giây với bước nhảy 2 giây.
- Luôn thêm một cửa sổ kết thúc tại cuối tệp nếu các bước nhảy trước chưa bao phủ hết.
- Score toàn tệp là trung bình xác suất spoof của các cửa sổ.
- Demo hiển thị thêm cửa sổ có score lớn nhất và timeline, nhưng không dùng max score làm kết luận chính.
- Threshold lấy từ closed development và không thay đổi theo tệp người dùng.

## Cổng trước baseline

Baseline chỉ được chạy khi:

1. `scripts/audio_smoke_test.py` trả trạng thái ĐẠT;
2. toàn bộ unit test đạt;
3. pilot ghi rõ một trong ba policy ứng viên; baseline đầy đủ chỉ chạy sau khi policy chiến thắng được khóa bằng development EER;
4. metric EER quy ước score cao hơn tương ứng với xác suất spoof.
