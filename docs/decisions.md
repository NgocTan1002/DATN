# Nhật ký quyết định

Tài liệu này ghi các quyết định ảnh hưởng đến phạm vi, giao thức và cách diễn giải kết quả. Mọi thay đổi phải có ngày, lý do và ảnh hưởng.

## 2026-09-01 — D001: Chọn VSASV làm dữ liệu chính

**Quyết định:** dùng VSASV làm corpus chính trong phiên bản đầu. SEA-Spoof là phần mở rộng nếu được cấp quyền.

**Lý do:** metadata VSASV đã có cục bộ, có 220.963 mẫu và 1.141 speaker; đủ để xây dựng pipeline và các baseline trước khi phụ thuộc vào nguồn ngoài.

**Ảnh hưởng:** toàn bộ audit, split và baseline đầu tiên phải chạy hoàn chỉnh chỉ với VSASV.

## 2026-09-01 — D002: Chốt bài toán nhị phân

**Quyết định:** đầu ra chính là `bonafide` hoặc `spoof`; đồng thời báo cáo metric riêng cho `voice_conversion`, `adversarial_attack` và `replay`.

**Lý do:** accuracy tổng có thể che khuất nhóm replay rất nhỏ. Kết quả theo `utt_type` giúp phân tích lỗi, nhưng không biến bài toán thành phân loại kiểu tấn công.

**Ảnh hưởng:** sampler, metric và bảng kết quả sau này phải hỗ trợ cả nhãn nhị phân lẫn `utt_type`.

## 2026-09-01 — D003: Giới hạn tuyên bố về unseen attack

**Quyết định:** nếu chỉ dùng VSASV, dùng cụm từ “unseen spoofing attacks”; không dùng “unseen TTS engines”.

**Lý do:** metadata hiện không có TTS, `generator_id` hoặc `source_id` để chứng minh tổng quát hóa theo engine.

**Ảnh hưởng:** tên thí nghiệm, biểu đồ, báo cáo và phần bảo vệ phải tuân theo giới hạn này.

## 2026-09-01 — D004: Dữ liệu và khả năng tái lập

**Quyết định:** metadata và báo cáo audit được quản lý phiên bản; audio, dữ liệu xử lý, embedding, checkpoint và log không đưa vào Git.

**Lý do:** metadata đủ nhỏ để tái lập split, trong khi các artifact còn lại lớn và có thể tái sinh.

**Ảnh hưởng:** mỗi lần thay metadata phải chạy lại audit và tạo phiên bản split mới.

## 2026-09-01 — D005: Cổng chất lượng trước huấn luyện

**Quyết định:** chưa huấn luyện mô hình trước khi có metadata audit đạt, split speaker-disjoint cố định và leakage checker đạt.

**Lý do:** sai split hoặc rò rỉ dữ liệu làm mọi kết quả mô hình không còn giá trị.

**Ảnh hưởng:** thứ tự bắt buộc là audit metadata → tạo split → kiểm tra leakage → audio smoke test → metric → baseline.

## 2026-09-22 — D006: Khóa split protocol phiên bản 1

**Quyết định:** dùng seed `2026`, chia speaker theo từng stratum với tỷ lệ 70/15/15 và sắp xếp ổn định bằng SHA-256. Toàn bộ 19 speaker replay được giữ riêng cho open unseen replay test.

**Lý do:** phân tầng giữ được các attack hiếm, còn cách xếp dựa trên hash cho phép tái tạo đúng cùng split mà không phụ thuộc thứ tự dòng metadata hoặc phiên bản bộ sinh số ngẫu nhiên.

**Ảnh hưởng:** closed protocol phủ toàn bộ metadata. Open protocol huấn luyện bằng bonafide và voice conversion; adversarial attack và replay chỉ xuất hiện trong unseen test. Seen VC test và unseen adversarial test cố ý dùng chung bonafide pool để so sánh trên cùng negative cohort. Mọi thay đổi seed hoặc quy tắc chia phải tăng phiên bản split và chạy lại toàn bộ kiểm tra.

## Việc cần giảng viên xác nhận

- XLS-R là mô hình chính và LFCC + LCNN là mô hình cơ sở.
- Đầu ra chỉ gồm tiếng nói thật hoặc giả mạo.
- Fine-tune toàn bộ XLS-R và AASIST là phần tùy chọn.

## 2026-09-24 — D007: Khóa giao thức âm thanh phiên bản 1

**Quyết định:** mọi waveform được chuyển về mono 16 kHz. Baseline đầu tiên dùng đoạn 4 giây (64.000 mẫu), random crop khi train, center crop khi đánh giá và repeat-then-trim cho tệp ngắn. Score EER cao hơn biểu thị khả năng là spoof.

**Lý do:** smoke test trên năm shard phát hiện toàn bộ 188 mẫu VC cục bộ ở 40 kHz trong khi bonafide, AP và replay ở 16 kHz. Nếu giữ sample rate gốc, mô hình có thể học băng thông thay vì dấu hiệu giả mạo. Thời lượng trung vị của các nhóm nằm quanh 3–4 giây nên đoạn 4 giây là điểm khởi đầu hợp lý.

**Ảnh hưởng:** mọi baseline phải dùng cùng cấu hình `configs/audio.json`; native sample rate chỉ được phép xuất hiện trong ablation có nhãn rõ ràng. Báo cáo chính phải có band-limit ablation và kết quả riêng theo attack type.

## 2026-09-25 — D008: Thu hẹp đề tài vào phát hiện nhị phân

**Quyết định:** tên đề tài chính thức là “Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần”. Đồ án chỉ dự đoán `bonafide` hoặc `spoof`; không dự đoán VC, AP hay replay. Giao thức `closed_train/dev/test` trở thành giao thức chính. LFCC + LCNN là mô hình cơ sở, XLS-R đóng băng là cấu hình đối chiếu và XLS-R tinh chỉnh từng phần là mô hình chính. Tinh chỉnh toàn bộ XLS-R và AASIST là tùy chọn.

**Lý do:** giảng viên yêu cầu chọn một bài toán để nghiên cứu sâu và tránh chồng lấn với hướng nhận diện kiểu tấn công đang được thực hiện bởi nghiên cứu sinh.

**Ảnh hưởng:** các split open-set vẫn được giữ nhưng chỉ dùng cho phân tích bổ sung. Channel consistency, external dataset và nhận diện kiểu tấn công bị loại khỏi phần bắt buộc. Các bảng theo `utt_type` chỉ phục vụ phân tích lỗi của bộ phân loại nhị phân.

## 2026-09-25 — D009: Khóa thêm ba mốc triển khai

**Quyết định:** chậm nhất ngày 18/10/2026 phải khóa manifest audio thực tế, gồm danh sách shard và số mẫu của từng split. Tuần 3 chỉ triển khai các chính sách `none`, peak và RMS. Tuần 4 chạy ba pilot B0 có cùng dữ liệu, seed và số bước; chọn chính sách bằng closed development EER và khóa trước ngày 25/10/2026. Chính sách đã chọn được áp dụng giống nhau cho mọi nhãn và split. Demo dùng cửa sổ 4 giây, bước nhảy 2 giây và lấy trung bình xác suất cửa sổ làm score toàn tệp.

**Lý do:** metadata bao phủ 220.963 mẫu nhưng máy hiện chỉ có audio của 2.558 mẫu; cần tránh phát hiện thiếu dung lượng quá muộn. Gain hoặc loudness có thể trở thành shortcut giống sample rate. Demo phải xử lý tệp dài bất kỳ mà không bỏ qua phần lớn nội dung.

**Ảnh hưởng:** giao thức âm thanh tăng lên phiên bản 2. Kết quả cuối phải ghi số shard và mẫu thực tế. Mốc 18/10 chỉ khóa dữ liệu; mốc 25/10 mới khóa chính sách biên độ sau pilot B0. Chính sách biên độ không được chọn trên test. Module inference phải dùng cùng tiền xử lý và threshold đã khóa từ development.

## 2026-09-28 — D010: Xác nhận tên đề tài chính thức

**Quyết định:** chốt tên đề tài “Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần”.

**Lý do:** đây là tên đề tài đã được xác nhận để sử dụng thống nhất trong đề cương, báo cáo, slide, mã nguồn và các sản phẩm bàn giao.

**Ảnh hưởng:** không tiếp tục dùng các tên cũ liên quan đến nhận diện kiểu tấn công, channel consistency hoặc tổng quát hóa unseen attack làm tên đề tài. Mọi tài liệu mới phải dùng đúng nguyên văn tên đã chốt.

## 2026-09-28 — D011: Hoàn thành giai đoạn đề cương

**Quyết định:** đề cương đã được gửi và được đánh giá hoàn thành. Phạm vi trong đề cương trở thành cơ sở triển khai cho giai đoạn kỹ thuật.

**Lý do:** không còn cổng phê duyệt đề cương cần chờ trước khi bắt đầu xây dựng pipeline và mô hình cơ sở.

**Ảnh hưởng:** ưu tiên tiếp theo là hoàn thiện dataset loader, tiền xử lý âm thanh và một vòng huấn luyện thử LFCC + LCNN. Chỉ quay lại sửa đề cương nếu có yêu cầu mới từ giảng viên hoặc phát hiện sai lệch ảnh hưởng trực tiếp đến thực nghiệm.

## 2026-09-28 — D012: Khóa đặc tả ứng viên chính sách biên độ

**Quyết định:** khóa ba policy ứng viên `none`, `peak` và `rms_dbfs` ở vị trí sau resample 16 kHz, trước chia đoạn. `peak` dùng đích 0,95; `rms_dbfs` dùng đích -25 dBFS. Waveform dưới -50 dBFS không được khuếch đại bởi `peak` hoặc `rms_dbfs`. RMS gain được giới hạn để peak không vượt 0,95; không dùng hard clipping.

**Lý do:** biên độ có thể là shortcut giữa bonafide và spoof. Cần ba định nghĩa xác định, an toàn với đoạn gần im lặng và không gây clipping để chạy ablation công bằng.

**Ảnh hưởng:** giao thức âm thanh tăng lên phiên bản 3. Đây mới là khóa quy tắc của ba ứng viên, chưa phải chọn policy chiến thắng. Việc lựa chọn vẫn phải dựa trên ba pilot B0 có cùng dữ liệu, seed và ngân sách, dùng development EER trước ngày 25/10/2026.
