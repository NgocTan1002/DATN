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

## 2026-10-01 — D013: Dùng LCNN tối thiểu có Max-Feature-Map cho cổng kỹ thuật B0

**Quyết định:** triển khai B0 smoke bằng bốn block convolution có Max-Feature-Map, ba tầng pooling, adaptive average pooling và head một logit. Cấu hình có 498.113 tham số trainable, nhận tensor `(batch, 1, 60, 401)` và dùng `BCEWithLogitsLoss`.

**Lý do:** cần một kiến trúc LCNN đủ nhỏ để kiểm tra toàn bộ forward, backward, optimizer và checkpoint trên CPU, đồng thời giữ đúng đặc trưng Max-Feature-Map của họ Light CNN. Adaptive pooling loại bỏ phụ thuộc của classifier vào kích thước không gian trung gian.

**Ảnh hưởng:** pipeline B0 đã vượt cổng kỹ thuật trên smoke subset. Kiến trúc và loss của lần chạy này là mốc triển khai có thể tái lập, chưa phải ngân sách huấn luyện hoặc kết luận khoa học cuối; mọi thay đổi kiến trúc dùng cho thí nghiệm chính phải được ghi phiên bản và so sánh công bằng.

## 2026-10-02 — D014: Giữ DataLoader batch 8, không dùng worker phụ cho chặng B0 kế tiếp

**Quyết định:** dùng `batch_size=8`, `num_workers=0` làm cấu hình kỹ thuật mặc định cho chặng B0 kế tiếp trên máy hiện tại. Đo lại quyết định này sau khi thay cách lưu/đọc Parquet hoặc bổ sung cache.

**Lý do:** trên cùng 16 mẫu, seed `2026` và ba lần lặp, cấu hình hai worker nhanh hơn 14,42% nhưng chưa vượt ngưỡng chấp nhận 15%. Peak RSS cây tiến trình tăng từ 530,64 MiB lên 1.831,23 MiB, trong khi batch size 4 chỉ nhanh hơn 1,25% và batch size 16 chậm hơn 6,57%.

**Ảnh hưởng:** dự báo thời gian 20.000, 40.000 và toàn bộ closed protocol dùng throughput end-to-end 0,871 mẫu/giây của cấu hình này. Đây là quyết định vận hành theo máy và layout dữ liệu hiện tại, không phải siêu tham số khoa học hoặc kết luận có thể suy rộng sang máy khác.

## 2026-10-02 — D015: Ghi nhận trạng thái bản phát hành công khai VSASV trên Hugging Face

**Quyết định:** ghi nhận tác giả xác nhận phiên bản trên Hugging Face là phiên bản chính thức được phát hành duy nhất và phần dữ liệu đã mất không thể khôi phục. Nguồn xác nhận: email từ tác giả Vũ Hoàng, gửi lúc 16:38 Thứ Ba ngày 22/09/2026 theo thời gian hiển thị trong ảnh. Dữ liệu sử dụng trong đồ án vẫn được định danh là `VSASV-HF-public-snapshot-v1`; không tuyên bố tái lập số liệu của bài báo gốc.

**Lý do:** xác nhận của tác giả làm rõ trạng thái phát hành hiện tại nhưng không xóa bỏ chênh lệch giữa số mẫu quan sát được trong snapshot và số liệu của bài báo. Định danh riêng giúp giới hạn đúng phạm vi dữ liệu thực sự được sử dụng và tránh so sánh metric trực tiếp không tương đương.

**Ảnh hưởng:** mọi báo cáo phải ghi số shard và số mẫu thực tế, đồng thời nêu rõ kết quả chỉ áp dụng cho `VSASV-HF-public-snapshot-v1`. Cần cân nhắc cập nhật trạng thái `provisional_public_snapshot` trong báo cáo xác minh, nhưng chưa sửa báo cáo; việc thay đổi trạng thái này cần người dùng quyết định.

## 2026-10-07 — D016: Khóa development manifest 20.000 mẫu phiên bản 1

**Quyết định:** dùng `data/manifests/development_20k_v1.csv` làm development subset phiên bản 1, gồm 15.885 mẫu `closed_train` và 4.115 mẫu `closed_dev`. Quota giữa hai split và giữa bốn `utt_type` được phân bổ theo phân bố đầy đủ của closed train/dev bằng phương pháp phần dư lớn nhất; trong mỗi stratum, mẫu được xếp ổn định theo SHA-256 của `2026|file`. Manifest có SHA-256 `63df695e232a424c95ebd869f1d71cae715040ec30428177967cac26c6c5e673`.

**Lý do:** 68 shard cục bộ đã cung cấp 34.782 audio khớp metadata, đủ mọi quota của mục tiêu 20.000 mẫu. Quy tắc này giữ nguyên split speaker-disjoint, giữ phân bố tham chiếu của dữ liệu phát hành và tái lập được mà không dùng `closed_test` để chọn quy mô hoặc phân bố.

**Ảnh hưởng:** các pilot phát triển kế tiếp dùng đúng 20.000 hàng của manifest này, gồm 7.134 bonafide, 4.348 voice conversion, 4.348 adversarial attack và 55 replay trong train; 1.524 bonafide, 1.289 voice conversion, 1.289 adversarial attack và 13 replay trong development. Manifest dùng 66 shard và 184 speaker, không có file lặp hoặc speaker giao giữa hai split. Mọi thay đổi quy mô, quota, seed hoặc quy tắc chọn phải tạo phiên bản manifest mới và ghi quyết định mới; test vẫn chỉ dùng cho đánh giá sau khi khóa mô hình và threshold.

## 2026-10-09 — D017: Khử trùng nội dung và tạo development manifest phiên bản 2

**Quyết định:** tạo `development-20k-v2` từ cùng closed train/development, seed `2026` và quota 15.885/4.115 của D016. Trước khi phân bổ quota, băm toàn bộ waveform trong pool audio cục bộ bằng SHA-256 của sample rate mã hóa signed int64 little-endian nối với toàn bộ mẫu `float64` little-endian. Trong mỗi nhóm content hash trùng, chỉ giữ file có khóa `SHA256("2026|file")` nhỏ nhất; loại các file còn lại khỏi pool. Khi chọn và bù quota, content hash được quản lý toàn cục trên train + development nên một nội dung không thể được chọn lần thứ hai. Nếu một stratum không đủ mẫu sau khử trùng thì dừng và báo thiếu, không đổi quota. Manifest v1 được giữ nguyên để truy vết nhưng không dùng cho kết quả khoa học.

**Lý do:** kiểm tra ngày 08/10 trên 34.782 waveform cục bộ phát hiện 91 nhóm/182 file bonafide trùng, trong đó manifest v1 giữ nhiều file của 38 nhóm và có 17 nhóm đi qua train/development. File path và speaker vẫn disjoint nhưng nội dung âm thanh trùng làm rò rỉ trực tiếp giữa tập huấn luyện và phát triển. Quy tắc đại diện theo hash cho kết quả xác định mà không ưu tiên thủ công train hay development.

**Kiểm chứng bắt buộc:** sau khi sinh v2, phải đọc và băm lại độc lập đúng 20.000 waveform của manifest, lưu hash từng dòng trong báo cáo JSON và xác nhận số nhóm hash lặp bằng 0 trên toàn manifest, gồm cả trùng trong cùng split và trùng đi qua train/development. Báo cáo phải ghi riêng số file bị loại và số file bù của từng split. Leakage checker theo speaker/file không thay thế cổng content hash này.

**Giới hạn và ảnh hưởng:** v2 xử lý trùng nội dung nhưng không chứng minh hoặc sửa quan hệ giữa hai speaker ID khi cùng waveform được gán cho hai speaker. Snapshot nguồn vẫn được ghi nhận có duplicate dù manifest v2 có thể đạt cổng khoa học. Trong pool cục bộ còn 4 nhóm đã xác nhận đi qua `closed_train` và `closed_test`, không có nhóm `closed_dev`–`closed_test`; D017 không thay đổi test và yêu cầu một quyết định riêng trước đánh giá cuối. Metadata, các split closed/open, seed `2026`, quota và `configs/audio.json` không thay đổi.

## 2026-10-12 — D018: Chọn Kaggle làm môi trường GPU chính

**Quyết định:** dùng Kaggle làm môi trường GPU chính cho các thí nghiệm cần tăng tốc, gồm ba pilot B0 khi pipeline dữ liệu đã đạt benchmark, XLS-R đóng băng (X0) và XLS-R fine-tune từng phần (X1). Máy cá nhân tiếp tục dùng cho phát triển code, unit test, smoke test nhỏ và kiểm tra artifact. Google Colab trả phí là phương án dự phòng về sau khi Kaggle không đáp ứng được thời lượng phiên, hạn mức, bộ nhớ hoặc khả năng cấp GPU; trước khi phát sinh chi phí phải có xác nhận riêng của người dùng.

**Lý do:** dự án không có GPU cục bộ; các phép đo CPU hiện tại cho thấy thời gian chạy dài, còn X0/X1 cần GPU để khả thi. Kaggle phù hợp làm điểm chạy chính có thể tái lập, trong khi Colab trả phí giữ vai trò mở rộng năng lực thay vì trở thành phụ thuộc bắt buộc ngay từ đầu.

**Quy tắc vận hành:** code huấn luyện và đánh giá phải nằm trong script của repository; notebook chỉ thiết lập môi trường, gắn dữ liệu và gọi script. Dữ liệu Kaggle phải để riêng tư và được định danh theo phiên bản; audio, embedding, checkpoint và log lớn không đưa vào Git. Không cài nguyên `requirements-lock.txt` hiện tại lên runtime GPU vì file này khóa PyTorch CPU; trước thí nghiệm khoa học phải ghi lại stack CUDA/PyTorch thực dùng. Mỗi run phải lưu commit, manifest/checksum, seed, cấu hình, loại GPU, thời gian, checkpoint và metric development đủ để tiếp tục sau khi phiên bị ngắt. Không dùng `closed_test` để chọn môi trường, tối ưu loader, policy, threshold, checkpoint hoặc mô hình.

**Ảnh hưởng:** D018 chỉ khóa môi trường tính toán, không thay đổi metadata, split, seed `2026`, manifest v2, cấu hình âm thanh, ngân sách pilot hay tiêu chí chọn bằng development EER. D014 vẫn là quyết định cho máy CPU và layout dữ liệu cục bộ; cấu hình DataLoader trên Kaggle phải được benchmark và ghi riêng trước khi sử dụng cho run dài. Việc mua hoặc kích hoạt Colab trả phí chưa được thực hiện trong quyết định này.

## 2026-10-12 — D019: Khóa quy tắc pilot B0, threshold và early stopping

**Quyết định:** ba pilot B0 cho `none`, `peak` và `rms_dbfs` phải dùng cùng `development-20k-v2`, seed `2026`, batch size, sampler, optimizer, learning rate và số optimizer step; chỉ amplitude policy được thay đổi. Ngân sách bước chưa được ấn định trước khi có smoke/benchmark Kaggle: chọn ngân sách chung lớn nhất có thể hoàn thành cả ba run trước ngày 25/10/2026 với thời gian dự phòng, và mỗi run đánh giá toàn bộ closed development. Không dùng early stopping trong ba pilot. Policy được chọn theo development EER thấp nhất; nếu EER bằng nhau về mặt số học thì dùng development loss thấp hơn, nếu vẫn bằng nhau thì chọn `none`.

**Quy tắc threshold và full training:** threshold nhị phân là threshold tại điểm EER trên `closed_dev`. Threshold này được đóng băng để tính macro-F1 trên test và dùng cho inference; EER và ROC-AUC tiếp tục được tính trực tiếp từ score. Early stopping chỉ được dùng cho lần huấn luyện B0 đầy đủ sau khi amplitude policy đã khóa, với checkpoint tốt nhất được chọn bằng development EER. Không dùng closed test để chọn threshold, policy, ngân sách, checkpoint hoặc thời điểm dừng.

**Lý do:** ngân sách cố định giữ so sánh ba policy công bằng, trong khi throughput Kaggle chưa được đo nên chưa thể khóa an toàn số epoch hoặc số bước. Tách pilot khỏi early stopping tránh mỗi policy nhận lượng tối ưu khác nhau. Dùng threshold tại development EER thống nhất với EER là metric chính và cung cấp một quy tắc cố định cho macro-F1 cùng demo.

**Ảnh hưởng:** trước ba pilot phải có training/evaluation loop thu đủ score/label, EER, ROC-AUC, macro-F1, development loss và checkpoint có thể tiếp tục. D019 không thay đổi manifest, split, seed, kiến trúc B0 hoặc ba công thức amplitude trong `configs/audio.json`; policy chiến thắng vẫn phải được ghi bằng một quyết định riêng sau khi có kết quả thực chạy.

## 2026-10-20 — D020: Khóa ngân sách vận hành cho ba pilot B0

**Quyết định:** ba pilot `none`, `peak` và `rms_dbfs` cùng dùng `batch_size=8`, `num_workers=0`, optimizer Adam, learning rate `0.001` và `step_budget=896`. Mỗi run dùng toàn bộ 15.885 mẫu train làm nguồn batch, đánh giá đủ 4.115 mẫu development sau khi hoàn thành đúng 896 optimizer step và không early stopping. Ba profile Kaggle chỉ được khác amplitude policy cùng tên checkpoint/report.

**Bằng chứng:** Kaggle kernel riêng tư version 8 chạy ba process độc lập trên Tesla T4 với source bundle SHA-256 `1624687b9f19fae76f0d829e2161f3caa13da11391ab81f7fe76ac08539d0f9e`. Mỗi repeat dùng cùng technical subset xác định gồm 256 train cân bằng 128/128, 90 speaker, 55 shard và 128 development cân bằng 64/64, 25 speaker, 29 shard; selection SHA-256 là `384952c550fe5df29589f5e1d40bb2be5d7a74e1075d5df97f272b17570c09a6`. Cả ba repeat hoàn thành 32/32 step với cùng identity. Thời gian xấu nhất là 12,123399 giây mỗi optimizer step và 1,546248 giây mỗi development sample; CV tương ứng là 7,20% và 3,11%.

**Quy tắc ngân sách:** dành tối đa 20 GPU-giờ cho bốn run-equivalent gồm ba pilot và một lượt dự phòng. Dùng thời gian lớn nhất trong ba repeat, chiếu đủ 4.115 development sample, rồi làm tròn số step xuống bội số 64. Kết quả là 896 step, dự kiến 4,785 giờ mỗi run và 19,139 GPU-giờ cho bốn run-equivalent, còn 0,861 giờ biên dự phòng.

**Ảnh hưởng và giới hạn:** D020 khóa cấu hình vận hành để so sánh công bằng ba policy; không chọn amplitude policy và không phải kết quả khoa học. Benchmark không dùng `closed_test`, không thay metadata, split, seed, manifest hoặc `configs/audio.json`. Báo cáo gốc và phép tính nằm tại `reports/b0_benchmark_repeat_{1,2,3}.*` và `reports/b0_benchmark_summary.*`.

## 2026-10-22 — D021: Chuẩn bị kỹ thuật XLS-R trong lúc chờ pilot B0

**Quyết định:** theo yêu cầu của người dùng, triển khai khung X0/X1 và kiểm thử CPU song song với thời gian chờ pilot. Dùng giao diện Wav2Vec2Model → mean/std pooling → classifier một logit. X0 chỉ học classifier; X1 chỉ mở các block Transformer cuối, giữ feature encoder và các phần ngoài block đóng băng. Kiểm chứng bằng backbone thu nhỏ khởi tạo ngẫu nhiên và waveform tổng hợp; không dùng test, pretrained checkpoint hoặc quota GPU. Transformers `4.57.6` là dependency tùy chọn cho cổng kỹ thuật CPU, chưa xác nhận stack XLS-R CUDA.

**Lý do:** phát hiện sớm lỗi shape, gradient, freeze/unfreeze và lưu/nạp trọng số mà không phải đợi chọn amplitude policy. Tách dependency và entrypoint để các pilot B0 tiếp tục dùng source và cấu hình đã khóa.

**Giới hạn:** mean/std pooling dùng phương sai tổng thể cộng epsilon `1e-8` trước căn bậc hai. Phần đóng băng giữ eval ngay cả khi huấn luyện head; dropout chỉ bật trong block mở/head, không bật LayerDrop hoặc SpecAugment của backbone. Đây là thiết kế kỹ thuật ban đầu; chưa khóa checkpoint XLS-R, số tầng mở, head, learning rate hoặc ngân sách thí nghiệm chính. Không tự zero-mean/unit-variance normalize waveform vì sẽ thêm biến đổi ngoài giao thức biên độ hiện hành. Khác biệt với processor pretrained phải được giải quyết và ghi rõ trước run khoa học; nếu cần đổi cấu hình âm thanh thì xin xác nhận theo quy định hiện hành.

**Ảnh hưởng:** không đổi D019/D020, metadata, split, seed `2026`, manifest hoặc `configs/audio.json`; không publish lại source Kaggle. Smoke chỉ chứng minh luồng tính toán và khôi phục `state_dict`, chưa chứng minh exact resume huấn luyện hoặc hiệu năng pretrained. Hướng dẫn tại `docs/guides/xlsr_technical_smoke.md`, bằng chứng tại `reports/xlsr_cpu_smoke.json`.

## 2026-10-23 — D022: Chọn amplitude policy peak theo ba pilot B0

**Quyết định:** chọn `peak` cho chặng B0 đầy đủ kế tiếp theo tiêu chí development EER của D019. Ba pilot `none` (kernel v9), `peak` (v10) và `rms_dbfs` (v12) đều hoàn thành đúng D020: nguồn train 15.885 mẫu, đủ 4.115 development, manifest 20.000 mẫu/66 shard, seed `2026`, batch 8, worker 0, Adam, learning rate `0.001`, 896 optimizer step, không early stopping và không dùng test. Cả ba dùng source SHA-256 `12d91121abac99367724d63f336951a36cd2a47075e79ed4c6d670b3c3d060db`.

**Bằng chứng:** development EER lần lượt `0,10039370078740158`, `0,09120734908136482`, `0,11548556430446194` cho `none`, `peak`, `rms_dbfs`. `peak` có EER thấp nhất nên không cần tie-break; loss của `peak` là `0,32683539048814975`. Báo cáo tổng hợp và kiểm chứng tại `reports/b0_pilot_amplitude_comparison.{json,md}`; ba report gốc cùng workflow/runtime được giữ trong `reports/`. Metric được tính lại từ score, metadata development khớp manifest và checkpoint/checksum/identity/progress/sampler/optimizer/RNG đều đạt kiểm chứng. Version 11 lỗi mount source trước runner; version 12 chạy lại cùng source/cấu hình và hoàn thành, không resume.

**Giới hạn:** mỗi pilot có 7.168 lượt mẫu train, chưa đủ một epoch và mới một seed; không tuyên bố ý nghĩa thống kê hoặc hiệu năng baseline cuối. `none` có ROC-AUC cao hơn `peak`, nên không khẳng định `peak` tốt hơn mọi metric. Replay development chỉ có 13 mẫu và rủi ro shortcut sample rate/utt_type vẫn tồn tại. Kết quả áp dụng `VSASV-HF-public-snapshot-v1`, không so trực tiếp với protocol bài báo gốc.

**Ảnh hưởng:** D022 chọn một ứng viên đã định nghĩa ở D012; không sửa `configs/audio.json`, metadata, split, seed, manifest hoặc công thức policy. Các run mới thuộc chặng B0 đầy đủ phải truyền `--amplitude-policy peak` rõ ràng và có ngân sách/early stopping riêng được đặc tả trước khi chạy. Không sửa hoặc chạy lại ba pilot để tối ưu kết quả. Threshold `0,2584670518678943` của pilot `peak` chỉ gắn với checkpoint pilot tương ứng; full training phải chọn checkpoint bằng development EER, xác định threshold trên development của chính checkpoint đó rồi đóng băng trước test/inference. D022 chưa khóa normalization hoặc cấu hình khoa học XLS-R và không cho phép tự tạo packed artifact/evaluation manifest đang chờ xác nhận.

## 2026-10-23 — D023: Khóa đặc tả B0 đầy đủ có ngân sách giới hạn trên manifest 20k

**Quyết định:** khóa lượt `b0-peak-development-20k-v2-full-v1` trong `configs/b0_full_training.json`: khởi tạo mới seed 2026, giữ LFCC/LCNN/audio của pilot, policy peak theo D022; toàn bộ 15.885 train/4.115 development của manifest 20.000 mẫu/66 shard, đường Parquet (pool cục bộ 68 shard). Adam, learning rate cố định 0,001, batch 8, worker 0, float32, không scheduler/AMP/clipping/class weighting. `drop_last=false`, 1.986 bước/epoch, tối đa 3 epoch/5.958 bước/47.655 lượt mẫu train. Không warm-start hoặc resume từ checkpoint pilot.

**Chọn checkpoint và dừng:** full development sau mỗi epoch hoàn chỉnh, trước epoch kế tiếp. Cải thiện khi EER giảm nghiêm ngặt (min_delta 0); hòa giữ best sớm nhất, không tie-break bằng loss. Patience 1 lần đánh giá không cải thiện; lần đầu tạo best, dừng sớm nhất sau epoch 2, trần epoch 3. Chỉ xuất threshold development của chính checkpoint best; threshold pilot không áp dụng cho trọng số mới. Đây là lượt B0 có ngân sách giới hạn, không tuyên bố ngân sách tối ưu, hội tụ hoặc dùng toàn bộ closed train của metadata.

**Lý do và tài nguyên:** giữ cấu hình đã kiểm chứng để cô lập việc huấn luyện thêm sau chọn policy. Dự toán dùng maximum seconds/step và seconds/development sample trên benchmark khóa và ba pilot thực chạy. Báo cáo `reports/b0_full_training_budget.{json,md}` tính 8,764 giờ/epoch, tối đa 26,291 giờ; cộng 20% dự phòng là 31,550 giờ. Trần 3 epoch là lựa chọn vận hành để giới hạn chi phí Parquet; patience 1 có thể dừng sớm và cần được báo đúng như một giới hạn thiết kế, không suy diễn là được dữ liệu xác nhận tối ưu. Đây là phép chiếu, không phải thời gian thực chạy hoặc cam kết đủ quota trong một tuần.

**Chia phiên và resume:** tối đa 512 optimizer step/invocation; checkpoint sau mỗi 64 bước hoàn thành, biên epoch và điểm chủ động dừng. Phải giữ controller early stopping/history/best/latest/RNG và trạng thái evaluation pending trong transaction generation, commit pointer nguyên tử. Checkpoint full training dự kiến version 4; không làm thay version 3 hoặc source/identity của ba pilot. Quota thiếu thì chờ rồi resume nguyên identity; không reset patience hoặc giảm trần giữa run. Không tự dùng tài nguyên trả phí/monitor tự động.

**Cổng và ảnh hưởng:** đây là đặc tả, `ready_for_training=false`. Runner hiện chưa đáp ứng lịch đánh giá, early stopping và transaction best/latest; cần triển khai cùng kiểm thử fault/resume CPU và smoke CUDA trước run dài. Hướng dẫn chi tiết `docs/guides/b0_full_training.md`, script dự toán `scripts/estimate_b0_full_training_budget.py`. D023 không sửa metadata/split/seed/manifest/configs/audio.json, không dùng test và không cho phép packed artifact hoặc evaluation manifest đang chờ xác nhận. Không publish source hoặc gửi kernel mới trong phiên đặc tả.

**Kiểm chứng triển khai D023 ngày 24/10:** runner riêng/checkpoint v4 đã qua 10 test mới, toàn suite 172/172, real-audio CPU và CUDA v13 continuous/partial/resume khớp model/optimizer/sampler/RNG/controller/best; đối chiếu lại checkpoint tải về cũng đạt. Readiness đã mở sau các cổng. Người dùng đã xác nhận cụ thể upload source mới, cập nhật readiness và gửi phiên đầu nếu đạt; private kernel v14 b0-full-first-session đã gửi, quan sát RUNNING. Không đổi ngân sách/quy tắc khoa học của D023 hoặc báo phiên đang chạy là kết quả hoàn thành.

## 2026-10-30 — D024: So sánh ghép cặp normalization waveform cho XLS-R bằng development

**Quyết định:** theo xác nhận của người dùng, chuẩn bị một pilot X0 ghép cặp để chọn đúng một chế độ normalization waveform trước thí nghiệm XLS-R chính. Cả hai nhánh dùng cùng `development-20k-v2`, đúng 15.885 train/4.115 development, 66 shard được tham chiếu từ pool cục bộ 68 shard, seed `2026`, cùng file/thứ tự sampler, cùng khởi tạo classifier, cùng checkpoint pretrained bất biến và cùng endpoint huấn luyện. Không dùng `closed_test` để chọn normalization, checkpoint, threshold hoặc ngân sách.

**Hai nhánh:** pipeline chung vẫn resample mono 16 kHz, áp dụng policy `peak` theo D022 rồi tạo đoạn 4 giây/64.000 mẫu. Nhánh `pipeline_only` truyền nguyên đoạn này vào encoder. Nhánh `processor_zero_mean_unit_variance` chuẩn hóa riêng từng đoạn sau bước cố định độ dài theo `(x - mean) / sqrt(population_variance + 1e-7)`, tương ứng hành vi processor của ứng viên đã ghim. Phép biến đổi mới là lựa chọn riêng của XLS-R trong pilot, không sửa `configs/audio.json`, công thức `peak`, dữ liệu nguồn hoặc kết quả B0.

**Mô hình và tiêu chí:** dùng X0, đóng băng toàn backbone để cô lập normalization và giảm chi phí; mean/std pooling, head 128 chiều, ReLU, dropout 0,2 và một spoof logit giống nhau giữa hai nhánh. Adam chỉ học classifier với learning rate `0,001`; đây là hằng số pilot, không tuyên bố tối ưu cho X0/X1 cuối. Chọn EER development thấp hơn tại cùng endpoint; nếu EER bằng nhau chính xác thì giữ `pipeline_only` vì ít biến đổi hơn. Mỗi nhánh tự xác định threshold trên development của chính nó; không dùng lại threshold giữa hai nhánh hoặc từ B0.

**Cổng trước khi chạy:** `configs/xlsr_normalization_pilot.json` phải giữ `ready_for_gpu_smoke=false` và `ready_for_pilot_training=false` cho đến khi (1) tải/băm đúng trọng số ở revision `1a640f32ac3e39899438a2931f9924c02f080a54`, (2) load offline đạt, (3) smoke CUDA đo VRAM/runtime, (4) khóa một batch size và một step budget chung từ bằng chứng đó, và (5) entrypoint/controller/history/best tích hợp exact-resume đã kiểm thử. Không ngoại suy ngân sách B0 sang XLS-R và không chạy hai nhánh với ngân sách khác nhau.

**Giới hạn và ảnh hưởng:** D024 chỉ khóa thiết kế so sánh và thứ tự preprocessing; chưa chọn normalization thắng, chưa tải trọng số, chưa cho phép upload/gửi Kaggle XLS-R và chưa tạo metric khoa học. Kết quả pilot sau này chỉ là lựa chọn vận hành bằng một development subset/một seed, không chứng minh ý nghĩa thống kê. D024 không đổi metadata, split, manifest, seed, `configs/audio.json`, D022/D023, packed artifact hoặc evaluation manifest đang chờ xác nhận.

## 2026-11-02 — D025: Mở rộng và khóa tập đánh giá (dev-full, test-full) trước khi có metric mới

**Quyết định:**

1. **Train giữ nguyên.** Mọi mô hình (B0, X0, X1) huấn luyện trên đúng 15.885 mẫu `closed_train` của `development-20k-v2`, seed `2026`, policy `peak` theo D022.
2. **Dev-full thay dev cũ cho việc chọn mô hình.** Từ nay, chọn checkpoint, early stopping của mô hình mới, threshold và pilot XLS-R dùng toàn bộ `closed_dev` (39.116 dòng gốc, 172 speaker) sau khử trùng nội dung. Dev 4.115 mẫu của manifest v2 chỉ còn là lịch sử.
3. **Test-full.** Đánh giá cuối dùng toàn bộ `closed_test` (30.848 dòng gốc, 171 speaker) sau khử trùng. Test chỉ được đọc qua entrypoint đánh giá có cổng khóa (nhiệm vụ T10), mỗi mô hình đúng một lần.
4. **Quy tắc khử trùng.** Dùng đúng hàm băm của D017 (sample rate mã hóa signed int64 little-endian nối với toàn bộ mẫu `float64`), theo thứ tự ưu tiên giữ mẫu: train manifest luôn được giữ; dev-full loại mọi mẫu có hash trùng train manifest, trong nội bộ dev mỗi nhóm trùng chỉ giữ file có `SHA256("2026|file")` nhỏ nhất; test-full loại mọi mẫu có hash trùng train manifest hoặc dev-full, trong nội bộ test áp cùng quy tắc đại diện. Không bù mẫu. Split gốc, nhãn, speaker và metadata không đổi. Trùng với train được tính theo train manifest 15.885 mẫu, không theo toàn bộ `closed_train`, vì chỉ phần này được dùng để huấn luyện.
5. **Điều kiện dừng khai báo trước.** Nếu một cặp (split, `utt_type`) mất hơn 5% số dòng gốc, hoặc một nhóm hash trộn hai nhãn nhị phân, thì dừng và báo cáo; không tự điều chỉnh quy tắc hay quota, cần quyết định mới.
6. **Xử lý B0.** Huấn luyện theo D023 giữ nguyên, không train lại. Sau khi chuỗi D023 dừng hợp lệ, chấm các checkpoint cuối epoch đã lưu (tối đa 3) trên dev-full và chọn lại bằng quy tắc của D023: EER giảm nghiêm ngặt, hòa giữ checkpoint sớm nhất. Threshold lấy tại EER dev-full của checkpoint được chọn.
7. **Phạm vi hiệu lực.** D019 (nghĩa của "development"), D023 (chọn best cho registry) và D024 (pilot normalization dùng dev-full) được sửa theo D025. D022 (chọn `peak`) giữ nguyên, không chạy lại hoặc chấm lại ba pilot.
8. **Không thay đổi:** metadata, `data/splits/*`, seed `2026`, `configs/audio.json`, manifest v2 và ba công thức amplitude.

**Lý do:** phân tích ngày 02/11 cho thấy dev cũ chỉ có 31 speaker, trong đó VC đến từ 3 speaker; ở epoch 2, 192/214 lỗi VC đến từ một speaker (`id01070`) và CI bootstrap theo speaker của dev EER rộng (khoảng 2–16%). Test cục bộ chỉ phủ 4.967/30.848 mẫu với 4 speaker VC/AP. Dev-full có 22 speaker VC/AP và test-full có 21; độ phủ speaker này quyết định có kết luận so sánh được giữa B0, X0 và X1 hay không.

**Số liệu tham chiếu** (từ `reports/vsasv_snapshot_verification.json`, trước khử trùng):

| Split | Bonafide | Voice conversion | Adversarial attack | Replay | Speaker |
|---|---:|---:|---:|---:|---:|
| `closed_dev` | 14.490 | 12.253 | 12.253 | 120 | 172 |
| `closed_test` | 16.006 | 7.361 | 7.361 | 120 | 171 |

**Giới hạn và rủi ro:** (a) đổi dev sau khi B0 đã chạy có thể bị xem là chọn hậu nghiệm; giảm thiểu bằng cách ghi D025 trước khi chấm dev-full lần đầu và báo cáo cả kết quả dev cũ. (b) Early stopping của B0 dùng dev cũ, vốn là tập con của dev-full, nên có thể hơi thiên lợi cho B0 trên dev-full; dev-full không phải test nhưng phải nêu trong báo cáo. (c) Kết luận chỉ áp dụng cho mô hình huấn luyện trên 15.885 mẫu (khoảng 10,5% `closed_train`) và `VSASV-HF-public-snapshot-v1`. (d) Replay chỉ có 120 mẫu/3 speaker mỗi split, báo cáo riêng và không kết luận. (e) D025 không giải quyết quan hệ giữa hai speaker ID khi cùng waveform, như D017 đã ghi.

**Điều kiện kích hoạt và ảnh hưởng:** D025 chỉ khóa giao thức. Nó không cho phép tải shard, sinh packed artifact hoặc upload lên Kaggle; mỗi việc cần xác nhận riêng. Bốn nhóm trùng train–test đã biết được xử lý bằng quy tắc ở mục 4. Yêu cầu xác nhận evaluation manifest content-disjoint trước đây trong `docs/current_state.md` được xem là đã có quyết định. Không đổi dữ liệu, split, seed hoặc cấu hình âm thanh.

## 2026-11-02 — D026: Thay pilot normalization D024, phương án X1 theo bậc và mốc kết quả mô hình

**Quyết định:**

1. **Normalization XLS-R.** Mọi run X0/X1 dùng `processor_zero_mean_unit_variance` đã có trong `src/data/xlsr.py`: chuẩn hóa từng đoạn sau resample 16 kHz, policy `peak` và tạo đoạn 4 giây, epsilon `1e-7`. Không chạy pilot hai nhánh của D024. D024 và `configs/xlsr_normalization_pilot.json` giữ nguyên làm lịch sử; phần "chọn nhánh bằng pilot" của D024 bị thay thế.
2. **X1 theo trục số tầng Transformer mở.** Chọn cấu hình bằng EER trên dev-full (D025); nếu hòa, giữ cấu hình mở ít tầng hơn. Learning rate khởi đầu và giá trị cụ thể của các mức số tầng được khai báo ở D027 trước khi chạy.
3. **Ba bậc phương án:** A — 3 mức số tầng, kiểm tra 2 learning rate ở mức đã chọn, 3 seed cho X1 cuối, B0 và X0. B (mục tiêu) — 2 mức số tầng, 3 seed cho X1 cuối, B0 và X0. C (sàn dự phòng) — 2 mức số tầng, một seed.
4. **Chọn bậc.** Sau smoke CUDA (T14), tính chi phí từ số đo thật (giờ mỗi epoch, giờ mỗi lượt dev-full) theo số epoch tối đa, không giả định dừng sớm. Chọn bậc cao nhất có chi phí không quá 80% quota GPU còn lại trước 06/12; nếu buộc phải tính theo số epoch dự kiến thay vì tối đa thì dùng ngưỡng 70%. Kết quả ghi vào D027. Rơi xuống bậc thấp hơn khi quota giảm, lỗi vận hành làm mất hơn 2 ngày, hoặc đến 29/11 chưa chọn xong cấu hình X1.
5. **Seed.** Seed `2026` của dữ liệu, split, manifest và D025 không đổi. Seed huấn luyện chính là `2026`; seed huấn luyện bổ sung (bậc A/B) là `2027` và `2028`, áp cho khởi tạo mô hình, sampler và random crop. Run B0 seed `2026` theo D023 trên Parquet giữ nguyên, không chạy lại. Seed bổ sung của B0 dùng đúng cấu hình D023, chỉ đổi seed huấn luyện, và chạy trên backend packed sau khi kiểm parity với Parquet (T08).
6. **Tổng hợp và báo cáo.** Báo cáo kết quả test của mọi seed cùng trung bình ± độ lệch chuẩn mẫu. So sánh chính dựa trên trung bình theo seed, kèm CI bootstrap theo speaker cho từng seed. Không chọn seed theo test; checkpoint dùng cho demo chọn theo dev-full EER.
7. **Mốc.** 29/11: chọn xong cấu hình X1. 06/12: mọi mô hình/seed đăng ký trong registry, không còn quyết định cấu hình mở. Khoảng 13/12: đánh giá test một lần cùng ablation shortcut. Demo làm sau 14/12, trước hạn 03/01/2027.

**Lý do:** checkpoint `facebook/wav2vec2-xls-r-300m` đã ghim có `do_normalize=true`, nên dùng đúng phân phối đầu vào lúc pretrain là lựa chọn mặc định dễ bảo vệ. Pilot một seed khó phân định được: CI bootstrap theo speaker của hiệu EER `none − peak` ở ba pilot B0 là [−3,1; +3,3] điểm. Nhiều seed cho nhiều thông tin hơn nhiều cấu hình một seed, vì chúng đo được dao động do huấn luyện mà CI theo speaker không đo. Ước tính thời gian trước đây khá sát (pilot `none` dự kiến 4,785 giờ, thực tế 4,807 giờ), và tính theo số epoch tối đa vốn đã bảo thủ, nên chọn ngưỡng 80%.

**Giới hạn:** normalization là lựa chọn có lý do, không phải giá trị tối ưu bằng thực nghiệm. XLS-R gần như bất biến với biên độ trong khi B0 dùng `peak`; phải nêu khi so sánh. Kết quả X1 là một cấu hình fine-tune từng phần hợp lý, không phải cấu hình tối ưu. Nếu rơi về bậc C, kết luận chỉ dựa trên một seed. Run B0 seed `2026` dùng Parquet còn seed bổ sung dùng packed; chỉ chấp nhận khi T08 chứng minh đầu vào giống hệt.

**Không thay đổi:** D022, D023, D025, metadata, split, `configs/audio.json`. Không dùng mixed precision hay chạy 2 GPU song song nếu chưa ghi trong D027. Không đọc test trước T19.

## 2026-11-02 — D027: Sửa LFCC tính riêng từng mẫu và huấn luyện lại B0 (B0 v2)

**Bối cảnh:** review độc lập phát hiện, và đã được kiểm chứng lại, rằng với `log_lf=false` hàm `torchaudio.functional.amplitude_to_DB` coi chiều batch của đầu vào `(batch, samples)` là chiều kênh, nên cả batch dùng chung mức cắt `top_db=80`. Đặc trưng LFCC của một mẫu vì thế phụ thuộc các mẫu khác trong batch, cả khi train lẫn khi đánh giá. Đo trên 1.064 mẫu bonafide của dev cũ với checkpoint best `g-4174b60757a44324a9b864a4931e7a18`: chấm theo batch tái dựng đúng score đã lưu (lệch tối đa 3,6e-6); chấm riêng từng mẫu làm 126 score đổi hơn 1e-4 (lớn nhất 0,213) và đảo 7/1.064 quyết định tại threshold 0,0070. Phía spoof và EER toàn tập chưa đo.

**Quyết định:**

1. **Định nghĩa đặc trưng B0.** Mọi lần huấn luyện, đánh giá và suy luận B0 từ nay dùng `lfcc_features` trong `src/models/lfcc_lcnn.py`, tức LFCC tính riêng từng mẫu. Tham số trong `configs/lfcc_lcnn.json` không đổi. Test hồi quy `tests/test_lfcc_batch_independence.py` bảo đảm đặc trưng không đổi khi đổi thành phần hoặc thứ tự batch.
2. **B0 v2.** Huấn luyện lại từ đầu với LFCC đã sửa, giữ cấu hình D023: 15.885 mẫu train của `development-20k-v2`, policy `peak`, LCNN không đổi, Adam, learning rate cố định 0,001, batch 8, float32, tối đa 3 epoch, patience 1, cải thiện nghiêm ngặt theo EER, hòa giữ checkpoint sớm nhất. Khác D023 ở ba điểm: đánh giá sau mỗi epoch và chọn checkpoint/threshold trên dev-full (D025), giống X0/X1; đọc dữ liệu từ packed artifact sau khi qua kiểm parity với Parquet (T08); seed huấn luyện theo mục 3.
3. **Seed của B0 v2.** Luôn chạy đủ ba seed huấn luyện `2026`, `2027`, `2028` nếu benchmark trên dữ liệu thật cho thấy mỗi seed chạy được trong vài giờ trên CPU cục bộ, bất kể XLS-R rơi vào bậc nào của D026. Nếu không đạt điều kiện này, số seed theo bậc của D026. Seed dữ liệu `2026` không đổi.
4. **Chuỗi D023 cũ (v14–v25) và ba pilot** giữ làm lịch sử, ghi nhãn "LFCC phụ thuộc batch". Không chạy phiên 12. Output v25 đã được kiểm nhẹ để lưu hồ sơ: workflow `complete`, cả ba lệnh mã 0; resume input đạt từ bước 5.120 (191 tệp); report `paused` ở bước 5.632, epoch 2/cursor 1.660, 512 bước mới, 15.885 train/4.115 development; `LATEST.json` và checksum latest/best khớp, 104 generation; source tree trùng phiên 10; best vẫn là `g-4174…` (EER 8,3333%); runtime báo cáo 1,919 giờ. Các con số của chuỗi này không dùng làm kết quả chính thức của B0.
5. **Đo ảnh hưởng của lỗi.** Khi nạp đủ dev cũ, chấm checkpoint `g-4174…` theo cả hai cách (theo batch và riêng từng mẫu) trên 4.115 mẫu, báo cáo EER, hiệu EER kèm CI ghép cặp và số quyết định bị đảo. Kết quả chỉ dùng cho phần giới hạn và thảo luận.
6. **D022 giữ nguyên** (`peak`), nhưng ghi rõ ba pilot cùng mang lỗi này; mức cắt chung trong batch có thể tác động khác nhau lên `none`, `peak` và `rms_dbfs`, nên phép so sánh bị nhiễu. Không chạy lại pilot vì CI vốn đã không phân biệt được các policy.
7. **Môi trường chạy B0 v2** được chốt sau benchmark packed trên dữ liệu thật; ưu tiên CPU cục bộ nếu đủ nhanh để quota GPU Kaggle dành cho XLS-R. Thiết bị chạy được ghi vào identity của run.
8. **Thay thế quyết định cũ.** Mục 6 của D025 (chấm lại checkpoint cuối epoch của chuỗi D023 trên dev-full) được thay bằng B0 v2. Câu "run B0 seed 2026 theo D023 trên Parquet giữ nguyên, không chạy lại" ở mục 5 của D026 bị thay thế. Quyết định chốt ngân sách sau T14 mà D026 gọi là "D027" nay mang số D028.
9. **Cổng trước khi chạy B0 v2:** có manifest dev-full theo D025; packed artifact qua kiểm parity với Parquet; runner nhận seed huấn luyện tách khỏi seed dữ liệu, nhận dev-full và backend packed; test CPU/resume và toàn suite đạt.

**Lý do:** đặc trưng của một mẫu phải được xác định duy nhất từ chính mẫu đó; nếu không, threshold không chuyển được sang test hay demo chấm từng tệp, và kết quả B0 phụ thuộc cách ghép batch. Chỉ sửa khi đánh giá sẽ để lại lệch giữa train và đánh giá, vì mô hình cũ học trên đặc trưng bị nhiễu; huấn luyện lại là cách sạch nhất. Chọn checkpoint B0 v2 trên dev-full còn xóa sự bất đối xứng về lịch sử lựa chọn giữa B0 và XLS-R. Với packed, chi phí huấn luyện lại dự kiến thấp hơn nhiều so với chuỗi Parquet, nhưng phải được đo.

**Giới hạn:** B0 v2 không tương đương với các số liệu B0 đã báo trước đây (dev EER 8,33%); phải trình bày như một mô hình mới. Lựa chọn `peak` dựa trên các pilot mang lỗi này. Kết quả vẫn chỉ áp dụng cho `VSASV-HF-public-snapshot-v1` và 15.885 mẫu train.

**Kiểm chứng cổng ngày 03/11:** packed artifact train/dev-full/test-full đã lắp và qua kiểm parity với Parquet (0 lệch; train phủ 100%, dev-full 18,14%, test-full 16,01% do chỉ 68 shard còn Parquet), runner nhận seed huấn luyện, dev-full và packed (mục 9). Benchmark CPU 1 luồng chiếu 3,29 giờ/seed và 9,86 giờ cho ba seed theo số epoch tối đa, nên điều kiện của mục 3 được xem là đáp ứng để chạy đủ ba seed; đây là phép chiếu từ 60 bước, chưa phải thời gian thực chạy. Cổng `ready_for_training` vẫn đóng cho tới khi người dùng xác nhận. Chi tiết trong `docs/current_state.md` và `reports/`.

**Không thay đổi:** metadata, split, seed dữ liệu `2026`, `configs/audio.json`, `configs/lfcc_lcnn.json`, kiến trúc LCNN, D025 (trừ mục 6), D026 (trừ các điểm ở mục 8). Source khóa trên Kaggle của chuỗi D023 không đổi. Không dùng test.

## 2026-11-03 — D028: Chốt bậc A, ngân sách epoch chung và cấu hình huấn luyện XLS-R sau T14

**Bối cảnh:** T14 version 2 (`reports/xlsr_t14_cuda_benchmark.{json,md}`, Tesla T4, Transformers 4.57.6, 12/12 cấu hình ok, 0 OOM) đã được Claude tính lại độc lập từ 30 raw timing mỗi cấu hình: công thức chiếu khớp, loss hữu hạn, tầng đóng băng không đổi, không dùng test. Với batch 16, một epoch train 15.885 mẫu tốn 0,24 giờ (X0), 0,28 (2 tầng), 0,31 (4 tầng), 0,39 (8 tầng), 0,46 (12 tầng); một lượt dev-full 38.777 mẫu tốn 0,65–0,67 giờ. Đánh giá dev-full mỗi epoch tốn hơn một epoch train, nên chi phí chủ yếu theo số epoch. Cả ba seed B0 v2 đều đạt EER tốt nhất đúng ở epoch 3 (trần của D023/D027) và EER còn đang giảm; trần đó xuất phát từ chi phí Parquet/Kaggle (8,76 GPU-giờ/epoch), không còn áp dụng với packed (khoảng 1,1 giờ CPU/epoch/seed).

**Quyết định:**

1. **Bậc A của D026.** Ba mức số tầng Transformer mở `{2, 4, 8}` (N khối cuối), kiểm hai learning rate encoder `{1e-5, 3e-5}` ở mức tầng được chọn, ba seed huấn luyện `2026/2027/2028` cho X1 cuối, X0 và B0 v2. Mức 1 và 12 tầng không đưa vào lưới.
2. **Trần epoch chung.** B0 v2 và mọi run X0/X1 dùng `max_epochs = 10`, `patience = 2`, tối thiểu 1 epoch hoàn tất, đánh giá toàn bộ dev-full sau mỗi epoch, chọn checkpoint và threshold theo EER dev-full, cải thiện nghiêm ngặt, hòa giữ checkpoint sớm nhất. Chọn trên dev, không dùng test. Với B0, mục này thay `max_epochs=3` và `patience=1` của D023/D027 mục 2; mọi tham số khác của D027 giữ nguyên.
3. **Nối dài B0 v2.** Ba run B0 v2 đã kết thúc ở bước 5.958 do chạm trần, và ở cả ba chưa từng có epoch không cải thiện (best đúng epoch 3, EER giảm liên tục), nên dưới patience 2 ba run đó trùng với run dài liền đến hết epoch 3. Vì vậy được nối dài từ checkpoint cuối (trọng số, Adam, sampler, RNG, lịch sử, bộ đếm patience) thay vì chạy lại. Runner phải có chế độ nối dài có kiểm soát (chỉ cho phép khi `stop_reason == max_epochs`, bộ đếm patience bằng 0, các trường identity khác không đổi, ghi rõ trần cũ/mới trong identity) và có test chứng minh nối dài trùng với run dài liền bằng từng bit trên mô hình nhỏ. Báo cáo ba run 3 epoch cũ giữ làm kết quả trung gian có nhãn "trần 3 epoch"; kết quả B0 v2 chính thức là run đã nối dài.
4. **Cấu hình XLS-R.** Batch 16; Adam hai nhóm tham số; learning rate classifier `1e-3`; learning rate encoder theo lưới mục 1 (X0 chỉ có classifier); head `128` đơn vị, dropout `0,2` như đã dùng trong T14; learning rate cố định, không scheduler, không gradient clipping; float32, không AMP; một GPU; `num_workers=0` như T14 trừ khi ghi khác trong identity; normalization theo D026. Các giá trị head/dropout trong T14 là kỹ thuật giữ chỗ, nay được khóa ở D028.
5. **Thứ tự chạy và quy tắc chọn.** (a) Seed `2026`: ba mức tầng với encoder LR `1e-5`, cộng X0. (b) Mức tầng có EER dev-full tốt nhất (hòa chọn ít tầng hơn) được chạy thêm với LR `3e-5`; cấu hình X1 cuối là cặp (tầng, LR) có EER dev-full tốt nhất, hòa chọn LR thấp hơn. (c) Cấu hình X1 cuối và X0 chạy thêm seed `2027` và `2028`; B0 v2 chạy nối dài theo mục 3. Không chọn cấu hình hay seed theo test. Chọn xong X1 trước 29/11.
6. **Ngân sách (D026 mục 4, tính theo trần 10 epoch, không giả định dừng sớm).** Chi phí tối đa mỗi run 10 epoch kèm đánh giá mỗi epoch, batch 16, theo T14: X0 8,9 giờ; 2 tầng 9,4; 4 tầng 9,8; 8 tầng 10,5. Bậc A tối đa 87,7 GPU-giờ (lựa chọn tầng 29,7; LR thứ hai tối đa 10,5; X0 seed 2026 8,9; X1 cuối seed 2027/2028 tối đa 20,9; X0 seed 2027/2028 17,8), 109,6 giờ nếu cộng hệ số an toàn 25% cho checkpoint, khởi động và biến thiên. Quota còn 24,09/30 giờ, làm mới ngày 07/11 (`reports/xlsr_t14_output_verification.json`); nếu giả định quota 30 giờ làm mới hằng tuần vào 07/11, 14/11, 21/11, 28/11 và 05/12 thì tổng đến 06/12 là khoảng 174 giờ, và 80% là khoảng 139 giờ, nên bậc A đạt ngưỡng. Giả định chu kỳ làm mới hằng tuần chưa được xác nhận ngoài ngày làm mới đầu tiên; phải đọc lại quota qua API sau mỗi lần làm mới. Phần còn lại (20%) dành cho đánh giá test-full, ablation shortcut và lỗi vận hành. Quy tắc rơi bậc của D026 mục 4 giữ nguyên.
7. **Thay thế.** Mục 3 của D026 chốt bậc A; "D027" trong D026 mục 4 và 7 đọc là D028. Không đổi mốc 29/11, 06/12, khoảng 13/12, sau 14/12 và hạn 03/01/2027.

**Lý do:** chi phí thật thấp hơn nhiều so với ước lượng của D023, nên trần 3 epoch của B0 chỉ còn là hạn chế nhân tạo; để B0 dừng ở trần trong khi XLS-R chạy dài hơn sẽ làm lệch phép so sánh về phía XLS-R. Cùng một quy tắc dừng cho mọi mô hình giữ so sánh công bằng. Batch 16 nhanh hơn khoảng 9% so với batch 8 ở các mức tầng đã đo. Hai learning rate được thử ở một mức tầng thay vì cả lưới để giữ chi phí trong bậc A.

**Giới hạn:** chi phí chiếu từ 30 bước đo và 512 mẫu dev, chưa gồm checkpoint, khởi động và biến thiên Kaggle; hệ số an toàn 25% là ước tính, không phải số đo. Lưới tầng, learning rate và trần 10 epoch là lựa chọn có lý do, không phải tối ưu; EER vẫn có thể còn giảm sau epoch 10. Learning rate cố định có thể bất lợi cho XLS-R so với lịch có warmup. B0 và XLS-R vẫn khác batch (8 và 16). Seed bổ sung của X0/X1 chỉ đo dao động khởi tạo, sampler và crop. Kết quả chỉ áp dụng cho `VSASV-HF-public-snapshot-v1` và 15.885 mẫu train.

**Không thay đổi:** metadata, split, seed dữ liệu `2026`, `configs/audio.json`, D022, D025, D026 (trừ mục 7), D027 (trừ mục 2). Không dùng test; cổng `ready_for_training` của B0 nối dài và XLS-R vẫn đóng cho tới khi người dùng xác nhận.

## 2026-11-05 — D029: Ghi riêng cấu hình và trạng thái hiệu lực LayerDrop/SpecAugment XLS-R

**Xác nhận:** người dùng trả lời nguyên văn "đồng ý" ngày 05/11/2026 cho đề xuất D029 và tiếp tục T15 từ bước 6.

**Quyết định:** giữ hành vi wrapper đã dùng trong T14: config pretrained `layerdrop=0.1`, `apply_spec_augment=true` không đổi; backbone và encoder luôn eval, chỉ các block mở và classifier ở train. Do đó effective LayerDrop bằng 0 và effective SpecAugment tắt. Identity/report ghi riêng configured/effective, không sửa config pretrained hoặc trọng số. Điều kiện dừng bước 5 T15 đã được giải quyết theo cách diễn giải này, cho phép tiếp tục công việc cục bộ bước 6–12.

**Bằng chứng:** `reports/xlsr_t15_eol_normalization.json`, source T14 trong bundle khớp HEAD, probe mode nhỏ exit 0; Transformers 4.57.6 kiểm cờ training ở encoder cho LayerDrop và backbone cho masking.

**Phạm vi:** không mở ready_for_training, không chạy huấn luyện khoa học, test-full, upload/push/chạy Kaggle, commit hoặc merge. D025/D026/D028 giữ nguyên. Quyết định được ghi trong worktree XLS-R; không sửa docs/decisions.md ở thư mục chính.

**Ghi chú chép sang main (11/11/2026):** mục trên được chép nguyên văn từ docs/decisions.md của worktree XLS-R (nhánh feat/xlsr-runner-d028) theo xác nhận nguyên văn "Viết prompt D029" của người dùng ngày 11/11/2026, để thư mục chính chứa quyết định mà D030–D032 đã tham chiếu. Câu cuối của mục Phạm vi phản ánh thời điểm 05/11; nội dung quyết định không đổi.

## 2026-11-06 — D030: Đăng ký trước tiêu chí nâng trần epoch cho XLS-R

**Xác nhận:** người dùng đồng ý ngày 06/11/2026, nguyên văn "việc nhỏ, bạn tự ghi giúp tôi", cho đề xuất ghi tiêu chí trong cuộc trò chuyện. Mục này được ghi trong khi T18 (X1-L2/L4/L8 seed 2026) đang chạy, **trước khi có kết quả X1 nào**.

**Bối cảnh:** X0 seed 2026 chạy hết trần 10 epoch của D028 (`max_epochs`, bước 9.930); EER dev-full giảm đều từ 8,80% (epoch 1) xuống 4,90% (epoch 8) và 4,20% (epoch 10, best). Nguồn: `kaggle_outputs/xlsr-d028-x0-seed2026-s1/reports/xlsr_X0_seed2026.json`, report T17 `reports/xlsr_t17_x0_seed2026_session1.json` trong worktree XLS-R. Số đo thật: khoảng 0,72 GPU-giờ mỗi epoch với X0 (0,766 giây/bước train, 1.826 giây/lượt dev-full). Quyết định D029 (LayerDrop/SpecAugment) hiện chỉ có trong `docs/decisions.md` của nhánh `feat/xlsr-runner-d028`, chưa merge.

**Quyết định:**

1. **Thời điểm quyết định.** Xét việc nâng trần một lần, ngay sau khi xong ba run X1-L2/L4/L8 LR `1e-5` seed 2026 và trước khi chạy LR `3e-5` (D028 mục 5b). Không xét lại sau khi đã chạy seed 2027/2028.
2. **Run dùng để xét.** Cấu hình X1 thắng ở bước chọn mức tầng theo luật D028 (EER dev-full thấp nhất, hòa chọn ít tầng hơn). X0 chỉ là mốc tham chiếu, không quyết định việc nâng trần.
3. **Giữ trần 10** nếu run đó có best ở epoch 8 trở về trước hoặc dừng sớm do patience.
4. **Xem xét nâng trần** chỉ khi đồng thời: best ở epoch 9 hoặc 10, và EER epoch 10 thấp hơn EER epoch 8 ít nhất **0,3 điểm phần trăm** (lớn hơn độ lệch chuẩn theo seed khoảng 0,18 điểm % đã đo ở B0 v2, `reports/b0_v2_summary.json`).
5. **Nếu nâng:** trần mới đề xuất 15 epoch, patience giữ 2, luật chọn best giữ nguyên D028. Áp dụng cho **mọi run XLS-R trong tập so sánh** (X0 và mọi X1 đã/sẽ chạy), nối dài từ checkpoint cuối theo cơ chế có kiểm soát như B0 (D028 mục 3), không chạy lại từ đầu và không nâng riêng cho cấu hình đang thắng. B0 v2 không đổi vì đã dừng do patience ở epoch 5, không do trần. Trước khi chạy phải có: chế độ nối dài trong runner XLS-R kèm test exact-resume, tính lại quota còn đủ phần dự phòng 20% của D026, và người dùng xác nhận mức trần cụ thể trong một mục quyết định mới.

**Lý do:** đăng ký tiêu chí trước khi thấy kết quả X1 để việc nâng hay giữ trần là áp dụng quy tắc, không phải chọn hậu nghiệm. Ngưỡng 0,3 điểm % tránh nâng trần vì dao động nhỏ hơn mức nhiễu giữa seed.

**Không được làm:** dùng test-full để quyết định trần; đổi patience, luật chọn best hoặc lưới cấu hình cùng lúc với việc nâng trần; nâng trần riêng cho một cấu hình.

**Giới hạn:** ngưỡng 0,3 điểm % và mức 15 epoch là lựa chọn có lý do, không phải tối ưu. Nếu giữ trần 10 thì X0 và có thể cả X1 được báo cáo là chưa hội tụ ở trần; kết luận so sánh với B0 phải nêu giới hạn này.

**Không thay đổi:** D025, D026, D027, D028, D029; metadata, split, seed dữ liệu `2026`, `configs/audio.json`. Không mở thêm cổng nào.

## 2026-11-09 — D031: Kết quả band-limit ablation T19b và cách diễn giải

**Xác nhận:** người dùng đồng ý ngày 09/11/2026, nguyên văn "được hãy thực hiện", cho đề xuất ghi D031 và cập nhật báo cáo trong cuộc trò chuyện. Mục này được ghi sau khi T19b hoàn tất và đã được Claude kiểm độc lập; T20 (X1-L8-lr3e-5 seed 2026) chưa có kết quả.

**Bối cảnh:** D007 yêu cầu báo cáo chính có band-limit ablation vì toàn bộ VC có sample rate gốc 40 kHz, còn bonafide, nhiễu đối kháng (AP) và replay ở 16 kHz. Kiểm tra ngày 08/11/2026 xác nhận tỉ lệ này là 100% ở cả `closed_train` (`data/manifests/development_20k_v2.csv`) và dev-full (`data/manifests/closed_dev_full_v1.csv`). Mọi waveform đều được resample về 16 kHz, nhưng bộ lọc chống aliasing khi hạ từ 40 kHz có thể để lại dấu vết ở dải sát 8 kHz, trùng hoàn toàn với nhãn VC. T19b là phép thử lúc chấm: không train, không đọc test, chỉ dùng dev-full và threshold cố định của từng checkpoint. B0 v2 ba seed được chấm trên đủ 38.777 mẫu/186 shard; XLS-R X1-L8-lr1e-5 seed 2026 được chấm trên tập con phân tầng 3.120 mẫu/169 shard (1.000 bonafide, 1.000 VC, 1.000 AP, 120 replay; SHA-256 danh sách `869d8f87…`). Có bốn điều kiện: `original`; `roundtrip_40k` (16→40→16 kHz bằng `torchaudio.functional.resample`, chỉ áp cho bonafide); `lowpass_7k` và `lowpass_4k` (FIR Hann 513 tap, pha tuyến tính, có bù trễ, áp lại `peak`). Ngưỡng chỉ báo được khai báo trước khi chạy. Chỉ báo mạnh khi có một trong hai điều kiện: `roundtrip_40k` làm FPR bonafide tăng ≥ 10 điểm %; hoặc `lowpass_7k` làm EER VC tăng ≥ 3 lần trong khi EER AP tăng < 1,5 lần. `lowpass_4k` chỉ dùng để đo độ bền. Nguồn: `reports/shortcut_probe_dev.json`.

**Kết quả (dev-full, không phải test):**

- XLS-R X1-L8-lr1e-5 seed 2026: chỉ báo **yếu**. Round-trip làm FPR bonafide tăng 0,5 điểm %; `lowpass_7k` làm EER VC tăng ×1,23 và EER AP tăng ×2,0. Với `lowpass_4k`, EER tổng trên tập con là 2,30%, so với 1,46% ở `original`.
- B0 v2: chỉ báo **mạnh** ở seed 2027 và 2028; seed 2026 yếu nhưng sát ngưỡng. Round-trip làm FPR bonafide tăng lần lượt 7,67 / 11,71 / 10,45 điểm % (seed 2026/2027/2028). `lowpass_7k` làm EER AP tăng mạnh hơn EER VC (AP ×11,34 / ×5,89 / ×2,29; VC ×1,39 / ×1,82 / ×1,77). Với `lowpass_4k`, EER tổng là 38,97–51,54%.
- Claude kiểm độc lập ngày 09/11/2026 bằng cách chạy lại 376 test, tính lại toàn bộ metric từ các khối score, và chấm lại mẫu bằng bộ lọc tự viết. Mọi kết quả khớp report.

**Quyết định:**

1. T19b là band-limit ablation phiên bản đầu theo D007, áp dụng cho B0 v2 và XLS-R X1-L8-lr1e-5 seed 2026.
2. Giữ giao thức âm thanh (D007, `configs/audio.json`) và giao thức huấn luyện D028. Không thêm lọc băng tần khi train; không train lại B0 hay XLS-R vì lý do này.
3. Cách diễn giải: ở mức phép thử này phát hiện được, EER dev-full của XLS-R L8 không được giải thích bởi dấu vết resample 40→16 kHz. B0 v2 phụ thuộc một phần vào đặc trưng dải cao. Phụ thuộc này thể hiện rõ nhất qua phản ứng với round-trip, và qua mức suy giảm của AP và replay khi lọc thông thấp. Báo cáo phải nêu điều này khi trình bày kết quả B0 và khi so sánh B0 với XLS-R.
4. Chạy lại phép thử với cùng đặc tả (tập con, bốn điều kiện, bộ lọc, ngưỡng chỉ báo) cho cấu hình X1 cuối cùng, sau khi chọn theo D028 mục 5, và trước khi chấm test-full. Nếu cấu hình đó chạm chỉ báo mạnh, phải dừng và lập quyết định mới trước khi chấm test. Chạy thêm cho X0 seed 2026 là tùy chọn.
5. Báo cáo phải cập nhật chương phương pháp/B0 (mục 4.2, 9.3, 10 của `baocao/phuong_phap_du_lieu_va_b0.md`) và thêm mục band-limit ablation. Số liệu lấy từ `reports/shortcut_probe_dev.json` và từ report phân tích phổ do script trong repo sinh ra.

**Lý do:** phép thử được đăng ký trước, có kiểm tái tạo score gốc, và cho kết quả khác biệt rõ giữa hai họ mô hình. Mô hình chính không cần đổi giao thức. Mô hình cơ sở cần được diễn giải kèm giới hạn. Train lại toàn bộ với dữ liệu đã lọc băng tần sẽ tốn khoảng một ngày GPU và đổi giao thức đã khóa, trong khi không có chỉ báo nào đòi hỏi việc đó cho XLS-R.

**Giới hạn:** với XLS-R, phép thử mới làm trên một cấu hình, một seed và tập con 3.120 mẫu; ở mức 1.000 bonafide, mỗi mẫu tương ứng 0,1 điểm %. Đây là phép thử lúc chấm, không phải train lại trên dữ liệu lọc băng tần. Round-trip chỉ mô phỏng dấu vết của resampler torchaudio. Ngưỡng chỉ báo là lựa chọn có lý do, không phải kiểm định thống kê. Kết quả chỉ đúng trên dev-full của `VSASV-HF-public-snapshot-v1`.

**Không được làm:** dùng test-full để đánh giá shortcut; đổi ngưỡng chỉ báo hoặc đặc tả phép thử sau khi đã thấy kết quả; tuyên bố XLS-R "không có shortcut" một cách tuyệt đối.

**Không thay đổi:** D007, D025, D026, D027, D028, D029, D030; metadata, split, seed dữ liệu `2026`, `configs/audio.json`. Không mở thêm cổng nào.

## 2026-11-09 — D032: Đăng ký trước giao thức chấm test-full

**Xác nhận:** người dùng đồng ý ngày 09/11/2026, nguyên văn "Ghi D032", cho bản nháp giao thức chấm test-full do Claude trình bày trong cuộc trò chuyện. Mục này được ghi **trước khi đọc bất kỳ dữ liệu test nào**, khi T21 (seed 2027/2028 của X1 cuối và X0) và T22 (phép thử shortcut cho X1 cuối) đang chạy.

**Bối cảnh:** D025 mục 3 quy định test-full chỉ được đọc qua entrypoint có cổng khóa, mỗi mô hình đúng một lần. D026 mục 7 đặt mốc 06/12 cho registry đầy đủ và khoảng 13/12 cho lần chấm test cùng ablation shortcut. Theo D028 mục 5b, cấu hình X1 cuối là X1-L8-lr3e-5: EER dev-full seed 2026 là 0,8162%, so với 0,8833% của X1-L8-lr1e-5. Nguồn: `kaggle_outputs/xlsr-d028-X1-L8-lr3e-5-seed2026/reports/xlsr_X1-L8-lr3e-5_seed2026.json`, đã được Claude kiểm độc lập. Test-full gồm `data/manifests/closed_test_full_v1.csv` (30.698 dòng, SHA-256 `0a7a70bc8ededdd83e5f6df06580fc954c6f2da848366e37bc265331ca17d266`, theo `reports/closed_eval_full_v1.json`) và packed artifact `data/processed/packed/closed_test_full_v1`.

**Quyết định:**

1. **Mô hình và checkpoint.** Chấm B0 v2, X0 và X1 cuối (X1-L8-lr3e-5), mỗi loại đủ ba seed 2026, 2027, 2028, tổng 9 mục. Mỗi mục dùng checkpoint best trên dev-full và `decision_threshold` lấy từ dev-full của chính checkpoint đó. Toàn bộ được khóa trong registry `reports/model_registry_v1.json` (SHA-256 checkpoint, generation, threshold, EER dev-full, source/identity) trước khi chấm. Hạn đăng ký: 06/12/2026.
2. **Dữ liệu và tiền xử lý.** Dùng toàn bộ test-full nêu trên, cùng pipeline âm thanh hiện hành (`configs/audio.json`, `peak`, center crop 4 giây). Không lọc, không bù mẫu, không đổi nhãn.
3. **Metric chính.** EER test-full cho từng mô hình và từng seed; báo cáo thêm trung bình ± SD mẫu qua ba seed.
4. **So sánh chính.** So X1 cuối với B0 v2, và X1 cuối với X0. Với từng seed, tính CI 95% của hiệu EER bằng bootstrap ghép cặp theo người nói (`paired_speaker_bootstrap` trong `src/metrics/bootstrap.py`, 1.000 lần lặp, seed 2026). Mỗi hệ thống được báo cáo kèm CI bootstrap theo người nói của riêng nó.
5. **Metric phụ.** ROC-AUC; macro-F1; FPR và FNR tại threshold dev; EER theo kiểu tấn công (bonafide chung so với từng kiểu); ma trận nhầm lẫn. Replay (khoảng 120 mẫu) chỉ được mô tả, không dùng để kết luận.
6. **Band-limit ablation trên test.** Áp đúng bốn điều kiện và đặc tả bộ lọc của T19b/D031 (`original`, `roundtrip_40k` chỉ cho bonafide, `lowpass_7k`, `lowpass_4k`) cho B0 v2 và X1 cuối, cả ba seed, trên toàn bộ test-full. X0 chỉ chấm `original`. Threshold giữ nguyên từ dev. Kết quả chỉ để báo cáo và đối chiếu với dev; không được dùng để đổi mô hình, threshold hay giao thức.
7. **Luật một lần.** Test chỉ được đọc qua entrypoint có khóa. Mỗi mô hình-seed tạo một file khóa trước khi mở dữ liệu và bị từ chối nếu khóa đã tồn tại. Nếu phát hiện lỗi sau khi chấm, phải ghi nhận lỗi và lập quyết định mới nêu rõ lần chấm lại cùng lý do; không chấm lại âm thầm.
8. **Điều kiện trước khi chấm:** registry đủ 9 mục và khớp `configs/test_evaluation.json`; entrypoint và bootstrap đã được thử trên dev-full và được Claude kiểm; T22 cho X1 cuối có chỉ báo yếu theo D031 (nếu chỉ báo mạnh thì dừng và lập quyết định mới); đã kiểm dataset Kaggle có test-full chưa (upload mới cần người dùng xác nhận riêng); người dùng mở cổng chấm test trong cuộc trò chuyện.

**Lý do:** đăng ký trước mô hình, threshold, metric và phép so sánh để kết quả test là phép áp dụng quy tắc, không phải lựa chọn sau khi thấy số. Báo cáo đủ ba seed và CI theo người nói tránh việc chỉ chọn seed tốt nhất hoặc coi chênh lệch nhỏ là có ý nghĩa.

**Không được làm:** dùng test để chọn checkpoint, threshold, hiệu chuẩn score, seed hoặc cấu hình; bỏ seed hay chỉ báo cáo seed tốt nhất; thêm mô hình hay điều kiện sau khi đã thấy kết quả test mà không ghi rõ là phân tích hậu nghiệm; đọc test ngoài entrypoint có khóa.

**Giới hạn:** bootstrap 1.000 lần lặp và CI 95% là lựa chọn quy ước; các so sánh trên nhiều seed và nhiều cặp không hiệu chỉnh đa so sánh, nên phải diễn giải thận trọng. Kết quả chỉ áp dụng cho `VSASV-HF-public-snapshot-v1`, mô hình huấn luyện trên 15.885 mẫu, và không phải tái lập official VSASV.

**Không thay đổi:** D007, D025, D026, D027, D028, D029, D030, D031; metadata, split, seed dữ liệu `2026`, `configs/audio.json`. Không mở cổng test.

## 2026-11-10 — D033: Chạy 2 GPU song song trên Kaggle và dùng Colab Google AI Pro làm môi trường GPU bổ sung

**Xác nhận:** người dùng ngày 10/11/2026, nguyên văn "được, có thể chạy 2 gpu song song"; người dùng cho biết đã đăng ký gói Google AI Pro có 200 đơn vị điện toán để chạy GPU Colab. Nội dung mục này được người dùng duyệt ngày 10/11/2026, nguyên văn "Duyệt D033".

**Bối cảnh:** D026 cấm chạy 2 GPU song song khi chưa ghi quyết định; D018 coi Colab trả phí là dự phòng cần xác nhận riêng trước khi phát sinh chi phí. Kernel Kaggle `NvidiaTeslaT4` cấp 2 GPU nhưng mỗi run D028 chỉ dùng một (`gpus_per_run = 1`); quota Kaggle tính theo thời gian phiên, không theo số GPU. Runner `scripts/train_xlsr.py` đã đặt `torch.backends.cuda.matmul.allow_tf32 = False` và `torch.backends.cudnn.allow_tf32 = False`; identity mỗi run ghi `device = cuda:0`.

**Quyết định:**

1. **Kaggle 2 GPU song song.** Một kernel được chạy tối đa 2 run D028 độc lập, mỗi run là một tiến trình riêng chỉ thấy một GPU qua `CUDA_VISIBLE_DEVICES` và dùng `--device cuda:0`. Mỗi run giữ nguyên cấu hình, identity, checkpoint, report và quy tắc dừng của D028/D029; không dùng DataParallel/DDP, không đổi `gpus_per_run`. Hai run dùng chung hạn giờ phiên; một run lỗi không được làm dừng hay sửa run còn lại.
2. **Colab là môi trường GPU bổ sung.** Dùng Colab theo gói Google AI Pro với ngân sách 200 đơn vị điện toán. Vượt 200 đơn vị hoặc mua thêm cần xác nhận riêng. GPU mặc định là L4 nếu phép đo tốc độ cho thấy chi phí mỗi run thấp hơn T4 Colab; dùng A100 hoặc H100 cần xác nhận riêng.
3. **Quy tắc phần cứng.** Mọi run giữ float32, không AMP, TF32 tắt. Report ghi tên GPU, driver, phiên bản CUDA, torch, transformers và numpy. Một run chỉ chạy trên một loại GPU: resume trên loại GPU khác bị từ chối. Ba seed của cùng một mô hình ở cùng quy mô dữ liệu chạy trên cùng một loại GPU.
4. **Kiểm tương đương trước khi dùng Colab cho run khoa học.** Chấm lại checkpoint best X1-L8-lr3e-5 seed 2026 (generation `g-eb35f3c2379f4a47bd22744c9be1dd8f`) trên toàn bộ dev-full bằng Colab, so với 38.777 score đã lưu trong report Kaggle. Đạt khi đồng thời: |ΔEER| ≤ 0,02 điểm %; số mẫu đổi quyết định tại threshold đã khóa ≤ 10; max |Δ score| ≤ 0,05 theo đơn vị score đã lưu. Không đạt thì không dùng Colab cho run khoa học và lập quyết định mới.
5. **Registry v1 (D032) giữ trên Kaggle T4.** X0 seed 2027/2028 chạy trên Kaggle T4 (được phép theo mục 1). Chín lượt chấm test của D032 chạy cùng một môi trường. Colab không được dùng cho registry v1, ngoại trừ phép kiểm ở mục 4 vốn chỉ dùng dev-full.

**Lý do:** chạy 2 run trên 2 GPU sẵn có giảm gần một nửa quota Kaggle cho mỗi cặp run mà không đổi thuật toán. Colab thêm năng lực tính toán cho các giai đoạn sau. Ghi phần cứng, tắt TF32 và đăng ký trước ngưỡng tương đương giữ cho kết quả từ hai môi trường so sánh được.

**Giới hạn:** chạy song song có thể chậm hơn do hai tiến trình dùng chung CPU của kernel. Giá đơn vị Colab theo loại GPU không được Google công bố chính thức và có thể thay đổi. Gói Google AI Pro có thể không có chạy nền và không đảm bảo loại GPU được cấp. Khác phần cứng vẫn gây sai khác số học nhỏ, cùng cỡ với tính không tất định của GPU; ngưỡng ở mục 4 là lựa chọn quy ước.

**Không thay đổi:** D025, D028, D029, D030, D031, D032; metadata, split, seed dữ liệu `2026`, `configs/audio.json`. D033 không quyết định việc mở rộng tập train; việc đó cần quyết định riêng và phải ghi trước khi mở cổng test của D032.

## 2026-11-13 — D034: Sửa tiêu chí tái tạo logit của T22 cho score lưu dạng float32

**Xác nhận:** người dùng duyệt ngày 13/11/2026, nguyên văn "Duyệt D034", cho bản nháp do Claude trình bày trong cuộc trò chuyện.

**Bối cảnh:** T22 (phép thử shortcut cho X1 cuối X1-L8-lr3e-5 seed 2026, D031/D032 mục 8) dừng ở `blocked_reproduction`: trên tập con 3.120 mẫu/169 shard, sai khác tối đa giữa `raw_logit` chấm lại trên CPU và `stable_logit(score lưu)` là 0,0689, vượt ngưỡng 0,05 (`reports/shortcut_probe_dev_x1_final.json`, mục `reproduction.full_subset`). Score dev-full trong report Kaggle được lưu dạng float32; gần p = 1, hai giá trị float32 liền kề (ví dụ 0,99999905 và 0,99999917) ứng với logit chênh khoảng 0,134, nên logit suy ngược từ score lưu có sai số lượng tử vượt 0,05. Claude đối chiếu từng mẫu (`.cache/claude-check-t22/analyze.log`): với score lưu ≤ 0,999, sai khác tối đa 0,000365; 0 mẫu đổi quyết định tại threshold; sai khác xác suất tối đa 2,3e-5.

**Quyết định:** một mẫu đạt điều kiện tái tạo khi |`raw_logit` − `stable_logit(score lưu)`| ≤ 0,05 + g, trong đó g là khoảng cách logit từ score lưu tới giá trị float32 liền kề, lấy giá trị lớn hơn trong hai phía. Giữ điều kiện 0 mẫu đổi quyết định ngoài biên 0,05. Áp dụng cho cả 200 mẫu đầu và toàn tập con. T22 tiếp tục từ giai đoạn đang dừng, dùng đúng bộ lọc, khối dữ liệu và ngưỡng chỉ báo của T19b/D031; không chấm lại các khối đã có.

**Lý do:** sai khác quan sát được nằm trong độ phân giải của score đã lưu, không phải do mô hình chấm lệch; tiêu chí cũ không thể đạt ở vùng score bão hòa dù mô hình tái tạo đúng.

**Thời điểm:** quyết định được ghi khi chưa có chỉ báo shortcut nào của X1 cuối (các điều kiện round-trip và lọc thông thấp chưa chạy), nên không phải chọn tiêu chí sau khi thấy kết quả chỉ báo.

**Không thay đổi:** ngưỡng chỉ báo shortcut, bộ lọc, tập con, checkpoint, threshold; D031, D032, D033; metadata, split, seed dữ liệu `2026`, `configs/audio.json`. Không mở cổng test.

## 2026-11-13 — D035: Biến thể X1 lọc thông thấp 7 kHz để đo mức phụ thuộc vào dải cao (X1R)

**Xác nhận:** người dùng duyệt ngày 13/11/2026, nguyên văn "duyệt nguyên văn", cho bản nháp do Claude trình bày trong cuộc trò chuyện.

**Bối cảnh:** 100% mẫu VC có sample rate gốc 40 kHz; bonafide, AP và replay ở 16 kHz (D007, D031). Hạ VC từ 40 kHz xuống 16 kHz để lại dấu vết bộ lọc chống răng cưa sát 8 kHz. Phép đo phổ R2 cho thấy trung vị công suất dải 7,95–8 kHz là −59,02 dB với bonafide, −69,99 dB với VC, và −66,12 dB với bonafide sau round-trip 16→40→16 kHz (`reports/highband_spectrum_dev.json`). Theo D031, B0 v2 có chỉ báo shortcut mạnh ở 2/3 seed; X1-L8-lr1e-5 có chỉ báo yếu. Chỉ báo của X1 cuối đang chờ T22 chạy tiếp theo D034. Phép thử lúc chấm (T19b, T22, D032 mục 6) chỉ đo được mô hình đã học nhạy với dải cao đến đâu, không cho biết mô hình còn tốt bao nhiêu nếu không được dùng dải cao ngay từ đầu.

**Quyết định:**

1. **Mô hình.** Thêm biến thể X1R (`run_id` = `X1-L8-lr3e-5-lp7k`). Cấu hình giống hệt X1 cuối theo D028/D029: mở 8 tầng, lr encoder 3e-5, lr classifier 1e-3, batch 16, Adam, float32, trần 10 epoch, patience 2, chọn best theo EER dev-full. Chỉ khác ở bước đầu vào ở mục 2.
2. **Đầu vào.** Mọi mẫu train và dev-full của X1R, và test-full khi được phép chấm, đi qua đúng bộ lọc `lowpass_7k` của T19b. Bộ lọc là FIR pha tuyến tính, windowed-sinc 513 tap, cửa sổ Hann đối xứng, cắt 7.000 Hz, bù trễ 256 mẫu, biên zero-extension, peak lại sau khi lọc. SHA-256 hệ số float32 là `973b82a36db8dab4f802de2102c5feed2bfb8cf6bb5846472c07d9984ab4c977`. Bộ lọc áp lên đoạn 4 giây sau khi resample 16 kHz, crop và peak, trước chuẩn hóa processor của XLS-R, theo đúng thứ tự `apply_condition` của T19b. Tham số bộ lọc nằm trong identity của X1R. Không sửa `configs/audio.json`; 9 mô hình của registry v1 giữ nguyên pipeline.
3. **Seed và dữ liệu.** Seed huấn luyện 2026, 2027, 2028; seed dữ liệu 2026. Dùng cùng 15.885 mẫu train và 38.777 mẫu dev-full. Threshold lấy tại EER dev-full đã lọc của chính checkpoint best.
4. **Vị trí trong giao thức.** X1R không thuộc registry v1 của D032; registry v1 vẫn đúng 9 mục. X1R có registry phụ riêng (`reports/model_registry_x1r_draft.json`) với cùng các trường như D032 mục 1. Phải đăng ký đủ trước khi người dùng mở cổng test, và được chấm test-full đúng một lần qua entrypoint có khóa (D032 mục 7).
5. **So sánh và metric**, trên test-full, sau khi cổng mở: (a) EER X1R so với EER X1 cuối ở điều kiện `original`; hiệu số đo cái giá của việc bỏ dải trên 7 kHz, gồm cả thông tin hợp lệ lẫn dấu vết resample. (b) EER X1R so với EER X1 cuối ở điều kiện `lowpass_7k` (đã có trong D032 mục 6); phép này tách hiệu ứng "được huấn luyện quen với đầu vào bị lọc" khỏi hiệu ứng "bị lọc lúc chấm". Với từng seed, tính CI 95% của hiệu EER bằng `paired_speaker_bootstrap` (1.000 lần lặp, seed 2026). Báo cáo thêm trung bình ± SD mẫu qua ba seed, EER theo kiểu tấn công và FPR bonafide. Replay chỉ mô tả.
6. **Kiểm tính hợp lệ trên dev.** Chấm X1R trên tập con T19b (3.120 mẫu) với điều kiện `roundtrip_40k`. Kỳ vọng FPR bonafide tăng không quá 0,5 điểm %, vì vùng chứa dấu vết đã bị lọc. Đây chỉ là kiểm tính hợp lệ của pipeline, không dùng để chọn mô hình.
7. **Thứ tự và điều kiện chạy.** Ưu tiên 1 là X0 seed 2027/2028 cho registry v1 (mốc 06/12). X1R chạy sau đó trên Kaggle T4, được dùng chế độ cặp của D033. Nếu T22 cho X1 cuối chỉ báo yếu: X1R chạy khi còn quota, mục tiêu đủ 3 seed trước khi mở cổng test. Nếu T22 cho chỉ báo mạnh: X1R là bắt buộc, cổng test không mở cho tới khi X1R đủ 3 seed và đã đăng ký; đây là quyết định mới mà D032 mục 8 yêu cầu. Nếu tới lúc người dùng mở cổng test mà X1R chưa đủ 3 seed: báo cáo mọi seed đã hoàn tất và ghi rõ là chưa đủ; không bỏ seed, không chọn seed theo test.
8. **Ngân sách.** Theo D028, mỗi run X1 tám tầng tốn tối đa 10,5 GPU-giờ cho 10 epoch. Ba seed tối đa 31,5 giờ, khoảng 39 giờ khi cộng hệ số an toàn 25%. Ba seed X1 cuối thực tế dừng sớm ở epoch 4–8, run seed 2028 tốn khoảng 7,5 GPU-giờ. Đọc quota qua API trước và sau mỗi phiên.

**Lý do:** phép thử lúc chấm chỉ cho chỉ báo. Huấn luyện một mô hình ngay từ đầu không thấy dải trên 7 kHz cho biết trực tiếp hiệu năng còn lại khi lối tắt resample bị loại khỏi đầu vào. Dùng lại đúng bộ lọc của T19b giúp kết quả so được với ablation đã có và tránh thêm một tham số tự chọn. Lọc đồng loạt mọi lớp đơn giản và tất định hơn tăng cường round-trip ngẫu nhiên, vốn cần chọn thêm xác suất áp dụng.

**Giới hạn:** lọc 7 kHz bỏ cả thông tin hợp lệ ở dải cao, nên hiệu ở mục 5(a) không phải thuần do shortcut. Chỉ một cấu hình, không tinh chỉnh lại siêu tham số cho đầu vào đã lọc. Không loại trừ được các lối tắt khác như codec, khoảng lặng, độ dài hay đặc trưng vocoder dưới 7 kHz. Kết quả chỉ áp dụng cho `VSASV-HF-public-snapshot-v1` và 15.885 mẫu train. Phương án tăng cường round-trip không được thử.

**Không thay đổi:** D007 và `configs/audio.json`, D025, D028, D029, D031, D032 (registry v1 vẫn 9 mục), D033, D034; metadata, split, seed dữ liệu `2026`. Không mở cổng test.

## 2026-11-13 — Ghi chú đính chính D034

**Xác nhận:** người dùng duyệt ngày 13/11/2026, nguyên văn "Duyệt đính chính", cho nội dung do Claude trình bày trong cuộc trò chuyện.

**Đính chính:** ví dụ trong phần Bối cảnh của D034 ("0,99999905 và 0,99999917 … chênh khoảng 0,134") gọi hai giá trị này là float32 liền kề; thực ra chúng cách nhau hai bước float32 (2 × 2⁻²⁴). Hai giá trị float32 liền kề trong vùng này ứng với logit chênh khoảng 0,074, cũng là bước lượng tử lớn nhất đo được khi chạy T22 theo D034 (`max_float32_gap_logit` = 0,0741 trong `reports/shortcut_probe_dev_x1_final.json`). Phần Quyết định và cách tính g của D034 không đổi; kết quả T22 không bị ảnh hưởng.

## 2026-11-15 — D036: Thí nghiệm mở rộng train ×2 cho X1 cuối và B0 v2 (scale2)

**Xác nhận:** người dùng duyệt ngày 15/11/2026, nguyên văn "duyệt, bạn có thể thực hiện thay đổi luôn mà không cần tạo prompt yêu cầu codex thực hiện", cho bản nháp do Claude trình bày trong cuộc trò chuyện kèm các sửa đổi về quota Kaggle 45 giờ (mục 6, Bối cảnh, Lý do).

**Bối cảnh:** D025 mục 1 khóa train của mọi mô hình ở 15.885 mẫu `closed_train` (khoảng 10,5% của 150.999 mẫu), gồm 153 trên 798 người nói và 57 shard. Registry v1 của D032 (9 mục, complete=true) và kết quả chính của đồ án giữ nguyên ở quy mô này. Ở quy mô hiện tại, EER dev-full trung bình của B0 v2 là 8,139320% và của X1 cuối khoảng 0,77% (ba seed). Ít dữ liệu có thể bất lợi cho B0 (học từ đầu) hơn so với XLS-R (đã pretrain), nên khoảng cách giữa hai mô hình có thể phụ thuộc vào quy mô train. Pool Parquet cục bộ (68 shard) chỉ có 10.562 bonafide, 7.173 VC, 4.887 AP và 145 replay thuộc closed_train (`reports/development_manifest_20k_v2.md`), không đủ cho ×2. Từ 15/11/2026, quota GPU Kaggle của tài khoản là 45 giờ mỗi tuần (đọc qua API: total 45,00h, refresh 21/11/2026 00:00 UTC).

**Quyết định:**

1. **Câu hỏi.** Đo mức cải thiện EER khi gấp đôi dữ liệu train, cho từng mô hình. Đây là thí nghiệm bổ sung về quy mô, không thay registry v1 và không đổi kết luận chính ở quy mô ×1.
2. **Dữ liệu ×2 (`development-scale2-v1`).** Train gồm 31.770 mẫu `closed_train`. **Chứa trọn 15.885 mẫu hiện tại** và thêm 15.885 mẫu mới, để hai mức lồng nhau. Quota tổng theo phân bố closed_train, chia theo phương pháp phần dư lớn nhất: bonafide 14.267, VC 8.697, AP 8.697, replay 109. Phần thêm của mỗi loại bằng quota tổng trừ số mẫu loại đó đã có.
   - **Ứng viên:** mẫu `closed_train` có trong pool cục bộ, chưa thuộc train hiện tại.
   - **Loại khỏi ứng viên** mọi mẫu có content hash (hàm băm của D017) trùng với train hiện tại, dev-full hoặc test-full. Trong nội bộ ứng viên, mỗi nhóm trùng chỉ giữ file có `SHA256("2026|file")` nhỏ nhất.
   - **Chọn:** trong từng `utt_type`, xếp ứng viên theo `SHA256("2026|file")` tăng dần rồi lấy đủ phần thêm.
   - **Không đổi** dev-full, test-full, split, metadata hay seed dữ liệu.
3. **Tải thêm shard.** Chỉ tải shard train của `VSASV-HF-public-snapshot-v1` (cùng revision với D015) chưa có trên máy, theo thứ tự `SHA256("2026|shard")` tăng dần, cho tới khi đủ mọi quota sau khi khử trùng.
   - Trần 120 shard hoặc 10 GB. Chạm trần mà chưa đủ thì dừng và lập quyết định mới.
   - Mỗi shard kiểm kích thước, SHA-256, đọc được và khớp metadata như D016.
   - Không xem hay dùng `closed_test` để chọn shard.
4. **Mô hình.** **X1 cuối** (X1-L8-lr3e-5) và **B0 v2**, mỗi mô hình ba seed huấn luyện 2026, 2027, 2028. Cấu hình y hệt D028/D029 (X1) và D027/D028 (B0); chỉ khác dữ liệu train và số bước mỗi epoch, tính lại theo 31.770 mẫu. Trần 10 epoch, patience 2, chọn best và threshold theo EER dev-full. Không chạy X0 và X1R ở ×2.
5. **Cấu hình và identity.** Dùng file cấu hình riêng cho scale2. Không sửa `configs/xlsr_training.json` hay cấu hình B0 hiện hành, để identity của mọi run ×1 giữ nguyên byte. Identity của run scale2 ghi manifest, artifact và SHA mới.
6. **Nơi chạy.** B0 ×2 chạy trên CPU máy cục bộ như B0 v2. X1 ×2 chạy trên Kaggle T4 (D033 mục 1, được dùng chế độ cặp), cùng loại GPU với registry v1, sau X1R. Mỗi run có thể cần một phiên resume vì có thể vượt hạn phiên 11 giờ; mỗi lần resume cần người dùng xác nhận. Colab chỉ là phương án dự phòng khi quota Kaggle không đủ để xong trước khi mở cổng test, và khi đó áp dụng phép kiểm tương đương của D033 mục 4, ba seed trên cùng một loại GPU, ngân sách tối đa 180 trong 200 đơn vị điện toán.
7. **Registry phụ.** `reports/model_registry_scale2_draft.json`, 6 mục, cùng các trường như D032 mục 1. Phải đăng ký trước khi mở cổng test. Mỗi mục chấm test-full đúng một lần qua entrypoint có khóa (D032 mục 7), cùng pipeline và điều kiện `original`.
8. **So sánh và metric**, trên test-full sau khi mở cổng: với từng mô hình và từng seed, hiệu EER ×1 trừ ×2 kèm CI 95% bằng `paired_speaker_bootstrap` (1.000 lần, seed 2026). Báo cáo trung bình ± SD mẫu của các hiệu qua ba seed, và mô tả chênh lệch giữa mức cải thiện của B0 và của X1. Metric phụ như D032 mục 5; replay chỉ mô tả.
9. **Thứ tự và điều kiện.** X1R (D035) giữ ưu tiên trên Kaggle. Cổng test chỉ mở khi người dùng quyết định. Nếu khi đó scale2 chưa đủ ba seed của một mô hình, báo cáo mọi seed đã hoàn tất kèm ghi chú chưa đủ; không bỏ seed, không chọn seed theo test. Tải shard, đóng gói, upload dataset và dùng Colab, mỗi bước cần người dùng xác nhận riêng (D025, D033).

**Lý do:** gấp đôi dữ liệu tốn ít hơn nhiều so với toàn bộ closed_train: X1 tối đa khoảng 14 GPU-giờ mỗi run (theo số đo T14 trong D028), so với khoảng 45 giờ ở ×10. Mức này vẫn đủ để thấy xu hướng. Tập lồng nhau và cách chọn mẫu giống D016/D017 giúp so ×1 với ×2 sạch. Thêm người nói và shard mới cũng tăng độ đa dạng của train. Chạy X1 ×2 trên Kaggle T4 giữ cùng phần cứng với registry v1, nên so sánh ×1 với ×2 không lẫn yếu tố phần cứng.

**Giới hạn:** chỉ hai điểm quy mô, chưa đủ để vẽ một đường cong học đầy đủ. Mẫu mới lấy từ một số shard, không phải mẫu ngẫu nhiên đều trên toàn bộ closed_train. Train ×2 vẫn chỉ khoảng 21% closed_train. Không tinh chỉnh lại siêu tham số cho dữ liệu lớn hơn; trần 10 epoch có thể bất lợi cho mô hình còn đang cải thiện. Ở ×2 một epoch X1 khoảng 1,3 giờ gồm đánh giá dev, nên 10 epoch có thể vượt hạn phiên 11 giờ và cần resume. Kết quả chỉ áp dụng cho `VSASV-HF-public-snapshot-v1`.

**Không thay đổi:** D025 (kết quả chính ở 15.885 mẫu), D028, D029, D032 (registry v1 vẫn 9 mục), D033, D035; dev-full, test-full, split, metadata, seed dữ liệu `2026`, `configs/audio.json`. Không mở cổng test.

## 2026-11-15 — Ghi chú đính chính D033

**Xác nhận:** người dùng ngày 15/11/2026 cho biết gói Colab của mình có tính năng chạy nền, và yêu cầu nguyên văn "có hãy thực hiện giúp tôi" cho đề xuất ghi chú này do Claude trình bày trong cuộc trò chuyện.

**Đính chính:** phần Giới hạn của D033 ghi "Gói Google AI Pro có thể không có chạy nền". Theo người dùng, gói Colab đang dùng có chạy nền. Giới hạn còn lại: thời gian tối đa mỗi phiên và việc cấp loại GPU không được bảo đảm; đơn vị điện toán vẫn bị trừ khi chạy nền. Từ 15/11/2026, quota GPU Kaggle của tài khoản là 45 giờ mỗi tuần (D036).

**Không thay đổi:** các mục Quyết định 1–5 của D033, gồm phép kiểm tương đương bắt buộc ở mục 4 trước khi dùng Colab cho run khoa học, quy tắc một loại GPU cho ba seed ở mục 3 và việc giữ registry v1 trên Kaggle T4 ở mục 5.

## 2026-11-18 — D037: Scale2 chạy trước với hạn chót 07/12, chấm test XLS-R trên Colab L4

**Xác nhận:** người dùng chọn phương án C ngày 18/11/2026 (nguyên văn "được tôi duyệt C") và duyệt bản nháp do Claude trình bày trong cuộc trò chuyện, nguyên văn "duyệt nguyên văn".

**Bối cảnh:** registry v1 (9 mục) và registry phụ X1R (3 mục) đã đủ và được Claude kiểm. Các kiểm tra lối tắt (T22, REG-X1R, SC-01, SC-02) không cho chỉ báo mạnh với mô hình XLS-R nào. Kho test-full cục bộ khớp 3/3 SHA (`reports/pre_test_inventory.json`). Chấm XLS-R trên CPU mất khoảng 1,65 giây/mẫu nên không khả thi cho test-full. Ngày 18/11/2026, phép kiểm tương đương D033 mục 4 trên Colab GPU NVIDIA L4 đạt: chênh EER 0,000000 điểm %, 0 mẫu đổi quyết định, chênh score tối đa 0,00002 (`G:\My Drive\DoAnTTNT_colab_outputs\parity_x1_seed2026`, bản sao `.cache/colab_parity/`). Scale2 (D036) chưa chạy.

**Quyết định:**

1. **Thứ tự và hạn chót.** Scale2 chạy trước theo D036. Cổng test được mở khi scale2 hoàn tất, hoặc vào ngày 07/12/2026, tùy mốc nào đến trước, với điều kiện các mục ở mục 6 đã đạt và người dùng mở cổng trong cuộc trò chuyện. Nếu đến 07/12 scale2 chưa đủ ba seed của một mô hình, chỉ các seed đã hoàn tất được đăng ký và chấm test; báo cáo ghi rõ chưa đủ (D036 mục 9). Không bỏ seed, không chọn seed theo test.
2. **Huấn luyện scale2.** X1 ×2 chạy trên Kaggle T4 như D036 mục 6, không dùng Colab để huấn luyện, giữ cùng loại GPU với ×1. B0 ×2 chạy trên CPU cục bộ.
3. **Nơi chấm test.** B0 v2 (×1 và ×2) chấm trên CPU cục bộ qua `scripts/evaluate_full_split.py`. X0, X1, X1R và X1 ×2 chấm trên Colab GPU L4, đúng loại GPU đã qua phép kiểm tương đương, cùng code, pipeline và điều kiện như entrypoint (D032 mục 6, D035, D036 mục 7). Ba seed của cùng một mô hình chấm trên cùng loại GPU. Nếu Colab không cấp L4, chờ hoặc chạy phép kiểm tương đương mới cho loại GPU khác trước khi dùng. Kaggle T4 là phương án dự phòng; khi dùng thì test-full phải được upload lên Kaggle, và việc upload cần người dùng xác nhận riêng.
4. **Luật một lần khi chạy trên Colab.** Trước khi chấm, entrypoint trên máy cục bộ kiểm cổng, tạo khóa cho các mô hình sẽ chấm, rồi ghi một file cấp phép (danh sách mô hình, điều kiện, SHA registry, SHA checkpoint, SHA từng khóa). File cấp phép được đưa lên Drive cùng test-full. Notebook Colab từ chối chạy nếu thiếu file cấp phép, nếu SHA registry, checkpoint hay test-full lệch, hoặc nếu mô hình không có trong danh sách.
5. **Sự cố kỹ thuật.** Nếu phiên Colab dừng trước khi ghi bất kỳ score test nào (lỗi môi trường, mất phiên, hết đơn vị), được chạy lại với cùng file cấp phép, ghi nhận lý do. Score của mỗi mô hình × điều kiện đã ghi xong là cuối cùng và không chấm lại; phần chưa chấm được tiếp tục ở lần chạy sau với cùng cấp phép. Mọi trường hợp khác áp dụng D032 mục 7 (lập quyết định mới).
6. **Điều kiện trước khi mở cổng:**
   - Các registry chính thức (`reports/model_registry_v1.json`, `reports/model_registry_x1r.json`, và `reports/model_registry_scale2.json` với các mục đã hoàn tất) được tạo từ bản nháp đã kiểm, SHA điền vào các config.
   - `t22_claude_verified` và `test_dataset_inventory_checked` bật true.
   - Test-full có trên Drive, SHA khớp bản cục bộ.
   - Notebook chấm test Colab được chạy thử trên dev-full với một mô hình, và tái tạo đạt.
   - Người dùng mở cổng trong cuộc trò chuyện.

**Lý do:** Colab L4 đã qua phép kiểm tương đương, nhanh gấp khoảng hai lần Kaggle T4, và chấm toàn bộ test XLS-R ước tính chỉ khoảng 8–10 đơn vị Colab. Nhờ đó quota Kaggle (khoảng 106 giờ đến 07/12) dành trọn cho scale2. Scale2 trả lời trực tiếp câu hỏi về tác động của quy mô dữ liệu. Hạn chót 07/12 giữ hơn ba tuần để viết báo cáo trước 03/01/2027. Khóa tạo trên máy cục bộ giữ được luật một lần dù việc chấm chạy trên máy tạm.

**Giới hạn:** phép kiểm tương đương chỉ dùng một checkpoint (X1 seed 2026) trên dev-full; tính tương đương của các mô hình khác trên Colab L4 được suy ra, không đo riêng. Nếu scale2 không xong trước 07/12, so sánh ×1 với ×2 thiếu seed.

**Không thay đổi:** D025, D028, D031, D032 (trừ cách thực hiện luật một lần ở mục 4–5), D033, D034, D035, D036 (trừ thứ tự ở mục 1 và nơi chấm test ở mục 3); metadata, split, seed dữ liệu `2026`, `configs/audio.json`. Không mở cổng test.
