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
