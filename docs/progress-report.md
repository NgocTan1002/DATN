# Báo cáo tiến độ đồ án tốt nghiệp

**Đề tài:** Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần

**Sinh viên thực hiện:** Nguyễn Ngọc Tân, MSSV 2251262637, lớp 64TTNT1, Khoa Công nghệ Thông tin, Trường Đại học Thủy Lợi

**Giảng viên hướng dẫn:** PGS.TS Nguyễn Quang Hoan

**Kỳ báo cáo:** 21/11/2026. Số liệu chốt đến ngày 21/11/2026.

**Tính chất tài liệu:** đây là bản thảo làm dần của báo cáo đồ án. Các chương đã có cơ sở được viết ở dạng hoàn chỉnh. Các mục còn phụ thuộc thí nghiệm chưa chạy được đánh dấu rõ và để trống phần kết quả. Mọi chỉ số nhận dạng trong báo cáo đo trên tập phát triển; tập kiểm thử chưa được dùng cho mô hình nào.

---

## Quy ước trạng thái và mức bằng chứng

Báo cáo dùng các mức sau, theo thứ tự bằng chứng tăng dần:

- **Chưa thực hiện**: Không có mã, sản phẩm dữ liệu hoặc lượt chạy đáp ứng tiêu chí.
- **Đã triển khai**: Có mã nguồn hoặc cấu hình; chưa có bằng chứng chạy đúng.
- **Đã kiểm thử**: Có kiểm thử tự động đạt, thường trên dữ liệu kiểm thử tổng hợp hoặc mô hình thu nhỏ. Chưa đồng nghĩa chạy đúng trên dữ liệu thật.
- **Đã kiểm chứng trên dữ liệu thật**: Có lượt chạy trên dữ liệu VSASV hoặc GPU thật, kèm báo cáo đã được lưu.
- **Lịch sử**: Kết quả có thật nhưng đã bị quyết định sau thay thế; chỉ dùng để thảo luận, không dùng làm kết quả chính.

Mọi chỉ số trong báo cáo đều đo trên **tập phát triển mở rộng (dev-full)**. **Tập kiểm thử (test) chưa được dùng** cho mô hình nào; cổng đánh giá cuối vẫn đóng.

---

## Tóm tắt

Đồ án xây dựng hệ thống phân loại nhị phân một đoạn tiếng nói tiếng Việt thành *bonafide* (thật) hoặc *spoof* (giả mạo). Bộ dữ liệu được sử dụng là bản phát hành công khai của VSASV trên Hugging Face. LFCC kết hợp LCNN là mô hình cơ sở (B0). XLS-R 300M đóng băng (X0) là cấu hình đối chiếu, XLS-R tinh chỉnh từng phần (X1) là mô hình chính.

Nền tảng dữ liệu và quy trình đã hoàn tất: chia tập theo người nói, khử trùng nội dung, tập đánh giá mở rộng, tiền xử lý thống nhất, dữ liệu đóng gói đã đối chiếu với dữ liệu gốc, các chỉ số kèm khoảng tin cậy theo người nói, cơ chế huấn luyện tiếp tục chính xác sau gián đoạn, sổ đăng ký mô hình chính thức (9 mục, đóng băng ngày 21/11/2026) cùng sổ phụ cho biến thể X1R (3 mục), kiểm tương đương môi trường GPU Colab L4 và cổng đánh giá cuối khóa một lần (đã thử trên tập phát triển, cổng đang đóng).

Về thực nghiệm, ba nhóm mô hình đã được huấn luyện đủ ba seed (2026, 2027, 2028) và đánh giá trên dev-full (38.777 mẫu, 186 shard), huấn luyện trên 15.885 mẫu:

- **B0 (LFCC–LCNN):** EER trung bình 8,14% ± 0,18 điểm phần trăm. Cả ba seed dừng sớm ở epoch 5 và tốt nhất ở epoch 3.
- **X0 (XLS-R đóng băng):** EER trung bình 4,43% ± 0,32. Cả ba seed còn cải thiện ở epoch 10 (trần cho phép), nên X0 chưa hội tụ.
- **X1 (XLS-R mở 8 tầng cuối, tốc độ học bộ mã hóa 3·10⁻⁵):** EER trung bình 0,77% ± 0,09.

Trên dev-full, XLS-R có EER thấp hơn rõ rệt so với B0, và X1 thấp hơn X0 với khoảng tin cậy 95% theo người nói của hiệu không chứa 0 (seed 2026). Các kết luận này **chưa phải kết quả cuối**: các cấu hình được chọn bằng chính dev-full, nên EER trên tập này có thể lạc quan; kết quả trên tập kiểm thử chưa có.

EER của X1 thấp (dưới 1%), nên đồ án đã chạy một loạt kiểm tra tính hợp lệ. Dữ liệu không có người nói hay nội dung trùng giữa các tập. Toàn bộ mẫu chuyển đổi giọng nói có tần số lấy mẫu gốc 40 kHz, trong khi các nhóm còn lại là 16 kHz, nên phép hạ tần số có thể để lại dấu vết dải cao gắn với nhãn. Phép thử lúc chấm cho thấy B0 phụ thuộc một phần vào dải cao (chỉ báo mạnh ở hai trong ba seed), còn X1 chỉ có chỉ báo yếu. Biến thể X1R, huấn luyện trên âm thanh đã lọc bỏ dải trên 7 kHz, đạt EER trung bình 0,90%, gần với X1. Các kiểm tra này giảm khả năng X1 dựa vào đường tắt tần số lấy mẫu, nhưng không loại trừ các đường tắt khác. Hai kiểm tra bổ sung sau ngày 17/11 mở rộng bức tranh: một bộ phân loại tuyến tính chỉ dùng các đặc trưng thô (khoảng lặng, nền nhiễu, tỉ số dải thấp...) đạt EER 8,76% khi tách bonafide khỏi VC trên dev-full (chỉ báo mạnh ở mức dữ liệu); và phép thử cắt khoảng lặng/thêm nền nhiễu cho thấy B0 nhạy mạnh, còn X1 và X1R ở mức trung bình (mục 4.8).

Thí nghiệm gấp đôi dữ liệu huấn luyện (31.770 mẫu) đã hoàn tất phần chuẩn bị dữ liệu và đang chạy: B0 ×2 huấn luyện trên CPU cục bộ, X1 ×2 trên Kaggle; chưa có kết quả chốt nên báo cáo này không nêu số (mục 2.9 và 4.10). Chương trình minh họa, đánh giá trên tập kiểm thử và phân tích lỗi cuối chưa thực hiện. Theo bộ cổng kiểm soát G0–G7, 5/8 cổng đạt và một cổng đạt một phần (mục 5.2).

---

## MỞ ĐẦU

### 1. Lý do chọn đề tài

Các kỹ thuật tổng hợp tiếng nói (text-to-speech, TTS) và chuyển đổi giọng nói (voice conversion, VC) hiện có thể tạo ra tiếng nói gần với giọng của một người cụ thể. Điều này tạo nguy cơ cho các hệ thống dùng giọng nói làm yếu tố xác thực như tổng đài, ngân hàng hoặc dịch vụ trực tuyến. Bộ phát hiện giả mạo (spoofing countermeasure, CM) là thành phần đứng trước hoặc song song với hệ thống xác thực người nói (automatic speaker verification, ASV), có nhiệm vụ từ chối các đầu vào không phải tiếng nói thật.

Phần lớn nghiên cứu và dữ liệu công khai về chống giả mạo tập trung vào tiếng Anh, tiêu biểu là chuỗi ASVspoof [2], [3]. Với tiếng Việt, VSASV [1] là một trong số ít bộ dữ liệu công khai được thiết kế cho bài toán này. Việc đánh giá có hệ thống các phương pháp phổ biến trên dữ liệu tiếng Việt, theo một giao thức kiểm soát được rò rỉ dữ liệu, vì vậy có giá trị thực tiễn và học thuật.

### 2. Mục tiêu và câu hỏi nghiên cứu

Mục tiêu tổng quát là xây dựng và đánh giá một hệ thống phân loại nhị phân bonafide/spoof cho tiếng nói tiếng Việt, theo quy trình có thể tái lập. Hai câu hỏi nghiên cứu:

- **RQ1.** XLS-R kết hợp bộ phân loại nhị phân có đạt EER thấp hơn mô hình cơ sở LFCC–LCNN trên cùng giao thức dữ liệu hay không?
- **RQ2.** Tinh chỉnh từng phần XLS-R có cải thiện kết quả so với đóng băng toàn bộ bộ mã hóa hay không, và chi phí tính toán tăng bao nhiêu?

Giả thuyết làm việc là X1 đạt EER thấp hơn X0 và B0 trong giới hạn tài nguyên GPU của đồ án. **Trên tập phát triển, số liệu hiện có phù hợp với giả thuyết này (mục 4.7). Giả thuyết chỉ được kết luận sau khi đánh giá trên tập kiểm thử.**

### 3. Đối tượng và phạm vi

**Phần bắt buộc:**

- **Dữ liệu và nhãn:** dữ liệu là bản phát hành công khai đang khảo sát, nhãn 0 = bonafide, 1 = spoof.
- **Gộp nhãn:** chuyển đổi giọng nói (VC), nhiễu đối kháng (adversarial attack, AP) và phát lại (replay) được gộp thành spoof. Trường loại phát ngôn chỉ dùng để thống kê và phân tích lỗi.
- **Chia tập và chuẩn âm thanh:** chia tập theo người nói (speaker-disjoint) với hạt giống sinh số ngẫu nhiên (seed) 2026; âm thanh chuẩn hóa về mono 16 kHz, đoạn 4 giây.
- **Mô hình:** B0 (LFCC–LCNN), X0 (XLS-R đóng băng), X1 (XLS-R tinh chỉnh từng phần).
- **Chỉ số:** EER là chỉ số chính, kèm ROC-AUC và macro-F1.

**Phần tùy chọn:** tinh chỉnh toàn bộ XLS-R, AASIST [4], tăng cường dữ liệu, các tập dữ liệu open-set, dữ liệu ngoài miền.

**Ngoài phạm vi:** phân loại kiểu tấn công, nhận dạng người nói, huấn luyện mô hình nền từ đầu, xây dựng mô hình sinh giọng nói.

**Giới hạn về dữ liệu:** bản phát hành công khai khác đáng kể so với dữ liệu mô tả trong bài báo VSASV. Số mẫu replay chỉ còn khoảng 1,32% so với bài báo. Vì vậy mọi kết luận chỉ áp dụng cho bản phát hành này. Đồ án **không** tuyên bố tái lập giao thức chính thức của VSASV hay đánh giá trên "unseen TTS engines".

### 4. Phương pháp nghiên cứu

- **Nghiên cứu tài liệu:** khảo sát bài toán chống giả mạo, đặc trưng âm học và mô hình tiền huấn luyện tự giám sát.
- **Thực nghiệm có kiểm soát:**
 - Mọi mô hình dùng cùng tập huấn luyện, cùng tiền xử lý và cùng quy tắc chọn trạng thái lưu/ngưỡng trên development.
 - Mọi mô hình huấn luyện đủ ba seed; kết quả báo cáo kèm trung bình và độ lệch chuẩn qua seed.
 - Tập test chỉ được đánh giá một lần, sau khi mọi mô hình đã được đăng ký.
 - Mỗi thay đổi giao thức được ghi lại cùng lý do và phạm vi áp dụng trước khi thực hiện, và mọi tiêu chí kiểm tra quan trọng được đăng ký trước khi biết kết quả.
- **Kỹ thuật phần mềm phục vụ tái lập:** kiểm thử tự động; mã băm kiểm tra toàn vẹn cho dữ liệu, cấu hình và mã nguồn; trạng thái lưu lưu đủ trạng thái để tiếp tục chính xác.
- **Lưu trữ mã nguồn:** [CÁCH LƯU TRỮ MÃ NGUỒN — người dùng điền: ví dụ nộp kèm tệp nén / kho GitHub].

### 5. Đóng góp dự kiến

Đồ án **không** đề xuất kiến trúc học sâu mới. Đóng góp dự kiến gồm:

1. Đánh giá có hệ thống LFCC–LCNN và XLS-R (đóng băng, tinh chỉnh từng phần) cho tiếng nói tiếng Việt trên cùng một giao thức.
2. Một giao thức dữ liệu speaker-disjoint, có khử trùng nội dung và kiểm tra rò rỉ, trên bản phát hành công khai của VSASV.
3. Báo cáo minh bạch các vấn đề chất lượng của bản phát hành, như chênh lệch so với bài báo và nguy cơ mô hình học "đường tắt" (shortcut) từ tần số lấy mẫu hoặc biên độ, kèm các phép thử đã thực hiện để đo mức phụ thuộc.

### 6. Bố cục báo cáo

- **Chương 1:** cơ sở lý thuyết.
- **Chương 2:** dữ liệu và giao thức thực nghiệm.
- **Chương 3:** phương pháp và kiến trúc hệ thống.
- **Chương 4:** các thực nghiệm đã thực hiện và kết quả hiện có.
- **Chương 5:** đánh giá tiến độ, rủi ro và kế hoạch.
- **Phần cuối:** kết luận tạm thời và tài liệu tham khảo.

---

## CHƯƠNG 1. CƠ SỞ LÝ THUYẾT

> Trạng thái chương: bản thảo. Chương này không phụ thuộc kết quả thực nghiệm. Cần bổ sung phần công trình liên quan chi tiết hơn ở bản cuối.

### 1.1. Bài toán phát hiện tiếng nói giả mạo

Cho một đoạn tín hiệu tiếng nói $x$, bộ phát hiện giả mạo tính một điểm số $s(x) \in \mathbb{R}$. Trong đồ án, điểm cao hơn biểu thị bằng chứng giả mạo mạnh hơn. Nhãn dự đoán được xác định bằng ngưỡng $\tau$:

$$\hat{y} = \begin{cases} 1\ (\text{spoof}) & \text{nếu } s(x) \ge \tau \\ 0\ (\text{bonafide}) & \text{nếu } s(x) < \tau \end{cases}$$

ASVspoof chia tấn công thành hai kịch bản [2]:

- **Truy cập logic (logical access, LA):** tiếng nói giả được đưa trực tiếp vào hệ thống, ví dụ VC.
- **Truy cập vật lý (physical access, PA):** tiếng nói được phát lại qua loa và thu lại bằng micro (replay).

### 1.2. Các dạng tấn công trong dữ liệu của đồ án

Bản phát hành VSASV hiện dùng có ba nhóm giả mạo [1]:

- **Chuyển đổi giọng nói (VC):** biến đổi tiếng nói của người nguồn sao cho mang đặc điểm giọng của người đích.
- **Nhiễu đối kháng (AP):** thêm nhiễu được tối ưu có chủ đích để làm sai lệch hệ thống nhận dạng hoặc xác thực.
- **Phát lại (replay):** ghi lại tiếng nói thật rồi phát qua thiết bị.

Đồ án không phân loại riêng ba nhóm này. Chúng được gộp thành một lớp spoof khi huấn luyện, còn loại phát ngôn chỉ dùng để phân tích lỗi theo nhóm.

### 1.3. Bộ dữ liệu liên quan

- **ASVspoof 2019** [2]: bộ dữ liệu quy mô lớn dựa trên VCTK, gồm hai kịch bản LA và PA, có giao thức huấn luyện/phát triển/đánh giá cố định.
- **ASVspoof 2021** [3]: bổ sung điều kiện kênh truyền, codec và nhánh deepfake. Bộ này cho thấy hiệu năng thường giảm mạnh khi điều kiện kiểm thử khác điều kiện huấn luyện.
- **VSASV** [1]: bộ dữ liệu tiếng Việt cho xác thực người nói có xét đến giả mạo. Bản trên Hugging Face [5] có 220.963 bản ghi của 1.141 người nói (mục 2.1).

### 1.4. Đặc trưng LFCC

LFCC (linear-frequency cepstral coefficients) có cách tính tương tự MFCC (Mel Frequency Cepstral Coefficients), nhưng dùng dãy bộ lọc tam giác phân bố **tuyến tính** trên trục tần số thay vì theo thang Mel. Nhờ vậy, vùng tần số cao, nơi thường chứa dấu vết của quá trình tổng hợp, được biểu diễn với độ phân giải ngang vùng tần số thấp [6]. LFCC là đặc trưng của các hệ thống cơ sở trong ASVspoof 2019 [2].

Với khung thời gian $t$, phép biến đổi Fourier thời gian ngắn $X_t(k)$ và $M$ bộ lọc tuyến tính $H_m(k)$, năng lượng dải lọc được tính như sau:

$$E_m(t) = \sum_k |X_t(k)|^2\, H_m(k), \qquad m = 1,\dots,M$$

Năng lượng được đưa về thang log (dB), rồi biến đổi cosin rời rạc loại II (DCT-II) chuẩn hóa trực giao để lấy $N$ hệ số đầu:

$$c_n(t) = \alpha_n \sum_{m=1}^{M} \log E_m(t)\, \cos\!\left(\frac{\pi n (2m-1)}{2M}\right), \qquad n = 0,\dots,N-1$$

trong đó $\alpha_0 = \sqrt{1/M}$ và $\alpha_n = \sqrt{2/M}$ với $n \ge 1$.

Thông số đã khóa để tính LFCC như sau:

- $M = 80$ bộ lọc trên dải 0–8.000 Hz, $N = 60$ hệ số.
- FFT 512 điểm, cửa sổ Hann 400 mẫu (25 ms), bước nhảy 160 mẫu (10 ms).

Với đoạn 4 giây ở 16 kHz, đầu ra có kích thước $60 \times 401$.

Trong cài đặt dùng thư viện torchaudio, bước đổi sang dB có thao tác cắt động (giới hạn dải động 80 dB): mọi giá trị thấp hơn mức cực đại 80 dB bị kẹp lại. Mức cực đại này phải được tính **riêng cho từng mẫu**. Nếu tính chung trên cả lô, đặc trưng của một mẫu sẽ phụ thuộc các mẫu khác. Đây chính là lỗi được phát hiện và sửa trong đồ án (mục 3.2.3, mục 4.4).

### 1.5. Mạng LCNN và cơ chế Max-Feature-Map

LCNN (Light CNN) là mạng tích chập dùng hàm kích hoạt Max-Feature-Map (MFM) thay cho ReLU [7]. MFM chia bản đồ đặc trưng đầu vào gồm $2C$ kênh thành hai nửa, rồi lấy cực đại theo từng phần tử:

$$y_c = \max\left(x_c,\ x_{c+C}\right), \qquad c = 1,\dots,C$$

MFM vừa đóng vai trò hàm phi tuyến, vừa giảm một nửa số kênh, đồng thời hoạt động như một cơ chế chọn đặc trưng cạnh tranh. LCNN dùng đặc trưng LFCC hoặc phổ là một trong những hệ thống mạnh tại ASVspoof 2019 [8]. Vì vậy, LFCC–LCNN là mô hình cơ sở phù hợp cho đồ án.

### 1.6. Học biểu diễn tự giám sát: wav2vec 2.0 và XLS-R

wav2vec 2.0 [9] học biểu diễn tiếng nói trực tiếp từ dạng sóng, không cần nhãn, qua ba khối:

1. **Bộ mã hóa đặc trưng tích chập (CNN feature encoder):** biến dạng sóng 16 kHz thành chuỗi vectơ ẩn, mỗi vectơ ứng với khoảng 20 ms.
2. **Mạng ngữ cảnh Transformer:** tính biểu diễn theo ngữ cảnh của toàn chuỗi.
3. **Mô-đun lượng tử hóa:** tạo mục tiêu rời rạc cho tác vụ tương phản (contrastive). Trong tác vụ này, mô hình phải nhận ra mục tiêu đúng tại các vị trí bị che (masked).

XLS-R [10] áp dụng kiến trúc wav2vec 2.0 ở quy mô đa ngôn ngữ, tiền huấn luyện trên gần nửa triệu giờ tiếng nói công khai của 128 ngôn ngữ. Phiên bản 300M dùng trong đồ án (mô hình XLS-R 300M công khai của Meta) có 24 tầng Transformer, chiều ẩn 1.024. Cấu hình xử lý đầu vào của trạng thái lưu này bật chuẩn hóa từng đoạn, tức mỗi đoạn được chuẩn hóa về trung bình 0, phương sai 1. Các thông tin này được đối chiếu với cấu hình tiền huấn luyện đã cố định phiên bản.

Nhiều nghiên cứu cho thấy dùng mô hình tự giám sát làm bộ trích xuất đặc trưng (front-end) giúp cải thiện đáng kể khả năng tổng quát hóa của bộ phát hiện giả mạo so với đặc trưng thủ công [11], [12].

### 1.7. Đóng băng và tinh chỉnh từng phần

Có ba cách dùng một mô hình tiền huấn luyện cho tác vụ đích:

- **Đóng băng (frozen):** giữ nguyên toàn bộ tham số của bộ mã hóa, chỉ huấn luyện tầng gộp (gộp theo thời gian) và bộ phân loại. Chi phí bộ nhớ thấp, vì không cần lưu gradient cho bộ mã hóa.
- **Tinh chỉnh từng phần (partial fine-tuning):** mở một số tầng Transformer cuối để cập nhật, thường với tốc độ học nhỏ hơn bộ phân loại. Cách này cho phép biểu diễn thích nghi với tác vụ đích mà vẫn giữ phần lớn tri thức đã học, đồng thời giảm nguy cơ quá khớp khi dữ liệu huấn luyện nhỏ.
- **Tinh chỉnh toàn bộ:** cập nhật mọi tham số. Chi phí cao nhất; đây là phần tùy chọn của đồ án.

### 1.8. Chỉ số đánh giá

Coi spoof là lớp dương. Tại ngưỡng $\tau$:

$$\text{FPR}(\tau) = \frac{|\{x \in \text{bonafide}: s(x) \ge \tau\}|}{N_{\text{bonafide}}}, \qquad \text{FNR}(\tau) = \frac{|\{x \in \text{spoof}: s(x) < \tau\}|}{N_{\text{spoof}}}$$

- **EER (Equal Error Rate):** giá trị lỗi tại ngưỡng $\tau^*$ mà $\text{FPR}(\tau^*) = \text{FNR}(\tau^*)$. Trên dữ liệu rời rạc, EER được nội suy tại điểm hai đường cắt nhau. EER càng thấp càng tốt và không phụ thuộc việc chọn ngưỡng trước.
- **ROC-AUC:** diện tích dưới đường ROC, bằng xác suất một mẫu spoof ngẫu nhiên có điểm cao hơn một mẫu bonafide ngẫu nhiên. Càng cao càng tốt.
- **Macro-F1:** trung bình cộng F1 của hai lớp tại một ngưỡng cố định. Chỉ số này phụ thuộc ngưỡng, nên đồ án lấy ngưỡng tại điểm EER **trên development** rồi đóng băng ngưỡng đó cho test và suy luận.
- **Khoảng tin cậy bootstrap theo người nói:** các mẫu của cùng một người nói không độc lập với nhau. Vì vậy khoảng tin cậy được ước lượng bằng cách lấy mẫu lại có hoàn lại **theo người nói**, rồi lấy khoảng phân vị 95% [13].

---

## CHƯƠNG 2. DỮ LIỆU VÀ GIAO THỨC THỰC NGHIỆM

> Trạng thái chương: các mục 2.1 đến mục 2.7 và mục 2.9 mô tả sản phẩm dữ liệu đã tạo và kiểm chứng. Mục 2.8 là nguyên tắc đã khóa.

### 2.1. Bản phát hành dữ liệu và kiểm kê thông tin mô tả

Thông tin mô tả của bản phát hành công khai đang khảo sát đã được kiểm tra cấu trúc. Trong đợt rà soát độc lập ngày 03/11/2026, phép kiểm tra cấu trúc được chạy lại và không phát hiện dòng trùng hoặc giá trị thiếu; số lượng bản ghi và phân bố nhãn được nêu ở Bảng 2.1.

Bảng 2.1 tổng hợp bản phát hành dữ liệu và kiểm kê thông tin mô tả.

**Bảng 2.1 Bản phát hành dữ liệu và kiểm kê thông tin mô tả**

| Thuộc tính | Giá trị |
|---|---:|
| Tổng số bản ghi | 220.963 |
| Số người nói | 1.141 |
| Bonafide | 98.305 |
| Voice conversion | 60.949 |
| Adversarial attack | 60.949 |
| Replay | 760 |
| Dòng trùng / giá trị thiếu | 0 / 0 |

Tác giả bộ dữ liệu đã xác nhận đây là phiên bản duy nhất được phát hành và phần dữ liệu thất lạc không thể khôi phục. Số mẫu VC bằng đúng số mẫu AP là cấu trúc quan sát được; đồ án không suy diễn nguyên nhân.

Một đặc điểm quan trọng của thông tin mô tả là tần số lấy mẫu gốc gắn với loại phát ngôn: toàn bộ mẫu chuyển đổi giọng nói (VC) có tần số lấy mẫu gốc 40 kHz, còn bonafide, nhiễu đối kháng và replay ở 16 kHz. Tỉ lệ này đúng 100% ở cả tập huấn luyện 15.885 mẫu và dev-full. Mọi dạng sóng đều được đưa về 16 kHz, nhưng phép hạ tần số có thể để lại dấu vết gắn với nhãn VC; mức ảnh hưởng được kiểm tra ở mục 4.8.

### 2.2. Giao thức chia tập theo người nói

Giao thức chính gồm ba tập huấn luyện, phát triển và kiểm thử, được chia độc lập theo người nói với seed 2026:

Bảng 2.2 tổng hợp phân bố các tập độc lập theo người nói.

**Bảng 2.2 Phân bố các tập độc lập theo người nói**

| Tập dữ liệu | Mẫu | Người nói | Bonafide | Spoof |
|---|---:|---:|---:|---:|
| Train | 150.999 | 798 | 67.809 | 83.190 |
| Development | 39.116 | 172 | 14.490 | 24.626 |
| Test | 30.848 | 171 | 16.006 | 14.842 |

Để bảo đảm các tập độc lập, từng cặp tập được đối chiếu theo mã người nói và tên tệp; không có người nói hay tệp nào xuất hiện đồng thời ở hai tập. Chia theo người nói ngăn mô hình đạt điểm cao nhờ ghi nhớ đặc điểm giọng của từng người.

### 2.3. Phạm vi âm thanh truy cập được

Bản phát hành chia dữ liệu thành 432 phân mảnh Parquet (shard). Đồ án không tải toàn bộ mà làm việc với các tập con có kiểm soát:

- **Pool huấn luyện:** 68 shard với 34.782 dạng sóng, tổng 10.580.529.098 byte (9,85 GiB). Mỗi shard đã được kiểm tra kích thước, mã băm kiểm tra toàn vẹn, khả năng đọc Parquet và mức khớp với thông tin mô tả.
- **Tập đánh giá mở rộng (mục 2.5):** 265 shard được nạp riêng, cho 69.964 bản ghi phát triển/kiểm thử trước khử trùng.

Mọi kết quả phải ghi rõ số mẫu và số shard thực dùng. Đồ án không gọi phần này là toàn bộ VSASV.

### 2.4. Tập dữ liệu phát triển 20.000 mẫu và khử trùng nội dung

Khi băm lại toàn bộ dạng sóng, gồm cả tần số lấy mẫu và mọi mẫu số thực dấu phẩy động 64 bit, quá trình kiểm tra phát hiện 91 nhóm (182 tệp) bonafide có nội dung giống hệt nhau dù tên tệp khác nhau. Trong đó, 33 nhóm đi qua ranh giới người nói và 30 nhóm đi qua ranh giới tập dữ liệu. Như vậy, việc tách theo tên tệp và người nói là chưa đủ: cùng một bản ghi có thể xuất hiện ở cả train và development, làm EER thấp một cách giả tạo.

Để tránh cùng một nội dung xuất hiện ở hai tập, quy tắc khử trùng được áp dụng như sau:

- Mỗi nhóm nội dung chỉ giữ một đại diện, xác định bằng giá trị nhỏ nhất của mã băm của chuỗi ghép seed 2026 với tên tệp.
- Khử trùng thực hiện trước khi chọn hạn mức GPU.

Kết quả là danh sách phát triển gồm 20.000 mẫu đã khử trùng, gọi là tập phát triển sau khử trùng:

Bảng 2.3 tổng hợp tập dữ liệu phát triển sau khử trùng.

**Bảng 2.3 Tập dữ liệu phát triển sau khử trùng**

| Tập dữ liệu | Tổng | Bonafide | VC | AP | Replay |
|---|---:|---:|---:|---:|---:|
| Phần huấn luyện được chọn | 15.885 | 7.134 | 4.348 | 4.348 | 55 |
| Phần phát triển được chọn (dev cũ) | 4.115 | 1.524 | 1.289 | 1.289 | 13 |

- Danh sách mẫu dùng 66 shard, 184 người nói.
- Phép kiểm tra độc lập đọc và băm lại 20.000/20.000 dạng sóng, cho 20.000 mã băm khác nhau và 0 nhóm trùng đi qua ranh giới huấn luyện/phát triển.
- Tập huấn luyện 15.885 mẫu tương đương khoảng 10,5% tập huấn luyện gốc. Đây là giới hạn quy mô cần nêu khi diễn giải kết quả.

### 2.5. Tập đánh giá mở rộng

Tập phát triển mở rộng dùng để chọn mô hình được gọi là dev-full; tập kiểm thử mở rộng dùng để đánh giá cuối được gọi là test-full. Tập dev cũ chỉ có 4.115 mẫu, trong đó replay chỉ có 13 mẫu và nhóm VC đến từ ít người nói. Kích thước này chưa đủ để chọn mô hình ổn định. Vì vậy, giao thức mới mở rộng tập chọn mô hình và tập đánh giá cuối lên **toàn bộ** tập phát triển gốc và tập kiểm thử gốc sau khử trùng, trong khi **giữ nguyên tập huấn luyện 15.885 mẫu**.

Bảng 2.4 tổng hợp tập đánh giá mở rộng sau khử trùng.

**Bảng 2.4 Tập đánh giá mở rộng sau khử trùng**

| Tập | Dòng gốc | Giữ lại | Bonafide giữ | VC | AP | Replay | Lý do loại (đều là bonafide) |
|---|---:|---:|---:|---:|---:|---:|---|
| dev-full | 39.116 | 38.777 | 14.151 | 12.253 | 12.253 | 120 | 301 trùng nội bộ, 38 trùng với train |
| test-full | 30.848 | 30.698 | 15.856 | 7.361 | 7.361 | 120 | 106 trùng nội bộ, 41 trùng với dev, 3 trùng với train |

Kiểm tra toàn vẹn cho các kết quả sau:

- Mã băm trong từng tập là duy nhất.
- Số trùng chéo huấn luyện–phát triển, huấn luyện–kiểm thử và phát triển–kiểm thử đều bằng 0.
- Không có người nói chung giữa danh sách mẫu huấn luyện, dev-full và test-full.
- Tỷ lệ mất mẫu của mọi nhóm loại phát ngôn đều dưới ngưỡng 5% đã đặt trước. Mất nhiều nhất là bonafide ở dev, 2,34%.

**Giới hạn:** replay chỉ có 120 mẫu của 3 người nói ở mỗi tập, nên kết quả trên nhóm này chỉ được báo cáo riêng và không dùng để kết luận.

### 2.6. Tiền xử lý âm thanh

Mọi mô hình dùng chung một quy trình tiền xử lý đã khóa, được minh họa trong Hình 2.1.

Âm thanh được đọc từ dữ liệu gốc, kiểm tra có đúng một kênh và giá trị hữu hạn, lấy mẫu lại về 16 kHz khi cần; sau đó áp dụng chính sách biên độ và tạo đoạn 4 giây gồm 64.000 mẫu trước khi trích xuất đặc trưng hoặc đưa vào mô hình.

**Hình 2.1 Quy trình tiền xử lý âm thanh**

- **Cắt đoạn:** khi huấn luyện, cắt ngẫu nhiên (cắt ngẫu nhiên) có kiểm soát theo seed và vòng huấn luyện. Khi đánh giá, cắt đoạn giữa (center crop). Tệp ngắn hơn 4 giây được lặp lại rồi cắt (repeat-then-trim).
- **Lý do chuẩn hóa 16 kHz:** thông tin mô tả ghi nhận tần số lấy mẫu gốc khác nhau giữa các nhóm (VC ở 40 kHz, các nhóm còn lại ở 16 kHz). Nếu không chuẩn hóa, mô hình có thể dùng băng thông làm đường tắt để đoán nhãn. Chuẩn hóa không loại bỏ hoàn toàn rủi ro này, vì bộ lọc chống răng cưa khi hạ tần số có thể để lại dấu vết gần 8 kHz; do đó đồ án thực hiện thêm các phép thử ở mục 4.8.
- **Ba chính sách biên độ đã triển khai:**
 - giữ nguyên biên độ: giữ nguyên.
 - chuẩn hóa theo đỉnh: nhân với hệ số $g = 0{,}95 / \max|x|$.
 - chuẩn hóa theo RMS: đưa mức RMS về −25 dBFS, với hệ số $g = \min\left(10^{(-25 - \text{RMS}_{\text{dBFS}})/20},\ 0{,}95/\max|x|\right)$ để không cắt ngọn.
 - Đoạn gần im lặng (RMS < −50 dBFS) không bị khuếch đại.
- **Lý do cần chọn chính sách biên độ:** khảo sát trên 2.558 dạng sóng cho thấy RMS trung vị khác nhau giữa các nhóm, ví dụ −21,29 dBFS với bonafide và −23,91 dBFS với VC. Biên độ vì thế là một nguy cơ đường tắt.

### 2.7. Dữ liệu dạng sóng đóng gói

Các phép đo ban đầu cho thấy đọc và giải mã Parquet chiếm khoảng 99,9% thời gian của mỗi bước huấn luyện B0 trên Kaggle (mục 4.2). Vì vậy dữ liệu được đóng gói để đọc trực tiếp dưới dạng sóng:

- Dạng sóng lưu ở số thực dấu phẩy động 32 bit, **sau** resample 16 kHz và **trước** chính sách biên độ và cắt đoạn.
- Có chỉ mục và mã băm từng mẫu.
- Được đọc bằng cơ chế ánh xạ tệp vào bộ nhớ.

Kết quả đóng gói:

Bảng 2.5 tổng hợp quy mô dữ liệu dạng sóng đóng gói.

**Bảng 2.5 Quy mô dữ liệu dạng sóng đóng gói**

| Artifact | Danh sách mẫu | Mẫu | Shard nguồn | Dung lượng |
|---|---|---:|---:|---:|
| train | danh sách huấn luyện đã khử trùng | 15.885 | 57 | 5,39 GiB |
| dev-full | danh sách phát triển mở rộng | 38.777 | 186 | 12,53 GiB |
| test-full | danh sách kiểm thử mở rộng | 30.698 | 162 | 11,40 GiB |

Shard có thể dùng chung giữa các tập, nên không cộng cột shard thành tổng.

**Trạng thái:** đã đối chiếu dữ liệu đóng gói với dữ liệu gốc, không có sai lệch trên phần đối chiếu được: train 15.885/15.885 mẫu (100%), dev-full 7.035/38.777 mẫu (18,14%) và test-full 4.915/30.698 mẫu (16,01%). Phần còn lại không còn bản gốc cục bộ, được kiểm bằng mã băm lúc nạp. Phép đối chiếu test-full chỉ kiểm dữ liệu, không chạy mô hình hay tính chỉ số nhận dạng.

### 2.8. Nguyên tắc sử dụng tập test

- Tập test không được dùng để chọn mô hình, trạng thái lưu, ngưỡng, chính sách biên độ hay quy mô dữ liệu.
- Đánh giá test diễn ra **một lần**, sau khi mọi mô hình và seed đã được đăng ký. Theo quyết định D037 (18/11/2026), cổng chỉ được mở khi thí nghiệm gấp đôi dữ liệu hoàn tất hoặc đến ngày 07/12/2026, tùy mốc nào đến trước, và khi người dùng quyết định mở; nếu đến hạn mà thiếu seed thì chỉ các seed đã xong được đăng ký và chấm, kèm ghi chú chưa đủ.
- Giao thức chấm đã được đăng ký trước khi đọc bất kỳ dữ liệu test nào: chấm B0, X0 và X1 cuối, mỗi loại đủ ba seed (9 mô hình); dùng trạng thái lưu tốt nhất và ngưỡng đã chọn trên dev-full; báo cáo EER của từng seed cùng trung bình ± độ lệch chuẩn; so sánh X1 với B0 và với X0 bằng bootstrap ghép cặp theo người nói; lặp lại phép thử băng tần; replay chỉ mô tả.
- Mỗi mô hình chỉ được chấm một lần: cổng đánh giá tạo một tệp khóa trước khi mở dữ liệu test và từ chối chạy lại nếu khóa đã tồn tại; nếu phát hiện lỗi sau khi chấm, phải ghi nhận và lập quyết định riêng.
- Đến thời điểm báo cáo, dữ liệu test-full đã được đóng gói, băm lại và đối chiếu; cổng đánh giá và sổ đăng ký đã được thử trên dev-full; cổng vẫn đóng và **chưa có mô hình nào chạy trên test**.
- **Sổ đăng ký chính thức.** Ngày 21/11/2026 sổ chính thức v1 (9 mục: B0, X0, X1 cuối, mỗi loại ba seed; 9/9 mục có mặt và đã được kiểm, `complete = true`) và sổ phụ X1R (3/3 mục) được đóng băng từ bản nháp đã kiểm (`reports/model_registry_v1.md`, `reports/model_registry_x1r.md`). Hai sổ đều ghi rõ chưa chấm test. Các mục thí nghiệm gấp đôi dữ liệu sẽ vào sổ phụ riêng khi hoàn tất.
- **Nơi chấm test.** Chấm XLS-R trên CPU mất khoảng 1,65 giây mỗi mẫu nên không khả thi cho test-full. D037 giao việc chấm X0, X1, X1R và X1 ×2 cho Colab GPU L4, và chấm B0 cho CPU cục bộ. Quyền chấm được khóa bằng tệp cấp phép tạo trên máy cục bộ (gồm mã băm sổ đăng ký, trạng thái lưu và từng khóa một lần), notebook Colab từ chối chạy nếu thiếu tệp này hoặc nếu mã băm lệch.
- **Điều kiện còn lại trước khi mở cổng** (D037 mục 6): bật hai cờ kiểm tra (`t22_claude_verified`, `test_dataset_inventory_checked`), đưa test-full lên Drive và đối chiếu mã băm, chạy thử notebook chấm test trên dev-full, và người dùng mở cổng. Kho test-full cục bộ đã được kiểm kê: 30.698 mẫu / 162 shard, khớp 3/3 mã băm; việc kiểm không đọc nhãn hay nội dung test (`reports/pre_test_inventory.md`).

### 2.9. Dữ liệu huấn luyện gấp đôi (scale2)

Theo quyết định D036 (15/11/2026), thí nghiệm bổ sung đo mức cải thiện khi gấp đôi dữ liệu huấn luyện, từ 15.885 lên 31.770 mẫu `closed_train`. Đây là thí nghiệm bổ sung về quy mô, **không thay thế** kết quả chính ở 15.885 mẫu hay sổ đăng ký v1. Tập mới **chứa trọn 15.885 mẫu hiện tại** và thêm 15.885 mẫu mới, nên hai mức lồng nhau; dev-full và test-full không đổi. Phần dữ liệu này đã hoàn tất ngày 18/11/2026 qua bốn bước: chọn mẫu và tải shard, danh sách mẫu, đóng gói, đối chiếu và đưa lên Kaggle.

Bảng 2.6 tổng hợp dữ liệu huấn luyện gấp đôi.

**Bảng 2.6 Dữ liệu huấn luyện gấp đôi (scale2) so với dữ liệu hiện hành**

| Thuộc tính | Train hiện hành (×1) | Train ×2 (scale2) |
|---|---:|---:|
| Số mẫu | 15.885 | 31.770 |
| Bonafide / VC / AP / replay | 7.134 / 4.348 / 4.348 / 55 | 14.267 / 8.697 / 8.697 / 109 |
| Số người nói | 153 | 225 |
| Shard nguồn dùng | 57 | 89 |
| Dung lượng đóng gói | 5,39 GiB | 11,29 GiB |

Nguồn: `reports/development_manifest_scale2_v1.md`, `reports/packed_artifact_scale2_v1.md`.

- **Tải thêm shard:** 35 shard mới, tổng 6.753.103.437 byte (khoảng 6,29 GiB), không vượt trần 120 shard / 10 GB đã đặt; mỗi shard được kiểm kích thước, mã băm và khả năng đọc (`reports/scale2_download.md`). Thứ tự tải xác định bằng mã băm của seed và tên shard, không dùng `closed_test`.
- **Khử trùng:** ứng viên có nội dung trùng với train, dev-full hoặc test-full bị loại (60 bonafide trùng train, 45 trùng dev, 0 trùng test; 70 trùng nội bộ). Băm lại 31.770/31.770 dạng sóng từ Parquet cho 31.770 mã băm khác nhau; 0 trùng nội bộ, 0 trùng với dev-full và test-full; 0 người nói trùng với tập phát triển; 15.885 mẫu đầu khớp tuyệt đối với kết quả kiểm của train ×1.
- **Đóng gói và đối chiếu:** 15.885 mẫu đầu của dữ liệu đóng gói ×2 trùng từng byte với dữ liệu đóng gói ×1; đối chiếu với Parquet đạt 31.770/31.770 mẫu (100%), 0 lệch siêu dữ liệu và 0 lệch dạng sóng (`reports/packed_parity_train_scale2_v1.md`).
- **Đưa lên Kaggle:** bộ dữ liệu riêng tư gồm 3 tệp (12.130.061.692 byte), mã băm khớp 3/3 tệp khi kiểm trong lượt chạy thử trên Kaggle (`reports/scale2_kaggle_dataset.md`; báo cáo kiểm thử cặp GPU ở mục 4.10).
- **Giới hạn:** mẫu mới được lấy từ một số shard, không phải mẫu ngẫu nhiên đều trên toàn bộ `closed_train`; train ×2 vẫn chỉ khoảng 21% `closed_train`; chỉ có hai điểm quy mô nên chưa đủ vẽ đường cong học.

---

## CHƯƠNG 3. PHƯƠNG PHÁP VÀ KIẾN TRÚC HỆ THỐNG

### 3.1. Kiến trúc tổng thể

Hệ thống được tổ chức thành các tầng có ranh giới rõ ràng:

Hình 3.1 trình bày kiến trúc tổng thể gồm tám tầng: (1) tổ chức dữ liệu và khử trùng; (2) tiền xử lý về 16 kHz, điều chỉnh biên độ và tạo đoạn 4 s; (3) trích xuất đặc trưng và phân loại bằng B0 hoặc X0/X1; (4) điều khiển huấn luyện, dừng sớm và lưu trạng thái; (5) đánh giá EER, ROC-AUC, macro-F1, ngưỡng phát triển và khoảng tin cậy theo người nói; (6) vận hành cục bộ và GPU Kaggle; (7) sổ đăng ký mô hình và cổng đánh giá cuối khóa một lần; (8) suy luận cửa sổ trượt và chương trình minh họa. Tầng (7) đã triển khai và thử trên dev-full; tầng (8) chưa được triển khai.

**Hình 3.1 Kiến trúc tổng thể của hệ thống**

Ba nguyên tắc thiết kế chung:

1. **Xác định danh tính (danh tính lượt chạy) của lượt chạy.** Trạng thái lưu lưu mã băm của danh sách mẫu, cấu hình, seed, chính sách biên độ, kích thước lô, bộ tối ưu và tốc độ học. Khi tiếp tục, bộ điều khiển chạy từ chối nếu bất kỳ thành phần nào lệch.
2. **Kiểm tra sớm và dừng có thông báo.** Bộ đọc dữ liệu kiểm tra schema, nhãn, người nói, shard và tần số lấy mẫu ngay khi đọc. Lỗi được báo bằng thông điệp tiếng Việt thay vì để quá trình huấn luyện tiếp tục với đầu vào sai.
3. **Mọi thay đổi giao thức đều có quyết định ghi lại.** Quyết định sau có thể thay thế quyết định trước (quy tắc huấn luyện lại thay thế một phần đặc tả cũ), nhưng quyết định cũ không bị xóa.

### 3.2. Mô hình cơ sở B0: LFCC–LCNN

#### 3.2.1. Kiến trúc

Bảng 3.1 tổng hợp kiến trúc mô hình cơ sở LFCC–LCNN.

**Bảng 3.1 Kiến trúc mô hình cơ sở LFCC–LCNN**

| Thành phần | Cấu hình |
|---|---|
| Đầu vào | Dạng sóng 64.000 mẫu, 16 kHz, chính sách chuẩn hóa theo đỉnh |
| Đặc trưng | LFCC 60 hệ số × 401 khung (mục 1.4), tính riêng từng mẫu |
| Khối tích chập 1–4 | Tích chập hai chiều với số kênh đầu ra 2×{32, 64, 128, 128}, kích thước nhân {5, 3, 3, 3}, theo sau là MFM |
| Gộp | Gộp cực đại 2×2 sau các khối 1, 2, 3; gộp trung bình thích nghi về 1×1 |
| Bộ phân loại | tầng tuyến tính (128 → 128) → MFM → loại ngẫu nhiên kích hoạt với xác suất 0,2 → tầng tuyến tính (64 → 1) |
| Đầu ra | Một điểm đầu ra trước hàm sigmoid thô cho mỗi đoạn |
| Hàm mất mát | entropy chéo nhị phân tính trực tiếp trên điểm đầu ra trước hàm sigmoid |
| Số tham số huấn luyện | 498.113 |

**Trạng thái:** đã triển khai, đã kiểm thử, đã kiểm chứng trên dữ liệu thật (CPU và GPU Tesla T4).

#### 3.2.2. Cấu hình huấn luyện

Bảng 3.2 tổng hợp cấu hình huấn luyện lại mô hình cơ sở.

**Bảng 3.2 Cấu hình huấn luyện lại mô hình cơ sở**

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | 15.885 mẫu train; chọn trạng thái lưu trên dev-full 38.777 mẫu |
| Bộ tối ưu, tốc độ học | Adam, 0,001 cố định; không lịch điều chỉnh tốc độ học, không tính toán độ chính xác hỗn hợp, không cắt chuẩn gradient |
| Lô, số vòng huấn luyện | Lô 8; tối đa 10 epoch (1.986 bước/epoch) |
| Dừng sớm | Dừng sau 2 lần đánh giá không cải thiện; chỉ coi là cải thiện khi EER giảm nghiêm ngặt; nếu hòa, giữ trạng thái lưu sớm hơn |
| Ngưỡng | Lấy tại điểm EER trên development của trạng thái lưu được chọn |
| Seed huấn luyện | 2026, 2027, 2028 (áp cho khởi tạo, bộ lấy mẫu, cắt ngẫu nhiên); seed dữ liệu cố định 2026 |

Quy tắc chung được chốt ngày 03/11/2026 cho B0 và XLS-R là tối đa 10 epoch, dừng sớm sau 2 lần đánh giá không cải thiện.

**Trạng thái:** ba seed đã hoàn tất trên CPU với dữ liệu đóng gói, dùng 15.885 mẫu train/57 shard và 38.777 mẫu dev-full/186 shard. Lượt chạy đầu giới hạn 3 epoch, sau đó được nối dài từ trạng thái lưu cuối theo quy tắc chung; cả ba seed dừng sớm sau epoch 5 và tốt nhất ở epoch 3. Kết quả ở mục 4.5.

#### 3.2.3. Lỗi LFCC phụ thuộc lô và cách khắc phục

**Nguyên nhân.** Khi thư viện torchaudio chuyển năng lượng dải lọc sang dB, cách sắp xếp đầu vào khiến chiều chứa các mẫu trong lô bị hiểu là chiều kênh. Vì vậy, mức cắt dải động được tính chung cho cả lô thay vì riêng từng mẫu; đặc trưng của một mẫu phụ thuộc các mẫu khác cùng lô, cả khi huấn luyện lẫn đánh giá.

**Ảnh hưởng.** Lỗi này vi phạm một yêu cầu cơ bản: đặc trưng của một mẫu phải chỉ phụ thuộc chính mẫu đó. Nếu không, ngưỡng chọn trên development không chuyển được sang suy luận từng tệp ở chương trình minh họa.

**Khắc phục.**

- Bộ trích xuất LFCC thêm một chiều kênh đơn, để mức cắt được tính riêng cho từng mẫu.
- Kiểm thử hồi quy kiểm tra đặc trưng không đổi khi thay đổi thành phần hoặc thứ tự lô.
- Bản sửa đã được lưu trong lịch sử mã nguồn.

**Trạng thái:** đã triển khai và đã kiểm thử. Ảnh hưởng định lượng của lỗi được trình bày ở mục 4.4.

### 3.3. Mô hình chính X0/X1: XLS-R kết hợp bộ phân loại nhị phân

#### 3.3.1. Kiến trúc dự kiến

Hình 3.2 minh họa kiến trúc dự kiến: đoạn âm thanh 16 kHz, dài 4 s được chuẩn hóa riêng theo công thức $(x − μ)/√(σ² + 1e−7)$ rồi đưa qua XLS-R 300M với 24 tầng Transformer và chiều ẩn 1.024. Gộp trung bình và độ lệch chuẩn theo thời gian tạo vectơ 2.048 chiều; bộ phân loại gồm tầng tuyến tính 2.048 → 128, ReLU, loại ngẫu nhiên kích hoạt với xác suất 0,2 và tầng tuyến tính 128 → 1 để tạo điểm đầu ra trước hàm sigmoid.

**Hình 3.2 Kiến trúc dự kiến của mô hình XLS-R**

Bảng 3.3 tổng hợp phần tham số được cập nhật trong X0 và X1.

**Bảng 3.3 Phần tham số được cập nhật trong X0 và X1**

| Cấu hình | Phần được cập nhật | Ghi chú |
|---|---|---|
| X0 | Chỉ bộ phân loại | Bộ mã hóa chạy trong chế độ không tính gradient, không lưu gradient |
| X1 | $k$ tầng Transformer cuối và bộ phân loại | Hai nhóm tốc độ học riêng cho bộ mã hóa và bộ phân loại; đã thử $k$ thuộc {2, 4, 8}, cấu hình cuối $k = 8$ |

Chuẩn hóa đầu vào theo bộ xử lý tiền huấn luyện thay thế thiết kế so sánh hai nhánh chuẩn hóa trước đây. Lý do là cấu hình tiền huấn luyện đã cố định có bật chuẩn hóa từng đoạn, nên dùng đúng phân phối đầu vào lúc tiền huấn luyện là lựa chọn mặc định dễ biện luận. Đây là lựa chọn có lý do, **không** phải giá trị được tối ưu bằng thực nghiệm.

#### 3.3.2. Cấu hình huấn luyện và trạng thái triển khai

Bảng 3.4 tổng hợp cấu hình huấn luyện XLS-R.

**Bảng 3.4 Cấu hình huấn luyện XLS-R**

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | 15.885 mẫu train; chọn trạng thái lưu và ngưỡng trên dev-full 38.777 mẫu |
| Bộ tối ưu | Adam hai nhóm tham số, tốc độ học cố định, không lịch điều chỉnh; số thực 32 bit, không tính toán độ chính xác hỗn hợp, không cắt chuẩn gradient |
| Tốc độ học | Bộ phân loại 10⁻³; bộ mã hóa 10⁻⁵ hoặc 3·10⁻⁵ (chỉ X1); X0 chỉ cập nhật bộ phân loại |
| Số tầng mở | 2, 4 hoặc 8 tầng Transformer cuối; X1 cuối mở 8 tầng, tốc độ học bộ mã hóa 3·10⁻⁵ |
| Bộ phân loại | 128 đơn vị ẩn, xác suất loại ngẫu nhiên 0,2 |
| Lô và số vòng | Lô 16; tối đa 10 epoch; dừng sau 2 lần đánh giá không cải thiện; chọn theo EER dev-full, ngưỡng tại điểm EER của trạng thái được chọn |
| Seed huấn luyện | 2026, 2027, 2028 |
| Chuẩn hóa đầu vào | Chuẩn hóa từng đoạn theo bộ xử lý của mô hình tiền huấn luyện |
| Phần cứng | Kaggle, một GPU Tesla T4 cho mỗi lượt chạy |

Trong huấn luyện, bộ mã hóa luôn ở chế độ đánh giá. Cấu hình tiền huấn luyện công bố có bật kỹ thuật bỏ tầng ngẫu nhiên (LayerDrop) và che phổ (SpecAugment), nhưng trong cách dùng này hai kỹ thuật đó thực tế tắt; chỉ các tầng được mở và bộ phân loại ở chế độ huấn luyện. Cấu hình khai báo và cấu hình hiệu lực được ghi riêng trong từng lượt chạy.

Thứ tự chạy đã đăng ký trước: seed 2026 với ba mức tầng ở tốc độ học bộ mã hóa 10⁻⁵ và X0; mức tầng tốt nhất theo dev-full được thử thêm tốc độ học 3·10⁻⁵; sau đó cấu hình X1 cuối và X0 chạy thêm hai seed 2027 và 2028. Khi hòa EER, ưu tiên ít tầng hơn; khi chọn tốc độ học ở mức tầng đã chọn, ưu tiên tốc độ học thấp hơn nếu hòa. Test không tham gia lựa chọn. Thứ tự này đã được thực hiện đầy đủ (mục 4.7).

Bảng 3.5 tổng hợp trạng thái triển khai XLS-R và giới hạn bằng chứng.

**Bảng 3.5 Trạng thái triển khai XLS-R và giới hạn bằng chứng**

| Hạng mục | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| Khung mô hình, đóng băng/mở tầng, gộp theo thời gian, nhóm tốc độ học | Đã triển khai, đã kiểm thử, đã kiểm chứng trên GPU | Chạy trên dữ liệu thật cho X0, X1 và X1R |
| Trọng số tiền huấn luyện | Đã tải, kiểm mã băm, tải offline | 1.269.737.156 byte; đoạn 4 giây cho đầu ra 199 khung × 1.024 chiều |
| Chương trình huấn luyện khoa học cho X0/X1 | Đã triển khai, đã kiểm chứng trên GPU Kaggle | Hỗ trợ ba seed, dừng sớm, lưu và tiếp tục chính xác |
| Trạng thái lưu tiếp tục chính xác | Đã kiểm chứng trên GPU | Được dùng thực tế khi phiên GPU hết hạn (X1 seed 2027, X1R seed 2026) |
| Hai lượt chạy độc lập song song trên hai GPU | Đã kiểm chứng | Mỗi lượt giữ nguyên cấu hình và quy tắc dừng riêng |
| Kết quả nhận dạng | Có trên dev-full | Mục 4.7; chưa có kết quả test |

**Nhận xét:** XLS-R với trọng số tiền huấn luyện đã được huấn luyện đủ ba seed trên dữ liệu thật và có kết quả trên dev-full. Chưa có kết quả trên tập kiểm thử.

### 3.4. Huấn luyện có thể tiếp tục chính xác

Phiên GPU trên Kaggle giới hạn thời gian. Một lượt huấn luyện B0 vì thế phải chia thành nhiều phiên, mỗi phiên tối đa 512 bước tối ưu. Để kết quả không phụ thuộc việc bị chia phiên, hệ thống được thiết kế như sau:

- **Nội dung trạng thái lưu:** tham số mô hình, trạng thái bộ tối ưu, số bước, vòng huấn luyện, vị trí trong vòng huấn luyện (vị trí đọc tiếp), trạng thái bộ lấy mẫu, trạng thái sinh số ngẫu nhiên của Python/NumPy/PyTorch/CUDA, lịch sử đánh giá, bộ đếm số lượt không cải thiện cho phép và danh tính lượt chạy.
- **Kho trạng thái lưu theo thế hệ lưu:** mỗi lần lưu tạo một thế hệ lưu mới, ghi nguyên tử; bản chỉ dẫn ghi vị trí trạng thái mới nhất và mô hình tốt nhất, kèm mã băm.
- **Bộ lấy mẫu hỗ trợ tiếp tục sau gián đoạn:** tái tạo đúng thứ tự lô từ seed và vị trí đọc tiếp.

Để kiểm tra việc chia phiên không làm thay đổi kết quả, chạy liên tục được đối chiếu với chạy dừng–tiếp tục trên CPU và cho 12/12 tensor mô hình trùng tuyệt đối. Trên GPU, kiểm tra sau từng phiên trong 11 phiên hoàn tất của chuỗi B0 (mục 4.3) xác nhận mọi thế hệ lưu nhập vào giữ nguyên từng byte và chuỗi hàm mất mát trùng với phiên trước. Cơ chế này cũng được dùng trong thực tế cho XLS-R khi phiên GPU hết hạn trước lúc huấn luyện xong.

### 3.5. Hạ tầng tính toán và quy trình vận hành

- **Máy cục bộ (CPU):** phát triển, kiểm thử, xử lý dữ liệu, đóng gói sản phẩm dữ liệu và huấn luyện B0 v2. Từ tối 20/11/2026 máy này cũng chạy B0 v2 ×2 (scale2): ba seed song song, mỗi seed một tiến trình, do một bộ điều phối khởi chạy theo từng đợt 512 bước (`reports/b0_v2_scale2_launch.md`). Lượt đầu đạt 1,23 giây mỗi bước khi ba seed chạy song song, chậm 1,147 lần so với phép đo một luồng; phép chiếu của lượt đầu là khoảng 1,94 giờ mỗi epoch cho mỗi seed. Đây là phép chiếu, không phải kết quả.
- **Kaggle (GPU chính):** Tesla T4. Mỗi phiên cấp hai GPU; hệ thống chạy tối đa hai lượt huấn luyện độc lập song song, mỗi lượt chỉ dùng một GPU, số thực 32 bit, tắt chế độ tăng tốc TF32. Hạn mức GPU là 45 giờ mỗi tuần từ 15/11/2026 (trước đó 30 giờ). Với X1 ×2, chế độ cặp hai GPU đã được kiểm bằng một lượt chạy thử ngày 19/11/2026 (hai GPU Tesla T4, mã băm dữ liệu scale2 khớp 3/3 tệp, không dùng test, trạng thái hoàn tất; `W/reports/xlsr_x1s02_pair_dry_run.md`). Sau đó một phiên huấn luyện thật cho seed 2026 và 2027 dùng 10,99 giờ quota và tạm dừng có kiểm soát để tiếp tục ở phiên sau (mục 4.10).
- **Colab (gói Google AI Pro, 200 đơn vị điện toán):** môi trường bổ sung; chỉ dùng cho lượt chạy khoa học sau khi kiểm tương đương với Kaggle trên dev-full. Các mô hình của sổ đăng ký chính vẫn được huấn luyện trên Kaggle T4. Ngày 18/11/2026 phép kiểm tương đương trên GPU NVIDIA L4 đạt: chấm lại 38.777 mẫu dev-full với checkpoint X1 cuối seed 2026 cho chênh EER 0,000000 điểm phần trăm (ngưỡng cho phép 0,02), 0 mẫu đổi quyết định (cho phép tối đa 10) và chênh điểm số tối đa 0,00002 (cho phép 0,05); phép kiểm không dùng dữ liệu test (`.cache/colab_parity/reports/xlsr_parity_X1-L8-lr3e-5_seed2026.md`, hướng dẫn `docs/guides/colab_parity_check.md`). Vì vậy D037 cho phép dùng Colab L4 để **chấm** test XLS-R, không dùng để huấn luyện. Giới hạn: phép kiểm chỉ dùng một checkpoint; tính tương đương của các mô hình khác được suy ra, không đo riêng.
- **Quy trình vận hành bằng dòng lệnh** gồm kiểm tra tài khoản, đóng gói mã nguồn xác định có bản kê mã băm từng tệp, tạo bộ dữ liệu riêng tư, khởi chạy phiên GPU, theo dõi trạng thái và tải kết quả.
- **Kiểm soát sai lệch mã nguồn:** bộ điều khiển chạy trên Kaggle từ chối chạy nếu mã nguồn thực tế không khớp mã băm đã khai báo. Sau khi một phiên lỗi trước huấn luyện vì bộ chạy chứa mã nguồn sai phiên, hệ thống bổ sung bước so khớp bộ chạy với cấu hình thực thi.
- **Sao lưu:** toàn bộ dữ liệu, mã nguồn và kết quả được sao lưu lên ổ đĩa đám mây; số tệp và dung lượng được kiểm khớp, và đã thử khôi phục trọng số từ bản sao lưu.

### 3.6. Kiểm thử và kiểm soát chất lượng

- Bộ kiểm thử đơn vị bao phủ: cấu trúc dữ liệu, chia đoạn, chính sách biên độ, tách tập, khử trùng, bộ đọc dữ liệu, LFCC (gồm tính độc lập theo lô), LCNN, XLS-R, chỉ số, bootstrap (cả ghép cặp), trạng thái lưu, tiếp tục chính xác, bộ điều khiển B0 v2 và XLS-R, đóng gói và đối chiếu tính tương đương, sổ đăng ký mô hình, cổng đánh giá cuối, quy trình Kaggle.
- Lần chạy toàn phần gần nhất trên nhánh chính (20/11/2026, lúc chuẩn bị B0 v2 ×2) đạt **510/510** kiểm thử (`reports/b0_v2_scale2_prep.md`); nhánh làm việc XLS-R đạt 456/456 (19/11/2026). Cổng đánh giá cuối cũng được thử (15/11/2026) và từ chối chạy đúng như thiết kế vì cổng đang đóng, không mở dữ liệu test.
- **Phân biệt mức bằng chứng:** nhiều kiểm thử dùng dữ liệu kiểm thử tổng hợp. Các kiểm thử này chứng minh logic đúng, **không** chứng minh lượt chạy trên dữ liệu thật đã đạt; mức bằng chứng của lượt chạy thật nằm ở Chương 4.
- **Tái lập:** các thay đổi mã nguồn đã được ghi nhận phiên bản; dữ liệu đã xử lý, trạng thái lưu và kết quả lớn nằm ngoài kho mã và được sao lưu riêng.

### 3.7. Ma trận trạng thái thành phần

Bảng 3.6 tổng hợp trạng thái các thành phần hệ thống.

**Bảng 3.6 Trạng thái các thành phần hệ thống**

| Thành phần | Đã triển khai | Đã kiểm thử | Kiểm chứng dữ liệu thật |
|---|:---:|:---:|:---:|
| Kiểm kê thông tin mô tả, chia tập, kiểm tra rò rỉ | ✓ | ✓ | ✓ |
| Khử trùng nội dung, danh sách tập phát triển sau khử trùng | ✓ | ✓ | ✓ |
| Danh sách mẫu dev-full / test-full | ✓ | ✓ | ✓ |
| Tiền xử lý âm thanh, ba chính sách biên độ | ✓ | ✓ | ✓ |
| Dữ liệu đóng gói và đối chiếu với Parquet | ✓ | ✓ | ✓ (trong phạm vi còn bản gốc) |
| Chỉ số EER, ROC-AUC, macro-F1, ngưỡng dev | ✓ | ✓ | ✓ |
| Bootstrap theo người nói, bootstrap ghép cặp | ✓ | ✓ | ✓ (dev-full) |
| B0 với LFCC cũ (phụ thuộc lô) | ✓ | ✓ | ✓ (lịch sử) |
| B0 v2 với LFCC tính riêng từng mẫu, ba seed | ✓ | ✓ | ✓ (dev-full) |
| XLS-R: X0, X1 cuối, ba seed | ✓ | ✓ | ✓ (dev-full) |
| XLS-R: X1R, ba seed | ✓ | ✓ | ✓ (dev-full; đã đăng ký sổ phụ) |
| Trạng thái lưu tiếp tục chính xác | ✓ | ✓ | ✓ (CPU, GPU) |
| Phân tích phổ và phép thử băng tần | ✓ | ✓ | ✓ (dev-full) |
| Sổ đăng ký chính thức v1 (9 mục) và sổ phụ X1R (3 mục), cổng đánh giá cuối khóa một lần | ✓ | ✓ | ✓ (đóng băng 21/11; thử trên dev-full; cổng đóng) |
| Kiểm tương đương Colab L4 (chấm XLS-R) | ✓ | ✓ | ✓ (dev-full, một checkpoint) |
| Probe lối tắt mức dữ liệu và mô hình (SC-01, SC-02) | ✓ | ✓ | ✓ (dev-full) |
| Dữ liệu huấn luyện ×2 (danh sách mẫu, đóng gói, đối chiếu) | ✓ | ✓ | ✓ |
| B0 v2 ×2 và X1 ×2 (huấn luyện) | ✓ | ✓ | Đang chạy, chưa có kết quả chốt |
| Suy luận cửa sổ trượt, chương trình minh họa | — | — | — |

---

## CHƯƠNG 4. THỰC NGHIỆM VÀ KẾT QUẢ HIỆN CÓ

> Mọi kết quả trong chương đo trên **development**, không có kết quả test. Các mục 4.2 đến 4.4 mang nhãn **Lịch sử — LFCC phụ thuộc lô** do đặc trưng cũ bị lỗi. Chúng không phải kết quả chính thức của mô hình cơ sở; kết quả chính thức của B0 nằm ở mục 4.5.

### 4.1. Thiết lập chung

Bảng 4.1 tổng hợp thiết lập chung của các thí nghiệm lịch sử.

**Bảng 4.1 Thiết lập chung của các thí nghiệm lịch sử**

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | tập phát triển sau khử trùng: 15.885 train, 4.115 dev cũ, 66 shard |
| Seed | 2026 |
| Mô hình | LFCC–LCNN (mục 3.2.1) |
| Huấn luyện | Lô 8, không worker phụ, Adam, tốc độ học 0,001 |
| Phần cứng | Kaggle, một GPU Tesla T4 |

### 4.2. Thí nghiệm 1: chọn chính sách biên độ (ba pilot)

**Mục đích:** chọn một trong ba chính sách giữ nguyên biên độ, chuẩn hóa theo đỉnh, chuẩn hóa theo RMS bằng EER trên development.

**Thiết kế:**

- Ba lượt chạy chỉ khác nhau ở chính sách biên độ.
- Mỗi lượt có cùng ngân sách 896 bước tối ưu (7.168 lượt mẫu train, chưa hết một vòng huấn luyện). Ngân sách này được xác định từ phép đo hiệu năng trên Kaggle.
- Không dừng sớm; đánh giá đủ 4.115 mẫu development.

Kết quả đo trên cùng tập phát triển được trình bày dưới đây.

Bảng 4.2 tổng hợp so sánh ba chính sách biên độ trên tập phát triển (lịch sử).

**Bảng 4.2 So sánh ba chính sách biên độ trên tập phát triển (lịch sử)**

| Chính sách | Hàm mất mát dev | EER (%) | ROC-AUC | Macro-F1 (%) | Thời gian (giờ) |
|---|---:|---:|---:|---:|---:|
| Giữ nguyên biên độ | 0,4011 | 10,04 | 0,9715 | 89,36 | 4,81 |
| **Chuẩn hóa theo đỉnh** | **0,3268** | **9,12** | 0,9701 | **90,33** | 4,95 |
| Chuẩn hóa theo RMS | 0,3769 | 11,55 | 0,9569 | 87,79 | 4,84 |

**Quyết định:** chọn chuẩn hóa theo đỉnh, là chính sách có EER thấp nhất.

**Diễn giải và giới hạn:**

- chuẩn hóa theo đỉnh có EER và hàm mất mát tốt nhất, nhưng ROC-AUC của giữ nguyên biên độ cao hơn. Vì vậy không thể nói chuẩn hóa theo đỉnh tốt hơn trên mọi chỉ số.
- Khoảng tin cậy 95% bootstrap theo người nói của hiệu EER giữ nguyên biên độ trừ chuẩn hóa theo đỉnh là [−3,1; +3,3] điểm phần trăm (giao thức hiện hành). Như vậy, **khác biệt giữa các chính sách không có ý nghĩa thống kê** ở quy mô này. Lựa chọn chuẩn hóa theo đỉnh là lựa chọn vận hành theo quy tắc đã khóa trước, không phải bằng chứng chuẩn hóa theo đỉnh vượt trội.
- Mỗi chính sách chỉ chạy một seed và chưa đủ một vòng huấn luyện.
- Cả ba pilot mang lỗi LFCC phụ thuộc lô. Mức cắt chung trong lô có thể tác động khác nhau lên ba chính sách. Quy tắc sau khi phát hiện lỗi vẫn giữ chuẩn hóa theo đỉnh và không chạy lại pilot, vì khoảng tin cậy vốn đã không phân biệt được các chính sách.
- Phân tích tài nguyên cho thấy thời gian nạp dữ liệu chiếm khoảng 99,9% tổng thời gian (nạp dữ liệu + mô hình). Đây là lý do chuyển sang dữ liệu đóng gói (mục 2.7).

### 4.3. Thí nghiệm 2: huấn luyện B0 đầy đủ theo đặc tả đã khóa (lịch sử)

**Thiết kế:** chính sách chuẩn hóa theo đỉnh, khởi tạo mới, tối đa 3 vòng huấn luyện, cho phép 1 vòng đánh giá không cải thiện trước khi dừng, đánh giá toàn bộ dev cũ sau mỗi vòng huấn luyện. Lượt chạy gồm 11 phiên GPU Kaggle hoàn tất; một phiên lỗi trước khi huấn luyện do bộ chạy chứa mã nguồn sai phiên và đã được chạy lại.

Hai lần đánh giá đều dùng đủ 4.115 mẫu phát triển cũ; kết quả vẫn mang nhãn lịch sử.

Bảng 4.3 tổng hợp kết quả huấn luyện mô hình cơ sở cũ (lịch sử).

**Bảng 4.3 Kết quả huấn luyện mô hình cơ sở cũ (lịch sử)**

| Điểm đánh giá | Bước | EER (%) | Hàm mất mát dev | ROC-AUC | Macro-F1 (%) | Nguồn |
|---|---:|---:|---:|---:|---:|---|
| Cuối vòng huấn luyện 1 | 1.986 | 8,60 | 0,6780 | 0,9702 | 90,89 | phiên 4 |
| Cuối vòng huấn luyện 2 (tốt nhất) | 3.972 | 8,33 | 0,6344 | 0,9751 | 91,17 | phiên 10 |

**Diễn biến:**

- Vòng huấn luyện 2 cải thiện EER, nên theo quy tắc dừng sớm, huấn luyện được tiếp tục sang vòng huấn luyện 3.
- Chuỗi dừng ở bước 5.632, còn 326 bước nữa mới đến cuối vòng huấn luyện 3. Lý do dừng là đợt rà soát xác định chuỗi này mang lỗi LFCC.
- Không có đánh giá cuối vòng huấn luyện 3. Trạng thái lưu best vẫn là trạng thái lưu ở bước 3.972.

**Diễn giải:**

- Hàm mất mát trên development **tăng** từ 0,33 (pilot chuẩn hóa theo đỉnh, bước 896) lên 0,63–0,68 ở các vòng huấn luyện sau, trong khi EER giảm. Điều này có thể do mô hình trở nên quá tự tin trên các mẫu bị phân loại sai. Đây chỉ là nhận định cần kiểm tra lại trên B0 v2, chưa phải kết luận.
- Kết quả **không** được dùng làm kết quả chính thức của mô hình cơ sở.

Để kiểm chứng việc tiếp tục huấn luyện qua nhiều phiên GPU, trạng thái mô hình, bản chỉ dẫn và mã băm được kiểm tra sau mỗi phiên; chuỗi chạy cũ cung cấp bằng chứng kỹ thuật cho cơ chế này.

### 4.4. Thí nghiệm 3: đo ảnh hưởng của lỗi LFCC phụ thuộc lô

**Mục đích:** định lượng mức sai lệch do lỗi gây ra, phục vụ phần thảo luận.

**Thiết kế:**

- Dùng trạng thái lưu tốt nhất của mục 4.3, chấm 4.115 mẫu dev cũ theo hai cách: (a) theo lô 8 như khi huấn luyện, và (b) riêng từng mẫu bằng bộ trích xuất LFCC riêng từng mẫu.
- Cách (a) tái tạo đúng điểm số đã lưu, sai lệch tối đa 1,1·10⁻⁵, xác nhận phép đo được dựng lại chính xác.

Kết quả đo trên cùng tập phát triển được trình bày dưới đây.

Bảng 4.4 tổng hợp ảnh hưởng của lỗi LFCC phụ thuộc lô.

**Bảng 4.4 Ảnh hưởng của lỗi LFCC phụ thuộc lô**

| Chỉ số | Chấm theo lô (cũ) | Chấm riêng từng mẫu |
|---|---:|---:|
| EER toàn tập | 8,33% | 8,40% |
| EER nhóm AP | 3,35% | 3,74% |
| EER nhóm VC | 11,95% | 12,14% |
| EER nhóm replay (13 mẫu) | 7,69% | 7,69% |

- **Số điểm số thay đổi:** 896 mẫu có điểm thay đổi hơn 10⁻⁴, gồm 233 bonafide, 656 VC và 7 replay. Thay đổi lớn nhất là 0,946; trung vị là 1,19·10⁻⁷.
- **Số quyết định bị đảo:** 31 mẫu đổi nhãn dự đoán tại ngưỡng cũ 0,007008 (20 bonafide, 11 VC).
- **Hiệu EER:** chấm riêng trừ chấm theo lô được +0,066 điểm, khoảng tin cậy 95% theo người nói là [−0,436; +1,634] (999/1.000 lần lấy mẫu hợp lệ).

**Diễn giải:**

- Ở mức toàn tập, lỗi làm thay đổi EER nhỏ, và khoảng tin cậy bao gồm 0.
- Tuy vậy, lỗi làm một số điểm số thay đổi rất lớn (tới 0,946) và đảo 31 quyết định. Như vậy lỗi **không** vô hại ở mức từng mẫu, vốn là mức mà chương trình minh họa hoạt động.
- Mô hình đã được huấn luyện trên đặc trưng bị nhiễu, nên số 8,40% **không** phải hiệu năng của một mô hình huấn luyện đúng. Đó là lý do chọn huấn luyện lại (B0 v2) thay vì chỉ sửa ở bước đánh giá.
- **Giới hạn:** phép đo chỉ dùng một trạng thái lưu và dev cũ (31 người nói; VC chỉ đến từ 3 người nói).

### 4.5. B0 v2: kết quả chính thức của mô hình cơ sở

B0 v2 dùng đặc trưng LFCC đã sửa lỗi, chạy trên CPU với dữ liệu đóng gói. Quy tắc huấn luyện chung cho B0 và XLS-R là tối đa 10 epoch, dừng sớm sau 2 lần đánh giá không cải thiện, chọn trạng thái lưu và ngưỡng phân loại theo EER trên dev-full; mỗi seed có ngưỡng riêng. Mỗi mô hình huấn luyện trên 15.885 mẫu của 57 shard và đánh giá trên 38.777 mẫu của 186 shard dev-full (172 người nói).

Ba seed ban đầu chạy với giới hạn 3 epoch, sau đó được nối dài từ trạng thái cuối theo quy tắc chung. Ở cả ba seed, epoch 4 và epoch 5 đều không cải thiện EER, nên huấn luyện dừng sớm sau epoch 5 và trạng thái tốt nhất vẫn là epoch 3. Kết quả chính thức vì vậy trùng với kết quả ở giới hạn 3 epoch. Các chỉ số được tính lại độc lập từ điểm số đã lưu và khớp kết quả gốc.

Bảng 4.5 tổng hợp kết quả B0 v2 theo epoch và seed.

**Bảng 4.5 Kết quả B0 v2 trên dev-full theo epoch và seed**

| Seed | EER epoch 1 | EER epoch 2 | EER epoch 3 | EER epoch 4 | EER epoch 5 | Tốt nhất | ROC-AUC | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 13,91% | 10,81% | **8,23%** | 11,38% | 8,98% | epoch 3 | 0,973 | 91,23% |
| 2027 | 13,03% | 10,67% | **8,26%** | 9,91% | 10,27% | epoch 3 | 0,973 | 91,20% |
| 2028 | 13,96% | 13,34% | **7,93%** | 8,20% | 10,01% | epoch 3 | 0,976 | 91,55% |
| Trung bình ± độ lệch chuẩn | | | 8,14% ± 0,18 | | | | 0,974 ± 0,002 | 91,33% ± 0,19 |

Độ lệch chuẩn là độ lệch chuẩn mẫu (mẫu số n−1) qua ba seed, tính bằng điểm phần trăm; đây không phải khoảng tin cậy theo người nói. Hình 4.1 minh họa diễn biến EER theo epoch.

**Hình 4.1 Diễn biến EER của B0 v2 theo epoch trên dev-full**

Có ba nhận xét. Thứ nhất, EER dao động mạnh giữa các epoch liền kề của cùng một seed (ví dụ seed 2026 tăng từ 8,23% lên 11,38% rồi giảm xuống 8,98%), lớn hơn nhiều so với độ lệch chuẩn qua seed. Vì trạng thái tốt nhất được chọn bằng chính dev-full, EER được báo cáo có thể hơi lạc quan so với hiệu năng thực trên dữ liệu mới. Thứ hai, khoảng tin cậy 95% theo người nói của EER còn rộng vì dev-full chỉ có 172 người nói: [4,96%; 12,73%] cho seed 2026, [5,05%; 12,47%] cho seed 2027 và [5,32%; 11,42%] cho seed 2028 (1.000 lần lấy mẫu lại). Các khoảng này có điều kiện trên trạng thái đã chọn bằng dev-full. Thứ ba, hàm mất mát không được báo cáo cho B0 v2 vì chỉ lưu xác suất sau hàm sigmoid, bão hòa gần 0 và 1, nên không kiểm lại chính xác được.

Bảng 4.6 và Hình 4.2 tổng hợp EER theo kiểu tấn công; mỗi nhóm dùng chung tập bonafide và ngưỡng EER riêng của nhóm.

**Bảng 4.6 EER của B0 v2 theo kiểu tấn công trên dev-full**

| Nhóm spoof | Số mẫu spoof | Seed 2026 | Seed 2027 | Seed 2028 | Trung bình ± độ lệch chuẩn |
|---|---:|---:|---:|---:|---:|
| Chuyển đổi giọng nói | 12.253 | 11,51% | 11,17% | 11,47% | 11,38% ± 0,19 |
| Nhiễu đối kháng | 12.253 | 2,68% | 2,52% | 1,69% | 2,29% ± 0,53 |
| Phát lại (không đủ để kết luận) | 120 | 5,51% | 11,24% | 10,83% | 9,19% ± 3,19 |

**Hình 4.2 EER của B0 v2 theo kiểu tấn công trên dev-full**

Chuyển đổi giọng nói là nhóm khó nhất: tại ngưỡng riêng của từng seed, khoảng 15,8% đến 16,4% mẫu VC bị bỏ lọt, trong khi chỉ 0,03% đến 0,14% mẫu nhiễu đối kháng bị bỏ lọt. Nhóm replay chỉ có 120 mẫu spoof của 3 người nói nên dao động giữa các seed lớn và **không đủ để kết luận**. Mức dễ của nhóm nhiễu đối kháng cần đọc cùng phân tích phổ ở mục 4.8, vì nhóm này có năng lượng dải cao cao hơn hẳn bonafide.

### 4.6. Đo chi phí XLS-R trên GPU

Trước khi chốt cấu hình, chi phí XLS-R được đo trên một GPU Tesla T4; 12/12 cấu hình (0, 1, 2, 4, 8, 12 tầng mở, lô 8 và 16) chạy được, không có lỗi thiếu bộ nhớ. Mỗi cấu hình có 5 bước khởi động, 30 bước đo và 512 mẫu dev thuộc 160 shard. Đây là phép đo chi phí và phép chiếu, không phải kết quả nhận dạng.

Bảng 4.7 trình bày chi phí XLS-R với lô 16.

**Bảng 4.7 Chi phí XLS-R trên một GPU Tesla T4 với lô 16**

| Cấu hình | Giờ/epoch train | Giờ/lượt dev-full | VRAM reserved (GiB) | Tham số huấn luyện |
|---|---:|---:|---:|---:|
| X0 | 0,24 | 0,65 | 2,38 | 262.401 |
| 2 tầng | 0,28 | 0,67 | 2,78 | 25.454.849 |
| 4 tầng | 0,31 | 0,67 | 3,17 | 50.647.297 |
| 8 tầng | 0,39 | 0,66 | 3,96 | 101.032.193 |

Một lượt đánh giá dev-full tốn nhiều thời gian hơn một epoch huấn luyện, nên chi phí chủ yếu tỉ lệ với số epoch. Khi chạy thật, X0 tốn khoảng 0,72 giờ mỗi epoch gồm cả đánh giá dev-full, thấp hơn mức chiếu 0,89 giờ. Các run X1 mở nhiều tầng thực tế kết thúc ở epoch 4 đến epoch 10, nên ngân sách tối đa tính theo 10 epoch cho mỗi run là ước lượng bảo thủ.

### 4.7. Kết quả XLS-R: X0, X1 và so sánh ba mô hình

#### 4.7.1. Chọn cấu hình X1

Cấu hình X1 được chọn theo thứ tự đã đăng ký trước. Với seed 2026, thử mở 2, 4 và 8 tầng Transformer cuối ở tốc độ học bộ mã hóa 10⁻⁵, cùng X0 làm mốc. Mức tầng tốt nhất theo EER dev-full được thử thêm tốc độ học 3·10⁻⁵, và cặp (số tầng, tốc độ học) có EER thấp nhất trở thành X1 cuối. Test không tham gia lựa chọn.

Bảng 4.8 tổng hợp các cấu hình đã thử với seed 2026.

**Bảng 4.8 Các cấu hình XLS-R thử với seed 2026 trên dev-full**

| Cấu hình | Tốt nhất ở epoch | Kết thúc | EER dev-full |
|---|---:|---|---:|
| X0 (chỉ bộ phân loại) | 10 | hết trần 10 epoch | 4,20% |
| X1, mở 2 tầng, tốc độ học 10⁻⁵ | 9 | hết trần 10 epoch | 2,46% |
| X1, mở 4 tầng, tốc độ học 10⁻⁵ | 5 | dừng sớm | 1,08% |
| X1, mở 8 tầng, tốc độ học 10⁻⁵ | 2 | dừng sớm | 0,88% |
| **X1, mở 8 tầng, tốc độ học 3·10⁻⁵** | 2 | dừng sớm | **0,82%** |

X1 cuối mở 8 tầng cuối với tốc độ học bộ mã hóa 3·10⁻⁵. EER giảm khi mở thêm tầng, và mức tối ưu nằm ở biên lưới đã thử (8 tầng), nên chưa biết mở thêm nữa có tốt hơn hay không. Tiêu chí nâng trần epoch, đăng ký trước khi biết kết quả X1, không bị kích hoạt vì cấu hình thắng đạt tốt nhất ở epoch 2, nên trần 10 epoch được giữ nguyên.

#### 4.7.2. Kết quả ba seed

Bảng 4.9 tổng hợp kết quả ba seed của X0, X1 cuối và biến thể X1R (mục 4.8).

**Bảng 4.9 Kết quả XLS-R ba seed trên dev-full**

| Mô hình | Seed 2026 | Seed 2027 | Seed 2028 | EER trung bình ± độ lệch chuẩn | ROC-AUC trung bình | Macro-F1 trung bình |
|---|---:|---:|---:|---:|---:|---:|
| X0 | 4,20% | 4,30% | 4,79% | 4,43% ± 0,32 | 0,9925 | 95,25% |
| X1 (8 tầng, 3·10⁻⁵) | 0,82% | 0,84% | 0,67% | 0,77% ± 0,09 | 0,9992 | 99,17% |
| X1R (lọc 7 kHz) | 0,86% | 0,92% | 0,92% | 0,90% ± 0,04 | 0,9975 | 99,03% |

Epoch tốt nhất của X0 là epoch 10 ở cả ba seed. Của X1 là epoch 2, 8 và 5; của X1R là epoch 8, 6 và 6. Các chỉ số của X0, X1 và X1R đã được tính lại độc lập từ điểm số lưu và khớp kết quả gốc. Trong bản báo cáo ngày 17/11, X1R còn ở mức sơ bộ vì kiểm tra toàn vẹn của lượt chạy tiếp tục ở seed 2026 chưa đạt; sau đó cả ba seed đã được kiểm lại (T30B, T31B và lượt tiếp tục T32B), đưa vào sổ phụ X1R 3/3 mục (`reports/model_registry_x1r.md`) và bổ sung phép kiểm round-trip ở mục 4.8. Bảng 4.9 vì vậy không còn nhãn sơ bộ cho X1R.

Có hai điểm cần lưu ý khi đọc bảng. Thứ nhất, **X0 chưa hội tụ ở trần**: EER của X0 giảm ở mọi epoch và đạt tốt nhất đúng ở epoch 10 (seed 2026: từ 8,80% ở epoch 1 xuống 4,20% ở epoch 10). Tiêu chí nâng trần chỉ áp dụng cho X1, nên EER của X0 có thể còn giảm thêm và khoảng cách X1 với X0 có thể bị phóng đại. Thứ hai, EER của X1 dao động giữa các epoch cỡ 0,4 đến 0,5 điểm phần trăm (seed 2027: từ 0,84% đến 1,26%), lớn hơn độ lệch chuẩn qua seed. Việc chọn epoch tốt nhất trên dev-full vì thế làm EER báo cáo có thể thấp hơn hiệu năng thật. Điểm số AUC của X1R seed 2026 (0,9939) thấp hơn hai seed còn lại (0,9994 và 0,9992) trong khi EER tương đương; chưa có giải thích.

#### 4.7.3. So sánh ba mô hình và trả lời sơ bộ các câu hỏi nghiên cứu

Bảng 4.10 so sánh ba mô hình trên dev-full.

**Bảng 4.10 So sánh ba mô hình trên dev-full (chưa phải kết quả kiểm thử)**

| Mô hình | Bộ mã hóa | EER | ROC-AUC | Macro-F1 | Giờ/epoch huấn luyện (lô 16) | VRAM reserved (GiB) |
|---|---|---:|---:|---:|---:|---:|
| LFCC + LCNN (B0 v2) | Không áp dụng | 8,14% ± 0,18 | 0,974 | 91,33% | Chạy trên CPU | Không áp dụng |
| XLS-R + bộ phân loại (X0) | Đóng băng | 4,43% ± 0,32 | 0,993 | 95,25% | 0,24 | 2,38 |
| XLS-R + bộ phân loại (X1) | Mở 8 tầng cuối | 0,77% ± 0,09 | 0,999 | 99,17% | 0,39 | 3,96 |

Để đo khác biệt có ý nghĩa hay không, bootstrap ghép cặp theo người nói (1.000 lần lấy mẫu lại) được tính cho seed 2026 trên dev-full. Kết quả ở Bảng 4.11.

**Bảng 4.11 Hiệu EER ghép cặp theo người nói trên dev-full, seed 2026**

| So sánh | Hiệu EER (điểm %) | Khoảng tin cậy 95% của hiệu |
|---|---:|---|
| B0 v2 trừ X1 | 7,41 | [4,23; 11,49] |
| X0 trừ X1 | 3,38 | [1,23; 6,15] |

Khoảng tin cậy riêng của từng hệ thống ở seed 2026 là [4,96%; 12,73%] với B0, [1,75%; 8,00%] với X0 và [0,16%; 2,15%] với X1.

Trả lời sơ bộ, trên dev-full:

- **RQ1:** XLS-R có EER thấp hơn B0: 4,43% (X0) và 0,77% (X1) so với 8,14%. Với X1, hiệu so với B0 có khoảng tin cậy không chứa 0. Với X0, các khoảng tin cậy riêng của X0 và B0 chồng lấn và chưa có phép so sánh ghép cặp giữa hai mô hình này, nên chưa kết luận được về X0 so với B0 ở mức ý nghĩa thống kê.
- **RQ2:** X1 có EER thấp hơn X0 và khoảng tin cậy của hiệu không chứa 0. Chi phí tăng khoảng 63% giờ huấn luyện mỗi epoch (0,24 lên 0,39) và khoảng 66% bộ nhớ GPU (2,38 lên 3,96 GiB). Tuy nhiên X1 đạt tốt nhất ở epoch 2 đến 8, còn X0 vẫn đang cải thiện ở epoch 10, nên chưa so được tổng chi phí đến hội tụ.

Các kết luận này có bốn giới hạn: (a) chỉ trên dev-full, nơi cấu hình và trạng thái lưu được chọn; (b) bootstrap ghép cặp chỉ cho một seed và không hiệu chỉnh so sánh nhiều lần; (c) B0 dùng lô 8 còn XLS-R dùng lô 16, tốc độ học cố định, không tối ưu riêng; (d) B0 phụ thuộc một phần vào dải tần cao (mục 4.8), nên khoảng cách giữa B0 và XLS-R không chỉ phản ánh chất lượng biểu diễn.

### 4.8. Phân tích đường tắt (shortcut) theo tần số lấy mẫu

**Bối cảnh.** Theo thông tin mô tả, toàn bộ mẫu chuyển đổi giọng nói (VC) có tần số lấy mẫu gốc 40 kHz, còn bonafide, nhiễu đối kháng và replay ở 16 kHz; tỉ lệ này đúng 100% ở cả tập huấn luyện và dev-full. Mọi dạng sóng đều được hạ về 16 kHz, nhưng bộ lọc chống răng cưa khi hạ từ 40 kHz có thể để lại dấu vết gần 8 kHz, trùng hoàn toàn với nhãn VC. Nếu mô hình dùng dấu vết này, EER thấp sẽ không phản ánh khả năng phát hiện giả mạo.

**Đo phổ dải cao.** Phổ được đo trên một tập con phân tầng của dev-full gồm 3.120 mẫu thuộc 169 shard (1.000 bonafide, 1.000 VC, 1.000 nhiễu đối kháng, 120 replay), kèm nhóm thứ năm là 1.000 bonafide được cho qua chu trình lấy mẫu lại 16→40→16 kHz để mô phỏng dấu vết hạ tần số. Bảng 4.12 trình bày công suất tương đối ở các dải gần 8 kHz; Hình 4.3 minh họa phổ trung bình theo từng ô 50 Hz từ 4 đến 8 kHz.

**Bảng 4.12 Công suất tương đối dải cao theo nhóm trên dev-full, trung vị [P25; P75] (dB)**

| Nhóm | Số mẫu | 6–7 kHz | 7,5–7,8 kHz | 7,95–8 kHz |
|---|---:|---:|---:|---:|
| Bonafide | 1.000 | −36,16 [−40,33; −30,97] | −47,63 [−52,60; −41,77] | −59,02 [−64,08; −53,37] |
| Chuyển đổi giọng nói (VC) | 1.000 | −38,10 [−47,04; −33,63] | −54,88 [−80,18; −49,14] | −69,99 [−91,58; −63,44] |
| Nhiễu đối kháng (AP) | 1.000 | −25,64 [−27,20; −23,96] | −33,39 [−35,29; −31,68] | −43,32 [−45,12; −41,53] |
| Phát lại | 120 | −27,31 [−34,71; −19,50] | −33,72 [−50,53; −26,48] | −41,23 [−63,20; −34,42] |
| Bonafide sau chu trình 16→40→16 kHz | 1.000 | −37,47 [−41,58; −32,25] | −53,94 [−58,89; −48,06] | −66,12 [−71,18; −60,47] |

**Hình 4.3 Phổ dải cao trung bình theo nhóm trên dev-full**

VC có năng lượng sát 8 kHz thấp hơn bonafide, và bonafide sau chu trình lấy mẫu lại dịch về gần VC. Nhóm nhiễu đối kháng và replay có năng lượng dải cao lớn hơn bonafide. Đây là mô tả phân bố phổ; bản thân nó chưa chứng minh mô hình dùng đường tắt.

**Phép thử can thiệp lúc chấm.** Để đo mức phụ thuộc, mô hình đã huấn luyện được chấm lại trên bốn điều kiện: âm thanh gốc; bonafide qua chu trình 16→40→16 kHz; lọc thông thấp 7 kHz; lọc thông thấp 4 kHz (chỉ để đo độ bền). Ngưỡng phân loại giữ nguyên từ dev-full. Ngưỡng chỉ báo được đăng ký trước khi chạy: chỉ báo **mạnh** nếu chu trình lấy mẫu lại làm tỉ lệ bonafide bị báo nhầm tăng từ 10 điểm phần trăm trở lên, hoặc lọc 7 kHz làm EER của VC tăng từ 3 lần trở lên trong khi EER của nhiễu đối kháng tăng dưới 1,5 lần. B0 được chấm trên đủ 38.777 mẫu; XLS-R chấm trên tập con 3.120 mẫu, nên số tuyệt đối của hai họ mô hình không so trực tiếp được.

Bảng 4.13 và Hình 4.4 tổng hợp kết quả.

**Bảng 4.13 EER theo điều kiện can thiệp băng tần (dev-full, không phải test)**

| Mô hình | Âm thanh gốc | Chu trình 16→40→16 kHz | Lọc 7 kHz | Lọc 4 kHz | Tỉ lệ bonafide bị báo nhầm: gốc → chu trình |
|---|---:|---:|---:|---:|---|
| B0 v2, seed 2026 | 8,23% | 11,67% | 24,53% | 51,54% | 8,23% → 15,89% |
| B0 v2, seed 2027 | 8,26% | 13,81% | 17,36% | 38,97% | 8,26% → 19,97% |
| B0 v2, seed 2028 | 7,93% | 12,37% | 14,41% | 50,77% | 7,93% → 18,38% |
| X1 mở 8 tầng, 10⁻⁵, seed 2026 (tập con) | 1,46% | 1,89% | 1,93% | 2,30% | 1,00% → 1,50% |
| X1 cuối, seed 2026 (tập con) | 1,32% | 1,42% | 1,70% | 2,00% | 1,20% → 1,30% |

**Hình 4.4 EER tổng theo bốn điều kiện can thiệp băng tần (B0 ba seed và X1 một cấu hình)**

Theo ngưỡng đã đăng ký:

- **B0 v2: chỉ báo mạnh ở seed 2027 và 2028**, yếu nhưng sát ngưỡng ở seed 2026. Chu trình lấy mẫu lại làm tỉ lệ bonafide bị báo nhầm tăng 7,67, 11,71 và 10,45 điểm phần trăm. Lọc 7 kHz làm EER của nhiễu đối kháng tăng ×11,34, ×5,89 và ×2,29, lớn hơn mức tăng của VC (×1,39, ×1,82 và ×1,77); vì nhóm nhiễu đối kháng cũng suy giảm mạnh, ảnh hưởng không chỉ do dấu vết riêng của VC mà phản ánh B0 dùng rộng rãi thông tin dải cao. Lọc 4 kHz đẩy EER lên 39% đến 52%.
- **X1: chỉ báo yếu.** Chu trình lấy mẫu lại làm tỉ lệ bonafide bị báo nhầm tăng 0,50 điểm phần trăm (cấu hình 10⁻⁵) và 0,10 điểm phần trăm (X1 cuối). Lọc 7 kHz làm EER của VC tăng ×1,23 và ×1,17, chưa chạm ngưỡng ×3. EER của nhiễu đối kháng tăng ×2,0 và ×3,0, nhưng ứng với chỉ 1 đến 4 mẫu lỗi trên 1.000 mẫu, nên không ổn định. Với X1 cuối, kết quả chỉ ở một seed.

**Huấn luyện không có dải trên 7 kHz (X1R).** Phép thử lúc chấm chỉ cho biết mô hình nhạy với dải cao đến đâu, không cho biết hiệu năng còn lại nếu mô hình không được dùng dải này ngay từ đầu. Vì vậy biến thể X1R có cấu hình giống hệt X1 cuối nhưng mọi mẫu huấn luyện và đánh giá đều qua bộ lọc thông thấp 7 kHz. EER trung bình ba seed của X1R là 0,90% ± 0,04, so với 0,77% ± 0,09 của X1 (Bảng 4.9): chênh lệch 0,12 điểm phần trăm, nhỏ so với khoảng cách giữa X1 và B0, và chưa có khoảng tin cậy ghép cặp. Kết quả cho thấy hiệu năng cao của X1 không phụ thuộc vào dải trên 7 kHz, nơi chứa dấu vết hạ tần số. Lưu ý lọc 7 kHz cũng bỏ cả thông tin hợp lệ ở dải cao, nên hiệu 0,12 điểm không hoàn toàn do đường tắt.

**Kiểm tra round-trip cho X1R.** Với 1.000 mẫu bonafide của tập con phép thử băng tần, tỉ lệ bonafide bị báo nhầm của X1R tại ngưỡng của chính nó thay đổi +0,00, −0,10 và +0,00 điểm phần trăm (seed 2026, 2027, 2028) khi cho qua chu trình 16→40→16 kHz, thấp hơn ngưỡng 0,5 điểm đã đăng ký; đây chỉ là phép kiểm trên dev, không đọc test (`reports/x1r_roundtrip_dev.md`).

**Kiểm tra đường tắt mức dữ liệu (SC-01).** Phép thử ở trên dùng mô hình; phép kiểm này đo trực tiếp trên dữ liệu thô, độc lập với mô hình. Với mỗi mẫu, đồ án tính các đặc trưng thô gồm độ dài, tỉ lệ đệm, tỉ lệ khoảng lặng, khoảng lặng đầu và cuối, nền nhiễu, độ lệch một chiều và tỉ số năng lượng dải thấp, trên toàn tệp và trên đoạn 4 giây. Một hồi quy logistic được học trên train (15.885 mẫu) và chấm trên dev-full (38.777 mẫu, 172 người nói). Ngưỡng phân loại chỉ báo được khóa trước: mạnh nếu EER ≤ 10%, trung bình nếu trong (10%; 25%], yếu nếu trên 25%. Bảng 4.14 trình bày kết quả.

**Bảng 4.14 EER dev-full của hồi quy logistic trên đặc trưng thô (SC-01)**

| Phép tách | Vùng toàn tệp | Vùng đoạn 4 giây |
|---|---:|---:|
| Bonafide so với spoof (VC + AP + replay) | 11,89% | 11,76% |
| Bonafide so với VC | **8,76%** | 9,34% |
| Bonafide so với AP | 13,68% | 13,19% |

Nguồn: `reports/shortcut_feature_audit.md`. Mức tổng là **mạnh**: phép tách bonafide với VC trên toàn tệp đạt EER dev 8,76%, khoảng tin cậy 95% theo người nói [4,18%; 17,52%]. Nghĩa là chỉ riêng khoảng lặng, nền nhiễu và các đặc trưng thô tương tự đã tách được VC khỏi bonafide ở mức gần với EER của B0 (8,14%), và gần cả EER tổng hợp spoof (11,89%). Đây là chỉ báo của dữ liệu, không chứng minh mô hình nào dùng đường tắt; metadata cục bộ không có nguồn corpus hay codec nên hai nguồn đường tắt này chưa kiểm được. EER của X1 (0,77%) thấp hơn nhiều so với mức tách bằng đặc trưng thô này.

**Kiểm tra mô hình với khoảng lặng và nền nhiễu (SC-02).** Để xem mô hình có dựa vào các đặc trưng thô ở trên không, các mô hình seed 2026 được chấm lại trên 2.000 mẫu (1.000 bonafide, 1.000 VC) của tập con phép thử băng tần, sau khi cắt khoảng lặng ở hai đầu tệp (`trim_edges`), nâng nền nhiễu (`noise_floor`) và, riêng B0, bỏ độ lệch một chiều (`dc_remove`). Ngưỡng phân loại giữ nguyên từ dev-full. Bảng 4.15 trình bày EER trên tập con này; tập con này khác dev-full nên EER gốc của X1 là 1,20%, không phải 0,82%.

**Bảng 4.15 EER theo điều kiện cắt khoảng lặng và nâng nền nhiễu (SC-02, dev-full, seed 2026)**

| Mô hình | Gốc | Cắt khoảng lặng | Nâng nền nhiễu | Bỏ độ lệch một chiều | Mức tổng |
|---|---:|---:|---:|---:|---|
| B0 v2 | 12,00% | 21,00% | 19,90% | 12,10% | Mạnh |
| X1 cuối | 1,20% | 1,40% | 2,50% | không áp dụng | Trung bình |
| X1R | 1,40% | 1,60% | 2,10% | không áp dụng | Trung bình |

Nguồn: `reports/shortcut_probe_sc02.md`. B0 nhạy rõ với cắt khoảng lặng: EER tăng 9,00 điểm phần trăm và tỉ lệ VC bị bỏ lọt tăng từ 17,8% lên 42,5%. X1 và X1R gần như không đổi khi cắt khoảng lặng (tăng 0,2 điểm), nhưng EER tăng khi nâng nền nhiễu (lần lượt 2,08 và 1,50 lần), kèm tỉ lệ bonafide bị báo nhầm tăng từ 1,2% lên 5,6% ở X1. Mức tăng này đồng thời là mất độ bền trước nhiễu cộng, nên không thuần túy là đường tắt; vì vậy X1 và X1R được xếp mức trung bình chứ không phải yếu. Giới hạn: chỉ một seed mỗi họ mô hình, chỉ 2.000 mẫu, và cắt khoảng lặng cũng làm thay đổi nội dung hợp lệ của đoạn.

**Kết luận từ phân tích.** Ở mức các phép thử này phát hiện được, EER của X1 không được giải thích bởi dấu vết hạ tần số 40 kHz→16 kHz. B0 phụ thuộc một phần vào đặc trưng dải cao, nên EER của B0 và khoảng cách giữa B0 với XLS-R phải được diễn giải kèm giới hạn này. Các phép thử không chứng minh X1 hoàn toàn không có đường tắt: chưa loại trừ được codec, độ dài phát ngôn hay đặc trưng của bộ tổng hợp ở dải dưới 7 kHz. Với khoảng lặng, SC-02 cho thấy X1 và X1R gần như không nhạy khi cắt khoảng lặng nhưng kém bền trước nền nhiễu cộng (chỉ báo trung bình, một seed); B0 nhạy mạnh với cả hai. Trong khi đó, SC-01 cho thấy dữ liệu có chỉ báo đường tắt mạnh ở mức đặc trưng thô, nên EER của B0 đặc biệt cần được đọc kèm giới hạn này. Ngưỡng chỉ báo là lựa chọn có lý do, không phải kiểm định thống kê, và phép thử can thiệp chỉ áp dụng lúc chấm. Các phép thử này sẽ được lặp lại trên tập kiểm thử theo giao thức đã đăng ký.

### 4.9. Kiểm tra tính hợp lệ khi EER thấp bất thường

Với EER dưới 1% của X1, các kiểm tra sau đã thực hiện trước khi coi kết quả là đáng tin:

- **Rò rỉ người nói và tệp:** không có người nói hoặc tệp nào chung giữa tập huấn luyện và dev-full; kiểm lại trực tiếp với mô hình X1 cuối.
- **Rò rỉ nội dung:** hai tập đã được khử trùng theo mã băm nội dung và không có nhóm trùng chéo (mục 2.4 và 2.5).
- **Tần số lấy mẫu:** mọi dạng sóng được chuẩn hóa về 16 kHz; ảnh hưởng của dấu vết hạ tần số được đo ở mục 4.8.
- **Biên độ:** chính sách chuẩn hóa theo đỉnh áp dụng chung cho mọi nhóm.
- **Tái tạo điểm số:** chấm lại mô hình trên một máy khác cho điểm số khớp kết quả lưu (0 mẫu đổi quyết định); sai khác còn lại nằm trong độ phân giải của điểm số lưu ở dạng số thực 32 bit.
- **Chỉ số:** EER và ROC-AUC của mọi mô hình được tính lại từ điểm số lưu bằng mã độc lập và khớp.

Các kiểm tra **chưa** đủ để loại trừ: ảnh hưởng của độ dài phát ngôn gốc (khoảng lặng và nền nhiễu mới được thử ở mức một seed, mục 4.8); ảnh hưởng của độ lớn âm ngoài chuẩn hóa theo đỉnh (chưa có phép thử khuếch đại ngẫu nhiên); các đường tắt do cách thu thập dữ liệu; và hiện tượng chọn lọc do cấu hình được chọn trên dev-full. Kết quả trên tập kiểm thử là phép kiểm tra tiếp theo.

### 4.10. Các kết quả chưa có

**Trạng thái thí nghiệm gấp đôi dữ liệu huấn luyện (D036).** Phần chuẩn bị dữ liệu đã xong (mục 2.9). Huấn luyện đang chạy; **chưa có kết quả chốt nào**, nên báo cáo này không nêu EER hay số epoch tốt nhất của các lượt chạy ×2, và việc so sánh ×1 với ×2 chỉ được thực hiện sau khi cả hai mức hoàn tất.

- **B0 v2 ×2 (CPU cục bộ):** cấu hình riêng cho scale2 (15 khóa khác cấu hình ×1, gồm danh sách mẫu, dữ liệu đóng gói và số bước mỗi epoch 3.972); chuẩn bị đạt ngày 20/11/2026 (`reports/b0_v2_scale2_prep.md`) và khởi chạy lúc 22:50 cùng ngày với ba seed song song (`reports/b0_v2_scale2_launch.md`). Trạng thái: đang huấn luyện trên CPU cục bộ, chưa có kết quả chốt. Phép chiếu thời gian của lượt đầu: trần 10 epoch khoảng 19,4 giờ cho mỗi seed khi ba seed chạy song song.
- **X1 ×2 (Kaggle, hai GPU T4 mỗi phiên):** cấu hình riêng `configs/xlsr_training_scale2.json` (1.986 bước mỗi epoch, chỉ cho phép run X1 cuối và ba seed 2026–2028; cấu hình từ chối chạy X0); lượt chạy thử cặp GPU đạt ngày 19/11/2026. Phiên 1 (seed 2026 và 2027) dùng 10,99 giờ quota rồi tạm dừng có kiểm soát; payload tiếp tục và mô phỏng khôi phục đạt ngày 19–20/11/2026 (`W/reports/xlsr_x1s03_pair_session1.md`, `W/reports/xlsr_x1s04a_resume_prep.md`). Sau khi quota Kaggle làm mới (21/11/2026), hai bộ dữ liệu tiếp tục riêng tư đã được tạo và phiên 2 (seed 2026 tiếp tục cùng seed 2028 mới) được đẩy lên Kaggle lúc 11:24 ngày 21/11/2026 và ở trạng thái đang chạy ngay sau khi đẩy, theo báo cáo X1S-04B (`W/reports/xlsr_x1s04b_s2_push.md`; báo cáo này chưa qua kiểm độc lập). Phiên cho seed 2027 và phần còn lại của seed 2028 chưa chạy. Trạng thái: đang huấn luyện, chưa có kết quả chốt.
- **Chuẩn bị chấm test và đăng ký:** các mục ×2 sẽ vào sổ phụ riêng (6 mục) khi hoàn tất; tối đa 07/12/2026 theo D037 (mục 2.8).

Bảng 4.16 tổng hợp các thí nghiệm chưa có kết quả.

**Bảng 4.16 Các thí nghiệm chưa có kết quả**

| Thí nghiệm | Trạng thái | Điều kiện để bắt đầu |
|---|---|---|
| Đánh giá tập kiểm thử cho B0, X0, X1 (9 mô hình), kèm khoảng tin cậy và so sánh ghép cặp | Chưa chạy; sổ đăng ký chính thức v1 đủ 9 mục đã đóng băng, cổng đang đóng | Hoàn tất các điều kiện D037 mục 6 (mục 2.8), người dùng mở cổng |
| Phép thử băng tần trên tập kiểm thử (B0 và X1, ba seed) và đánh giá X1R | Chưa chạy; sổ phụ X1R đủ 3 mục đã đóng băng | Sau khi mở cổng |
| Thí nghiệm gấp đôi dữ liệu huấn luyện (31.770 mẫu) cho B0 và X1 | Dữ liệu sẵn sàng; B0 ×2 đang chạy trên CPU, X1 ×2 đang chạy trên Kaggle; chưa có kết quả chốt | Hoàn tất ba seed mỗi mô hình (hoặc đến 07/12/2026), đăng ký sổ phụ scale2 |
| Phân tích lỗi theo VC, AP, replay cho XLS-R | Chưa chạy | Sau đánh giá tập kiểm thử |
| Phép thử khuếch đại ngẫu nhiên và phân tích độ dài | Chưa thực hiện (khoảng lặng và nền nhiễu đã thử ở một seed, mục 4.8) | Sau khi khóa mô hình |
| Suy luận cửa sổ trượt và chương trình minh họa | Chưa thực hiện | Sau khi khóa mô hình |

Bảng 4.17 dành cho so sánh ba mô hình trên tập kiểm thử; hiện chưa có kết quả để điền.

**Bảng 4.17 Bảng so sánh ba mô hình trên tập kiểm thử chưa có kết quả**

| Mô hình | Bộ mã hóa | EER | ROC-AUC | Macro-F1 |
|---|---|---:|---:|---:|
| LFCC + LCNN (B0 v2) | Không áp dụng | — | — | — |
| XLS-R + bộ phân loại (X0) | Đóng băng | — | — | — |
| XLS-R + bộ phân loại (X1) | Mở 8 tầng cuối | — | — | — |

---

## CHƯƠNG 5. ĐÁNH GIÁ TIẾN ĐỘ, RỦI RO VÀ KẾ HOẠCH

### 5.1. Đối chiếu với lộ trình 14 tuần

Việc đối chiếu dùng đầu ra của lộ trình 14 tuần (từ 28/09/2026 đến 03/01/2027) và kế hoạch từng tuần, theo trạng thái ngày 21/11/2026, tức đang ở tuần 8 (16–22/11).

Bảng 5.1 tổng hợp đối chiếu tiến độ với lộ trình 14 tuần.

**Bảng 5.1 Đối chiếu tiến độ với lộ trình 14 tuần**

| Tuần | Đầu ra theo kế hoạch | Thực tế | Đánh giá |
|---|---|---|---|
| 1 (28/09–04/10) | Khóa phạm vi, giao thức, chạy thử kỹ thuật quy trình | Đã khóa phạm vi và tên đề tài; chạy thử kỹ thuật B0 | Đạt |
| 2 (05–11/10) | Danh sách mẫu phát triển, kiểm kê dữ liệu | Danh sách mẫu khử trùng, audit 20.000 dạng sóng | Đạt |
| 3 (12–18/10) | Khóa danh sách mẫu, chạy thử kỹ thuật GPU, ba chính sách biên độ | Danh sách mẫu khóa, chạy thử kỹ thuật Kaggle đạt | Đạt |
| 4 (19–25/10) | Vòng lặp huấn luyện, ba pilot, chọn chính sách biên độ | Ba pilot hoàn tất, chọn chuẩn hóa theo đỉnh | Đạt; kết quả pilot mang nhãn lịch sử |
| 5 (26/10–01/11) | B0 có EER, ROC-AUC, macro-F1 hợp lệ | Phát hiện lỗi LFCC; B0 huấn luyện lại, hoàn tất ba seed vào tuần 6 | Chậm một tuần, nay đạt |
| 6 (02–08/11) | XLS-R đóng băng chạy được một vòng huấn luyện | X0 seed 2026 chạy đủ 10 epoch; bắt đầu thử các mức tầng của X1 | Đạt |
| 7 (09–15/11) | Huấn luyện X0, bảng so sánh ban đầu với B0 | Chọn xong cấu hình X1 cuối ngày 09/11; X0 và X1 đủ ba seed; so sánh trên dev-full; sổ đăng ký bản nháp đủ 9 mục | Vượt kế hoạch |
| 8–10 (16/11–06/12) | Tinh chỉnh từng phần, khóa cấu hình, chạy lại với seed cố định | Cấu hình X1 đã khóa sớm hơn mốc 29/11; X1R đã kiểm và đăng ký sổ phụ; sổ chính thức v1 đóng băng 21/11; dữ liệu ×2 hoàn tất, B0 ×2 và X1 ×2 đang huấn luyện | Đúng hoặc sớm hơn kế hoạch |
| 11–14 (07/12–03/01) | Đánh giá tập kiểm thử, phân tích lỗi, chương trình minh họa, bàn giao | Cổng đánh giá và sổ đăng ký đã sẵn sàng; kiểm tương đương Colab L4 đạt; chương trình minh họa chưa bắt đầu | Chưa đến hạn |

**Nhận xét:**

- Nhánh XLS-R vượt kế hoạch nhờ chạy hai lượt song song trên hai GPU của Kaggle và nhờ phần lớn các cấu hình X1 dừng sớm; hạn mức GPU được nâng lên 45 giờ mỗi tuần từ 15/11/2026.
- Khối lượng thực hiện lớn hơn lộ trình gốc: tập đánh giá mở rộng, ba seed cho mọi mô hình, huấn luyện lại B0, các phép thử đường tắt, biến thể X1R và thí nghiệm gấp đôi dữ liệu.
- Hạng mục chưa bắt đầu và có khối lượng đáng kể nhất là chương trình minh họa (suy luận cửa sổ trượt 4 giây, bước nhảy 2 giây) cùng báo cáo, slide cuối.
- Mốc đã điều chỉnh sau D037 (18/11/2026): cổng kiểm thử được mở khi scale2 hoàn tất hoặc chậm nhất vào 07/12/2026 (nếu người dùng quyết định mở), sớm hơn mốc khoảng 13/12 trước đó; sổ đăng ký v1 và sổ phụ X1R đã đóng băng từ 21/11, sổ phụ scale2 chỉ gồm các seed đã xong khi đến hạn. Mốc bàn giao trước hạn 03/01/2027 giữ nguyên; chương trình minh họa vẫn dự kiến sau khi chấm test.

### 5.2. Đánh giá theo cổng kiểm soát G0–G7

**Phương pháp:**

- Mỗi cổng được định nghĩa bằng tiêu chí đầu ra trong lộ trình.
- Một cổng chỉ được tính **đạt** khi mọi tiêu chí của cổng có bằng chứng ở mức "đã kiểm chứng trên dữ liệu thật".
- Các cổng có khối lượng công việc khác nhau. Vì vậy số cổng đạt **không** được quy đổi thành phần trăm khối lượng hoàn thành.

Bảng 5.2 tổng hợp trạng thái các cổng kiểm soát.

**Bảng 5.2 Trạng thái các cổng kiểm soát**

| Cổng | Tiêu chí chính | Trạng thái |
|---|---|---|
| G0. Dữ liệu sẵn sàng | Thông tin mô tả audit, tập dữ liệu không rò rỉ, chạy thử kỹ thuật âm thanh, EER có kiểm thử | **Đạt** |
| G1. Dữ liệu thí nghiệm được khóa | Danh sách mẫu khóa, speaker-disjoint, thống kê từng tập dữ liệu | **Đạt** |
| G2. Pipeline huấn luyện hoạt động | Loader, hàm mất mát hữu hạn, trạng thái lưu, ROC-AUC/macro-F1/ngưỡng, tiếp tục được | **Đạt** |
| G3. Mô hình cơ sở hoàn thành | Ba pilot, khóa chính sách biên độ, B0 có chỉ số hợp lệ, cấu hình và nhật ký | **Đạt** (B0 v2 ba seed trên dev-full; pilot mang nhãn lịch sử) |
| G4. XLS-R đóng băng hoàn thành | Bộ mã hóa không cập nhật, hội tụ, so sánh với B0 | **Đạt một phần**: đã huấn luyện và so sánh với B0 trên dev-full, nhưng X0 chưa hội tụ ở trần 10 epoch |
| G5. XLS-R tinh chỉnh từng phần hoàn thành | Ghi rõ số tầng, trạng thái lưu, nhật ký tài nguyên, không dùng test để chọn | **Đạt** (trên dev-full; X1R đã kiểm và đăng ký sổ phụ) |
| G6. Đánh giá cuối hoàn thành | Bảng ba mô hình trên tập kiểm thử, phân tích lỗi, shortcut | Chưa đạt |
| G7. Bàn giao hoàn thành | Chương trình minh họa, hướng dẫn tổng quan, báo cáo và slide thống nhất | Chưa đạt |

**Kết quả: 5/8 cổng đạt (G0, G1, G2, G3, G5), một cổng đạt một phần (G4).** Hai cổng còn lại chứa tập kiểm thử và sản phẩm bàn giao.

### 5.3. Trạng thái các sản phẩm bắt buộc

Bảng 5.3 tổng hợp trạng thái các sản phẩm bắt buộc.

**Bảng 5.3 Trạng thái các sản phẩm bắt buộc**

| # | Sản phẩm | Trạng thái |
|---|---|---|
| 1 | Quy trình dữ liệu tái lập từ thông tin mô tả đến huấn luyện/phát triển/kiểm thử | Phần lớn đã có. Công cụ tải thông tin mô tả còn phụ thuộc gói datasets chưa có trong danh sách thư viện phụ thuộc |
| 2 | Tập dữ liệu theo người nói và báo cáo rò rỉ | Đã có |
| 3 | LFCC + LCNN hoạt động ổn định | Đã có kết quả ba seed trên dev-full, đã đăng ký sổ chính thức v1; chờ đánh giá tập kiểm thử |
| 4 | XLS-R đóng băng | Đã có ba seed trên dev-full (chưa hội tụ ở trần), đã đăng ký sổ chính thức v1; chờ tập kiểm thử |
| 5 | XLS-R tinh chỉnh từng phần | Đã có cấu hình cuối và ba seed trên dev-full, đã đăng ký sổ chính thức v1; biến thể X1R đã đăng ký sổ phụ; chờ tập kiểm thử |
| 6 | Bảng so sánh trên cùng tập kiểm thử | Chưa có |
| 7 | Phân tích lỗi theo VC, AP, replay | Một phần: B0 theo kiểu tấn công trên dev-full; XLS-R chưa |
| 8 | Chương trình minh họa | Chưa có |
| 9 | Báo cáo, slide, mã nguồn, hướng dẫn | Đang làm dần; báo cáo này là bản thảo |

### 5.4. Rủi ro chính

Bảng 5.4 tổng hợp rủi ro và biện pháp xử lý.

**Bảng 5.4 Rủi ro và biện pháp xử lý**

| Mức | Rủi ro | Biện pháp |
|---|---|---|
| Cao | Chương trình minh họa, phân tích lỗi cuối, báo cáo và slide chưa làm, trong khi hạn 03/01/2027 cố định | Viết suy luận cửa sổ trượt ngay bằng mô hình X1 đã chọn (không phụ thuộc kết quả kiểm thử), chạy song song với các phiên GPU |
| Cao | Đánh giá tập kiểm thử chỉ được chạy một lần mỗi mô hình; lỗi phát hiện sau đó không thể chấm lại âm thầm | Thử đầy đủ quy trình trên dev-full (đã thực hiện), kiểm tập kiểm thử trên Kaggle trước khi mở cổng; mọi lần chấm lại phải có quyết định riêng |
| Trung bình | EER của X1 dưới 1% và được chọn trên dev-full nên có thể lạc quan; đường tắt ngoài tần số lấy mẫu chưa loại trừ; chỉ báo dữ liệu ở mức đặc trưng thô là mạnh (SC-01) và X1 kém bền trước nền nhiễu (SC-02) | Báo cáo kết quả tập kiểm thử làm kết quả chính; giữ kèm phép thử băng tần, X1R, SC-01/SC-02; bổ sung phép thử khuếch đại và độ dài nếu còn thời gian |
| Trung bình | B0 phụ thuộc một phần vào dải cao nên so sánh B0 với XLS-R có thể bị lệch | Nêu rõ trong mọi bảng so sánh; không diễn giải khoảng cách chỉ là chất lượng biểu diễn |
| Trung bình | X0 chưa hội tụ ở trần 10 epoch, có thể làm khoảng cách X1 và X0 bị phóng đại | Nêu rõ khi so sánh; không nâng trần riêng cho X0 sau khi đã biết kết quả |
| Trung bình | Hạn mức GPU Kaggle 45 giờ mỗi tuần: phiên 1 của X1 ×2 đã dùng 10,99 giờ; sau khi làm mới ngày 21/11 còn 45,0 giờ, phiên 2 đang chạy và hạn mức làm mới kế tiếp ngày 28/11 (`W/reports/xlsr_x1s04b_s2_push.md`) | Đọc hạn mức qua API trước và sau mỗi phiên; dành Kaggle cho X1 ×2 theo thứ tự đã đăng ký; chấm test XLS-R chuyển sang Colab L4 (D037) để không tiêu quota Kaggle |
| Trung bình | Thí nghiệm gấp đôi dữ liệu cần ba seed cho hai mô hình; X1 mỗi run có thể vượt hạn phiên 11 giờ và cần tiếp tục; B0 ×2 chiếm CPU cục bộ nhiều giờ (phép chiếu khoảng 19,4 giờ mỗi seed cho 10 epoch khi ba seed song song) | Hạn chót 07/12/2026 (D037); nếu chưa kịp thì chỉ đăng ký và chấm các seed đã xong, ghi rõ chưa đủ, không bỏ seed, không chọn seed theo kết quả; không sửa mã nguồn trên máy chạy B0 trong lúc chạy vì lượt tiếp tục kiểm mã băm mã nguồn |
| Trung bình | B0 dùng lô 8 còn XLS-R dùng lô 16; tốc độ học cố định, chưa tối ưu riêng | Ghi rõ giới hạn; không diễn giải cấu hình đã chọn là tối ưu |
| Trung bình | Quy mô huấn luyện nhỏ (15.885 mẫu, khoảng 10,5% tập huấn luyện gốc); replay chỉ 120 mẫu mỗi tập đánh giá | Nêu rõ giới hạn; thí nghiệm gấp đôi dữ liệu đo xu hướng; không kết luận cho replay |
| Thấp | Cấu hình được chọn ở biên lưới tầng (8 tầng) | Ghi rõ; không mở rộng lưới vì ngoài ngân sách đã đăng ký |

### 5.5. Kế hoạch tiếp theo

Bảng 5.5 tổng hợp các bước tiếp theo và tiêu chí chấp nhận.

**Bảng 5.5 Các bước tiếp theo và tiêu chí chấp nhận**

| # | Công việc | Tiêu chí chấp nhận | Mốc |
|---|---|---|---|
| 1 | Hoàn tất huấn luyện thí nghiệm gấp đôi dữ liệu: B0 ×2 (CPU) và X1 ×2 (Kaggle: phiên 2 đang chạy, sau đó phiên cho seed 2027 và phần còn lại của seed 2028), ba seed mỗi mô hình | Đủ 6 mục trong sổ phụ scale2 hoặc báo cáo đúng số seed đã xong; chưa nêu kết quả trước khi được kiểm | Hạn 07/12 |
| 2 | Hoàn tất điều kiện trước khi mở cổng (D037 mục 6): bật hai cờ kiểm tra, đưa test-full lên Drive và đối chiếu mã băm, chạy thử notebook chấm test Colab trên dev-full | Dữ liệu khớp, mọi điều kiện đạt; người dùng quyết định mở cổng | Trước 07/12 |
| 3 | Chạy thêm probe lối tắt cho X1 ×2 và lập sổ đăng ký phụ scale2 từ các mục đã xong | Đủ mục đã hoàn tất, đúng cấu hình đánh giá | Trước khi mở cổng |
| 4 | (Đã xong 21/11/2026) Đóng băng sổ đăng ký chính thức v1 (9 mục) và sổ phụ X1R (3 mục) | 9/9 và 3/3 mục có mặt và đã kiểm; chưa chấm test | Hoàn tất |
| 5 | Đánh giá tập kiểm thử một lần (XLS-R trên Colab L4, B0 trên CPU), so sánh ghép cặp, phép thử băng tần, so sánh ×1 với ×2 | Bảng B0/X0/X1 (và X1R, scale2) đủ nguồn gốc; khoảng tin cậy theo người nói | Sau khi mở cổng (theo D037, chậm nhất 07/12 nếu người dùng quyết định mở) |
| 6 | Phân tích lỗi theo kiểu tấn công và phép thử bổ sung (khuếch đại, độ dài, khoảng lặng) nếu còn thời gian | Bảng phân tích kèm giới hạn | 14–20/12 |
| 7 | Suy luận cửa sổ trượt (4 giây, bước nhảy 2 giây) và chương trình minh họa | Chạy được với tệp ngắn và dài, cùng tiền xử lý và ngưỡng dev | sau 14/12 |
| 8 | Hoàn thiện báo cáo, slide, hướng dẫn tổng quan, kiểm tra tái lập | Số liệu thống nhất giữa các tài liệu | 03/01/2027 |

---

## KẾT LUẬN TẠM THỜI

Đến ngày 21/11/2026, đồ án đã hoàn thành phần nền tảng và phần lớn thực nghiệm trên tập phát triển, có bằng chứng trên dữ liệu thật:

- Giao thức dữ liệu speaker-disjoint có khử trùng nội dung, tập đánh giá mở rộng 38.777 / 30.698 mẫu và dữ liệu dạng sóng đóng gói đã đối chiếu với dữ liệu gốc.
- Quy trình tiền xử lý thống nhất, các chỉ số đánh giá kèm khoảng tin cậy theo người nói và cơ chế huấn luyện tiếp tục chính xác qua nhiều phiên.
- Sổ đăng ký mô hình chính thức v1 (9 mục) và sổ phụ X1R (3 mục) đã đóng băng ngày 21/11/2026; cổng đánh giá cuối khóa một lần đã thử trên tập phát triển và đang đóng. Kiểm tương đương Colab L4 đạt (ΔEER 0,000000 điểm phần trăm, 0 mẫu đổi quyết định), cho phép chấm test XLS-R trên Colab theo D037.
- Dữ liệu huấn luyện gấp đôi (31.770 mẫu) đã chọn, khử trùng, đóng gói và đối chiếu đạt 100%; huấn luyện B0 ×2 và X1 ×2 đang chạy, chưa có kết quả chốt.

Về thực nghiệm, cả ba mô hình đã được huấn luyện đủ ba seed. Trên dev-full, EER trung bình là 8,14% ± 0,18 với B0, 4,43% ± 0,32 với X0 và 0,77% ± 0,09 với X1. X1 thấp hơn X0 và B0 với khoảng tin cậy ghép cặp theo người nói không chứa 0 (seed 2026), phù hợp với giả thuyết nghiên cứu; chi phí của X1 cao hơn X0 khoảng 63% giờ huấn luyện mỗi epoch và 66% bộ nhớ GPU. X0 chưa hội tụ ở trần 10 epoch.

Phân tích đường tắt cho thấy B0 phụ thuộc một phần vào dải tần cao (chỉ báo mạnh ở hai trong ba seed), còn X1 chỉ có chỉ báo yếu và biến thể huấn luyện không có dải trên 7 kHz đạt EER gần tương đương (0,90%). Các phép thử này giảm khả năng X1 dựa vào dấu vết hạ tần số nhưng không chứng minh X1 không có đường tắt. Hai kiểm tra bổ sung cho thấy dữ liệu có chỉ báo đường tắt mạnh ở mức đặc trưng thô (EER 8,76% khi tách bonafide khỏi VC chỉ bằng khoảng lặng, nền nhiễu và các đặc trưng thô tương tự), B0 nhạy mạnh với cắt khoảng lặng và nền nhiễu, còn X1 và X1R chỉ ở mức trung bình (chủ yếu do kém bền trước nền nhiễu cộng, một seed).

Hai câu hỏi nghiên cứu RQ1 và RQ2 đã có trả lời sơ bộ trên tập phát triển nhưng **chưa được kết luận**, vì cấu hình được chọn bằng chính tập này và tập kiểm thử chưa được dùng. Đường găng còn lại là: hoàn tất thí nghiệm gấp đôi dữ liệu (hạn 07/12/2026), hoàn tất các điều kiện trước khi mở cổng, đánh giá tập kiểm thử một lần sau khi người dùng mở cổng, chương trình minh họa, báo cáo và slide, trước hạn bàn giao 03/01/2027.

---

## TÀI LIỆU THAM KHẢO

[1] V. Hoang *et al.*, "VSASV: A Vietnamese Spoofing-Aware Speaker Verification Dataset," in *Proc. Interspeech 2024*, pp. 4288–4292, doi: 10.21437/Interspeech.2024-1972.

[2] X. Wang *et al.*, "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," *Computer Speech & Language*, vol. 64, 2020.

[3] X. Liu *et al.*, "ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild," *IEEE/ACM Trans. Audio, Speech, and Language Processing*, vol. 31, 2023.

[4] J. Jung *et al.*, "AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks," in *Proc. ICASSP 2022*.

[5] HUST-EP Lab, "VSASV-Dataset — public dataset snapshot and card," Hugging Face. [Online]. Available: https://huggingface.co/datasets/hustep-lab/VSASV-Dataset

[6] M. Sahidullah, T. Kinnunen, and C. Hanilçi, "A comparison of features for synthetic speech detection," in *Proc. Interspeech 2015*.

[7] X. Wu, R. He, Z. Sun, and T. Tan, "A light CNN for deep face representation with noisy labels," *IEEE Trans. Information Forensics and Security*, vol. 13, no. 11, 2018.

[8] G. Lavrentyeva *et al.*, "STC antispoofing systems for the ASVspoof2019 challenge," in *Proc. Interspeech 2019*.

[9] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *Proc. NeurIPS 2020*.

[10] A. Babu *et al.*, "XLS-R: Self-supervised cross-lingual speech representation learning at scale," in *Proc. Interspeech 2022*.

[11] H. Tak *et al.*, "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation," in *Proc. Odyssey 2022*.

[12] X. Wang and J. Yamagishi, "Investigating self-supervised front ends for speech spoofing countermeasures," in *Proc. Odyssey 2022*.

[13] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.
