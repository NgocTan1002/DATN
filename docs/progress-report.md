# Báo cáo tiến độ đồ án tốt nghiệp

**Đề tài:** Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần

**Sinh viên thực hiện:** Nguyễn Ngọc Tân, MSSV 2251262637, lớp 64TTNT1, Khoa Công nghệ Thông tin, Trường Đại học Thủy Lợi

**Giảng viên hướng dẫn:** PGS.TS Nguyễn Quang Hoan

**Kỳ báo cáo:** 04/11/2026. Số liệu chốt đến tối ngày 03/11/2026.

**Tính chất tài liệu:** đây là bản thảo làm dần của báo cáo đồ án. Các chương đã có cơ sở được viết ở dạng hoàn chỉnh. Các mục còn phụ thuộc thí nghiệm chưa chạy được đánh dấu rõ và để trống phần kết quả. Các số liệu và nhận định phản ánh phạm vi đã được kiểm tra trong đợt rà soát độc lập ngày 03/11/2026.

---

## Quy ước trạng thái và mức bằng chứng

Báo cáo dùng các mức sau, theo thứ tự bằng chứng tăng dần:

- **Chưa thực hiện**: Không có mã, sản phẩm dữ liệu hoặc lượt chạy đáp ứng tiêu chí.
- **Đã triển khai**: Có mã nguồn hoặc cấu hình; chưa có bằng chứng chạy đúng.
- **Đã kiểm thử**: Có kiểm thử tự động đạt, thường trên dữ liệu kiểm thử tổng hợp hoặc mô hình thu nhỏ. Chưa đồng nghĩa chạy đúng trên dữ liệu thật.
- **Đã kiểm chứng trên dữ liệu thật**: Có lượt chạy trên dữ liệu VSASV hoặc GPU thật, kèm báo cáo đã được lưu.
- **Lịch sử**: Kết quả có thật nhưng đã bị quyết định sau thay thế; chỉ dùng để thảo luận, không dùng làm kết quả chính.

Mọi chỉ số trong báo cáo đều đo trên **tập phát triển (development)**. **Tập kiểm thử (test) chưa được dùng** cho mô hình nào.

---

## Tóm tắt

Đồ án xây dựng hệ thống phân loại nhị phân một đoạn tiếng nói tiếng Việt thành *bonafide* (thật) hoặc *spoof* (giả mạo). Bộ dữ liệu được sử dụng là bản phát hành công khai của VSASV trên Hugging Face. LFCC kết hợp LCNN là mô hình cơ sở (B0). XLS-R 300M đóng băng (X0) là cấu hình đối chiếu, XLS-R tinh chỉnh từng phần (X1) là mô hình chính.

Đến hiện tại, các thành phần sau đã có bằng chứng hoạt động trên dữ liệu thật:

- Nền tảng dữ liệu: kiểm kê thông tin mô tả, chia tập theo người nói, khử trùng nội dung, mở rộng tập đánh giá và đóng gói dữ liệu.
- Quy trình tiền xử lý âm thanh.
- Các chỉ số đánh giá.
- Cơ chế huấn luyện có thể tiếp tục chính xác sau gián đoạn, chạy trên GPU Kaggle.

Ba thí nghiệm sơ bộ đã chọn chính sách chuẩn hóa biên độ theo đỉnh. Chuỗi huấn luyện B0 đầy đủ đạt EER 8,33% trên 4.115 mẫu development ở cuối vòng huấn luyện 2.

Sau đó, quá trình rà soát phát hiện lỗi LFCC phụ thuộc lô. Lỗi đã được sửa và có kiểm thử hồi quy; phép đo trên 4.115 mẫu cho thấy EER đổi từ 8,33% sang 8,40% khi chấm riêng từng mẫu. Kết quả cũ được giữ làm lịch sử. B0 v2 với đặc trưng đã sửa đã chạy xong 3 epoch cho ba seed trên CPU, đạt EER dev-full trung bình 8,14% ± 0,18 điểm phần trăm. Đây là kết quả trung gian vì cả ba seed còn cải thiện ở epoch cuối; bản chính thức chờ nối dài theo trần 10 epoch và dừng sớm sau 2 lần đánh giá không cải thiện.

XLS-R đã tải và kiểm mã băm trọng số, tải offline trên CPU và đo chi phí trên GPU; 12/12 cấu hình chạy được, chưa có kết quả nhận dạng khoa học. Cấu hình tinh chỉnh và ngân sách đã chốt, nhưng chương trình huấn luyện khoa học còn thiếu. Chưa có đánh giá test, phân tích lỗi cuối và chương trình minh họa. Mô hình cơ sở chính thức của tuần 5 vẫn chờ bản nối dài; tuần 6 đạt một phần. Có 3/8 cổng kiểm soát đạt (mục 5.2).

---

## MỞ ĐẦU

### 1. Lý do chọn đề tài

Các kỹ thuật tổng hợp tiếng nói (text-to-speech, TTS) và chuyển đổi giọng nói (voice conversion, VC) hiện có thể tạo ra tiếng nói gần với giọng của một người cụ thể. Điều này tạo nguy cơ cho các hệ thống dùng giọng nói làm yếu tố xác thực như tổng đài, ngân hàng hoặc dịch vụ trực tuyến. Bộ phát hiện giả mạo (spoofing countermeasure, CM) là thành phần đứng trước hoặc song song với hệ thống xác thực người nói (automatic speaker verification, ASV), có nhiệm vụ từ chối các đầu vào không phải tiếng nói thật.

Phần lớn nghiên cứu và dữ liệu công khai về chống giả mạo tập trung vào tiếng Anh, tiêu biểu là chuỗi ASVspoof [2], [3]. Với tiếng Việt, VSASV [1] là một trong số ít bộ dữ liệu công khai được thiết kế cho bài toán này. Việc đánh giá có hệ thống các phương pháp phổ biến trên dữ liệu tiếng Việt, theo một giao thức kiểm soát được rò rỉ dữ liệu, vì vậy có giá trị thực tiễn và học thuật.

### 2. Mục tiêu và câu hỏi nghiên cứu

Mục tiêu tổng quát là xây dựng và đánh giá một hệ thống phân loại nhị phân bonafide/spoof cho tiếng nói tiếng Việt, theo quy trình có thể tái lập. Hai câu hỏi nghiên cứu:

- **RQ1.** XLS-R kết hợp bộ phân loại nhị phân có đạt EER thấp hơn mô hình cơ sở LFCC–LCNN trên cùng giao thức dữ liệu hay không?
- **RQ2.** Tinh chỉnh từng phần XLS-R có cải thiện kết quả so với đóng băng toàn bộ bộ mã hóa hay không, và chi phí tính toán tăng bao nhiêu?

Giả thuyết làm việc là X1 đạt EER thấp hơn X0 và B0 trong giới hạn tài nguyên GPU của đồ án. **Đến thời điểm báo cáo, chưa có dữ liệu thực nghiệm để kiểm định giả thuyết này.**

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
 - Tập test chỉ được đánh giá một lần, sau khi mọi cấu hình đã đăng ký.
 - Mỗi thay đổi giao thức được ghi lại cùng lý do và phạm vi áp dụng trước khi thực hiện.
- **Kỹ thuật phần mềm phục vụ tái lập:** kiểm thử tự động; mã băm kiểm tra toàn vẹn cho dữ liệu, cấu hình và mã nguồn; trạng thái lưu lưu đủ trạng thái để tiếp tục chính xác.


### 5. Đóng góp dự kiến

Đồ án **không** đề xuất kiến trúc học sâu mới. Đóng góp dự kiến gồm:

1. Đánh giá có hệ thống LFCC–LCNN và XLS-R (đóng băng, tinh chỉnh từng phần) cho tiếng nói tiếng Việt trên cùng một giao thức.
2. Một giao thức dữ liệu speaker-disjoint, có khử trùng nội dung và kiểm tra rò rỉ, trên bản phát hành công khai của VSASV.
3. Báo cáo minh bạch các vấn đề chất lượng của bản phát hành, như chênh lệch so với bài báo và nguy cơ mô hình học "đường tắt" (shortcut) từ tần số lấy mẫu hoặc biên độ.

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

- **Truy cập logic (logical access, LA):** tiếng nói giả được đưa trực tiếp vào hệ thống, ví dụ TTS hoặc VC.
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

> Trạng thái chương: các mục 2.1 đến mục 2.7 mô tả sản phẩm dữ liệu đã tạo và kiểm chứng. Mục 2.8 là nguyên tắc đã khóa.

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
- **Lý do chuẩn hóa 16 kHz:** thông tin mô tả ghi nhận tần số lấy mẫu gốc khác nhau giữa các nhóm. Nếu không chuẩn hóa, mô hình có thể dùng băng thông làm đường tắt để đoán nhãn.
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
- Đánh giá test diễn ra **một lần**, sau khi mọi mô hình và seed đã được đăng ký, dự kiến khoảng ngày 13/12/2026 (giao thức hiện hành).
- Đến thời điểm báo cáo, sản phẩm dữ liệu test-full mới chỉ được đóng gói và băm lại; **chưa có mô hình nào chạy trên test**.

---

## CHƯƠNG 3. PHƯƠNG PHÁP VÀ KIẾN TRÚC HỆ THỐNG

### 3.1. Kiến trúc tổng thể

Hệ thống được tổ chức thành các tầng có ranh giới rõ ràng:

Hình 3.1 trình bày kiến trúc tổng thể gồm bảy tầng: (1) tổ chức dữ liệu và khử trùng; (2) tiền xử lý về 16 kHz, điều chỉnh biên độ và tạo đoạn 4 s; (3) trích xuất đặc trưng và phân loại bằng B0 hoặc X0/X1; (4) điều khiển huấn luyện, dừng sớm và lưu trạng thái; (5) đánh giá EER, ROC-AUC, macro-F1, ngưỡng phát triển và khoảng tin cậy theo người nói; (6) vận hành cục bộ và GPU Kaggle; (7) đăng ký mô hình, kiểm soát đánh giá cuối và chương trình suy luận. Tầng cuối chưa được triển khai.

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

**Trạng thái:** ba seed đã chạy xong 3 epoch trên CPU với dữ liệu đóng gói, dùng 15.885 mẫu train/57 shard và 38.777 mẫu dev-full/186 shard. Bản chạy nối dài từ trạng thái lưu cuối chưa chạy. Kết quả chính thức sẽ lấy từ bản nối dài theo quy tắc chung; các kết quả hiện có chỉ là trung gian với giới hạn 3 epoch.

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
| X1 | $k$ tầng Transformer cuối và bộ phân loại | Hai nhóm tốc độ học riêng cho bộ mã hóa và bộ phân loại; $k$ thuộc {2, 4, 8} |

Chuẩn hóa đầu vào theo bộ xử lý tiền huấn luyện thay thế thiết kế so sánh hai nhánh chuẩn hóa trước đây. Lý do là cấu hình tiền huấn luyện đã cố định có bật chuẩn hóa từng đoạn, nên dùng đúng phân phối đầu vào lúc tiền huấn luyện là lựa chọn mặc định dễ biện luận. Đây là lựa chọn có lý do, **không** phải giá trị được tối ưu bằng thực nghiệm.

#### 3.3.2. Trạng thái triển khai

Bảng 3.4 tổng hợp trạng thái triển khai XLS-R và giới hạn bằng chứng.

**Bảng 3.4 Trạng thái triển khai XLS-R và giới hạn bằng chứng**

| Hạng mục | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| Khung mô hình, đóng băng/mở tầng, gộp theo thời gian, nhóm tốc độ học | Đã triển khai, đã kiểm thử | Chạy thử CPU trên bộ mã hóa **thu nhỏ, trọng số ngẫu nhiên**, 2 dạng sóng tổng hợp mỗi chế độ, **0 mẫu VSASV** |
| Chuẩn hóa đầu vào theo giao thức hiện hành | Đã triển khai, đã kiểm thử | Chưa được kết nối với bộ điều khiển thí nghiệm chính thức |
| Trạng thái lưu XLS-R tiếp tục chính xác | Đã kiểm thử trên CPU | Cơ chế đối chiếu danh tính lượt chạy hiện **từ chối seed khác 2026** |
| Trọng số tiền huấn luyện | Đã tải và kiểm mã băm | 1.269.737.156 byte |
| Tải offline trên CPU | Đã kiểm chứng | Đoạn 4 giây cho đầu ra 199 khung × 1.024 chiều |
| Đo VRAM và thời gian trên GPU | Đã hoàn thành | Chỉ đo chi phí, chưa phải kết quả nhận dạng |
| Số tầng mở, tốc độ học, lô và ngân sách | Đã chốt ngày 03/11/2026 | Mở 2, 4 hoặc 8 tầng cuối; lô 16; tối đa 10 epoch, dừng sau 2 lần đánh giá không cải thiện |
| Chương trình huấn luyện khoa học cho X0/X1 | Chưa có | Cần tích hợp bộ điều khiển thí nghiệm chính thức và hỗ trợ seed huấn luyện bổ sung |

Cấu hình XLS-R dùng Adam, tốc độ học cố định, không lịch điều chỉnh. Bộ mã hóa thử tốc độ học 10⁻⁵ và 3·10⁻⁵ (mức thứ hai chỉ thử ở mức tầng tốt nhất); bộ phân loại dùng tốc độ học 10⁻³, 128 đơn vị và dropout 0,2. Cấu hình X1 cuối, X0 và B0 dùng ba seed 2026, 2027, 2028.

Thứ tự chạy: seed 2026 với ba mức tầng ở tốc độ học bộ mã hóa 10⁻⁵ và X0; mức tầng tốt nhất theo dev-full được thử thêm tốc độ học 3·10⁻⁵; sau đó cấu hình X1 cuối và X0 chạy thêm hai seed 2027 và 2028. Khi hòa EER, ưu tiên ít tầng hơn; khi chọn tốc độ học ở mức tầng đã chọn, ưu tiên tốc độ học thấp hơn nếu hòa. Test không tham gia lựa chọn.

**Nhận xét:** XLS-R tiền huấn luyện đã chạy trên GPU để đo chi phí với dữ liệu thật; chưa chạy hết một epoch và chưa có kết quả nhận dạng khoa học.

### 3.4. Huấn luyện có thể tiếp tục chính xác

Phiên GPU trên Kaggle giới hạn thời gian. Một lượt huấn luyện B0 vì thế phải chia thành nhiều phiên, mỗi phiên tối đa 512 bước tối ưu. Để kết quả không phụ thuộc việc bị chia phiên, hệ thống được thiết kế như sau:

- **Nội dung trạng thái lưu:** tham số mô hình, trạng thái bộ tối ưu, số bước, vòng huấn luyện, vị trí trong vòng huấn luyện (vị trí đọc tiếp), trạng thái bộ lấy mẫu, trạng thái sinh số ngẫu nhiên của Python/NumPy/PyTorch/CUDA, lịch sử đánh giá, bộ đếm số lượt không cải thiện cho phép và danh tính lượt chạy.
- **Kho trạng thái lưu theo thế hệ lưu:** mỗi lần lưu tạo một thế hệ lưu mới, ghi nguyên tử; bản chỉ dẫn ghi vị trí trạng thái mới nhất và mô hình tốt nhất, kèm mã băm.
- **Bộ lấy mẫu hỗ trợ tiếp tục sau gián đoạn:** tái tạo đúng thứ tự lô từ seed và vị trí đọc tiếp.

Để kiểm tra việc chia phiên không làm thay đổi kết quả, chạy liên tục được đối chiếu với chạy dừng–tiếp tục trên CPU và cho 12/12 tensor mô hình trùng tuyệt đối. Trên GPU, kiểm tra sau từng phiên trong 11 phiên hoàn tất của chuỗi B0 (mục 4.3) xác nhận mọi thế hệ lưu nhập vào giữ nguyên từng byte và chuỗi hàm mất mát trùng với phiên trước.

### 3.5. Hạ tầng tính toán và quy trình vận hành

- **Máy cục bộ (CPU):** phát triển, kiểm thử, xử lý dữ liệu, đóng gói sản phẩm dữ liệu. Môi trường: PyTorch 2.11.0+cpu.
- **Kaggle (GPU chính):** Tesla T4, PyTorch 2.11.0+cu128, CUDA 12.8.
- **Quy trình vận hành bằng dòng lệnh** gồm kiểm tra tài khoản, đóng gói mã nguồn xác định có bản kê mã băm từng tệp, tạo bộ dữ liệu riêng tư, khởi chạy phiên GPU, theo dõi trạng thái và tải kết quả.
- **Kiểm soát sai lệch mã nguồn:** bộ điều khiển chạy trên Kaggle từ chối chạy nếu mã nguồn thực tế không khớp mã băm đã khai báo. Sau khi một phiên lỗi trước huấn luyện vì bộ chạy chứa mã nguồn sai phiên, hệ thống bổ sung bước so khớp bộ chạy với cấu hình thực thi.

### 3.6. Kiểm thử và kiểm soát chất lượng

- Bộ kiểm thử đơn vị bao phủ: cấu trúc dữ liệu, chia đoạn, chính sách biên độ, tách tập, khử trùng, bộ đọc dữ liệu, LFCC (gồm tính độc lập theo lô), LCNN, XLS-R, chỉ số, bootstrap, trạng thái lưu, tiếp tục chính xác, bộ điều khiển B0 v2, đóng gói và đối chiếu tính tương đương, quy trình Kaggle.
- Bộ kiểm thử toàn phần đạt **302/302** trên trạng thái mã nguồn được rà soát. Kết quả này được ghi trong đợt rà soát độc lập ngày 03/11/2026 (mục 6 của bản rà soát) và lần chạy được ghi nhận khi soạn bản nguồn cũng đạt (302 kiểm thử thành công trong 209 giây).
- **Phân biệt mức bằng chứng:** nhiều kiểm thử dùng dữ liệu kiểm thử tổng hợp, ví dụ bộ điều khiển B0 v2 và đối chiếu dữ liệu đóng gói. Các kiểm thử này chứng minh logic đúng, **không** chứng minh lượt chạy trên dữ liệu thật đã đạt.
- **Vấn đề tái lập:** một số thay đổi quan trọng (bộ điều khiển B0 v2, bộ tạo dữ liệu đóng gói, công cụ đối chiếu tính tương đương) vẫn **chưa được ghi nhận phiên bản mã nguồn**. Vì vậy bản mã nguồn đã ghi nhận gần nhất chưa tái tạo đủ năng lực đã kiểm thử.

### 3.7. Ma trận trạng thái thành phần

Bảng 3.5 tổng hợp trạng thái các thành phần hệ thống.

**Bảng 3.5 Trạng thái các thành phần hệ thống**

| Thành phần | Đã triển khai | Đã kiểm thử | Kiểm chứng dữ liệu thật |
|---|:---:|:---:|:---:|
| Kiểm kê thông tin mô tả, chia tập, kiểm tra rò rỉ | ✓ | ✓ | ✓ |
| Khử trùng nội dung, danh sách tập phát triển sau khử trùng | ✓ | ✓ | ✓ |
| Danh sách mẫu dev-full / test-full | ✓ | ✓ | ✓ |
| Tiền xử lý âm thanh, ba chính sách biên độ | ✓ | ✓ | ✓ |
| Dữ liệu đóng gói (cấu trúc, chỉ mục) | ✓ | ✓ | ✓ |
| Đối chiếu tính tương đương dữ liệu đóng gói và Parquet | ✓ | ✓ | ✓ (trong phạm vi còn bản gốc) |
| Chỉ số EER, ROC-AUC, macro-F1, ngưỡng dev | ✓ | ✓ | ✓ |
| Bootstrap theo người nói | ✓ | ✓ | ✓ (dev cũ) |
| B0 với LFCC cũ (phụ thuộc lô) | ✓ | ✓ | ✓ (lịch sử) |
| B0 v2 với LFCC tính riêng từng mẫu | ✓ | ✓ | ✓ (3 epoch, trung gian) |
| Trạng thái lưu tiếp tục chính xác B0 | ✓ | ✓ | ✓ (CPU, GPU) |
| Khung XLS-R X0/X1 | ✓ | ✓ (mô hình thu nhỏ) | — |
| XLS-R với trọng số tiền huấn luyện | ✓ | ✓ | ✓ (đo chi phí GPU) |
| Sổ đăng ký mô hình, cổng đánh giá test | — | — | — |
| Suy luận cửa sổ trượt, chương trình minh họa | — | — | — |

---

## CHƯƠNG 4. THỰC NGHIỆM VÀ KẾT QUẢ HIỆN CÓ

> Mọi kết quả trong chương đo trên **development**, không có kết quả test. Các mục 4.2 và 4.3 mang nhãn **Lịch sử — LFCC phụ thuộc lô** do đặc trưng cũ bị lỗi. Chúng không phải kết quả chính thức của mô hình cơ sở.

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

### 4.5. B0 v2 với giới hạn 3 epoch (kết quả trung gian)

Ba seed chạy trên CPU với 15.885 mẫu train/57 shard và 38.777 mẫu dev-full/186 shard. Các chỉ số được tính lại từ điểm số đã lưu và khớp kết quả gốc.

Bảng 4.5 tổng hợp kết quả trung gian B0 v2 trên dev-full.

**Bảng 4.5 B0 v2 với giới hạn 3 epoch**

| Seed | EER epoch 1 | EER epoch 2 | EER epoch 3 | ROC-AUC tốt nhất | Macro-F1 tốt nhất |
|---|---:|---:|---:|---:|---:|
| 2026 | 13,91% | 10,81% | 8,23% | 0,973 | 91,23% |
| 2027 | 13,03% | 10,67% | 8,26% | 0,973 | 91,20% |
| 2028 | 13,96% | 13,34% | 7,93% | 0,976 | 91,55% |

EER tốt nhất qua ba seed có trung bình 8,14% ± 0,18 điểm phần trăm, với độ lệch chuẩn mẫu dùng mẫu số n−1.

Cả ba seed đạt tốt nhất đúng ở epoch cuối được phép và EER còn giảm, nên đây chưa phải kết quả chính thức; kết quả chính thức là bản chạy nối dài. Không so sánh trực tiếp với EER 8,33% của mô hình cũ vì khác tập đánh giá (dev-full 38.777 mẫu so với dev cũ 4.115 mẫu) và đặc trưng đã được sửa lỗi. Không báo cáo hàm mất mát cho B0 v2 vì chỉ lưu xác suất sau sigmoid, không thể kiểm lại chính xác từ dữ liệu đã lưu. Mọi chỉ số ở đây đều trên dev, chưa dùng test.

### 4.6. Đo chi phí XLS-R trên GPU

Phép đo dùng một GPU Tesla T4; 12/12 cấu hình chạy được, không có lỗi thiếu bộ nhớ. Mỗi cấu hình có 5 bước khởi động, 30 bước đo và 512 mẫu dev thuộc 160 shard. Các cấu hình lô 16 đọc 560 mẫu train/57 shard trong phép đo; dữ liệu nền gồm 15.885 mẫu train/57 shard và 38.777 mẫu dev-full/186 shard. Đây là đo chi phí và phép chiếu, không phải kết quả nhận dạng hoặc một epoch huấn luyện hoàn chỉnh.

Bảng 4.6 trình bày chi phí XLS-R với lô 16.

**Bảng 4.6 Chi phí XLS-R trên một GPU Tesla T4 với lô 16**

| Cấu hình | Giờ/epoch train | Giờ/lượt dev-full | VRAM reserved (GiB) | Tham số huấn luyện |
|---|---:|---:|---:|---:|
| X0 | 0,24 | 0,65 | 2,38 | 262.401 |
| 2 tầng | 0,28 | 0,67 | 2,78 | 25.454.849 |
| 4 tầng | 0,31 | 0,67 | 3,17 | 50.647.297 |
| 8 tầng | 0,39 | 0,66 | 3,96 | 101.032.193 |

Ngân sách tối đa là 87,7 GPU-giờ (109,6 giờ nếu cộng 25% dự phòng), theo trần 10 epoch và không giả định dừng sớm. Quota còn 24,09/30 giờ. Việc quota được làm mới hằng tuần mới chỉ là giả định, chưa được xác nhận.

### 4.7. Các kết quả chưa có

Bảng 4.7 tổng hợp các thí nghiệm chưa có kết quả.

**Bảng 4.7 Các thí nghiệm chưa có kết quả**

| Thí nghiệm | Trạng thái | Điều kiện để bắt đầu |
|---|---|---|
| B0 v2 nối dài, seed 2026/2027/2028 | Chưa chạy nối dài; đã có 3 epoch trung gian | Chế độ nối dài có kiểm soát, kiểm thử tính tương đương và xác nhận mở cổng |
| X0, XLS-R đóng băng | Chưa có kết quả khoa học; đã đo chi phí GPU | Chương trình huấn luyện khoa học, hỗ trợ ba seed và xác nhận mở cổng |
| X1, XLS-R tinh chỉnh từng phần | Chưa có kết quả khoa học; cấu hình đã chốt | Chương trình huấn luyện khoa học, hỗ trợ ba seed và chạy theo thứ tự đã chốt |
| Đánh giá test B0/X0/X1, trung bình ± độ lệch chuẩn theo seed, khoảng tin cậy | Chưa chạy | Sổ đăng ký mô hình và cổng đánh giá test; mọi mô hình đã đăng ký |
| Ablation băng thông và độ lớn âm (shortcut) | Chưa chạy | Sau khi khóa mô hình |
| Phân tích lỗi theo VC/AP/replay | Chưa chạy | Sau đánh giá test |

Bảng 4.8 dành cho so sánh ba mô hình; hiện chưa có kết quả để điền.

**Bảng 4.8 Bảng so sánh ba mô hình chưa có kết quả**

| Mô hình | Bộ mã hóa | EER | ROC-AUC | Macro-F1 | Thời gian huấn luyện | Bộ nhớ GPU cực đại |
|---|---|---:|---:|---:|---:|---:|
| LFCC + LCNN (B0 v2) | Không áp dụng | — | — | — | — | — |
| XLS-R + bộ phân loại (X0) | Đóng băng | — | — | — | — | — |
| XLS-R + bộ phân loại (X1) | Tinh chỉnh từng phần | — | — | — | — | — |

---

## CHƯƠNG 5. ĐÁNH GIÁ TIẾN ĐỘ, RỦI RO VÀ KẾ HOẠCH

### 5.1. Đối chiếu với lộ trình 14 tuần

Việc đối chiếu dùng đầu ra của mục 11 trong lộ trình 14 tuần và kế hoạch từng tuần, kết hợp kết quả đợt rà soát độc lập ngày 03/11/2026 ở mục 7 của bản rà soát.

Bảng 5.1 tổng hợp đối chiếu tiến độ với lộ trình 14 tuần.

**Bảng 5.1 Đối chiếu tiến độ với lộ trình 14 tuần**

| Tuần | Đầu ra theo kế hoạch | Thực tế | Đánh giá |
|---|---|---|---|
| 1 (28/09–04/10) | Khóa phạm vi, giao thức, chạy thử kỹ thuật quy trình | Đã khóa phạm vi và tên đề tài; chạy thử kỹ thuật B0 | Đạt |
| 2 (05–11/10) | Danh sách mẫu phát triển, kiểm kê dữ liệu | Danh sách mẫu v2, audit 20.000 dạng sóng | Đạt |
| 3 (12–18/10) | Khóa danh sách mẫu, chạy thử kỹ thuật GPU, ba chính sách biên độ | Danh sách mẫu v2 khóa, chạy thử kỹ thuật Kaggle đạt | Đạt |
| 4 (19–25/10) | Vòng lặp huấn luyện, ba pilot, chọn chính sách biên độ | Ba pilot hoàn tất, chọn chính sách ngày 23/10 | Đạt; kết quả sau đó mang nhãn lịch sử |
| 5 (26/10–01/11) | B0 có EER, ROC-AUC, macro-F1 hợp lệ | B0 v2 đã chạy 3 epoch; bản chính thức chờ nối dài | **Chưa đạt** |
| 6 (02–08/11) | XLS-R đóng băng chạy được một vòng huấn luyện | XLS-R đã chạy GPU ở mức đo chi phí, chưa chạy hết một epoch | Đạt một phần |
| 7–10 (09/11–06/12) | X0, X1, khóa cấu hình | Chưa bắt đầu phần chạy | Chưa đến hạn |
| 11–14 (07/12–03/01) | Test, phân tích, demo, bàn giao | Chưa bắt đầu | Chưa đến hạn |

**Nhận xét:**

- Đầu ra chính thức của tuần 5 chưa đạt vì B0 v2 còn chờ nối dài; tuần 6 đạt một phần nhờ phép đo chi phí XLS-R trên GPU, chưa đạt tiêu chí chạy trọn một epoch.
- Ngoài ra, các điều chỉnh giao thức đã bổ sung khối lượng so với lộ trình gốc: tập đánh giá mở rộng, ba seed huấn luyện, và huấn luyện lại B0.
- Lộ trình đã được điều chỉnh bằng các mốc trong giao thức hiện hành:
 - 29/11: khóa cấu hình X1.
 - 06/12: đăng ký mọi mô hình và seed.
 - Khoảng 13/12: đánh giá test một lần.
 - Sau 14/12: làm chương trình minh họa, trước hạn 03/01/2027.

### 5.2. Đánh giá theo cổng kiểm soát G0–G7

**Phương pháp:**

- Mỗi cổng được định nghĩa bằng tiêu chí đầu ra trong mục 12 của lộ trình.
- Một cổng chỉ được tính **đạt** khi mọi tiêu chí của cổng có bằng chứng ở mức "đã kiểm chứng trên dữ liệu thật".
- Các cổng có khối lượng công việc khác nhau. Vì vậy số cổng đạt **không** được quy đổi thành phần trăm khối lượng hoàn thành.

Bảng 5.2 tổng hợp trạng thái các cổng kiểm soát.

**Bảng 5.2 Trạng thái các cổng kiểm soát**

| Cổng | Tiêu chí chính | Trạng thái |
|---|---|---|
| G0. Dữ liệu sẵn sàng | Thông tin mô tả audit, tập dữ liệu không rò rỉ, chạy thử kỹ thuật âm thanh, EER có kiểm thử | **Đạt** |
| G1. Dữ liệu thí nghiệm được khóa | Danh sách mẫu khóa, speaker-disjoint, thống kê từng tập dữ liệu | **Đạt** (đã bổ sung tập phát triển và kiểm thử mở rộng) |
| G2. Pipeline huấn luyện hoạt động | Loader, hàm mất mát hữu hạn, trạng thái lưu, ROC-AUC/macro-F1/ngưỡng, tiếp tục được | **Đạt** |
| G3. Mô hình cơ sở hoàn thành | Ba pilot, khóa chính sách biên độ, B0 có chỉ số hợp lệ | **Chưa đạt**: pilot và chính sách đã có; B0 v2 mới có kết quả trung gian ở trần 3 epoch, còn chờ bản nối dài |
| G4. XLS-R đóng băng hoàn thành | Bộ mã hóa không cập nhật, hội tụ, so sánh với B0 | Chưa đạt |
| G5. XLS-R tinh chỉnh từng phần hoàn thành | Ghi rõ số tầng, trạng thái lưu, log tài nguyên | Chưa đạt |
| G6. Đánh giá cuối hoàn thành | Bảng ba mô hình, phân tích lỗi, shortcut | Chưa đạt |
| G7. Bàn giao hoàn thành | Demo, hướng dẫn tổng quan, báo cáo và slide thống nhất | Chưa đạt |

**Kết quả: 3/8 cổng đạt (G0, G1, G2).** Các cổng còn lại chứa các thí nghiệm cốt lõi trả lời RQ1 và RQ2.

### 5.3. Trạng thái các sản phẩm bắt buộc

Bảng 5.3 tổng hợp trạng thái các sản phẩm bắt buộc.

**Bảng 5.3 Trạng thái các sản phẩm bắt buộc**

| # | Sản phẩm (theo mục 3 của lộ trình) | Trạng thái |
|---|---|---|
| 1 | Quy trình dữ liệu tái lập từ thông tin mô tả đến huấn luyện/phát triển/kiểm thử | Phần lớn đã có. Công cụ tải thông tin mô tả còn phụ thuộc gói datasets chưa có trong danh sách thư viện phụ thuộc |
| 2 | Tập dữ liệu theo người nói và báo cáo rò rỉ | Đã có |
| 3 | LFCC + LCNN hoạt động ổn định | Pipeline đã kiểm chứng; B0 v2 đã có 3 epoch trung gian; bản chính thức chờ nối dài |
| 4 | XLS-R đóng băng | Chưa có |
| 5 | XLS-R tinh chỉnh từng phần | Chưa có |
| 6 | Bảng so sánh trên cùng tập test | Chưa có |
| 7 | Phân tích lỗi theo VC, AP, replay | Chưa có |
| 8 | Chương trình minh họa | Chưa có (phần chương trình hiện chưa có nội dung) |
| 9 | Báo cáo, slide, mã nguồn, hướng dẫn | Đang làm dần; báo cáo này là bản thảo |

### 5.4. Rủi ro chính

Bảng 5.4 tổng hợp rủi ro và biện pháp xử lý.

**Bảng 5.4 Rủi ro và biện pháp xử lý**

| Mức | Rủi ro | Biện pháp |
|---|---|---|
| Cao | B0 v2 còn chờ nối dài và XLS-R chưa có chương trình huấn luyện khoa học, trong khi các mốc cố định | Ưu tiên nối dài B0 v2 và tích hợp chương trình huấn luyện XLS-R |
| Cao | Ngân sách 87,7 GPU-giờ, hoặc 109,6 giờ có dự phòng, phụ thuộc giả định quota được làm mới hằng tuần | Kiểm tra quota thực tế sau mỗi lần làm mới và điều chỉnh bậc phương án theo giao thức |
| Trung bình | B0 dùng lô 8 còn XLS-R dùng lô 16 | Nêu khác biệt khi diễn giải so sánh |
| Trung bình | Tốc độ học cố định có thể bất lợi cho XLS-R | Ghi rõ giới hạn; không diễn giải cấu hình đã chọn là tối ưu |
| Trung bình | Seed huấn luyện khác 2026 chưa được danh tính lượt chạy XLS-R chấp nhận | Tách seed dữ liệu và seed huấn luyện như đã làm ở B0 v2, kèm kiểm thử |
| Trung bình | Chưa có sổ đăng ký mô hình và cổng đánh giá test; loader mới kiểm soát ở mức quyền đọc tập kiểm thử | Triển khai sổ đăng ký mô hình ghi trạng thái lưu, mã băm, ngưỡng, chỉ số dev trước 06/12 |
| Trung bình | Thay đổi quan trọng chưa ghi nhận phiên bản mã nguồn; bản mã nguồn đã ghi nhận chưa tái tạo được trạng thái làm việc | Rà soát rồi ghi nhận phiên bản mã nguồn theo từng nhóm thay đổi sau khi được duyệt |
| Trung bình | Quy mô train nhỏ (15.885 mẫu, khoảng 10,5% tập huấn luyện gốc); replay ít (120 mẫu mỗi tập đánh giá) | Nêu rõ giới hạn; báo cáo theo nhóm kèm khoảng tin cậy; không kết luận cho replay |
| Thấp | Tài liệu vận hành có đoạn trạng thái cũ (hướng dẫn tổng quan và thiết kế so sánh trước đó) | Đồng bộ theo giao thức hiện hành và sản phẩm dữ liệu đã kiểm chứng |

### 5.5. Kế hoạch tiếp theo

Đợt rà soát độc lập ngày 03/11/2026 đề xuất thứ tự ưu tiên ở mục 9 của bản rà soát; mỗi bước dưới đây có tiêu chí chấp nhận riêng.

Bảng 5.5 tổng hợp các bước tiếp theo và tiêu chí chấp nhận.

**Bảng 5.5 Các bước tiếp theo và tiêu chí chấp nhận**

| # | Công việc | Tiêu chí chấp nhận | Mốc |
|---|---|---|---|
| 1 | Nối dài B0 v2 cho ba seed từ trạng thái lưu cuối | Kiểm thử nối dài trùng chạy liên tục, ghi trần cũ/mới, chọn trạng thái trên dev-full | Tuần 6–7 |
| 2 | Hoàn thiện chương trình huấn luyện khoa học XLS-R | Bộ điều khiển, lịch sử, trạng thái tốt nhất/mới nhất, ba seed, kiểm thử và xác nhận mở cổng | Tuần 6–7 |
| 3 | Seed 2026: chạy X1 với 2, 4, 8 tầng ở tốc độ học 10⁻⁵ và X0 | Đánh giá dev-full sau mỗi epoch theo trần 10 epoch, dừng sớm sau 2 lần không cải thiện | Trước 29/11 |
| 4 | Thử tốc độ học 3·10⁻⁵ ở mức tầng tốt nhất | Khóa cấu hình X1 bằng EER dev-full | 29/11 |
| 5 | Chạy thêm seed 2027 và 2028 cho X1 cuối và X0 | Đủ ba seed, giữ nguyên cấu hình đã chọn | Trước 06/12 |
| 6 | Sổ đăng ký mô hình và cổng đánh giá test | Mọi mô hình và seed được đăng ký | 06/12 |
| 7 | Đánh giá test, phân tích lỗi, ablation shortcut | Bảng B0/X0/X1 đủ nguồn gốc | khoảng 13/12 |
| 8 | Suy luận cửa sổ trượt (4 s, bước 2 s) và chương trình minh họa | Chạy được với tệp ngắn và dài, cùng tiền xử lý và ngưỡng dev | sau 14/12 |
| 9 | Hoàn thiện báo cáo, slide, hướng dẫn tổng quan, kiểm tra tái lập | Số liệu thống nhất giữa các tài liệu | 03/01/2027 |

---

## KẾT LUẬN TẠM THỜI

Đến hiện tại, đồ án đã hoàn thành phần nền tảng có bằng chứng trên dữ liệu thật:

- Giao thức dữ liệu speaker-disjoint có khử trùng nội dung.
- Tập đánh giá mở rộng 38.777 / 30.698 mẫu.
- Dữ liệu dạng sóng đóng gói.
- Quy trình tiền xử lý thống nhất.
- Các chỉ số đánh giá kèm khoảng tin cậy theo người nói.
- Cơ chế huấn luyện có thể tiếp tục chính xác qua nhiều phiên GPU.

Ba cổng kiểm soát đầu tiên (G0–G2) đã đạt.

Về thực nghiệm, chính sách biên độ đã được chọn từ các pilot lịch sử. Lỗi LFCC phụ thuộc lô đã được định lượng và sửa bằng kiểm thử hồi quy. B0 v2 đã hoàn tất ba seed ở giới hạn 3 epoch trên CPU với dữ liệu đóng gói; EER tốt nhất trên dev-full trung bình 8,14% ± 0,18 điểm phần trăm. Cả ba seed còn cải thiện ở epoch cuối, nên bản chạy nối dài mới là kết quả chính thức. Không so sánh trực tiếp số này với 8,33% trên dev cũ.

Hai câu hỏi nghiên cứu RQ1 và RQ2 chưa thể trả lời vì chưa có kết quả chính thức của B0 v2, X0 và X1. XLS-R đã được đo chi phí trên GPU, với ngân sách tối đa 87,7 GPU-giờ, hoặc 109,6 giờ có 25% dự phòng; tính khả thi còn phụ thuộc giả định quota được làm mới hằng tuần. Đường găng là nối dài B0 v2, hoàn thiện chương trình huấn luyện XLS-R và thực hiện các lượt chạy đã chốt, giữ mốc khóa X1 ngày 29/11/2026, đăng ký mô hình ngày 06/12, đánh giá test khoảng 13/12, làm chương trình minh họa sau 14/12 và bàn giao trước hạn 03/01/2027.

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
