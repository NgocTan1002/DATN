# Đánh giá độc lập dự án sau các thay đổi — 02/11/2026

> Báo cáo này lưu nội dung đánh giá đã trình bày trong cuộc trò chuyện. Các trạng thái, số liệu và kết quả kiểm tra phản ánh thời điểm review ngày 02/11/2026, không phải trạng thái trực tiếp của các tiến trình đang chạy. Việc lưu báo cáo không đồng nghĩa các đề xuất đã được triển khai hoặc các quyết định nghiên cứu đã được thay đổi.

## Phần 1 — Executive Summary

**Dự án đã tiến bộ thực chất về vận hành và đã có kết quả development có thể kiểm tra lại. Tuy nhiên, chưa đủ cơ sở để coi baseline B0 đã hoàn thiện hoặc kết quả nghiên cứu chính đã đáng tin cậy để nghiệm thu.**

Phát hiện quan trọng nhất của phiên review là **đặc trưng LFCC hiện phụ thuộc vào các mẫu khác trong batch**. Đã kiểm chứng trên audio development thật: cùng checkpoint, cùng waveform và cùng threshold, chạy trong batch ban đầu và chạy riêng có thể cho **hai quyết định phân loại khác nhau**. Đây là vấn đề cần xử lý trước khi sử dụng B0 làm đối chứng cuối cho XLS-R.

Những tiến bộ được xác minh:

- Ba pilot amplitude đã hoàn tất; EER, ROC-AUC, macro-F1 và threshold tính lại khớp score lưu trong báo cáo.
- B0 đã có hai lần đánh giá development và checkpoint best; checkpoint best/latest cục bộ có checksum khớp.
- Cơ chế huấn luyện nhiều phiên, khôi phục trạng thái và kiểm tra cấu hình Kaggle đã có triển khai cùng bằng chứng thực chạy.
- D025 mở rộng development/test theo độ phủ speaker; D026 thu gọn thí nghiệm normalization là những điều chỉnh có lý do.

Những phần còn thiếu:

- Chưa có B0 cuối theo giao thức đánh giá mới.
- Chưa có kết quả pretrained XLS-R trên VSASV.
- Chưa có evaluation manifest đầy đủ, đường đánh giá test được khóa, bảng so sánh B0/X0/X1 hoặc demo.
- Code chưa hỗ trợ đầy đủ kế hoạch nhiều seed của D026.

**Mức độ tin cậy:** cao đối với các lỗi tái hiện được, số liệu tính lại và trạng thái code; có giới hạn đối với toàn bộ lịch sử GPU và chất lượng dữ liệu nguồn.

Phạm vi đối chiếu là commit `823bd17` ngày 21/10 → HEAD `0df18d3`, cộng thay đổi chưa commit. Có [bản phân tích tiến độ trước đây](phan_tich_tong_the_tien_do_du_an.md), nhưng **không tìm thấy báo cáo review có thể xác định là của Claude Code**. Vì vậy, phần đối chiếu dưới đây không gán các nhận định cũ cho Claude.

Phiên review giữ chế độ chỉ đọc: không sửa file, không training, không tải model/dataset, không cài dependency hoặc gửi Kaggle. Báo cáo Markdown này được tạo sau đó theo yêu cầu riêng của người dùng; không chạy lại thí nghiệm khi lưu báo cáo.

## Phần 2 — Đánh giá những thay đổi gần đây

| Thay đổi | Mục tiêu | Tác động thực tế | Đánh giá | Bằng chứng |
|---|---|---|---|---|
| Hoàn tất ba pilot, chọn `peak` | Chọn amplitude theo development | Cả ba đủ 896 bước và 4.115 development; metric tính lại khớp | **Cải thiện đã xác minh**, nhưng chưa chứng minh ưu thế thống kê | [So sánh pilot](../reports/b0_pilot_amplitude_comparison.md), D022 |
| Runner B0 đầy đủ | Đánh giá theo epoch, early stopping, giữ best/latest | Đã chạy qua hai biên epoch; best ở bước 3.972 | **Cải thiện vận hành đã xác minh**; kế thừa lỗi LFCC cũ | Commit `246da0c`, [controller, dòng 171](../src/training/full_training.py#L171) |
| Chuyển checkpoint giữa phiên Kaggle | Tiếp tục run dài qua giới hạn phiên | Có chuỗi báo cáo đến bước 5.120; checkpoint hiện có khớp checksum | **Cải thiện đã xác minh**; tăng chi phí quản lý artifact | Commit `2b9773d`, [kiểm chứng phiên 10](../reports/b0_full_session_10_verification.md) |
| Kiểm tra runner/config trước gửi | Ngăn gửi sai cấu hình nhúng như v24 | Có so khớp cấu hình và test hồi quy được lưu | **Xử lý đúng nguyên nhân đã xác định**; v25 chưa có output được kiểm tra trong phiên này | [Verifier, dòng 413](../scripts/kaggle_workflow.py#L413), [sự cố v24](../reports/b0_full_session_11_v24_failure.md) |
| Khung XLS-R và checkpoint riêng | Chuẩn bị X0/X1 trong lúc chờ B0 | Có freeze/unfreeze, pooling, optimizer groups, exact-resume CPU | **Cải thiện kỹ thuật**, chưa có bằng chứng hiệu năng pretrained | Commit `f032820`, [CPU smoke](../reports/xlsr_cpu_smoke.json) |
| D025: dev-full/test-full | Giảm phụ thuộc vào ít speaker | Quy tắc khử trùng đã khai báo; builder đã có; chưa có manifest thật | **Hợp lý nhưng chưa hoàn tất**; cổng xác minh đầu vào còn thiếu | [D025, dòng 225](decisions.md#L225) |
| Nạp shard tăng dần | Hạn chế dung lượng đĩa | Đã có chunk thật; luồng lỗi vẫn có thể xóa nguồn quá sớm | **Cải thiện một phần, phát sinh lỗi vận hành** | [Luồng xóa, dòng 269](../scripts/ingest_evaluation_shards.py#L269) |
| Bootstrap theo speaker | Đánh giá bất định thay vì chỉ nhìn EER điểm | Test đạt; CI tính lại cho thấy ưu thế `peak` chưa chắc chắn | **Cải thiện đã xác minh** | [Bootstrap](../src/metrics/bootstrap.py), [tests](../tests/test_bootstrap_metrics.py) |
| D026 thay pilot normalization, thêm nhiều seed | Giảm tìm kiếm cấu hình, đo dao động huấn luyện | Quyết định đã ghi; identity XLS-R vẫn từ chối seed bổ sung | **Kế hoạch hợp lý, triển khai chưa theo kịp** | [D026, dòng 251](decisions.md#L251), [ràng buộc seed, dòng 84](../src/training/xlsr_checkpoint.py#L84) |

Không có thay đổi ở metadata, split hoặc `configs/audio.json` trong diff từ mốc đối chiếu.

## Phần 3 — Kiểm tra lại vấn đề trước đây

“Trạng thái cũ” trong bảng lấy từ tài liệu lưu trong repository, không suy đoán nội dung báo cáo ngoài repository.

| Vấn đề | Trạng thái cũ | Trạng thái hiện tại | Bằng chứng | Kết luận |
|---|---|---|---|---|
| Chưa chọn amplitude policy | Ba pilot chưa hoàn tất trong bản phân tích cũ | Đã hoàn tất, chọn `peak` theo quy tắc khai báo | Ba report và score | **Đã giải quyết về lựa chọn vận hành** |
| Chưa có B0 đầy đủ | Mới đạt cổng kỹ thuật | Đã chạy đến bước 5.120, có metric hai epoch; chưa kết thúc | Report phiên 10 | **Đã xử lý một phần** |
| Rò rỉ nội dung manifest v1 | Có duplicate xuyên train/dev | V2 có audit 20.000 hash duy nhất; checksum manifest hiện tại khớp | [Audit v2](../reports/development_manifest_20k_v2_content_audit.md) | **Đã giải quyết trong phạm vi v2 được audit** |
| Duplicate train–test | Đã thấy overlap trong pool cục bộ | D025 có quy tắc xử lý; manifest cuối chưa được tạo | D025, builder | **Chưa giải quyết bằng artifact thực tế** |
| Parquet là nút thắt | Đã đo và có packed PoC | B0 hiện vẫn dùng Parquet; chưa có packed production đầy đủ | [Packed PoC](../reports/packed_artifact_benchmark.md) | **Chưa giải quyết ở quy mô production** |
| XLS-R chưa triển khai | Chưa có X0/X1 | Có scaffold và kiểm chứng CPU tổng hợp | Report XLS-R | **Đã xử lý phần kỹ thuật ban đầu** |
| Normalization XLS-R chưa rõ | D024 dự kiến pilot hai nhánh | D026 chọn normalization theo processor | D026 | **Đổi cách tiếp cận hợp lý; cần tích hợp vào runner** |
| Chưa có test cuối/phân tích lỗi/demo | Chưa thực hiện | Vẫn chưa có sản phẩm cuối | Thư mục `app` chỉ có `.gitkeep` | **Chưa giải quyết** |
| Tiền xử lý “nhất quán” | Tài liệu mô tả pipeline đã thống nhất | Waveform thống nhất nhưng LFCC phụ thuộc batch | Kiểm tra thực tế phiên review | **Nhận định cũ cần thu hẹp** |

## Phần 4 — Những điểm đang tốt

**1. Dữ liệu được quản lý bằng artifact và checksum cụ thể.**

Manifest hiện tại đúng SHA-256 đã khóa, gồm 15.885 train và 4.115 development trên 66 shard. Kiểm tra lại split gốc cho thấy overlap speaker/file giữa ba closed split bằng 0. Báo cáo content audit chứa đúng tập file của manifest và 20.000 hash duy nhất. Không băm lại toàn bộ waveform trong phiên review.

**2. Metric có thể kiểm tra từ đầu vào đã lưu.**

Ba pilot và development của checkpoint B0 best đều lưu đủ score/label/record. Tính lại EER, threshold, AUC và macro-F1 cho kết quả khớp hoàn toàn; từng tập có đúng 4.115 file development duy nhất và metadata khớp manifest.

**3. Việc tiếp tục training có kiểm soát trạng thái thực chất.**

Controller có lịch đánh giá theo epoch, strict improvement, giữ checkpoint sớm khi hòa, patience và trạng thái evaluation pending. Đây là những cơ chế cần thiết trong môi trường phải chia nhiều phiên, không phải độ phức tạp thuần hình thức.

**4. Dự án đã loại phương án tối ưu không đạt bằng benchmark.**

Việc gỡ batch-read sau khi không đạt cổng và giữ packed ở mức PoC là hợp lý. Không có bằng chứng để yêu cầu quay lại phương án batch-read đã bị loại.

**5. Giới hạn nghiên cứu được ghi tương đối thẳng thắn.**

D022 không tuyên bố `peak` tốt hơn mọi metric; D025 thừa nhận thay development sau khi B0 đã chạy; XLS-R được mô tả là scaffold kỹ thuật. Các giới hạn này nên được giữ trong báo cáo cuối.

## Phần 5 — Những điểm chưa tốt

| Vấn đề | Mức độ | Bằng chứng | Tác động | Hướng xử lý |
|---|---|---|---|---|
| LFCC phụ thuộc thành phần batch | **P0 trước nghiệm thu B0/test/demo** | [Lệnh LFCC, dòng 303](../scripts/train_b0.py#L303), cấu hình `log_lf=false`; đã tái hiện trên audio thật | Cùng file có thể đổi score và quyết định; threshold không chuyển nguyên vẹn sang chạy từng file | Xác định phiên bản đặc trưng đúng, kiểm thử tính độc lập theo mẫu và đánh giá ảnh hưởng trên dev |
| Builder tin chỉ mục chunk chưa xác minh | **P1** | [Đọc chunk, dòng 97](../scripts/make_evaluation_manifest.py#L97) chỉ kiểm sự tồn tại của marker/index | Có thể chứng nhận “content-disjoint” từ chỉ mục đã hỏng | Kiểm checksum/identity và tính nhất quán nhãn trước khử trùng |
| Builder không kiểm nhãn theo `utt_type` | **P1**, cùng cổng trên | [Kiểm metadata, dòng 189](../scripts/make_evaluation_manifest.py#L189) chỉ đối chiếu speaker/type | Đầu vào tổng hợp `bonafide, binary_label=1` vẫn trả trạng thái đạt | Từ chối nhãn không đúng ánh xạ nhị phân |
| D026 nhiều seed chưa được code hỗ trợ | **P1** | Identity XLS-R từ chối 2027/2028; [B0 runner, dòng 96](../scripts/train_b0_full.py#L96) hardcode 2026 | Không thể thực hiện nguyên trạng phương án A/B | Tách seed dữ liệu khỏi seed huấn luyện trong cấu hình và identity |
| Chưa có đường đánh giá đầy đủ | **P1** | Chưa có manifest full; [packed loader, dòng 410](../src/data/packed.py#L410) chỉ nhận train/dev | Chưa thể vận hành test-full qua đường packed hiện hành | Hoàn thiện artifact, loader đánh giá và registry/test gate |
| Xóa download trong `finally` dù ingest lỗi | **P2** | [Dòng 269–270](../scripts/ingest_evaluation_shards.py#L269) | Mất bản nguồn đã tải để chẩn đoán/thử lại; phải tải lại | Chỉ xóa nguồn sau khi chunk được xác minh thành công |
| Tài liệu trạng thái mâu thuẫn | **P2** | `current_state` còn ghi code đã commit là chưa commit; đoạn cũ vẫn yêu cầu pilot D024 | Phiên sau dễ tiếp tục sai bước | Đồng bộ trạng thái hiện hành, ghi rõ tài liệu lịch sử |

### Phát hiện LFCC: bằng chứng và giới hạn

Đường hiện tại đưa waveform dạng `(batch, time)` vào LFCC. Với `log_lf=false`, phép đổi sang dB nhận tensor ba chiều và dùng **một cutoff chung**. Hành vi này được mô tả trong [tài liệu chính thức Torchaudio](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.amplitude_to_DB.html).

Đã nạp checkpoint best bước 3.972, dùng policy `peak`, và so sánh cùng mẫu trên CPU:

| File development | Score trong batch 8 ban đầu | Score chạy riêng | Threshold | Kết quả |
|---|---:|---:|---:|---|
| `id01126/bonafide/00001.wav` | 0,007011 | 0,220251 | 0,007008 | Score thay đổi lớn; quyết định giữ nguyên |
| `id01070/voice_conversion/id01070_vc_01698.wav` | 0,006975 | 0,009152 | 0,007008 | **Quyết định thay đổi** |

Hai phép kiểm tra dùng các batch gốc, mỗi batch 8 mẫu, lần lượt trên 7 và 4 shard. Trước đó, kiểm tra 16 mẫu/11 shard cũng xác nhận đặc trưng và score thay đổi khi đổi cách ghép batch.

Đây là **lỗi đã có từ commit `d083558`**, được runner mới tái sử dụng; không phải regression mới do D023 tạo ra.

Chưa thể kết luận EER toàn tập bị tăng hay giảm bao nhiêu, hoặc policy thắng sẽ đổi. **Metric lưu vẫn đúng với score đã sinh; tính ổn định của cách sinh score mới là vấn đề.**

Hai trục review độc lập cùng phát hiện lỗi xóa nguồn khi ingest thất bại và thiếu kiểm tra chỉ mục. Trục **quy ước** xác nhận lỗi xóa trái hợp đồng của module; trục **đặc tả** xác nhận cổng chỉ mục chưa bảo đảm các ràng buộc D025. Không có lý do để đề xuất refactor kiến trúc lớn chỉ từ các lỗi này.

## Phần 6 — Chất lượng AI/ML và tính hợp lệ thực nghiệm

### Dữ liệu và tiền xử lý

Ánh xạ `0 = bonafide`, `1 = spoof` nhất quán trong loader chính. Tuy nhiên, điều này chỉ xác minh ánh xạ theo metadata; chưa chứng minh mọi bản ghi nguồn đều có nhãn ngữ nghĩa đúng.

Speaker/file disjointness được kiểm tra lại và đạt. Content disjointness của v2 có báo cáo audit phù hợp với manifest hiện tại. Còn hai giới hạn:

- Không xác minh được mọi speaker ID tương ứng với các cá nhân khác nhau.
- Hash waveform chính xác không chứng minh không tồn tại các bản ghi cùng nội dung đã qua biến đổi.

**D025 là điều chỉnh đáng giữ.** Development cũ có 31 speaker, trong đó VC chỉ đến từ 3 speaker. Tính lại lỗi VC ở checkpoint best cho thấy 192/214 lỗi tập trung vào `id01070`. Dev-full có 22 speaker VC; vì vậy mở rộng độ phủ speaker có giá trị nghiên cứu rõ ràng.

Nhưng dev-full vẫn là tập dùng để lựa chọn. Nó không thay thế test độc lập, và việc B0 đã được dừng/chọn theo dev cũ cần được công bố khi so sánh.

### Mô hình và huấn luyện

B0 là LCNN tối thiểu có Max-Feature-Map, một raw logit và BCEWithLogitsLoss; cách ghép đầu ra/loss phù hợp bài toán nhị phân. Không có cơ sở gọi đây là tái lập đầy đủ một baseline chính thức.

D023 có cấu hình rõ: Adam, learning rate cố định, batch 8, tối đa ba epoch, đánh giá sau epoch và patience 1. Đây là **ngân sách vận hành**, chưa phải bằng chứng hội tụ hoặc cấu hình tối ưu.

XLS-R đã có các phần quan trọng: freeze/unfreeze, giữ module đóng băng ở eval, mean/std pooling và optimizer groups. Nhưng báo cáo CPU dùng backbone tổng cộng 7.249 tham số với audio tổng hợp, **không phải pretrained XLS-R 300M trên VSASV**. Chưa thể kết luận về chất lượng hay khả năng tổng quát hóa của X0/X1.

D026 chọn normalization theo processor là mặc định có lý do theo cấu hình pretrained đã ghim. Tuy nhiên, đây vẫn là lựa chọn thiết kế; runner khoa học phải thực sự áp dụng nó ở cả train, evaluation và inference.

### Metric và kết quả

Kết quả tính lại từ report:

| Run | Development | Shard của manifest | EER | ROC-AUC | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Pilot `none` | 4.115 | 66 | 10,0394% | 0,971496 | 0,893614 |
| Pilot `peak` | 4.115 | 66 | 9,1207% | 0,970066 | 0,903263 |
| Pilot `rms_dbfs` | 4.115 | 66 | 11,5486% | 0,956941 | 0,877914 |
| B0 best epoch 2 | 4.115 | 66 | 8,3333% | 0,975127 | 0,911656 |

Mỗi run dùng nguồn train 15.885 mẫu của manifest 20.000 mẫu. Các số này là **kết quả development lịch sử với cách tạo score hiện tại**, không phải kết quả test hoặc dev-full. Nguồn: ba report [none](../reports/b0_pilot_none.json), [peak](../reports/b0_pilot_peak.json), [rms_dbfs](../reports/b0_pilot_rms_dbfs.json) và [B0 phiên 10](../reports/b0_peak_full_session_10.json).

Bootstrap theo speaker, 1.000 lần lấy mẫu, seed 2026:

- Pilot peak: CI 95% EER khoảng **[2,47%; 15,30%]**.
- Hiệu EER `none − peak`: **0,92 điểm phần trăm**, CI khoảng **[−2,73; +3,61] điểm**.
- Có 999 lần lấy mẫu hợp lệ chứa đủ hai lớp.

Vì CI hiệu chứa 0, lựa chọn `peak` đúng quy tắc khai báo nhưng **chưa chứng minh `peak` tốt hơn `none` một cách chắc chắn**. Bootstrap này giữ các score đã lưu cố định; nó không sửa lỗi LFCC.

Không có bằng chứng trong các đường code/report đã xem cho thấy dùng metric test để chọn mô hình. Điều đó không xác minh được hoạt động ngoài repository.

### Những kết luận chưa được chứng minh

- XLS-R tốt hơn B0.
- Fine-tune từng phần tốt hơn đóng băng.
- B0 đã hội tụ hoặc có hiệu năng cuối 8,3333%.
- Resample 16 kHz đã loại hết shortcut băng thông.
- Hệ thống tổng quát hóa theo vùng miền, nguồn ghi hoặc TTS engine.

Band-limit, phân tích theo nhóm và kiểm tra đầu vào thực tế vẫn cần thiết. Metadata hiện tại không đủ để kiểm chứng đầy đủ một số chiều sai lệch đó.

## Phần 7 — Đánh giá tiến độ thực tế

| Hạng mục | Trạng thái | Bằng chứng | Phần còn thiếu | Ưu tiên |
|---|---|---|---|---|
| Split và development v2 | Có artifact, audit và kiểm tra phù hợp | Manifest/checksum/audit khớp | Giới hạn danh tính speaker và nội dung biến đổi | Duy trì |
| Metric/bootstrap | **Đã kiểm thử lại** | 26 test đạt; metric report khớp | Tích hợp vào đánh giá cuối | P1 |
| Pipeline B0 | Đã hoạt động trên GPU theo report; kiểm tra cục bộ checkpoint đạt | Chuỗi phiên, best/latest | Xử lý LFCC và hoàn tất/chốt run hợp lệ | P0 |
| Packed production | Có PoC và chunk ingest | PoC 24 mẫu; tại lần kiểm tra có 115 bộ chunk | Lắp artifact, kiểm toàn vẹn, parity, benchmark | P1 |
| Dev-full/test-full | Có quyết định và builder | D025, code/tests | Manifest thật và cổng xác minh đầy đủ | P1 |
| X0/X1 | Có code, kiểm chứng kỹ thuật CPU | Report tổng hợp | Pretrained load, CUDA, runner, training | P1 |
| Nhiều seed | Có kế hoạch | D026 | Cấu hình/identity/runner hỗ trợ | P1 |
| Kết quả nghiên cứu cuối | **Chưa hoàn thiện** | Chưa có bảng test B0/X0/X1 | Thực nghiệm, CI, phân tích lỗi | P1 |
| Demo | Chưa triển khai | `app/.gitkeep` | Inference, sliding window, giao diện | P1 sau mô hình |
| Báo cáo/slide cuối | Có nền tài liệu | Đề cương, báo cáo tiến độ, hướng dẫn | Kết quả chính và diễn giải thống nhất | P1 |

Đánh giá riêng:

- **Kỹ thuật:** nền tảng vận hành khá đầy đủ, còn lỗi đúng đắn đáng kể.
- **Pipeline ML:** B0 đã hoạt động; pipeline khoa học XLS-R còn thiếu.
- **Độ tin cậy thực nghiệm:** phép tính metric đáng tin; suy luận từ kết quả còn bị giới hạn bởi batching, speaker và thiếu test.
- **Tài liệu/tái lập:** có dấu vết tốt, nhưng trạng thái hiện hành bị phân tán và mâu thuẫn.
- **Sẵn sàng bảo vệ:** có thể trình bày tốt phần chuẩn bị và phương pháp; chưa đủ sản phẩm để bảo vệ kết luận chính về XLS-R.

Không đưa phần trăm hoàn thành vì chưa có trọng số và bộ tiêu chí nghiệm thu định lượng.

## Phần 8 — Kế hoạch cũ và kế hoạch mới

**Nên giữ:**

- B0 → X0 → X1; nhị phân; speaker-disjoint; threshold từ development.
- Giới hạn phạm vi snapshot công khai.
- Ưu tiên đo thực tế, giữ artifact/checksum và kiểm chứng trước run dài.
- Loại AASIST, open-set, external dataset và full fine-tuning khỏi đường găng bắt buộc.

**Nên điều chỉnh:**

- Đưa kiểm tra LFCC lên trước việc “chốt B0 rồi chuyển sang XLS-R”.
- Dùng D025/D026 làm căn cứ hiện hành; roadmap ngày 12/10 và bản phân tích ngày 21/10 chỉ còn là lịch sử tiến độ.
- Tách “đã có builder” khỏi “đã có evaluation artifact đạt”.
- Tách seed split/manifest 2026 khỏi seed huấn luyện được D026 cho phép.
- Hoàn thiện packed vì chi phí đọc hiện tại đã thành phụ thuộc của dev-full và nhiều seed.

**Nên bỏ khỏi công việc tiếp theo:**

- Chạy pilot normalization D024 đã bị thay thế.
- Tiếp tục tối ưu batch-read đã không đạt benchmark.
- Mở rộng train lên 40.000 khi câu hỏi chính trên 15.885 mẫu chưa được trả lời.

**Cần xem lại điều kiện “không chạy lại B0”.**

Giữ run lịch sử là đúng để bảo toàn bằng chứng. Nhưng sau phát hiện LFCC, chưa thể khóa trước rằng B0 tuyệt đối không cần chạy lại. Nếu sửa đặc trưng làm thay đổi định nghĩa mô hình hoặc phân phối train/evaluation, phải có quyết định mới về tái huấn luyện và cách so sánh. Không nên sửa âm thầm source của chuỗi D023 đang tồn tại.

D025 có lý do tốt hơn phương án chỉ dùng development ít speaker. Đổi lại, B0 và XLS-R sẽ có lịch sử lựa chọn khác nhau; báo cáo cuối phải mô tả điều đó thay vì gọi phép so sánh hoàn toàn đồng nhất.

## Phần 9 — Roadmap ưu tiên

### 1. P0 — Khóa cách tạo đặc trưng B0 đúng và xác định ảnh hưởng

- **Mục tiêu/vấn đề:** một file có đầu vào mô hình và score ổn định; lỗi đã làm đổi quyết định trên development thật.
- **Phương án:** bổ sung kiểm thử chạy riêng/ghép batch/đổi thứ tự; áp LFCC với chiều kênh rõ ràng hoặc cách tính độc lập từng mẫu, rồi đối chiếu bằng chứng.
- **Phụ thuộc:** trước chốt baseline, threshold, test và demo.
- **Rủi ro:** thay đổi đặc trưng có thể đòi hỏi huấn luyện lại; không được sửa tiếp giữa chuỗi cũ.
- **Nghiệm thu:** đặc trưng và score không đổi ngoài sai số số học khi đổi batch; ảnh hưởng trên development được báo cáo.
- **Kiểm chứng:** test hồi quy, kiểm tra checkpoint thật, quyết định mới ghi rõ cách xử lý kết quả lịch sử.

### 2. P1 — Hoàn thiện cổng dữ liệu đánh giá

- **Mục tiêu/vấn đề:** manifest chỉ được tạo từ chunk toàn vẹn, nhãn đúng; hiện builder bỏ qua checksum và nhận nhãn sai.
- **Phương án:** xác minh marker/index/identity, dữ liệu chunk và ánh xạ nhãn; giữ quy tắc D025.
- **Phụ thuộc:** trước tạo manifest chính thức và chọn checkpoint trên dev-full.
- **Rủi ro:** có thể phát hiện chunk cần tái tạo; phải giữ đủ dấu vết nguồn.
- **Nghiệm thu:** từ chối index bị sửa, nhãn sai, chunk thiếu/hỏng; manifest hợp lệ có số mẫu/shard và ledger rõ ràng.
- **Kiểm chứng:** test lỗi có chủ đích và audit artifact thật.

### 3. P2 — Sửa giữ nguồn khi ingest thất bại; P1 — hoàn thiện packed

- **Mục tiêu/vấn đề:** tiếp tục ingest an toàn và tạo backend đủ nhanh cho đánh giá/multi-seed.
- **Phương án:** chỉ xóa Parquet sau xác minh thành công; lắp chunk thành artifact theo cách giới hạn đỉnh dung lượng; hỗ trợ test split.
- **Phụ thuộc:** cổng dữ liệu ở bước 2; cần cho các run mới.
- **Rủi ro:** thiếu đĩa, mất ledger/hash gốc, thay `source_index` làm lệch crop.
- **Nghiệm thu:** có thể ngắt–tiếp tục; waveform/metadata/crop khớp đường Parquet; toàn bộ artifact qua kiểm toàn vẹn.
- **Kiểm chứng:** fault tests, parity trên đầy đủ dữ liệu áp dụng và benchmark Kaggle đại diện. Không dùng tốc độ PoC 24 mẫu làm dự toán production.

### 4. P1 — Hoàn tất XLS-R pretrained và runner khoa học

- **Mục tiêu/vấn đề:** chuyển scaffold thành X0/X1 thực sự.
- **Phương án:** kiểm trọng số revision ghim, load offline, tích hợp normalization D026, checkpoint/controller/best và seed huấn luyện.
- **Phụ thuộc:** dữ liệu/packed và quyết định đặc trưng/so sánh B0.
- **Rủi ro:** VRAM, khác biệt stack CUDA, sai freeze hoặc preprocessing.
- **Nghiệm thu:** pretrained CUDA smoke đạt; chỉ phần cho phép có gradient; ngắt–resume đúng; seed bổ sung được hỗ trợ.
- **Kiểm chứng:** kiểm thử tích hợp và smoke CUDA có log tài nguyên; sau đó mới mở run khoa học.

### 5. P1 — Khóa D027 và thực hiện B0/X0/X1 theo ngân sách thật

- **Mục tiêu/vấn đề:** trả lời câu hỏi nghiên cứu chính trong quota.
- **Phương án:** đo train và dev-full; chọn bậc A/B/C theo D026, khai báo tầng mở, learning rate, epoch, patience và seed trước chạy.
- **Phụ thuộc:** bước 1–4; B0 phải có quyết định xử lý rõ ràng.
- **Rủi ro:** dành quá nhiều quota cho tìm kiếm cấu hình hoặc đánh giá lặp.
- **Nghiệm thu:** mỗi run có identity, dữ liệu, score dev, best, threshold và lý do dừng; báo mọi seed.
- **Kiểm chứng:** đối chiếu registry với artifact; không chọn seed/cấu hình bằng test.

### 6. P1 — Khóa registry và đánh giá test

- **Mục tiêu/vấn đề:** hiện chưa có bằng chứng tổng quát hóa cuối.
- **Phương án:** khóa model/checkpoint/preprocessing/threshold và manifest; triển khai test gate rồi đánh giá các run đăng ký.
- **Phụ thuộc:** hoàn tất lựa chọn bằng development.
- **Rủi ro:** dùng kết quả test để sửa mô hình; độ bất định cao ở replay.
- **Nghiệm thu:** bảng cùng protocol cho B0/X0/X1, mọi seed, CI theo speaker và kết quả từng attack.
- **Kiểm chứng:** tính lại metric từ score; kiểm tra threshold đến từ đúng development/checkpoint.

### 7. P1/P2 — Phân tích lỗi, demo và bàn giao

- **Mục tiêu/vấn đề:** hoàn thành sản phẩm và diễn giải đúng kết quả.
- **Phương án:** phân tích speaker/attack/băng thông/biên độ; khóa trước quy tắc ablation; dùng chung preprocessing trong demo; đồng bộ tài liệu.
- **Phụ thuộc:** mô hình và đánh giá đã khóa.
- **Rủi ro:** lấy ablation test để quay lại tối ưu mô hình; gọi sigmoid là xác suất đã hiệu chuẩn.
- **Nghiệm thu:** demo xử lý file ngắn/dài đúng giao thức; báo cáo trả lời RQ bằng số liệu truy vết được.
- **Kiểm chứng:** integration tests inference, kiểm bảng báo cáo với report và thử quy trình tái lập.

Các mốc 29/11, 06/12 và khoảng 13/12 của D026 có thể giữ làm mục tiêu; **chưa có timing pretrained XLS-R để xác nhận tính khả thi**.

## Phần 10 — Kết luận độc lập

**1. Dự án có tốt hơn trước không?**

Có, về triển khai và bằng chứng: ba pilot hoàn tất, B0 có evaluation thật, resume nhiều phiên hoạt động và XLS-R đã có scaffold. Chưa thể nói chất lượng mô hình cuối hoặc mục tiêu nghiên cứu chính đã được cải thiện tương ứng.

**2. Quyết định thay đổi nào hợp lý?**

D025 mở rộng độ phủ speaker; D026 dùng normalization theo processor và ưu tiên nhiều seed có điều kiện; giữ ngân sách khai báo trước; sửa đúng nguyên nhân cấu hình nhúng v24.

**3. Quyết định nào cần xem lại?**

Cam kết giữ nguyên B0 mà không xét tái huấn luyện sau lỗi LFCC; coi kế hoạch nhiều seed là sẵn sàng khi identity còn cấm; tiếp tục dựa vào các mô tả trạng thái đã lỗi thời.

**4. Vấn đề quan trọng nào chưa giải quyết?**

Tính độc lập theo mẫu của B0, artifact đánh giá đầy đủ, pretrained X0/X1, hỗ trợ multi-seed, test cuối, shortcut analysis và demo.

**5. Có rủi ro nghiêm trọng với kết quả không?**

Có. Lỗi LFCC đã được chứng minh làm thay đổi quyết định. Đồng thời, development cũ có độ phủ speaker hạn chế. Hai vấn đề này cần xử lý trước khi dùng các số hiện tại làm kết luận cuối.

**6. Ba việc quan trọng nhất tiếp theo là gì?**

1. Xử lý và định lượng ảnh hưởng của LFCC phụ thuộc batch.
2. Hoàn thiện cổng toàn vẹn dữ liệu và artifact dev-full/test-full.
3. Đưa pretrained XLS-R lên runner CUDA, đo ngân sách và khóa thực nghiệm X0/X1.

**7. Nếu giữ hướng hiện tại, rủi ro lớn nhất là gì?**

Hoàn thiện thêm hạ tầng và nhật ký nhưng vẫn thiếu thí nghiệm chính; hoặc hoàn thành bảng kết quả bằng các đường preprocessing không tương đương rồi rút ra kết luận vượt bằng chứng.

### Kiểm tra đã thực hiện trong phiên review

**34 kiểm thử được chọn an toàn đã đạt:**

```text
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test*metrics.py' -v
→ 26/26 đạt

.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_b0_full_training_budget.py -v
→ 3/3 đạt

.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_xlsr_normalization.py -v
→ 5/5 đạt
```

Ngoài ra đã kiểm tra split overlap, checksum manifest/checkpoint, tính lại metric/bootstrap và tái hiện các lỗi nêu trên. `git diff --check` đạt. Không chạy toàn suite vì một số test ghi thư mục tạm trong repository hoặc thực hiện tối ưu mô hình, vượt phạm vi chỉ đọc của phiên review.

Trạng thái Git cuối phiên review giữ nguyên các thay đổi đã có; không cập nhật tài liệu hoặc tự commit trong phiên review. Bước lưu Markdown sau đó chỉ tạo báo cáo này theo yêu cầu của người dùng, không sửa code, cấu hình, dữ liệu hoặc quyết định nghiên cứu.
