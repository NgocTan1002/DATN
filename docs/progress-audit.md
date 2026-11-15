# Kiểm toán tiến độ kỹ thuật đồ án tốt nghiệp

Ngày kiểm toán: **03/11/2026 (UTC+7)**. Vai trò: Technical Project Auditor.

Đề tài: **Nghiên cứu phát hiện tiếng nói giả mạo tiếng Việt dựa trên mô hình XLS-R và kỹ thuật tinh chỉnh từng phần**.

## 1. Phạm vi, phương pháp và kết luận chính

Mốc Git được kiểm tra: nhánh `main`, HEAD `86189ee7430fc4c1491658fc82b598deeb5d98a8`, cộng các thay đổi có sẵn trong working tree. Kết luận về mã hiện tại không được hiểu là kết luận chỉ về HEAD. Kiểm toán không sửa source code, cấu hình nghiên cứu, metadata hay split; không tải dữ liệu/trọng số, chạy thí nghiệm khoa học, gửi Kaggle, commit hoặc push.

Nguồn đối chiếu gồm [AGENTS.md](../AGENTS.md), toàn bộ [trạng thái bàn giao](current_state.md), [README](../README.md), [quyết định D001–D027](decisions.md), lộ trình gốc trong Git, lộ trình 14 tuần hiện lưu, kế hoạch tuần W40–W44, các báo cáo thực nghiệm liên quan, source và test của đường dữ liệu/mô hình/huấn luyện/metric/checkpoint. Có kiểm kê toàn repository và đọc các đường thực thi chính; đây không phải cam kết review từng dòng mã hoặc kiểm chứng lại toàn bộ phiên GPU lịch sử.

Phân loại bằng chứng:

- **Xác minh trong phiên:** có lệnh chạy hoặc kiểm tra artifact trực tiếp trong phiên này.
- **Có bằng chứng lịch sử:** có report/log lưu trong repository; chỉ xác nhận trong phạm vi report, không mặc định môi trường hiện tại vẫn tương đương.
- **Đang triển khai:** có mã hoặc artifact nhưng còn cổng chưa đạt.
- **Chưa có bằng chứng thực hiện:** không tìm thấy sản phẩm/lượt chạy đáp ứng tiêu chí trong phạm vi đã kiểm tra; không đồng nghĩa chứng minh không tồn tại ở máy khác.

**Kết luận:** nền tảng dữ liệu, metric, tiền xử lý và cơ chế huấn luyện đã có bằng chứng hoạt động. Tuy nhiên, **baseline khoa học B0 v2 chưa hoàn thành, pretrained XLS-R chưa được chứng minh hoạt động trên VSASV, và chưa có bảng so sánh test B0/X0/X1 hay demo**. Không đủ cơ sở nghiệm thu kết quả nghiên cứu chính. Không suy đoán tỷ lệ hoàn thành. Bằng chứng: ma trận tại mục 3–5, kiểm thử tại mục 6 và D027.

## 2. Mốc Git và tình trạng bàn giao

| Mốc | Nội dung có thể xác nhận từ Git | Giới hạn diễn giải |
|---|---|---|
| `d1d3f49` (01/09) | Khởi tạo repository, metadata audit, đề cương và lộ trình ban đầu | Chỉ chứng minh hồ sơ/mã đã được lưu |
| `740df8e` (02/10) | Triển khai LFCC–LCNN smoke | Smoke không phải baseline khoa học cuối |
| `5af07f0` (09/10) | Manifest v2 khử trùng nội dung | Phạm vi train/dev subset |
| `246da0c`, `2b9773d` (02/11) | Engine B0 đầy đủ và chuyển checkpoint giữa phiên | Độ tin cậy kết quả mô hình còn phụ thuộc đặc trưng LFCC |
| `f032820`, `4b34c71` (02/11) | Khung XLS-R/checkpoint và bootstrap theo speaker | Không phải kết quả XLS-R pretrained |
| `7e1eb27`, `d6958ad` (02/11) | Sửa LFCC theo mẫu và đo ảnh hưởng lỗi cũ | Chưa chứng minh B0 v2 đã huấn luyện |
| `c20c549`, `86189ee` (02/11) | Manifest dev/test đầy đủ có kiểm toàn vẹn; D026/D027 và review độc lập | D027 thay cách diễn giải chuỗi B0 cũ |

Nguồn: `git log -15 --date=short --format='%h %ad %s'`, `git log --reverse`, `git show <commit>:<path>`.

Working tree đầu phiên đã có thay đổi tại `scripts/train_b0_full.py`, `scripts/ingest_evaluation_shards.py`, `src/data/packed.py`, `src/data/__init__.py`, test ingest và hai báo cáo leakage. Các file B0 v2, builder packed, parity và test liên quan còn chưa được theo dõi. Vì vậy, **checkout HEAD sạch chưa tái tạo đầy đủ năng lực của working tree được kiểm tra**. Hai DOCX tham khảo cũng chưa được theo dõi; kiểm toán không thay đổi chúng. Bằng chứng: `git status --short`, `git diff --stat`, bản chụp đầu phiên tại `.cache/progress-audit-2026-11-03/git-status-before.txt`.

Ngày commit không được dùng thay ngày thực hiện thí nghiệm: nhiều thay đổi được commit cùng ngày 02/11 trong khi report/plan ghi hoạt động diễn ra trước đó.

## 3. Hạng mục đã hoàn thành trong phạm vi có bằng chứng

| Hạng mục | Kết luận và phạm vi đã xác minh | Bằng chứng |
|---|---|---|
| Tên đề tài, phạm vi nhị phân, ba cấu hình B0/X0/X1 | Đã khóa trong hồ sơ; trạng thái đề cương được tài liệu ghi nhận, không xác minh độc lập việc phê duyệt bên ngoài | D008, D010, D011; [W40](plans/weekly/2026-W40.md) |
| Metadata audit | Chạy lại đạt trên 220.963 dòng/1.141 speaker; không suy ra chất lượng ngữ nghĩa của mọi nhãn âm thanh | [Script](../scripts/audit_metadata.py), mục 6 |
| Split theo người nói | Kiểm tra lại schema/coverage và điều kiện speaker/file của closed/open split đạt; seed dữ liệu giữ 2026 | [Checker](../scripts/check_leakage.py), [split protocol](split_protocol.md), mục 6 |
| Tiền xử lý và amplitude | Có mono/resample 16 kHz, đoạn 64.000 mẫu, random/center crop, repeat-then-trim, ba policy; test và smoke peak trên audio thật đạt | [Loader](../src/data/vsasv.py), [segment](../src/data/segment.py), [amplitude](../src/data/amplitude.py), [audio protocol](audio_protocol.md), mục 6 |
| Manifest train/dev v2 | Audit lịch sử đọc lại đủ 20.000 waveform, không trùng hash; manifest/loader được kiểm lại ở phiên này | [Audit nội dung v2](../reports/development_manifest_20k_v2_content_audit.md), [manifest](../data/manifests/development_20k_v2.csv) |
| Dev-full/test-full sau khử trùng | Đã có manifest thật; giữ 38.777 dev và 30.698 test; report ghi hash gốc trùng chéo bằng 0 và speaker-disjoint đạt | [Report D025](../reports/closed_eval_full_v1.md), [builder](../scripts/make_evaluation_manifest.py), commit `c20c549` |
| Đóng gói dữ liệu T07 | Có ba artifact thật; kiểm tra hiện tại đối chiếu index checksum, kích thước waveform, checksum manifest và thứ tự file; chưa tương đương cổng parity T08 | [Report packed](../reports/packed_artifacts_v1.md), [builder](../scripts/build_packed_artifacts.py), mục 6 |
| Metric và threshold | EER, ROC-AUC, macro-F1, chọn threshold trên dev và tính ngoài dev với threshold được cung cấp có test; tính lại metric lịch sử khớp | [evaluation](../src/metrics/evaluation.py), [EER](../src/metrics/eer.py), [classification](../src/metrics/classification.py), mục 6 |
| CI bootstrap theo speaker | Có API, test và phép áp dụng trên dev lịch sử; chưa phải CI test của mô hình cuối | [bootstrap](../src/metrics/bootstrap.py), [tests](../tests/test_bootstrap_metrics.py), commit `4b34c71` |
| Pipeline LFCC–LCNN và exact resume | Có kiểm chứng CPU hiện tại và report CPU/CUDA lịch sử; không đồng nghĩa baseline khoa học đã hoàn thiện | [full training](../src/training/full_training.py), [sampler](../src/training/sampler.py), [CPU smoke](../reports/b0_full_cpu_smoke.json), [CUDA smoke](../reports/b0_full_cuda_smoke.json) |
| Workflow Kaggle | Có report thực chạy GPU, chuyển checkpoint nhiều phiên và kiểm cấu hình nhúng; phiên này không truy vấn lại dịch vụ | [Workflow](../scripts/kaggle_workflow.py), [resume bootstrap](../deploy/kaggle/resume_checkpoint.py), [phiên 10](../reports/b0_full_session_10_verification.md), [sự cố v24](../reports/b0_full_session_11_v24_failure.md) |
| Ba pilot amplitude | Đã chạy đủ cùng ngân sách, metric tính lại khớp; D022 chọn peak; tất cả thuộc lịch sử LFCC phụ thuộc batch theo D027 | [So sánh pilot](../reports/b0_pilot_amplitude_comparison.md), D022/D027 |
| Sửa LFCC phụ thuộc batch | Mã đã dùng chiều kênh riêng, có test độc lập thành phần/thứ tự batch; đã commit | [`lfcc_features`](../src/models/lfcc_lcnn.py), [test hồi quy](../tests/test_lfcc_batch_independence.py), commit `7e1eb27` |
| Đo ảnh hưởng lỗi LFCC | Có phép đo đủ 4.115 dev cũ, không còn chỉ là ví dụ vài mẫu | [Report](../reports/lfcc_batch_effect.md), [script](../scripts/measure_lfcc_batch_effect.py), commit `d6958ad` |
| Khung kỹ thuật XLS-R | X0/X1 có freeze/unfreeze, pooling, head, optimizer groups và restore trên mô hình nhỏ ngẫu nhiên; smoke CPU được chạy lại | [model](../src/models/xlsr.py), [smoke](../scripts/smoke_test_xlsr.py), mục 6 |
| Checkpoint XLS-R mức thấp | Có exact-resume CPU và khóa identity trong test; chưa bao gồm controller khoa học hoàn chỉnh | [checkpoint](../src/training/xlsr_checkpoint.py), [tests](../tests/test_xlsr_checkpoint.py), [preflight](../reports/xlsr_exact_resume_preflight.md) |

Quy mô dữ liệu phải được phân biệt:

| Artifact | Mẫu thực dùng | Shard nguồn | Ý nghĩa |
|---|---:|---:|---|
| Manifest development v2 ban đầu | 15.885 train + 4.115 dev | 66 tổng hợp | Phạm vi ba pilot và chuỗi B0 cũ |
| Packed train | 15.885 | 57 | Train giữ nguyên theo D025/D027 |
| Packed dev-full | 38.777 | 186 | Dùng chọn checkpoint/threshold cho run mới |
| Packed test-full | 30.698 | 162 | Đã chuẩn bị dữ liệu; chưa có kết quả mô hình cuối |

Nguồn: [manifest v2](../reports/development_manifest_20k_v2.md), [packed](../reports/packed_artifacts_v1.md). Shard có thể dùng chung giữa các tập, không cộng các cột shard thành tổng. Report D025 ghi 265 shard nạp cho dev/test, 69.964 bản ghi trước khử trùng. Phiên audit không băm lại toàn bộ waveform gốc hoặc toàn bộ tệp packed hàng chục GiB.

## 4. Hạng mục đang triển khai

| Hạng mục | Bằng chứng hiện có | Cổng còn thiếu; trạng thái kiểm toán |
|---|---|---|
| Parity packed/Parquet T08 | [Script](../scripts/check_packed_parity.py) và [test](../tests/test_check_packed_parity.py) đã có trong working tree | Không tìm thấy `reports/packed_parity_v1.*`; chưa xác minh tương đương trên dữ liệu thật. Script chỉ so các dòng có Parquet cục bộ, phải công bố coverage khi chạy |
| Runner B0 v2 | [Runner](../scripts/train_b0_full.py) nhận packed/dev-full, tách data seed và training seed; [tests](../tests/test_b0_v2_runner.py) có đối chứng resume tổng hợp | [Config](../configs/b0_v2_training.json) vẫn `ready_for_training=false`, checksum waveform train/dev còn `null`; chưa có benchmark CPU dữ liệu thật và report run B0 v2 |
| Chuyển đường dữ liệu sang packed | [Packed loader](../src/data/packed.py) đã có test split và `allow_test` mặc định false; artifact T07 tồn tại | Còn parity, benchmark và xác nhận cổng vận hành; không coi đã thay thế production chỉ vì loader đã viết |
| Sửa luồng ingest xóa download quá sớm | Working tree [ingest](../scripts/ingest_evaluation_shards.py) giữ bản tải đúng hash khi ingest lỗi, tái sử dụng lần sau; [tests](../tests/test_ingest_evaluation_shards.py) kiểm các nhánh này | Có kiểm chứng hồi quy bằng fixture; chưa có lỗi ingest thật được tái hiện lại trong phiên này; thay đổi chưa commit |
| Chuẩn bị pretrained XLS-R | [Candidate config](../configs/xlsr_pretrained_candidate.json) ghim revision/config/preprocessor | `weights_downloaded=false`, `weights_sha256=null`; chưa có load offline và CUDA trên checkpoint thật |
| Áp dụng normalization D026 | [Hàm normalization](../src/data/xlsr.py) có test; quyết định dùng processor normalization đã ghi | Candidate config/README còn mô tả lựa chọn cũ; chưa có runner khoa học nối toàn đường dữ liệu–normalization–model |
| Phương án nhiều seed | B0 v2 nhận 2026/2027/2028 trong working tree | [`XLSRTrainingRunIdentity.__post_init__`](../src/training/xlsr_checkpoint.py) còn từ chối seed khác 2026; chưa thể thực hiện nguyên trạng D026 cho XLS-R |
| Báo cáo học thuật | Có [báo cáo tiến độ cũ](bao_cao_tien_do_do_an_2026-10-20.md), đề cương DOCX và tài liệu giải thích | Không coi là báo cáo cuối; chưa có kết quả chính để hoàn thiện phần thực nghiệm/kết luận |

## 5. Hạng mục chưa có bằng chứng hoàn thành

1. **B0 v2 huấn luyện trên dữ liệu thật đến điều kiện dừng hợp lệ**, chọn checkpoint/threshold bằng dev-full và chạy các seed đã đăng ký. Không tìm thấy report `b0_v2*`; config còn đóng cổng. D027 yêu cầu train lại, không được thay bằng EER chuỗi D023 cũ.
2. **X0 pretrained XLS-R trên VSASV:** chưa có checkpoint/metric khoa học, một epoch thực tế, số đo VRAM và throughput trên model đã ghim. Report [CPU smoke](../reports/xlsr_cpu_smoke.json) ghi rõ 0 mẫu/0 shard VSASV, chỉ mô hình thu nhỏ ngẫu nhiên.
3. **X1 pretrained tinh chỉnh từng phần:** chưa có các mức số tầng, learning rate, ngân sách và bậc A/B/C được khóa bằng D028; chưa có pilot/run dữ liệu thật. Bằng chứng: D026/D027 và các trường chưa khóa trong candidate config.
4. **Registry và entrypoint đánh giá test có cổng khóa:** chưa tìm thấy phần triển khai trong `src/`, `scripts/`, `tests/`. Cờ `allow_test=True` của loader là kiểm soát truy cập mức thấp, chưa phải registry đăng ký model, checkpoint, threshold và cơ chế đánh giá cuối theo D025.
5. **Bảng test B0/X0/X1, mean ± SD giữa seed, CI theo speaker và phân tích lỗi cuối:** chưa có report tương ứng. [Metric bootstrap](../src/metrics/bootstrap.py) và metric lịch sử chỉ là năng lực nền, không thay cho thí nghiệm cuối.
6. **Ablation band-limit/loudness và phân tích shortcut cuối:** chưa tìm thấy runner/report tương ứng; đây là yêu cầu trong mục 10.4 của [lộ trình](../lo_trinh_hoan_chinh_do_an_deepfake_tieng_viet.md) và D007.
7. **Suy luận sliding-window và demo nhận tệp dài:** `app/` hiện chỉ có `.gitkeep`; không tìm thấy triển khai sliding-window trong source/script/test. D009 yêu cầu cửa sổ 4 giây, bước nhảy 2 giây, trung bình xác suất.
8. **Bộ bàn giao cuối:** chưa có báo cáo/slide cuối thống nhất số liệu, demo và kiểm tra tái lập trọn quy trình. Có hướng dẫn thành phần trong `docs/guides/`, chưa đủ tiêu chí G7 của lộ trình.

AASIST, fine-tune toàn bộ XLS-R, open-set và dữ liệu ngoài miền là **tùy chọn** theo D008 và mục 4.2 của lộ trình; việc chưa thực hiện không phải thiếu sót của phần bắt buộc.

## 6. Kiểm chứng thực tế trong phiên

### 6.1. Điều kiện chạy và giới hạn

Các lệnh dùng `.venv/Scripts/python.exe`, không dùng `python` toàn hệ thống đang trỏ sang Python 3.11. Thiết lập `PYTHONUTF8=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TEMP`/`TMP` trỏ tới `.cache/progress-audit-2026-11-03/tmp` trong workspace. Không cài dependency. Kết quả và log kiểm toán cục bộ đặt trong `.cache/progress-audit-2026-11-03/`, không đưa cache/artifact lớn vào Git.

Các kiểm tra độc lập có thời gian chạy chồng nhau; số giây ở đây là thời gian kiểm tra, **không phải benchmark huấn luyện**. Compileall chỉ xác minh biên dịch Python; repository không có quy trình build sản phẩm/demo để gọi là build ứng dụng thành công.

### 6.2. Lệnh và kết quả

| Kiểm tra | Kết quả quan sát | Thời gian thực đo / log |
|---|---|---|
| Toàn bộ unittest | **302/302 đạt**, `OK`, không có test skip; mã thoát 0 | Unittest báo 272,910 giây; process gồm khởi động/kết thúc 287,438 giây; `unit-tests.log` |
| Compileall `src scripts tests` | Đạt, mã thoát 0 | 0,844 giây; `compileall.log` |
| Verify environment | Đạt; torch/torchaudio `2.11.0+cpu`, CUDA khả dụng `False` | 18,125 giây; `environment.log` |
| Leakage checker | Đạt speaker/file/protocol; không chạy audio-root hashing | 8,719 giây; `leakage.log`, `leakage.json` |
| Metadata audit | Đạt 220.963 dòng/1.141 speaker | 2,766 giây; `metadata.log`, `metadata.json` |
| Smoke loader manifest v2, peak | Đạt train 15.885/dev 4.115, coverage 100%, batch hữu hạn và tái lập | 29,079 giây; `manifest-smoke.log` |
| XLS-R smoke CPU | X0/X1 đạt trên backbone nhỏ ngẫu nhiên, 2 waveform tổng hợp mỗi mode, 0 mẫu/0 shard VSASV | 86,047 giây; `xlsr-smoke.log`, `xlsr-smoke.json` |

Các log trong bảng nằm dưới `.cache/progress-audit-2026-11-03/`; tổng hợp lệnh/mã thoát có trong `validation.json` và `smoke-validation.json`. Kết quả 302 test chỉ áp dụng working tree đầu phiên được kiểm tra; không gán ngược cho HEAD hoặc source GPU lịch sử.

Trong suite này, `test_b0_v2_runner` đã xác minh resume liên tục/ngắt trên dữ liệu packed **tổng hợp**, ghi seed 2027/2026, từ chối đổi seed/checksum/đường Parquet; `test_ingest_evaluation_shards` xác minh giữ/tái sử dụng bản tải khi ingest lỗi; các test LFCC kiểm tính độc lập theo mẫu; nhóm XLS-R kiểm freeze/unfreeze, normalization và exact-resume CPU. Bằng chứng là các dòng `... ok` trong log và các test tương ứng ở mục 3–4. Không nâng các kết quả fixture này thành xác nhận run dữ liệu thật.

Lệnh cụ thể (đường dẫn đầu ra audit được chuyển hướng để không ghi đè report người dùng):

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m compileall -q src scripts tests
.venv/Scripts/python.exe scripts/verify_environment.py
.venv/Scripts/python.exe scripts/check_leakage.py --json-report .cache/progress-audit-2026-11-03/leakage.json --markdown-report .cache/progress-audit-2026-11-03/leakage.md
.venv/Scripts/python.exe scripts/audit_metadata.py --json-output .cache/progress-audit-2026-11-03/metadata.json --markdown-output .cache/progress-audit-2026-11-03/metadata.md
.venv/Scripts/python.exe scripts/smoke_test_development_manifest_loader.py --amplitude-policy peak
.venv/Scripts/python.exe scripts/smoke_test_xlsr.py --report .cache/progress-audit-2026-11-03/xlsr-smoke.json
```

### 6.3. Kiểm tra artifact và metric

Kiểm tra trực tiếp ba packed artifact đều đạt các điều kiện: số dòng index khớp manifest theo split, thứ tự file khớp, SHA-256 index khớp metadata, SHA-256 manifest khớp metadata và kích thước `waveforms.f32` bằng `total_samples × 4`. Kết quả: train 15.885 mẫu/57 shard; dev 38.777/186; test 30.698/162. Giao nhau file, speaker và **hash float32 được lưu trong index** giữa từng cặp tập đều bằng 0. Bằng chứng: `.cache/progress-audit-2026-11-03/artifact-check.json`, metadata/index trong `data/processed/packed/`.

Đây là kiểm tra chỉ mục và cấu trúc artifact. Hash float32 của packed không phải hash waveform gốc float64 theo D017; trong phiên này không băm lại nội dung toàn bộ tệp waveform. Kết luận khử trùng hash gốc dựa trên report D025 và audit D017 đã dẫn, không dựa riêng vào hash packed.

Kiểm lại output v25 cục bộ: `LATEST.json` dẫn tới latest/best có checksum `checkpoint.pt` và `state.json` khớp ở cả hai generation; report ghi `paused`, global step 5.632, train 15.885/dev 4.115. Log: `.cache/progress-audit-2026-11-03/v25-check.json`; nguồn `kaggle_outputs/b0-full-session-11-v25/`. Chỉ kiểm hai pointer đích, không xác minh lại cả 104 generation và toàn bộ RNG/optimizer. D027 vẫn yêu cầu giữ chuỗi này làm lịch sử.

Metric được tính lại từ `labels`, `scores`, `loss`, `utt_type` đã lưu, qua `evaluate_development_scores`: ba pilot none/peak/rms_dbfs và checkpoint best trong report phiên 10, mỗi tập 4.115 mẫu. EER, EER threshold, ROC-AUC, macro-F1 và decision threshold đều khớp giá trị lưu, sai khác quan sát bằng 0. Log JSON: `.cache/progress-audit-2026-11-03/metric-recheck.json`.

**Diễn giải:** tính metric đúng từ score không chứng minh cách sinh score cũ phù hợp. Theo D027, ba pilot và EER best cũ 8,3333% chỉ được giữ làm lịch sử “LFCC phụ thuộc batch”. [Phép đo ảnh hưởng](../reports/lfcc_batch_effect.md) ghi EER 8,3333% → 8,3990%, 31 quyết định đảo ở threshold cũ; không được gọi 8,3990% là kết quả B0 v2.

Không chạy lại full training, không chạy mô hình/metric trên test, không chạy toàn bộ parity packed/Parquet, không xác minh lại toàn bộ generation CUDA lịch sử. Smoke loader chỉ đọc batch đại diện; coverage 100% là coverage manifest đối với shard hiện có, không phải đã đọc lại mọi waveform trong phiên.

Kiểm tra bàn giao: SHA-256 đầu/cuối phiên của **127 file** thuộc `src/`, `scripts/`, `tests/`, `configs/`, `data/metadata/`, `data/splits/` không đổi, không có file mới trong các thư mục đó. `git diff --check` trả mã 0; các liên kết tương đối trong báo cáo đều tồn tại. Bản trạng thái Git cuối được so với đầu phiên để giữ nguyên mọi thay đổi có sẵn. Chỉ thêm báo cáo này và cập nhật ghi chú kiểm toán trong `docs/current_state.md` theo AGENTS.md; không sửa quyết định nghiên cứu. Bằng chứng kiểm tra cục bộ: `final-integrity.json`, bản chụp Git đầu/cuối phiên trong thư mục cache audit.

## 7. Đối chiếu với kế hoạch ban đầu và kế hoạch hiện hành

Nguồn gốc thực sự là lộ trình tại commit `d1d3f49` (16 tuần, có các nhánh nghiên cứu rộng). D008/D010 đã thu hẹp và khóa đề tài; vì vậy không dùng channel consistency, cross-dataset hay AASIST trong bản cũ làm nghĩa vụ nghiệm thu hiện tại. Nguồn đối chiếu tiến độ chính là mục 11–12 của [lộ trình 14 tuần](../lo_trinh_hoan_chinh_do_an_deepfake_tieng_viet.md), các file tuần và các quyết định sửa đổi sau đó.

| Mốc kế hoạch | Thực tế có bằng chứng | Đánh giá |
|---|---|---|
| Tuần 1, 28/09–04/10: phạm vi và smoke pipeline | D010/D011, W40 và report smoke; test hiện tại kiểm lại các thành phần | Đạt cổng kỹ thuật trong phạm vi smoke, không suy thành mô hình cuối |
| Tuần 2, 05–11/10: manifest phát triển và audit dữ liệu | Manifest v2, audit độc lập 20.000 waveform, W41 | Có bằng chứng hoàn thành |
| Tuần 3, 12–18/10: benchmark, GPU smoke, khóa manifest, ba policy | W42, report batch-read/packed PoC/Kaggle và D019 | Có bằng chứng lịch sử; phương án batch-read bị loại theo cổng đo, không phải tính năng cần khôi phục |
| Tuần 4, 19–25/10: runner, ba pilot, chọn amplitude | W43, ba report pilot, D022 ngày 23/10 | Hoàn tất lựa chọn trước hạn; D027 bổ sung giới hạn do LFCC cũ |
| Tuần 5, 26/10–01/11: baseline hoàn chỉnh | W44 có hai evaluation; D027 ghi v25 dừng ở bước 5.632, chưa đạt trần cũ 5.958; baseline phải train lại | **Chưa đạt đầu ra baseline hợp lệ**; không tiếp tục chuỗi cũ để “hoàn tất kế hoạch” |
| Tuần 6, 02–08/11: X0, embedding và một epoch | Có scaffold CPU tổng hợp; chưa có trọng số/pretrained epoch | Đang trong tuần thực hiện tại ngày audit; chưa đạt đầu ra, chưa thể gọi trễ hạn tuần 6 |
| Tuần 7–10, 09/11–06/12: X0/X1 và khóa cấu hình | D026/D027 điều chỉnh thành các bậc ngân sách, nhiều seed; chưa có số đo CUDA thật | Chưa đến hạn; cần xử lý đường găng B0 v2 và XLS-R để bảo vệ mốc 29/11, 06/12 |
| Tuần 11–14, 07/12–03/01: test, ablation, demo, bàn giao | Chưa có sản phẩm cuối; D026 đặt test khoảng 13/12 và demo sau 14/12 | Chưa đến hạn; không đánh dấu hoàn thành trước khi có artifact và kết quả chạy |

**Các thay đổi được ghi nhận chính thức:** D017 khử trùng; D025 mở rộng dev/test nhưng giữ train 15.885; D026 chọn normalization theo processor và các bậc X1; D027 train lại B0 bằng LFCC đúng, chọn trên dev-full, đọc packed, dừng chuỗi cũ. Không phải mọi ô chưa tích trong kế hoạch tuần cũ đều là backlog hiện hành: mục tiếp tục D023 bị D027 thay thế, pilot normalization D024 bị D026 thay thế.

Theo tiêu chí G0–G7: G0/G1 có bằng chứng dữ liệu trong phạm vi nêu trên, G2 có bằng chứng pipeline kỹ thuật; **G3 chưa đạt theo D027, G4/G5 chưa có run pretrained thực, G6/G7 chưa đạt**. Đây là đánh giá theo đầu ra, không phải phần trăm tiến độ.

## 8. Rủi ro và vấn đề còn tồn tại

| Mức ưu tiên | Vấn đề và bằng chứng | Hệ quả / việc cần giải quyết |
|---|---|---|
| P0 trước kết luận nghiên cứu | B0 cũ dùng LFCC phụ thuộc batch; D027 hủy vai trò kết quả chính | Phải hoàn thành B0 v2 trước bảng so sánh; giữ nhãn lịch sử trên mọi bảng/slide cũ |
| P1 trước chạy B0 v2 | Config còn checksum `null`, readiness false; chưa có report parity/benchmark thật | Không vượt cổng bằng cách chỉ đổi readiness; cần report T08, khóa identity và đo chi phí |
| P1 trước run XLS-R | Chưa tải/băm trọng số, chưa offline load/CUDA benchmark, chưa khóa D028 | Chưa xác nhận được VRAM và ngân sách cho 300M; dùng số đo thật chọn bậc theo D026 |
| P1 cho kế hoạch nhiều seed | XLS-R identity khóa seed 2026 | Không chạy được 2027/2028 qua hợp đồng hiện tại; cần tách data seed và training seed có test |
| P1 trước test cuối | Chưa có registry/test entrypoint; loader có thể mở test bằng cờ tường minh | Cờ loader không bảo đảm quy trình khóa model/threshold và một lượt đánh giá; cần cổng cấp thí nghiệm |
| P1 cho tái lập | Các thay đổi quan trọng packed/B0 v2/parity chưa commit; trạng thái cục bộ không tương đương HEAD | Cần review và lưu phiên bản sau khi người dùng yêu cầu; audit không tự commit |
| P1 cho diễn giải dữ liệu | Snapshot khác bài báo gốc, replay dev/test chỉ 120 mẫu/3 speaker mỗi tập; train chỉ 15.885 | Chỉ kết luận trong snapshot/quy mô thật; phân tích theo attack và CI, không tuyên bố tái lập official VSASV hoặc unseen TTS engines. Nguồn: D015/D025, report D025 |
| P2 về đồng bộ tài liệu | README còn v25 đang chạy/D024 pilot; current_state có đoạn nói chưa có manifest hoặc packed chỉ PoC, xen đoạn đã hoàn tất; review 02/11 cũng có trạng thái cũ | Dễ chạy nhầm bước. Dùng D027 và artifact đã xác minh; cập nhật trạng thái vận hành có kiểm soát |
| P2 về đường chạy mặc định | `train_b0_full.py` mặc định chọn `b0_full_training.json` của D023, cấu hình cũ vẫn có readiness true | Có nguy cơ người vận hành chọn cấu hình lịch sử; hướng dẫn run mới phải chỉ rõ D027 và cổng checksum, không suy rằng mặc định đã là B0 v2 |
| P2 về benchmark | Các script benchmark LFCC vẫn gọi trực tiếp `lfcc(batch)` ở benchmark_batch_read/data_pipeline/dataloader_options/packed_artifact | Không dùng số đo cũ làm benchmark B0 v2 đã sửa; cần đo lại đúng đường `lfcc_features` |
| P2 về tái lập tải metadata | `download_vsasv_metadata.py` dùng `datasets`, không có trong requirements B0 | Pipeline từ metadata đang có hoạt động; quy trình tải metadata từ máy sạch chưa được chứng minh đầy đủ |

Các vấn đề của review 02/11 cần cập nhật: lỗi LFCC đã có fix/test và phép đo đủ dev; builder đã kiểm checksum chỉ mục/nhãn; manifest full đã tạo; packed đã có artifact thật; lỗi xóa download đã sửa trong working tree và có test. Những mục này không nên tiếp tục được báo là hoàn toàn chưa xử lý. Riêng parity, run khoa học và test gate vẫn chưa có bằng chứng đạt. Nguồn: commit `7e1eb27`, `d6958ad`, `c20c549`, diff hiện tại và mục 3–6.

## 9. Đề xuất thứ tự triển khai và tiêu chí chấp nhận

Đây là đề xuất triển khai, không phải quyết định nghiên cứu mới hoặc xác nhận đã được chạy.

1. **Hoàn tất cổng dữ liệu T08:** chạy parity dữ liệu thật, ghi số dòng/shard so được với Parquet và phần không còn nguồn cục bộ; kiểm hash/index/metadata đầy đủ. Chấp nhận khi có report đạt theo phạm vi đã khai báo, không chỉ test fixture. Cơ sở: D027 và script parity hiện có.
2. **Khóa và smoke B0 v2:** khóa checksum artifact trong cấu hình được duyệt; kiểm train/dev-full, seed 2026/2027/2028, thiết bị và exact-resume trên dữ liệu thật. Benchmark đúng LFCC theo mẫu để quyết định CPU/GPU và số seed theo D027. Chấp nhận khi report ghi sample/shard, identity và đối chứng dừng–resume đạt; không lấy PoC 24 mẫu làm runtime toàn tập.
3. **Huấn luyện B0 v2:** fresh initialization, peak, batch 8, LR 0,001, tối đa 3 epoch/patience 1 theo D027; chọn checkpoint/threshold trên dev-full. Chấp nhận khi có điều kiện dừng, history, best/latest, metric và checksum cho từng seed; chưa dùng test.
4. **Đưa XLS-R qua cổng pretrained:** tải/băm đúng revision theo quyền thực hiện riêng, kiểm offline load; nối packed → normalization D026 → backbone → head. Hoàn thiện seed huấn luyện riêng và runner/controller có test. Smoke CUDA lấy VRAM/thời gian thật, sau đó ghi D028 khóa bậc, số tầng, LR và ngân sách theo D026.
5. **Chạy X0/X1 và đăng ký mô hình:** giữ cùng train và dev-full, chọn cấu hình bằng dev; chốt X1 trước 29/11, đăng ký toàn bộ model/seed trước 06/12 theo D026. Chấp nhận khi registry chứa checkpoint/hash/threshold/dev metric/cấu hình và không còn lựa chọn dựa trên test.
6. **Hoàn thiện test gate trước đánh giá:** xác thực registry/manifest/checkpoint/threshold, lưu dấu lần đánh giá và report theo seed/attack/CI. Sau đó mới thực hiện test khoảng 13/12 và ablation theo giao thức đã khóa. Chấp nhận khi bảng B0/X0/X1 có cùng protocol và đủ provenance.
7. **Demo và báo cáo:** triển khai sliding-window theo D009 sau khi khóa mô hình; kiểm tệp ngắn/dài, cùng preprocessing và threshold dev. Viết dần chương dữ liệu/phương pháp ngay; phần kết quả chỉ dùng report hợp lệ, ghi rõ giới hạn LFCC/pilot/snapshot. Chấp nhận khi demo chạy được và báo cáo/slide/README cùng một bộ số liệu.

Trong lần báo cáo tiến độ gần nhất, có thể trình bày dữ liệu, giao thức, pipeline và phát hiện/sửa lỗi LFCC như kết quả đã làm; phải ghi B0 v2 và XLS-R pretrained đang là công việc tiếp theo. Chưa có bằng chứng để khẳng định XLS-R cải thiện B0 hoặc đồ án đã hoàn thành phần thực nghiệm chính.
