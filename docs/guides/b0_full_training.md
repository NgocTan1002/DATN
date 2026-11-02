# Đặc tả B0 huấn luyện đầy đủ trên development-20k-v2

## Trạng thái và phạm vi

D023 khóa lượt `b0-peak-development-20k-v2-full-v1` trong `configs/b0_full_training.json`. Runner đã triển khai tại `scripts/train_b0_full.py` cùng engine `src/training/full_training.py`; cổng CPU/CUDA đã đạt ngày 24/10, gồm đối chứng nhiều process trên private kernel v13 và kiểm tra lại checkpoint tải về. `implementation.ready_for_training=true`; phiên B0 đầu tiên private v14 đã hoàn thành và kiểm chứng paused512 ngày 25/10; chờ chuẩn bị resume phiên 2. Luồng pilot `scripts/train_b0.py`/checkpoint v3 giữ nguyên; full training dùng checkpoint v4 và entrypoint mới.

“Đầy đủ” ở đây là lượt huấn luyện sau chọn policy, có các epoch hoàn chỉnh trên 15.885 mẫu train của development manifest 20k; không có nghĩa dùng toàn bộ closed train của metadata. Manifest có 20.000 mẫu/66 shard: 15.885 train, 4.115 development; pool cục bộ có 68 shard. Không mở rộng dữ liệu, tạo packed artifact hoặc đọc test trong lượt này.

## Khởi tạo và cấu hình cố định

- Khởi tạo mới LFCC + LCNN theo `configs/lfcc_lcnn.json`, seed `2026`; không tiếp tục từ trọng số/optimizer hoặc threshold của pilot peak.
- Mono 16 kHz, 64.000 mẫu, policy `peak` theo D022; random crop theo seed/epoch khi train, center crop khi development. Giữ nguyên `configs/audio.json`.
- Batch 8, worker 0, sampler shuffle xác định theo seed + epoch, `drop_last=false`. Mỗi epoch có 1.986 optimizer step; batch cuối có 5 mẫu và phải được học.
- Adam, learning rate cố định 0,001, BCEWithLogitsLoss, float32; không thêm scheduler, AMP, class weighting, clipping hoặc thay kiến trúc ở lượt này.
- Tối đa 3 epoch = 5.958 optimizer step = 47.655 lượt mẫu train nếu không dừng sớm. Đây là trần vận hành có giới hạn, không phải cam kết mô hình hội tụ hoặc baseline tối ưu cuối.

## Lịch đánh giá và chọn mô hình

1. Chạy hết epoch, lưu checkpoint an toàn rồi đánh giá toàn bộ 4.115 development trước khi lấy batch của epoch kế tiếp. Không đánh giá ở step 0; không chọn checkpoint ở điểm chia phiên giữa epoch.
2. Lần đánh giá epoch đầu hữu hạn tạo best ban đầu. Chỉ coi là cải thiện nếu `current_eer < best_eer`, `min_delta=0`. EER bằng nhau giữ checkpoint sớm nhất; loss chỉ để báo cáo, không dùng tie-break chọn checkpoint. Tie-break policy ở D019 là một quy tắc khác.
3. Khi cải thiện, cập nhật best và đặt `bad_evaluations=0`; nếu không thì tăng bộ đếm. Dừng khi `bad_evaluations >= 1` hoặc đủ 3 epoch. Với một lần đánh giá mỗi epoch, lần dừng sớm nhất có thể xảy ra sau epoch 2. Đây là lựa chọn ngân sách thận trọng trong một lượt đơn, chưa được xác nhận tối ưu bằng thực nghiệm.
4. Sau mỗi lần đánh giá, lưu đủ score/label/file/speaker/utt_type, loss, EER, ROC-AUC, macro-F1 và threshold tại EER. Score hoặc metric không hữu hạn phải làm run thất bại; không cập nhật best/patience từ kết quả chưa hoàn chỉnh.
5. Report cuối phải hiển thị riêng `latest` và `selected_best`: epoch/global step/checksum/metric của từng checkpoint; không gắn metric best vào trọng số latest. Điểm dừng gồm `early_stopped`, `max_epochs` hoặc `paused`; `paused` không phải kết quả huấn luyện đã hoàn thành.

## Threshold và đánh giá cuối

Threshold dùng cho inference/đánh giá cuối được lấy từ full development của **chính checkpoint best được chọn trong lượt này**, rồi đóng băng cùng checksum checkpoint, manifest và cấu hình. Không dùng threshold pilot `0,2584670518678943` cho mô hình mới chỉ vì cùng policy.

Nếu checkpoint best đã có report đầy đủ, khớp checksum, không cần đánh giá development thêm chỉ để xuất report cuối. EER/ROC-AUC luôn tính trực tiếp từ score; macro-F1 dùng threshold development đã đóng băng khi đánh giá ngoài development. Evaluation manifest content-disjoint và audit test vẫn cần quyết định riêng trước đánh giá test; đặc tả này không mở cổng test.

## Chia phiên và khôi phục

- Một invocation học tối đa 512 optimizer step; checkpoint định kỳ sau mỗi 64 **bước đã hoàn thành optimizer**, đồng thời lưu ở biên epoch và khi chủ động dừng. Quota thiếu thì chờ và resume nguyên identity, không giảm ngân sách khoa học hoặc reset patience để vừa phiên.
- Nếu chạm biên epoch ở bước cuối invocation, ưu tiên hoàn tất full development và commit kết quả trước khi trả về. Nếu phiên bị ngắt trước/trong development, khôi phục ở trạng thái `needs_evaluation`, đánh giá lại từ đầu; không tăng patience hai lần và không bước sang epoch mới trước khi đánh giá hoàn chỉnh.
- Đầu mỗi epoch, `train_base.set_epoch(sampler.epoch)` phải nhất quán. Nếu resume giữa epoch thì giữ nguyên cursor; iterator DataLoader dùng generator riêng, không được làm lệch RNG dropout. Không lấy batch tiếp theo trước khi ghi nhận giới hạn phiên hoặc quyết định early stopping.
- Checkpoint full training dự kiến dùng phiên bản 4, giữ version 3 cho pilot. Identity phải khóa manifest/selection/model/audio/policy/seed/optimizer/lr/batch và mọi trường ngữ nghĩa của training plan: trần epoch/step, lịch đánh giá, patience, min_delta, quy tắc hòa và precision. Đường dẫn output và mốc thời gian không thuộc identity khoa học; source bundle thực dùng phải được ghi và đối chiếu trước resume. Không nạp checkpoint pilot vào full training.
- Một generation bất biến gồm checkpoint model/optimizer/progress/sampler/Python/NumPy/PyTorch/CUDA RNG, controller state, lịch sử đánh giá, best reference và report liên quan. Controller gồm `completed_epochs`, `last_evaluated_step`, `needs_evaluation`, `best_epoch`, `best_step`, `best_eer`, `bad_evaluations`, `stop_reason`.
- Ghi các file generation vào vị trí mới, kiểm tra checksum rồi thay `LATEST.json` nguyên tử. Chỉ pointer đã commit mới có thẩm quyền; file đang ghi hoặc generation mồ côi không làm tăng epoch/patience. Best/report phải được giữ cùng tham chiếu/checksum trước khi commit pointer. Không ghi đè best khi chỉ lưu latest hoặc dọn best còn được tham chiếu.

## Ngân sách vận hành và giới hạn dự toán

`reports/b0_full_training_budget.{json,md}` chứa phép tính tái lập từ manifest và số đo benchmark/ba pilot, kèm SHA-256 nguồn. Sử dụng giá trị lớn nhất của training seconds/step và development seconds/sample, không ngoại suy PoC packed 24 mẫu. Tính đủ 1.986 bước/epoch và 4.115 development/epoch; cộng 20% dự phòng vận hành do khởi tạo, checkpoint, gián đoạn và các overhead chưa đo riêng.

Một session có thể gồm tối đa 512 bước và một lượt development ở biên epoch. Trần invocation giới hạn số bước, không bảo đảm thời gian cứng. Trước gửi phải kiểm tra quota và giới hạn phiên thực tế, giữ biên dự phòng cho checkpoint/development; nếu không đủ thì chưa gửi. Dự toán nhiều epoch có thể phải đi qua nhiều kỳ quota. Các phép chiếu không phải thời gian huấn luyện thực tế, metric hoặc bằng chứng hội tụ; không tự kích hoạt tài nguyên trả phí.

## Cổng triển khai trước run dài

- Runner đọc và kiểm tra plan, checksum/số mẫu/split/coverage, từ chối test, subset limit, policy/config/seed sai và resume khác identity. Tạo source snapshot mới sau kiểm thử, giữ snapshot pilot cũ để truy vết; không dùng lại source cũ mà tuyên bố đã hỗ trợ full training.
- Unit/integration test bao phủ tail batch, một lần đánh giá mỗi epoch, hòa EER, patience/reset, dừng sớm, đạt trần và điểm chia phiên đúng tại biên epoch.
- Đối chứng liên tục so với resume ở giữa epoch, sau optimizer trước evaluation, trong evaluation, sau tạo best trước commit và sau commit; phải khớp loss/parameter cuối, chuỗi batch, history, best và quyết định dừng. Test không chỉ kiểm tra state_dict load được.
- Lỗi ghi checkpoint/report hoặc thiếu best/checksum sai phải dừng rõ bằng tiếng Việt; không silently chọn latest. Sau restore phải phát hiện controller/progress/sampler mâu thuẫn.
- Smoke CPU fixture cho vòng đời đầy đủ, rồi smoke CUDA hai process và đánh giá nhỏ để xác nhận RNG GPU. Technical smoke tách khỏi cấu hình khoa học; không dùng metric fixture/subset làm kết quả hoặc thay ngân sách từ smoke.
- Chỉ khi cổng CPU/CUDA và report đạt mới ghi `ready_for_training=true`, chuẩn bị private Kaggle profile mới và gửi phiên đầu. Người dùng tự theo dõi run dài; không bật lịch tự động. Khôi phục output/checkpoint đúng generation trước khi gửi phiên tiếp theo.

## Bằng chứng và báo cáo cần lưu

Mỗi report phải ghi training-plan checksum/source checksum, seed/config/manifest/selection, số shard và mẫu thực dùng, bước/epoch/phiên, best/latest, lịch sử development và lý do dừng, tài nguyên/stack GPU/timing, trạng thái resume/RNG và threshold provenance. Chỉ report nhỏ/cấu hình/source được đưa vào Git; checkpoint/audio/log thô giữ ngoài Git. Tài liệu này không phải báo cáo đã thực hiện huấn luyện.

## Kết quả triển khai ngày 24/10

- 10 kiểm thử D023 bao phủ dừng sớm/hòa EER, tail batch, fault trước/trong evaluation và trước/sau commit, resume/identity/checksum/controller. Toàn suite cuối đạt 172/172 trong 160,643 giây.
- CPU đối chứng trên 4 train/2 development từ 6 shard: continuous so với dừng step 1 rồi process mới resume cùng kết thúc step 4/epoch 2 do early stopping; model/optimizer/sampler/RNG/loss/batch trace/history/best khớp tuyệt đối. Report tại `reports/b0_full_cpu_smoke.json` và `reports/b0_full_cpu_{continuous,partial,resumed}.json`. Đây là technical smoke, không phải kết quả B0 đầy đủ.
- CPU real-audio smoke phát hiện tuple metric khác biểu diễn list trong JSON; đã chuẩn hóa controller trước serialize và chạy lại đối chứng/test.
- Source pilot cũ được giữ tại `.kaggle/pilot-source-locked/`. Gói source mới 56 file/checksum và quyền upload cụ thể được ghi tại `reports/b0_full_source_preflight.*`; source/runtime trên Kaggle riêng tư. Không có audio/checkpoint/secret file trong bundle.
- Profile cục bộ: `b0-full-smoke` chạy continuous, partial, resume qua các process và comparator; `b0-full-first-session` đã được gửi thành v14 sau cổng CUDA đạt.

## Bàn giao phiên dài đầu tiên — version 14

Private kernel: https://www.kaggle.com/code/ngctnnguyn1111/doanttnt-kaggle-cli-runner . Phiên đầu đã hoàn thành là v14, profile `b0-full-first-session`, source SHA-256 `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`. Không gửi profile khác khi phiên còn chạy; CLI status phản ánh phiên latest.

Kiểm tra thủ công mỗi 20–30 phút từ repository:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py status
```

Khi COMPLETE, tải vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-1-v14
```

Không cần force nếu thư mục mới. Sau tải, kiểm tra workflow `complete`/mọi return code 0 và source checksum; report `reports/b0_peak_full_session_1.json` phải có mode `full_training`, 15.885 train/4.115 development, 66 shard và `controller.global_step=512` nếu phiên hoàn thành đúng giới hạn. Trạng thái runner dự kiến `paused`, history development rỗng và best chưa có vì chưa hết epoch đầu; đây là chia phiên dự kiến, không phải kết quả B0 cuối. Không gắn metric pilot hoặc smoke vào phiên này.

Giữ **toàn bộ** `checkpoints/b0_peak_full_v1/`, gồm LATEST.json và các generation, để kiểm chứng và chuyển sang phiên resume tiếp theo; không chỉ lấy một file model. Cần xác minh pointer/checksum/identity và chuẩn bị input checkpoint private cho phiên sau trước khi gửi. Không chạy lại profile first-session để thay resume: thư mục mới sẽ khởi tạo lại từ đầu. Khi ERROR/CANCELLED, giữ nguyên log/output và bàn giao để kiểm tra generation đã commit; chưa tuyên bố tiến độ/bước nếu chưa có output hợp lệ. Người dùng tự theo dõi, không bật automation.

## Chốt output phiên 1 ngày 25/10

Version 14 đã COMPLETE, workflow complete và cả hai lệnh return code 0. Runner paused đúng bước 512, sampler epoch0/cursor512, 4.096 lượt mẫu train; chưa có full development, best hoặc threshold. Report training ghi 1,956 giờ trên Tesla T4. Cả 10 generation, pointer/checksum, model/optimizer/RNG, identity/source và chuỗi batch đã kiểm chứng; `reports/b0_full_session_1_verification.{json,md}` chứa bằng chứng. Không cần theo dõi v14 nữa.

Bước tiếp theo là tạo input checkpoint private và profile resume giữ nguyên identity/source, hướng tới bước 1.024 ở phiên 2. Checkpoint latest SHA-256 `71464c4f3ebabfc6f3634c7963ced14d8056dc5ad056e6284ee1883fe46c6872`. Giữ toàn bộ thư mục generation cục bộ; chưa upload checkpoint hoặc gửi phiên 2 trong phiên kiểm chứng output. Đối chiếu artifact không thay thế bằng chứng GPU resume thực tế của phiên tiếp theo.

## Bàn giao phiên tiếp tục thứ hai — version 15, ngày 25/10

Version15 đã COMPLETE và kiểm chứng GPU resume512→1.024 ngày25/10, profile `b0-full-resume-session-2`; paused đúng giới hạn thêm512 bước. Báo cáo `reports/b0_full_session_2_verification.*`. Hướng dẫn theo dõi bên dưới được dùng lúc phiên còn chạy; hiện không cần theo dõi v15 nữa.

`deploy/kaggle/resume_checkpoint.py` thực hiện chuyển checkpoint; file này nằm ngoài allowlist source huấn luyện đã khóa và được nhúng riêng vào lệnh bootstrap. Source 56 file, bundle SHA-256 `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1` và identity D023 giữ nguyên. Bootstrap kiểm tra manifest được ghim checksum, từng file, pointer/controller/identity và chỉ công bố thư mục run sau khi kiểm tra đạt. Lệnh training dùng `--resume`.

Gói đầu vào có đủ 10 generation, 21 file checkpoint/pointer/trạng thái và manifest; dataset `ngctnnguyn1111/doantnnt-b0-full-resume` đã xác nhận riêng tư, ready, đủ 22 file tải lên với kích thước khớp. Metadata tạo dataset được giữ cục bộ. Người dùng đã xác nhận cụ thể upload checkpoint và chạy tiếp sau yêu cầu của bộ duyệt tự động. Manifest SHA-256 `316d9c7c67298f70c2d0b5f9b11cecb4e9e15f711de5d28a904299e8695b23de` được tải lại từ Kaggle và đối chiếu thành công.

Cổng cục bộ: 7 kiểm thử mới đạt, bao gồm ngắt–chuyển–resume khớp chạy liên tục, bảo toàn best, từ chối checksum/source/step sai, đường dẫn không an toàn và ghi đè. Gói checkpoint thật và bootstrap CLI nhúng cũng đã khôi phục đúng bước 512; payload trước/sau chuyển khớp tuyệt đối. Toàn suite 179/179 đạt trong 239,697 giây. Bằng chứng: `reports/b0_full_session_2_preflight.{json,md}`, `reports/b0_full_session_2_transfer_check.json` và `reports/b0_full_session_2_embedded_check.json`.

### Theo dõi thủ công

Mở private kernel https://www.kaggle.com/code/ngctnnguyn1111/doanttnt-kaggle-cli-runner và chọn version 15, hoặc từ thư mục repository kiểm tra mỗi 20–30 phút:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py status
```

- `QUEUED`/`RUNNING`: tiếp tục chờ. CLI status phản ánh version mới nhất; không gửi phiên khác trong lúc v15 đang chạy.
- `COMPLETE`: tải output vào thư mục riêng bằng lệnh dưới đây.
- `ERROR`/`CANCELLED`: tải và giữ output/log nếu có, gửi lại thông báo lỗi để kiểm tra checkpoint đã commit trước khi quyết định chạy lại.
- Cảnh báo nâng phiên bản Kaggle CLI không tự chứng minh phiên huấn luyện bị lỗi.

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-2-v15
```

Sau khi tải, đối chiếu:

1. `reports/kaggle_cli_workflow.json`: đúng profile, source checksum, trạng thái `complete` và cả 3 lệnh trả mã 0.
2. `reports/b0_full_session_2_resume_input.json`: `passed=true`, `restored_global_step=512`, `files_verified=21`, manifest checksum đúng. Báo cáo này xác nhận chuyển checkpoint; riêng nó chưa chứng minh optimizer đã chạy tiếp.
3. `reports/b0_peak_full_session_2.json`: `run_mode=full_training`, `steps_this_invocation=512`, `controller.global_step=1024`, trạng thái `paused` nếu hoàn thành đúng giới hạn. Phạm vi phải là 15.885 train/4.115 development trên 66 shard của manifest 20.000 mẫu.
4. `checkpoints/b0_peak_full_v1/LATEST.json` và các generation: kiểm tra checksum, controller, sampler, optimizer và RNG, bảo toàn trace từ v14. Giữ toàn bộ thư mục này để tiếp tục phiên sau.

Tại mục tiêu bước 1.024 chưa đủ epoch đầu 1.986 bước, nên dự kiến chưa có development evaluation, best hoặc threshold. Đây là trạng thái dự kiến để kiểm tra, chưa phải kết quả đã quan sát của v15.

Profile trong `.kaggle/project.json` lưu các lệnh, còn bộ kernel đã gửi nằm tại `.kaggle/resume/session-2-kernel/`. Metadata trong bộ này gắn thêm private checkpoint dataset. Lệnh `push-kernel` chung sẽ tạo lại metadata chỉ có source/data và làm mất mount checkpoint bổ sung; khi cần gửi lại phải kiểm tra bộ kernel đã chuẩn bị và metadata trước, không chạy profile first-session.

## Chốt output phiên 2 ngày25/10

V15 COMPLETE/cả3 lệnh mã0. Bootstrap đúng checkpoint512, training CUDA tiếp tục thêm512 bước tới1.024; paused/cursor1.024/epoch0. Runtime training1,746giờ, tổng8.192 lượt mẫu train, 15.885train/4.115dev trên manifest20.000/66shard (pool68). Chưa đủ epoch1.986 bước, chưa evaluation/best/threshold.

Đã đọc lại19 generation; 10 generation nhập v14 giữ nguyên từng byte. Prefix loss/batch bảo toàn, toàn bộ1.024 batch khớp sampler seed2026, 12 Adam state đều bước1.024, model/optimizer hữu hạn, đầy đủ2 CUDA RNG. Source/plan/manifest/identity không đổi; kiểm chứng chi tiết tại `reports/b0_full_session_2_verification.*`. Không có đối chứng GPU liên tục1.024 bước hoặc run mới trong phiên kiểm chứng này.

Giữ toàn bộ `kaggle_outputs/b0-full-session-2-v15/checkpoints/b0_peak_full_v1/`, latest generation `g-9de1bc836e934c1ab892bdf6683afaec`, checkpoint SHA-256 `40812f8331d4b4d615ce91c6af4b72c863a4e17cb946041c42f8e94686c620ca`. Bước tiếp theo là chuẩn bị input checkpoint từ v15 cho phiên1.024→1.536, ghim manifest/checksum mới và cùng source/identity. Dataset checkpoint hiện vẫn chứa input512 của v15; không dùng lại cấu hình session2 để thay phiên tiếp theo. Chưa upload checkpoint mới hoặc gửi phiên3.

## Checklist trước lần đánh giá development đầu tiên

Đây là checklist chuẩn bị ngày 26/10 theo D023, chưa phải kết quả đánh giá thực tế. Sau khi phiên 3 đạt bước 1.536 và được kiểm chứng, phiên tiếp theo còn 450 bước để chạm cuối epoch 1 ở bước 1.986. Nếu invocation đó hoàn tất tối đa 512 bước và không gặp lỗi, mốc dự kiến là bước 2.048, sau khi đã đánh giá đủ development ở bước 1.986.

- Kiểm tra trước chạy: input checkpoint đã kiểm chứng ở bước 1.536, source/identity D023 và manifest không đổi, đủ quota cho cả training lẫn 4.115 mẫu development. Không chỉ dự trù thời gian 512 bước train.
- Trước batch epoch 2: `needs_evaluation` phải được xử lý, full development hoàn tất, history và best được commit. Rà code ngày 26/10 xác nhận nhánh evaluation đứng trước nhánh lấy batch/đổi epoch trong `src/training/full_training.py`; đây là xác minh luồng code đã khóa, không thay thế output GPU thực tế.
- Evaluation epoch 1: `sample_count=4115`, đủ score/label/records và `utt_type_counts` khớp records; thứ tự file/speaker/nhãn khớp toàn bộ closed development của manifest; không có giới hạn subset hoặc test. Đối chiếu lại EER, threshold, AUC và macro-F1 từ score bằng metric hiện hành.
- Controller sau evaluation đầu: `completed_epochs=1`, `last_evaluated_step=1986`, `needs_evaluation=false`, history có đúng một entry epoch 1/bước 1.986, `bad_evaluations=0`, `stop_reason=null`.
- Best đầu tiên phải ở epoch 1/bước 1.986; pointer best phải trỏ generation chứa chính model/controller/metric đã đánh giá. Kiểm tra hash của cả best và latest, kể cả khi chúng khác generation.
- Threshold phải là `decision_threshold` từ development của best hiện tại và khớp `eer_threshold`. Best có thể còn thay đổi ở epoch sau; chưa khóa đây là threshold cuối và không dùng threshold pilot thay thế.
- Nếu phiên tiếp tục đến bước 2.048: dự kiến sampler epoch 1/cursor 62, `completed_epochs=1`, `last_evaluated_step=1986`; history vẫn chỉ một evaluation. Đây là giá trị dự kiến theo ngân sách và lịch D023 để kiểm tra khi có output.
- Nếu ngắt trong evaluation: giữ generation/pointer đã commit, khôi phục pending evaluation và chạy lại toàn bộ development trước epoch kế tiếp; không tăng patience hai lần.
- Nếu EER thấp bất thường: kiểm tra speaker/file/content leakage, sample rate, biên độ và độ dài trước khi diễn giải kết quả. Báo cáo phải ghi 15.885 train/4.115 development trên manifest 20.000 mẫu/66 shard, pool 68 shard, cùng giới hạn replay development của dữ liệu hiện có.

Hàm `_evaluate` trong `scripts/train_b0.py` đọc toàn bộ development không shuffle, ghi file/speaker/nhãn/score và gọi metric development hiện hành. Runner D023 chọn best bằng EER giảm nghiêm ngặt; hòa giữ best sớm nhất. Cổng CPU/CUDA và 179/179 test đã có từ phiên trước; chỉ ghi đánh giá epoch 1 đạt sau khi đối chiếu output thực tế.

## Bàn giao phiên tiếp tục thứ ba — version 16, ngày 26/10

Version16 đã hoàn thành và output được kiểm chứng ngày 26/10: GPU resume 1.024 → 1.536, paused đúng giới hạn thêm 512 bước, profile `b0-full-resume-session-3`. Báo cáo `reports/b0_full_session_3_verification.*`. Hướng dẫn theo dõi bên dưới được dùng lúc phiên còn chạy; hiện không cần theo dõi v16 nữa.

Đầu vào có đủ19 generation/39 file checkpoint, pointer và state; manifest SHA-256 `76b604d3ff793f3ccec4f47e21ae916510172ccea14c76f74002ee16ce67ae97`. Dataset checkpoint riêng tư đã cập nhật, ready, có40 file dữ liệu upload và inventory khớp. Source bundle giữ `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`, source tree/identity D023 không đổi. Restore gói thật và bootstrap CLI nhúng khớp trạng thái gốc; báo cáo `reports/b0_full_session_3_preflight.*` cùng `b0_full_session_3_transfer_check.json` và `b0_full_session_3_embedded_check.json`.

Theo dõi mỗi20–30 phút bằng `.venv/Scripts/python.exe scripts/kaggle_workflow.py status`, hoặc mở private kernel https://www.kaggle.com/code/ngctnnguyn1111/doanttnt-kaggle-cli-runner và chọn version16. Chỉ một kernel active; không gửi phiên khác khi v16 còn QUEUED/RUNNING.

Khi COMPLETE, tải vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-3-v16
```

Đối chiếu output:

1. `reports/kaggle_cli_workflow.json`: profile đúng, source checksum đúng, complete và cả3 lệnh mã0.
2. `reports/b0_full_session_3_resume_input.json`: `passed=true`, `restored_global_step=1024`, `files_verified=39`, manifest checksum đúng. Report bootstrap riêng chỉ xác nhận chuyển checkpoint; phải đọc cả report training.
3. `reports/b0_peak_full_session_3.json`: dự kiến full_training/paused, thêm512 bước tới1.536, epoch0/cursor1.536, trace tổng12.288 lượt mẫu; nguồn15.885train/4.115dev/66shard của manifest20.000 (pool68).
4. Giữ toàn bộ `checkpoints/b0_peak_full_v1/`. Kiểm tra pointer/hash,19 generation nhập giữ nguyên, prefix1.024 loss/batch, sampler/optimizer/RNG/controller trước khi chuẩn bị phiên tiếp theo. Dự kiến chưa evaluation/best/threshold vì chưa đến1.986 bước.

Nếu ERROR/CANCELLED, tải và giữ log/output nếu có; xác minh generation cuối đã commit trước khi gửi lại. Cảnh báo phiên bản CLI cũ không tự chứng minh training lỗi. Bộ kernel đã gửi nằm tại `.kaggle/resume/session-3-kernel/`; lệnh push-kernel chung tạo lại metadata thiếu checkpoint mount, nên không dùng để gửi lại bộ này mà chưa kiểm tra metadata.

Sau khi v16 được kiểm chứng đạt1.536, mới chuẩn bị phiên đi qua biên epoch1 theo checklist phía trên. Không gán metric pilot hoặc smoke cho mô hình hiện tại.

## Chốt output phiên 3 ngày 26/10

Workflow complete/cả 3 mã 0; log, report và checkpoint xác nhận GPU resume 1.024 → 1.536, thêm 512 bước, epoch 0/cursor 1.536. Runtime training report 1,762 giờ trên Tesla T4. Tổng trace 12.288 lượt mẫu; chưa hết epoch đầu hoặc có evaluation/best/threshold.

Đã đọc lại 28 generation và kiểm tra pointer/hash/controller, source/identity, sampler/optimizer/RNG; 19 generation nhập v15 giữ nguyên từng byte, phần 1.024 loss/batch bảo toàn. Bằng chứng `reports/b0_full_session_3_verification.*`; output do người dùng tải được kiểm chứng offline, không chạy thêm training/GPU hoặc unit suite.

Giữ toàn bộ `kaggle_outputs/b0-full-session-3-v16/checkpoints/b0_peak_full_v1/`; latest generation `g-794ca799c6fb4cf6b988a8a1c920dcc4`, checkpoint SHA-256 `3cc00d46d9a4676950eb6bcd3c9a4ae6006ebadcdcc52418f3e44d4eda6d8772`. Phiên tiếp theo cần gói input bước 1.536 và manifest/checksum mới; dataset hiện vẫn là input 1.024 đã dùng cho v16, không gửi lại profile session3 để thay resume từ 1.536.

Còn 450 bước đến epoch 1 ở mốc 1.986; phiên 4 phải hoàn tất full development và commit best trước batch epoch 2 theo checklist phía trên. Mốc kết thúc dự kiến nếu chạy đủ 512 bước là 2.048; chưa phải kết quả. Chưa upload checkpoint mới hoặc gửi phiên 4.

## Chuẩn bị phiên tiếp tục thứ tư — ngày 27/10

Đã đóng gói toàn bộ output v16 ở bước 1.536: 28 generation/57 file checkpoint, pointer và state, tổng 169.861.762 byte. Manifest chuyển phiên SHA-256 `b11075f5f1f0524b09f114667d5e7420767288727c0b79bf6586f47eeaf29551`; gói dữ liệu dự kiến upload gồm 58 file/169.876.829 byte, không có audio, log hoặc thông tin đăng nhập. Restore trực tiếp và bootstrap CLI nhúng đều khớp run root nguồn; source 56 file/tree/bundle và identity D023 không đổi. Bằng chứng tại `reports/b0_full_session_4_preflight.*`, `b0_full_session_4_transfer_check.json` và `b0_full_session_4_embedded_check.json`.

Profile `b0-full-resume-session-4` và bộ kernel khóa `.kaggle/resume/session-4-kernel/` đã sẵn sàng: private, internet tắt, gắn data/source/checkpoint dataset, expected input 1.536, `--resume` và tối đa 512 bước. Kiểm tra trước upload lúc 14:43 UTC+7 ghi latest kernel `COMPLETE`, ba dataset `ready` và quota còn 24,46/30,00 giờ GPU; refresh theo CLI `2026-10-31T00:00:00`.

Upload payload mới ban đầu bị bộ duyệt an toàn chặn vì xác nhận trước chỉ bao phủ checkpoint của các phiên cũ. Sau khi người dùng xác nhận cụ thể đúng payload/manifest/private dataset, upload đã hoàn tất; tải lại xác nhận private/ready, đủ58file/169.876.829byte và manifest checkpoint/source khớp. Bộ kernel khóa được push trực tiếp; private kernel version17 quan sát `RUNNING` lúc14:51 UTC+7. Không dùng `push-kernel --profile` vì lệnh đó tạo lại metadata và bỏ mount checkpoint.

Khi phiên 4 được gửi, theo dõi mỗi 20–30 phút:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py status
```

Khi `COMPLETE`, tải vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-4-v17
```

Nếu `QUEUED`/`RUNNING`, tiếp tục chờ và không gửi kernel khác. Nếu `ERROR`/`CANCELLED`, tải/giữ log và output, xác định generation cuối đã commit trước khi quyết định gửi lại. Output thành công phải được đối chiếu theo checklist development epoch 1 ở trên; không xem các mốc 1.986/2.048 hay best đầu tiên là kết quả cho đến khi report/checkpoint thực tế được kiểm chứng.

## Chốt output phiên 4 ngày 27/10

Version 17 đã `COMPLETE`; workflow và cả ba lệnh con đều thành công. Bootstrap khôi phục đúng input bước 1.536/57 file. Training CUDA thêm 512 optimizer step, dừng `paused` ở bước 2.048; sampler đã sang epoch 2 với cursor 62. Do batch cuối epoch 1 có 5 mẫu, tổng exposure thực tế là 16.381 chứ không phải phép nhân 2.048 × 8.

Evaluation cuối epoch 1 chạy đủ 4.115 development tại bước 1.986: loss `0,6779944593369743`, EER `0,08595800524934383`, ROC-AUC `0,9702422376670303`, macro-F1 `0,9088565064507652`, threshold EER `0,00960088725961999`. Best đầu tiên là `g-c7418c4439ac45e9b051fa77b768709d`; latest bước 2.048 là `g-f0940a29982742ebb7cf7aa4a252bb26`. Threshold này chỉ thuộc checkpoint best hiện hành và có thể thay đổi nếu epoch sau cải thiện.

Kiểm chứng offline đọc đủ 39 generation; 28 generation nhập từ v16 giữ nguyên từng byte. Metric tính lại từ score/label khớp dưới `1e-12`; pointer, hash, controller, sampler, optimizer, RNG, source và identity đều đạt. Cổng kiểm tra EER thấp bất thường cũng đạt: speaker/file overlap 0, 20.000 content hash duy nhất, duplicate chéo split 0, metadata sample-rate gốc 16/40 kHz khớp, preprocessing vẫn về 16 kHz/4 giây với policy `peak`. Test set không được dùng.

Lần tải đầu vào `kaggle_outputs/b0-full-session-4-v17/` có hai tệp bị thay bằng trang HTML 502, đều 332 byte. Bản tải lại `kaggle_outputs/b0-full-session-4-v17-redownload/` có đủ 85 tệp sạch và là nguồn chuẩn cho báo cáo/phiên sau; giữ bản đầu để truy vết, không dùng làm checkpoint. Bằng chứng nằm tại `reports/b0_full_session_4_verification.*` và các report phiên 4 trong `reports/`.

Không cần theo dõi version 17 nữa. Phiên kế tiếp phải đóng gói từ latest bước 2.048, kiểm tra lại quota/dataset/source/manifest và chỉ sau đó tiếp tục tối đa 512 bước đến mốc dự kiến 2.560. Việc chuẩn bị và gửi phiên này được ghi ở mục tiếp theo.

## Bàn giao phiên tiếp tục thứ năm — version 18, ngày 28/10

Đầu vào version 18 là output sạch v17 tại bước 2.048: 39 generation/79 tệp checkpoint, pointer và state, tổng 245.056.669 byte. Gói upload có 80 tệp/245.077.609 byte; manifest chuyển phiên SHA-256 `31bac0c9f339264b9484e653a648908b73be72c166bd675cb0632ae7ff7a48c1`. Restore trực tiếp và bootstrap CLI nhúng đều khớp model/optimizer/sampler/RNG/controller/history/best; source 56 tệp, identity D023 và best bước 1.986 không đổi.

Sau xác nhận cụ thể của người dùng, gói được upload lên private checkpoint dataset. Tải lại từ Kaggle xác nhận dataset private/`ready`, đúng 80 tệp/245.077.609 byte và manifest checkpoint/source khớp. Bộ kernel khóa `.kaggle/resume/session-5-kernel/` là private, internet tắt, gắn đúng data/source/checkpoint dataset; được push trực tiếp để không làm mất mount checkpoint. Private kernel version 18 được quan sát `RUNNING` lúc 13:54 UTC+7; quota ngay sau gửi còn 20,87/30,00 giờ GPU.

Theo dõi mỗi 20–30 phút, không gửi kernel khác khi version 18 còn `QUEUED`/`RUNNING`:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_workflow.py status
```

Khi trạng thái `COMPLETE`, tải output vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-5-v18
```

Sau khi tải, chưa dùng ngay làm input phiên kế tiếp. Cần kiểm tra workflow/cả lệnh con, bootstrap input bước 2.048/79 tệp, pointer/hash và toàn bộ 39 generation nhập giữ nguyên, sampler/optimizer/RNG/controller/source/identity, loss/batch trace và môi trường GPU. Nếu phiên chạy đủ 512 bước, kết quả dự kiến là `paused` ở bước 2.560, epoch 2/cursor 574 và tổng 20.477 lượt mẫu train; history vẫn một evaluation, best/threshold vẫn ở bước 1.986 vì chưa tới evaluation epoch 2 tại bước 3.972. Đây là giá trị kỳ vọng để đối chiếu, chưa phải kết quả thực tế.

Nếu `ERROR`/`CANCELLED`, tải và giữ log/output nếu có, xác định generation cuối đã commit trước khi quyết định resume; không tự gửi lại. Cảnh báo Kaggle CLI 2.2.4 cũ hơn 2.2.5 không tự chứng minh training lỗi. Bằng chứng preflight nằm tại `reports/b0_full_session_5_preflight.*`, `reports/b0_full_session_5_transfer_check.json` và `reports/b0_full_session_5_embedded_check.json`.

## Chốt output phiên 5 ngày 29/10

Version 18 đã `COMPLETE`; output do người dùng tải về được kiểm chứng offline và đạt toàn bộ cổng. Workflow complete/cả ba lệnh mã 0, bootstrap khôi phục đúng input bước 2.048/79 tệp. Training CUDA thêm 512 optimizer step và dừng `paused` ở bước 2.560; sampler epoch 2/cursor 574, tổng trace 2.560 batch/20.477 lượt mẫu. Runtime training theo report là 1,802 giờ trên Tesla T4.

Đã đọc đủ 48 generation; 39 generation nhập từ v17 giữ nguyên từng byte. Prefix 2.048 loss/batch, pointer/hash/controller, model và 12 Adam state, sampler, hai CUDA RNG state, source/plan/manifest/identity đều đạt. Latest là generation `g-2d1394dcd5d34ed5bd7fe2bdc26af49f`, checkpoint SHA-256 `25220c5a8058716b69dd5ca35ec340fbfa4d87f75c60f9e5dbc62495b4549f66`.

Phiên này chưa đến cuối epoch 2 ở bước 3.972 nên không chạy development mới. History vẫn có đúng một evaluation epoch 1/bước 1.986; best vẫn generation `g-c7418c4439ac45e9b051fa77b768709d`. Metric epoch 1 được tính lại và giữ nguyên: loss `0,6779944593369743`, EER `0,08595800524934383`, ROC-AUC `0,9702422376670303`, macro-F1 `0,9088565064507652`, threshold `0,00960088725961999`. Cổng leakage/sample-rate/amplitude/duration vẫn đạt; test không được dùng.

Output chuẩn `kaggle_outputs/b0-full-session-5-v18/` có 103/103 tệp sạch, không phát hiện HTML/502 và không cần tải lại. Bằng chứng tại `reports/b0_full_session_5_verification.*` và các report phiên 5 trong `reports/`. Không cần theo dõi version 18 nữa.

Phiên 6, nếu được thực hiện, phải đóng gói từ latest bước 2.560, sinh manifest/checksum mới, restore/bootstrap thử, kiểm tra quota/dataset/kernel và xin xác nhận cụ thể trước upload. Với giới hạn 512 bước, mốc tiếp theo dự kiến là bước 3.072/epoch 2/cursor 1.086; đây chưa phải kết quả. Full development epoch 2 và quyết định early stopping chỉ ở bước 3.972.

## Chuẩn bị phiên 6 ngày 29/10

Preflight đã hoàn tất từ output chuẩn v18. Gói exact-resume bảo toàn đủ 48 generation và có 97 tệp checkpoint/pointer/state. Restore trực tiếp cùng bootstrap CLI nhúng đều khớp model, optimizer, sampler epoch 2/cursor 574, RNG, controller, history và best. Source 56 tệp, identity D023 và lịch evaluation giữ nguyên.

Phần sẽ upload lên private dataset `ngctnnguyn1111/doantnnt-b0-full-resume` có 98 tệp/317.861.018 byte; manifest SHA-256 `906bd9622f38d09c95db19369db03694e51e3d712c9104f6945d7ce988604612`. Gói không có audio, log hoặc thông tin đăng nhập. Bộ kernel khóa nằm tại `.kaggle/resume/session-6-kernel/`, profile `b0-full-resume-session-6`, private, internet tắt và gắn đúng ba dataset.

Remote preflight lúc 15:45 UTC+7 ghi nhận latest kernel `COMPLETE`, cả ba dataset `ready` và quota GPU còn 19,06/30,00 giờ. Sau xác nhận cụ thể của người dùng, payload được upload lên private checkpoint dataset; tải lại xác nhận private/`ready`, đúng 98 tệp/317.861.018 byte cùng manifest checkpoint/source khớp. Private kernel version 19 được gửi đúng một lần từ bộ khóa session6 và quan sát `RUNNING` lúc 15:58 UTC+7; quota sau gửi còn 19,05/30,00 giờ. Nếu phiên chạy lâu, theo dõi mỗi 20–30 phút:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_workflow.py status
```

Chỉ khi trạng thái `COMPLETE` mới tải output vào thư mục mới theo version và kiểm chứng. Nếu `ERROR`/`CANCELLED`, giữ output/log, xác định generation cuối đã commit và dừng chẩn đoán; không tự gửi lại. Mốc bước3.072/epoch2/cursor1.086 và24.573 lượt mẫu là kỳ vọng, chưa phải kết quả thực tế.

## Chốt output phiên 6 ngày 30/10

Version 19 đã `COMPLETE`; output do người dùng tải về được kiểm chứng offline và đạt toàn bộ cổng. Workflow complete/cả ba lệnh mã 0, bootstrap khôi phục đúng input bước 2.560/97 tệp. Training CUDA thêm 512 optimizer step và dừng `paused` ở bước 3.072; sampler epoch 2/cursor 1.086, tổng trace 3.072 batch/24.573 lượt mẫu. Runtime training theo report là 1,785 giờ trên Tesla T4.

Đã đọc đủ 57 generation; 48 generation nhập từ v18 giữ nguyên từng byte. Prefix 2.560 loss/batch, pointer/hash/controller, model và 12 Adam state, sampler, hai CUDA RNG state, source/plan/manifest/identity đều đạt. Latest là generation `g-a4b58c79412c435994f4ce0188b168b8`, checkpoint SHA-256 `aacea6dabcc9572f4c96b52b1647cea519713c1866243961d4c2dea9846003fa`.

Phiên này chưa đến cuối epoch 2 ở bước 3.972 nên không chạy development mới. History vẫn có đúng một evaluation epoch 1/bước 1.986; best vẫn generation `g-c7418c4439ac45e9b051fa77b768709d`. Metric epoch 1 được tính lại và giữ nguyên: loss `0,6779944593369743`, EER `0,08595800524934383`, ROC-AUC `0,9702422376670303`, macro-F1 `0,9088565064507652`, threshold `0,00960088725961999`. Cổng leakage/sample-rate/amplitude/duration vẫn đạt; test không được dùng.

Output chuẩn `kaggle_outputs/b0-full-session-6-v19/` có 121/121 tệp sạch, không phát hiện HTML/502 và không cần tải lại. Bằng chứng tại `reports/b0_full_session_6_verification.*` và các report phiên 6 trong `reports/`. Không cần theo dõi version 19 nữa.

Phiên 7, nếu được thực hiện, phải đóng gói từ latest bước 3.072, sinh manifest/checksum mới, restore/bootstrap thử, kiểm tra quota/dataset/kernel và xin xác nhận cụ thể trước upload. Với giới hạn 512 bước, mốc tiếp theo dự kiến là bước 3.584/epoch 2/cursor 1.598; đây chưa phải kết quả. Full development epoch 2 và quyết định early stopping chỉ ở bước 3.972.

## Chuẩn bị phiên 7 ngày 30/10

Preflight đã hoàn tất từ output chuẩn v19. Gói exact-resume bảo toàn đủ 57 generation và có 115 tệp checkpoint/pointer/state. Restore trực tiếp cùng bootstrap CLI nhúng đều khớp model, optimizer, sampler epoch 2/cursor 1.086, RNG, controller, history và best. Source 56 tệp, identity D023 và lịch evaluation giữ nguyên.

Phần sẽ upload lên private dataset `ngctnnguyn1111/doantnnt-b0-full-resume` có 116 tệp/391.458.890 byte; manifest chuyển phiên SHA-256 `688af75ea12f99df92af5e29b25858534d5cab1e324d358e44ecc30970886369`. Gói không có audio, log hoặc thông tin đăng nhập. Bộ kernel khóa nằm tại `.kaggle/resume/session-7-kernel/`, profile `b0-full-resume-session-7`, private, GPU bật, internet tắt và gắn đúng ba dataset.

Remote preflight lúc 10:02 UTC+7 ghi nhận latest kernel `COMPLETE`, cả ba dataset private/`ready` và quota GPU còn 17,27/30,00 giờ. Sau xác nhận cụ thể của người dùng, payload đã upload; tải lại xác nhận private/`ready`, đúng 116 tệp/391.458.890 byte, inventory và manifest checkpoint/source khớp. Private kernel version 20 được gửi đúng một lần từ bộ khóa session7 và quan sát `RUNNING` lúc 10:18 UTC+7; quota sau gửi còn 17,26/30,00 giờ.

Theo dõi mỗi 20–30 phút, không gửi kernel khác khi version 20 còn `QUEUED`/`RUNNING`:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_workflow.py status
```

Khi trạng thái `COMPLETE`, tải output vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-7-v20
```

Sau khi tải, phải kiểm tra workflow/các lệnh con, bootstrap input bước3.072/115tệp, 57 generation nhập giữ nguyên, pointer/hash/model/optimizer/sampler/RNG/controller/history/best/source/identity và trace mới. Nếu chạy đủ512 bước, kết quả dự kiến là `paused` ở bước3.584, epoch2/cursor1.598 và28.669 lượt mẫu; đây chưa phải kết quả cho đến khi artifact được kiểm chứng. Nếu `ERROR`/`CANCELLED`, giữ output/log, xác định generation cuối đã commit và dừng chẩn đoán; không tự gửi lại.

## Chốt output phiên 7 ngày 30/10

Version20 đã `COMPLETE` và output được tải đúng một lần vào `kaggle_outputs/b0-full-session-7-v20/`. Inventory có139/139tệp sạch, không có tệp rỗng hoặc HTML/502. Workflow complete/cả ba lệnh mã0; bootstrap khôi phục đúng input bước3.072/115tệp. Training CUDA thêm512 optimizer step, dừng `paused` ở bước3.584; sampler epoch2/cursor1.598, tổng trace3.584 batch/28.669 lượt mẫu. Runtime training1,770giờ trên Tesla T4.

Đã đọc đủ66 generation;57 generation nhập v19 giữ nguyên từng byte. Prefix3.072 loss/batch, pointer/hash/controller, model và12 Adam state, sampler, hai CUDA RNG, source/plan/manifest/identity đều đạt. Latest là `g-e1fbbfcb17644703a8f530c4c7ce0300`, checkpoint SHA-256 `e3133e30ad9e0def016f9020ac61cdaaa0130266e773e3788bdc66dedd587f4f`. Best vẫn `g-c7418c4439ac45e9b051fa77b768709d` ở bước1.986.

Phiên này chưa đến biên epoch2 bước3.972 nên không chạy development mới. Metric epoch1 giữ nguyên: loss `0,6779944593369743`, EER `0,08595800524934383`, ROC-AUC `0,9702422376670303`, macro-F1 `0,9088565064507652`, threshold `0,00960088725961999`. Test không được dùng. Bằng chứng tại `reports/b0_full_session_7_verification.*`; không cần theo dõi version20 nữa.

## Chuẩn bị phiên 8 ngày 30/10

Preflight đã hoàn tất từ output chuẩn v20. Gói exact-resume bảo toàn đủ66 generation và có133tệp checkpoint/pointer/state, 465.837.119byte. Restore trực tiếp cùng bootstrap nhúng đều khớp bước3.584 và bảo toàn model/optimizer/sampler epoch2/cursor1.598, RNG, controller, history và best.

Phần sẽ upload lên private dataset `ngctnnguyn1111/doantnnt-b0-full-resume` có134tệp/465.871.883byte; manifest SHA-256 `ec65e82c8e326f0a2d212fc618d194c2315614bcd1f7dcaf50580218ff8db67f`. Gói cục bộ kể cả metadata có135tệp/465.872.132byte; không có audio, log hoặc thông tin đăng nhập. Source B0 vẫn là bộ khóa56tệp/tree `a46681120d36511e5d1bc7d5290e5488eecf136bf1ad55a93052afdb0e788925`; không publish source D024 mới trong workspace vào chuỗi này.

Bộ kernel khóa nằm tại `.kaggle/resume/session-8-kernel/`, profile `b0-full-resume-session-8`, private, GPU bật, internet tắt, gắn đúng ba dataset, expected input3.584 và report riêng phiên8. Preflight online ghi nhận latest kernel `COMPLETE`, ba dataset private/`ready`, quota còn15,49/30,00giờ GPU.

Sau xác nhận cụ thể, payload đã được upload thành private checkpoint dataset version7. Inventory remote khớp đúng134tệp/465.871.883byte, không thiếu/thừa/sai kích thước; manifest checkpoint tải ngược đúng SHA-256 và source manifest vẫn có56tệp/bundle đã khóa. Private kernel version21 được gửi đúng một lần từ bộ khóa session8, quan sát `RUNNING` lúc15:56 UTC+7; quota sau gửi còn15,48/30,00giờ.

Phiên8 chạm biên epoch2 sau388 bước, tại bước3.972, rồi đánh giá đủ4.115 development. Nếu EER không thấp hơn nghiêm ngặt EER epoch1 thì controller dừng sớm ngay tại3.972. Nếu EER cải thiện, best/threshold mới được ghi và phiên chạy tiếp trong giới hạn512 bước đến tối đa4.096, tức epoch3/cursor124. Không dùng test ở bất kỳ nhánh nào.

Theo dõi mỗi20–30phút, không gửi kernel khác khi version21 còn `QUEUED`/`RUNNING`:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_workflow.py status
```

Khi `COMPLETE`, tải output vào thư mục mới bằng `download-output --path kaggle_outputs/b0-full-session-8-v21` rồi kiểm chứng workflow, bootstrap input3.584/133tệp, generation nhập, development epoch2, early stopping/best/threshold và toàn bộ checkpoint state trước khi dùng kết quả.

## Chốt output phiên 8 ngày 31/10

Version21 đã `COMPLETE`; output chuẩn tại `kaggle_outputs/b0-full-session-8-v21/` có161/161tệp sạch, workflow complete/cả3 mã0 và bootstrap đúng input3.584/133tệp. Training CUDA thêm512 bước đến4.096, `paused`, sampler epoch3/cursor124 và tổng32.762 lượt mẫu train.

Đã đọc77 generation;66 generation nhập v20 giữ nguyên từng byte. Prefix3.584 loss/batch, pointer/hash/controller, model/12 Adam state, sampler, hai CUDA RNG, source/plan/manifest/identity đều đạt. Latest là `g-7568c302b08644449b1e1f9f9c866591`, checkpoint SHA-256 `7038d36e1ce621c931e7d8665a08fdb54d56f9a462b786cbed3c70fa52d1e6de`.

Epoch2 được đánh giá đủ4.115 development ở bước3.972. Loss `0,634388257066786`, EER `0,08333333333333333`, ROC-AUC `0,975126649790158`, macro-F1 `0,9116562633142766`, threshold `0,0070081922846535845`. EER cải thiện nghiêm ngặt so với epoch1 nên best chuyển sang generation `g-4174b60757a44324a9b864a4931e7a18` ở bước3.972; `bad_evaluations=0`, chưa có stop reason và epoch3 phải tiếp tục. Test không được dùng. Bằng chứng tại `reports/b0_full_session_8_verification.*`.

## Chuẩn bị phiên 9 ngày 31/10

Payload exact-resume từ latest4.096 đã được tạo và restore thử thành công. Gói có155tệp checkpoint/pointer/state của77 generation; phần upload gồm156tệp/557.835.353byte, manifest SHA-256 `cc9ebf87f898162980f2c24de5c4b80bf04dafa0c5d34af3cbdb345b0ed997d8`. Không có audio, log hoặc thông tin đăng nhập.

Bộ kernel khóa tại `.kaggle/resume/session-9-kernel/`, profile `b0-full-resume-session-9`, private/GPU/offline, gắn đúng ba dataset và yêu cầu input bước4.096. Source B0 vẫn là bộ khóa56tệp; source D024 trong workspace không được publish vào chuỗi này. Ba dataset hiện `ready`; quota GPU đã làm mới còn30/30giờ đến ngày07/11/2026.

Sau xác nhận cụ thể, payload đã được upload thành private checkpoint dataset version8. Remote `ready`, inventory khớp156tệp/557.835.353byte, manifest checkpoint tải ngược đúng SHA-256 và source manifest vẫn là bộ khóa56tệp. Private kernel version22 được gửi đúng một lần và quan sát `QUEUED`; quota sau gửi còn30/30giờ.

Theo dõi mỗi20–30phút, không gửi kernel khác khi version22 còn `QUEUED`/`RUNNING`:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_workflow.py status
```

Khi `COMPLETE`, tải output vào thư mục mới:

```powershell
.\.venv\Scripts\python.exe scripts\kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-9-v22
```

Phiên9 dự kiến tiếp tục4.096→tối đa4.608, epoch3/cursor636. Không có development mới ở mốc này; biên epoch3 và full development cuối ở bước5.958. Nếu `ERROR`/`CANCELLED`, giữ log/output và dừng để xác định generation cuối đã commit; không tự gửi lại.

## Chốt output phiên 9 ngày 01/11

Version22 đã `COMPLETE`; output chuẩn tại `kaggle_outputs/b0-full-session-9-v22/` có179/179tệp sạch, workflow complete/cả3 mã0 và bootstrap đúng input4.096/155tệp. Training CUDA thêm512 bước đến4.608, `paused`, sampler epoch3/cursor636 và tổng36.858 lượt mẫu train.

Đã đọc86 generation;77 generation nhập v21 giữ nguyên từng byte. Prefix4.096 loss/batch, pointer/hash/controller, model/12 Adam state, sampler, hai CUDA RNG, source/plan/manifest/identity đều đạt. Latest là `g-a4e2c835d3914007a0b01936dbadaa15`, checkpoint SHA-256 `60ba1744ef2065a12cd897fa9023978c4fa34faaf0753173fc322c7b29122974`.

Phiên này chưa đến biên epoch3 ở bước5.958 nên không chạy development mới. Best vẫn `g-4174b60757a44324a9b864a4931e7a18` tại bước3.972 với EER `0,08333333333333333`, loss `0,634388257066786`, ROC-AUC `0,975126649790158`, macro-F1 `0,9116562633142766` và threshold `0,0070081922846535845`. Test không được dùng. Bằng chứng tại `reports/b0_full_session_9_verification.*`.

## Chuẩn bị phiên 10 ngày 01/11

Payload exact-resume từ latest4.608 đã được tạo và restore thử thành công. Gói có173tệp checkpoint/pointer/state của86 generation; phần upload gồm174tệp/633.860.608byte, manifest SHA-256 `f7b8ccbdf7b03402324699a92f86b2018d67e393e0237adc66f25c48e15ebde1`. Không có audio, log hoặc thông tin đăng nhập.

Bộ kernel khóa tại `.kaggle/resume/session-10-kernel/`, profile `b0-full-resume-session-10`, private/GPU/offline, gắn đúng ba dataset và yêu cầu input bước4.608. Runner/run config/bootstrap nhúng khớp; source B0 vẫn là bộ khóa56tệp và source D024 trong workspace không được publish. Remote latest kernel `COMPLETE`, ba dataset `ready`, quota GPU còn28,13/30giờ tới ngày07/11/2026.

Sau xác nhận cụ thể, payload đã được upload thành private checkpoint dataset version9. Remote `ready`, inventory khớp đúng174tệp/633.860.608byte, không thiếu/thừa/sai kích thước; manifest tải ngược trùng từng byte và đúng SHA-256. Private kernel version23 được gửi đúng một lần từ bộ khóa session10 và quan sát `RUNNING` lúc07:44 UTC+7; quota sau gửi còn28,13/30,00giờ. Nếu phiên chạy đủ512 bước, mốc dự kiến là5.120/epoch3/cursor1.148 và40.954 lượt mẫu; vẫn chưa có development mới, sau đó còn838 bước đến biên epoch3/full development cuối ở5.958. Chỉ theo dõi version23; nếu `ERROR`/`CANCELLED`, giữ output/log và không tự gửi lại.

## Chốt output phiên 10 ngày 01/11

Version23 đã `COMPLETE`; output chuẩn tại `kaggle_outputs/b0-full-session-10-v23/` có197/197tệp sạch, workflow complete/cả3 mã0 và bootstrap đúng input4.608/173tệp. Training CUDA thêm512 bước đến5.120, `paused`, sampler epoch3/cursor1.148 và tổng40.954 lượt mẫu train.

Đã đọc95 generation;86 generation nhập v22 giữ nguyên từng byte. Prefix4.608 loss/batch, pointer/hash/controller, model/12 Adam state, sampler, hai CUDA RNG, source/plan/manifest/identity đều đạt. Latest là `g-a8df6e8856cc4dfba2b628cbc0e71ca5`, checkpoint SHA-256 `4172f7c5802e89c2fa0823c03288b9d8a5cb7399d2a73e65c1e80e881e0db5f7`.

Phiên này chưa đến biên epoch3 ở bước5.958 nên không chạy development mới. Best vẫn ở bước3.972 với EER `0,08333333333333333`; test không được dùng. Bằng chứng tại `reports/b0_full_session_10_verification.*`.

## Chuẩn bị phiên 11 ngày 01/11

Payload exact-resume từ latest5.120 đã được tạo và restore thử thành công. Gói có191tệp checkpoint/pointer/state của95 generation; phần upload gồm192tệp/710.702.335byte, manifest SHA-256 `189c5472cd2faf5cbc588f01284792f640f7c3b3deb2e8ff5beabfced34fe831`. Không có audio, log hoặc thông tin đăng nhập; 7/7 kiểm thử chuyển checkpoint đạt.

Bộ kernel khóa tại `.kaggle/resume/session-11-kernel/`, profile dự kiến `b0-full-resume-session-11`, private/GPU/offline, gắn đúng ba dataset và yêu cầu input bước5.120 trong `run-config.json`. Kiểm tra sau sự cố cho thấy bootstrap nhúng khớp nhưng `runner.py` còn nhúng run config phiên10; nhận định preflight trước đó rằng runner khớp run config là sai. Source B0 vẫn là bộ khóa56tệp và source D024 trong workspace không được publish.

Sau xác nhận cụ thể, payload đã được upload thành private checkpoint dataset version10. Remote `ready`, inventory khớp đúng192tệp/710.702.335byte, không thiếu/thừa/sai kích thước; manifest tải ngược trùng từng byte và đúng SHA-256. Private kernel version24 được gửi đúng một lần, ban đầu `RUNNING` rồi chuyển `ERROR`. Workflow v24 thực tế dùng profile phiên10, manifest cũ và expected step4.608; cổng checksum dừng trước training, không tạo checkpoint/metric mới. Runner tải trực tiếp từ v24 trùng SHA-256 với runner cục bộ nên không phải cache Kaggle.

Ngày02/11 đã bổ sung cổng so khớp cấu hình nhúng với `run-config.json`; mismatch làm lệnh gửi dừng trước khi gọi Kaggle. Nhóm workflow đạt12/12 và toàn suite191/191 trong152,672giây. Runner phiên11 được tái tạo với SHA-256 `abab4fe7b3d52a5dc1ac9902572e54d54c12f86c3b5cc04e20fbeb1c7d083030`, nhúng đúng profile/checksum/expected step5.120. Sau xác nhận của người dùng, private kernel version25 được gửi qua cổng mới và quan sát `RUNNING`; mã kéo ngược từ Kaggle trùng runner cục bộ. Dataset version10 được dùng lại, không upload lại. Chỉ theo dõi version25; khi `COMPLETE`, tải vào `kaggle_outputs/b0-full-session-11-v25/` và kiểm chứng, còn `ERROR`/`CANCELLED` thì giữ output/log và không tự gửi lại. Xem `reports/b0_full_session_11_preflight.*` và `reports/b0_full_session_11_v24_failure.*`.
