# Huấn luyện XLS-R X0/X1 — D025, D026, D028, D029, D033

T15 hoàn thiện với cổng đóng. T17 đã mở `ready_for_training=true` cục bộ theo xác nhận
"Mở cổng, chạy" ngày 06/11/2026. T17a chỉ bổ sung đo thời gian và kiểm cục bộ;
bundle mới phải chờ Claude kiểm trước khi tiếp tục publish hoặc gửi phiên X0.
Không đọc test-full. Smoke chỉ kiểm pipeline, không dùng metric smoke làm kết quả khoa học.

## Môi trường và dữ liệu

Chạy PowerShell từ `C:/Users/Admin/Documents/DoAnTTNT-xlsr`:

```powershell
$python = 'C:/Users/Admin/Documents/DoAnTTNT/.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TEMP = "$PWD/.cache/t15/temp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$train = 'C:/Users/Admin/Documents/DoAnTTNT/data/processed/packed/train_development_20k_v2'
$dev = 'C:/Users/Admin/Documents/DoAnTTNT/data/processed/packed/closed_dev_full_v1'
$weights = 'C:/Users/Admin/Documents/DoAnTTNT/data/raw/xlsr/wav2vec2-xls-r-300m'
```

Train có 15.885 mẫu/57 shard; dev-full có 38.777 mẫu/186 shard. Manifest, audio config,
metadata, index và pretrained được kiểm checksum. Dry-run/smoke kiểm checksum metadata/index,
kích thước và digest waveform đã khóa trong metadata; không băm lại toàn waveform.
Run khoa học mới băm toàn waveform trước khi train. Resume giữ nguyên artifact và identity.

Theo D029, config pretrained giữ `layerdrop=0.1`, `configured_apply_spec_augment=true`.
Backbone/encoder luôn eval; chỉ block mở và classifier train. Vì vậy `effective_layerdrop=0`
và `effective_apply_spec_augment=false`. Các giá trị configured/effective được ghi riêng vào identity.

## Dry-run và technical smoke

```powershell
& $python scripts/train_xlsr.py --plan configs/xlsr_training.json --run-id X1-L4-lr1e-5 --training-seed 2027 --device cpu --dry-run --train-artifact $train --dev-artifact $dev --pretrained-dir $weights
& $python scripts/train_xlsr.py --plan configs/xlsr_training.json --run-id X1-L2-lr1e-5 --training-seed 2026 --device cpu --technical-smoke --smoke-session-steps 2 --train-artifact $train --dev-artifact $dev --pretrained-dir $weights --report-json .cache/t15/smoke-first.json
$smoke = Get-Content .cache/t15/smoke-first.json -Raw | ConvertFrom-Json
& $python scripts/train_xlsr.py --plan configs/xlsr_training.json --run-id X1-L2-lr1e-5 --training-seed 2026 --device cpu --technical-smoke --smoke-session-steps 1 --resume --run-root $smoke.run_root --train-artifact $train --dev-artifact $dev --pretrained-dir $weights --report-json .cache/t15/smoke-resumed.json
```

Smoke dùng batch 2, 4 mẫu train cân bằng và 8 mẫu dev cân bằng; hai bước đầu kết thúc một epoch
kỹ thuật và đánh giá đủ 8 mẫu, phiên resume thêm một bước. Root mới được tạo bằng thư mục tạm
`.cache/xlsr-technical-smoke/run-*`; không được dùng root run khoa học. Mỗi lần gọi hash lại pretrained.
Report ghi loss, trainable count, thời gian, RSS đỉnh, identity và checkpoint. Report tổng hợp của T15:
`reports/xlsr_t15_cpu_smoke.json`; run root và lệnh thực tế nằm trong report này.

## Run khoa học và resume (chỉ sau xác nhận mở cổng)

```powershell
& $python scripts/train_xlsr.py --plan configs/xlsr_training.json --run-id X1-L2-lr1e-5 --training-seed 2026 --device cuda:0 --train-artifact $train --dev-artifact $dev --pretrained-dir $weights
# Phiên tiếp theo: thêm --resume, giữ mọi trường semantic và cùng run root.
```

Batch 16, Adam classifier 1e-3 và encoder 1e-5 hoặc 3e-5; head 128/dropout 0,2;
float32, một GPU, worker 0; không scheduler, clipping, AMP. Tối đa 10 epoch/patience 2;
đánh giá đủ dev-full sau mỗi epoch. Best theo EER giảm nghiêm ngặt, hòa giữ checkpoint sớm nhất;
threshold lấy từ dev-full của best. Cục bộ mặc định `--session-steps 512`; có thể đổi số nguyên dương này giữa các phiên,
và giá trị vận hành này không nằm trong identity. Checkpoint vẫn mỗi 64 bước.
Kaggle truyền `--session-steps 9930` để chạy liên tục nhiều epoch trong một kernel,
đánh giá ngay ở mỗi biên epoch rồi train tiếp nếu còn ngân sách. Runner dừng khi run kết thúc
hoặc không còn đủ thời gian cho bước train/lượt dev-full và 120 giây dành cho commit.
Evaluation pending được ưu tiên khi resume; phải chuyển checkpoint giữa các kernel theo D023.

Bootstrap chụp `start_unix` ngay khi vào `main`, trước checksum, venv/pip và restore checkpoint.
Hạn tuyệt đối là `start_unix + sessions.max_session_hours * 3600` (11 giờ); bootstrap truyền
`--deadline-unix` cho runner. Runner dùng đồng hồ Unix cho hạn tuyệt đối, ưu tiên hạn này
hơn `--max-session-hours`. Hạn đã hết, NaN hoặc vô hạn bị từ chối trước khi đọc pretrained.
Không có hạn tuyệt đối thì ngân sách cục bộ tiếp tục dùng monotonic như trước.
Profile dry-run không truyền các điều khiển training mới và không chạy training.

```powershell
# Tham số vận hành tùy chọn khi đã được phép training khoa học:
# --session-steps 9930 --deadline-unix <timestamp Unix hữu hạn còn ở tương lai>
# Phiên resume có thể đổi --session-steps; giữ mọi trường identity khác.
```

`LATEST.json` công bố nguyên tử latest/best trong các thư mục `g-*/checkpoint.pt` và `state.json`.
Checkpoint chỉ chứa tham số trainable, Adam, sampler, RNG, controller/history/best/threshold và identity.
Prune chỉ bật riêng XLS-R; giữ latest, best và pending evaluation. B0 mặc định không prune.
Resume tải lại backbone từ pretrained đã kiểm SHA-256; từ chối identity, danh sách/shape/dtype trainable lệch.
Không đổi code, plan, seed, LR, số tầng, thiết bị, dữ liệu hoặc pretrained giữa các phiên.

## Kaggle: chỉ chuẩn bị cục bộ trong T15

```powershell
& $python .kaggle/xlsr-d028/preflight.py
& $python -m unittest discover -s tests -v
```

Preflight chặn socket và chỉ cho phép subprocess Git cục bộ; không gọi API Kaggle.
Nó tạo `.kaggle/xlsr-d028/project.json`, bundle tất định, `source_manifest.json`, và artifact kernel
riêng cho các profile dưới `.kaggle/xlsr-d028/prepared/`. Profile mặc định là dry-run.
Chỉ dùng `prepared/<profile>/` đã được thêm đủ dataset weights/offline, không gọi trực tiếp
`prepare-kernel` để gửi vì workflow chung chỉ gắn data/source mặc định.

Mount: packed `ngctnnguyn1111/doanttnt-t14-packed`, weights `ngctnnguyn1111/doanttnt-t14-weights`,
offline `ngctnnguyn1111/doanttnt-t14-offline-deps`; source mới dự kiến
`ngctnnguyn1111/doanttnt-xlsr-d028-source`. Bootstrap nhúng trong profile, dùng venv riêng,
pip `--no-index`, kiểm checksum wheels và giữ PyTorch/CUDA/NumPy nền như T14.
Cài wheel Linux cp313 và chạy GPU chưa được kiểm thực tế trong T15.

Chuyển checkpoint dùng nguyên module `deploy/kaggle/resume_checkpoint.py` của D023:
`build_payload(run_root, destination, dataset_ref)` đóng gói phẳng;
`restore_payload(input_root, run_root, manifest_sha256, expected_step, source_tree_sha256, report_path)`
kiểm mọi file, pointer, bước, source và identity trước khi công bố root mới.
Preflight nhúng đúng byte module này trong bootstrap vì module nằm ngoài source allowlist D023.
Test T15 kiểm roundtrip trainable-only và từ chối file truyền bị sửa.

Sau mỗi phiên, tải output và kiểm checkpoint; dựng private dataset chuyển checkpoint bằng build_payload,
khóa manifest SHA-256 và expected-step từ gói thực. Trước khi chuẩn bị phiên resume, thêm dataset checkpoint
vào metadata profile, thêm `--resume-manifest-sha256 <SHA thực>` và `--expected-step <bước thực>`
vào lệnh bootstrap, kiểm lại artifact rồi mới gửi sau xác nhận. Không tự đoán bước hoặc checksum.
Không sửa source/plan giữa phiên; dùng lại cùng bundle và identity. Checkpoint/report nằm ngoài project
(`/kaggle/working/checkpoints`, `/kaggle/working/reports`) để không bị runner cleanup xóa.

Các thao tác còn cần người dùng xác nhận: publish source/checkpoint dataset, push hoặc chạy kernel
(kể cả dry-run GPU), mở gate XLS-R và training khoa học. Không merge, commit hoặc push Git trong T15.

## Dự kiến số phiên theo timing của plan (T15b)

Nguồn số tính: `reports/xlsr_t15_kaggle_preflight.json`, mục `session_controls.budget_projection`.
Dùng đủ 10 epoch, không giả định early stopping:
`T = 9930 * train_step_estimated_seconds[layers] + 10 * dev_full_estimated_seconds`.
Mẫu số mỗi phiên là `11*3600 - 120` giây; số phiên bằng `ceil(T / mẫu số)`.

| Cấu hình | Train + dev-full tối đa (giờ) | Số phiên dự kiến |
|---|---:|---:|
| X0 | 11.375 | 2 |
| X1-L2 | 11.875 | 2 |
| X1-L4 | 12.250 | 2 |
| X1-L8 | 13.250 | 2 |

Đây là phép chiếu từ ước lượng bảo thủ trong plan, không phải số đo chạy Kaggle mới.
Chưa cộng bootstrap, hash toàn artifact lần đầu, checkpoint/transfer và biến thiên runtime;
các chi phí đó cũng nằm trong hạn phiên thực tế và có thể làm cần thêm phiên.
Đánh giá dev-full là một lượt nguyên khối; runner sẽ pause trước evaluation khi không đủ thời gian.
Mỗi kernel mới đều cần restore checkpoint cùng source/identity theo D023.

T15b chỉ sửa vận hành và kiểm offline. CPU smoke trong `reports/xlsr_t15_cpu_smoke.json`
là bằng chứng T15 trước bản sửa; T15b không chạy lại smoke pretrained hoặc training khoa học.

## Thứ tự D028 mục 5

1. Seed 2026: X1-L2, X1-L4, X1-L8 với encoder LR 1e-5, cộng X0.
2. Chọn mức tầng theo EER dev-full, hòa chọn ít tầng hơn; chạy thêm LR 3e-5 ở mức đó.
3. Chọn cặp tầng/LR theo EER dev-full, hòa chọn LR thấp hơn.
4. X1 cuối và X0 chạy thêm seed 2027/2028. Không dùng test chọn cấu hình hoặc seed.

Tất cả profile trong project chỉ là công thức lệnh; không được chạy toàn bộ lưới chéo.
Không đọc test-full, thay checksum cho qua kiểm, dùng root khoa học cho smoke hoặc diễn giải metric smoke.

## Log tiến độ và thời gian từng pha (T17a)

Runner in từng dòng JSON với `timestamp_utc` (ISO UTC) và `event`, flush ngay:
`backbone_load_start/end`, `model_build_start/end`, `rehash_train_start/end`,
`rehash_dev_start/end`, `train_start`, `checkpoint`, `evaluate_start/end`, `stop`.
Rehash chỉ xuất hiện khi thực sự gọi kiểm toàn artifact ở phiên khoa học đầu;
dry-run, technical smoke và resume vẫn bỏ qua bước này theo hành vi có sẵn.

`checkpoint` được in sau mỗi commit thành công, gồm commit định kỳ 64 bước và
commit ở biên epoch/đánh giá/dừng. Dòng có `global_step`, epoch hiện tại (từ 1),
`completed_epochs`, `mean_loss_last_64`, `mean_step_seconds`, `timing_window_steps`.
Loss lấy tối đa 64 bước gần nhất trong controller; thời gian lấy tối đa 64 bước
đã đo trong lần gọi hiện tại, nên cửa sổ thời gian có thể ngắn hơn sau resume.
`evaluate_end` có EER dạng tỷ lệ, loss và toàn bộ số đo lượt đánh giá.
`stop.reason` là `paused`, `early_stopped` hoặc `max_epochs`; khi paused,
`pause_cause` phân biệt `session_steps`, `budget_before_train` và
`budget_before_evaluation`. Log chỉ ghi nhận quyết định của controller hiện có.

Khối `timing` trong report cuối chứa:

- `rehash_train_seconds`, `rehash_dev_seconds` và `rehash_performed`: thời lượng
  kiểm toàn artifact đã thực hiện trong lần gọi này; 0/false khi bỏ qua.
- `backbone_load_seconds`, `model_build_seconds`: nạp backbone và dựng mô hình;
  dry-run cộng thời gian dựng cả 7 cấu hình.
- `train_steps_this_invocation`, `train_loader_seconds`, `train_model_seconds`,
  `train_seconds`, `mean_step_seconds`: tổng nạp batch và tính toán các bước train,
  không gồm checkpoint/đánh giá. Phần model gồm chuyển tensor sang thiết bị,
  normalization, forward, backward, kiểm gradient và optimizer.
- `evaluations_this_invocation`: mỗi lượt có `loader_seconds`, `model_seconds`,
  `total_seconds`, `samples`, `throughput_samples_per_second`. Total gồm cả gom
  records và tính metric; model gồm chuyển tensor/normalization/forward/loss và
  lấy scores về CPU. Cùng khối này được giữ trong history của checkpoint.
- `wall_seconds`: tổng thời gian runner đến lúc tạo report; `deadline_unix` là hạn
  tuyệt đối nhận qua CLI (null nếu dùng ngân sách monotonic cục bộ),
  `seconds_left_at_exit` và `deadline_overrun_seconds` phản ánh ngân sách khi tạo
  report. Chi phí ghi report cuối chưa nằm trong số đo này.
- `estimates`: ước tính bước train, dev-full và giới hạn giờ lấy nguyên từ plan,
  dùng đối chiếu số đo thực tế, không tự cập nhật ngân sách hoặc ngưỡng dừng.

Các phép đo dùng `perf_counter`; trên CUDA đồng bộ thiết bị trước khi đọc đồng hồ.
Thời gian bắt đầu lại từ 0 mỗi lần gọi, không được đưa vào identity, RNG hoặc
tiêu chí chọn best. Kiểm exact-resume vẫn so từng bit các tensor, optimizer, RNG,
sampler, loss/metric/history và best/threshold; riêng thời gian quan sát và UUID
generation không thể so bằng nhau giữa hai lần chạy. Không đổi source giữa các
phiên resume; bản thêm telemetry này phải được khóa bundle trước phiên đầu tiên.

Bootstrap in JSON UTC có `elapsed_seconds` từ thời điểm bắt đầu phiên tại các mốc
`bootstrap_start`, `offline_checksum_end`, `venv_pip_end`, `restore_end` (nếu resume),
`runner_start`, `runner_end` (kèm `exit_code`, kể cả runner lỗi).
Lệnh runner, hạn giờ tính từ đầu bootstrap và biên commit 120 giây giữ nguyên.
Số đo CPU technical smoke chỉ xác nhận pipeline/log; không dùng làm metric khoa học
hoặc thay ước tính thời gian GPU. T17a chỉ kiểm cục bộ, chờ Claude kiểm bundle mới
trước khi publish hoặc gửi phiên X0.

## Chế độ cặp 2 run song song trên Kaggle (D033 mục 1, T27)

Theo D033 mục 1, một kernel Kaggle `NvidiaTeslaT4` (2 GPU) được chạy tối đa 2 run D028 độc lập:

- Mỗi run là một tiến trình riêng với `--device cuda:0` (identity giữ nguyên `device = "cuda:0"`),
  tiến trình A có `CUDA_VISIBLE_DEVICES=0`, tiến trình B có `CUDA_VISIBLE_DEVICES=1`.
- Mỗi tiến trình đặt `OMP_NUM_THREADS=2` và `MKL_NUM_THREADS=2`, ghi rõ vào log `runner_start`/`runner_end`.
- Cả hai tiến trình dùng chung `--deadline-unix` tính từ `started_unix` của bootstrap.
- Mỗi run có `run_root` (`checkpoints/xlsr_{run_id}_seed{training_seed}`), `report`
  (`reports/xlsr_{run_id}_seed{training_seed}.json`) và `log` (`xlsr_{run_id}_seed{training_seed}.log`) tách biệt.
- Trước khi chạy, bootstrap kiểm số GPU bằng `detect_gpu_count`; nếu khác 2 GPU thì dừng ngay toàn bộ kernel
  với thông báo tiếng Việt, không tự chuyển sang chạy tuần tự.
- Bootstrap đợi cả hai tiến trình kết thúc; một run lỗi không làm dừng hay sửa run còn lại, và mã thoát bootstrap
  khác 0 nếu bất kỳ run nào có mã thoát khác 0.
- Profile cặp `X0-seed2027-X0-seed2028` và `X0-seed2027-X0-seed2028-dry-run` được sinh qua
  `scripts/kaggle_workflow.py::build_xlsr_pair_profiles` và `prepare_xlsr_pair_kernel` (private, `enable_gpu=true`,
  `machine_shape="NvidiaTeslaT4"`, `enable_internet=false`, đúng 4 dataset như profile X0 một run). Chế độ một run
  giữ nguyên lệnh và biến môi trường trùng byte với baseline.

Log nằm ngay dưới `working_root`, khớp bộ lọc tải output `^[^/]+\.log$`.
Mỗi dòng stdout/stderr được ghi nguyên byte vào log riêng và in ngay vào stdout bootstrap với tiền tố
`[A X0 seed2027] ` hoặc `[B X0 seed2028] `; mỗi tiến trình có luồng đọc riêng.
SDK Kaggle không liệt kê tên accelerator hợp lệ: lần push dry-run cặp đầu tiên phải xác nhận
`machine_shape="NvidiaTeslaT4"` thực sự cấp đúng 2 GPU và đo peak RSS.
Mỗi run một mình dùng khoảng 21,5 GB theo report X0 seed 2026 và X1 seed 2028;
phần lớn là memmap dữ liệu packed dùng chung. Chỉ thực hiện push sau xác nhận riêng.

## Chạy trên Google Colab AI Pro và kiểm tương đương môi trường (D033 mục 2, 3, 4, T27)

Runner `scripts/train_xlsr.py` ghi khối `hardware` vào report và vào `state.json` của mỗi generation (nằm ngoài
`identity`, không đổi `checkpoint_version=4`): tên GPU (`torch.cuda.get_device_name`), số GPU thấy được, driver nếu
đọc được, `torch.version.cuda`, phiên bản `torch`/`transformers`/`numpy`, hai cờ TF32 và `OMP_NUM_THREADS` (trên CPU
ghi rõ `không có GPU`). Cờ `--expected-gpu <chuỗi>` từ chối chạy nếu tên GPU thực tế không chứa chuỗi kỳ vọng; khi
`--resume`, nếu `state.json` có khối `hardware` với tên GPU khác GPU hiện tại thì runner từ chối resume (checkpoint v4
cũ không có `hardware` vẫn resume bình thường).

Trước khi dùng Colab cho bất kỳ run khoa học nào, phải chạy kiểm tương đương môi trường (parity) bằng
`scripts/check_xlsr_env_parity.py` (hoặc qua `colab/xlsr_colab_entry.py --mode parity` trong `colab/run_xlsr.ipynb`):

```powershell
& $python scripts/check_xlsr_env_parity.py `
  --plan configs/xlsr_training.json `
  --checkpoint <đường_dẫn_checkpoint_X1-L8-lr3e-5_seed2026> `
  --generation g-eb35f3c2379f4a47bd22744c9be1dd8f `
  --reference-report <đường_dẫn_report_Kaggle_X1-L8-lr3e-5_seed2026.json> `
  --dev-artifact $dev `
  --pretrained-dir $weights `
  --device cuda:0 `
  --expected-gpu L4 `
  --output-json .cache/t27/parity_report.json `
  --output-md .cache/t27/parity_report.md
```

Ba ngưỡng D033 mục 4 được khóa cứng làm hằng số trong `scripts/check_xlsr_env_parity.py` (không nhận từ CLI):
`|ΔEER| <= 0,02` điểm %, số mẫu đổi quyết định tại `decision_threshold` tham chiếu `<= 10`, và `max |Δ score| <= 0,05`.
Entrypoint `colab/xlsr_colab_entry.py` kiểm phiên bản thư viện theo `effective_environment` của T14 (`torch 2.11.0`,
`transformers 4.57.6`, `numpy 2.1.3`), ghi `pip freeze` vào log, chép packed `train_development_20k_v2`,
`closed_dev_full_v1` và `pretrained` từ Google Drive về ổ cục bộ (`/content/work`) rồi kiểm đủ checksum trước khi chạy
(cấm đọc memmap trực tiếp trên Drive và cấm truyền artifact test), đồng thời đồng bộ checkpoint, report và log về Drive
sau mỗi lần runner commit checkpoint. Registry v1 (D032) giữ trên Kaggle T4; Colab không dùng cho registry v1.

## X1R: lowpass 7 kHz theo D035 (T29)

Run `X1-L8-lr3e-5-lp7k` lấy nguyên cấu hình của `X1-L8-lr3e-5` từ GRID:
mở 8 tầng, encoder lr 3e-5, classifier lr 1e-3, batch 16, trần 10 epoch và patience 2.
Runner suy điều kiện `lowpass_7k` từ `run_id` qua `VARIANTS`; plan và GRID không đổi.

`src/data/lowpass.py` port bộ lọc T19b: FIR windowed-sinc 513 tap, Hann đối xứng,
cắt 7.000 Hz ở sample rate 16 kHz, FFT 65.536, bù trễ 256 mẫu, zero-extension,
rồi áp lại `apply_amplitude_policy(..., "peak")`. Khi nạp module, SHA hệ số float32
phải bằng `973b82a36db8dab4f802de2102c5feed2bfb8cf6bb5846472c07d9984ab4c977`.
Wrapper train/dev chỉ thay `item["waveform"]` sau khi packed dataset trả đoạn 64.000 mẫu;
mọi metadata, seed/crop stream giữ nguyên. Processor normalization vẫn thực hiện sau bước lọc.
Các run thuộc GRID dùng trực tiếp dataset cũ.

Identity X1R giữ `run_id` biến thể, `run` chứa cấu hình base, thêm `input_condition` gồm
`decision`, `condition`, `base_run_id`, toàn bộ `filter_parameters` và SHA file triển khai.
Identity run cũ chỉ đổi trung thực `source_files` và `source_tree_sha256` theo source mới;
mọi khóa còn lại phải trùng baseline T29. Logic băm source giữ nguyên.
Checkpoint/report theo template `xlsr_X1-L8-lr3e-5-lp7k_seed{seed}`.
X1R thuộc registry phụ D035, không thêm vào registry v1 D032.

Preflight sinh các profile mới:

- `X1-L8-lr3e-5-lp7k-seed2026`, `X1-L8-lr3e-5-lp7k-seed2027`,
  `X1-L8-lr3e-5-lp7k-seed2028` và bản `-dry-run` tương ứng.
- `X1-L8-lr3e-5-lp7k-seed2026-X1-L8-lr3e-5-lp7k-seed2027` và bản `-dry-run`.

Metadata X1R private, GPU `NvidiaTeslaT4`, internet tắt và cùng 4 dataset như X1 hiện có.
Cặp dùng bootstrap D033: hai tiến trình có `CUDA_VISIBLE_DEVICES=0/1`, OMP/MKL mỗi tiến trình 2 thread,
chung deadline, checkpoint/report/log riêng. Profile X0 cũ giữ nguyên lệnh và env.

Ví dụ CPU dry-run từ worktree W (dùng Python, artifact và pretrained có sẵn ở main):

```powershell
$main = 'C:/Users/Admin/Documents/DoAnTTNT'
$python = "$main/.venv/Scripts/python.exe"
& $python -B scripts/train_xlsr.py --run-id X1-L8-lr3e-5-lp7k `
  --training-seed 2026 --device cpu --dry-run `
  --train-artifact "$main/data/processed/packed/train_development_20k_v2" `
  --dev-artifact "$main/data/processed/packed/closed_dev_full_v1" `
  --pretrained-dir "$main/data/raw/xlsr/wav2vec2-xls-r-300m" `
  --report-json .cache/t29/dry_run_x1r_seed2026.json
```

T29 chỉ chạy preflight trên `.cache/t29/preflight_copy/{base,new}/`, không ghi đè prepared thật.
Bundle mới cần Claude kiểm và xác nhận riêng trước khi publish/push.
