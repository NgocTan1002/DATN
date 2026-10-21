# Điều khiển quy trình Kaggle bằng CLI

## Mục tiêu

Workflow này cho phép thực hiện toàn bộ chu trình từ repository cục bộ:

1. kiểm tra Kaggle CLI, xác thực và dataset VSASV;
2. đóng gói snapshot code hiện hành thành private source dataset;
3. tạo hoặc cập nhật version của source dataset;
4. sinh private GPU kernel gắn data dataset và source dataset;
5. gửi kernel, theo dõi trạng thái và tải output về workspace.

Audio, Parquet, checkpoint và log lớn không được đưa vào Git. File cấu hình thật nằm tại `.kaggle/project.json`, đã bị `.gitignore` loại trừ; bản mẫu có thể commit nằm tại `configs/kaggle_cli.example.json`.

## Kiến trúc

```text
Repository cục bộ
  -> project_source.zip
  -> private source dataset
                         \
                          -> private GPU kernel -> reports/checkpoints
                         /
Private VSASV dataset
```

Kernel vật chất hóa snapshot code vào `/kaggle/working/DoAnTTNT`, rồi chạy các lệnh của profile đã chọn. Runner hỗ trợ cả file ZIP và thư mục `project_source` do Kaggle tự giải nén, đồng thời tự tìm source/data dataset ở layout trực tiếp hoặc cấu trúc lồng `/kaggle/input/datasets/<owner>/<slug>/`. Mỗi source bundle có SHA-256 và danh sách checksum từng file; runner từ chối chạy nếu manifest, danh sách file hoặc checksum không khớp run config. Sau khi ghi report, bản sao source trong `/kaggle/working` được dọn để output chỉ giữ report/checkpoint cần thiết.

## Cấu hình lần đầu

Kaggle CLI phải có trong `PATH` và đã xác thực. Sao chép file mẫu nếu `.kaggle/project.json` chưa tồn tại:

```powershell
New-Item -ItemType Directory -Force .kaggle
Copy-Item configs/kaggle_cli.example.json .kaggle/project.json
```

Điền ba ref theo dạng `<owner>/<slug>`:

- `data_dataset_ref`: private dataset VSASV đã có;
- `source_dataset_ref`: private dataset nhỏ dùng lưu snapshot code;
- `kernel_ref`: private kernel điều khiển bằng CLI.

Không đặt API key hoặc token trong file này.

## Kiểm tra kết nối và dữ liệu

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py doctor
```

Cổng đạt khi CLI chạy được, API xác thực thành công, dataset đầu vào có 68 Parquet và đủ `development_20k_v2.csv`, `smoke_train.csv`, `smoke_dev.csv`.

Chỉ kiểm tra cấu hình cục bộ, không gọi mạng:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py doctor --offline
```

## Đóng gói và cập nhật source dataset

Chỉ đóng gói cục bộ:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py bundle-source
```

Tạo source dataset nếu chưa có, hoặc tạo version mới nếu đã tồn tại:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py publish-source `
  --message "Update runner and training code" `
  --confirm-private-upload
```

Source dataset luôn được tạo ở chế độ private. Cờ `--confirm-private-upload` là bắt buộc để tránh vô tình gửi source code ra ngoài máy. Bundle chỉ gồm Python source, cấu hình JSON, runner và `requirements-kaggle.txt`; không gồm audio, Parquet, checkpoint, report hay secret.

## Chuẩn bị và chạy kernel

Profile `runtime` chỉ kiểm tra GPU và ghi `kaggle_runtime.json`:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py prepare-kernel `
  --profile runtime

.\.venv\Scripts\python.exe scripts/kaggle_workflow.py push-kernel `
  --profile runtime
```

Profile `b0-smoke-resume` chạy kiểm tra GPU rồi gọi entrypoint production hai lần: pha đầu dừng có chủ đích tại `global_step=1`, pha sau nạp checkpoint và hoàn tất `global_step=2`. Cả hai pha dùng `development-20k-v2`, 68 shard và giới hạn kỹ thuật 4 mẫu train/2 mẫu development; kết quả không dùng để chọn policy:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py push-kernel `
  --profile b0-smoke-resume
```

Profile `b0-benchmark` chạy ba repeat độc lập với cùng technical subset cân bằng 256 train/128 development, batch 8 và 32 step. Tổng hợp timing sau khi tải output:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py push-kernel `
  --profile b0-benchmark

.\.venv\Scripts\python.exe scripts/summarize_b0_benchmark.py `
  --reports reports/b0_benchmark_repeat_1.json reports/b0_benchmark_repeat_2.json reports/b0_benchmark_repeat_3.json `
  --output-json reports/b0_benchmark_summary.json `
  --output-md reports/b0_benchmark_summary.md
```

Benchmark version 8 ngày 20/10/2026 đã khóa ngân sách D020 ở 896 step với batch 8, worker 0, Adam và learning rate 0,001. Đây là quyết định vận hành; metric của technical subset không dùng để chọn policy.

Có thể publish source và gửi kernel bằng một lệnh:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py deploy `
  --profile runtime `
  --message "CLI runtime smoke" `
  --confirm-private-upload
```

Mỗi lần `push-kernel` tạo một version/run mới của cùng private kernel. Metadata chỉ gắn các dataset được lệnh của profile tham chiếu: `runtime` chỉ cần source dataset, còn `b0-smoke-resume` gắn thêm data dataset. Không dùng profile smoke để báo cáo metric khoa học.

## Theo dõi và tải kết quả

Đọc trạng thái một lần:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py status
```

Chờ tối đa 30 phút, kiểm tra mỗi 20 giây:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py wait `
  --timeout-seconds 1800 `
  --interval-seconds 20
```

Tải output về thư mục đã bị Git bỏ qua:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py download-output `
  --path kaggle_outputs/latest `
  --force
```

Kiểm tra `kaggle_cli_workflow.json` trước. Chỉ trạng thái `complete` với mọi `return_code=0` mới chứng minh workflow thành công.

Trên Windows, workflow ép subprocess dùng UTF-8 để lệnh log/output không lỗi khi Kaggle trả văn bản tiếng Việt.

## Chạy profile pilot

Thêm profile vào `.kaggle/project.json` dưới dạng danh sách các lệnh, mỗi lệnh là một danh sách chuỗi. Các placeholder được hỗ trợ:

- `{data_root}` → `/kaggle/input/<data-dataset-slug>`;
- `{source_root}` → `/kaggle/input/<source-dataset-slug>`;
- `{project_root}` → `/kaggle/working/DoAnTTNT`;
- `{working_root}` → `/kaggle/working`.

Ba profile đã khóa là `b0-pilot-none`, `b0-pilot-peak` và `b0-pilot-rms-dbfs`. Chạy từng profile thành một kernel version riêng, chờ hoàn tất và tải output trước khi gửi profile kế tiếp:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py push-kernel `
  --profile b0-pilot-none
```

Thay tên profile lần lượt cho `peak` và `rms_dbfs`. Cả ba dùng cùng manifest v2, seed 2026, sampler, batch 8, worker 0, Adam, learning rate 0,001, 896 optimizer step và full development. Chỉ amplitude policy cùng tên checkpoint/report được phép khác. Không sửa profile sau khi pilot đầu tiên đã bắt đầu nếu chưa hủy cả phép so sánh và ghi quyết định mới.

## An toàn và giới hạn

- Data dataset và source dataset phải để private.
- `enable_internet` mặc định là `false`; chỉ bật khi có lý do đã ghi lại.
- Không cài `requirements-lock.txt` trên Kaggle vì file đó khóa PyTorch CPU.
- Không dùng `closed_test` để chọn policy, threshold, checkpoint, ngân sách hoặc mô hình.
- Không tải checkpoint hoặc output lớn vào Git.
- Không xóa version dataset cũ khi cập nhật source; lịch sử version là bằng chứng tái lập.
