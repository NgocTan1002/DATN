# Hướng dẫn kiểm chứng Kaggle GPU

## Mục tiêu

Quy trình này kiểm tra môi trường GPU chính trước mọi run dài. Đây là smoke kỹ thuật, không phải kết quả khoa học và không dùng `closed_test` để chọn cấu hình, policy, threshold hoặc checkpoint.

## 1. Chuẩn bị dataset riêng tư

Tạo một Kaggle Dataset ở chế độ **Private** và đặt tên phiên bản rõ ràng, ví dụ `vsasv-development-v2-20261013`. Dataset cần chứa:

- 68 shard Parquet cục bộ đã xác minh; tối thiểu phải có đủ 66 shard mà `development-20k-v2` tham chiếu;
- `data/manifests/development_20k_v2.csv`;
- các split smoke train/development nếu không lấy chúng từ repository;
- một file mô tả ghi số shard, tổng byte và SHA-256 manifest.

Không đưa token Kaggle, audio, Parquet, artifact đóng gói, checkpoint hoặc log lớn vào Git. Sau mỗi lần thay dữ liệu, tạo một version Kaggle Dataset mới thay vì ghi đè không truy vết.

## 2. Tạo notebook GPU

Trong Notebook settings, bật GPU accelerator và tắt Internet nếu repository đã được gắn dưới dạng dataset đầu vào. Nếu cần clone repository, bật Internet chỉ trong giai đoạn lấy đúng commit rồi ghi lại commit đó trong báo cáo.

Không chạy `pip install -r requirements.txt` hoặc `pip install -r requirements-lock.txt`: hai file này cố ý khóa PyTorch CPU cho máy Windows cục bộ. Giữ nguyên `torch` và `torchaudio` CUDA do Kaggle cung cấp, chỉ cài phần phụ thuộc còn thiếu:

```bash
python -m pip install -r requirements-kaggle.txt
```

## 3. Xác minh runtime trước khi đọc dữ liệu

```bash
python scripts/verify_gpu_runtime.py \
  --require-cuda \
  --output /kaggle/working/reports/kaggle_runtime.json
```

Cổng đạt yêu cầu `cuda_available=true`, có ít nhất một GPU, và báo cáo ghi Python, PyTorch, CUDA, cuDNN, tên GPU, VRAM cùng commit. Nếu lệnh báo CPU-only, dừng tại đây; không cài lại PyTorch một cách mù quáng.

## 4. Smoke B0 lần đầu

Thay `<dataset-slug>` bằng đường dẫn dataset riêng tư đang gắn. Lệnh sau chạy một optimizer step, đánh giá toàn bộ smoke development, lưu checkpoint và đo peak VRAM:

```bash
python scripts/smoke_train_lfcc_lcnn.py \
  --device cuda \
  --parquet-dir /kaggle/input/<dataset-slug>/vsasv_parquet/data \
  --epochs 1 \
  --max-steps 1 \
  --checkpoint /kaggle/working/checkpoints/b0_kaggle_phase1.pt \
  --report-json /kaggle/working/reports/b0_kaggle_phase1.json \
  --report-md /kaggle/working/reports/b0_kaggle_phase1.md
```

## 5. Kiểm tra tiếp tục từ checkpoint

Chạy trong một process mới để chứng minh model state, optimizer state và bộ đếm được nạp lại rồi hoàn thành thêm một optimizer step:

```bash
python scripts/smoke_train_lfcc_lcnn.py \
  --device cuda \
  --parquet-dir /kaggle/input/<dataset-slug>/vsasv_parquet/data \
  --epochs 1 \
  --max-steps 1 \
  --resume-from /kaggle/working/checkpoints/b0_kaggle_phase1.pt \
  --checkpoint /kaggle/working/checkpoints/b0_kaggle_resumed.pt \
  --report-json /kaggle/working/reports/b0_kaggle_resumed.json \
  --report-md /kaggle/working/reports/b0_kaggle_resumed.md
```

Cổng resume đạt khi báo cáo thứ hai ghi `resume.used=true`, `optimizer_state_restored=true`, `initial_steps=1`, `additional_steps_completed=1`, tổng bước bằng 2, development hữu hạn và checkpoint restore difference không vượt `1e-7`.

Smoke resume này chứng minh có thể tiếp tục model/optimizer trong process mới. Nó chưa cam kết tiếp tục chính xác giữa một epoch ở đúng batch kế tiếp; hợp đồng resume đầy đủ của training loop tuần 4 phải lưu sampler/RNG và batch cursor riêng.

## 6. Artifact cần tải xuống sau smoke

Tải về hoặc lưu thành output version:

- `kaggle_runtime.json`;
- hai báo cáo B0 JSON/Markdown;
- checkpoint phase 1 và resumed;
- log lệnh chạy có commit, Kaggle Dataset version, manifest SHA-256 và thời gian thực tế.

Không cập nhật dự báo runtime từ smoke một bước. Benchmark ngân sách pilot phải dùng cùng manifest, batch, policy và số bước đã khóa theo D019. Chỉ cân nhắc Colab trả phí nếu Kaggle không đáp ứng sau khi đã ghi lại giới hạn hoặc lỗi thực tế và người dùng xác nhận chi phí.
