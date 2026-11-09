# XLS-R T14 — đo chi phí kỹ thuật

- Status: failed
- Chỉ đo chi phí; head/LR là giá trị kỹ thuật giữ chỗ. Không chọn bậc, epoch hoặc LR, chưa có D028.
- Môi trường: {"python": "3.13.16", "torch": "2.11.0+cu130", "cuda": "13.0", "transformers": "5.20.2"}
- Dữ liệu: {}

| Mode | Tầng mở | Batch | Status | s/bước mean | median | CV | loader mean | compute mean | s/mẫu dev | VRAM allocated/reserved byte | Trainable | giờ/epoch train | giờ/dev-full |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

- Lỗi/cổng dừng: Transformers thiếu hoặc khác 4.57.6; dừng và hỏi người dùng, không cài thêm.
