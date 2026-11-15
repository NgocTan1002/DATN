# Kiểm tương đương môi trường Colab (D033 mục 4)

Hướng dẫn này dùng để chạy phép kiểm tương đương trên Google Colab trước khi dùng Colab cho bất kỳ run khoa học nào. Phép kiểm chấm lại checkpoint best X1-L8-lr3e-5 seed 2026 trên toàn bộ dev-full và so với 38.777 score đã lưu trong report Kaggle. Không train, không đọc dữ liệu test.

Ngưỡng đạt (khóa trong `scripts/check_xlsr_env_parity.py`, theo D033 mục 4), phải đạt đồng thời:

- |ΔEER| ≤ 0,02 điểm %;
- số mẫu đổi quyết định tại threshold đã khóa ≤ 10;
- max |Δ score| ≤ 0,05.

Không đạt thì không dùng Colab cho run khoa học và phải lập quyết định mới.

## Dữ liệu đã có trên Drive

Tất cả nằm trong thư mục `DoAnTTNT_backup` (tạo bởi BACKUP-01/02). Trong Colab, Drive gắn tại `/content/drive/MyDrive`, nên các đường dẫn là:

| Thành phần | Đường dẫn trong Colab |
|---|---|
| Source đã kiểm (ảnh chụp worktree XLS-R ngày 12/11, trùng code T27b) | `/content/drive/MyDrive/DoAnTTNT_backup/_local_only/2026-11-12/xlsr` |
| Packed dev-full (khoảng 13 GB) | `/content/drive/MyDrive/DoAnTTNT_backup/data/processed/packed/closed_dev_full_v1` |
| Trọng số XLS-R 300M | `/content/drive/MyDrive/DoAnTTNT_backup/data/raw/xlsr/wav2vec2-xls-r-300m` |
| Checkpoint X1 seed 2026 (thư mục run) | `/content/drive/MyDrive/DoAnTTNT_backup/kaggle_outputs/xlsr-d028-X1-L8-lr3e-5-seed2026/checkpoints/xlsr_X1-L8-lr3e-5_seed2026` |
| Report Kaggle tham chiếu | `/content/drive/MyDrive/DoAnTTNT_backup/kaggle_outputs/xlsr-d028-X1-L8-lr3e-5-seed2026/reports/xlsr_X1-L8-lr3e-5_seed2026.json` |

Claude đã kiểm ngày 15/11/2026: các file source chính trong ảnh chụp (`scripts/check_xlsr_env_parity.py`, `scripts/train_xlsr.py`, `src/data/packed.py`, `src/data/xlsr.py`, `src/models/xlsr.py`, `configs/xlsr_training.json`, `colab/xlsr_colab_entry.py`) có MD5 trùng bản code T27b đã kiểm; thư mục checkpoint có generation `g-eb35f3c2379f4a47bd22744c9be1dd8f`.

Không sửa, không xóa gì trong `DoAnTTNT_backup`. Đầu ra ghi vào thư mục mới `DoAnTTNT_colab_outputs`.

## Các bước trong Colab

Mở một notebook Colab mới (không cần file `colab/run_xlsr.ipynb`; notebook đó thiếu bước lấy source và cài torch). Chọn **Runtime → Change runtime type → GPU L4** (mặc định theo D033 mục 2; nếu không có L4 thì dùng T4 và ghi lại). Bật chạy nền nếu muốn đóng trình duyệt.

**Ô 1 — gắn Drive và kiểm GPU:**

```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

**Ô 2 — chép source vào ổ cục bộ:**

```python
!rm -rf /content/xlsr && cp -r /content/drive/MyDrive/DoAnTTNT_backup/_local_only/2026-11-12/xlsr /content/xlsr
%cd /content/xlsr
!ls colab scripts/check_xlsr_env_parity.py configs/xlsr_training.json
```

**Ô 3 — cài đúng phiên bản thư viện** (entry point dừng nếu torch khác bản 2.11.0, transformers khác 4.57.6 hoặc numpy khác 2.1.3):

```python
import torch; print(torch.__version__)
```

Nếu không phải `2.11.0…`, cài:

```python
!pip install torch==2.11.0 torchaudio==2.11.0
```

Sau đó cài các gói còn lại:

```python
!pip install --no-deps transformers==4.57.6 huggingface-hub==0.36.2 tokenizers==0.22.2 numpy==2.1.3
```

Sau ô này, **Runtime → Restart session** (bắt buộc vì numpy và torch đã đổi), rồi chạy lại ô 1 và `%cd /content/xlsr`. Kiểm lại:

```python
import torch, numpy, transformers
print(torch.__version__, torch.version.cuda, numpy.__version__, transformers.__version__, torch.cuda.get_device_name(0))
```

Nếu `torch.cuda.is_available()` là `False` hoặc báo lỗi driver CUDA, dừng và chụp lỗi gửi Claude; không thử phiên bản torch khác.

**Ô 4 — tạo thư mục liên kết dữ liệu** (entry point cần dev-full và trọng số nằm chung một thư mục; liên kết đặt ngoài `/content/drive` và không sửa Drive):

```python
!mkdir -p /content/links
!ln -sfn /content/drive/MyDrive/DoAnTTNT_backup/data/processed/packed/closed_dev_full_v1 /content/links/closed_dev_full_v1
!ln -sfn /content/drive/MyDrive/DoAnTTNT_backup/data/raw/xlsr/wav2vec2-xls-r-300m /content/links/wav2vec2-xls-r-300m
!ls -la /content/links
```

**Ô 5 — chạy phép kiểm** (đổi `--expected-gpu L4` thành `T4` nếu dùng T4):

```python
!python colab/xlsr_colab_entry.py \
  --drive-data-dir /content/links \
  --work-dir /content/work \
  --drive-output-dir /content/drive/MyDrive/DoAnTTNT_colab_outputs/parity_x1_seed2026 \
  --run-id X1-L8-lr3e-5 \
  --training-seed 2026 \
  --expected-gpu L4 \
  --mode parity \
  --checkpoint /content/drive/MyDrive/DoAnTTNT_backup/kaggle_outputs/xlsr-d028-X1-L8-lr3e-5-seed2026/checkpoints/xlsr_X1-L8-lr3e-5_seed2026 \
  --generation g-eb35f3c2379f4a47bd22744c9be1dd8f \
  --reference-report /content/drive/MyDrive/DoAnTTNT_backup/kaggle_outputs/xlsr-d028-X1-L8-lr3e-5-seed2026/reports/xlsr_X1-L8-lr3e-5_seed2026.json
```

Entry point chép dev-full và trọng số sang `/content/work` rồi kiểm checksum (bước này mất khoảng 10–20 phút vì dev-full khoảng 13 GB), chấm 38.777 mẫu, ghi report và đồng bộ về Drive. Trên T4 Kaggle một lượt dev-full mất khoảng 1.830 giây; tổng thời gian dự kiến khoảng 45–60 phút. Mã thoát 0 là đạt, mã 2 là không đạt ngưỡng.

**Ô 6 — ghi nhận mức tiêu hao:** trước ô 5 và sau khi ô 5 xong, mở **Runtime → View resources** hoặc trang quản lý đơn vị điện toán của Colab, ghi lại số đơn vị còn lại và thời điểm. Số này dùng để ước tính chi phí run khoa học theo D036 mục 6.

## Gửi kết quả cho Claude kiểm

rclone của dự án chỉ thấy file do chính nó tạo (scope `drive.file`), nên Claude không đọc được file do Colab tạo. Tải thủ công từ Drive web thư mục `DoAnTTNT_colab_outputs/parity_x1_seed2026` (gồm `xlsr_parity_X1-L8-lr3e-5_seed2026.json`, `.md` và file log `colab_parity_…log`), đặt vào `C:\Users\Admin\Documents\DoAnTTNT\.cache\colab_parity\`, rồi báo Claude kèm:

- loại GPU được cấp (ô 1), phiên bản torch/CUDA/numpy/transformers (ô 3);
- mã thoát của ô 5 và thời gian chạy;
- số đơn vị điện toán trước/sau (ô 6).

Claude sẽ kiểm ba ngưỡng, khối `hardware`, thứ tự file và so với report Kaggle trước khi Colab được dùng cho run khoa học.

## Lưu ý

- Không chạy `--mode train` hay `resume` trước khi phép kiểm được Claude xác nhận đạt.
- Không trỏ bất kỳ đường dẫn nào tới `closed_test_full_v1`; entry point từ chối đường dẫn test.
- Ba seed của cùng một mô hình ở cùng quy mô phải chạy trên cùng một loại GPU (D033 mục 3); nếu phép kiểm chạy trên L4 thì run khoa học trên Colab cũng dùng L4.
