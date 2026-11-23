# Chấm bằng cấp phép trên Colab L4 (D037 mục 3–5)

Hướng dẫn này dùng cho `scripts/evaluate_full_split.py --permit ... --device cuda`. Có hai chế độ; **không upload gì lên Drive trong tài liệu này**, người dùng tự thực hiện khi xác nhận.

- **Chế độ 1 — dev dry-run (PREGATE-02):** cấp phép `split="dev"` ghi trên máy, không cần cổng, không tạo khóa. Chỉ để chứng minh notebook tái tạo được score dev đã lưu.
- **Chế độ 2 — test:** chỉ sau khi người dùng mở cổng (`test_gate_open=true`) và tạo cấp phép test trên máy bằng `--split test --write-permit`. Lệnh này tạo khóa trong `.cache/test_once` (hoặc `_x1r`) và ghi cấp phép, không chấm. Mỗi mô hình × điều kiện đã ghi xong là cuối cùng; không chấm lại. Phần chưa chấm tiếp tục với cùng cấp phép.

B0-v2 chỉ chấm trên CPU máy cục bộ (script từ chối B0 khi `--device cuda`). Colab chỉ chấm X0, X1, X1R.

## Trên máy: tạo cấp phép

Dev dry-run (ví dụ X1 seed 2026, điều kiện original):

```powershell
$env:PYTHONUTF8=1; .\.venv\Scripts\python.exe scripts/evaluate_full_split.py --split dev --write-permit .cache\pregate02\permit_dev_x1.json --models X1-L8-lr3e-5-seed2026 --conditions original
```

Test (chỉ sau khi người dùng mở cổng; X1R dùng thêm `--config configs/test_evaluation_x1r.json`):

```powershell
$env:PYTHONUTF8=1; .\.venv\Scripts\python.exe scripts/evaluate_full_split.py --split test --write-permit <đường dẫn cấp phép> --models <model_id ...>
```

## File cần có trên Drive

Chế độ dev dry-run:

| Thành phần | Ghi chú |
|---|---|
| Source snapshot của repo (có `scripts/`, `src/`, `configs/`, `colab/xlsr_colab_entry.py`) | phải gồm bản `evaluate_full_split.py` mới và `configs/test_evaluation.json` đã điền SHA |
| `reports/model_registry_v1.json` | đúng bản có SHA trong cấp phép |
| File cấp phép dev | tạo trên máy |
| `closed_dev_full_v1.csv` và thư mục `closed_dev_full_v1/` (13 GB) | dev-full đã có trên Drive (`DoAnTTNT_backup`) |
| Trọng số `wav2vec2-xls-r-300m` | đã có trên Drive |
| Checkpoint X1 seed 2026 và report Kaggle nguồn (để kiểm tái tạo) | đã có trên Drive |

Chế độ test thêm: `closed_test_full_v1.csv` và `closed_test_full_v1/` (test-full 30.698 mẫu, khoảng 12 GB; **cần người dùng xác nhận riêng khi đưa lên**, SHA khớp `reports/pre_test_inventory.json`), checkpoint của từng mô hình trong cấp phép, cấp phép test, và registry chính thức tương ứng (`model_registry_v1.json` hoặc `model_registry_x1r.json`).

## Các ô Colab

Chọn **Runtime → Change runtime type → GPU L4**.

**Ô 1 — gắn Drive, kiểm GPU:**

```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

**Ô 2 — chép source vào ổ cục bộ:**

```python
!rm -rf /content/src_tree && cp -r <thư mục source snapshot trên Drive> /content/src_tree
%cd /content/src_tree
!ls scripts/evaluate_full_split.py configs/test_evaluation.json colab/xlsr_colab_entry.py
```

**Ô 3 — cài đúng phiên bản** (script dừng nếu torch, CUDA, transformers hoặc numpy lệch hằng trong `colab/xlsr_colab_entry.py`: 2.11.0+cu130, 13.0, 4.57.6, 2.1.3). Các bước cài và restart như `docs/guides/colab_parity_check.md` ô 3, rồi kiểm:

```python
import torch, numpy, transformers
print(torch.__version__, torch.version.cuda, numpy.__version__, transformers.__version__, torch.cuda.get_device_name(0))
```

**Ô 4 — chép dữ liệu và checkpoint vào ổ cục bộ** (không đọc memmap trực tiếp trên Drive). Với dev (đổi `dev` thành `test` ở chế độ test):

```python
!mkdir -p /content/data /content/ckpt /content/out
!cp /content/drive/MyDrive/DoAnTTNT_backup/data/manifests/closed_dev_full_v1.csv /content/data/
!cp -r /content/drive/MyDrive/DoAnTTNT_backup/data/processed/packed/closed_dev_full_v1 /content/data/
!cp -r /content/drive/MyDrive/DoAnTTNT_backup/data/raw/xlsr/wav2vec2-xls-r-300m /content/data/
# checkpoint: giữ đúng đường dẫn tương đối ghi trong registry (trường "checkpoint")
!mkdir -p /content/ckpt/<thư mục cha của checkpoint trong registry>
!cp <checkpoint trên Drive> /content/ckpt/<đường dẫn tương đối trong registry>
```

**Ô 5 — chạy chấm bằng cấp phép:**

```python
!python scripts/evaluate_full_split.py \
  --permit /content/drive/MyDrive/DoAnTTNT_colab_outputs/<file cấp phép> \
  --registry <đường dẫn registry chính thức> \
  --data-root /content/data \
  --checkpoint-root /content/ckpt \
  --reference-root /content/src_tree \
  --device cuda \
  --output-dir /content/out
```

Dev dry-run có thể thêm `--limit 200` để chạy thử nhanh. Test không dùng `--limit`. Script kiểm cấp phép, SHA registry, SHA checkpoint và SHA ba file artifact trước khi mở dữ liệu; lệch thì dừng. Mỗi mô hình × điều kiện ghi một file `<split>__<model_id>__<điều kiện>.json` trong `--output-dir`. Chạy lại với cùng cấp phép sẽ bỏ qua file đã hoàn tất (`complete=true`) và chỉ chấm lại file dở. `--reference-root` chỉ để kiểm tái tạo score dev bằng report nguồn; nếu report không có thì bỏ qua bước tái tạo.

**Ô 6 — chép kết quả về Drive:**

```python
!mkdir -p /content/drive/MyDrive/DoAnTTNT_colab_outputs/<tên lần chạy>
!cp /content/out/*.json /content/drive/MyDrive/DoAnTTNT_colab_outputs/<tên lần chạy>/
```

Sau đó tải thủ công về máy (như `colab_parity_check.md`) để Claude kiểm.

## Lưu ý

- Không sửa registry, cấp phép hay config sau khi tạo cấp phép; mọi SHA lệch làm script dừng.
- Không xóa khóa trong `.cache/test_once*` và không tạo cấp phép test thứ hai cho cùng mô hình.
- Dev dry-run phải đạt tái tạo theo D034 trước khi dùng notebook cho test.
