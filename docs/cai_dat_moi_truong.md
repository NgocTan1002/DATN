# Cài đặt môi trường Python cho dự án

Môi trường hiện tại được khóa cho Windows 64-bit, Python 3.12 và PyTorch CPU. Máy đang dùng không phát hiện CUDA, vì vậy không cài bản CUDA ở giai đoạn này.

## 1. Tạo môi trường mới

Chạy trong thư mục gốc dự án:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Nếu `.venv` đã tồn tại, chỉ cần kích hoạt và chạy lệnh cài từ `requirements-lock.txt`.

## 2. Kiểm tra sau khi cài

```powershell
python scripts/verify_environment.py
python -m unittest discover -s tests -v
```

Kết quả hợp lệ phải cho thấy:

- `torch` và `torchaudio` cùng phiên bản `2.11.0+cpu`.
- CUDA là `False` trên máy hiện tại.
- Resample 40 kHz xuống 16 kHz thành công.
- LFCC chỉ chứa giá trị hữu hạn.
- Toàn bộ kiểm thử dự án đạt.

## 3. Vai trò của hai tệp requirements

- `requirements.txt`: các thư viện trực tiếp mà dự án chủ động sử dụng, tất cả đã ghim phiên bản.
- `requirements-lock.txt`: toàn bộ thư viện trực tiếp và phụ thuộc gián tiếp của môi trường đã kiểm tra. Dùng tệp này để tái tạo đúng môi trường.

Khi chủ động nâng phiên bản thư viện, cài và kiểm thử trong `.venv` trước, sau đó cập nhật cả hai tệp. Không chạy nâng cấp hàng loạt ngay trước một thí nghiệm chính.

## 4. Nếu chuyển sang máy có NVIDIA GPU

Không giữ bản `+cpu` rồi kỳ vọng CUDA tự hoạt động. Cần chọn đúng lệnh cài theo phiên bản driver/CUDA từ trang cài đặt chính thức của PyTorch, giữ `torch` và `torchaudio` tương thích, rồi chạy lại:

```powershell
python scripts/verify_environment.py
python -m unittest discover -s tests -v
```

Chỉ thay tệp khóa của dự án sau khi pipeline dữ liệu và một bước huấn luyện đã chạy đúng trên máy GPU mục tiêu.
