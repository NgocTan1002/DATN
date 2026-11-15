# Sao lưu dữ liệu nặng lên Google Drive

## Mục đích và phạm vi

Sao lưu một chiều bằng `rclone copy` từ repository cục bộ lên `gdrive:DoAnTTNT_backup`. Máy cục bộ vẫn là nơi làm việc; Git quản lý code, cấu hình và báo cáo. Sáu thư mục nguồn là `checkpoints/`, `kaggle_outputs/`, `.kaggle/`, `kaggle_raw/`, `data/processed/`, `data/raw/`.

Không chạy huấn luyện, tải output Kaggle hoặc công việc khác ghi vào các thư mục nguồn trong lúc sao lưu, kiểm và tạo manifest. `copy` giữ các file dư trên Drive; đây không phải bản snapshot có lịch sử phiên bản. File cùng đường dẫn có thay đổi có thể được cập nhật ở lần sao lưu tiếp theo.

## Điều kiện

- Người dùng tự cài rclone và cấu hình remote tên `gdrive`, dùng client_id OAuth riêng, scope `drive.file`. Agent không đọc cấu hình, token hay client secret.
- Chạy các lệnh dưới đây từ `C:\Users\Admin\Documents\DoAnTTNT` trong PowerShell; rclone phải có trong PATH.
- Kiểm tra `rclone about gdrive: -v`: mã thoát 0, trống trên 200 GiB, không có cảnh báo `shared Google Drive client_id`.
- Ứng dụng OAuth hiện ở trạng thái Testing: quyền truy cập hết hạn sau 7 ngày. Khi có lỗi xác thực/token hết hạn, dừng; người dùng tự chạy `rclone config reconnect gdrive:` và đăng nhập lại. Agent không thực hiện đăng nhập lại.
- Dry-run phải được kiểm và người dùng phải xác nhận trực tiếp trước khi tải thật.

## Sao lưu lại

Giữ nguyên bộ lọc `configs/drive_backup_filter.txt`. Kiểm tra tiến trình và dung lượng trước khi chạy:

```powershell
rclone about gdrive: -v
rclone size . --filter-from configs/drive_backup_filter.txt
rclone copy . gdrive:DoAnTTNT_backup --filter-from configs/drive_backup_filter.txt --dry-run --log-file logs/drive_backup_dryrun.log --log-level INFO
```

rclone nối thêm vào log nếu log đã tồn tại; trước một dry-run mới, chỉ xóa đúng file log cũ `logs/drive_backup_dryrun.log`, không xóa dữ liệu nguồn. Kiểm log không chứa đường dẫn bị loại hoặc cảnh báo shared client_id; mọi file thuộc sáu thư mục. Thông báo bỏ qua cập nhật thời gian thư mục cha `data` và sáu thư mục gốc trong dry-run là hợp lệ.

Sau xác nhận của người dùng:

```powershell
rclone copy . gdrive:DoAnTTNT_backup --filter-from configs/drive_backup_filter.txt --transfers 4 --checkers 8 --tpslimit 8 --drive-chunk-size 128M --log-file logs/drive_backup_copy.log --log-level INFO --stats 5m
```

Nếu bị ngắt vì mạng, chạy lại đúng lệnh trên để bỏ qua file đã có và khớp. Nếu có lỗi xác thực, quota hoặc giới hạn tải, dừng và lưu nguyên văn lỗi trước khi xử lý; không tự thay tài khoản, remote, bộ lọc hoặc thông số.

## Kiểm và manifest

```powershell
rclone check . gdrive:DoAnTTNT_backup --filter-from configs/drive_backup_filter.txt --one-way --log-file logs/drive_backup_check.log --log-level INFO
rclone hashsum sha256 . --filter-from configs/drive_backup_filter.txt --output-file reports/drive_backup_manifest_sha256.txt
rclone copyto reports/drive_backup_manifest_sha256.txt gdrive:DoAnTTNT_backup/drive_backup_manifest_sha256.txt
rclone size gdrive:DoAnTTNT_backup
(Get-Content -LiteralPath reports/drive_backup_manifest_sha256.txt | Measure-Object -Line).Lines
rclone lsf gdrive:DoAnTTNT_backup -R --files-only --include ".git/**" --include ".cache/**" --include ".xlsr-offline-env/**" --include "kaggle.json"
```

Chỉ ghi sao lưu hoàn tất khi `check` exit 0, log ghi `0 differences found` và không có lỗi. Manifest SHA-256 phải có số dòng bằng số file nguồn đã lọc, tồn tại cả cục bộ lẫn trên Drive. Số file trên Drive phải bằng số file nguồn cộng một manifest cho bản sao lưu đầu tiên. Kiểm `--one-way` chỉ yêu cầu mọi file nguồn có bản tương ứng; không xác nhận Drive không có file dư. Lệnh `lsf --files-only` kiểm các file loại trừ phải không in đường dẫn. Nếu bỏ `--files-only`, lệnh còn in thư mục thông thường để duyệt cây; không được coi mọi dòng đó là file bị cấm. Để kiểm cả thư mục và đủ các mẫu loại trừ, đọc toàn bộ `rclone lsf gdrive:DoAnTTNT_backup -R` rồi kiểm các thành phần đường dẫn `.git`, `.cache`, `.venv`, `.kilo`, `.xlsr-offline-env`, `__pycache__`, tên file `kaggle.json`, `.env`, `.env.*` và tiền tố `.kaggle/tmp`.

`rclone check` sử dụng checksum chung mà hai bên hỗ trợ; không mặc định đó là SHA-256. Manifest SHA-256 được sinh từ file nguồn để có chứng cứ kiểm độc lập khi khôi phục.

Nếu check lỗi/khác biệt, chạy lại copy tối đa hai lần rồi kiểm lại. Nếu vẫn lỗi, dừng, trích log và không ghi đạt. Không sửa dữ liệu nguồn để làm cho check khớp.

## Khôi phục một thư mục

Người dùng chủ động chạy khi cần khôi phục, từ gốc repository:

```powershell
rclone copy gdrive:DoAnTTNT_backup/checkpoints checkpoints
```

Lệnh có thể cập nhật file cục bộ cùng tên; kiểm vị trí đích trước khi chạy và đối chiếu manifest SHA-256 sau khi khôi phục. Không chạy lệnh khôi phục trong phiên sao lưu.

## Loại trừ

Các mẫu không có `/` đầu khớp ở mọi độ sâu:

- `kaggle.json`, `.env`, `.env.*`: tránh đưa thông tin xác thực và biến môi trường lên Drive.
- `.git/**`: bản Git lồng trong dữ liệu; code đã được quản lý riêng.
- `.cache/**`, `__pycache__/**`: cache có thể tạo lại.
- `.venv/**`, `.xlsr-offline-env/**`: môi trường Python có thể cài lại.
- `.kilo/**`: nội dung công cụ ngoài phạm vi sao lưu.
- `/.kaggle/tmp/**`: file tạm của workflow.
- `- *` cuối bộ lọc: bỏ mọi file ngoài sáu cây nguồn đã cho phép.

`.kaggle/` được sao lưu theo bộ lọc; không đọc hay in nội dung bí mật để kiểm tra.

## Lệnh bị cấm trong nhiệm vụ

Không dùng `rclone sync`, `move`, `delete`, `purge`, `dedupe`, `cleanup`, `link`; không đổi quyền chia sẻ, không sửa cấu hình rclone hoặc xử lý token/client secret. Không xóa, sửa, di chuyển file nguồn; không thay split, metadata hoặc `configs/audio.json`; không commit/push khi chưa được người dùng yêu cầu.

## Bằng chứng BACKUP-01

Người dùng xác nhận tải thật nguyên văn **“xác nhận”**, ngày **10/11/2026**. Dry-run trước tải chọn 5.551 file / 130.167.623.644 byte; log `logs/drive_backup_dryrun.log`. Kết quả hoàn tất, thời gian và trạng thái kiểm phải đối chiếu `logs/drive_backup_copy.log`, `logs/drive_backup_check.log` và `docs/current_state.md`; không suy ra hoàn tất từ việc đã bắt đầu copy.

Lần đầu hoàn tất kiểm ngày 11/11/2026: copy exit 0, thời gian rclone 3 giờ 54 phút 12,9 giây; check MD5 exit 0 với `0 differences found` và `5551 matching files`. Manifest SHA-256 có 5.551 dòng, đã tải và đối chiếu checksum bản Drive; Drive có 5.552 file gồm manifest. Lệnh kiểm `lsf` ban đầu không có `--files-only` in 1.676 thư mục thông thường, 0 file; quét toàn cây và chạy lại với `--files-only` đều không thấy mục bị cấm. Đây là bằng chứng cho lần sao lưu này; lần tiếp theo phải chạy kiểm mới.

## Đã xóa trên máy (CLEANUP-01, 11/11/2026)

Người dùng xác nhận nguyên văn “xác nhận” ngày 11/11/2026. Đã xóa sáu nhóm, gồm đúng 20 thư mục dưới đây. data/raw/vsasv_parquet/ được giữ lại trên máy vì hai smoke test đọc trực tiếp.

| Thư mục đã xóa | Số file | Byte |
|---|---:|---:|
| `data/processed/eval_chunks/` | 796 | 25936718610 |
| `data/processed/train_chunks/` | 171 | 5796315730 |
| `.kaggle/t14/packed/` | 7 | 19257600479 |
| `.kaggle/t14/weights/` | 4 | 1269739083 |
| `.kaggle/resume/` | 1501 | 4932494713 |
| `kaggle_outputs/b0-full-smoke-v13/` | 42 | 82195817 |
| `kaggle_outputs/b0-full-session-1-v14/` | 26 | 57278126 |
| `kaggle_outputs/b0-full-session-2-v15/` | 45 | 113298505 |
| `kaggle_outputs/b0-full-session-3-v16/` | 63 | 170119357 |
| `kaggle_outputs/b0-full-session-4-v17/` | 85 | 240105062 |
| `kaggle_outputs/b0-full-session-4-v17-redownload/` | 85 | 246503781 |
| `kaggle_outputs/b0-full-session-5-v18/` | 103 | 319363535 |
| `kaggle_outputs/b0-full-session-6-v19/` | 121 | 393037695 |
| `kaggle_outputs/b0-full-session-7-v20/` | 139 | 467526970 |
| `kaggle_outputs/b0-full-session-8-v21/` | 161 | 559563541 |
| `kaggle_outputs/b0-full-session-9-v22/` | 179 | 635665317 |
| `kaggle_outputs/b0-full-session-10-v23/` | 197 | 712583467 |
| `kaggle_outputs/b0-full-session-11-v24/` | 3 | 53174 |
| `kaggle_outputs/b0-full-session-11-v24-error/` | 3 | 53174 |
| `kaggle_outputs/b0-full-session-11-v25/` | 215 | 790318781 |

Tổng kích thước logic 3.946 file: 61.980.534.917 byte. Dung lượng trống ổ C ngay trước/sau xóa: 69.136.007.168 / 110.597.459.968 byte; tăng đo được 41.461.452.800 byte. Staging T14 packed/weights được tạo bằng os.link trong .cache/t14_g2.py; chúng dùng chung dữ liệu đĩa với bản gốc còn giữ, nên không cộng dung lượng logic của hardlink vào dung lượng thu hồi.

File không có trên Drive và đã xóa cùng thư mục: `.kaggle/resume/session-8-kernel/__pycache__/runner.cpython-312.pyc`, 32.101 byte. Các file `.cache` của `data/raw/vsasv_parquet` không có trên Drive; chúng vẫn được giữ trên máy cùng thư mục Parquet.

Từ nay tuyệt đối không dùng `rclone sync`: đồng bộ từ máy lên Drive sẽ xóa trên Drive các mục đã bỏ khỏi máy. Chỉ sao lưu bằng copy; không tự dọn phần dư trên Drive. Manifest BACKUP-01 mô tả bản sao lưu trước cleanup, không còn là danh sách đầy đủ của file đang hiện diện cục bộ.

### Khôi phục khi cần

Từ gốc repository, chọn và chạy đúng cặp lệnh cho thư mục cần lấy lại bên dưới; đây là hướng dẫn, không phải các lệnh đã chạy trong CLEANUP-01. Lệnh kiểm đặt Drive làm nguồn và dùng --one-way để xác nhận đủ bản khôi phục.

```powershell
rclone copy gdrive:DoAnTTNT_backup/data/processed/eval_chunks data/processed/eval_chunks -P
rclone check gdrive:DoAnTTNT_backup/data/processed/eval_chunks data/processed/eval_chunks --one-way

rclone copy gdrive:DoAnTTNT_backup/data/processed/train_chunks data/processed/train_chunks -P
rclone check gdrive:DoAnTTNT_backup/data/processed/train_chunks data/processed/train_chunks --one-way

rclone copy gdrive:DoAnTTNT_backup/.kaggle/t14/packed .kaggle/t14/packed -P
rclone check gdrive:DoAnTTNT_backup/.kaggle/t14/packed .kaggle/t14/packed --one-way

rclone copy gdrive:DoAnTTNT_backup/.kaggle/t14/weights .kaggle/t14/weights -P
rclone check gdrive:DoAnTTNT_backup/.kaggle/t14/weights .kaggle/t14/weights --one-way

rclone copy gdrive:DoAnTTNT_backup/.kaggle/resume .kaggle/resume -P
rclone check gdrive:DoAnTTNT_backup/.kaggle/resume .kaggle/resume --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-smoke-v13 kaggle_outputs/b0-full-smoke-v13 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-smoke-v13 kaggle_outputs/b0-full-smoke-v13 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-1-v14 kaggle_outputs/b0-full-session-1-v14 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-1-v14 kaggle_outputs/b0-full-session-1-v14 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-2-v15 kaggle_outputs/b0-full-session-2-v15 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-2-v15 kaggle_outputs/b0-full-session-2-v15 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-3-v16 kaggle_outputs/b0-full-session-3-v16 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-3-v16 kaggle_outputs/b0-full-session-3-v16 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-4-v17 kaggle_outputs/b0-full-session-4-v17 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-4-v17 kaggle_outputs/b0-full-session-4-v17 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-4-v17-redownload kaggle_outputs/b0-full-session-4-v17-redownload -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-4-v17-redownload kaggle_outputs/b0-full-session-4-v17-redownload --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-5-v18 kaggle_outputs/b0-full-session-5-v18 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-5-v18 kaggle_outputs/b0-full-session-5-v18 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-6-v19 kaggle_outputs/b0-full-session-6-v19 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-6-v19 kaggle_outputs/b0-full-session-6-v19 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-7-v20 kaggle_outputs/b0-full-session-7-v20 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-7-v20 kaggle_outputs/b0-full-session-7-v20 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-8-v21 kaggle_outputs/b0-full-session-8-v21 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-8-v21 kaggle_outputs/b0-full-session-8-v21 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-9-v22 kaggle_outputs/b0-full-session-9-v22 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-9-v22 kaggle_outputs/b0-full-session-9-v22 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-10-v23 kaggle_outputs/b0-full-session-10-v23 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-10-v23 kaggle_outputs/b0-full-session-10-v23 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-11-v24 kaggle_outputs/b0-full-session-11-v24 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-11-v24 kaggle_outputs/b0-full-session-11-v24 --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-11-v24-error kaggle_outputs/b0-full-session-11-v24-error -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-11-v24-error kaggle_outputs/b0-full-session-11-v24-error --one-way

rclone copy gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-11-v25 kaggle_outputs/b0-full-session-11-v25 -P
rclone check gdrive:DoAnTTNT_backup/kaggle_outputs/b0-full-session-11-v25 kaggle_outputs/b0-full-session-11-v25 --one-way

```

Bằng chứng: logs/local_cleanup_a_before.txt (kiểm kê và mốc kiểm thử), logs/local_cleanup_a_check.log (20 check riêng cho 20 thư mục; mỗi lệnh exit 0/0 differences), logs/local_cleanup_a_after.txt (xác nhận xóa, kiểm kê và kiểm thử sau xóa). Bản Drive vẫn 5.552 file / 130.168.433.095 byte.

Kiểm sau xóa ngày 11/11/2026: unittest 444/444, exit 0 (trước 321,187 giây; sau 322,182 giây); hai smoke test trước/sau đều exit 0, nội dung chức năng khớp sau khi bỏ thời gian đo; leakage trước/sau exit 0 và report gốc được phục hồi nguyên byte/mtime. Kiểm kê giữ lại 0 khác biệt về tên, kích thước, mtime; Parquet vẫn 68 file/10.580.529.098 byte. Chưa chạy bất kỳ lệnh khôi phục nào trong CLEANUP-01.

## BACKUP-02 (12/11/2026)

Người dùng xác nhận nguyên văn “Xác nhận tải thật” ngày 12/11/2026. Sao lưu bằng copy, không xóa phần dư BACKUP-01 trên Drive.

- A: bổ sung 8 file / 2738461961 byte bằng configs/drive_backup_filter.txt tới gdrive:DoAnTTNT_backup.
- B1: ảnh chụp main trên Drive 2902 file / 273858929 byte tại gdrive:DoAnTTNT_backup/_local_only/2026-11-12/main; bộ lọc configs/drive_backup_local_main_filter.txt.
- B2: ảnh chụp W trên Drive 1512 file / 96375881 byte tại gdrive:DoAnTTNT_backup/_local_only/2026-11-12/xlsr; bộ lọc configs/drive_backup_local_xlsr_filter.txt.
- C: DoAnTTNT_all_2026-11-12.bundle nằm trong ảnh chụp main/.cache/backup02/, 13.767.377 byte, SHA-256 42714b0816786b4d7b8e62a4dcb7c599b89ebf6299d485a9571e2c4f3d2ce4ec; verify đạt, có main 146805445893b062c37ce021532c76940358f1cc và feat/xlsr-runner-d028 4e04ac11c672ae08ca0136eff2bff5debef7899e.

Bộ lọc main thêm - /.cache/r1/edge*/** ngay trước + /.cache/r1/** để loại hồ sơ trình duyệt Edge chứa dữ liệu phiên. Giữ nguyên 17 file bằng chứng R1; không xóa hồ sơ trên máy. Không phát hiện hồ sơ Edge hoặc tên file bị cấm trong cây _local_only/2026-11-12 trên Drive.

Sau lỗi MD5 của chính log copy B1 đang ghi, người dùng xác nhận nguyên văn “Cho phép thêm ngoại lệ log” ngày 12/11/2026. Thêm - /logs/drive_backup02_*.log trước + **; log BACKUP-02 giữ trên máy. Dừng lượt B1 cũ, dry-run lại sạch rồi tiếp tục B1 và B2. Không tạo lại bundle.

Main trên Drive có tổng 2914 file / 275016192 byte, gồm tập đã lọc ở trên và 12 log BACKUP-02 dư được tải trước ngoại lệ. Giữ nguyên các file dư theo quy tắc copy; check/manifest chỉ bao phủ tập đã lọc, không xác nhận các log dư này.

Ba check one-way exit 0, 0 differences found; mọi lần kiểm/copy lại nằm trong .cache/backup02/checks_uploaded.json và logs/drive_backup02_{heavy,local_main,local_xlsr}_{copy,check}.log. Log của phiên có thể thay đổi trong lúc ghi; đối chiếu bàn giao để biết chênh lệch theo thời điểm.

Tổng Drive trước/sau: 5552 / 9989 file; 130168433095 / 133279034916 byte. Không giảm số file; đã kiểm kích thước bundle trên Drive.

Sinh cả ba manifest SHA-256 vào .cache/backup02/ trước, sau đó mới chép cả ba sang reports/, nên manifest main không chứa chính nó hoặc hai manifest mới còn lại. Ba bản được tải riêng vào gdrive:DoAnTTNT_backup/_local_only/2026-11-12/; MD5 bản Drive khớp bản máy (SHA-256 của nội dung manifest ghi bên dưới). Các ảnh chụp được tạo trước lần cập nhật tài liệu này.

- reports/drive_backup02_manifest_heavy_sha256.txt: 1614 dòng; SHA-256 74b93a75f4056a01d38c044b5485c25c0a8973ac68e7ec599477207803c7b23c; MD5 máy/Drive 464306d4e84bd1d0b1928b0c5926168c.
- reports/drive_backup02_manifest_local_main_sha256.txt: 2902 dòng; SHA-256 8c1e40e75bff0b5a2dcc78a3bfa021bb6ac52d5debf15a0667d5dcd2cd4b151d; MD5 máy/Drive 9d0f492f1474e547478e10d998c44f5a.
- reports/drive_backup02_manifest_local_xlsr_sha256.txt: 1512 dòng; SHA-256 5be65dc7f5cf516c26a86140016311b0671a3c332876b2c1549e650649043a55; MD5 máy/Drive e35e6929df4b5b23a2c056fdccf5e422.

Khôi phục Git: tải bundle từ main/.cache/backup02/ về một thư mục cục bộ rồi, từ thư mục chứa bundle, chạy vào một thư mục mới chưa tồn tại:

```powershell
git clone DoAnTTNT_all_2026-11-12.bundle DoAnTTNT_restored
```

Bundle bảo vệ lịch sử/refs; thay đổi chưa commit được giữ trong ảnh chụp B1/B2, cần lấy lại riêng. Không chạy các lệnh khôi phục trong phiên sao lưu này.

Các log `logs/drive_backup02_*.log` (14 file) được đẩy riêng sau khi mọi lệnh rclone ghi log đã kết thúc, ngày 12/11/2026; check 0 differences, 14 matching files. Output tại `.cache/backup02c/`.
