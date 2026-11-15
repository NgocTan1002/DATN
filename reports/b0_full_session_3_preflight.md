# Chuẩn bị B0 full training phiên 3 — ngày 26/10/2026

- Preflight đạt; private kernel version 16 đã gửi thành công, API quan sát RUNNING, profile `b0-full-resume-session-3`. Chưa xác nhận GPU resume hoặc bước mới bằng output.
- Mục tiêu: tiếp tục checkpoint v15 từ bước 1.024 → 1.536, tối đa thêm 512 bước theo D023. Điểm đã kiểm chứng gần nhất vẫn là bước 1.024; chưa có tiến độ mới.
- Phạm vi giữ nguyên: 15.885 train/4.115 development trên manifest 20.000 mẫu/66 shard; pool cục bộ 68 shard, seed 2026, policy peak.
- Đã đối chiếu hash của cả 19 generation với `b0_full_session_2_verification.json`; input latest SHA-256 `40812f8331d4b4d615ce91c6af4b72c863a4e17cb946041c42f8e94686c620ca`.
- Gói có 39 file checkpoint/pointer/trạng thái, tổng 113.121.458 byte. Thêm manifest và metadata cục bộ thành 41 file/113.132.175 byte; metadata tạo dataset không phải file dữ liệu upload.
- Manifest chuyển phiên SHA-256: `76b604d3ff793f3ccec4f47e21ae916510172ccea14c76f74002ee16ce67ae97`.
- Source bundle SHA-256: `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`; source tree `a46681120d36511e5d1bc7d5290e5488eecf136bf1ad55a93052afdb0e788925`. Cả 56 file source huấn luyện giữ nguyên.
- Restore gói thật và chạy bootstrap CLI nhúng đều đạt: model/optimizer/sampler/RNG/controller khớp payload gốc; mọi file đã chuyển khớp checksum. Báo cáo `b0_full_session_3_transfer_check.json` và `b0_full_session_3_embedded_check.json`.
- Profile `b0-full-resume-session-3` có `--resume`, expected input step 1.024, manifest checksum mới, output report riêng cho phiên 3. Bộ kernel chuẩn bị nằm ngoài Git tại `.kaggle/resume/session-3-kernel/`, metadata private/internet off và đủ source/data/checkpoint dataset.
- Quota đọc trước phiên: còn 26,24 giờ GPU. Metadata checkpoint dataset ID 12838409 xác nhận private; source và data dataset đều ready. Chưa cam kết đủ hạn mức cho toàn bộ nhiều epoch.
- Bộ duyệt tự động ban đầu chặn payload mới vì quyền cũ chỉ bao phủ input512. Người dùng đã xác nhận cụ thể **“Cho phép upload checkpoint và chạy phiên 3”** cho gói/manifest/đích nêu trên; sau đó upload version mới được thực hiện.
- Không sửa code trong phiên này nên không chạy lại unit suite. Cổng 179/179 từ phiên triển khai vẫn là bằng chứng trước đó; kiểm tra hiện tại là hash 19 generation, restore payload thật, bootstrap CLI nhúng, source/profile và tài liệu.
- Sau upload, checkpoint dataset xác nhận private/ready, có 40 file dữ liệu upload với tên/kích thước khớp inventory. Manifest checkpoint và source tải lại từ Kaggle khớp; source vẫn 56 file/bundle đã khóa. Bộ kernel đã gửi có đủ ba dataset đầu vào.

## Chuẩn bị cho biên epoch đầu

Checklist đã bổ sung trong `docs/guides/b0_full_training.md`: từ input1.536 còn450 bước tới1.986, phải đánh giá đủ4.115 development/commit best trước batch epoch2; kiểm tra history, best/latest, threshold và pending evaluation. Code đã khóa được rà lại, chưa có evaluation thực tế hoặc metric mới trong phiên chuẩn bị này.

## Bàn giao

Version 16 RUNNING. Theo dõi thủ công mỗi 20–30 phút:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py status
```

Khi COMPLETE:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-3-v16
```

Đối chiếu workflow complete/cả3 mã0, bootstrap `restored_global_step=1024`/`files_verified=39` và report training dự kiến `paused`, `steps_this_invocation=512`, `controller.global_step=1536`, epoch0/cursor1.536. Giữ toàn bộ checkpoint/generation để kiểm chứng rồi tiếp tục phiên sau. Nếu ERROR/CANCELLED, giữ log/output để xác định generation cuối đã commit trước khi chạy lại.

JSON đi kèm lưu inventory, checksum và kết quả kiểm tra; artifact lớn nằm ngoài Git. Điểm đã kiểm chứng gần nhất vẫn1.024; mốc mục tiêu1.536 chưa hết epoch đầu nên dự kiến chưa có development/best/threshold. Không gửi thêm kernel trong lúc v16 còn chạy.
