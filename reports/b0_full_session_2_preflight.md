# Chuẩn bị và gửi B0 full training phiên 2 — Kaggle v15

- Preflight đạt; version 15 đã gửi thành công, API quan sát RUNNING. Mục tiêu tiếp tục 512 → 1.024, tối đa thêm 512 bước. Chưa xác nhận GPU resume hoặc bước mới bằng output v15.
- Điểm đã kiểm chứng gần nhất: v14 paused512. Giữ phạm vi manifest 20.000 mẫu/66 shard (15.885 train/4.115 development; pool 68 shard), policy peak và seed 2026 theo D023.
- Source bundle SHA-256: `68e07cfe4ac8884e21ea15a1f750ebb0aace03d4ac4a5dd42f3b0473030c92a1`; source tree `a46681120d36511e5d1bc7d5290e5488eecf136bf1ad55a93052afdb0e788925`. 56 file huấn luyện giữ nguyên; bootstrap chuyển checkpoint nằm riêng tại `deploy/kaggle/resume_checkpoint.py` và được nhúng trong lệnh kernel.
- Input latest SHA-256: `71464c4f3ebabfc6f3634c7963ced14d8056dc5ad056e6284ee1883fe46c6872`; manifest chuyển phiên SHA-256: `316d9c7c67298f70c2d0b5f9b11cecb4e9e15f711de5d28a904299e8695b23de`.
- Gói chuyển giữ 10 generation/21 file checkpoint, trạng thái và pointer, tổng 57.192.679 byte; thêm manifest thành 22 file trên Kaggle. Metadata tạo dataset được giữ cục bộ. Inventory remote/từng kích thước và manifest tải lại khớp.
- Dataset `ngctnnguyn1111/doantnnt-b0-full-resume`, ID 12838409, metadata xác nhận riêng tư và status ready. Kernel riêng tư, internet tắt, gắn data/source/checkpoint dataset.
- Bộ duyệt tự động ban đầu yêu cầu quyền xuất checkpoint tới dataset mới. Người dùng đã xác nhận cụ thể “Cho phép upload checkpoint và chạy tiếp”; sau đó upload và gửi phiên thành công.
- Local transfer trên checkpoint thật khớp toàn bộ payload trước/sau chuyển; bootstrap CLI nhúng cũng khôi phục đúng bước 512 và source identity. Xem `b0_full_session_2_transfer_check.json` và `b0_full_session_2_embedded_check.json`.
- 7/7 kiểm thử mới đạt, gồm đối chứng ngắt–chuyển–resume với chạy liên tục; toàn suite 179/179 đạt trong 239,697 giây. Lệnh: `.venv/Scripts/python.exe -m unittest discover -s tests -v`, offline, TEMP/TMP trong `.cache/d023-tests`; log `.cache/d023-tests/full-suite-2026-10-25.log`. Compileall hai file mới và diff --check đạt.
- Quota kiểm tra trước phiên: GPU còn 28 giờ; không cam kết đủ toàn bộ nhiều epoch. Không chạy test set hoặc thay protocol D023.

## Bàn giao

Theo dõi mỗi 20–30 phút bằng `.venv/Scripts/python.exe scripts/kaggle_workflow.py status` hoặc mở version 15 trên private kernel. Khi COMPLETE, tải bằng `.venv/Scripts/python.exe scripts/kaggle_workflow.py download-output --path kaggle_outputs/b0-full-session-2-v15`.

Đối chiếu workflow 3 return code 0; `b0_full_session_2_resume_input.json` phải khôi phục bước 512, `b0_peak_full_session_2.json` dự kiến paused/bước 1.024/thêm 512 bước. Kiểm tra checkpoint/hash/sampler/optimizer/RNG/controller trước khi tiếp tục phiên sau. Tại mốc 1.024 dự kiến chưa có evaluation/best/threshold vì chưa hết epoch đầu 1.986 bước.

Chi tiết vận hành: `docs/guides/b0_full_training.md`. JSON đi kèm lưu inventory/checksum, quyền xác nhận, quota và trạng thái quan sát. Dữ liệu/checkpoint/log lớn nằm ngoài Git; chưa có kết quả đầu ra v15 trong phiên bàn giao.
