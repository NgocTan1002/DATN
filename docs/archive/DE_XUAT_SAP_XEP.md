# Đề xuất sắp xếp tài liệu hiện có — đã thực hiện

> Người dùng đã duyệt phương án và việc sắp xếp được thực hiện ngày 08/10/2026. File này được giữ trong `docs/archive/` làm dấu vết đối chiếu; không dùng làm trạng thái hiện hành.

## Phạm vi kiểm kê

Kiểm kê ngày 08/10/2026 ghi nhận 21 file Markdown trực tiếp trong `docs/`: 14 file được đề xuất gộp hoặc chuyển sang cấu trúc mới và 7 file giữ nguyên vị trí. `docs/current_state.md` đang nằm đúng vị trí mà `AGENTS.md` quy định.

Kết quả thực hiện: bảy file ngày đã được tạo trong `docs/plans/daily/`, hai file tuần trong `docs/plans/weekly/`, hai hướng dẫn được chuyển nguyên nội dung sang `docs/guides/` và 12 file nguồn được lưu trong `docs/archive/`. Các thay đổi nội dung của hai hướng dẫn chỉ là sửa đường dẫn liên kết theo vị trí mới.

## Bảng đề xuất

| File hiện tại | Loại (kế hoạch ngày / tổng kết ngày / kế hoạch tuần / tổng kết tuần / hướng dẫn / kế hoạch commit / tài liệu lịch sử / giữ nguyên) | Đề xuất đích | Lý do |
|---|---|---|---|
| `docs/audio_protocol.md` | giữ nguyên | Giữ tại `docs/audio_protocol.md` | Giao thức âm thanh đang được tham chiếu trực tiếp; thuộc nhóm được yêu cầu giữ nguyên. |
| `docs/cai_dat_moi_truong.md` | giữ nguyên | Giữ tại `docs/cai_dat_moi_truong.md` | Hướng dẫn nền tảng đã có đường dẫn ổn định và được yêu cầu giữ nguyên. |
| `docs/chuan_bi_du_lieu_tuan_02_2026-10-04.md` | hướng dẫn | `docs/guides/chuan_bi_du_lieu_vsasv.md` | Nội dung chuẩn hóa kiểm kê, đo băng thông và chuẩn bị manifest có thể dùng lại; tên đích mô tả nội dung thay vì ngày. |
| `docs/current_state.md` | giữ nguyên | Giữ tại `docs/current_state.md` | Đây là nguồn trạng thái hiện hành duy nhất theo `AGENTS.md`. |
| `docs/decisions.md` | giữ nguyên | Giữ tại `docs/decisions.md` | Nhật ký quyết định dài hạn được yêu cầu giữ nguyên. |
| `docs/huong_dan_hoan_thanh_cong_viec_tu_2026-10-05.md` | hướng dẫn | `docs/guides/quy_trinh_tai_va_xac_minh_du_lieu_vsasv.md` | Nội dung là quy trình tải, xác minh shard và tạo manifest; tên mới phản ánh công dụng dùng lại. |
| `docs/ke_hoach_commit_2026-10-02.md` | kế hoạch commit | `docs/plans/daily/2026-10-02.md`, trong phần kết quả hoặc bàn giao Git | Không giữ kế hoạch commit thành file riêng; gộp vào file ngày tương ứng. |
| `docs/ke_hoach_cong_viec_2026-10-05_den_2026-10-07.md` | kế hoạch ngày | `docs/plans/daily/2026-10-05.md`, `2026-10-06.md`, `2026-10-07.md` theo từng mục Thứ Hai/Thứ Ba/Thứ Tư | File đang gộp ba ngày; tách các mục theo ngày để tuân thủ một file cho mỗi ngày. Cần người dùng duyệt cách xử lý phần mục tiêu và trạng thái dùng chung trước khi thực hiện. |
| `docs/ke_hoach_ngay_2026-09-30.md` | kế hoạch ngày | `docs/plans/daily/2026-09-30.md`, phần **Kế hoạch** | Ghép với tổng kết cùng ngày trong một file. |
| `docs/ke_hoach_ngay_2026-10-01.md` | kế hoạch ngày | `docs/plans/daily/2026-10-01.md`, phần **Kế hoạch** | Ghép với tổng kết cùng ngày trong một file. |
| `docs/ke_hoach_ngay_2026-10-02.md` | kế hoạch ngày | `docs/plans/daily/2026-10-02.md`, phần **Kế hoạch** | Ghép với tổng kết và kế hoạch commit cùng ngày. |
| `docs/ke_hoach_ngay_2026-10-04.md` | kế hoạch ngày | `docs/plans/daily/2026-10-04.md` | File đã chứa kế hoạch và kết quả chốt ngày; chuẩn hóa tên và vị trí, không tạo tổng kết riêng. |
| `docs/ke_hoach_tuan_01_2026-09-28_2026-10-04.md` | kế hoạch tuần | `docs/plans/weekly/2026-W40.md`, phần **Kế hoạch** | Tuần 28/09–04/10/2026 là ISO week 40; ghép với tổng kết tuần 01. |
| `docs/ke_hoach_tuan_02_2026-10-05_2026-10-11.md` | kế hoạch tuần | `docs/plans/weekly/2026-W41.md` | Tuần 05/10–11/10/2026 là ISO week 41; cuối tuần điền kết quả vào cùng file. |
| `docs/kien_thuc_can_nam_vung_de_bao_ve_do_an.md` | tài liệu lịch sử | Giữ tại `docs/kien_thuc_can_nam_vung_de_bao_ve_do_an.md` | Tài liệu nền phục vụ bảo vệ đồ án, được yêu cầu giữ nguyên vị trí. |
| `docs/split_protocol.md` | giữ nguyên | Giữ tại `docs/split_protocol.md` | Giao thức split đang được tham chiếu trực tiếp; thuộc nhóm được yêu cầu giữ nguyên. |
| `docs/tong_ket_cong_viec_2026-09-30.md` | tổng kết ngày | `docs/plans/daily/2026-09-30.md`, phần **Kết quả** | Ghép với kế hoạch ngày 30/09; không giữ file tổng kết ngày riêng. |
| `docs/tong_ket_cong_viec_2026-10-01.md` | tổng kết ngày | `docs/plans/daily/2026-10-01.md`, phần **Kết quả** | Ghép với kế hoạch ngày 01/10; không giữ file tổng kết ngày riêng. |
| `docs/tong_ket_cong_viec_2026-10-02.md` | tổng kết ngày | `docs/plans/daily/2026-10-02.md`, phần **Kết quả** | Ghép với kế hoạch và nội dung bàn giao Git ngày 02/10. |
| `docs/tong_ket_tuan_01_2026-09-28_2026-10-04.md` | tổng kết tuần | `docs/plans/weekly/2026-W40.md`, phần **Kết quả** | Ghép với kế hoạch tuần 01 trong một file tuần. |
| `docs/tong_quan_cach_thuc_hoat_dong_do_an.md` | tài liệu lịch sử | Giữ tại `docs/tong_quan_cach_thuc_hoat_dong_do_an.md` | Tài liệu tổng quan nền, được yêu cầu giữ nguyên vị trí. |

Sau khi người dùng duyệt và nội dung đã được gộp đúng, các file nguồn cũ có thể được đề xuất chuyển vào `docs/archive/`; không di chuyển hoặc xóa tự động.

## Liên kết phải cập nhật nếu thực hiện đề xuất

Phép kiểm tra tìm chính xác tên của 14 file được đề xuất chuyển/gộp trong tất cả file `.md` có sẵn của repository trước nhiệm vụ này, không tính chính file đích đang được tìm và không tính file đề xuất tạm này. Có 12 file Markdown khác chứa tổng cộng 34 lần tham chiếu cần xem xét:

| File chứa liên kết | Số liên kết cần sửa | File đích hiện tại được tham chiếu |
|---|---:|---|
| `docs/current_state.md` | 3 | `huong_dan_hoan_thanh_cong_viec_tu_2026-10-05.md` (1), `ke_hoach_cong_viec_2026-10-05_den_2026-10-07.md` (1), `ke_hoach_tuan_02_2026-10-05_2026-10-11.md` (1) |
| `docs/huong_dan_hoan_thanh_cong_viec_tu_2026-10-05.md` | 2 | `chuan_bi_du_lieu_tuan_02_2026-10-04.md` (1), `ke_hoach_cong_viec_2026-10-05_den_2026-10-07.md` (1) |
| `docs/ke_hoach_commit_2026-10-02.md` | 7 | `ke_hoach_ngay_2026-10-01.md` (1), `ke_hoach_ngay_2026-10-02.md` (1), `ke_hoach_tuan_01_2026-09-28_2026-10-04.md` (1), `ke_hoach_tuan_02_2026-10-05_2026-10-11.md` (1), `tong_ket_cong_viec_2026-09-30.md` (1), `tong_ket_cong_viec_2026-10-01.md` (1), `tong_ket_cong_viec_2026-10-02.md` (1) |
| `docs/ke_hoach_cong_viec_2026-10-05_den_2026-10-07.md` | 2 | `chuan_bi_du_lieu_tuan_02_2026-10-04.md` (1), `ke_hoach_tuan_02_2026-10-05_2026-10-11.md` (1) |
| `docs/ke_hoach_ngay_2026-10-01.md` | 1 | `tong_ket_cong_viec_2026-09-30.md` (1) |
| `docs/ke_hoach_ngay_2026-10-02.md` | 3 | `ke_hoach_commit_2026-10-02.md` (1), `ke_hoach_tuan_02_2026-10-05_2026-10-11.md` (1), `tong_ket_cong_viec_2026-10-02.md` (1) |
| `docs/ke_hoach_ngay_2026-10-04.md` | 1 | `tong_ket_tuan_01_2026-09-28_2026-10-04.md` (1) |
| `docs/ke_hoach_tuan_02_2026-10-05_2026-10-11.md` | 3 | `chuan_bi_du_lieu_tuan_02_2026-10-04.md` (2), `tong_ket_tuan_01_2026-09-28_2026-10-04.md` (1) |
| `docs/tong_ket_cong_viec_2026-09-30.md` | 2 | `ke_hoach_ngay_2026-09-30.md` (1), `ke_hoach_tuan_01_2026-09-28_2026-10-04.md` (1) |
| `docs/tong_ket_cong_viec_2026-10-01.md` | 2 | `ke_hoach_ngay_2026-10-01.md` (1), `ke_hoach_tuan_01_2026-09-28_2026-10-04.md` (1) |
| `docs/tong_ket_cong_viec_2026-10-02.md` | 6 | `ke_hoach_commit_2026-10-02.md` (3), `ke_hoach_ngay_2026-10-02.md` (1), `ke_hoach_tuan_02_2026-10-05_2026-10-11.md` (2) |
| `docs/tong_ket_tuan_01_2026-09-28_2026-10-04.md` | 2 | `chuan_bi_du_lieu_tuan_02_2026-10-04.md` (1), `ke_hoach_tuan_02_2026-10-05_2026-10-11.md` (1) |

Không tìm thấy tham chiếu trong các file Markdown khác tới `docs/ke_hoach_ngay_2026-10-04.md`. Các tham chiếu trong bảng được đếm theo số lần tên file xuất hiện; khi thực hiện sắp xếp cần rà lại liên kết tương đối theo đường dẫn mới.

Sau khi thực hiện, phép tìm lại không còn thấy tên file cũ bên ngoài file đề xuất này; 34 lượt tham chiếu đã được chuyển sang đường dẫn mới. Các file nguồn trong `docs/archive/` cũng được sửa liên kết để tiếp tục điều hướng được.

## Quyết định đã áp dụng

1. Đã gộp ba cặp kế hoạch/tổng kết ngày 30/09, 01/10 và 02/10, cùng cặp kế hoạch/tổng kết tuần 01.
2. Đã đặt phần mục tiêu và trạng thái dùng chung vào ngày 05/10; file ngày 06/10 và 07/10 liên kết về file này.
3. Đã chuyển nguyên nội dung hai tài liệu hướng dẫn; chỉ cập nhật liên kết theo đường dẫn mới.
4. Đã cập nhật liên kết trước khi chuyển 12 file nguồn cũ vào `docs/archive/`; không xóa file lịch sử.
