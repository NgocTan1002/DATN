# Ảnh hưởng của LFCC phụ thuộc batch lên B0 (D027)

- Checkpoint: `g-4174b60757a44324a9b864a4931e7a18` (SHA-256 `050bbfff587bb48dc513e74065f0c395b5c1b5f5a1d2ddde8ae28ebdc37776d7`).
- Nguồn score: `reports/b0_peak_full_session_10.json`; 4115 mẫu development cũ (manifest v2), batch 8.
- Tái dựng chấm theo batch so với score lưu: lệch tối đa 1.138e-05.

| | Chấm theo batch (cũ) | Chấm riêng từng mẫu (đã sửa) |
|---|---:|---:|
| EER | 8.3333% | 8.3990% |
| EER adversarial_attack | 3.346% | 3.740% |
| EER replay | 7.692% | 7.692% |
| EER voice_conversion | 11.947% | 12.139% |

- Score đổi hơn 0.0001: 896 ({'bonafide': 233, 'replay': 7, 'voice_conversion': 656}); lớn nhất 0.9461, trung vị 1.19e-07.
- Quyết định đảo tại threshold 0.007008: 31 ({'bonafide': 20, 'voice_conversion': 11}).
- Hiệu EER (riêng từng mẫu − theo batch): +0.066 điểm, CI 95% theo speaker [-0.436; +1.634], 999/1000 lần lấy mẫu hợp lệ, seed 2026.

## Giới hạn

- Chỉ đo trên một checkpoint và dev cũ (31 speaker, VC từ 3 speaker); không thay thế đánh giá B0 v2 trên dev-full.
- Mô hình được huấn luyện với đặc trưng cũ, nên số liệu cho thấy độ nhạy của score với cách ghép batch, không phải hiệu năng của mô hình sửa lỗi.
