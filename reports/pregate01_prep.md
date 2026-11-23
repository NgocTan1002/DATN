# PREGATE-01 — chuẩn bị trước cổng test

- Registry chính thức v1 (9 mục): SHA `4a4feef965b9ca50d8b609b9f9bdea7561dfc821427972cd0fd4320ef23a3c5c`; X1R (3 mục): SHA `d5e404ef6eaec0aa67324c6d4838455aefb9e5dfb9aee4bbbc7ff93bb3bd8d09`. Cả hai role official, entries trùng bản nháp (draft_sha256 ghi trong registry).
- Config: `test_gate_open=false` cả hai. D032 đổi `registry_sha256`, `t22_claude_verified`, `test_dataset_inventory_checked`; X1R chỉ đổi `registry_sha256`. protocol_sha256 không đổi.
- Cấp phép dev (`361615ca340619bec708f071ca3b7a32c67a02b925bee856de2170f2210d6329`): score qua cấp phép trùng tuyệt đối đường dev cũ trên 200 mẫu cho X1 seed 2026 và B0-v2 seed 2026; tái tạo D034 đạt; chạy lại bỏ qua 2/2 kết quả; sửa một ký tự SHA checkpoint bị từ chối trước khi mở dữ liệu; cấp phép test bị từ chối khi cổng đóng, `.cache/test_once*` giữ 0 file.
- Toàn suite: 545/545 (exit 0), `.cache/pregate01/suite.log`.
- Lưu ý: mô phỏng chạy CPU máy cục bộ (numpy 2.5.3), khác ghim Colab (numpy 2.1.3); chưa chạy Colab.
- Điều kiện D037 mục 6: (1) đạt; (2) đạt; (3) chưa; (4) chưa (PREGATE-02); (5) chưa.
- Hướng dẫn Colab: `docs/guides/colab_test_scoring.md`.
