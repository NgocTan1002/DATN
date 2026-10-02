# Kế hoạch gom commit — 02/10/2026

Các thay đổi được chia thành ba commit độc lập để có thể xem xét, hoàn nguyên và đối chiếu báo cáo với mã nguồn. Tài liệu này chỉ chuẩn bị phạm vi commit; chưa tự động stage hoặc commit.

## Commit 1 — Mô hình và cổng kỹ thuật B0

**Thông điệp đề xuất:** `feat(model): implement and validate LFCC-LCNN smoke baseline`

Tệp:

- `configs/lfcc_lcnn.json`
- `src/models/__init__.py`
- `src/models/lfcc_lcnn.py`
- `scripts/smoke_train_lfcc_lcnn.py`
- `tests/test_lcnn.py`
- `reports/b0_smoke_training.json`
- `reports/b0_smoke_training.md`

Ý nghĩa: tạo baseline LCNN tối thiểu có Max-Feature-Map, kiểm tra đầy đủ forward/backward/optimizer/checkpoint và lưu minh chứng smoke pilot.

Kiểm tra trước commit: chạy toàn bộ unit test; xác nhận checkpoint `.pt` không nằm trong danh sách stage.

## Commit 2 — Benchmark và dự báo tài nguyên

**Thông điệp đề xuất:** `perf(data): benchmark B0 loading and project runtime`

Tệp:

- `scripts/benchmark_data_pipeline.py`
- `scripts/benchmark_dataloader_options.py`
- `scripts/estimate_runtime.py`
- `reports/data_pipeline_benchmark.json`
- `reports/data_pipeline_benchmark.md`
- `reports/data_pipeline_optimization.json`
- `reports/data_pipeline_optimization.md`
- `reports/runtime_projection.json`
- `reports/runtime_projection.md`

Ý nghĩa: định lượng điểm nghẽn Parquet/DataLoader, so sánh batch size/worker trên cùng mẫu và chuyển throughput thành ngân sách 20.000, 40.000 và toàn bộ closed protocol.

Kiểm tra trước commit: chạy ba script với cấu hình mặc định; xác nhận JSON đọc được, Markdown khớp JSON và các kiểm tra tái lập đều đạt.

## Commit 3 — Tài liệu, quyết định và kế hoạch

**Thông điệp đề xuất:** `docs(project): record B0 progress and plan week 2`

Tệp:

- `README.md`
- `docs/decisions.md`
- `docs/ke_hoach_tuan_01_2026-09-28_2026-10-04.md`
- `docs/ke_hoach_tuan_02_2026-10-05_2026-10-11.md`
- `docs/ke_hoach_ngay_2026-10-01.md`
- `docs/ke_hoach_ngay_2026-10-02.md`
- `docs/ke_hoach_commit_2026-10-02.md`
- `docs/tong_ket_cong_viec_2026-09-30.md`
- `docs/tong_ket_cong_viec_2026-10-01.md`
- `docs/tong_ket_cong_viec_2026-10-02.md`
- `docs/kien_thuc_can_nam_vung_de_bao_ve_do_an.md`

Ý nghĩa: lưu lý do giữ cấu hình DataLoader, giới hạn diễn giải benchmark và backlog tuần 02 dựa trên số đo.

Kiểm tra trước commit: rà liên kết nội bộ, ngày tháng, số liệu lặp lại và `git diff --check`.

## Tệp không đưa vào commit

- `checkpoints/*.pt`
- `data/raw/**`
- `data/processed/**`
- `data/embeddings/**`
- log chạy cục bộ và file tạm

Trước mỗi commit, chỉ stage đúng nhóm tệp tương ứng rồi xem lại diff đã stage. Nếu một tệp chứa thay đổi không thuộc cùng mục đích, tách phần đó trước khi commit.
