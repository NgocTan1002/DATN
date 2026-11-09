# B0 v2 — tính lại metric từ score dev-full

- Đạt: True; 3 seed × 3 epoch, mỗi lượt 38.777 mẫu/186 shard dev-full (train 15.885/57 shard).
- Sai lệch tuyệt đối lớn nhất EER/threshold/ROC-AUC/macro-F1 so với số lưu: 0.0; dung sai 1e-12.
- Đối chiếu thêm EER/threshold/AUC bằng hàm numpy và macro-F1 từ confusion matrix; kiểm metadata/thứ tự với manifest, score/label giữa records và mảng lưu, history và best report/generation.
- Không đọc/chấm test, không nạp mô hình hoặc audio. Report/generation/manifest gốc giữ nguyên (SHA-256 trước/sau).
- Loss không được tuyên bố kiểm lại: chỉ có sigmoid probability, không lưu logits; sigmoid float32 có thể bão hòa 0/1. Số loss gốc và số score 0/1 nằm trong JSON.

| Seed | Epoch | Step | EER | ROC-AUC | Macro-F1 | Threshold | Lệch lớn nhất |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 1986 | 0.13914211009823 | 0.93583715012442 | 0.85289079735568 | 0.26372757014491 | 0.0 |
| 2026 | 2 | 3972 | 0.10811956752173 | 0.95368177657206 | 0.88520619918727 | 0.92676539266415 | 0.0 |
| 2026 | 3 | 5958 | 0.082255670977316 | 0.97302416247981 | 0.91234792976839 | 0.009960237331333 | 0.0 |
| 2027 | 1 | 1986 | 0.13030881209808 | 0.94274158957399 | 0.86207841391849 | 0.76963196720439 | 0.0 |
| 2027 | 2 | 3972 | 0.10671647851864 | 0.95462388980732 | 0.88668952217208 | 0.045099299724168 | 0.0 |
| 2027 | 3 | 5958 | 0.08263623812231 | 0.97328365612226 | 0.91196804602429 | 0.14591552048539 | 0.0 |
| 2028 | 1 | 1986 | 0.13956793632746 | 0.93006147745841 | 0.85246207601072 | 0.241952650842 | 0.0 |
| 2028 | 2 | 3972 | 0.13343620563632 | 0.9532514106604 | 0.85883535625125 | 0.12305466297396 | 0.0 |
| 2028 | 3 | 5958 | 0.079287682849269 | 0.9762978273407 | 0.91547204471722 | 0.00025503549158832 | 0.0 |

Lệnh tái lập (UTF-8):

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe .cache/recheck_b0_v2_metrics.py
```

Log: `logs/b0_v2_metric_recheck.log`. JSON đi kèm ghi đầy đủ checksum script, metric source, manifest, report và state đầu vào; generation, confusion matrix, sai lệch từng metric và kiểm chéo.
