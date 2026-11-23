# Báo cáo X1S-04B — Tạo 2 dataset resume private và push kernel S2 (X1 scale2)

- Trạng thái: **ĐẠT**
- Lưu ý: **Dataset seed 2027 đã sẵn sàng cho S3 nhưng S3 chưa push.**

## 1. Hai dataset resume private trên Kaggle
### Seed 2026 (`ngctnnguyn1111/doanttnt-xlsr-scale2-s2026-resume1`)
- `datasetId`: `13534901`
- `isPrivate`: `True`
- `manifest_sha256`: `9012282f20cd4dbe818f845f1cd02b133fee5cf0297fed2b40276675e0d8ccbd`
- Tập file từ xa (6 file, khớp local và tải về khớp SHA-256):
  - `LATEST.json`: `538` bytes
  - `b0_resume_manifest.json`: `2057` bytes
  - `g-6d3cd386b2514c7292a0e976f48ad28e__checkpoint.pt`: `1240034267` bytes
  - `g-6d3cd386b2514c7292a0e976f48ad28e__state.json`: `52024959` bytes
  - `g-f0deb62f783f4f5095a2f44a0702dfe2__checkpoint.pt`: `1245609947` bytes
  - `g-f0deb62f783f4f5095a2f44a0702dfe2__state.json`: `62615055` bytes
### Seed 2027 (`ngctnnguyn1111/doanttnt-xlsr-scale2-s2027-resume1`)
- `datasetId`: `13535177`
- `isPrivate`: `True`
- `manifest_sha256`: `86429f1520d02e7f14931be39e32c8340be34272a4cf7886de9774c771406ec3`
- Tập file từ xa (6 file, khớp local và tải về khớp SHA-256):
  - `LATEST.json`: `538` bytes
  - `b0_resume_manifest.json`: `2057` bytes
  - `g-2f64908aa31b4ea09320fa58bfec0277__checkpoint.pt`: `1250983067` bytes
  - `g-2f64908aa31b4ea09320fa58bfec0277__state.json`: `75253548` bytes
  - `g-b19748034808412f8a006986895313e6__checkpoint.pt`: `1251061211` bytes
  - `g-b19748034808412f8a006986895313e6__state.json`: `75544432` bytes

## 2. Push kernel S2
- Kernel: `ngctnnguyn1111/doanttnt-xlsr-d028`
- Version S2: **`18`**
- Thời điểm push (UTC+7): `2026-11-21T11:24:05.876081+07:00`
- Trạng thái sau push: `ngctnnguyn1111/doanttnt-xlsr-d028 has status "KernelWorkerStatus.RUNNING"`

## 3. Quota Kaggle GPU
- Trước khi tạo dataset (preflight): `45.0h`
- Trước khi push S2: `45.0h`
- Sau push 5–10 phút: `used=0.10h`, `remaining=44.90h`, `refreshAt=2026-11-28T00:00:00`
