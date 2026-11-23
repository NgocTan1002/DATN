# SCALE2-01 — Kế hoạch shard và khử trùng cho train ×2 (`development-scale2-v1`, D036)

- Thời điểm tạo (UTC): `2026-11-18T06:01:49.757785+00:00`
- Nguồn snapshot: `hustep-lab/VSASV-Dataset` revision `92de668616780ed4f3d7f58a82262f82b56b4557` (seed `2026`).
- Chính sách chống rò rỉ test: **Không đọc audio dev/test; không dùng bảng split của closed_test hay utt_type/speaker_id của closed_test; thứ tự chọn shard hoàn toàn chỉ phụ thuộc SHA256('2026|shard').**
- Kiểm 68 shard cục bộ vs `vsasv_shard_tree.json` & `vsasv_metadata.csv`: **68/68 shard khớp 100%** (10,580,529,098 B).

## 1. Quota D036, ứng viên cục bộ sạch và thiếu hụt còn lại

| `utt_type` | Quota tổng (×2) | Đã có (×1) | Phần thêm cần | Ứng viên cục bộ thô | Trùng `train` | Trùng `dev` | Trùng `test` | Trùng nội bộ | Sạch cục bộ | Còn thiếu sau cục bộ | Tỷ lệ loại cục bộ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bonafide` | 14,267 | 7,134 | 7,133 | 3,428 | 43 | 31 | 0 | 7 | 3,347 | 3,786 | 2.36% |
| `voice_conversion` | 8,697 | 4,348 | 4,349 | 2,825 | 0 | 0 | 0 | 0 | 2,825 | 1,524 | 0.00% |
| `adversarial_attack` | 8,697 | 4,348 | 4,349 | 539 | 0 | 0 | 0 | 0 | 539 | 3,810 | 0.00% |
| `replay` | 109 | 55 | 54 | 90 | 0 | 0 | 0 | 0 | 90 | 0 | 0.00% |
| **Tổng** | **31,770** | **15,885** | **15,885** | **6,882** | **43** | **31** | **0** | **7** | **6,801** | **9,120** | **1.18%** |

- Ghi chú `replay`: pool cục bộ có `90` mẫu sạch, dư `36` mẫu so với phần thêm cần `54` (đã đủ, thiếu hụt = `0`).
- Lưu ý tập hash loại trùng: Loại trùng theo content_sha256 của MỌI dòng closed_dev và closed_test trong ledger (gồm cả kept lẫn internal_duplicate/overlap, vì mọi dòng đều là nội dung thuộc dev/test gốc); không đọc hay dùng cột utt_type/speaker_id của closed_test.
- Số nhóm content hash trùng hoàn toàn trong `closed_train` cục bộ (nội bộ ứng viên): `7` nhóm.

## 2. Tóm tắt kế hoạch tải shard từ xa vs trần D036 (120 shard / 10 GB)

| Mức kế hoạch | Mục tiêu `bonafide` / `VC` / `AP` / `replay` | Số shard cần tải | Tổng dung lượng (byte) | GB (10⁹ B) | GiB (2³⁰ B) | Vượt trần 120 shard / 10 GB? |
|---|---|---:|---:|---:|---:|---|
| Cơ sở (chưa khử trùng shard mới) | 3,786 / 1,524 / 3,810 / 0 | 35 | 6,753,103,437 | 6.753 GB | 6.289 GiB | **Đạt (không vượt)** |
| Dự phòng 1 (*ước tính theo tỷ lệ trùng từng `utt_type` bước 4*) | 3,878 / 1,524 / 3,810 / 0 | 35 | 6,753,103,437 | 6.753 GB | 6.289 GiB | **Đạt (không vượt)** |
| Dự phòng 2 (*ước tính áp dụng tỷ lệ trùng chung 1.18% bước 4*) | 3,832 / 1,543 / 3,856 / 0 | 35 | 6,753,103,437 | 6.753 GB | 6.289 GiB | **Đạt (không vượt)** |

## 3. Danh sách shard cần tải theo thứ tự `SHA256("2026|shard")` tăng dần (kế hoạch cơ sở)

| # | Tên shard | Kích thước (byte) | SHA-256 (`vsasv_shard_tree.json`) | Dòng `closed_train` mới (`bonafide` / `VC` / `AP` / `replay`) | Cộng dồn (`bonafide` / `VC` / `AP` / `replay`) |
|---:|---|---:|---|---|---|
| 1 | `train-00204-of-00432.parquet` | 144,701,034 | `5d5073c80591143972cea80d7fd4085fd15f2fe16fed9e19a2d98db80d26deec` | 0 / 512 / 0 / 0 | 0 / 512 / 0 / 0 |
| 2 | `train-00200-of-00432.parquet` | 129,819,988 | `2973529c51b97be4d75b8f9d15ba8274814091350b3a104f9be39be689864e4d` | 9 / 0 / 0 / 0 | 9 / 512 / 0 / 0 |
| 3 | `train-00115-of-00432.parquet` | 122,751,249 | `a86a6025b2288ed6544ea1db25ac1428e80e9f7cc21829faf0638654650b87a4` | 386 / 0 / 0 / 0 | 395 / 512 / 0 / 0 |
| 4 | `train-00393-of-00432.parquet` | 39,306,423 | `fa94a6602a48a9ae47e78f0bb25cfd0586868eaf434070a6c272016554d56e42` | 511 / 0 / 0 / 0 | 906 / 512 / 0 / 0 |
| 5 | `train-00402-of-00432.parquet` | 181,099,873 | `20c5aab6e23294f260085e4114212bfac20304c61350662620a4679ec4b3e10b` | 446 / 0 / 0 / 15 | 1,352 / 512 / 0 / 15 |
| 6 | `train-00325-of-00432.parquet` | 145,604,183 | `87d10f2d592ab9cbbc60143c791b77ef94ae766ee1d7b1a6b9291846abeb52cc` | 0 / 511 / 0 / 0 | 1,352 / 1,023 / 0 / 15 |
| 7 | `train-00117-of-00432.parquet` | 222,136,829 | `a9ac0bc681eca7b53b5fbee1a83af3aa226b6e75072ebcc74ded29fd53ce88eb` | 506 / 0 / 0 / 0 | 1,858 / 1,023 / 0 / 15 |
| 8 | `train-00327-of-00432.parquet` | 117,536,081 | `a8d5d6258304869b2222f7e9c8b4852e2632eb224d847b965fd5f196681669d0` | 0 / 511 / 0 / 0 | 1,858 / 1,534 / 0 / 15 |
| 9 | `train-00191-of-00432.parquet` | 141,903,880 | `2a6105f517c44c68c3dc9467b40fa4aa477ad1eabc10872019168bb34f676817` | 0 / 0 / 0 / 0 | 1,858 / 1,534 / 0 / 15 |
| 10 | `train-00114-of-00432.parquet` | 138,682,063 | `9960e72fd0bae090c37b56232be1064af7d0fc6702d6041d93fd7f8b357cbaf7` | 474 / 0 / 0 / 0 | 2,332 / 1,534 / 0 / 15 |
| 11 | `train-00145-of-00432.parquet` | 175,607,729 | `93eb8e73ec6bb821341ce64f45041e207a8c2093be6d4175f462de65ee770b7b` | 405 / 0 / 0 / 0 | 2,737 / 1,534 / 0 / 15 |
| 12 | `train-00172-of-00432.parquet` | 165,364,857 | `6f6f475dc43445360924a603b8c650e6c1f0a4292ed89e72f21d14c996f95cc3` | 5 / 156 / 154 / 0 | 2,742 / 1,690 / 154 / 15 |
| 13 | `train-00429-of-00432.parquet` | 897,748,804 | `a28055ecde39801c8a662eb30e5d0001721c502eba32be76469d82f4d7f6f749` | 0 / 234 / 123 / 0 | 2,742 / 1,924 / 277 / 15 |
| 14 | `train-00281-of-00432.parquet` | 162,473,779 | `450173725060c04d40a784abfaa1d3a6f7051152b40f34f203ec47ed4d3fba9d` | 0 / 0 / 0 / 0 | 2,742 / 1,924 / 277 / 15 |
| 15 | `train-00053-of-00432.parquet` | 96,428,273 | `09f75f1fb797c243940e8d775aa613d13592b01411d3a30971a693764c29a0b7` | 179 / 0 / 0 / 0 | 2,921 / 1,924 / 277 / 15 |
| 16 | `train-00267-of-00432.parquet` | 228,418,310 | `b75c55a3b6c67e1d476d4bf12c4e0d5a39e1e487eafedfe434302861d68ace18` | 11 / 0 / 80 / 0 | 2,932 / 1,924 / 357 / 15 |
| 17 | `train-00039-of-00432.parquet` | 107,613,117 | `9bbe6604dd02a90ba9a8c3f4d1561814e59771df055ffa9f94cff4b83fca319f` | 375 / 0 / 0 / 0 | 3,307 / 1,924 / 357 / 15 |
| 18 | `train-00329-of-00432.parquet` | 133,742,230 | `ee7b9dce6217adf7f68858d27120d3f5f975c52a4a9c42c7923de0f17484a648` | 0 / 0 / 511 / 0 | 3,307 / 1,924 / 868 / 15 |
| 19 | `train-00302-of-00432.parquet` | 182,892,480 | `16c5740f63d180e40c746e72c979d8b7d8fcda893564904680b3359ad66ca7d9` | 6 / 82 / 82 / 0 | 3,313 / 2,006 / 950 / 15 |
| 20 | `train-00363-of-00432.parquet` | 164,917,600 | `09e031f81406047ce84175af1de9c6b3ee588a0600c9432057796c9438753133` | 0 / 0 / 511 / 0 | 3,313 / 2,006 / 1,461 / 15 |
| 21 | `train-00243-of-00432.parquet` | 158,779,058 | `03d7527337ea1b1df48e53f140caf7baa630195526b64b70223f22347123b4d9` | 0 / 0 / 0 / 0 | 3,313 / 2,006 / 1,461 / 15 |
| 22 | `train-00104-of-00432.parquet` | 153,434,517 | `7192bcbb6b0022c51155493f250c36a2e6883e59100f817ea202dc12ab9cdb71` | 278 / 0 / 0 / 0 | 3,591 / 2,006 / 1,461 / 15 |
| 23 | `train-00012-of-00432.parquet` | 90,955,955 | `27f201d80b586dde4af0e092eb7b3e59cf3a41141c16765c0c32c4c6ca0f2aa4` | 512 / 0 / 0 / 0 | 4,103 / 2,006 / 1,461 / 15 |
| 24 | `train-00368-of-00432.parquet` | 141,044,232 | `6d8453f721d9f55b12d924a574df52eaa64f28fa23c2c73e5f699d8f919ebe92` | 0 / 0 / 511 / 0 | 4,103 / 2,006 / 1,972 / 15 |
| 25 | `train-00410-of-00432.parquet` | 696,966,177 | `9b394604d254a0b508b332a0b8f313f610ee2a068ffeaf0772babdbdb84ac369` | 0 / 222 / 275 / 0 | 4,103 / 2,228 / 2,247 / 15 |
| 26 | `train-00158-of-00432.parquet` | 227,860,499 | `97686ebbfaf327c49c9de130b896c8bc4422ec2b68447f1328642a900331f3f1` | 371 / 0 / 0 / 0 | 4,474 / 2,228 / 2,247 / 15 |
| 27 | `train-00387-of-00432.parquet` | 125,697,135 | `f52b760cdcd4dddce4978e4f10debce92ab1f09869030c118a9489dd90e8ad08` | 138 / 0 / 373 / 0 | 4,612 / 2,228 / 2,620 / 15 |
| 28 | `train-00077-of-00432.parquet` | 134,186,202 | `76f9dde9aa9fc5aeed1ad177bb6d2b5928d2aa6578eca836923ea470445355a2` | 377 / 0 / 0 / 0 | 4,989 / 2,228 / 2,620 / 15 |
| 29 | `train-00178-of-00432.parquet` | 162,511,188 | `0fdc45d866db665828a6dee37283ef7aed2b6a821115178f21bdc5c583eb5403` | 0 / 454 / 58 / 0 | 4,989 / 2,682 / 2,678 / 15 |
| 30 | `train-00150-of-00432.parquet` | 204,352,196 | `201d918b092f05fef2335a9f749f95312899434aeb46c37c5434f221d0683c43` | 495 / 0 / 0 / 0 | 5,484 / 2,682 / 2,678 / 15 |
| 31 | `train-00057-of-00432.parquet` | 78,912,995 | `3cb8e53ffb99061509985d751b676664636d652bb98f76d5af57908e4b400003` | 512 / 0 / 0 / 0 | 5,996 / 2,682 / 2,678 / 15 |
| 32 | `train-00181-of-00432.parquet` | 180,326,974 | `1d6229580289d0ed38a87d5192173e875d7483b0826d62cee3e556e3dc8ffb48` | 0 / 0 / 512 / 0 | 5,996 / 2,682 / 3,190 / 15 |
| 33 | `train-00356-of-00432.parquet` | 151,498,074 | `c2b7603232f1e152cbd30676a96c5608ea59ead66eeb5589ee94d1ae4a825556` | 0 / 511 / 0 / 0 | 5,996 / 3,193 / 3,190 / 15 |
| 34 | `train-00362-of-00432.parquet` | 131,037,783 | `acc746c0b7d2beef7b2cb1f23f211e4a25aa4a4c4c4646528754099f1e87717a` | 0 / 0 / 511 / 0 | 5,996 / 3,193 / 3,701 / 15 |
| 35 | `train-00408-of-00432.parquet` | 416,791,670 | `ad58ff225ace793310eb9f46511c0b48ba3b1489dbf04df1f1675719c6fae9f9` | 0 / 139 / 163 / 0 | 5,996 / 3,332 / 3,864 / 15 |

- Với cả hai mức dự phòng (*ước tính theo tỷ lệ trùng ở bước 4*), danh sách shard dừng tại đúng cùng shard thứ `35` (`train-00408-of-00432.parquet`) mà không cần thêm shard nào ngoài bảng trên.

## 4. Nguồn hash và phạm vi chỉ đọc

- Train hiện tại: `reports/development_manifest_20k_v2_content_audit.json` (`15,885` dòng `closed_train`, SHA-256 file `25aec6dd922ec532ae3980f35d2ed232ca26f12dfada163ebb9787b32356d0df`).
- Dev-full & Test-full: `data/processed/eval_chunks/content_hash_ledger.csv` (`69,964` dòng, SHA-256 `6676ae932d80e3d8a4768ec21b4f339d1e51fd37319a7d1a088f6ef0160ea65a`, khớp `ledger_sha256` trong `reports/closed_eval_full_v1.json`).
- Không gọi mạng, không tải shard, không đọc audio `closed_dev` hay `closed_test`.
