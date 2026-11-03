# Packed artifact train / dev-full / test-full (T07)

- Ngày tạo: 2026-11-02T21:23:09+0700
- Định dạng: `vsasv-packed-v1`, float32 sau resample 16 kHz, trước amplitude/crop.
- Nguồn: chunk từng shard (`src/data/eval_ingest.py`); train nạp từ Parquet cục bộ, dev/test từ `data/processed/eval_chunks/`.
- Chunk train mới nạp/đã có: {'ingested': 0, 'already_done': 57}.

| Artifact | Split | Manifest | Mẫu | Shard | GiB | Build (s) | Băm lại (s) |
|---|---|---|---:|---:|---:|---:|---:|
| train | closed_train | `development_20k_v2.csv` | 15.885 | 57 | 5,39 | 39,5 | 25,8 |
| dev-full | closed_dev | `closed_dev_full_v1.csv` | 38.777 | 186 | 12,53 | 102,4 | 62,5 |
| test-full | closed_test | `closed_test_full_v1.csv` | 30.698 | 162 | 11,40 | 161,4 | 63,3 |

## Checksum

- train: waveforms `e09517c8e367bfb4e24563f92d90d2f7ee0ad7d5c7a58dc78862db742ab6614d`, index `b71825930d9a344484ecbc78102876b1ee5a873e2b2c7d724906cf5766094332`, metadata `9ce34ceb9883a56347de4aedf8d1a39fa7e8eae753dc006919bf634d376250a9`, manifest `f549423b1fb33665f7e5606cdf56321bc8cd07246a0cdf0b90cd918c0c09e728`.
- dev-full: waveforms `579e5e85a020aeab70c8ad912023f5ee4efaef19f8e61af4fd2c923ee692f648`, index `364f1993332ec889dd7301357a1aa6145d14c65c56bfa3db334668b93f01b8d6`, metadata `5537af3384c6ffd83093a5e4175676ddbef1b5a5b5607c5daa87285e6de13f14`, manifest `a79f1bb2595bc46b08ce30b8652234b1858cc26e558e2ccd33632435df1a5360`.
- test-full: waveforms `556bb99de8c39ed8f8568ae36d170907f4ac2959b9c4ab2f1a8d4e48423b32b2`, index `8f862032b6af42e1d35ff7d9e862308afb781308d13f24178a4bb1409237c78a`, metadata `7e16da487f4d066f9694acd62b099ab4b70cc9158429b21c6e85c5fd1185ee9b`, manifest `0a7a70bc8ededdd83e5f6df06580fc954c6f2da848366e37bc265331ca17d266`.

## Giới hạn

- Băm lại chỉ chứng minh artifact khớp chunk; đối chiếu với Parquet nằm ở báo cáo parity (T08).
- Test-full chỉ được đóng gói và băm lại; chưa chạy mô hình hay metric trên test.
