# SCALE2-02 — Báo cáo tải shard mở rộng cho `development-scale2-v1` (D036)

- Thời điểm tạo (UTC): `2026-11-18T07:35:06.101319+00:00`
- Trạng thái: **ĐẠT**
- Nguồn: `hustep-lab/VSASV-Dataset` revision `92de668616780ed4f3d7f58a82262f82b56b4557`.
- Xác thực tải: lần đầu dừng vì `token=False`; lần này dùng token đã lưu cục bộ theo mặc định thư viện (không truyền tham số `token`, không ghi giá trị token).
- Kiểm đĩa trước tải: trống `89,456,533,504` B (`89.457 GB` / `83.313 GiB`, ngưỡng tối thiểu `30 GB` → `True`).
- Tổng shard đã tải/xác minh trong lần này: `35` (kế hoạch ban đầu `35`, tải bổ sung ngoài 35 shard: `0` = **False**).
- Tổng dung lượng shard tải mới: `6,753,103,437` B (`6.753 GB` / `6.289 GiB`).
- So với trần D036 (`120` shard / `10 GB`): **Đạt (không vượt trần 120 shard / 10 GB)**.

## Danh sách shard đã tải/xác minh theo thứ tự `SHA256("2026|shard")`

| # | Shard | Kích thước (B) | SHA-256 | Trạng thái | Lần thử | Thời gian (s) | Bổ sung ngoài 35? |
|---:|---|---:|---|---|---:|---:|---|
| 1 | `train-00204-of-00432.parquet` | 144,701,034 | `5d5073c80591143972cea80d7fd4085fd15f2fe16fed9e19a2d98db80d26deec` | `downloaded_verified` | 1 | 14.986 | Không |
| 2 | `train-00200-of-00432.parquet` | 129,819,988 | `2973529c51b97be4d75b8f9d15ba8274814091350b3a104f9be39be689864e4d` | `downloaded_verified` | 1 | 11.886 | Không |
| 3 | `train-00115-of-00432.parquet` | 122,751,249 | `a86a6025b2288ed6544ea1db25ac1428e80e9f7cc21829faf0638654650b87a4` | `downloaded_verified` | 1 | 12.335 | Không |
| 4 | `train-00393-of-00432.parquet` | 39,306,423 | `fa94a6602a48a9ae47e78f0bb25cfd0586868eaf434070a6c272016554d56e42` | `downloaded_verified` | 1 | 4.491 | Không |
| 5 | `train-00402-of-00432.parquet` | 181,099,873 | `20c5aab6e23294f260085e4114212bfac20304c61350662620a4679ec4b3e10b` | `downloaded_verified` | 1 | 13.808 | Không |
| 6 | `train-00325-of-00432.parquet` | 145,604,183 | `87d10f2d592ab9cbbc60143c791b77ef94ae766ee1d7b1a6b9291846abeb52cc` | `downloaded_verified` | 1 | 12.273 | Không |
| 7 | `train-00117-of-00432.parquet` | 222,136,829 | `a9ac0bc681eca7b53b5fbee1a83af3aa226b6e75072ebcc74ded29fd53ce88eb` | `downloaded_verified` | 1 | 17.632 | Không |
| 8 | `train-00327-of-00432.parquet` | 117,536,081 | `a8d5d6258304869b2222f7e9c8b4852e2632eb224d847b965fd5f196681669d0` | `downloaded_verified` | 1 | 10.772 | Không |
| 9 | `train-00191-of-00432.parquet` | 141,903,880 | `2a6105f517c44c68c3dc9467b40fa4aa477ad1eabc10872019168bb34f676817` | `downloaded_verified` | 1 | 14.562 | Không |
| 10 | `train-00114-of-00432.parquet` | 138,682,063 | `9960e72fd0bae090c37b56232be1064af7d0fc6702d6041d93fd7f8b357cbaf7` | `downloaded_verified` | 1 | 8.675 | Không |
| 11 | `train-00145-of-00432.parquet` | 175,607,729 | `93eb8e73ec6bb821341ce64f45041e207a8c2093be6d4175f462de65ee770b7b` | `downloaded_verified` | 1 | 18.432 | Không |
| 12 | `train-00172-of-00432.parquet` | 165,364,857 | `6f6f475dc43445360924a603b8c650e6c1f0a4292ed89e72f21d14c996f95cc3` | `downloaded_verified` | 1 | 11.954 | Không |
| 13 | `train-00429-of-00432.parquet` | 897,748,804 | `a28055ecde39801c8a662eb30e5d0001721c502eba32be76469d82f4d7f6f749` | `downloaded_verified` | 1 | 58.715 | Không |
| 14 | `train-00281-of-00432.parquet` | 162,473,779 | `450173725060c04d40a784abfaa1d3a6f7051152b40f34f203ec47ed4d3fba9d` | `downloaded_verified` | 1 | 18.116 | Không |
| 15 | `train-00053-of-00432.parquet` | 96,428,273 | `09f75f1fb797c243940e8d775aa613d13592b01411d3a30971a693764c29a0b7` | `downloaded_verified` | 1 | 6.128 | Không |
| 16 | `train-00267-of-00432.parquet` | 228,418,310 | `b75c55a3b6c67e1d476d4bf12c4e0d5a39e1e487eafedfe434302861d68ace18` | `downloaded_verified` | 1 | 17.900 | Không |
| 17 | `train-00039-of-00432.parquet` | 107,613,117 | `9bbe6604dd02a90ba9a8c3f4d1561814e59771df055ffa9f94cff4b83fca319f` | `downloaded_verified` | 1 | 9.468 | Không |
| 18 | `train-00329-of-00432.parquet` | 133,742,230 | `ee7b9dce6217adf7f68858d27120d3f5f975c52a4a9c42c7923de0f17484a648` | `downloaded_verified` | 1 | 12.155 | Không |
| 19 | `train-00302-of-00432.parquet` | 182,892,480 | `16c5740f63d180e40c746e72c979d8b7d8fcda893564904680b3359ad66ca7d9` | `downloaded_verified` | 1 | 17.756 | Không |
| 20 | `train-00363-of-00432.parquet` | 164,917,600 | `09e031f81406047ce84175af1de9c6b3ee588a0600c9432057796c9438753133` | `downloaded_verified` | 1 | 11.486 | Không |
| 21 | `train-00243-of-00432.parquet` | 158,779,058 | `03d7527337ea1b1df48e53f140caf7baa630195526b64b70223f22347123b4d9` | `downloaded_verified` | 1 | 13.644 | Không |
| 22 | `train-00104-of-00432.parquet` | 153,434,517 | `7192bcbb6b0022c51155493f250c36a2e6883e59100f817ea202dc12ab9cdb71` | `downloaded_verified` | 1 | 13.619 | Không |
| 23 | `train-00012-of-00432.parquet` | 90,955,955 | `27f201d80b586dde4af0e092eb7b3e59cf3a41141c16765c0c32c4c6ca0f2aa4` | `downloaded_verified` | 1 | 8.745 | Không |
| 24 | `train-00368-of-00432.parquet` | 141,044,232 | `6d8453f721d9f55b12d924a574df52eaa64f28fa23c2c73e5f699d8f919ebe92` | `downloaded_verified` | 1 | 10.275 | Không |
| 25 | `train-00410-of-00432.parquet` | 696,966,177 | `9b394604d254a0b508b332a0b8f313f610ee2a068ffeaf0772babdbdb84ac369` | `downloaded_verified` | 1 | 31.599 | Không |
| 26 | `train-00158-of-00432.parquet` | 227,860,499 | `97686ebbfaf327c49c9de130b896c8bc4422ec2b68447f1328642a900331f3f1` | `downloaded_verified` | 1 | 13.295 | Không |
| 27 | `train-00387-of-00432.parquet` | 125,697,135 | `f52b760cdcd4dddce4978e4f10debce92ab1f09869030c118a9489dd90e8ad08` | `downloaded_verified` | 1 | 11.223 | Không |
| 28 | `train-00077-of-00432.parquet` | 134,186,202 | `76f9dde9aa9fc5aeed1ad177bb6d2b5928d2aa6578eca836923ea470445355a2` | `downloaded_verified` | 1 | 9.116 | Không |
| 29 | `train-00178-of-00432.parquet` | 162,511,188 | `0fdc45d866db665828a6dee37283ef7aed2b6a821115178f21bdc5c583eb5403` | `downloaded_verified` | 1 | 11.834 | Không |
| 30 | `train-00150-of-00432.parquet` | 204,352,196 | `201d918b092f05fef2335a9f749f95312899434aeb46c37c5434f221d0683c43` | `downloaded_verified` | 1 | 8.669 | Không |
| 31 | `train-00057-of-00432.parquet` | 78,912,995 | `3cb8e53ffb99061509985d751b676664636d652bb98f76d5af57908e4b400003` | `downloaded_verified` | 1 | 6.785 | Không |
| 32 | `train-00181-of-00432.parquet` | 180,326,974 | `1d6229580289d0ed38a87d5192173e875d7483b0826d62cee3e556e3dc8ffb48` | `downloaded_verified` | 1 | 17.743 | Không |
| 33 | `train-00356-of-00432.parquet` | 151,498,074 | `c2b7603232f1e152cbd30676a96c5608ea59ead66eeb5589ee94d1ae4a825556` | `downloaded_verified` | 1 | 12.082 | Không |
| 34 | `train-00362-of-00432.parquet` | 131,037,783 | `acc746c0b7d2beef7b2cb1f23f211e4a25aa4a4c4c4646528754099f1e87717a` | `downloaded_verified` | 1 | 14.157 | Không |
| 35 | `train-00408-of-00432.parquet` | 416,791,670 | `ad58ff225ace793310eb9f46511c0b48ba3b1489dbf04df1f1675719c6fae9f9` | `downloaded_verified` | 1 | 18.960 | Không |

## Ghi chú kỹ thuật

- Môi trường khóa không cài `pyarrow`, nên việc đọc Parquet và đối chiếu số dòng/cột với `vsasv_metadata.csv` thực hiện qua `duckdb` (`read_parquet`).
- Lần đầu dừng ở bước 2 vì truyền `token=False` vào gated repo; lần này gọi `hf_hub_download` không truyền tham số `token` để dùng token đã lưu cục bộ theo mặc định thư viện (giống `scripts/ingest_evaluation_shards.py`), không đọc hay ghi giá trị token.
