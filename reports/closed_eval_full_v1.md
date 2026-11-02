# Evaluation manifest dev-full/test-full (D025)

- Trạng thái: **ĐẠT**
- Nguồn: VSASV-HF-public-snapshot-v1; seed 2026; ngưỡng mất mẫu 5.0% mỗi (split, utt_type).
- Shard đã nạp: 265; mẫu trong sổ hash: 69964.

## closed_dev

- Dòng gốc 39116 → giữ 38777.

| utt_type | Gốc | Giữ | Mất (%) | Speaker gốc | Speaker giữ |
|---|---:|---:|---:|---:|---:|
| adversarial_attack | 12253 | 12253 | 0.00 | 22 | 22 |
| bonafide | 14490 | 14151 | 2.34 | 172 | 172 |
| replay | 120 | 120 | 0.00 | 3 | 3 |
| voice_conversion | 12253 | 12253 | 0.00 | 22 | 22 |
- Loại do `internal_duplicate`: {'bonafide': 301}
- Loại do `train_overlap`: {'bonafide': 38}

## closed_test

- Dòng gốc 30848 → giữ 30698.

| utt_type | Gốc | Giữ | Mất (%) | Speaker gốc | Speaker giữ |
|---|---:|---:|---:|---:|---:|
| adversarial_attack | 7361 | 7361 | 0.00 | 21 | 21 |
| bonafide | 16006 | 15856 | 0.94 | 171 | 171 |
| replay | 120 | 120 | 0.00 | 3 | 3 |
| voice_conversion | 7361 | 7361 | 0.00 | 21 | 21 |
- Loại do `dev_overlap`: {'bonafide': 41}
- Loại do `internal_duplicate`: {'bonafide': 106}
- Loại do `train_overlap`: {'bonafide': 3}

## Kiểm tra

- Hash duy nhất trong dev/test: True/True; trùng dev–test: 0; train–dev: 0; train–test: 0.
- Đối chiếu audit D017 cho dev: {'compared': 4115, 'mismatched': 0}.
- Speaker-disjoint train manifest/dev/test: True.

## Manifest
- `data/manifests/closed_dev_full_v1.csv`: 38777 dòng, SHA-256 `a79f1bb2595bc46b08ce30b8652234b1858cc26e558e2ccd33632435df1a5360`
- `data/manifests/closed_test_full_v1.csv`: 30698 dòng, SHA-256 `0a7a70bc8ededdd83e5f6df06580fc954c6f2da848366e37bc265331ca17d266`

## Giới hạn

- Không bù mẫu; số mẫu mất là hệ quả khử trùng, không phải chọn lọc theo metric.
- Hash tính lúc nạp shard; audit độc lập 100% cần tải lại shard ở revision ghim.
