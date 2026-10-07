# Hướng dẫn hoàn thành công việc tồn từ Thứ Hai 05/10/2026

> Dùng cùng [kế hoạch ba ngày](ke_hoach_cong_viec_2026-10-05_den_2026-10-07.md). Thực hiện tuần tự; không đánh dấu một bước hoàn thành khi chưa có đầu ra và kiểm tra tương ứng.

## 1. Đích cần đạt

Chuỗi công việc cần hoàn thành là:

```text
Xác nhận quyền truy cập
→ tải thử 1 shard và đo băng thông
→ xác minh shard
→ cập nhật kiểm kê storage và ETA
→ lập phương án shard cho 20.000 mẫu train + dev
→ tải đủ audio
→ tạo và audit manifest ứng viên
→ cập nhật checkpoint
```

Trong hôm nay, ưu tiên hoàn thành đến bước lập phương án shard. Manifest 20.000 chỉ được gọi là hoàn thành khi các audio tương ứng đã có cục bộ và được kiểm chứng.

## 2. Chuẩn bị môi trường

Mở PowerShell tại thư mục dự án:

```powershell
Set-Location C:\Users\Admin\Documents\DoAnTTNT
git status --short
.\.venv\Scripts\Activate.ps1
python scripts\verify_environment.py
hf auth whoami
```

Phải dùng Python trong `.venv`. Python mặc định của terminal hiện không có `duckdb`, trong khi `.venv` đã có các thư viện dự án.

Nếu `hf auth whoami` báo chưa đăng nhập:

1. mở [trang VSASV chính thức](https://huggingface.co/datasets/hustep-lab/VSASV-Dataset), đăng nhập và chấp nhận điều kiện truy cập dataset;
2. chạy `hf auth login`;
3. chạy lại `hf auth whoami`.

Không ghi token Hugging Face vào script, báo cáo, lịch sử lệnh được chia sẻ hoặc Git.

**Cổng qua bước:** môi trường báo ĐẠT và `hf auth whoami` trả về đúng tài khoản có quyền truy cập VSASV.

## 3. Hoàn thành việc tồn của Thứ Hai: chọn shard thử

Dùng shard sau để phép đo có thể kiểm chứng và không trùng các shard đã có trước phép đo:

| Trường | Giá trị |
|---|---|
| Repository | `hustep-lab/VSASV-Dataset` |
| Revision | `92de668616780ed4f3d7f58a82262f82b56b4557` |
| Shard | `data/train-00002-of-00432.parquet` |
| Kích thước | `70.504.633 byte` |
| SHA-256 nguồn | `a48e840f19e51a54c0e3d86a7658679b979dc8a147b5ca8bd76d1c069bcdf73e` |

Các giá trị trên lấy từ tree metadata đã được cache cục bộ ở đúng revision. Trang file chính thức hiện hiển thị kích thước làm tròn khoảng 70,5 MB. Nếu lệnh dry-run cho revision này trả về kích thước khác, dừng phép đo và kiểm tra lại revision trước khi tải.

Chạy dry-run:

```powershell
$vsasvRepo = 'hustep-lab/VSASV-Dataset'
$vsasvRevision = '92de668616780ed4f3d7f58a82262f82b56b4557'
$trialRemotePath = 'data/train-00002-of-00432.parquet'
$expectedBytes = 70504633
$expectedSha256 = 'a48e840f19e51a54c0e3d86a7658679b979dc8a147b5ca8bd76d1c069bcdf73e'

hf download $vsasvRepo $trialRemotePath --type dataset --revision $vsasvRevision --dry-run --format json
```

**Cổng qua bước:** dry-run nhận diện đúng một file và kích thước đúng `70.504.633 byte`.

## 4. Tải thử và đo băng thông

Chạy toàn bộ khối sau trong cùng một cửa sổ PowerShell. `--force-download` được dùng để phép đo phản ánh lần truyền mạng thay vì chỉ lấy file từ cache.

```powershell
$incomingRoot = 'data\raw\vsasv_parquet\incoming'
$trialFile = Join-Path $incomingRoot $trialRemotePath
New-Item -ItemType Directory -Force -Path $incomingRoot | Out-Null

if (Test-Path -LiteralPath $trialFile) {
    throw "File thử đã tồn tại tại $trialFile; không thể đo một lượt tải mới một cách tin cậy."
}

$startedAt = Get-Date
$downloadClock = [System.Diagnostics.Stopwatch]::StartNew()
hf download $vsasvRepo $trialRemotePath --type dataset --revision $vsasvRevision --local-dir $incomingRoot --force-download
$downloadExitCode = $LASTEXITCODE
$downloadClock.Stop()
$finishedAt = Get-Date

if ($downloadExitCode -ne 0) {
    throw "Tải shard thất bại với mã $downloadExitCode."
}
if (-not (Test-Path -LiteralPath $trialFile)) {
    throw "Lệnh tải kết thúc nhưng không tìm thấy $trialFile."
}

$elapsedSeconds = $downloadClock.Elapsed.TotalSeconds
$downloadedBytes = (Get-Item -LiteralPath $trialFile).Length
$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $trialFile).Hash.ToLowerInvariant()
$bytesPerSecond = $downloadedBytes / $elapsedSeconds
$mibPerSecond = $bytesPerSecond / 1MB
$megabitsPerSecond = $bytesPerSecond * 8 / 1000000

[pscustomobject]@{
    started_at = $startedAt.ToString('o')
    finished_at = $finishedAt.ToString('o')
    elapsed_seconds = $elapsedSeconds
    downloaded_bytes = $downloadedBytes
    sha256 = $actualSha256
    mib_per_second = $mibPerSecond
    megabits_per_second = $megabitsPerSecond
} | Format-List
```

Không đóng cửa sổ PowerShell vì các biến đo sẽ được dùng để sinh báo cáo ở bước 7.

## 5. Xác minh shard trước khi nhập vào dữ liệu chính

Kiểm tra byte và SHA-256:

```powershell
if ($downloadedBytes -ne $expectedBytes) {
    throw "Sai kích thước: nhận $downloadedBytes, kỳ vọng $expectedBytes byte."
}
if ($actualSha256 -ne $expectedSha256) {
    throw "SHA-256 không khớp nguồn."
}
```

Kiểm tra schema và waveform trên riêng shard vừa tải:

```powershell
python scripts\audio_smoke_test.py `
  --parquet-dir data\raw\vsasv_parquet\incoming\data `
  --samples-per-shard 8 `
  --json-output reports\download_trial_audio_smoke.json `
  --markdown-output reports\download_trial_audio_smoke.md
```

Chỉ khi lệnh trên báo ĐẠT mới chuyển file sang thư mục dữ liệu chính:

```powershell
$finalDirectory = 'data\raw\vsasv_parquet\data'
$finalFile = Join-Path $finalDirectory 'train-00002-of-00432.parquet'

if (Test-Path -LiteralPath $finalFile) {
    throw "Từ chối ghi đè file đã có: $finalFile"
}

Move-Item -LiteralPath $trialFile -Destination $finalFile
python scripts\audit_local_storage.py
```

Mở `reports/local_storage_audit.md` và xác nhận:

- có 6 shard;
- shard mới có đúng số hàng duy nhất;
- mọi hàng của shard mới khớp metadata;
- `hard_issues` rỗng trong JSON.

Nếu bất kỳ điều kiện nào không đạt, không tiếp tục tạo manifest. Giữ file ở khu vực `incoming` hoặc tách khỏi thư mục dữ liệu chính để điều tra.

## 6. Tính ETA 20.000 và 40.000 mẫu

Sau khi `audit_local_storage.py` chạy xong, dùng số byte dự báo mới trong JSON:

```powershell
$storageAudit = Get-Content -Raw reports\local_storage_audit.json | ConvertFrom-Json
$projection20k = $storageAudit.linear_sample_projections | Where-Object target_samples -eq 20000
$projection40k = $storageAudit.linear_sample_projections | Where-Object target_samples -eq 40000
$eta20kSeconds = $projection20k.estimated_bytes / $bytesPerSecond
$eta40kSeconds = $projection40k.estimated_bytes / $bytesPerSecond

[pscustomobject]@{
    target_20000_gib = $projection20k.estimated_gib
    eta_20000_hours = $eta20kSeconds / 3600
    target_40000_gib = $projection40k.estimated_gib
    eta_40000_hours = $eta40kSeconds / 3600
} | Format-List
```

Đây là ETA từ một lượt tải. Ghi rõ loại mạng, gián đoạn quan sát được và thời điểm đo; không trình bày nó như cam kết thời gian chắc chắn.

## 7. Sinh báo cáo tải thử

Chạy trong cùng PowerShell để dùng các biến của phép đo:

```powershell
$sourceUrl = "https://huggingface.co/datasets/$vsasvRepo/resolve/$vsasvRevision/$trialRemotePath"
$networkNotes = Read-Host 'Ghi chú mạng (Wi-Fi/LAN, gián đoạn nếu có)'
if ([string]::IsNullOrWhiteSpace($networkNotes)) {
    $networkNotes = 'Không được ghi nhận trong lần đo; chưa xác định Wi-Fi/LAN hoặc tình trạng gián đoạn.'
}

$bandwidthReport = [ordered]@{
    status = 'ĐẠT'
    source_url = $sourceUrl
    repository = $vsasvRepo
    revision = $vsasvRevision
    shard = [System.IO.Path]::GetFileName($trialRemotePath)
    started_at = $startedAt.ToString('o')
    finished_at = $finishedAt.ToString('o')
    elapsed_seconds = $elapsedSeconds
    downloaded_bytes = $downloadedBytes
    bytes_per_second = $bytesPerSecond
    mib_per_second = $mibPerSecond
    megabits_per_second = $megabitsPerSecond
    sha256 = $actualSha256
    source_checksum_or_etag = $expectedSha256
    checksum_status = 'khớp SHA-256 nguồn'
    network_notes = $networkNotes
    projections = @(
        [ordered]@{ target_samples = 20000; estimated_bytes = $projection20k.estimated_bytes; eta_hours = $eta20kSeconds / 3600 },
        [ordered]@{ target_samples = 40000; estimated_bytes = $projection40k.estimated_bytes; eta_hours = $eta40kSeconds / 3600 }
    )
}

$bandwidthReport | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 reports\download_bandwidth_trial.json

$bandwidthMarkdown = @"
# Báo cáo đo băng thông tải VSASV

- Trạng thái: **ĐẠT**
- Shard: ``$($bandwidthReport.shard)``
- Revision: ``$vsasvRevision``
- Dung lượng: **$downloadedBytes byte**
- Thời gian: **$([math]::Round($elapsedSeconds, 3)) giây**
- Tốc độ: **$([math]::Round($mibPerSecond, 3)) MiB/s** ($([math]::Round($megabitsPerSecond, 3)) Mb/s)
- SHA-256: ``$actualSha256``
- Đối chiếu checksum: **khớp SHA-256 nguồn**
- ETA 20.000 mẫu: **$([math]::Round($eta20kSeconds / 3600, 2)) giờ**
- ETA 40.000 mẫu: **$([math]::Round($eta40kSeconds / 3600, 2)) giờ**
- Ghi chú mạng: $networkNotes

ETA dùng dự báo dung lượng mới nhất từ ``reports/local_storage_audit.json`` và một phép đo tải. Đây là số liệu lập kế hoạch, không phải kết quả khoa học.
"@

$bandwidthMarkdown | Set-Content -Encoding utf8 reports\download_bandwidth_trial.md
```

Kiểm tra nhanh:

```powershell
Get-Content reports\download_bandwidth_trial.md
Get-Content -Raw reports\download_bandwidth_trial.json | ConvertFrom-Json | Format-List
```

## 8. Lập phương án shard cho ứng viên 20.000 mẫu

Không tải liên tiếp 40 shard đầu vì thứ tự shard có thể gắn với cấu trúc dữ liệu. Tạo danh sách ứng viên phân bố đều trên 432 shard, sau đó cộng các shard cục bộ đã có. Đây là phương án ứng viên để kiểm tra, chưa phải manifest đã khóa.

```powershell
$treeFile = Get-ChildItem data\raw\vsasv_parquet\.cache\huggingface\trees -Filter '*.json' -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$tree = Get-Content -Raw -LiteralPath $treeFile.FullName | ConvertFrom-Json
$gridIndices = 0..47 | ForEach-Object { [int][math]::Round($_ * 431 / 47) } | Sort-Object -Unique
$localNames = Get-ChildItem data\raw\vsasv_parquet\data -Filter '*.parquet' -File | Select-Object -ExpandProperty Name
$localIndices = $localNames | ForEach-Object {
    if ($_ -match '^train-(\d{5})-of-00432\.parquet$') { [int]$Matches[1] }
}
$candidateIndices = @($gridIndices + $localIndices) | Sort-Object -Unique

$shardPlan = foreach ($index in $candidateIndices) {
    $name = 'train-{0:D5}-of-00432.parquet' -f $index
    $remotePath = "data/$name"
    $entry = $tree.files.PSObject.Properties[$remotePath].Value
    if ($null -eq $entry) { throw "Không tìm thấy $remotePath trong tree metadata." }
    [pscustomobject]@{
        shard = $name
        size_bytes = $entry.size
        sha256 = $entry.lfs_sha256
        status = if ($localNames -contains $name) { 'available' } else { 'planned' }
    }
}

$shardPlan | Export-Csv -NoTypeInformation -Encoding utf8 reports\development_shard_candidates_20k.csv
$missingPlan = $shardPlan | Where-Object status -eq 'planned'
$plannedBytes = ($missingPlan | Measure-Object size_bytes -Sum).Sum
$candidateSummary = [ordered]@{
    source_csv = 'reports/development_shard_candidates_20k.csv'
    source_csv_sha256 = (Get-FileHash -Algorithm SHA256 reports\development_shard_candidates_20k.csv).Hash.ToLowerInvariant()
    selection_rule = '48 chỉ số phân bố đều round(k*431/47), hợp với các shard cục bộ, rồi loại trùng'
    candidate_shards = $shardPlan.Count
    available_shards = ($shardPlan | Where-Object status -eq 'available').Count
    missing_shards = $missingPlan.Count
    missing_bytes = $plannedBytes
    missing_gib = $plannedBytes / 1GB
    estimated_download_hours = ($plannedBytes / $bytesPerSecond) / 3600
    estimated_download_hours_with_20_percent_buffer = ($plannedBytes / $bytesPerSecond) / 3600 * 1.2
}
$candidateSummary | Format-List
$candidateSummary | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 reports\development_shard_candidates_20k_summary.json

$candidateMarkdown = @"
# Phương án shard ứng viên cho 20.000 mẫu train + development

- Shard ứng viên: **$($candidateSummary.candidate_shards)**
- Đã có: **$($candidateSummary.available_shards)**
- Cần tải: **$($candidateSummary.missing_shards)**
- Dung lượng cần tải: **$([math]::Round($candidateSummary.missing_gib, 2)) GiB** ($($candidateSummary.missing_bytes) byte)
- ETA theo phép đo: **$([math]::Round($candidateSummary.estimated_download_hours, 2)) giờ**
- ETA có dự phòng 20%: **$([math]::Round($candidateSummary.estimated_download_hours_with_20_percent_buffer, 2)) giờ**
- Quy tắc chọn: $($candidateSummary.selection_rule)
- SHA-256 CSV: ``$($candidateSummary.source_csv_sha256)``

Đây là phương án tải theo shard thực, chưa phải manifest audio đã khóa. Coverage 20.000 mẫu chỉ được xác nhận sau khi tải, giao với closed train/development và audit.
"@
$candidateMarkdown | Set-Content -Encoding utf8 reports\development_shard_candidates_20k_summary.md
```

Trước khi tải tiếp, mở CSV và ghi các số tổng vào kế hoạch. Tải theo đợt nhỏ, ví dụ 4–8 shard; mỗi đợt phải đối chiếu byte/SHA-256, chạy lại storage audit và xem phân bố `utt_type`. Dừng mở rộng khi có đủ audio train/dev để lập ứng viên 20.000 và vẫn bảo đảm dung lượng an toàn.

## 9. Tạo manifest ứng viên sau khi đủ audio

Hiện repository chưa có script tạo manifest 20.000 mẫu. Không ghép CSV thủ công. Cần triển khai `scripts/make_development_manifest.py` cùng kiểm thử trước khi sinh artifact chính thức.

Script phải thực hiện đúng các bước sau:

1. đọc toàn bộ Parquet cục bộ với tên shard và `audio.sampling_rate`;
2. giao theo `file` với `closed_train.csv` và `closed_dev.csv`; không đưa `closed_test.csv` vào tập chọn;
3. kiểm tra speaker, `utt_type` và nhãn khớp split/metadata;
4. xếp ổn định bằng SHA-256 của `2026|file` trong từng split và stratum;
5. chọn mục tiêu 15.885 train và 4.115 development nếu coverage cho phép;
6. ghi đủ schema đã nêu trong `docs/chuan_bi_du_lieu_tuan_02_2026-10-04.md`;
7. sinh báo cáo số mẫu mục tiêu, đã ánh xạ shard, đã có audio, phân bố và checksum manifest;
8. thất bại rõ ràng hoặc báo shortfall nếu audio chưa đủ, không lặp file và không chuyển speaker giữa split.

Kiểm thử tối thiểu phải bao phủ tính xác định, schema, nhãn nhị phân, file duy nhất, speaker-disjoint và trường hợp thiếu audio. Sau khi script và test tồn tại, chạy:

```powershell
python -m unittest discover -s tests -v
python scripts\make_development_manifest.py --target-total 20000 --seed 2026
```

Lệnh thứ hai là giao diện dự kiến; chỉ chạy sau khi script đã được triển khai đúng giao diện này. Khi chốt quy tắc lấy mẫu hoặc quy mô cuối, ghi một quyết định mới trong `docs/decisions.md`.

## 10. Kết thúc và bàn giao

Chạy các kiểm tra phù hợp sau khi thêm shard hoặc code:

```powershell
python -m unittest discover -s tests -v
python scripts\verify_environment.py
python scripts\check_leakage.py
python scripts\smoke_test_dataset_loader.py
git diff --check
git status --short
```

Cập nhật `docs/current_state.md` bằng số shard, số mẫu, băng thông, ETA, trạng thái manifest và blocker thực tế. Không commit audio, thư mục cache, checkpoint, token hoặc log lớn.

## 11. Khi nào được xem là hoàn thành phần việc từ Thứ Hai

- Có shard thứ sáu được tải và xác minh đúng byte/SHA-256/schema/metadata.
- Có `reports/download_bandwidth_trial.*` với số đo thật.
- Có phương án shard, tổng byte và ETA cho ứng viên 20.000 train + development.
- Có manifest ứng viên được tạo bằng script có kiểm thử, hoặc có báo cáo shortfall chính xác nếu audio chưa đủ.
- Manifest không trùng file, không rò rỉ speaker và không dùng test để chọn quy mô/phân bố.
- `docs/current_state.md` phản ánh đúng số shard/mẫu thực tế và việc đầu tiên tiếp theo.

Không chọn amplitude policy và chưa bắt đầu XLS-R trong chuỗi công việc này.
