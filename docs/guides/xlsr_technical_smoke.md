# Chuẩn bị và kiểm thử kỹ thuật XLS-R

Khung `src/models/xlsr.py` nhận waveform `float32` dạng `(batch, 64000)` ở 16 kHz từ pipeline hiện hành và trả một **logit spoof** cho mỗi mẫu. Dùng `BCEWithLogitsLoss` với nhãn `0 = bonafide`, `1 = spoof`; chỉ chuyển sigmoid khi cần score xác suất. Không truyền LFCC vào XLS-R.

## Chạy trên CPU

Cài dependency tùy chọn vào môi trường dự án đã có PyTorch; không cài đè requirements CPU lên Kaggle CUDA:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-xlsr.txt
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_xlsr.py -v
./.venv/Scripts/python.exe scripts/smoke_test_xlsr.py
```

Script tạo Wav2Vec2 thu nhỏ từ `configs/xlsr_smoke.json`, trọng số ngẫu nhiên và waveform tổng hợp. Không tải pretrained checkpoint, không đọc VSASV hoặc test, không dùng GPU. Báo cáo `reports/xlsr_cpu_smoke.json` ghi rõ số mẫu/shard dữ liệu thật bằng 0. Loss chỉ dùng kiểm tra luồng tính toán, không dùng so sánh chất lượng X0/X1. Các số chiều, số tầng và learning rate trong smoke không phải cấu hình thí nghiệm chính.

Transformers được import khi xây/tải backbone; B0 tiếp tục dùng requirements và entrypoint cũ. Máy không cài dependency tùy chọn vẫn chạy được các kiểm thử B0; nhóm kiểm thử encoder thật sẽ được đánh dấu bỏ qua, không được diễn giải là đã kiểm chứng XLS-R.

## Hành vi mô hình

- **X0:** `num_unfrozen_layers=0`; toàn bộ backbone đóng băng và ở eval, embedding không giữ graph. Chỉ classifier học.
- **X1:** số tầng mở phải dương và nhỏ hơn tổng số tầng Transformer. Chỉ các block cuối được mở; convolution, feature projection, positional convolution và normalization ngoài block giữ đóng băng. Graph vẫn đi qua các tầng mở.
- Gọi `train()` giữ phần đóng băng ở eval; dropout chỉ hoạt động trong block mở và classifier. LayerDrop và SpecAugment của backbone không hoạt động trong khung kỹ thuật này, kể cả khi cấu hình checkpoint gốc bật chúng.
- Pooling ghép mean và `sqrt(population_variance + 1e-8)` theo thời gian, không có tham số học; epsilon bảo vệ backward khi phương sai bằng 0. Đoạn đầu vào đã đủ 4 giây, không có padding cần loại khỏi pooling.
- `extract_embeddings()` cho phép lấy embedding; `optimizer_groups()` tạo hai group có tên `classifier`/`encoder`, tách learning rate và chỉ chứa tham số trainable, không trùng nhau. Khi cache embedding X0 sau này phải khóa preprocessing và identity trước; phiên này chưa tạo cache.
- `src/training/xlsr_checkpoint.py` đã có hợp đồng exact-resume riêng cho XLS-R. Checkpoint ghi atomic, chỉ lưu tham số trainable, optimizer, progress, sampler và Python/NumPy/PyTorch/CUDA RNG; backbone đóng băng phải được dựng lại từ model ID, revision và checksum trọng số đã ghim. Identity khóa manifest/audio/selection, model/revision/config/preprocessor/weights, normalization, X0/X1, head, batch, optimizer, learning rate, seed và ngân sách bước. Không dùng checkpoint identity B0 cho XLS-R.

## Kết nối pretrained ở bước tiếp theo

```python
from src.models.xlsr import XLSRBinaryClassifier, load_xlsr_backbone

# Thư mục checkpoint đã có sẵn, không tự truy cập mạng.
backbone = load_xlsr_backbone("data/processed/xlsr_checkpoint")
model = XLSRBinaryClassifier(backbone, num_unfrozen_layers=0)
logits = model(batch["waveform"])
```

Nếu dùng model ID trên Hub, phải truyền `revision` là commit SHA đủ 40 ký tự; mặc định chỉ đọc cache. Chỉ tải qua mạng khi truyền rõ `local_files_only=False`. Không dùng `main` làm revision và không tự thay bằng trọng số ngẫu nhiên khi tải lỗi. Trọng số lớn phải nằm ngoài Git.

Checkpoint `facebook/wav2vec2-xls-r-300m` là ứng viên kỹ thuật, chưa được tải hoặc khóa làm lựa chọn khoa học. Metadata chính thức đã được ghim tại revision bất biến `1a640f32ac3e39899438a2931f9924c02f080a54`; `config.json` có SHA-256 `0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac`, còn `preprocessor_config.json` có SHA-256 `a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335`. Cấu hình ứng viên nằm tại `configs/xlsr_pretrained_candidate.json`; checksum trọng số vẫn để trống vì chưa tải.

[Cấu hình phát hành](https://huggingface.co/facebook/wav2vec2-xls-r-300m/blob/1a640f32ac3e39899438a2931f9924c02f080a54/config.json) dùng kiến trúc Wav2Vec2 24 tầng, hidden size 1.024; [preprocessor phát hành](https://huggingface.co/facebook/wav2vec2-xls-r-300m/blob/1a640f32ac3e39899438a2931f9924c02f080a54/preprocessor_config.json) bật `do_normalize`. Theo [mã Hugging Face 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py), đây là zero-mean/unit-variance normalization trên waveform. Khung hiện tại truyền nguyên tensor đã qua pipeline, không gọi processor để tránh tự thêm phép biến đổi biên độ ngoài `configs/audio.json` và `docs/audio_protocol.md`.

Trước thí nghiệm pretrained, phải giải quyết rõ tương thích normalization với giao thức biên độ, khóa checkpoint/revision, số tầng mở, head, learning rate và ngân sách bằng development, rồi smoke GPU trên dữ liệu train/development. Nếu cần đổi tiền xử lý đã khóa, đề xuất quyết định mới và xin xác nhận trước khi sửa cấu hình âm thanh. Bộ ba pilot B0 vẫn chạy theo D019/D020 và source bundle đã khóa; phiên chuẩn bị này không publish lại source Kaggle.

## Rà soát đặc tả ngày 30/10/2026

Rà soát trong lúc B0 version20 đang chạy kết luận XLS-R **chưa sẵn sàng cho thí nghiệm khoa học**:

| Hạng mục | Đã có | Còn phải khóa |
|---|---|---|
| Checkpoint | Ứng viên `facebook/wav2vec2-xls-r-300m` đã ghim revision 40 ký tự và checksum hai cấu hình phát hành | Chưa khóa lựa chọn khoa học, tải/băm trọng số hoặc kiểm chứng load offline |
| Tiền xử lý | Đầu vào mono16kHz/4giây `float32`; code không tự normalize | Processor phát hành bật zero-mean/unit-variance, xung đột với waveform sau policy `peak`; cần quyết định mới nếu đổi audio protocol |
| X0/X1 | X0 đóng toàn backbone; X1 chỉ mở các block cuối | Số block mở X1 và cấu hình head/dropout khoa học |
| Optimizer | Có group encoder/classifier không trùng, chỉ chứa tham số trainable | Optimizer, learning rate từng group, scheduler/clipping/AMP và ngân sách |
| Resume | Checkpoint atomic, identity, trainable state, optimizer, sampler, progress và RNG đã đạt exact-resume CPU | Tích hợp vào entrypoint/controller/history/best của run khoa học sau khi cấu hình được khóa |
| Cổng chạy | CPU smoke X0/X1 đạt trên waveform tổng hợp | CUDA stack, pretrained load, dữ liệu train/development thật và runtime/VRAM |

Kiểm thử riêng exact-resume đạt 4/4; toàn bộ suite đạt 183/183 trong 186,276 giây với `TEMP`/`TMP` đặt trong workspace. Báo cáo kỹ thuật: `reports/xlsr_exact_resume_preflight.*`.

Phiên rà soát không tải pretrained weights, không gửi run XLS-R, không sửa `configs/audio.json`, không tạo packed artifact 20.000 mẫu hoặc evaluation manifest. Việc khóa normalization và các siêu tham số còn thiếu phải thành một đặc tả/decision riêng trước GPU smoke khoa học.

## Pilot normalization D024

D024 khóa thiết kế so sánh ghép cặp X0 trong `configs/xlsr_normalization_pilot.json`, chưa khóa nhánh thắng:

1. Cả hai nhánh dùng đúng `development-20k-v2`, seed `2026`, policy `peak`, cùng file/thứ tự sampler, cùng khởi tạo classifier và cùng endpoint.
2. Pipeline chung thực hiện resample 16 kHz → peak → đoạn 4 giây. `pipeline_only` giữ nguyên đoạn; `processor_zero_mean_unit_variance` mới chuẩn hóa từng đoạn theo `(x - mean) / sqrt(population_variance + 1e-7)`.
3. Backbone X0 đóng băng hoàn toàn; head 128 chiều/dropout 0,2 và Adam learning rate 0,001 chỉ là hằng số pilot, không phải cấu hình X0/X1 cuối.
4. Chọn development EER thấp hơn; nếu bằng nhau chính xác thì giữ `pipeline_only`. Mỗi nhánh có threshold development riêng. Test không tham gia bất kỳ quyết định nào.

Hàm `src/data/xlsr.py` áp dụng normalization tường minh sau đoạn cố định và từ chối mode, shape, dtype hoặc giá trị không hợp lệ. Năm kiểm thử mới bao phủ hai nhánh, công thức, đoạn hằng và cấu hình D024; nhóm XLS-R đạt 21/21, toàn suite đạt 188/188 trong 191,255 giây.

Hai cờ readiness vẫn là `false`. Trước khi chạy phải tải/băm trọng số đúng revision đã ghim, load offline, smoke CUDA để đo VRAM/runtime, khóa một batch size và step budget chung, rồi tích hợp entrypoint/controller/history/best với exact-resume. Không ngoại suy ngân sách B0. Bằng chứng: `reports/xlsr_normalization_pilot_preflight.*`.
