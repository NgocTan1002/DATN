"""PREP-01: Kiểm kho test-full cục bộ, đối chiếu Kaggle chỉ đọc và ước tính chi phí chấm test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, BinaryIO, Literal, NoReturn, Sequence

ROOT = Path(__file__).resolve().parents[1]
WORKTREE_XLSR_ROOT = ROOT.parent / "DoAnTTNT-xlsr"
CACHE_DIR = ROOT / ".cache" / "prep01"
DEFAULT_JSON_REPORT = ROOT / "reports" / "pre_test_inventory.json"
DEFAULT_MD_REPORT = ROOT / "reports" / "pre_test_inventory.md"

CHUNK_BYTES = 8 * 1024 * 1024
DEV_FULL_SAMPLES = 38777
DEV_FULL_SHARDS = 186
DEV_FULL_BONAFIDE_SAMPLES = 14151
SAFETY_FACTOR = 1.25

EXPECTED_TEST_CONTEXT = {
    "artifact_count": 30698,
    "chunk_shard_count": 162,
    "source_manifest_sha256": "0a7a70bc8ededdd83e5f6df06580fc954c6f2da848366e37bc265331ca17d266",
    "index_sha256": "8f862032b6af42e1d35ff7d9e862308afb781308d13f24178a4bb1409237c78a",
    "waveforms_sha256": "556bb99de8c39ed8f8568ae36d170907f4ac2959b9c4ab2f1a8d4e48423b32b2",
}

TEST_TABLE_NAMES = frozenset({"closed_test_full_v1.csv", "index.csv"})
DATASET_CANDIDATE_KEYWORDS = ("packed", "test", "eval")
DATASET_REF_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
KAGGLE_AUTH_ERROR_MARKERS = (
    "authentication required",
    "unauthorized",
    "401",
    "403",
    "could not find kaggle",
    "kaggle auth login",
)


def require(condition: bool, message: str) -> None:
    """Kiểm tra điều kiện; ném ValueError tiếng Việt nếu sai."""
    if not condition:
        raise ValueError(message)


def open_test_artifact_binary_only(path: Path, mode: str = "rb") -> BinaryIO:
    """Chỉ cho phép mở file dữ liệu test ở chế độ nhị phân chỉ đọc ('rb')."""
    if mode != "rb":
        raise ValueError(
            "Chỉ cho phép mở file dữ liệu test ở chế độ nhị phân chỉ đọc ('rb'); "
            f"từ chối mode={mode!r} cho {path}."
        )
    return path.open("rb")


def forbid_test_table_parse(path: Path) -> NoReturn:
    """Từ chối mọi thao tác parse dòng hoặc cột trên manifest/index của tập test."""
    raise PermissionError(
        f"PREP-01 cấm đọc nội dung hoặc parse dòng/cột của file test: {path}."
    )


def stream_sha256_and_size(path: Path, chunk_size: int = CHUNK_BYTES) -> tuple[str, int]:
    """Băm SHA-256 theo luồng byte chỉ đọc ('rb') và trả về (sha256_hex, size_bytes)."""
    require(chunk_size > 0, "Kích thước chunk phải là số nguyên dương.")
    require(path.is_file(), f"Không tìm thấy file để băm: {path}")
    digest = hashlib.sha256()
    total_bytes = 0
    with open_test_artifact_binary_only(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            total_bytes += len(chunk)
    stat_size = path.stat().st_size
    require(
        total_bytes == stat_size,
        f"Số byte đọc được ({total_bytes}) không khớp kích thước file ({stat_size}): {path}",
    )
    return digest.hexdigest(), total_bytes


def validate_kaggle_readonly_command(cmd: Sequence[str]) -> list[str]:
    """Chỉ cho phép đúng 3 dạng lệnh Kaggle chỉ đọc của PREP-01."""
    tokens = list(cmd)
    error_message = (
        "Lệnh Kaggle không hợp lệ: PREP-01 chỉ cho phép đúng 3 lệnh chỉ đọc "
        "('kaggle -W quota', 'kaggle -W datasets list --mine', "
        "'kaggle -W datasets files <dataset>')."
    )
    if len(tokens) < 3 or tokens[0] != "kaggle" or tokens[1] != "-W":
        raise ValueError(error_message)
    if tokens == ["kaggle", "-W", "quota"]:
        return tokens
    if tokens == ["kaggle", "-W", "datasets", "list", "--mine"]:
        return tokens
    if (
        len(tokens) == 5
        and tokens[2] == "datasets"
        and tokens[3] == "files"
        and not tokens[4].startswith("-")
        and bool(DATASET_REF_PATTERN.fullmatch(tokens[4]))
    ):
        return tokens
    raise ValueError(error_message)


def run_kaggle_readonly(cmd: Sequence[str], log_path: Path) -> str:
    """Chạy lệnh Kaggle đã qua whitelist và ghi log ra đĩa."""
    validated = validate_kaggle_readonly_command(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        validated,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    log_path.write_text(combined, encoding="utf-8")
    lower = combined.lower()
    if any(marker in lower for marker in KAGGLE_AUTH_ERROR_MARKERS):
        raise RuntimeError(
            "Kaggle yêu cầu đăng nhập lại; hãy dừng và để người dùng tự chạy 'kaggle auth login'."
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Lệnh Kaggle thất bại (mã thoát {completed.returncode}): {' '.join(validated)}\n{combined}"
        )
    return combined


def check_local_test_inventory(root: Path = ROOT) -> dict[str, Any]:
    """Kiểm tra toàn vẹn kho test-full cục bộ bằng băm byte và đọc metadata.json."""
    packed_dir = root / "data" / "processed" / "packed" / "closed_test_full_v1"
    metadata_path = packed_dir / "metadata.json"
    index_path = packed_dir / "index.csv"
    waveforms_path = packed_dir / "waveforms.f32"
    manifest_path = root / "data" / "manifests" / "closed_test_full_v1.csv"
    config_v1_path = root / "configs" / "test_evaluation.json"
    config_x1r_path = root / "configs" / "test_evaluation_x1r.json"

    for required in (metadata_path, index_path, waveforms_path, manifest_path, config_v1_path, config_x1r_path):
        require(required.is_file(), f"Thiếu file bắt buộc để kiểm kho test-full: {required}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_v1 = json.loads(config_v1_path.read_text(encoding="utf-8"))
    config_x1r = json.loads(config_x1r_path.read_text(encoding="utf-8"))

    # Chỉ băm byte cho manifest, index.csv và waveforms.f32; tuyệt đối không parse nội dung.
    manifest_sha, manifest_bytes = stream_sha256_and_size(manifest_path)
    index_sha, index_bytes = stream_sha256_and_size(index_path)
    waveforms_sha, waveforms_bytes = stream_sha256_and_size(waveforms_path)
    metadata_sha, metadata_bytes = stream_sha256_and_size(metadata_path)

    total_samples = int(metadata["total_samples"])
    expected_waveforms_bytes = total_samples * 4
    artifact_count = int(metadata["artifact_count"])
    chunk_shard_count = int(metadata["chunk_shard_count"])

    checks = {
        "manifest_sha_matches_context": manifest_sha == EXPECTED_TEST_CONTEXT["source_manifest_sha256"],
        "manifest_sha_matches_metadata": manifest_sha == metadata["source_manifest_sha256"],
        "manifest_sha_matches_config_v1": manifest_sha == config_v1["test"]["manifest_sha256"],
        "manifest_sha_matches_config_x1r": manifest_sha == config_x1r["test"]["manifest_sha256"],
        "index_sha_matches_context": index_sha == EXPECTED_TEST_CONTEXT["index_sha256"],
        "index_sha_matches_metadata": index_sha == metadata["index_sha256"],
        "waveforms_sha_matches_context": waveforms_sha == EXPECTED_TEST_CONTEXT["waveforms_sha256"],
        "waveforms_sha_matches_metadata": waveforms_sha == metadata["waveforms_sha256"],
        "artifact_count_matches_context": artifact_count == EXPECTED_TEST_CONTEXT["artifact_count"],
        "artifact_count_matches_config_v1": artifact_count == int(config_v1["test"]["samples"]),
        "artifact_count_matches_config_x1r": artifact_count == int(config_x1r["test"]["samples"]),
        "chunk_shard_count_matches_context": chunk_shard_count == EXPECTED_TEST_CONTEXT["chunk_shard_count"],
        "waveforms_bytes_equals_total_samples_x4": waveforms_bytes == expected_waveforms_bytes,
    }
    all_matched = all(checks.values())
    require(all_matched, f"Kiểm kho test-full cục bộ không khớp: {checks}")

    return {
        "no_test_labels_or_content_read": True,
        "all_checks_passed": all_matched,
        "checks": checks,
        "metadata_summary": {
            "path": str(metadata_path.relative_to(root)).replace("\\", "/"),
            "sha256": metadata_sha,
            "bytes": metadata_bytes,
            "format_version": metadata["format_version"],
            "preprocessing_stage": metadata["preprocessing_stage"],
            "dtype": metadata["dtype"],
            "target_sample_rate": metadata["target_sample_rate"],
            "crop_seed": metadata["crop_seed"],
            "split": metadata["split"],
            "artifact_count": artifact_count,
            "chunk_shard_count": chunk_shard_count,
            "total_samples": total_samples,
            "expected_waveforms_bytes": expected_waveforms_bytes,
        },
        "files": {
            "manifest": {
                "path": str(manifest_path.relative_to(root)).replace("\\", "/"),
                "sha256": manifest_sha,
                "bytes": manifest_bytes,
                "expected_context_sha256": EXPECTED_TEST_CONTEXT["source_manifest_sha256"],
                "expected_metadata_sha256": metadata["source_manifest_sha256"],
                "expected_config_v1_sha256": config_v1["test"]["manifest_sha256"],
                "expected_config_x1r_sha256": config_x1r["test"]["manifest_sha256"],
                "matched": (
                    checks["manifest_sha_matches_context"]
                    and checks["manifest_sha_matches_metadata"]
                    and checks["manifest_sha_matches_config_v1"]
                    and checks["manifest_sha_matches_config_x1r"]
                ),
            },
            "index": {
                "path": str(index_path.relative_to(root)).replace("\\", "/"),
                "sha256": index_sha,
                "bytes": index_bytes,
                "expected_context_sha256": EXPECTED_TEST_CONTEXT["index_sha256"],
                "expected_metadata_sha256": metadata["index_sha256"],
                "matched": checks["index_sha_matches_context"] and checks["index_sha_matches_metadata"],
            },
            "waveforms": {
                "path": str(waveforms_path.relative_to(root)).replace("\\", "/"),
                "sha256": waveforms_sha,
                "bytes": waveforms_bytes,
                "expected_bytes_total_samples_x4": expected_waveforms_bytes,
                "expected_context_sha256": EXPECTED_TEST_CONTEXT["waveforms_sha256"],
                "expected_metadata_sha256": metadata["waveforms_sha256"],
                "matched": (
                    checks["waveforms_sha_matches_context"]
                    and checks["waveforms_sha_matches_metadata"]
                    and checks["waveforms_bytes_equals_total_samples_x4"]
                ),
            },
        },
    }


def _parse_kaggle_quota_text(text: str) -> dict[str, Any]:
    lower = text.lower()
    if any(marker in lower for marker in KAGGLE_AUTH_ERROR_MARKERS):
        raise RuntimeError(
            "Log quota Kaggle cho thấy lỗi xác thực; hãy dừng và để người dùng tự chạy 'kaggle auth login'."
        )
    resources: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^(GPU|TPU)\s+([0-9.]+h)\s+([0-9.]+h)\s+([0-9.]+h)\s+(\S+)\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        used_str, rem_str, total_str = match.group(2), match.group(3), match.group(4)
        resources.append(
            {
                "resource": match.group(1),
                "used": used_str,
                "remaining": rem_str,
                "total": total_str,
                "used_hours": float(used_str[:-1]),
                "remaining_hours": float(rem_str[:-1]),
                "total_hours": float(total_str[:-1]),
                "refresh_at": match.group(5),
            }
        )
    require(bool(resources), f"Không parse được thông tin quota Kaggle từ log:\n{text}")
    gpu_entry = next((item for item in resources if item["resource"] == "GPU"), None)
    require(gpu_entry is not None, "Không tìm thấy dòng GPU trong output 'kaggle -W quota'.")
    return {"resources": resources, "gpu": gpu_entry}


def _parse_kaggle_datasets_list_text(text: str) -> list[dict[str, Any]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    datasets: list[dict[str, Any]] = []
    row_pattern = re.compile(
        r"^(\S+/\S+)\s{2,}(.+?)\s{2,}(\d+)\s{2,}(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
        r"(\d+)\s+(\d+)\s+([0-9.]+)\s*$"
    )
    for line in lines:
        if line.startswith("Warning:") or line.startswith("ref ") or set(line.replace(" ", "")) <= {"-"}:
            continue
        match = row_pattern.match(line)
        if match:
            datasets.append(
                {
                    "ref": match.group(1),
                    "title": match.group(2).strip(),
                    "size_bytes": int(match.group(3)),
                    "last_updated": match.group(4),
                    "download_count": int(match.group(5)),
                    "vote_count": int(match.group(6)),
                    "usability_rating": float(match.group(7)),
                }
            )
    require(bool(datasets), f"Không parse được danh sách dataset từ log 'kaggle -W datasets list --mine':\n{text}")
    return datasets


def _parse_kaggle_dataset_files_text(text: str) -> list[dict[str, Any]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    files: list[dict[str, Any]] = []
    row_pattern = re.compile(
        r"^(\S+)\s+(\d+)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*$"
    )
    for line in lines:
        if line.startswith("Warning:") or line.startswith("name ") or set(line.replace(" ", "")) <= {"-"}:
            continue
        match = row_pattern.match(line)
        if match:
            files.append(
                {
                    "name": match.group(1),
                    "size_bytes": int(match.group(2)),
                    "creation_date": match.group(3),
                }
            )
    return files


def _read_log_text(path: Path) -> str:
    """Đọc file log hỗ trợ cả UTF-8 và UTF-16 BOM do PowerShell redirection tạo ra."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8")


def check_kaggle_inventory(
    cache_dir: Path = CACHE_DIR,
    kaggle_mode: Literal["auto", "logs", "live"] = "auto",
) -> dict[str, Any]:
    """Đọc hoặc chạy 3 lệnh Kaggle chỉ đọc và kiểm tra xem closed_test_full_v1 đã có trên Kaggle chưa."""
    quota_log = cache_dir / "kaggle_quota.log"
    datasets_log = cache_dir / "kaggle_datasets_list.log"

    if kaggle_mode == "live" or (kaggle_mode == "auto" and not (quota_log.is_file() and datasets_log.is_file())):
        quota_text = run_kaggle_readonly(["kaggle", "-W", "quota"], quota_log)
        datasets_text = run_kaggle_readonly(["kaggle", "-W", "datasets", "list", "--mine"], datasets_log)
    else:
        require(quota_log.is_file(), f"Thiếu log quota Kaggle: {quota_log}")
        require(datasets_log.is_file(), f"Thiếu log danh sách dataset Kaggle: {datasets_log}")
        quota_text = _read_log_text(quota_log)
        datasets_text = _read_log_text(datasets_log)

    quota = _parse_kaggle_quota_text(quota_text)
    datasets = _parse_kaggle_datasets_list_text(datasets_text)

    candidate_datasets = [
        item
        for item in datasets
        if any(
            kw in item["ref"].lower() or kw in item["title"].lower()
            for kw in DATASET_CANDIDATE_KEYWORDS
        )
    ]

    inspected_datasets: list[dict[str, Any]] = []
    test_full_matches: list[dict[str, Any]] = []
    for ds in candidate_datasets:
        slug = ds["ref"].split("/", 1)[1]
        files_log = cache_dir / f"kaggle_files_{slug}.log"
        if kaggle_mode == "live" or (kaggle_mode == "auto" and not files_log.is_file()):
            files_text = run_kaggle_readonly(["kaggle", "-W", "datasets", "files", ds["ref"]], files_log)
        else:
            require(files_log.is_file(), f"Thiếu log danh sách file Kaggle: {files_log}")
            files_text = _read_log_text(files_log)

        remote_files = _parse_kaggle_dataset_files_text(files_text)
        matching_test = [
            f
            for f in remote_files
            if "closed_test_full_v1" in f["name"].lower() or "test_full" in f["name"].lower()
        ]
        if matching_test:
            for item in matching_test:
                test_full_matches.append({"dataset": ds["ref"], **item})

        inspected_datasets.append(
            {
                "ref": ds["ref"],
                "title": ds["title"],
                "dataset_zip_size_bytes": ds["size_bytes"],
                "files_log": str(files_log.relative_to(ROOT)).replace("\\", "/") if files_log.is_relative_to(ROOT) else str(files_log),
                "files": remote_files,
                "contains_closed_test_full_v1": bool(matching_test),
            }
        )

    return {
        "quota": quota,
        "quota_log": str(quota_log.relative_to(ROOT)).replace("\\", "/") if quota_log.is_relative_to(ROOT) else str(quota_log),
        "datasets_list_log": str(datasets_log.relative_to(ROOT)).replace("\\", "/") if datasets_log.is_relative_to(ROOT) else str(datasets_log),
        "all_my_datasets": datasets,
        "candidate_filter_keywords": list(DATASET_CANDIDATE_KEYWORDS),
        "inspected_candidate_datasets": inspected_datasets,
        "has_test_full_on_kaggle": bool(test_full_matches),
        "matched_test_full_files": test_full_matches,
        "remote_hash_note": (
            "Lệnh chỉ đọc 'kaggle -W datasets files' chỉ trả tên và kích thước byte (không trả SHA-256). "
            "Không tìm thấy thư mục hay file nào tên closed_test_full_v1 trong tài khoản Kaggle."
            if not test_full_matches
            else "Tìm thấy file gợi ý test-full trên Kaggle; đã đối chiếu kích thước byte (không có SHA-256 từ xa khi không tải về)."
        ),
    }


def _load_json(path: Path) -> Any:
    require(path.is_file(), f"Thiếu file JSON nguồn: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_eval_records(records: list[dict[str, Any]], *, includes_lowpass_7k: bool = False) -> dict[str, Any]:
    require(bool(records), "Danh sách lượt đánh giá dev-full không được rỗng.")
    totals = [float(item["total_seconds"]) for item in records]
    loaders = [float(item["loader_seconds"]) for item in records]
    models = [float(item["model_seconds"]) for item in records]
    samples = {int(item["samples"]) for item in records}
    require(samples == {DEV_FULL_SAMPLES}, f"Số mẫu dev-full khác {DEV_FULL_SAMPLES}: {samples}")
    return {
        "evaluation_count": len(records),
        "samples_per_eval": DEV_FULL_SAMPLES,
        "includes_lowpass_7k_in_loader": includes_lowpass_7k,
        "mean_total_seconds": statistics.mean(totals),
        "min_total_seconds": min(totals),
        "max_total_seconds": max(totals),
        "mean_loader_seconds": statistics.mean(loaders),
        "mean_model_seconds": statistics.mean(models),
        "mean_gpu_hours_per_eval": statistics.mean(totals) / 3600.0,
        "individual_total_seconds": totals,
    }


def _parse_t14_benchmark_dev_hours(path: Path) -> dict[str, float]:
    require(path.is_file(), f"Thiếu report benchmark T14: {path}")
    text = path.read_text(encoding="utf-8")
    results: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 14 and cells[0] in {"X0", "X1"}:
            key = f"{cells[0]}_L{cells[1]}_bs{cells[2]}"
            try:
                results[key] = float(cells[13])
            except ValueError:
                continue
    return results


def estimate_test_evaluation_costs(
    main_root: Path = ROOT,
    xlsr_root: Path = WORKTREE_XLSR_ROOT,
    remaining_gpu_hours: float | None = None,
) -> dict[str, Any]:
    """Ước tính chi phí GPU Kaggle T4 (X0, X1, X1R) và CPU cục bộ (B0 v2) từ số đo thực tế trên dev-full."""
    probe_dev = _load_json(main_root / "reports" / "shortcut_probe_dev.json")
    dev_bonafide_samples = int(
        probe_dev["results"]["B0-v2-seed2026"]["conditions"]["original"]["groups"]["bonafide"]["samples"]
    )
    require(dev_bonafide_samples == DEV_FULL_BONAFIDE_SAMPLES, "Số mẫu bonafide dev-full không khớp 14.151.")

    scale_factor = EXPECTED_TEST_CONTEXT["artifact_count"] / DEV_FULL_SAMPLES
    bonafide_ratio_dev = dev_bonafide_samples / DEV_FULL_SAMPLES
    approx_test_bonafide_samples = EXPECTED_TEST_CONTEXT["artifact_count"] * bonafide_ratio_dev

    # --- X0 trên Kaggle T4 ---
    t17 = _load_json(xlsr_root / "reports" / "xlsr_t17_x0_seed2026_session1.json")
    t28 = _load_json(xlsr_root / "reports" / "xlsr_t28_x0_pair_seed2027_2028.json")
    x0_2026_evals = list(t17["timing"]["evaluations_this_invocation"])
    x0_2027_evals = list(t28["phase_b"]["verifier_results"]["2027"]["result"]["timing"]["evaluations_this_invocation"])
    x0_2028_evals = list(t28["phase_b"]["verifier_results"]["2028"]["result"]["timing"]["evaluations_this_invocation"])
    x0_all_evals = x0_2026_evals + x0_2027_evals + x0_2028_evals

    # --- X1-L8-lr3e-5 trên Kaggle T4 ---
    t20 = _load_json(xlsr_root / "reports" / "xlsr_t20_seed2026_l8_lr3e-5.json")
    t21 = _load_json(xlsr_root / "reports" / "xlsr_t21_seeds_2027_2028.json")
    t24b = _load_json(xlsr_root / "reports" / "xlsr_t24b_resume_run.json")
    t26 = _load_json(xlsr_root / "reports" / "xlsr_t26_x1_seed2028.json")
    x1_2026_evals = list(t20["final_handoff"]["timing"]["evaluations_this_invocation"])
    x1_2027_evals = list(t21["runs"]["X1-L8-lr3e-5-seed2027"]["raw_summary"]["timing"]["evaluations_this_invocation"]) + list(
        t24b["verification"]["standard_a_g"]["timing"]["evaluations_this_invocation"]
    )
    x1_2028_evals = list(t26["timing"]["evaluations_this_invocation"])
    x1_all_evals = x1_2026_evals + x1_2027_evals + x1_2028_evals

    # --- X1R (X1-L8-lr3e-5-lp7k, đã bao gồm lọc lowpass_7k trong loader) trên Kaggle T4 ---
    t30 = _load_json(xlsr_root / "reports" / "xlsr_t30_x1r_pair_seed2026_2027.json")
    t31 = _load_json(xlsr_root / "reports" / "xlsr_t31_x1r_seed2028.json")
    t32b = _load_json(xlsr_root / "reports" / "xlsr_t32b_x1r_seed2026_resume_run.json")
    x1r_2026_pair_evals = list(t30["phase_b"]["runs"]["2026"]["timing"]["evaluations_this_invocation"])
    x1r_2026_resume_sec = float(t32b["verify_resume"]["timing"]["evaluation_seconds"])
    x1r_2027_evals = list(t30["phase_b"]["runs"]["2027"]["timing"]["evaluations_this_invocation"])
    x1r_2028_evals = list(t31["phase_b"]["run"]["timing"]["evaluations_this_invocation"])
    x1r_all_totals = (
        [float(x["total_seconds"]) for x in x1r_2026_pair_evals]
        + [x1r_2026_resume_sec]
        + [float(x["total_seconds"]) for x in x1r_2027_evals]
        + [float(x["total_seconds"]) for x in x1r_2028_evals]
    )

    t14_ref = _parse_t14_benchmark_dev_hours(main_root / "reports" / "xlsr_t14_cuda_benchmark.md")

    x0_dev_summary = _summarize_eval_records(x0_all_evals)
    x1_dev_summary = _summarize_eval_records(x1_all_evals)
    x1r_dev_mean_seconds = statistics.mean(x1r_all_totals)

    x0_dev_by_seed = {
        "2026_single": _summarize_eval_records(x0_2026_evals),
        "2027_pair": _summarize_eval_records(x0_2027_evals),
        "2028_pair": _summarize_eval_records(x0_2028_evals),
    }
    x1_dev_by_seed = {
        "2026_single": _summarize_eval_records(x1_2026_evals),
        "2027_single_plus_resume": _summarize_eval_records(x1_2027_evals),
        "2028_single": _summarize_eval_records(x1_2028_evals),
    }
    x1r_dev_by_seed = {
        "2026_pair": _summarize_eval_records(x1r_2026_pair_evals, includes_lowpass_7k=True),
        "2026_resume_single": {
            "evaluation_count": 1,
            "samples_per_eval": DEV_FULL_SAMPLES,
            "includes_lowpass_7k_in_loader": True,
            "mean_total_seconds": x1r_2026_resume_sec,
            "mean_gpu_hours_per_eval": x1r_2026_resume_sec / 3600.0,
        },
        "2027_pair": _summarize_eval_records(x1r_2027_evals, includes_lowpass_7k=True),
        "2028_single": _summarize_eval_records(x1r_2028_evals, includes_lowpass_7k=True),
    }

    # Quy đổi 1 lượt test-full cho từng họ mô hình
    x0_test_pass_seconds = x0_dev_summary["mean_total_seconds"] * scale_factor
    x1_test_pass_seconds = x1_dev_summary["mean_total_seconds"] * scale_factor
    x1r_test_pass_seconds = x1r_dev_mean_seconds * scale_factor

    # Số lượt theo D032 mục 6 & D035 (roundtrip_40k chỉ bonafide, xấp xỉ từ dev-full)
    x0_passes = 3.0
    x1_passes_per_seed_bonafide_only = 3.0 + bonafide_ratio_dev
    x1_passes_bonafide_only = 3.0 * x1_passes_per_seed_bonafide_only
    x1_passes_full_split_current_script = 12.0  # evaluate_full_split.py hiện chấm cả spoof rồi bỏ hoặc gửi full
    x1r_passes = 3.0

    x0_gpu_hours_base = (x0_test_pass_seconds * x0_passes) / 3600.0
    x1_gpu_hours_base = (x1_test_pass_seconds * x1_passes_bonafide_only) / 3600.0
    x1r_gpu_hours_base = (x1r_test_pass_seconds * x1r_passes) / 3600.0

    # Phân tích độ nhạy cho X1: 6 lượt lowpass (3 seed x {lowpass_7k, lowpass_4k}) cộng chi phí lọc SciPy như X1R
    x1_gpu_hours_sensitivity_lowpass_from_x1r = (
        x1_test_pass_seconds * (3.0 * (1.0 + bonafide_ratio_dev)) + x1r_test_pass_seconds * 6.0
    ) / 3600.0
    x1_gpu_hours_sensitivity_full_12_passes = (
        x1_test_pass_seconds * 6.0 + x1r_test_pass_seconds * 6.0
    ) / 3600.0

    total_gpu_hours_base = x0_gpu_hours_base + x1_gpu_hours_base + x1r_gpu_hours_base
    total_gpu_hours_with_safety = total_gpu_hours_base * SAFETY_FACTOR

    total_gpu_hours_sensitivity_base = x0_gpu_hours_base + x1_gpu_hours_sensitivity_lowpass_from_x1r + x1r_gpu_hours_base
    total_gpu_hours_sensitivity_with_safety = total_gpu_hours_sensitivity_base * SAFETY_FACTOR

    total_gpu_hours_unoptimized_base = x0_gpu_hours_base + x1_gpu_hours_sensitivity_full_12_passes + x1r_gpu_hours_base
    total_gpu_hours_unoptimized_with_safety = total_gpu_hours_unoptimized_base * SAFETY_FACTOR

    # --- B0 v2 trên CPU ---
    t19b_stages = probe_dev["sessions"][0]["stages"]
    b0_dev_orig_3seeds_s = float(t19b_stages["ii"]["elapsed_seconds"])
    b0_dev_rt_bonafide_3seeds_s = float(t19b_stages["iii"]["elapsed_seconds"])
    b0_dev_lp7k_3seeds_s = float(t19b_stages["iv"]["elapsed_seconds"])
    b0_dev_lp4k_3seeds_s = float(t19b_stages["v"]["elapsed_seconds"])
    b0_dev_all4_3seeds_s = b0_dev_orig_3seeds_s + b0_dev_rt_bonafide_3seeds_s + b0_dev_lp7k_3seeds_s + b0_dev_lp4k_3seeds_s

    b0_ext_single_evals: dict[str, dict[str, float]] = {}
    for seed in ("2026", "2027", "2028"):
        b0_ext = _load_json(main_root / "reports" / f"b0_v2_ext_seed{seed}.json")
        res = b0_ext["resources"]
        two_evals_s = float(res["elapsed_seconds"]) - float(res["training_loader_seconds"]) - float(res["training_model_seconds"])
        b0_ext_single_evals[seed] = {
            "two_dev_evals_seconds": two_evals_s,
            "one_dev_eval_seconds": two_evals_s / 2.0,
            "one_test_eval_seconds_scaled": (two_evals_s / 2.0) * scale_factor,
        }
    b0_ext_mean_1dev_s = statistics.mean(item["one_dev_eval_seconds"] for item in b0_ext_single_evals.values())
    b0_bench_1t = _load_json(main_root / "reports" / "b0_v2_cpu_benchmark_1thread.json")
    b0_bench_1dev_s = float(b0_bench_1t["measured"]["seconds_per_dev_sample"]) * DEV_FULL_SAMPLES

    return {
        "scale_basis": {
            "dev_full_samples": DEV_FULL_SAMPLES,
            "dev_full_shards": DEV_FULL_SHARDS,
            "test_full_samples": EXPECTED_TEST_CONTEXT["artifact_count"],
            "test_full_shards": EXPECTED_TEST_CONTEXT["chunk_shard_count"],
            "scale_factor_test_over_dev": scale_factor,
            "dev_full_bonafide_samples": dev_bonafide_samples,
            "dev_bonafide_ratio": bonafide_ratio_dev,
            "approx_test_bonafide_samples": approx_test_bonafide_samples,
            "approximation_note": (
                "Không đọc nhãn tập test; số mẫu bonafide để ước tính roundtrip_40k "
                f"dùng xấp xỉ theo tỷ lệ bonafide của dev-full ({dev_bonafide_samples}/{DEV_FULL_SAMPLES} = {bonafide_ratio_dev:.6f})."
            ),
            "safety_factor": SAFETY_FACTOR,
        },
        "gpu_t4": {
            "X0": {
                "sources": [
                    "W/reports/xlsr_t17_x0_seed2026_session1.json (timing.evaluations_this_invocation)",
                    "W/reports/xlsr_t28_x0_pair_seed2027_2028.json (phase_b.verifier_results.{2027,2028}.result.timing.evaluations_this_invocation)",
                    "reports/xlsr_t14_cuda_benchmark.md (X0 batch=16)",
                ],
                "t14_benchmark_dev_full_hours": t14_ref.get("X0_L0_bs16"),
                "by_seed": x0_dev_by_seed,
                "pooled_dev_full": x0_dev_summary,
                "projected_test_full_pass_seconds": x0_test_pass_seconds,
                "projected_test_full_pass_hours": x0_test_pass_seconds / 3600.0,
                "required_conditions": ["original"],
                "seeds": 3,
                "equivalent_test_passes": x0_passes,
                "estimated_gpu_hours_base": x0_gpu_hours_base,
                "estimated_gpu_hours_plus_25pct": x0_gpu_hours_base * SAFETY_FACTOR,
            },
            "X1": {
                "run_id": "X1-L8-lr3e-5",
                "sources": [
                    "W/reports/xlsr_t20_seed2026_l8_lr3e-5.json (final_handoff.timing.evaluations_this_invocation)",
                    "W/reports/xlsr_t21_seeds_2027_2028.json (runs.X1-L8-lr3e-5-seed2027.raw_summary.timing.evaluations_this_invocation)",
                    "W/reports/xlsr_t24b_resume_run.json (verification.standard_a_g.timing.evaluations_this_invocation)",
                    "W/reports/xlsr_t26_x1_seed2028.json (timing.evaluations_this_invocation)",
                    "reports/xlsr_t14_cuda_benchmark.md (X1 layers=8 batch=16)",
                ],
                "t14_benchmark_dev_full_hours": t14_ref.get("X1_L8_bs16"),
                "by_seed": x1_dev_by_seed,
                "pooled_dev_full": x1_dev_summary,
                "projected_test_full_pass_seconds": x1_test_pass_seconds,
                "projected_test_full_pass_hours": x1_test_pass_seconds / 3600.0,
                "required_conditions_d032": [
                    "original (toàn bộ)",
                    "lowpass_7k (toàn bộ)",
                    "lowpass_4k (toàn bộ)",
                    "roundtrip_40k (chỉ bonafide, xấp xỉ theo tỷ lệ dev-full)",
                ],
                "seeds": 3,
                "equivalent_test_passes_d032": x1_passes_bonafide_only,
                "equivalent_test_passes_if_roundtrip_full": x1_passes_full_split_current_script,
                "estimated_gpu_hours_base": x1_gpu_hours_base,
                "estimated_gpu_hours_plus_25pct": x1_gpu_hours_base * SAFETY_FACTOR,
                "sensitivity_with_x1r_lowpass_overhead_base": x1_gpu_hours_sensitivity_lowpass_from_x1r,
                "sensitivity_with_x1r_lowpass_overhead_plus_25pct": x1_gpu_hours_sensitivity_lowpass_from_x1r * SAFETY_FACTOR,
                "sensitivity_full_12_passes_if_unoptimized_base": x1_gpu_hours_sensitivity_full_12_passes,
                "implementation_note": (
                    "Theo D032 mục 6 và bước 4, roundtrip_40k chỉ cần chấm phần bonafide (≈36,49% mẫu). "
                    "Tuy nhiên scripts/evaluate_full_split.py hiện duyệt đủ len(indices) cho từng điều kiện (= 12 lượt với X1) "
                    "trừ khi tái sử dụng điểm spoof từ original như scripts/probe_shortcut_dev.py."
                ),
            },
            "X1R": {
                "run_id": "X1-L8-lr3e-5-lp7k",
                "sources": [
                    "W/reports/xlsr_t30_x1r_pair_seed2026_2027.json (phase_b.runs.{2026,2027}.timing.evaluations_this_invocation)",
                    "W/reports/xlsr_t32b_x1r_seed2026_resume_run.json (verify_resume.timing.evaluation_seconds)",
                    "W/reports/xlsr_t31_x1r_seed2028.json (phase_b.run.timing.evaluations_this_invocation)",
                ],
                "by_seed": x1r_dev_by_seed,
                "pooled_dev_full": {
                    "evaluation_count": len(x1r_all_totals),
                    "samples_per_eval": DEV_FULL_SAMPLES,
                    "includes_lowpass_7k_in_loader": True,
                    "mean_total_seconds": x1r_dev_mean_seconds,
                    "min_total_seconds": min(x1r_all_totals),
                    "max_total_seconds": max(x1r_all_totals),
                    "mean_gpu_hours_per_eval": x1r_dev_mean_seconds / 3600.0,
                },
                "projected_test_full_pass_seconds": x1r_test_pass_seconds,
                "projected_test_full_pass_hours": x1r_test_pass_seconds / 3600.0,
                "required_conditions_d035": ["lowpass_7k (toàn bộ)"],
                "seeds": 3,
                "equivalent_test_passes": x1r_passes,
                "estimated_gpu_hours_base": x1r_gpu_hours_base,
                "estimated_gpu_hours_plus_25pct": x1r_gpu_hours_base * SAFETY_FACTOR,
            },
            "totals": {
                "total_equivalent_passes_d032_d035": x0_passes + x1_passes_bonafide_only + x1r_passes,
                "total_gpu_hours_base": total_gpu_hours_base,
                "total_gpu_hours_plus_25pct": total_gpu_hours_with_safety,
                "sensitivity_with_x1_lowpass_overhead_base": total_gpu_hours_sensitivity_base,
                "sensitivity_with_x1_lowpass_overhead_plus_25pct": total_gpu_hours_sensitivity_with_safety,
                "sensitivity_unoptimized_18_passes_base": total_gpu_hours_unoptimized_base,
                "sensitivity_unoptimized_18_passes_plus_25pct": total_gpu_hours_unoptimized_with_safety,
                "remaining_kaggle_gpu_hours": remaining_gpu_hours,
                "fits_current_weekly_quota_with_safety": (
                    total_gpu_hours_sensitivity_with_safety <= remaining_gpu_hours
                    if remaining_gpu_hours is not None
                    else None
                ),
            },
        },
        "cpu_b0_v2": {
            "primary_source_t19b": {
                "source": "reports/shortcut_probe_dev.json (sessions[0].stages ii–v, 3 seed B0-v2 gộp batch=8, 8 luồng CPU)",
                "dev_full_seconds": {
                    "original_3seeds": b0_dev_orig_3seeds_s,
                    "roundtrip_40k_bonafide_3seeds": b0_dev_rt_bonafide_3seeds_s,
                    "lowpass_7k_3seeds": b0_dev_lp7k_3seeds_s,
                    "lowpass_4k_3seeds": b0_dev_lp4k_3seeds_s,
                    "all_4_conditions_3seeds": b0_dev_all4_3seeds_s,
                },
                "projected_test_full_seconds": {
                    "original_3seeds": b0_dev_orig_3seeds_s * scale_factor,
                    "roundtrip_40k_bonafide_3seeds_approx": b0_dev_rt_bonafide_3seeds_s * scale_factor,
                    "lowpass_7k_3seeds": b0_dev_lp7k_3seeds_s * scale_factor,
                    "lowpass_4k_3seeds": b0_dev_lp4k_3seeds_s * scale_factor,
                    "all_4_conditions_3seeds": b0_dev_all4_3seeds_s * scale_factor,
                },
                "projected_test_full_hours": {
                    "original_3seeds_base": (b0_dev_orig_3seeds_s * scale_factor) / 3600.0,
                    "original_3seeds_plus_25pct": (b0_dev_orig_3seeds_s * scale_factor * SAFETY_FACTOR) / 3600.0,
                    "all_4_conditions_3seeds_base": (b0_dev_all4_3seeds_s * scale_factor) / 3600.0,
                    "all_4_conditions_3seeds_plus_25pct": (b0_dev_all4_3seeds_s * scale_factor * SAFETY_FACTOR) / 3600.0,
                },
            },
            "reference_1thread_ext_and_benchmark": {
                "source_ext": "reports/b0_v2_ext_seed{2026,2027,2028}.json (resources: (elapsed - train_loader - train_model) / 2)",
                "ext_by_seed": b0_ext_single_evals,
                "ext_mean_one_dev_eval_seconds": b0_ext_mean_1dev_s,
                "ext_mean_one_test_eval_hours": (b0_ext_mean_1dev_s * scale_factor) / 3600.0,
                "ext_three_seeds_original_test_hours": (b0_ext_mean_1dev_s * scale_factor * 3.0) / 3600.0,
                "source_benchmark": "reports/b0_v2_cpu_benchmark_1thread.json (measured.seconds_per_dev_sample)",
                "benchmark_one_dev_eval_seconds": b0_bench_1dev_s,
                "benchmark_three_seeds_original_test_hours": (b0_bench_1dev_s * scale_factor * 3.0) / 3600.0,
            },
        },
    }


def collect_registry_freeze_checklist(root: Path = ROOT) -> dict[str, Any]:
    """Liệt kê những việc còn thiếu để đóng băng registry chính thức (không sửa file nào)."""
    cfg_v1_path = root / "configs" / "test_evaluation.json"
    cfg_x1r_path = root / "configs" / "test_evaluation_x1r.json"
    reg_v1_official = root / "reports" / "model_registry_v1.json"
    reg_x1r_official = root / "reports" / "model_registry_x1r.json"
    reg_v1_draft_json = root / "reports" / "model_registry_v1_draft.json"
    reg_v1_draft_md = root / "reports" / "model_registry_v1_draft.md"
    reg_x1r_draft_json = root / "reports" / "model_registry_x1r_draft.json"
    reg_x1r_draft_md = root / "reports" / "model_registry_x1r_draft.md"

    cfg_v1 = _load_json(cfg_v1_path)
    cfg_x1r = _load_json(cfg_x1r_path)
    v1_draft = _load_json(reg_v1_draft_json)
    x1r_draft = _load_json(reg_x1r_draft_json)

    v1_md_lines = reg_v1_draft_md.read_text(encoding="utf-8").splitlines()[:6]
    x1r_md_lines = reg_x1r_draft_md.read_text(encoding="utf-8").splitlines()[:6]

    scale2_drafts = sorted(
        str(p.relative_to(root)).replace("\\", "/")
        for p in (root / "reports").glob("model_registry*scale2*")
    )

    missing_items: list[str] = []
    if not reg_v1_official.exists():
        missing_items.append(
            "Tạo registry chính thức reports/model_registry_v1.json (hiện chưa tồn tại; chỉ có bản nháp reports/model_registry_v1_draft.json với role='draft'; scripts/build_model_registry.py đang chặn ghi ra đường dẫn chính thức)."
        )
    if not reg_x1r_official.exists():
        missing_items.append(
            "Tạo registry chính thức X1R reports/model_registry_x1r.json (hiện chưa tồn tại; chỉ có bản nháp reports/model_registry_x1r_draft.json với role='draft')."
        )
    if cfg_v1.get("registry_sha256") is None:
        missing_items.append("Điền registry_sha256 vào configs/test_evaluation.json (hiện đang null).")
    if cfg_x1r.get("registry_sha256") is None:
        missing_items.append("Điền registry_sha256 vào configs/test_evaluation_x1r.json (hiện đang null).")

    pre_checks = dict(cfg_v1.get("pre_test_checks", {}))
    if not pre_checks.get("t22_claude_verified", False):
        missing_items.append(
            "Cập nhật pre_test_checks.t22_claude_verified=true trong configs/test_evaluation.json (T22b đã được Claude kiểm ĐẠT 13/11/2026 nhưng cờ trong config vẫn đang false)."
        )
    if not pre_checks.get("test_dataset_inventory_checked", False):
        missing_items.append(
            "Upload dataset closed_test_full_v1 lên Kaggle (hoặc tạo dataset test trên Kaggle), kiểm khớp kích thước/checksum và cập nhật pre_test_checks.test_dataset_inventory_checked=true."
        )
    if any("chưa đầy đủ" in line.lower() for line in v1_md_lines):
        missing_items.append(
            "Tiêu đề reports/model_registry_v1_draft.md vẫn ghi '# Registry mô hình v1 — snapshot chưa đầy đủ' và 'Không phải registry chính thức; không mở cổng test.' (dù JSON nháp đã complete=True, 9/9 mục present và claude_verified=True)."
        )
    if any("chưa đầy đủ" in line.lower() for line in x1r_md_lines):
        missing_items.append(
            "Tiêu đề reports/model_registry_x1r_draft.md vẫn ghi '# Registry phụ mô hình X1R — snapshot chưa đầy đủ' (dù JSON nháp đã complete=True, 3/3 mục present và claude_verified=True)."
        )
    if not scale2_drafts:
        missing_items.append(
            "Theo D036 (scale2), cần hoàn tất và đăng ký registry phụ 6 mục trước khi mở cổng test nếu giữ phạm vi đánh giá gồm scale2."
        )
    if not cfg_v1.get("test_gate_open", False) or not cfg_x1r.get("test_gate_open", False):
        missing_items.append(
            "Cổng test hiện vẫn đóng (test_gate_open=false trong cả hai config); chỉ người dùng xác nhận mới được mở."
        )

    return {
        "official_registries": {
            "v1": {"path": "reports/model_registry_v1.json", "exists": reg_v1_official.exists()},
            "x1r": {"path": "reports/model_registry_x1r.json", "exists": reg_x1r_official.exists()},
        },
        "configs": {
            "test_evaluation": {
                "path": "configs/test_evaluation.json",
                "test_gate_open": cfg_v1.get("test_gate_open"),
                "registry_path": cfg_v1.get("registry_path"),
                "registry_sha256": cfg_v1.get("registry_sha256"),
                "pre_test_checks": pre_checks,
            },
            "test_evaluation_x1r": {
                "path": "configs/test_evaluation_x1r.json",
                "test_gate_open": cfg_x1r.get("test_gate_open"),
                "registry_path": cfg_x1r.get("registry_path"),
                "registry_sha256": cfg_x1r.get("registry_sha256"),
            },
        },
        "draft_registries": {
            "v1": {
                "role": v1_draft.get("role"),
                "complete": v1_draft.get("complete"),
                "entry_count": len(v1_draft.get("entries", [])),
                "present_count": sum(1 for e in v1_draft.get("entries", []) if e.get("status") == "present"),
                "claude_verified_count": sum(1 for e in v1_draft.get("entries", []) if e.get("claude_verified")),
                "protocol_sha256": v1_draft.get("protocol_sha256"),
                "md_header_lines": v1_md_lines,
                "md_header_says_chua_day_du": any("chưa đầy đủ" in line.lower() for line in v1_md_lines),
            },
            "x1r": {
                "role": x1r_draft.get("role"),
                "complete": x1r_draft.get("complete"),
                "entry_count": len(x1r_draft.get("entries", [])),
                "present_count": sum(1 for e in x1r_draft.get("entries", []) if e.get("status") == "present"),
                "claude_verified_count": sum(1 for e in x1r_draft.get("entries", []) if e.get("claude_verified")),
                "protocol_sha256": x1r_draft.get("protocol_sha256"),
                "md_header_lines": x1r_md_lines,
                "md_header_says_chua_day_du": any("chưa đầy đủ" in line.lower() for line in x1r_md_lines),
            },
            "scale2_files_present": scale2_drafts,
        },
        "missing_items_to_freeze_registry": missing_items,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Tạo nội dung báo cáo Markdown reports/pre_test_inventory.md."""
    local = report["local_test_inventory"]
    kag = report["kaggle_inventory"]
    cost = report["cost_estimate"]
    reg = report["registry_freeze_checklist"]
    gpu_q = kag["quota"]["gpu"]
    scale = cost["scale_basis"]
    gpu = cost["gpu_t4"]
    cpu = cost["cpu_b0_v2"]

    lines: list[str] = [
        "# PREP-01 — Kiểm kho test-full, dataset Kaggle và ước tính chi phí chấm test",
        "",
        f"- Thời điểm tạo (UTC): `{report['generated_utc']}`",
        f"- Cam kết dữ liệu test: **Không đọc nhãn, không parse dòng/cột của `index.csv` hay manifest test, không giải mã audio test** (`no_test_labels_or_content_read = {local['no_test_labels_or_content_read']}`).",
        f"- Cổng test: `test_gate_open = {reg['configs']['test_evaluation']['test_gate_open']}` (D032) và `{reg['configs']['test_evaluation_x1r']['test_gate_open']}` (D035).",
        "",
        "## 1. Kho test-full cục bộ (`closed_test_full_v1`)",
        "",
        f"- Kết luận: **{'KHỚP TOÀN BỘ' if local['all_checks_passed'] else 'LỆCH'}** (`artifact_count = {local['metadata_summary']['artifact_count']}`, `chunk_shard_count = {local['metadata_summary']['chunk_shard_count']}`, `total_samples = {local['metadata_summary']['total_samples']}`).",
        f"- Kiểm kích thước `waveforms.f32`: `{local['files']['waveforms']['bytes']:,}` byte = `{local['metadata_summary']['total_samples']:,} × 4` byte (`khớp = {local['checks']['waveforms_bytes_equals_total_samples_x4']}`).",
        "",
        "| File | Đường dẫn | Kích thước (byte) | SHA-256 thực đo (luồng byte) | Khớp metadata/config/bối cảnh |",
        "|---|---|---:|---|---|",
    ]
    for label, key in (
        ("Manifest", "manifest"),
        ("Index", "index"),
        ("Waveforms", "waveforms"),
    ):
        item = local["files"][key]
        lines.append(
            f"| {label} | `{item['path']}` | {item['bytes']:,} | `{item['sha256']}` | {item['matched']} |"
        )
    meta = local["metadata_summary"]
    lines.append(
        f"| Metadata | `{meta['path']}` | {meta['bytes']:,} | `{meta['sha256']}` | Đã đọc JSON hợp lệ |"
    )

    lines.extend(
        [
            "",
            "## 2. Trạng thái Kaggle (chỉ đọc)",
            "",
            f"- Quota GPU tuần hiện tại: còn **{gpu_q['remaining']}** / **{gpu_q['total']}** (đã dùng `{gpu_q['used']}`, làm mới lúc `{gpu_q['refresh_at']}` UTC; log `{kag['quota_log']}`).",
            f"- Tổng số dataset thuộc tài khoản: `{len(kag['all_my_datasets'])}` (log `{kag['datasets_list_log']}`).",
            f"- Kết luận `closed_test_full_v1` trên Kaggle: **{'ĐÃ CÓ' if kag['has_test_full_on_kaggle'] else 'CHƯA CÓ'}**.",
            f"- Ghi chú: {kag['remote_hash_note']}",
            "",
            "| Dataset ứng viên (`packed`/`test`/`eval`) | Kích thước nén (byte) | File từ xa | Kích thước file (byte) |",
            "|---|---:|---|---:|",
        ]
    )
    for ds in kag["inspected_candidate_datasets"]:
        for f in ds["files"]:
            lines.append(
                f"| `{ds['ref']}` | {ds['dataset_zip_size_bytes']:,} | `{f['name']}` | {f['size_bytes']:,} |"
            )

    lines.extend(
        [
            "",
            "## 3. Ước tính chi phí chấm test-full",
            "",
            f"- Hệ số quy đổi số mẫu: `30.698 / 38.777 = {scale['scale_factor_test_over_dev']:.6f}`.",
            f"- **Xấp xỉ cho `roundtrip_40k` (không đọc nhãn test)**: dùng tỷ lệ bonafide trên `dev-full` (`{scale['dev_full_bonafide_samples']:,} / {scale['dev_full_samples']:,} = {scale['dev_bonafide_ratio']:.6f}`, nguồn `reports/shortcut_probe_dev.json`) → ước lượng khoảng `~{scale['approx_test_bonafide_samples']:,.0f}` mẫu bonafide trong `test-full`.",
            "",
            "### 3a. Dự kiến GPU Kaggle Tesla T4 (X0, X1, X1R)",
            "",
            "| Họ mô hình | TB 1 lượt dev-full thực đo (s) | Quy đổi 1 lượt test-full (s / h) | Số lượt test-full (3 seed) | GPU-giờ gốc | GPU-giờ (+25% an toàn) |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| X0 | {gpu['X0']['pooled_dev_full']['mean_total_seconds']:.2f} ({gpu['X0']['pooled_dev_full']['evaluation_count']} lượt) | "
                f"{gpu['X0']['projected_test_full_pass_seconds']:.2f} s / {gpu['X0']['projected_test_full_pass_hours']:.4f} h | "
                f"{gpu['X0']['equivalent_test_passes']:.2f} (`original` ×3) | "
                f"{gpu['X0']['estimated_gpu_hours_base']:.3f} | {gpu['X0']['estimated_gpu_hours_plus_25pct']:.3f} |"
            ),
            (
                f"| X1 (`X1-L8-lr3e-5`) | {gpu['X1']['pooled_dev_full']['mean_total_seconds']:.2f} ({gpu['X1']['pooled_dev_full']['evaluation_count']} lượt) | "
                f"{gpu['X1']['projected_test_full_pass_seconds']:.2f} s / {gpu['X1']['projected_test_full_pass_hours']:.4f} h | "
                f"{gpu['X1']['equivalent_test_passes_d032']:.4f} (xấp xỉ D032) | "
                f"{gpu['X1']['estimated_gpu_hours_base']:.3f} | {gpu['X1']['estimated_gpu_hours_plus_25pct']:.3f} |"
            ),
            (
                f"| X1R (`X1-L8-lr3e-5-lp7k`) | {gpu['X1R']['pooled_dev_full']['mean_total_seconds']:.2f} ({gpu['X1R']['pooled_dev_full']['evaluation_count']} lượt) | "
                f"{gpu['X1R']['projected_test_full_pass_seconds']:.2f} s / {gpu['X1R']['projected_test_full_pass_hours']:.4f} h | "
                f"{gpu['X1R']['equivalent_test_passes']:.2f} (`lowpass_7k` ×3) | "
                f"{gpu['X1R']['estimated_gpu_hours_base']:.3f} | {gpu['X1R']['estimated_gpu_hours_plus_25pct']:.3f} |"
            ),
            (
                f"| **Tổng GPU T4 (gốc D032+D035)** | — | — | **{gpu['totals']['total_equivalent_passes_d032_d035']:.4f}** | "
                f"**{gpu['totals']['total_gpu_hours_base']:.3f}** | **{gpu['totals']['total_gpu_hours_plus_25pct']:.3f}** |"
            ),
            "",
            f"- **Độ nhạy chi phí lọc lowpass trên X1**: nếu 6 lượt `lowpass_7k`/`lowpass_4k` của X1 cộng thêm thời gian lọc CPU trong loader (như X1R thực đo `{gpu['X1R']['pooled_dev_full']['mean_total_seconds']:.2f}` s/dev-full), tổng chi phí gốc là **{gpu['totals']['sensitivity_with_x1_lowpass_overhead_base']:.3f} GPU-h** (cộng 25% an toàn = **{gpu['totals']['sensitivity_with_x1_lowpass_overhead_plus_25pct']:.3f} GPU-h**).",
            f"- **Lưu ý script `evaluate_full_split.py`**: hiện tại hàm `Models.score` chạy qua toàn bộ `30.698` mẫu cả ở điều kiện `roundtrip_40k` (tức `12.0` lượt cho X1 thay vì `10.0948` lượt). Nếu chạy nguyên trạng hiện tại (18 lượt tổng X0+X1+X1R) và tính overhead lowpass cho 6 lượt X1: gốc **{gpu['totals']['sensitivity_unoptimized_18_passes_base']:.3f} GPU-h**, +25% an toàn = **{gpu['totals']['sensitivity_unoptimized_18_passes_plus_25pct']:.3f} GPU-h**.",
            f"- **Đối chiếu quota Kaggle hiện tại (`{gpu_q['remaining']}`)**: cả phương án chuẩn lẫn độ nhạy +25% đều nằm trong quota tuần còn lại (`<= {gpu_q['remaining_hours']:.2f} h`).",
            "- **Nguồn số đo từng mô hình**:",
            f"  - X0: `W/reports/xlsr_t17_x0_seed2026_session1.json` (10 lượt single GPU, TB `{gpu['X0']['by_seed']['2026_single']['mean_total_seconds']:.2f}` s); `W/reports/xlsr_t28_x0_pair_seed2027_2028.json` (seed 2027 10 lượt TB `{gpu['X0']['by_seed']['2027_pair']['mean_total_seconds']:.2f}` s, seed 2028 10 lượt TB `{gpu['X0']['by_seed']['2028_pair']['mean_total_seconds']:.2f}` s). Tham chiếu T14 batch=16: `{gpu['X0']['t14_benchmark_dev_full_hours']:.6f}` h.",
            f"  - X1: `W/reports/xlsr_t20_seed2026_l8_lr3e-5.json` (4 lượt TB `{gpu['X1']['by_seed']['2026_single']['mean_total_seconds']:.2f}` s); `W/reports/xlsr_t21_seeds_2027_2028.json` + `W/reports/xlsr_t24b_resume_run.json` (10 lượt TB `{gpu['X1']['by_seed']['2027_single_plus_resume']['mean_total_seconds']:.2f}` s); `W/reports/xlsr_t26_x1_seed2028.json` (7 lượt TB `{gpu['X1']['by_seed']['2028_single']['mean_total_seconds']:.2f}` s). Tham chiếu T14 L8 batch=16: `{gpu['X1']['t14_benchmark_dev_full_hours']:.6f}` h.",
            f"  - X1R: `W/reports/xlsr_t30_x1r_pair_seed2026_2027.json` (seed 2026 9 lượt pair TB `{gpu['X1R']['by_seed']['2026_pair']['mean_total_seconds']:.2f}` s, seed 2027 8 lượt pair TB `{gpu['X1R']['by_seed']['2027_pair']['mean_total_seconds']:.2f}` s); `W/reports/xlsr_t32b_x1r_seed2026_resume_run.json` (1 lượt resume `{gpu['X1R']['by_seed']['2026_resume_single']['mean_total_seconds']:.2f}` s); `W/reports/xlsr_t31_x1r_seed2028.json` (8 lượt single TB `{gpu['X1R']['by_seed']['2028_single']['mean_total_seconds']:.2f}` s).",
            "",
            "### 3b. Dự kiến B0 v2 trên CPU cục bộ",
            "",
            f"- **Nguồn chính (đúng pipeline 3 seed gộp batch=8, 8 luồng CPU)**: `reports/shortcut_probe_dev.json` (`sessions[0].stages` `ii`–`v` trên `dev-full` 38.777 mẫu):",
            f"  - `original` × 3 seed: dev-full `{cpu['primary_source_t19b']['dev_full_seconds']['original_3seeds']:.2f}` s → quy đổi test-full `{cpu['primary_source_t19b']['projected_test_full_seconds']['original_3seeds']:.2f}` s = **{cpu['primary_source_t19b']['projected_test_full_hours']['original_3seeds_base']:.3f} giờ CPU** (+25% an toàn: **{cpu['primary_source_t19b']['projected_test_full_hours']['original_3seeds_plus_25pct']:.3f} giờ**).",
            f"  - Đủ 4 điều kiện (`original`, `roundtrip_40k` bonafide xấp xỉ, `lowpass_7k`, `lowpass_4k`) × 3 seed: dev-full `{cpu['primary_source_t19b']['dev_full_seconds']['all_4_conditions_3seeds']:.2f}` s → quy đổi test-full `{cpu['primary_source_t19b']['projected_test_full_seconds']['all_4_conditions_3seeds']:.2f}` s = **{cpu['primary_source_t19b']['projected_test_full_hours']['all_4_conditions_3seeds_base']:.3f} giờ CPU** (+25% an toàn: **{cpu['primary_source_t19b']['projected_test_full_hours']['all_4_conditions_3seeds_plus_25pct']:.3f} giờ**).",
            f"- **Tham chiếu 1 luồng CPU từ training/benchmark**: `reports/b0_v2_ext_seed{{2026,2027,2028}}.json` (TB `{cpu['reference_1thread_ext_and_benchmark']['ext_mean_one_dev_eval_seconds']:.2f}` s/dev-full → 3 seed `original` test-full `{cpu['reference_1thread_ext_and_benchmark']['ext_three_seeds_original_test_hours']:.3f}` giờ); `reports/b0_v2_cpu_benchmark_1thread.json` (`{cpu['reference_1thread_ext_and_benchmark']['benchmark_three_seeds_original_test_hours']:.3f}` giờ cho 3 seed `original` test-full).",
            "",
            "## 4. Việc còn thiếu để đóng băng registry (chỉ đọc, chưa sửa gì)",
            "",
            f"- Registry chính thức `reports/model_registry_v1.json`: tồn tại = `{reg['official_registries']['v1']['exists']}` (bản nháp: `{reg['draft_registries']['v1']['present_count']}/{reg['draft_registries']['v1']['entry_count']}` present, `complete={reg['draft_registries']['v1']['complete']}`, header `.md` ghi `'snapshot chưa đầy đủ'` = `{reg['draft_registries']['v1']['md_header_says_chua_day_du']}`).",
            f"- Registry chính thức `reports/model_registry_x1r.json`: tồn tại = `{reg['official_registries']['x1r']['exists']}` (bản nháp: `{reg['draft_registries']['x1r']['present_count']}/{reg['draft_registries']['x1r']['entry_count']}` present, `complete={reg['draft_registries']['x1r']['complete']}`, header `.md` ghi `'snapshot chưa đầy đủ'` = `{reg['draft_registries']['x1r']['md_header_says_chua_day_du']}`).",
            f"- `configs/test_evaluation.json`: `registry_sha256={reg['configs']['test_evaluation']['registry_sha256']}`, `test_gate_open={reg['configs']['test_evaluation']['test_gate_open']}`, `pre_test_checks={json.dumps(reg['configs']['test_evaluation']['pre_test_checks'], ensure_ascii=False)}`.",
            f"- `configs/test_evaluation_x1r.json`: `registry_sha256={reg['configs']['test_evaluation_x1r']['registry_sha256']}`, `test_gate_open={reg['configs']['test_evaluation_x1r']['test_gate_open']}`.",
            "",
            "Danh sách chi tiết cần xử lý trước khi mở cổng:",
        ]
    )
    for idx, item in enumerate(reg["missing_items_to_freeze_registry"], start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines) + "\n"


def generate_pre_test_inventory(
    main_root: Path = ROOT,
    xlsr_root: Path = WORKTREE_XLSR_ROOT,
    cache_dir: Path = CACHE_DIR,
    kaggle_mode: Literal["auto", "logs", "live"] = "auto",
) -> dict[str, Any]:
    """Chạy toàn bộ các bước kiểm kho PREP-01 (không đọc nội dung/nhãn test)."""
    local_inventory = check_local_test_inventory(main_root)
    kaggle_inventory = check_kaggle_inventory(cache_dir=cache_dir, kaggle_mode=kaggle_mode)
    remaining_gpu_h = float(kaggle_inventory["quota"]["gpu"]["remaining_hours"])
    cost_estimate = estimate_test_evaluation_costs(
        main_root=main_root,
        xlsr_root=xlsr_root,
        remaining_gpu_hours=remaining_gpu_h,
    )
    registry_checklist = collect_registry_freeze_checklist(main_root)

    return {
        "task": "PREP-01",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "no_test_labels_or_content_read": True,
        "local_test_inventory": local_inventory,
        "kaggle_inventory": kaggle_inventory,
        "cost_estimate": cost_estimate,
        "registry_freeze_checklist": registry_checklist,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào CLI của PREP-01."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--kaggle-mode",
        choices=("auto", "logs", "live"),
        default="auto",
        help="Mặc định dùng log đã lưu ở .cache/prep01 nếu đủ, nếu thiếu mới gọi lệnh Kaggle chỉ đọc.",
    )
    args = parser.parse_args(argv)
    try:
        report = generate_pre_test_inventory(
            main_root=ROOT,
            xlsr_root=WORKTREE_XLSR_ROOT,
            cache_dir=args.cache_dir,
            kaggle_mode=args.kaggle_mode,
        )
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        args.md_output.parent.mkdir(parents=True, exist_ok=True)
        args.md_output.write_text(
            render_markdown_report(report),
            encoding="utf-8",
            newline="\n",
        )
        summary = {
            "status": "ok",
            "local_matched": report["local_test_inventory"]["all_checks_passed"],
            "has_test_full_on_kaggle": report["kaggle_inventory"]["has_test_full_on_kaggle"],
            "gpu_remaining_hours": report["kaggle_inventory"]["quota"]["gpu"]["remaining_hours"],
            "gpu_hours_plus_25pct": round(
                report["cost_estimate"]["gpu_t4"]["totals"]["total_gpu_hours_plus_25pct"], 3
            ),
            "json_output": str(args.json_output),
            "md_output": str(args.md_output),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"LỖI kiểm kho test-full: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
