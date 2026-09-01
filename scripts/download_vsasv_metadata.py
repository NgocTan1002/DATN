import csv
from pathlib import Path

from datasets import load_dataset

REPO_ID = "hustep-lab/VSASV-Dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "metadata" / "vsasv_metadata.csv"
COLUMNS = ["file", "label", "utt_type"]

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Từ chối ghi đè metadata hiện có: {OUTPUT_FILE}. "
        "Hãy đổi tên hoặc sao lưu file trước khi tải lại."
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Đang kết nối tới VSASV...")

dataset = load_dataset(
    REPO_ID,
    split="train",
    streaming=True,
    columns=COLUMNS,
    token=True,
)

print("Đang xuất metadata...")

count = 0

with open(
    OUTPUT_FILE,
    mode="w",
    newline="",
    encoding="utf-8-sig",
) as output:
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()

    for count, row in enumerate(dataset, start=1):
        writer.writerow({
            "file": row.get("file"),
            "label": row.get("label"),
            "utt_type": row.get("utt_type"),
        })

        if count % 10_000 == 0:
            print(f"Đã xử lý {count:,} mẫu")

print(f"Hoàn thành: {count:,} mẫu")
print(f"Metadata được lưu tại: {OUTPUT_FILE}")
