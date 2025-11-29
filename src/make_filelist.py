import os
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_ROOT = PROJECT_ROOT / "data_raw"
META_DIR = PROJECT_ROOT / "meta"
FILELIST_PATH = META_DIR / "filelist.csv"
LABELS_PATH = META_DIR / "labels.txt"

def load_label_set():
    labels = []
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                labels.append(l)
    return set(labels)

def main():
    target_labels = load_label_set()

    rows = []
    for label_name in os.listdir(DATA_RAW_ROOT):
        class_dir = DATA_RAW_ROOT / label_name
        if not class_dir.is_dir():
            continue

        # chỉ lấy những folder trùng labels.txt
        if label_name not in target_labels:
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".mp4"):
                video_path = class_dir / fname
                rows.append({
                    "video_path": str(video_path.relative_to(PROJECT_ROOT)),
                    "label_name": label_name
                })

    META_DIR.mkdir(parents=True, exist_ok=True)
    with FILELIST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {FILELIST_PATH}")

if __name__ == "__main__":
    main()
