import os
import json
import shutil
from pathlib import Path

# ====== SỬA CHO ĐÚNG ĐƯỜNG DẪN KAGGLE_ROOT CỦA BẠN ======
KAGGLE_ROOT = "/home/dangkhoi/dev/Projects/WorldLevel_DIP/data_raw"
# =========================================================

# Tính project root = folder cha của thư mục src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_ROOT = PROJECT_ROOT / "data_raw"
META_DIR = PROJECT_ROOT / "meta"
LABELS_PATH = META_DIR / "labels.txt"
JSON_PATH = Path(KAGGLE_ROOT) / "WLASL_v0.3.json"
VIDEOS_DIR = Path(KAGGLE_ROOT) / "videos"


def load_label_map(labels_path: Path):
    """Đọc meta/labels.txt → trả về dict {gloss: index}"""
    labels = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return {name: i for i, name in enumerate(labels)}


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("KAGGLE_ROOT :", KAGGLE_ROOT)
    print("DATA_RAW_ROOT:", DATA_RAW_ROOT)

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {JSON_PATH}")

    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy folder videos: {VIDEOS_DIR}")

    # Đọc danh sách class bạn đã chọn trong meta/labels.txt
    label_map = load_label_map(LABELS_PATH)
    target_glosses = set(label_map.keys())
    print("Target glosses (labels.txt):", target_glosses)

    # Đọc WLASL_v0.3.json
    with JSON_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # WLASL_v0.3.json: list các entry, mỗi entry có 'gloss' và 'instances'
    copied_count = 0
    missing_videos = 0

    os.makedirs(DATA_RAW_ROOT, exist_ok=True)

    for entry in meta:
        gloss = entry.get("gloss")
        if gloss not in target_glosses:
            continue  # bỏ qua những từ không có trong labels.txt

        instances = entry.get("instances", [])
        class_dir = DATA_RAW_ROOT / gloss
        class_dir.mkdir(parents=True, exist_ok=True)

        for inst in instances:
            # theo mô tả dataset: mỗi instance có 'video_id'
            video_id = inst.get("video_id")
            if not video_id:
                continue

            # file video trong folder 'videos' tên dạng '<video_id>.mp4'
            src = VIDEOS_DIR / f"{video_id}.mp4"
            if not src.exists():
                # không phải video nào trong JSON cũng tồn tại trong videos/
                missing_videos += 1
                # print(f"WARNING: missing video file: {src}")
                continue

            dst = class_dir / src.name
            if not dst.exists():  # tránh copy trùng
                shutil.copy2(src, dst)
                copied_count += 1
                print(f"Copied: {src} -> {dst}")

    print("=== DONE ===")
    print("Total videos copied :", copied_count)
    print("Missing video files :", missing_videos)
    print("Data organized under:", DATA_RAW_ROOT)


if __name__ == "__main__":
    main()
