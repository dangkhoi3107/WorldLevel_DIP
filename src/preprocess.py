import os
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ==== THAM SỐ CƠ BẢN ====
SEQ_LEN = 32        # số frame cố định mỗi video
FEATURE_DIM = 63    # 21 điểm tay * 3 (x,y,z) cho 1 tay
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_ROOT = PROJECT_ROOT / "data_raw"
DATA_NPY_ROOT = PROJECT_ROOT / "data_npy"
META_DIR = PROJECT_ROOT / "meta"
FILELIST_PATH = META_DIR / "filelist.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints_from_frame(image_bgr: np.ndarray) -> np.ndarray:
    """Nhận 1 frame BGR → trả về vector (FEATURE_DIM,) chứa keypoints bàn tay."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    keypoints = []

    if result.multi_hand_landmarks:
        # lấy tay đầu tiên
        hand_landmarks = result.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])

    if len(keypoints) == 0:
        # không thấy tay → vector 0
        keypoints = [0.0] * FEATURE_DIM

    return np.array(keypoints, dtype=np.float32)

def video_to_seq(video_path: str, seq_len: int = SEQ_LEN) -> np.ndarray | None:
    """Đọc 1 video → trả về tensor (SEQ_LEN, FEATURE_DIM)."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extract_keypoints_from_frame(frame)
        frames.append(kp)

    cap.release()

    if len(frames) == 0:
        print(f"No frames extracted from: {video_path}")
        return None

    frames = np.stack(frames, axis=0)  # (num_frames, FEATURE_DIM)
    num_frames = frames.shape[0]

    if num_frames >= seq_len:
        idxs = np.linspace(0, num_frames - 1, seq_len).astype(int)
        frames = frames[idxs]
    else:
        pad_len = seq_len - num_frames
        pad = np.zeros((pad_len, FEATURE_DIM), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)

    return frames  # (SEQ_LEN, FEATURE_DIM)

def preprocess_all(filelist_path: Path = FILELIST_PATH, out_root: Path = DATA_NPY_ROOT):
    out_root.mkdir(parents=True, exist_ok=True)

    with filelist_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Preprocessing {len(rows)} videos from {filelist_path} ...")

    for row in rows:
        rel_video_path = row["video_path"]
        label_name = row["label_name"]

        abs_video_path = PROJECT_ROOT / rel_video_path
        if not abs_video_path.exists():
            print(f"Video not found, skip: {abs_video_path}")
            continue

        seq = video_to_seq(str(abs_video_path))
        if seq is None:
            continue

        class_dir = out_root / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        base = os.path.splitext(os.path.basename(rel_video_path))[0]
        out_path = class_dir / f"{base}.npy"
        np.save(out_path, seq)

        print(f"Saved: {out_path}")

    print("=== DONE PREPROCESS ===")
    print("Data_npy at:", out_root)

# test nhanh 1 video
def debug_one_video():
    # lấy 1 video bất kỳ trong data_raw
    for label_name in os.listdir(DATA_RAW_ROOT):
        class_dir = DATA_RAW_ROOT / label_name
        if not class_dir.is_dir():
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".mp4"):
                video_path = class_dir / fname
                print("Test video:", video_path)
                seq = video_to_seq(str(video_path))
                if seq is not None:
                    print("seq shape:", seq.shape)  # kỳ vọng (SEQ_LEN, FEATURE_DIM)
                return

if __name__ == "__main__":
    # chạy test nhanh hoặc preprocess toàn bộ
    # debug_one_video()
    preprocess_all()
