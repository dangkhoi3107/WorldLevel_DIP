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
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    keypoints = []

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # 1. Lấy toạ độ gốc (Wrist - điểm số 0)
        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

        for lm in hand_landmarks.landmark:
            # 2. Trừ đi toạ độ gốc để có toạ độ tương đối (Relative)
            # Giúp model nhận diện được dù tay ở góc trái hay phải màn hình
            relative_x = lm.x - wrist_x
            relative_y = lm.y - wrist_y
            relative_z = lm.z - wrist_z

            keypoints.extend([relative_x, relative_y, relative_z])

    if len(keypoints) == 0:
        keypoints = [0.0] * FEATURE_DIM

    return np.array(keypoints, dtype=np.float32)

def video_to_seq(video_path: str, seq_len: int = SEQ_LEN) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    frames = []
    # ... (đoạn đọc frame giữ nguyên) ...
    while True:
        ret, frame = cap.read()
        if not ret: break
        kp = extract_keypoints_from_frame(frame) # Gọi hàm đã sửa ở trên
        frames.append(kp)
    cap.release()

    if len(frames) == 0: return None

    frames = np.stack(frames, axis=0)  # (num_frames, FEATURE_DIM)
    num_frames = frames.shape[0]

    # CÁCH SỬA: Luôn luôn dùng Linear Interpolation để đưa về đúng seq_len
    # Bất kể video ngắn hay dài, nó sẽ được co giãn đều ra 32 frames
    if num_frames > 0:
        idxs = np.linspace(0, num_frames - 1, seq_len).astype(int)
        frames = frames[idxs]

    return frames  # Luôn luôn là (SEQ_LEN, FEATURE_DIM)

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
