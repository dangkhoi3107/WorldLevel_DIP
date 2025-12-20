import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ================= CONFIG =================
PROJECT_ROOT = Path("/home/dangkhoi/dev/Projects/WorldLevel_DIP")
FILELIST_PATH = PROJECT_ROOT / "meta/filelist.csv"
VIDEO_ROOT = PROJECT_ROOT / "data_raw/videos"
OUT_DIR = PROJECT_ROOT / "meta"

FRAME_STRIDE = 5          # chỉ check mỗi 5 frame cho nhanh
TWO_HAND_THRESHOLD = 5    # >=5 frame có 2 tay -> video 2-hand
# =========================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FILELIST_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

video_results = []
class_video_hand = defaultdict(list)

print("Scanning videos with MediaPipe Hands...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    video_rel = row["video_path"]
    label = row["label_name"]

    video_path = PROJECT_ROOT / video_rel
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        continue

    frame_idx = 0
    two_hand_frames = 0
    total_checked = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            if len(result.multi_hand_landmarks) >= 2:
                two_hand_frames += 1

        total_checked += 1
        frame_idx += 1

    cap.release()

    is_two_hand = two_hand_frames >= TWO_HAND_THRESHOLD

    video_results.append({
        "video_path": video_rel,
        "label_name": label,
        "two_hand_frames": two_hand_frames,
        "checked_frames": total_checked,
        "is_two_hand": is_two_hand
    })

    class_video_hand[label].append(is_two_hand)

hands.close()

# ================= SAVE VIDEO STATS =================
video_df = pd.DataFrame(video_results)
video_stats_path = OUT_DIR / "video_hand_stats.csv"
video_df.to_csv(video_stats_path, index=False)

print("Saved:", video_stats_path)

# ================= CLASS-LEVEL DECISION =================
only_1hand_classes = []
mixed_classes = []
twohand_classes = []

for cls, flags in class_video_hand.items():
    if all(not f for f in flags):
        only_1hand_classes.append(cls)
    elif all(f for f in flags):
        twohand_classes.append(cls)
    else:
        mixed_classes.append(cls)

# ================= SAVE RESULTS =================
(Path(OUT_DIR / "classes_only_1hand.txt")
 .write_text("\n".join(sorted(only_1hand_classes))))

(Path(OUT_DIR / "classes_twohand.txt")
 .write_text("\n".join(sorted(twohand_classes))))

(Path(OUT_DIR / "classes_mixed.txt")
 .write_text("\n".join(sorted(mixed_classes))))

print("Classes only 1-hand:", len(only_1hand_classes))
print("Classes 2-hand only:", len(twohand_classes))
print("Classes mixed:", len(mixed_classes))

# ================= CREATE NEW FILELIST =================
df_1hand = video_df[video_df["is_two_hand"] == False]
filelist_1hand_path = OUT_DIR / "filelist_1hand.csv"
df_1hand[["video_path", "label_name"]].to_csv(filelist_1hand_path, index=False)

print("Saved:", filelist_1hand_path)
