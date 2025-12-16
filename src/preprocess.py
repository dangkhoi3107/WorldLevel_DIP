import os
import csv
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

# ==== THAM SỐ CƠ BẢN ====
SEQ_LEN = 32        # Model GRU cần đầu vào cố định
FEATURE_DIM = 63    # 21 * 3
# =========================

# SỬA LỖI PATH (Để chạy được cả script lẫn notebook)
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_RAW_ROOT = PROJECT_ROOT / "data_raw"
DATA_NPY_ROOT = PROJECT_ROOT / "data_npy"
META_DIR = PROJECT_ROOT / "meta"
FILELIST_PATH = META_DIR / "filelist.csv"

# Khởi tạo module (nhưng chưa khởi tạo object Hands ở đây)
mp_hands = mp.solutions.hands

def extract_keypoints(hand_landmarks) -> np.ndarray:
    keypoints = []
    wrist = hand_landmarks.landmark[0]

    # Tính kích thước bàn tay (Scale)
    # Dùng khoảng cách từ Cổ tay (0) đến Khớp ngón giữa (9) làm chuẩn
    middle_mcp = hand_landmarks.landmark[9]
    scale = ((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2 + (wrist.z - middle_mcp.z)**2)**0.5

    # Tránh chia cho 0
    if scale == 0: scale = 1.0

    for lm in hand_landmarks.landmark:
        # 1. Relative (Trừ tâm)
        relative_x = lm.x - wrist.x
        relative_y = lm.y - wrist.y
        relative_z = lm.z - wrist.z

        # 2. Normalize Scale (Chia cho kích thước tay) -> QUAN TRỌNG
        relative_x /= scale
        relative_y /= scale
        relative_z /= scale

        keypoints.extend([relative_x, relative_y, relative_z])

    return np.array(keypoints, dtype=np.float32)

def video_to_seq(video_path: str, seq_len: int = SEQ_LEN) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_list = []

    # --- QUAN TRỌNG: Khởi tạo Hands BÊN TRONG hàm xử lý video ---
    # Dùng 'with' để đảm bảo tài nguyên được giải phóng sau mỗi video
    # Và quan trọng nhất: Reset trạng thái tracking cho video mới
    with mp_hands.Hands(
        static_image_mode=False, # Vẫn giữ False để tracking mượt trong 1 video
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chuyển màu
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                # Nếu tìm thấy tay
                kp = extract_keypoints(result.multi_hand_landmarks[0])
                frames_list.append(kp)
            else:
                # Nếu KHÔNG thấy tay (do mờ, di chuyển nhanh)
                if len(frames_list) > 0:
                    # Cách Tốt: Dùng lại vị trí của frame trước đó (Giữ nguyên tay)
                    # Giúp đường đi mượt mà, không bị giật về 0
                    frames_list.append(frames_list[-1])
                else:
                    # Nếu ngay frame đầu đã không thấy tay thì đành chịu, điền số 0
                    frames_list.append(np.zeros(FEATURE_DIM, dtype=np.float32))

    cap.release()

    # --- HẬU XỬ LÝ (RESAMPLING) ---
    original_len = len(frames_list)
    if original_len == 0:
        return None

    data_matrix = np.stack(frames_list, axis=0) # (original_len, 63)

    # Thuật toán nội suy (Interpolation) để co/giãn về đúng SEQ_LEN
    # Logic này của bạn đã đúng, giữ nguyên
    idxs = np.linspace(0, original_len - 1, seq_len).astype(int)
    data_matrix = data_matrix[idxs] # (32, 63)

    return data_matrix

def preprocess_all(filelist_path: Path = FILELIST_PATH, out_root: Path = DATA_NPY_ROOT):
    out_root.mkdir(parents=True, exist_ok=True)

    if not filelist_path.exists():
        print(f"Lỗi: Không tìm thấy {filelist_path}")
        return

    with filelist_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Bắt đầu xử lý {len(rows)} video...")

    count = 0
    for row in rows:
        rel_video_path = row["video_path"] # Ví dụ: "book/001.mp4"
        label_name = row["label_name"]     # Ví dụ: "book"

        # Đường dẫn tuyệt đối
        abs_video_path = PROJECT_ROOT / rel_video_path

        # Check file tồn tại
        if not abs_video_path.exists():
            # Thử tìm trong data_raw xem sao (đôi khi csv path sai lệch)
            abs_video_path = DATA_RAW_ROOT / label_name / os.path.basename(rel_video_path)
            if not abs_video_path.exists():
                continue

        # Xử lý
        seq = video_to_seq(str(abs_video_path))
        if seq is None:
            continue

        # Lưu file
        class_dir = out_root / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(rel_video_path))[0]
        out_path = class_dir / f"{base_name}.npy"

        np.save(out_path, seq)
        count += 1

        if count % 10 == 0:
            print(f"Đã xử lý {count}/{len(rows)} video...")

    print(f"=== HOÀN TẤT: {count} videos đã được lưu tại {out_root} ===")

if __name__ == "__main__":
    preprocess_all()