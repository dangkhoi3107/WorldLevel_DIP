import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# project root = thư mục cha của src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_NPY_ROOT = PROJECT_ROOT / "data_npy"
META_DIR = PROJECT_ROOT / "meta"
LABELS_PATH = META_DIR / "labels.txt"

def load_label_map(labels_path: Path = LABELS_PATH):
    """Đọc labels.txt → {'book': 0, 'hello': 1, ...}"""
    labels = []
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                l = line.strip()
                if l:
                    labels.append(l)
    return {name: i for i, name in enumerate(labels)}

class WLASLKeypointDataset(Dataset):
    """
    Đọc các file .npy trong data_npy/<label_name>/*.npy
    Mỗi sample: (SEQ_LEN, FEATURE_DIM), label_idx
    """
    # QUAN TRỌNG: Đã thêm tham số augment=False vào đây
    def __init__(self, root_dir: Path | str = DATA_NPY_ROOT, label_map=None, augment=False):
        self.root_dir = Path(root_dir)
        self.augment = augment  # Lưu biến này để dùng trong __getitem__

        if label_map is None:
            self.label_map = load_label_map()
        else:
            self.label_map = label_map

        self.samples = []  # list[(path, label_idx)]

        for label_name, idx in self.label_map.items():
            class_dir = self.root_dir / label_name
            if not class_dir.is_dir():
                continue

            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    path = class_dir / fname
                    self.samples.append((path, idx))

        if len(self.samples) == 0:
            print(f"WARNING: Dataset trống tại {self.root_dir}. Hãy kiểm tra lại preprocess.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        seq = np.load(path).astype(np.float32)  # (32, 63)

        # --- DATA AUGMENTATION (Chỉ chạy khi self.augment = True) ---
        if self.augment:
            # 1. Thêm nhiễu (Noise)
            noise = np.random.normal(0, 0.005, seq.shape).astype(np.float32)
            seq += noise

            # 2. Co giãn (Scaling) - Phóng to/nhỏ biên độ cử chỉ
            scale = np.random.uniform(0.9, 1.1)
            seq *= scale

            # 3. Dịch chuyển (Shifting) - Dời tay sang trái/phải/lên/xuống
            # Shift chỉ áp dụng cho x, y (2/3 features), nhưng shift cả 3 cũng không sao
            shift = np.random.uniform(-0.05, 0.05, size=(1, 63)).astype(np.float32)
            seq += shift
        # -----------------------------------------------------------

        return seq, label