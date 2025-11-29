import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

# project root = thư mục cha của src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_NPY_ROOT = PROJECT_ROOT / "data_npy"
META_DIR = PROJECT_ROOT / "meta"
LABELS_PATH = META_DIR / "labels.txt"


def load_label_map(labels_path: Path = LABELS_PATH):
    """Đọc labels.txt → {'book': 0, 'hello': 1, ...}"""
    labels = []
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

    def __init__(self, root_dir: Path | str = DATA_NPY_ROOT, label_map=None):
        self.root_dir = Path(root_dir)
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
            print("WARNING: dataset trống, kiểm tra lại data_npy/.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        seq = np.load(path).astype(np.float32)  # (SEQ_LEN, FEATURE_DIM)
        return seq, label
