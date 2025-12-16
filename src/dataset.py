import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# ===== PATH =====
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_NPY_ROOT = PROJECT_ROOT / "data_npy"


def build_label_map_from_data(root_dir: Path):
    """
    Sinh label_map trực tiếp từ data_npy/
    Chỉ lấy class có ít nhất 1 file .npy
    """
    classes = []
    for d in sorted(os.listdir(root_dir)):
        class_dir = root_dir / d
        if class_dir.is_dir():
            if any(f.endswith(".npy") for f in os.listdir(class_dir)):
                classes.append(d)

    label_map = {name: idx for idx, name in enumerate(classes)}
    return label_map


class WLASLKeypointDataset(Dataset):
    def __init__(self, root_dir=DATA_NPY_ROOT, augment=False, target_len=32):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.target_len = target_len

        # 🔥 label_map sinh từ data_npy
        self.label_map = build_label_map_from_data(self.root_dir)
        self.num_classes = len(self.label_map)

        self.samples = []

        for label_name, idx in self.label_map.items():
            class_dir = self.root_dir / label_name
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    self.samples.append((class_dir / fname, idx))

        if len(self.samples) == 0:
            raise RuntimeError("Dataset rỗng – kiểm tra data_npy")

        print(f"[Dataset] Loaded {len(self.samples)} samples")
        print(f"[Dataset] Num classes = {self.num_classes}")

    def __len__(self):
        return len(self.samples)

    def pad_or_truncate(self, seq):
        L, F = seq.shape
        if L > self.target_len:
            return seq[:self.target_len]
        elif L < self.target_len:
            pad = np.zeros((self.target_len - L, F), dtype=np.float32)
            return np.vstack((seq, pad))
        return seq

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)

        seq = self.pad_or_truncate(seq)

        # Augmentation (chỉ khi train)
        if self.augment:
            if np.random.rand() < 0.5:
                seq *= np.random.uniform(0.9, 1.1)
            if np.random.rand() < 0.5:
                seq += np.random.normal(0, 0.002, seq.shape)

        return torch.FloatTensor(seq), torch.tensor(label, dtype=torch.long)
