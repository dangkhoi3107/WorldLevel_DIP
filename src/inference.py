import torch
import json
from model import GRUClassifier

# 1. Thông số đã train
CONFIG = {
    "input_size": 63,
    "hidden_size": 256,
    "num_layers": 2,
    "num_classes": 590
}

# 2. Load Model
device = torch.device('cpu')
model = GRUClassifier(**CONFIG)
model.load_state_dict(torch.load("word_level_gru2.pth", map_location=device))
model.eval()

print("Load thành công Model với cấu hình:", CONFIG)
