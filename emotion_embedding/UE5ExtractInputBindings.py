from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import torch
import numpy as np
from bvh_loader import BVHMotionParser


parser = BVHMotionParser(str(get_path("EMO_ANIM_SAMPLE_BVH", "data/sample.bvh")))
adj_matrix = parser.get_adjacency_matrix().numpy()
motion_tensor = parser.get_motion_tensor().unsqueeze(0)


stats = torch.load('motion_stats.pt')
motion_norm = (motion_tensor - stats['mean']) / stats['std']

total_frames = motion_norm.shape[2]
valid_frames = (total_frames // 128) * 128
motion_chopped = motion_norm[:, :, :valid_frames, :]


adj_matrix.astype(np.float32).tofile("adjacency_matrix.bin")
motion_chopped.numpy().astype(np.float32).tofile("neutral_motion.bin")
print(f"Exported {valid_frames} frames (exactly {valid_frames // 128} chunks) to binary!")
