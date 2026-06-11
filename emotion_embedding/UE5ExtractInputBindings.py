import torch
import numpy as np
from bvh_loader import BVHMotionParser

# 1. Get the data
parser = BVHMotionParser(r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_1_1.bvh")
adj_matrix = parser.get_adjacency_matrix().numpy() # Shape [75, 75]
motion_tensor = parser.get_motion_tensor().unsqueeze(0) # Shape [1, 3, Time, 75]

# 2. Normalize it! (Use your stats)
stats = torch.load('motion_stats.pt')
motion_norm = (motion_tensor - stats['mean']) / stats['std']

total_frames = motion_norm.shape[2]
valid_frames = (total_frames // 128) * 128  # Rounds down to the nearest 128
motion_chopped = motion_norm[:, :, :valid_frames, :]

# 4. Save as raw, flat C++ Float32 binaries
adj_matrix.astype(np.float32).tofile("adjacency_matrix.bin")
motion_chopped.numpy().astype(np.float32).tofile("neutral_motion.bin")
print(f"Exported {valid_frames} frames (exactly {valid_frames // 128} chunks) to binary!")