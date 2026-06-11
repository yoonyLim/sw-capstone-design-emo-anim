import numpy as np
import torch

stats = torch.load('motion_stats.pt')

# Assuming stats['mean'] is shape [1, 3, 1, 75]. 
# We use .expand() to stretch it across all 128 frames.
# Now it becomes exactly [1, 3, 128, 75] (28,800 floats)
expanded_mean = stats['mean'].expand(1, 3, 128, 75).numpy()
expanded_std = stats['std'].expand(1, 3, 128, 75).numpy()

# Save them alongside your other .bin files
expanded_mean.astype(np.float32).tofile("motion_mean.bin")
expanded_std.astype(np.float32).tofile("motion_std.bin")
print("Un-normalization stats exported!")