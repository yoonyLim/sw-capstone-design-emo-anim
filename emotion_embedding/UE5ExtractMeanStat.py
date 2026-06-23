import numpy as np
import torch

stats = torch.load('motion_stats.pt')




expanded_mean = stats['mean'].expand(1, 3, 128, 75).numpy()
expanded_std = stats['std'].expand(1, 3, 128, 75).numpy()


expanded_mean.astype(np.float32).tofile("motion_mean.bin")
expanded_std.astype(np.float32).tofile("motion_std.bin")
print("Un-normalization stats exported!")
