from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_builder import BEATMotionDataset




def compute_dataset_stats():
    print("Gathering all motion data (this may take a minute)...")

    bvh_dir = str(get_path("BEAT_MOTION_ROOT", "data/beat_motion"))
    audio_dir = str(get_path("BEAT_AUDIO_ROOT", "data/beat_audio"))


    beat_dataset = BEATMotionDataset(bvh_folder=bvh_dir, audio_folder=audio_dir, window_size=128)



    dataloader = DataLoader(beat_dataset, batch_size=32, shuffle=True, drop_last=True)

    all_motions = []
    for motion_batch, _ in tqdm(dataloader):
        all_motions.append(motion_batch)



    full_dataset = torch.cat(all_motions, dim=0)




    data_mean = full_dataset.mean(dim=(0, 2), keepdim=True)
    data_std = full_dataset.std(dim=(0, 2), keepdim=True)




    data_std[data_std < 1e-4] = 1.0


    torch.save({'mean': data_mean, 'std': data_std}, 'motion_stats.pt')
    print("\nSUCCESS: Normalization stats saved to 'motion_stats.pt'")
    print(f"Mean Shape: {data_mean.shape} | Std Shape: {data_std.shape}")

if __name__ == "__main__":
    compute_dataset_stats()
