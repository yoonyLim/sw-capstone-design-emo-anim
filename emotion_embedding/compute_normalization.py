import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_builder import BEATMotionDataset

# [IMPORT YOUR DATALOADER HERE]
# from your_dataset_file import dataloader 

def compute_dataset_stats():
    print("Gathering all motion data (this may take a minute)...")

    bvh_dir = r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1\beat_english_v0.2.1"
    audio_dir = r"D:\Capstone_Project\BEAT_Audio_Raw\beat_english_v0.2.1\beat_english_v0.2.1"

    # 2. Initialize the Dataset
    beat_dataset = BEATMotionDataset(bvh_folder=bvh_dir, audio_folder=audio_dir, window_size=128)

    # 3. Create the PyTorch DataLoader
    # This handles the GPU batching and shuffles the 128-frame chunks so the network doesn't memorize the order
    dataloader = DataLoader(beat_dataset, batch_size=32, shuffle=True, drop_last=True)
    
    all_motions = []
    for motion_batch, _ in tqdm(dataloader):
        all_motions.append(motion_batch)
        
    # Stack every single batch into one massive tensor
    # Shape: [Total_Samples, Channels(3), Time(128), Joints(75)]
    full_dataset = torch.cat(all_motions, dim=0)
    
    # Calculate Mean and Standard Deviation
    # We calculate across the Batch (dim 0) and Time (dim 2) axes.
    # This gives us the exact average for every specific joint's X, Y, and Z axis.
    data_mean = full_dataset.mean(dim=(0, 2), keepdim=True)
    data_std = full_dataset.std(dim=(0, 2), keepdim=True)
    
    # Safety Check: If a bone never moves (like the Neck base), its standard deviation is 0.
    # Dividing by 0 creates NaN (Not a Number) errors and crashes the network. 
    # We force any 0s to become 1s to prevent this.
    data_std[data_std < 1e-4] = 1.0
    
    # Save the statistics
    torch.save({'mean': data_mean, 'std': data_std}, 'motion_stats.pt')
    print("\nSUCCESS: Normalization stats saved to 'motion_stats.pt'")
    print(f"Mean Shape: {data_mean.shape} | Std Shape: {data_std.shape}")

if __name__ == "__main__":
    compute_dataset_stats()