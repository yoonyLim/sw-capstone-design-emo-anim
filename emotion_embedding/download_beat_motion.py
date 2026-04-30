import os
from huggingface_hub import snapshot_download

def download_beat_motion_only():
    # Keep this isolated in its own folder to prevent data mixing
    target_directory = "D:/Capstone_Project/BEAT_Motion_Raw"
    os.makedirs(target_directory, exist_ok=True)

    print(f"Target Directory created at: {target_directory}")
    print("Connecting to Hugging Face server to fetch 3D skeleton data...")

    try:
        # The Surgical Download for Motion
        # We explicitly ignore .zip files, .wav files, and raw text
        snapshot_download(
            repo_id="H-Liu1997/BEAT",
            repo_type="dataset",
            local_dir=target_directory,
            allow_patterns="*.bvh",   # <-- THE MAGIC FILTER: Only grab 3D skeleton files
            ignore_patterns=["*.zip", "*.wav", "*.json", "*.mp4", "*.TextGrid"], 
            max_workers=8,            
            resume_download=True      
        )
        
        print("\nSUCCESS: All BEAT motion (.bvh) files have been downloaded safely!")
        print(f"You can view them at: {target_directory}")

    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Tip: If it fails, verify your huggingface-cli login status.")

if __name__ == "__main__":
    download_beat_motion_only()