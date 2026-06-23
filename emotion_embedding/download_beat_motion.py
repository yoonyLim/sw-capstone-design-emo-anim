from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import os
from huggingface_hub import snapshot_download

def download_beat_motion_only():

    target_directory = str(get_path("BEAT_MOTION_DOWNLOAD_DIR", "data/BEAT_Motion_Raw"))
    os.makedirs(target_directory, exist_ok=True)

    print(f"Target Directory created at: {target_directory}")
    print("Connecting to Hugging Face server to fetch 3D skeleton data...")

    try:


        snapshot_download(
            repo_id="H-Liu1997/BEAT",
            repo_type="dataset",
            local_dir=target_directory,
            allow_patterns="*.bvh",
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
