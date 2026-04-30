import os
from huggingface_hub import snapshot_download

def download_beat_audio_only():
    # 1. THE SSD SHIELD
    # Change 'D:/' to the root of your 1TB drive. 
    # This keeps your project entirely isolated from your OS drive.
    target_directory = "D:/Capstone_Project/BEAT_Audio_Raw"
    os.makedirs(target_directory, exist_ok=True)

    print(f"Target Directory created at: {target_directory}")
    print("Connecting to Hugging Face server...")

    try:
        # 2. THE SURGICAL DOWNLOAD
        # snapshot_download pulls files directly from the repository.
        # allow_patterns="*.wav" acts as a strict filter. It will ignore every 
        # single .json, .csv, and .bvh file, completely bypassing our earlier errors.
        snapshot_download(
            repo_id="H-Liu1997/BEAT",
            repo_type="dataset",
            local_dir=target_directory,
            allow_patterns="*.wav",   # Only grab audio files
            ignore_patterns="*.zip",  # Ignore compressed archives if they exist
            max_workers=8,            # Use 8 parallel threads to speed up the download
            resume_download=True      # If your internet drops, it picks up right where it left off
        )
        
        print("\nSUCCESS: All BEAT audio files have been downloaded safely!")
        print(f"You can view them at: {target_directory}")

    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Tip: If it asks for authentication, you may need to run 'huggingface-cli login' in your terminal.")

if __name__ == "__main__":
    download_beat_audio_only()