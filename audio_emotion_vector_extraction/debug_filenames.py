from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import os


DATA_DIRECTORY = str(get_path("BEAT_AUDIO_DOWNLOAD_DIR", "data/BEAT_Audio_Raw"))

audio_files = []
for dirpath, _, filenames in os.walk(DATA_DIRECTORY):
    for file in filenames:
        if file.endswith(".wav"):
            audio_files.append(file)

print(f"Total files found: {len(audio_files)}\n")
print("--- FILENAME PARSER CHECK ---")


step = max(1, len(audio_files) // 20)
sample_files = audio_files[::step][:20]

for file in sample_files:
    base_name = file.replace('.wav', '')
    parts = base_name.split('_')

    try:

        emotion_id = int(parts[2])
        print(f"File: {file:<25} | Parsed ID: {emotion_id} | Split Parts: {parts}")
    except Exception as e:
        print(f"File: {file:<25} | FAILED TO PARSE (Fallback 0) | Split Parts: {parts}")
