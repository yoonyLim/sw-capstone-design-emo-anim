import os

# Make sure this matches your path
DATA_DIRECTORY = "D:/Capstone_Project/BEAT_Audio_Raw" 

audio_files = []
for dirpath, _, filenames in os.walk(DATA_DIRECTORY):
    for file in filenames:
        if file.endswith(".wav"):
            audio_files.append(file)

print(f"Total files found: {len(audio_files)}\n")
print("--- FILENAME PARSER CHECK ---")

# Let's look at a sample of 20 files from different parts of the folder
step = max(1, len(audio_files) // 20)
sample_files = audio_files[::step][:20]

for file in sample_files:
    base_name = file.replace('.wav', '')
    parts = base_name.split('_')
    
    try:
        # Our current logic
        emotion_id = int(parts[2])
        print(f"File: {file:<25} | Parsed ID: {emotion_id} | Split Parts: {parts}")
    except Exception as e:
        print(f"File: {file:<25} | FAILED TO PARSE (Fallback 0) | Split Parts: {parts}")