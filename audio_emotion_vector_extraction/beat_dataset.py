import os
import torch
import torchaudio
from torch.utils.data import Dataset

class LocalBEATAudioDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=16000, target_duration=10.0):
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        # AST expects fixed length. 10 seconds at 16kHz = 160,000 samples.
        self.target_length = int(target_sample_rate * target_duration) 
        self.audio_files = []
        
        print(f"Scanning '{root_dir}' for audio files...")
        
        # Safely traverse directories looking only for .wav files
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith(".wav"):
                    self.audio_files.append(os.path.join(dirpath, file))
                    
        print(f"Ready! Found {len(self.audio_files)} clean audio files.")

    def extract_emotion(self, filename):
        """
        Extracts the integer emotion ID from the filename.
        Adjust this logic if your downloaded BEAT files use a different naming convention.
        """
        base_name = os.path.basename(filename).replace('.wav', '')
        parts = base_name.split('_')
        
        try:
            recording_type = int(parts[2])
            sequence_id = int(parts[3])
            
            # According to BEAT docs, all "Conversations" (Odd numbers 1, 3, 5, 7) are labeled Neutral
            if recording_type % 2 != 0:
                return 0
                
            # If it is a "Speech" (Even numbers 0, 2, 4, 6), the Sequence ID dictates the emotion
            if 0 <= sequence_id <= 64: return 0      # Neutral
            elif 65 <= sequence_id <= 72: return 1   # Happiness
            elif 73 <= sequence_id <= 80: return 2   # Anger
            elif 81 <= sequence_id <= 86: return 3   # Sadness
            elif 87 <= sequence_id <= 94: return 4   # Contempt
            elif 95 <= sequence_id <= 102: return 5  # Surprise
            elif 103 <= sequence_id <= 110: return 6 # Fear
            elif 111 <= sequence_id <= 118: return 7 # Disgust
            else: return 0
            
        except Exception:
            return 0 # Fallback to Neutral if the filename is totally broken

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        
        try:
            # 1. Load Audio
            waveform, sample_rate = torchaudio.load(file_path)
            
            # 2. Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            
            # 3. Standardize Length (Pad or Crop)
            if waveform.shape[1] < self.target_length:
                padding = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.target_length:
                waveform = waveform[:, :self.target_length]

            if waveform.shape[0] > 1:
                # Averages the left and right channels together
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            emotion_label = self.extract_emotion(file_path)

            return {
                'audio': waveform,
                'emotion': torch.tensor(emotion_label, dtype=torch.long)
            }
            
        except Exception:
            # Fail-safe: If a file is completely corrupted, return an empty tensor
            # so the training loop doesn't crash halfway through the night.
            return {
                'audio': torch.zeros(1, self.target_length),
                'emotion': torch.tensor(0, dtype=torch.long)
            }