import os
import torch
import torchaudio
from torch.utils.data import Dataset

class LocalBEATAudioDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=16000, target_duration=10.0):
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate

        self.target_length = int(target_sample_rate * target_duration)
        self.audio_files = []

        print(f"Scanning '{root_dir}' for audio files...")


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


            if recording_type % 2 != 0:
                return 0


            if 0 <= sequence_id <= 64: return 0
            elif 65 <= sequence_id <= 72: return 1
            elif 73 <= sequence_id <= 80: return 2
            elif 81 <= sequence_id <= 86: return 3
            elif 87 <= sequence_id <= 94: return 4
            elif 95 <= sequence_id <= 102: return 5
            elif 103 <= sequence_id <= 110: return 6
            elif 111 <= sequence_id <= 118: return 7
            else: return 0

        except Exception:
            return 0

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]

        try:

            waveform, sample_rate = torchaudio.load(file_path)


            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)


            if waveform.shape[1] < self.target_length:
                padding = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.target_length:
                waveform = waveform[:, :self.target_length]

            if waveform.shape[0] > 1:

                waveform = torch.mean(waveform, dim=0, keepdim=True)

            emotion_label = self.extract_emotion(file_path)

            return {
                'audio': waveform,
                'emotion': torch.tensor(emotion_label, dtype=torch.long)
            }

        except Exception:


            return {
                'audio': torch.zeros(1, self.target_length),
                'emotion': torch.tensor(0, dtype=torch.long)
            }
