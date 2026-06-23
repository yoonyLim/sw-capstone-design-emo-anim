import os
import torch
from torch.utils.data import Dataset, DataLoader
from bvh_loader import BVHMotionParser
from audio_extractor import load_wav_file

class BEATMotionDataset(Dataset):
    def __init__(self, bvh_folder, audio_folder, window_size=128):
        """
        Recursively scans the parent folders, uses Tensor Caching to skip redundant parsing,
        and chunks the data into 128-frame windows.
        """
        self.window_size = window_size
        self.samples = []

        print(f"Scanning all subfolders inside {bvh_folder}...")

        for current_root, sub_directories, files in os.walk(bvh_folder):
            for file_name in files:
                if file_name.endswith('.bvh'):
                    bvh_path = os.path.join(current_root, file_name)


                    subfolder_name = os.path.basename(current_root)
                    audio_name = file_name.replace('.bvh', '.wav')
                    audio_path = os.path.join(audio_folder, subfolder_name, audio_name)

                    if not os.path.exists(audio_path):
                        audio_path = os.path.join(audio_folder, audio_name)

                    if not os.path.exists(audio_path):
                        continue




                    cache_path = bvh_path.replace('.bvh', '_cache.pt')

                    if os.path.exists(cache_path):

                        full_motion_tensor = torch.load(cache_path)
                    else:

                        parser = BVHMotionParser(bvh_path)
                        full_motion_tensor = parser.get_motion_tensor()
                        torch.save(full_motion_tensor, cache_path)


                    total_frames = full_motion_tensor.shape[1]

                    for start_frame in range(0, total_frames - window_size, window_size):
                        end_frame = start_frame + window_size
                        motion_chunk = full_motion_tensor[:, start_frame:end_frame, :]

                        self.samples.append({
                            'motion': motion_chunk,
                            'motion_path': bvh_path,
                            'audio_path': audio_path,
                            'start_frame': start_frame
                        })

        print(f"Dataset Built! Created {len(self.samples)} valid 128-frame training chunks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        motion = sample['motion']
        start_frame = sample['start_frame']


        fps = 60.0
        target_sr = 16000
        window_size = motion.shape[1]


        audio_start_sample = int((start_frame / fps) * target_sr)
        audio_samples_needed = int((window_size / fps) * target_sr)

        try:


            audio_tensor = load_wav_file(sample['audio_path'], target_sr=target_sr)


            audio_chunk = audio_tensor[:, :, audio_start_sample : audio_start_sample + audio_samples_needed]


            if audio_chunk.shape[2] < audio_samples_needed:
                pad_amount = audio_samples_needed - audio_chunk.shape[2]

                audio_chunk = torch.nn.functional.pad(audio_chunk, (0, pad_amount))

        except Exception as e:

            audio_chunk = torch.zeros((1, 1, audio_samples_needed))

        return motion, audio_chunk
