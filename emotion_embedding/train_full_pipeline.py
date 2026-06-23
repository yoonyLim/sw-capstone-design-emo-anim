from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import torchaudio

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
audio_folder_path = os.path.join(parent_dir, 'audio_emotion_vector_extraction')
sys.path.append(audio_folder_path)

from bvh_loader import BVHMotionParser
from motion_generator import MotionGenerator
from content_encoder import ContentEncoder
from ast_model import ASTEmotionExtractor




class BEATMultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, motion_dir, audio_dir):
        self.motion_dir = motion_dir
        self.audio_dir = audio_dir
        self.samples = []


        for root, _, filenames in os.walk(motion_dir):
            for filename in filenames:
                if filename.endswith('.bvh'):

                    base_name = filename.replace('.bvh', '')



                    relative_path = os.path.relpath(root, motion_dir)


                    wav_path = os.path.join(audio_dir, relative_path, f"{base_name}.wav")


                    if os.path.exists(wav_path):
                        self.samples.append({
                            'bvh': os.path.join(root, filename),
                            'wav': wav_path
                        })
                    else:
                        print(f"Warning: Missing audio for {filename}")

        print(f"Found {len(self.samples)} perfectly matched Audio/Motion pairs.")

        if len(self.samples) == 0:
            raise ValueError("Zero pairs found. Check your directory paths!")


        parser = BVHMotionParser(self.samples[0]['bvh'])
        self.adj_matrix = parser.get_adjacency_matrix()
        self.num_joints = len(parser.joints)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]


        parser = BVHMotionParser(sample['bvh'])
        motion_tensor = parser.get_motion_tensor()

        target_frames = 128
        current_frames = motion_tensor.shape[1]

        if current_frames > target_frames:
            start = torch.randint(0, current_frames - target_frames, (1,)).item()
            motion_tensor = motion_tensor[:, start:start+target_frames, :]
        elif current_frames < target_frames:
            padding = target_frames - current_frames
            motion_tensor = torch.nn.functional.pad(motion_tensor, (0, 0, 0, padding))


        waveform, sr = torchaudio.load(sample['wav'])
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        target_audio_len = 160000
        if waveform.shape[1] < target_audio_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_audio_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_audio_len]

        return motion_tensor, waveform




def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")


    MOTION_DIR = str(get_path("BEAT_MOTION_ROOT", "data/beat_motion"))
    AUDIO_DIR = str(get_path("BEAT_AUDIO_ROOT", "data/beat_audio"))


    dataset = BEATMultiModalDataset(MOTION_DIR, AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    adj_matrix = dataset.adj_matrix.to(device)



    audio_extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    audio_extractor.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    audio_extractor.eval()


    content_encoder = ContentEncoder(num_joints=dataset.num_joints).to(device)


    generator = MotionGenerator(num_joints=dataset.num_joints).to(device)












    criterion = nn.L1Loss()
    optimizer = optim.AdamW(list(content_encoder.parameters()) + list(generator.parameters()), lr=1e-4)
    start_epoch = 0
    total_epochs = 20

    stats = torch.load('motion_stats.pt', map_location=device)
    data_mean = stats['mean']
    data_std = stats['std']
    print("Loaded Normalization Statistics.")


    for epoch in range(start_epoch, total_epochs):
        content_encoder.train()
        generator.train()
        running_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False, dynamic_ncols=True)

        for batch_idx, (motion_batch, audio_batch) in enumerate(loop):
            motion_batch = motion_batch.to(device)
            audio_batch = audio_batch.to(device)




            if epoch == start_epoch and batch_idx == 0:
                print("\n\n" + "="*50)
                print("SYSTEM DIAGNOSTIC REPORT")
                print("="*50)


                m_min = motion_batch.min().item()
                m_max = motion_batch.max().item()
                print(f"Motion Data Bounds:  MIN: {m_min:.2f}  |  MAX: {m_max:.2f}")

                if m_max > 5.0 or m_min < -5.0:
                    print("  [CRITICAL WARNING] Data is not normalized! Values are too large for AdaIN.")
                    if m_max > 170.0:
                        print("  [CRITICAL WARNING] Euler Angles detected (-180 to 180). This will cause 358-degree jump penalties!")
                else:
                    print("  [PASS] Data appears to be normalized.")


                adj_min = adj_matrix.min().item()
                adj_max = adj_matrix.max().item()
                print(f"\nAdjacency Matrix Bounds: MIN: {adj_min:.4f} | MAX: {adj_max:.4f}")

                if adj_max > 1.1:
                    print("  [CRITICAL WARNING] Adjacency matrix is not symmetrically normalized.")
                    print("  Graph Convolutions will mathematically explode at deep layers.")
                else:
                    print("  [PASS] Adjacency matrix scaling looks healthy.")

                print("="*50 + "\n")


            optimizer.zero_grad()

            motion_batch_norm = (motion_batch - data_mean) / data_std


            with torch.no_grad():
                emotion_vector = audio_extractor(audio_batch)


            content_code = content_encoder(motion_batch_norm, adj_matrix)


            output_motion_norm = generator(content_code, emotion_vector, adj_matrix)

            raw_error = torch.abs(output_motion_norm - motion_batch_norm)
            joint_weights = torch.ones(75).to(device)


            right_arm_indices = range(9, 36)
            left_arm_indices = range(36, 63)
            right_leg_indices = range(63, 69)
            left_leg_indices = range(69, 75)

            joint_weights[left_arm_indices] = 3.0
            joint_weights[right_arm_indices] = 3.0
            joint_weights[right_leg_indices] = 2.0
            joint_weights[left_leg_indices] = 2.0

            weighted_error = raw_error * joint_weights.view(1, 1, 1, -1)
            loss_rec = weighted_error.mean()



            vel_generated = output_motion_norm[:, :, 1:, :] - output_motion_norm[:, :, :-1, :]
            vel_real = motion_batch_norm[:, :, 1:, :] - motion_batch_norm[:, :, :-1, :]
            loss_vel = criterion(vel_generated, vel_real)


            with torch.no_grad():
                target_content_code = content_encoder(motion_batch_norm, adj_matrix).detach()

            generated_content_code = content_encoder(output_motion_norm, adj_matrix)
            loss_content = criterion(generated_content_code, target_content_code)


            total_loss = loss_rec + (0.1 * loss_vel) + (0.05 * loss_content)


            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item(), rec=loss_rec.item(), vel=loss_vel.item())

        print(f"Epoch {epoch+1} Avg Loss: {running_loss/len(dataloader):.4f}")

        if (epoch + 1) % 50 == 0:
            ce_path = f"content_encoder_epoch_{epoch+1}.pth"
            gen_path = f"motion_generator_epoch_{epoch+1}.pth"

            torch.save(content_encoder.state_dict(), ce_path)
            torch.save(generator.state_dict(), gen_path)

            print(f"--> [CHECKPOINT] Saved: {ce_path} & {gen_path}")

    torch.save(content_encoder.state_dict(), "content_encoder_weights.pth")
    torch.save(generator.state_dict(), "motion_generator_weights.pth")
    print("SUCCESS: Pipeline Weights Saved!")

if __name__ == "__main__":
    train()
