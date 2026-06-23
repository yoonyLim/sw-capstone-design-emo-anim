from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_builder import BEATMotionDataset
from bvh_loader import BVHMotionParser
from motion_generator import MotionGenerator
from content_encoder import ContentEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
audio_folder_path = os.path.join(parent_dir, 'audio_emotion_vector_extraction')
sys.path.append(audio_folder_path)

from ast_model import ASTEmotionExtractor

def angular_distance_loss(pred, target):


    diff = pred - target



    loss = torch.mean(1.0 - torch.cos(diff))
    return loss

def train_pipeline():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")


    bvh_dir = str(get_path("BEAT_MOTION_ROOT", "data/beat_motion"))
    audio_dir = str(get_path("BEAT_AUDIO_ROOT", "data/beat_audio"))


    print("\nInitializing Dataset... (Scanning all subfolders 1-30)")
    dataset = BEATMotionDataset(bvh_folder=bvh_dir, audio_folder=audio_dir, window_size=128)


    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)



    print("Extracting skeletal hierarchy layout...")



    sample_bvh_path = dataset.samples[0]['motion_path'] if 'motion_path' in dataset.samples[0] else None


    if not sample_bvh_path:

        for root, _, files in os.walk(bvh_dir):
            bvh_files = [f for f in files if f.endswith('.bvh')]
            if bvh_files:
                sample_bvh_path = os.path.join(root, bvh_files[0])
                break

    print(f"Using skeletal template file: {os.path.basename(sample_bvh_path)}")
    parser = BVHMotionParser(sample_bvh_path)
    adj_matrix = parser.get_adjacency_matrix().to(device)
    num_joints = len(parser.joints)


    print("Loading Dataset Normalization parameters...")
    if not os.path.exists('motion_stats.pt'):
        raise FileNotFoundError("CRITICAL ERROR: 'motion_stats.pt' not found. Run compute_normalization.py first!")

    stats = torch.load('motion_stats.pt', map_location=device)
    data_mean = stats['mean'].to(device)
    data_std = stats['std'].to(device)


    print("Building Neural Network Architecture...")
    content_encoder = ContentEncoder(num_joints=num_joints).to(device)
    generator = MotionGenerator(num_joints=num_joints).to(device)


    audio_extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    audio_extractor.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    audio_extractor.eval()

    encoder_path = "content_encoder_epoch_18.pth"
    generator_path = "motion_generator_epoch_18.pth"

    if os.path.exists(encoder_path):
        print(f"Loading Content Encoder weights from '{encoder_path}'...")
        content_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    else:
        print(f"Warning: Could not find {encoder_path}. Starting fresh.")

    if os.path.exists(generator_path):
        print(f"Loading Generator weights from '{generator_path}'...")
        generator.load_state_dict(torch.load(generator_path, map_location=device))
    else:
        print(f"Warning: Could not find {generator_path}. Starting fresh.")


    optimizer = optim.Adam(list(content_encoder.parameters()) + list(generator.parameters()), lr=1e-4)
    criterion = nn.L1Loss()

    epochs = 30
    start_epoch = 18

    print(f"\nSUCCESS: Pipeline initialized cleanly. Training 75 joints across {len(dataset)} chunks.")
    print("Starting Training Loop...\n")

    for epoch in range(start_epoch, epochs):
        content_encoder.train()
        generator.train()
        running_loss = 0.0


        loop = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            dynamic_ncols=True
        )

        for motion_batch, audio_batch in loop:
            motion_batch = motion_batch.to(device)
            audio_batch = audio_batch.to(device)

            optimizer.zero_grad()





            motion_batch_norm = (motion_batch - data_mean) / data_std




            with torch.no_grad():
                audio_batch = audio_batch.view(audio_batch.size(0), 1, -1)
                emotion_vector = audio_extractor(audio_batch)


            content_code = content_encoder(motion_batch_norm, adj_matrix)


            output_motion_norm = generator(content_code, emotion_vector, adj_matrix)





            output_real_scale = (output_motion_norm * data_std) + data_mean
            target_real_scale = (motion_batch_norm * data_std) + data_mean



            output_rad = output_real_scale * (torch.pi / 180.0)
            target_rad = target_real_scale * (torch.pi / 180.0)



            diff = output_rad - target_rad
            raw_error = 1.0 - torch.cos(diff)

            joint_weights = torch.ones(num_joints).to(device)


            joint_weights[range(9, 36)] = 1.0
            joint_weights[range(36, 63)] = 1.0


            joint_weights[range(63, 69)] = 1.0
            joint_weights[range(69, 75)] = 1.0


            weighted_error = raw_error * joint_weights.view(1, 1, 1, -1)
            loss_rec = weighted_error.mean()





            vel_generated = output_motion_norm[:, :, 1:, :] - output_motion_norm[:, :, :-1, :]
            vel_real = motion_batch_norm[:, :, 1:, :] - motion_batch_norm[:, :, :-1, :]
            loss_vel = criterion(vel_generated, vel_real)


            with torch.no_grad():
                target_content_code = content_encoder(motion_batch_norm, adj_matrix).detach()
            generated_content_code = content_encoder(output_motion_norm, adj_matrix)
            loss_content = criterion(generated_content_code, target_content_code)






            total_loss = loss_rec + (0.05 * loss_vel) + (0.5 * loss_content)




            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item(), rec=loss_rec.item())


        print(f"Epoch {epoch+1} Avg Loss: {running_loss/len(dataloader):.4f}")




        ce_path = f"content_encoder_epoch_{epoch+1}.pth"
        gen_path = f"motion_generator_epoch_{epoch+1}.pth"

        torch.save(content_encoder.state_dict(), ce_path)
        torch.save(generator.state_dict(), gen_path)
        print(f"--> [CHECKPOINT SECURED] Saved: {ce_path} & {gen_path}")


    torch.save(content_encoder.state_dict(), "content_encoder_final.pth")
    torch.save(generator.state_dict(), "motion_generator_final.pth")
    print("\nSUCCESS: ST-GCN AdaIN Full Pipeline Training Complete!")

if __name__ == "__main__":
    train_pipeline()
