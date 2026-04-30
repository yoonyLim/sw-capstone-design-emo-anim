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

# ==========================================
# DATASET UPGRADE: Multi-Modal Loading
# ==========================================
class BEATMultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, motion_dir, audio_dir):
        self.motion_dir = motion_dir
        self.audio_dir = audio_dir
        self.samples = []
        
        # 1. Walk through the MOTION directory
        for root, _, filenames in os.walk(motion_dir):
            for filename in filenames:
                if filename.endswith('.bvh'):
                    # 2. Get the base name (e.g., "1_wayne_0_1_1")
                    base_name = filename.replace('.bvh', '')
                    
                    # 3. Calculate the relative subfolder path (e.g., "1")
                    # This handles the fact that the files are inside subfolders 1-30
                    relative_path = os.path.relpath(root, motion_dir)
                    
                    # 4. Construct the expected path for the AUDIO file
                    wav_path = os.path.join(audio_dir, relative_path, f"{base_name}.wav")
                    
                    # 5. Check if the matching audio file actually exists!
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
        
        # Parse the first file just to grab the Adjacency Matrix
        parser = BVHMotionParser(self.samples[0]['bvh'])
        self.adj_matrix = parser.get_adjacency_matrix()
        self.num_joints = len(parser.joints)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Motion
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
            
        # 2. Load Audio 
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

# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Define BOTH master directories
    MOTION_DIR = r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1\beat_english_v0.2.1"
    AUDIO_DIR = r"D:\Capstone_Project\BEAT_Audio_Raw\beat_english_v0.2.1\beat_english_v0.2.1"
    
    # Pass both to the new dataset loader
    dataset = BEATMultiModalDataset(MOTION_DIR, AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    adj_matrix = dataset.adj_matrix.to(device)

    # --- INITIALIZE PIPELINE ---
    # 1. The Audio Extractor (Frozen - we don't want to change its weights)
    audio_extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    audio_extractor.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    audio_extractor.eval()
    
    # 2. The Content Encoder
    content_encoder = ContentEncoder(num_joints=dataset.num_joints).to(device)
    
    # 3. The Motion Generator
    generator = MotionGenerator(num_joints=dataset.num_joints).to(device)
    
    # --- SETUP OPTIMIZERS ---
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(list(content_encoder.parameters()) + list(generator.parameters()), lr=1e-4)
    epochs = 5

    # --- TRAINING LOOP ---
    for epoch in range(epochs):
        content_encoder.train()
        generator.train()
        running_loss = 0.0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for motion_batch, audio_batch in loop:
            motion_batch = motion_batch.to(device)
            audio_batch = audio_batch.to(device)
            
            optimizer.zero_grad()
            
            # 1. Extract real emotion from the audio file
            with torch.no_grad():
                emotion_vector = audio_extractor(audio_batch)
                
            # 2. Strip the style from the motion file to create a sterile base
            content_code = content_encoder(motion_batch, adj_matrix)
            
            # 3. Generate the stylized animation
            output_motion = generator(content_code, emotion_vector, adj_matrix, motion_batch)
            
            # 4. Calculate Loss & Step
            loss = criterion(output_motion, motion_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {running_loss/len(dataloader):.4f}")

    torch.save(content_encoder.state_dict(), "content_encoder_weights.pth")
    torch.save(generator.state_dict(), "motion_generator_weights.pth")
    print("SUCCESS: Pipeline Weights Saved!")

if __name__ == "__main__":
    train()