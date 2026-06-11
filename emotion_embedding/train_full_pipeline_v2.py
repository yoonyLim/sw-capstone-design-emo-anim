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
    # Convert predictions and targets to radians if they aren't already!
    # Calculate the raw difference
    diff = pred - target
    
    # 1.0 - cos(diff) will be 0 when the angles are exactly the same, 
    # and it will perfectly wrap around the 360-degree mark.
    loss = torch.mean(1.0 - torch.cos(diff))
    return loss

def train_pipeline():
    # 1. Hardware Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    
    # 2. File Path Configuration
    bvh_dir = r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1\beat_english_v0.2.1"
    audio_dir = r"D:\Capstone_Project\BEAT_Audio_Raw\beat_english_v0.2.1\beat_english_v0.2.1"
    
    # 3. Initialize the Recursive Dataset & DataLoader
    print("\nInitializing Dataset... (Scanning all subfolders 1-30)")
    dataset = BEATMotionDataset(bvh_folder=bvh_dir, audio_folder=audio_dir, window_size=128)
    
    # Batch size 32, shuffling chunks so the network learns generalized style dynamics
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # 4. Generate the Symmetrically Normalized Graph Adjacency Matrix
    # We parse a single sample file just to extract the uniform skeleton structure
    print("Extracting skeletal hierarchy layout...")
    
    # Grab the very first chunk's audio/motion paths from our already-built dataset
    # This prevents us from having to hardcode a specific sample file path!
    sample_bvh_path = dataset.samples[0]['motion_path'] if 'motion_path' in dataset.samples[0] else None
    
    # If your dataset builder doesn't store the path explicitly, just pull a guaranteed file path:
    if not sample_bvh_path:
        # We manually find the first .bvh file in the entire tree to use as our template
        for root, _, files in os.walk(bvh_dir):
            bvh_files = [f for f in files if f.endswith('.bvh')]
            if bvh_files:
                sample_bvh_path = os.path.join(root, bvh_files[0])
                break

    print(f"Using skeletal template file: {os.path.basename(sample_bvh_path)}")
    parser = BVHMotionParser(sample_bvh_path)
    adj_matrix = parser.get_adjacency_matrix().to(device)
    num_joints = len(parser.joints)  # Dynamically sets to 75
    
    # 5. Load Z-Score Normalization Statistics
    print("Loading Dataset Normalization parameters...")
    if not os.path.exists('motion_stats.pt'):
        raise FileNotFoundError("CRITICAL ERROR: 'motion_stats.pt' not found. Run compute_normalization.py first!")
        
    stats = torch.load('motion_stats.pt', map_location=device)
    data_mean = stats['mean'].to(device)
    data_std = stats['std'].to(device)
    
    # 6. Initialize Networks from Scratch (Fresh Reset for Normalized Space)
    print("Building Neural Network Architecture...")
    content_encoder = ContentEncoder(num_joints=num_joints).to(device)
    generator = MotionGenerator(num_joints=num_joints).to(device)
    
    # Initialize and lock down the pre-trained Audio Transformer
    audio_extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    audio_extractor.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    audio_extractor.eval() # Audio encoder weights are completely frozen during this stage

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
    
    # 7. Optimizer & Loss Configuration
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
        
        # UPGRADE: dynamic_ncols and leave=False prevents terminal text stacking cascades
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
            
            # ===================================================
            # STEP 1: APPLY Z-SCORE NORMALIZATION
            # ===================================================
            # Compresses values from [-1400, 1396] to balanced variances around [-3.0, 3.0]
            motion_batch_norm = (motion_batch - data_mean) / data_std
            
            # ===================================================
            # STEP 2: FORWARD INFERENCE PASS
            # ===================================================
            with torch.no_grad():
                audio_batch = audio_batch.view(audio_batch.size(0), 1, -1)
                emotion_vector = audio_extractor(audio_batch)
                
            # Strip style using normalized input base
            content_code = content_encoder(motion_batch_norm, adj_matrix)
            
            # Synthesize stylized variations via AdaIN injection
            output_motion_norm = generator(content_code, emotion_vector, adj_matrix)
            
            # ===================================================
            # STEP 3: FULL-BODY BONE-WEIGHTED RECONSTRUCTION LOSS
            # ===================================================
            # 1. Un-normalize back to real scale for angular math
            output_real_scale = (output_motion_norm * data_std) + data_mean
            target_real_scale = (motion_batch_norm * data_std) + data_mean

            # 2. Convert to radians (Assuming your raw BVH data was in degrees. 
            # If your parser already outputs radians, you can delete this * pi/180 step).
            output_rad = output_real_scale * (torch.pi / 180.0)
            target_rad = target_real_scale * (torch.pi / 180.0)

            # 3. SOLUTION 2: Cosine Angular Distance Loss
            # This perfectly wraps the 360-degree boundary, eliminating the spin glitch.
            diff = output_rad - target_rad
            raw_error = 1.0 - torch.cos(diff)

            joint_weights = torch.ones(num_joints).to(device)
            
            # Arm/Hand chains (3.0 penalty forces left arm out of static dead-arm posture)
            joint_weights[range(9, 36)] = 1.0   # Right Arm & Fingers
            joint_weights[range(36, 63)] = 1.0  # Left Arm & Fingers
            
            # Leg chains (2.0 penalty breaks static leg freezing / gradient starvation)
            joint_weights[range(63, 69)] = 1.0  # Right Leg & Foot
            joint_weights[range(69, 75)] = 1.0  # Left Leg & Foot
            
            # Spine, neck, and head nodes remain at baseline 1.0 weight
            weighted_error = raw_error * joint_weights.view(1, 1, 1, -1)
            loss_rec = weighted_error.mean()
            
            # ===================================================
            # STEP 4: VELOCITY & CONTENT PRESERVATION LOSSES
            # ===================================================
            # Loss B: Velocity calculation evaluated entirely in normalized scale
            vel_generated = output_motion_norm[:, :, 1:, :] - output_motion_norm[:, :, :-1, :]
            vel_real = motion_batch_norm[:, :, 1:, :] - motion_batch_norm[:, :, :-1, :]
            loss_vel = criterion(vel_generated, vel_real)
            
            # Loss C: Content preservation checking to enforce core choreography retention
            with torch.no_grad():
                target_content_code = content_encoder(motion_batch_norm, adj_matrix).detach()
            generated_content_code = content_encoder(output_motion_norm, adj_matrix)
            loss_content = criterion(generated_content_code, target_content_code)
            
            # ===================================================
            # STEP 5: REBALANCED LOSS COMBINATION
            # ===================================================
            # - Scaled down Velocity to 0.05 to unlock locked/straight elbows
            # - Scaled up Content to 0.5 to force the arms closer to the body/choreography
            total_loss = loss_rec + (0.05 * loss_vel) + (0.5 * loss_content)
            
            # ===================================================
            # STEP 6: BACKWARD PASS & WEIGHT UPDATES
            # ===================================================
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item(), rec=loss_rec.item())
            
        # Clean terminal output summary printed at the conclusion of every epoch
        print(f"Epoch {epoch+1} Avg Loss: {running_loss/len(dataloader):.4f}")
        
        # ===================================================
        # STEP 7: PERIODIC CHECKPOINT SAVING
        # ===================================================
        ce_path = f"content_encoder_epoch_{epoch+1}.pth"
        gen_path = f"motion_generator_epoch_{epoch+1}.pth"
        
        torch.save(content_encoder.state_dict(), ce_path)
        torch.save(generator.state_dict(), gen_path)
        print(f"--> [CHECKPOINT SECURED] Saved: {ce_path} & {gen_path}")

    # Final Weights Export
    torch.save(content_encoder.state_dict(), "content_encoder_final.pth")
    torch.save(generator.state_dict(), "motion_generator_final.pth")
    print("\nSUCCESS: ST-GCN AdaIN Full Pipeline Training Complete!")

if __name__ == "__main__":
    train_pipeline()