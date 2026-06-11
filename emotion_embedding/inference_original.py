import os
import sys
import numpy as np
import torch
import torchaudio

from bvh_loader import BVHMotionParser
from bvh_writer import export_to_bvh
from content_encoder import ContentEncoder
from motion_generator import MotionGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
audio_folder_path = os.path.join(parent_dir, 'audio_emotion_vector_extraction')
sys.path.append(audio_folder_path)

from ast_model import ASTEmotionExtractor

def run_inference(audio_path, neutral_bvh_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on: {device}")

    # ==========================================
    # 1. LOAD THE DATA & STATISTICS
    # ==========================================
    parser = BVHMotionParser(neutral_bvh_path)
    adj_matrix = parser.get_adjacency_matrix().to(device)
    print(adj_matrix.shape)
    num_joints = len(parser.joints)
    
    # Extract original motion tensor [Channels, Time, Joints]
    original_motion = parser.get_motion_tensor().unsqueeze(0).to(device) 
    
    # Load the Z-Score stats created during training
    print("Loading Normalization Statistics...")
    stats = torch.load('motion_stats.pt', map_location=device)
    data_mean = stats['mean'].to(device)
    data_std = stats['std'].to(device)

    # ==========================================
    # 2. LOAD & FORMAT THE AUDIO
    # ==========================================
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1: 
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        
    target_audio_len = 160000
    if waveform.shape[1] < target_audio_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_audio_len - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_audio_len]
        
    # THE AUDIO DIMENSION FIX: Ensure shape is [Batch(1), Channel(1), Time]
    waveform = waveform.unsqueeze(0).to(device)

    # ==========================================
    # 3. LOAD THE NEURAL NETWORKS
    # ==========================================
    print("Loading Neural Networks...")
    audio_extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    audio_extractor.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    audio_extractor.eval()

    content_encoder = ContentEncoder(num_joints=num_joints).to(device)
    content_encoder.load_state_dict(torch.load("content_encoder_epoch_20.pth", map_location=device))
    content_encoder.eval()

    generator = MotionGenerator(num_joints=num_joints).to(device)
    generator.load_state_dict(torch.load("motion_generator_epoch_20.pth", map_location=device))
    generator.eval()

    # ==========================================
    # 4. GENERATE THE STYLIZED ANIMATION
    # ==========================================
    print("Injecting Emotion...")
    with torch.no_grad():
        # Step A: Crush input to the normalized [-3.0, 3.0] mathematical space
        original_motion_norm = (original_motion - data_mean) / data_std
        
        # Step B: Run Inference
        emotion_vector = audio_extractor(waveform)
        content_code = content_encoder(original_motion_norm, adj_matrix)
        stylized_motion_norm = generator(content_code, emotion_vector, adj_matrix, style_multiplier=1.0)
        
        # Step C: Re-inflate the output back to real-world Euler degrees!
        stylized_motion_real = (stylized_motion_norm * data_std) + data_mean

        # ANCHOR THE ROOT ROTATION
        stylized_motion_real[:, :, :, 0] = original_motion[:, :, :, 0]

    # ==========================================
    # 5. RECONSTRUCT THE HIERARCHY & EXPORT
    # ==========================================
    print("Exporting to BVH format...")
    
    # Depending on how your custom `bvh_writer.py` was built, it either expects 
    # the 3D PyTorch Tensor, or a flat Numpy array. 
    
    # IF your export_to_bvh expects a PyTorch Tensor [3, Time, 75] and handles root stitching internally:
    export_to_bvh(neutral_bvh_path, output_path, stylized_motion_real.squeeze(0))
    
    # ---------------------------------------------------------
    # ALTERNATIVE: If your BVH writer expects a raw Numpy Array with Root Translation included:
    # (Uncomment this block if the character is still detached from the hips)
    #
    # final_rotations = stylized_motion_real.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # final_rotations_flat = final_rotations.reshape(final_rotations.shape[0], -1)
    #
    # num_frames = final_rotations_flat.shape[0]
    # original_root_pos = parser.frames[:num_frames, 0:3] 
    #
    # reconstructed_bvh_data = np.concatenate((original_root_pos, final_rotations_flat), axis=1)
    # export_to_bvh(neutral_bvh_path, output_path, reconstructed_bvh_data)
    # ---------------------------------------------------------

if __name__ == "__main__":
    AUDIO_FILE = r"D:\Capstone_Project\Test_Data\arthur.wav"
    NEUTRAL_MOTION = r"D:\Capstone_Project\BEAT_Motion_Raw\beat_english_v0.2.1\beat_english_v0.2.1\1\1_wayne_0_1_1.bvh"
    OUTPUT_FILE = r"D:\Capstone_Project\Test_Data\arthur0605.bvh"
    
    run_inference(AUDIO_FILE, NEUTRAL_MOTION, OUTPUT_FILE)