from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

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




    parser = BVHMotionParser(neutral_bvh_path)
    adj_matrix = parser.get_adjacency_matrix().to(device)
    print(adj_matrix.shape)
    num_joints = len(parser.joints)


    original_motion = parser.get_motion_tensor().unsqueeze(0).to(device)


    print("Loading Normalization Statistics...")
    stats = torch.load('motion_stats.pt', map_location=device)
    data_mean = stats['mean'].to(device)
    data_std = stats['std'].to(device)




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


    waveform = waveform.unsqueeze(0).to(device)




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




    print("Injecting Emotion...")
    with torch.no_grad():

        original_motion_norm = (original_motion - data_mean) / data_std


        emotion_vector = audio_extractor(waveform)
        content_code = content_encoder(original_motion_norm, adj_matrix)
        stylized_motion_norm = generator(content_code, emotion_vector, adj_matrix, style_multiplier=1.0)


        stylized_motion_real = (stylized_motion_norm * data_std) + data_mean


        stylized_motion_real[:, :, :, 0] = original_motion[:, :, :, 0]




    print("Exporting to BVH format...")





    export_to_bvh(neutral_bvh_path, output_path, stylized_motion_real.squeeze(0))















if __name__ == "__main__":
    AUDIO_FILE = str(get_path("EMO_ANIM_DEFAULT_AUDIO", "data/input.wav"))
    NEUTRAL_MOTION = str(get_path("EMO_ANIM_SAMPLE_BVH", "data/sample.bvh"))
    OUTPUT_FILE = str(get_path("EMO_ANIM_ORIGINAL_OUTPUT_BVH", "outputs/generated_original.bvh"))

    run_inference(AUDIO_FILE, NEUTRAL_MOTION, OUTPUT_FILE)
