from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_int, get_path, get_str

import os
import sys
import time
import torch
import torchaudio
import math
from pythonosc import udp_client

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
audio_folder_path = os.path.join(parent_dir, 'audio_emotion_vector_extraction')
sys.path.append(audio_folder_path)

from ast_model import ASTEmotionExtractor

def setup_audio_extractor():
    device = torch.device("cpu")
    print("Loading AST Emotion Extractor...")
    extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    extractor.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    extractor.eval()
    return extractor, device

def process_and_send(audio_path, extractor, device, osc_client):
    print(f"\nProcessing Audio: {audio_path}")


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


    with torch.no_grad():
        emotion_vector = extractor(waveform)



    vector_list = emotion_vector.squeeze().tolist()

    rms = math.sqrt(sum(v * v for v in vector_list) / len(vector_list))
    l2 = math.sqrt(sum(v * v for v in vector_list))

    print(
        f"Vector stats: "
        f"min={min(vector_list):.4f}, "
        f"max={max(vector_list):.4f}, "
        f"mean={sum(vector_list)/len(vector_list):.4f}, "
        f"rms={rms:.4f}, "
        f"l2={l2:.4f}"
    )
    print("first 8:", [round(v, 4) for v in vector_list[:8]])


    osc_client.send_message("/emotion/vector", vector_list)
    print("SUCCESS: 64-D Emotion Vector sent to Unreal Engine!")

if __name__ == "__main__":
    extractor, device = setup_audio_extractor()


    osc_host = get_str("EMO_ANIM_OSC_HOST", "127.0.0.1")
    osc_port = get_int("EMO_ANIM_OSC_PORT", 9001)
    client = udp_client.SimpleUDPClient(osc_host, osc_port)
    print(f"OSC Server Running. Connected to {osc_host}:{osc_port}")



    TEST_AUDIO = str(get_path("EMO_ANIM_OSC_TEST_AUDIO", "audio_emotion_vector_extraction/sad.wav"))



    process_and_send(TEST_AUDIO, extractor, device, client)

