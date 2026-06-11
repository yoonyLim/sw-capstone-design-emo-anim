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
    device = torch.device("cpu") # Keep this on CPU so Unreal can hog the GPU!
    print("Loading AST Emotion Extractor...")
    extractor = ASTEmotionExtractor(target_latent_dim=64).to(device)
    extractor.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    extractor.eval()
    return extractor, device

def process_and_send(audio_path, extractor, device, osc_client):
    print(f"\nProcessing Audio: {audio_path}")
    
    # 1. Load and format the audio
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

    # 2. Extract the 64-D Emotion Vector
    with torch.no_grad():
        emotion_vector = extractor(waveform)
    
    # 3. Convert the PyTorch Tensor to a standard Python List of 64 floats
    # Flatten it from [1, 64] to [64]
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
    
    # 4. Blast it over the network to Unreal Engine!
    osc_client.send_message("/emotion/vector", vector_list)
    print("SUCCESS: 64-D Emotion Vector sent to Unreal Engine!")

if __name__ == "__main__":
    extractor, device = setup_audio_extractor()
    
    # Set up the OSC Client to target Unreal Engine on your local machine (Port 8000)
    client = udp_client.SimpleUDPClient("59.12.75.47", 9001)
    print("OSC Server Running. Connected to 59.12.75.47:9001")
    
    # For now, we will test it with a hardcoded file. 
    # Later, Unreal can trigger this script dynamically!
    TEST_AUDIO = r"D:\Capstone_Project\Test_Data\arthur.wav"
    #print("Spamming Unreal Engine every 1 second. Press Ctrl+C to stop.")
    #while True:
        #client.send_message("/test", 99.9)
    process_and_send(TEST_AUDIO, extractor, device, client)
        #time.sleep(1.0)