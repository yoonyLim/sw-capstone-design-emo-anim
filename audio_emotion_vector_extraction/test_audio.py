from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import torch
import torchaudio
import json
import torch.nn.functional as F
from ast_model import ASTEmotionExtractor

def test_single_audio(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading '{file_path}'...")


    model = ASTEmotionExtractor(target_latent_dim=64).to(device)
    model.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    model.eval()


    with open("master_emotion_anchors.json", "r") as f:
        anchors_dict = json.load(f)


    anchor_names = list(anchors_dict.keys())
    anchor_tensors = torch.stack([torch.tensor(anchors_dict[name]) for name in anchor_names]).to(device)


    try:
        waveform, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    if waveform.shape[0] > 1:

        waveform = torch.mean(waveform, dim=0, keepdim=True)

    max_amplitude = torch.max(torch.abs(waveform))

    if max_amplitude > 0:
        waveform = waveform / max_amplitude


    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)


    target_length = 160000
    if waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding))
    elif waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]


    waveform = waveform.unsqueeze(0).to(device)


    with torch.no_grad():
        output_vector = model(waveform)


        output_vector = F.normalize(output_vector, p=2, dim=1)



    similarities = F.cosine_similarity(output_vector, anchor_tensors)


    print("\n--- EMOTION ANALYSIS RESULTS ---")


    sorted_indices = torch.argsort(similarities, descending=True)

    for idx in sorted_indices:
        emotion = anchor_names[idx]
        score = similarities[idx].item()



        percentage = max(0, score) * 100
        print(f"{emotion:<12}: {percentage:>5.1f}% Match (Score: {score:.3f})")

if __name__ == "__main__":

    TEST_FILE = str(get_path("EMO_ANIM_AUDIO_TEST_FILE", "audio_emotion_vector_extraction/sad.wav"))
    test_single_audio(TEST_FILE)
