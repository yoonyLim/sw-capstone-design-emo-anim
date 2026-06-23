import os
import sys
import torch
import torchaudio


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
audio_folder_path = os.path.join(parent_dir, 'audio_emotion_vector_extraction')
sys.path.append(audio_folder_path)

from ast_model import ASTEmotionExtractor

def load_wav_file(file_path, target_sr=16000):
    """Loads a wav file, converts to mono, and resamples to 16kHz for the AST."""

    waveform, sample_rate = torchaudio.load(file_path)


    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)


    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)




    waveform = waveform.unsqueeze(0)

    return waveform

def bake_emotions():
    device = torch.device("cpu")


    audio_model = ASTEmotionExtractor(target_latent_dim=64).to(device)
    audio_model.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    audio_model.eval()


    emotion_files = {
        "Anger": "anger.wav"
    }

    print("Baking Emotion Vectors...\n")

    with torch.no_grad():
        for emotion_name, file_path in emotion_files.items():
            try:

                audio_tensor = load_wav_file(file_path, target_sr=16000).to(device)


                latent_code = audio_model(audio_tensor)


                flat_list = latent_code.squeeze().tolist()


                formatted_numbers = ", ".join([f"{num:.4f}" for num in flat_list])
                print(f"--- {emotion_name} ---")
                print(f"[{formatted_numbers}]\n")
            except Exception as e:
                print(f"Failed to process {emotion_name} ({file_path}): {e}")

if __name__ == "__main__":
    bake_emotions()


    -3.3865, 2.6052, -1.9240, 1.3173, 0.1550, -1.4064, -0.1747, -2.6550, -0.2466, 3.3881, 1.3281, 0.9517,
    1.0758, -1.9886, -0.9709, -2.1956, 0.8276, 2.6688, 0.2260, 0.2375, 1.1972, -0.3447, 1.3293, 1.8028, 0.4655, 0.5094, 0.0419, 1.8556, 1.5820,
    -1.2301, -1.4074, 0.2789, 0.6358, -1.7877, 0.7626, -0.2485, -0.2563, 2.3995, 0.2384, -0.5420, 0.6264, 1.6721, 1.7792, -2.2264, 0.2271,
    -0.3624, 1.6733, -0.4695, -0.0386, -1.7017, 0.1243, -0.3367, -0.1945, 1.5166, -1.3946, -0.3570, 1.5537, -0.3184, 0.3688, -1.3142, -2.4501, -0.9214, -1.8754, 0.2702
