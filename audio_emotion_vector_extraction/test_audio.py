import torch
import torchaudio
import json
import torch.nn.functional as F
from ast_model import ASTEmotionExtractor

def test_single_audio(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading '{file_path}'...")

    # 1. Load the pre-trained Brain
    model = ASTEmotionExtractor(target_latent_dim=64).to(device)
    model.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    model.eval()

    # 2. Load the JSON Anchors (The Translator)
    with open("master_emotion_anchors.json", "r") as f:
        anchors_dict = json.load(f)
    
    # Convert JSON lists back to PyTorch tensors
    anchor_names = list(anchors_dict.keys())
    anchor_tensors = torch.stack([torch.tensor(anchors_dict[name]) for name in anchor_names]).to(device)

    # 3. Process the new audio file
    try:
        waveform, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    if waveform.shape[0] > 1:
        # Averages the left and right channels together
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    max_amplitude = torch.max(torch.abs(waveform))

    if max_amplitude > 0:
        waveform = waveform / max_amplitude

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Pad or Crop to exactly 10 seconds (160,000 samples)
    target_length = 160000
    if waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding))
    elif waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]

    # Add the missing batch dimension: [1, 1, 160000]
    waveform = waveform.unsqueeze(0).to(device)

    # 4. Run the Network!
    with torch.no_grad():
        output_vector = model(waveform) # Shape: [1, 64]
        
        # Normalize the output so it can be compared purely by angle (Cosine Similarity)
        output_vector = F.normalize(output_vector, p=2, dim=1)

    # 5. Calculate Cosine Similarity against all 8 anchors
    # Cosine Similarity ranges from -1.0 (opposite) to 1.0 (exact match)
    similarities = F.cosine_similarity(output_vector, anchor_tensors)

    # 6. Print the Results
    print("\n--- EMOTION ANALYSIS RESULTS ---")
    
    # Sort the results from highest match to lowest
    sorted_indices = torch.argsort(similarities, descending=True)
    
    for idx in sorted_indices:
        emotion = anchor_names[idx]
        score = similarities[idx].item()
        
        # Convert score to a rough percentage (0 to 100%)
        # Note: Cosine space handles negatives, so we clamp it to 0 for display
        percentage = max(0, score) * 100 
        print(f"{emotion:<12}: {percentage:>5.1f}% Match (Score: {score:.3f})")

if __name__ == "__main__":
    # Change this path to ANY .wav file on your computer to test the network
    TEST_FILE = r"C:\Users\hayoo\Documents\DevProjects\emo-anim\audio_emotion_vector_extraction\sad.wav"
    test_single_audio(TEST_FILE)