import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

from beat_dataset import LocalBEATAudioDataset
from ast_model import ASTEmotionExtractor

def generate_anchors():
    DATA_DIRECTORY = "D:/Capstone_Project/BEAT_Audio_Raw" # Match your data folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and trained model
    dataset = LocalBEATAudioDataset(root_dir=DATA_DIRECTORY)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    model = ASTEmotionExtractor(target_latent_dim=64, export_mode=False).to(device)
    model.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    model.eval() # Inference mode

    # Dictionaries to hold the sums and counts for averaging
    emotion_sums = {i: torch.zeros(64).to(device) for i in range(8)}
    emotion_counts = {i: 0 for i in range(8)}

    print("Mapping the Latent Space...")
    with torch.no_grad(): # No need to track gradients for this
        for batch in tqdm(dataloader):
            audio_waveforms = batch['audio'].to(device)
            labels = batch['emotion'].to(device)

            # Get the 64-D vectors
            vectors = model.forward(audio_waveforms)

            # Add each vector to its corresponding emotion sum
            for i in range(vectors.size(0)):
                label = labels[i].item()
                emotion_sums[label] += vectors[i]
                emotion_counts[label] += 1

    # Calculate averages and format for export
    print("\nCalculating Master Anchors...")
    master_anchors = {}
    emotion_names = ["Neutral", "Happiness", "Anger", "Sadness", "Contempt", "Surprise", "Fear", "Disgust"]

    for label_id in range(8):
        if emotion_counts[label_id] > 0:
            avg_vector = emotion_sums[label_id] / emotion_counts[label_id]
            # Normalize the anchor for cosine similarity
            avg_vector = torch.nn.functional.normalize(avg_vector, p=2, dim=0)
            
            # Convert to a standard Python list for JSON serialization
            master_anchors[emotion_names[label_id]] = avg_vector.cpu().tolist()

    # Save to JSON
    with open("master_emotion_anchors.json", "w") as f:
        json.dump(master_anchors, f, indent=4)
        
    print("SUCCESS: Saved 'master_emotion_anchors.json'")

if __name__ == "__main__":
    generate_anchors()