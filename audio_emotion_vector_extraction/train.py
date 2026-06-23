from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from project_env import get_path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from beat_dataset import LocalBEATAudioDataset
from ast_model import ASTEmotionExtractor

def train_pipeline():


    DATA_DIRECTORY = str(get_path("BEAT_AUDIO_DOWNLOAD_DIR", "data/BEAT_Audio_Raw"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialization complete. Using device: {device}")



    dataset = LocalBEATAudioDataset(root_dir=DATA_DIRECTORY)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)


    model = ASTEmotionExtractor(target_latent_dim=64, export_mode=False).to(device)
    classifier_head = nn.Linear(64, 8).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier_head.parameters()),
        lr=5e-5,
        weight_decay=1e-4
    )


    epochs = 40

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        model.train()
        classifier_head.train()
        running_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in loop:
            audio_waveforms = batch['audio'].to(device)
            emotion_labels = batch['emotion'].to(device)

            optimizer.zero_grad()


            emotion_vectors = model.forward(audio_waveforms)
            predictions = classifier_head(emotion_vectors)

            loss = criterion(predictions, emotion_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.6f}")

        scheduler.step()

        print(f"Epoch {epoch+1} Avg Loss: {running_loss/len(dataloader):.4f}")


    print("Training complete! Slicing off the classification head...")
    torch.save(model.state_dict(), "ast_emotion_extractor_weights.pth")
    print("Saved -> ast_emotion_extractor_weights.pth")

if __name__ == "__main__":
    train_pipeline()
