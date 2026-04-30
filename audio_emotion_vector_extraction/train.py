import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the modules we just built
from beat_dataset import LocalBEATAudioDataset
from ast_model import ASTEmotionExtractor

def train_pipeline():
    # 1. Environment & Hardware Setup
    # Change this path to where your surgical downloader placed the files
    DATA_DIRECTORY = "D:/Capstone_Project/BEAT_Audio_Raw" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware initialization complete. Using device: {device}")

    # 2. Initialize Data
    # Batch size 8 is the safety limit for 8GB VRAM with a ViT.
    dataset = LocalBEATAudioDataset(root_dir=DATA_DIRECTORY)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    # 3. Initialize Models
    model = ASTEmotionExtractor(target_latent_dim=64, export_mode=False).to(device)
    classifier_head = nn.Linear(64, 8).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier_head.parameters()), 
        lr=5e-5,
        weight_decay=1e-4
    )

    # 4. The Main Loop
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
            
            # Extract 64-D vector, then map to 8 classes
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

    # 5. Export Final Weights
    print("Training complete! Slicing off the classification head...")
    torch.save(model.state_dict(), "ast_emotion_extractor_weights.pth")
    print("Saved -> ast_emotion_extractor_weights.pth")

if __name__ == "__main__":
    train_pipeline()