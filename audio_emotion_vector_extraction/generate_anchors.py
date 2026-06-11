from pathlib import Path
import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from beat_dataset import LocalBEATAudioDataset
from ast_model import ASTEmotionExtractor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = r"D:\Capstone_Project\BEAT_Audio_Raw"
EMOTION_NAMES = [
    "Neutral",
    "Happiness",
    "Anger",
    "Sadness",
    "Contempt",
    "Surprise",
    "Fear",
    "Disgust",
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def generate_anchors(data_dir, weights_path, raw_output, normalized_output, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LocalBEATAudioDataset(root_dir=str(data_dir))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ASTEmotionExtractor(target_latent_dim=64, export_mode=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    emotion_sums = {index: torch.zeros(64, device=device) for index in range(len(EMOTION_NAMES))}
    emotion_counts = {index: 0 for index in range(len(EMOTION_NAMES))}

    print("Mapping raw AST latent space...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            audio_waveforms = batch["audio"].to(device)
            labels = batch["emotion"].to(device)
            vectors = model(audio_waveforms)

            for item_index in range(vectors.size(0)):
                label = int(labels[item_index].item())
                if label not in emotion_sums:
                    continue
                emotion_sums[label] += vectors[item_index]
                emotion_counts[label] += 1

    raw_anchors = {}
    normalized_anchors = {}

    print("\nCalculating anchors...")
    for label_id, emotion_name in enumerate(EMOTION_NAMES):
        count = emotion_counts[label_id]
        if count <= 0:
            print(f"WARNING: No samples found for {emotion_name}; skipping.")
            continue

        raw_vector = emotion_sums[label_id] / count
        normalized_vector = F.normalize(raw_vector, p=2, dim=0)

        raw_anchors[emotion_name] = raw_vector.cpu().tolist()
        normalized_anchors[emotion_name] = normalized_vector.cpu().tolist()

        print(
            f"{emotion_name:10s} count={count:5d} "
            f"raw_l2={torch.linalg.vector_norm(raw_vector).item():.4f}"
        )

    write_json(raw_output, raw_anchors)
    write_json(normalized_output, normalized_anchors)

    print(f"SUCCESS: Saved raw motion anchors -> {raw_output}")
    print(f"SUCCESS: Saved normalized cosine anchors -> {normalized_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate emotion anchors from the trained AST extractor.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--weights", default=str(SCRIPT_DIR / "ast_emotion_extractor_weights.pth"))
    parser.add_argument("--raw-output", default=str(SCRIPT_DIR / "master_emotion_anchors_raw.json"))
    parser.add_argument("--normalized-output", default=str(SCRIPT_DIR / "master_emotion_anchors.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_anchors(
        data_dir=Path(args.data_dir),
        weights_path=Path(args.weights),
        raw_output=Path(args.raw_output),
        normalized_output=Path(args.normalized_output),
        batch_size=args.batch_size,
    )
