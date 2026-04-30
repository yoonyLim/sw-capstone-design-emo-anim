import torch
import torch.onnx
from ast_model import ASTEmotionExtractor

def export_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing ONNX Export...")

    # 1. Load the empty architecture
    model = ASTEmotionExtractor(target_latent_dim=64, export_mode=True).to(device)

    # 2. Inject your newly trained weights
    model.load_state_dict(torch.load("ast_emotion_extractor_weights.pth", map_location=device))
    
    # 3. CRITICAL: Set to evaluation mode (disables dropout/batchnorm updates)
    model.eval()

    # 4. Create dummy input matching the expected 10-second 16kHz audio
    dummy_waveform = torch.randn(1, 160000).to(device)

    # 5. Trace and Export
    output_filename = "Audio2Emotion.onnx"
    print("Tracing the graph... (This might take a minute)")
    
    try:
        torch.onnx.export(
            model, 
            (dummy_waveform,), # Passed as a tuple
            output_filename,
            export_params=True,
            opset_version=17,
            do_constant_folding=False,
            input_names=['input_audio'],
            output_names=['emotion_vector']
        )
        print(f"SUCCESS: Engine-ready model saved as '{output_filename}'")
    except Exception as e:
        print(f"ONNX Export Failed: {e}")
        print("Note: Sometimes torchaudio STFT operations clash with ONNX opsets. If so, we will adjust the Mel-transform logic.")

if __name__ == "__main__":
    export_model()