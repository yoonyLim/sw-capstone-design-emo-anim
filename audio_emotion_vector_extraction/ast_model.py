import torch
import torch.nn as nn
import torchaudio
import timm

class ASTEmotionExtractor(nn.Module):
    def __init__(self, target_latent_dim=64, export_mode=False):
        super().__init__()

        self.export_mode = export_mode
        
        # Load the base Vision Transformer
        self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
        
        # Collapse the 3 RGB channels into 1 Audio channel
        original_embedding_weights = self.v.patch_embed.proj.weight
        new_embedding_weights = original_embedding_weights.sum(dim=1, keepdim=True)
        self.v.patch_embed.proj.weight = nn.Parameter(new_embedding_weights)
        
        # Replace the 1000-class ImageNet head with our 64-D Latent Bottleneck
        self.emotion_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, target_latent_dim)
        )
        
        self.v.head = nn.Identity()
        self.v.head_dist = nn.Identity()

        # Define the Spectrogram Converter
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            n_mels=128,
            center=False,
            pad_mode="constant"
        )

    def forward(self, waveform):
        """
        Executes the forward pass using a raw waveform tensor from the DataLoader.
        Expected input shape: [Batch_Size, 1, 160000]
        """
        if self.export_mode:
            # 1. Grab the exact Hann window and Mel filters we trained with
            window = self.mel_transform.spectrogram.window
            mel_filters = self.mel_transform.mel_scale.fb.T # Transpose for matrix multiplication

            # 2. Pure PyTorch STFT (Natively supported in ONNX Opset 17)
            stft = torch.stft(
                waveform,
                n_fft=1024,
                hop_length=160,
                win_length=400,
                window=window,
                center=False,
                normalized=False,
                return_complex=False
            )

            # 3. Calculate Power Spectrogram (Magnitude squared)
            # stft shape is [Batch, Freq, Time, 2]. 
            # We square the Real and Imaginary parts and sum them on the last dimension.
            power_spec = stft.pow(2.0).sum(dim=-1)

            # Convert to Spectrogram
            spectrogram = torch.matmul(mel_filters, power_spec) # Output shape: [Batch, 128, Time]

            # Add channel dimension if missing: [Batch, 1, 128, Time_Frames]
            spectrogram = spectrogram.unsqueeze(1)
                
            # Normalize the spectrogram
            mean = spectrogram.mean(dim=[1, 2, 3], keepdim=True)
            std = spectrogram.std(dim=[1, 2, 3], keepdim=True)
            spectrogram = (spectrogram - mean) / (std * 2 + 1e-6) # Added epsilon for numerical stability

            # Interpolate to fit the ViT's expected 384x384 image size
            spectrogram = torch.nn.functional.interpolate(spectrogram, size=(384, 384), mode='bilinear', align_corners=False)

            # Forward pass through the Transformer
            x = self.v(spectrogram)

            # If the model is distilled, it returns a tuple: (class_token, dist_token)
            # The standard engineering practice for DeiT is to average the two tokens together.
            if isinstance(x, tuple):
                x = (x[0] + x[1]) / 2.0
            
            # Extract the 64-Dimensional Vector
            emotion_vector = self.emotion_head(x)
            
            return emotion_vector
        else:
            spectrogram = self.mel_transform(waveform) 

            # Add channel dimension if missing: [Batch, 1, 128, Time_Frames]
            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)
                
            # Normalize the spectrogram
            mean = spectrogram.mean()
            std = spectrogram.std()
            spectrogram = (spectrogram - mean) / (std * 2 + 1e-6) # Added epsilon for numerical stability

            # Interpolate to fit the ViT's expected 384x384 image size
            spectrogram = torch.nn.functional.interpolate(spectrogram, size=(384, 384), mode='bilinear', align_corners=False)

            # Forward pass through the Transformer
            x = self.v(spectrogram)

            # If the model is distilled, it returns a tuple: (class_token, dist_token)
            # The standard engineering practice for DeiT is to average the two tokens together.
            if isinstance(x, tuple):
                x = (x[0] + x[1]) / 2.0
            
            # Extract the 64-Dimensional Vector
            emotion_vector = self.emotion_head(x)
            
            return emotion_vector