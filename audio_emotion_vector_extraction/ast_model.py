import torch
import torch.nn as nn
import torchaudio
import timm

class ASTEmotionExtractor(nn.Module):
    def __init__(self, target_latent_dim=64, export_mode=False):
        super().__init__()

        self.export_mode = export_mode


        self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)


        original_embedding_weights = self.v.patch_embed.proj.weight
        new_embedding_weights = original_embedding_weights.sum(dim=1, keepdim=True)
        self.v.patch_embed.proj.weight = nn.Parameter(new_embedding_weights)


        self.emotion_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, target_latent_dim)
        )

        self.v.head = nn.Identity()
        self.v.head_dist = nn.Identity()


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

            window = self.mel_transform.spectrogram.window
            mel_filters = self.mel_transform.mel_scale.fb.T


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




            power_spec = stft.pow(2.0).sum(dim=-1)


            spectrogram = torch.matmul(mel_filters, power_spec)


            spectrogram = spectrogram.unsqueeze(1)


            mean = spectrogram.mean(dim=[1, 2, 3], keepdim=True)
            std = spectrogram.std(dim=[1, 2, 3], keepdim=True)
            spectrogram = (spectrogram - mean) / (std * 2 + 1e-6)


            spectrogram = torch.nn.functional.interpolate(spectrogram, size=(384, 384), mode='bilinear', align_corners=False)


            x = self.v(spectrogram)



            if isinstance(x, tuple):
                x = (x[0] + x[1]) / 2.0


            emotion_vector = self.emotion_head(x)

            return emotion_vector
        else:
            spectrogram = self.mel_transform(waveform)


            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)


            mean = spectrogram.mean()
            std = spectrogram.std()
            spectrogram = (spectrogram - mean) / (std * 2 + 1e-6)


            spectrogram = torch.nn.functional.interpolate(spectrogram, size=(384, 384), mode='bilinear', align_corners=False)


            x = self.v(spectrogram)



            if isinstance(x, tuple):
                x = (x[0] + x[1]) / 2.0


            emotion_vector = self.emotion_head(x)

            return emotion_vector
