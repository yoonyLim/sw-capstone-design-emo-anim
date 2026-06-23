import torch
import torch.nn as nn

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_t=9):
        super().__init__()


        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)



        pad = (kernel_size_t - 1) // 2
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(kernel_size_t, 1),
            padding=(pad, 0)
        )

        self.relu = nn.ReLU()

    def forward(self, x, A):






        x_spatial = torch.matmul(x, A)
        x_spatial = self.spatial_conv(x_spatial)



        x_temporal = self.temporal_conv(x_spatial)

        return self.relu(x_temporal)

class EmotionAdaIN(nn.Module):
    def __init__(self, emotion_dim=64, motion_channels=64):
        super().__init__()




        self.fc_gamma = nn.Linear(emotion_dim, motion_channels)
        self.fc_beta = nn.Linear(emotion_dim, motion_channels)

    def forward(self, motion_features, emotion_vector):





        gamma = self.fc_gamma(emotion_vector).view(-1, motion_features.size(1), 1, 1)
        beta = self.fc_beta(emotion_vector).view(-1, motion_features.size(1), 1, 1)



        mean = torch.mean(motion_features, dim=[2, 3], keepdim=True)
        std = torch.std(motion_features, dim=[2, 3], keepdim=True) + 1e-6

        normalized_motion = (motion_features - mean) / std


        stylized_motion = (gamma * normalized_motion) + beta

        return stylized_motion
