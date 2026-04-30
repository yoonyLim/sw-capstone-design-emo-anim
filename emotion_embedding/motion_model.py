import torch
import torch.nn as nn

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_t=9):
        super().__init__()
        
        # 1. Spatial Convolution: A 1x1 Conv to process the Adjacency Matrix math
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 2. Temporal Convolution: Slides across the Time dimension
        # We use padding to ensure the frame count stays exactly the same
        pad = (kernel_size_t - 1) // 2
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=(kernel_size_t, 1), 
            padding=(pad, 0)
        )
        
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # Expected input 'x' shape: [Batch, Channels, Time, Joints]
        # Expected 'A' shape: [Joints, Joints]

        # --- SPATIAL PASS ---
        # Multiply the skeleton data by the Adjacency Matrix to share data between connected bones.
        # torch.matmul natively handles the last two dimensions [Time, Joints] @ [Joints, Joints]
        x_spatial = torch.matmul(x, A)
        x_spatial = self.spatial_conv(x_spatial)
        
        # --- TEMPORAL PASS ---
        # Look across the frames to calculate momentum
        x_temporal = self.temporal_conv(x_spatial)
        
        return self.relu(x_temporal)
    
class EmotionAdaIN(nn.Module):
    def __init__(self, emotion_dim=64, motion_channels=64):
        super().__init__()
        
        # These Linear layers act as translators. 
        # They convert your 64-D audio vector into scaling (gamma) and shifting (beta) factors
        # that the 3D skeleton can mathematically understand.
        self.fc_gamma = nn.Linear(emotion_dim, motion_channels)
        self.fc_beta = nn.Linear(emotion_dim, motion_channels)

    def forward(self, motion_features, emotion_vector):
        # motion_features shape: [Batch, Channels, Time, Joints]
        # emotion_vector shape:  [Batch, 64]

        # 1. Extract the Emotion Style Parameters
        # We use .view() to reshape them so they broadcast cleanly across all frames and joints
        gamma = self.fc_gamma(emotion_vector).view(-1, motion_features.size(1), 1, 1)
        beta = self.fc_beta(emotion_vector).view(-1, motion_features.size(1), 1, 1)

        # 2. Strip the original style from the motion
        # We calculate the mean and standard deviation of the neutral animation
        mean = torch.mean(motion_features, dim=[2, 3], keepdim=True)
        std = torch.std(motion_features, dim=[2, 3], keepdim=True) + 1e-6
        
        normalized_motion = (motion_features - mean) / std

        # 3. Inject the new Emotion Style!
        stylized_motion = (gamma * normalized_motion) + beta

        return stylized_motion