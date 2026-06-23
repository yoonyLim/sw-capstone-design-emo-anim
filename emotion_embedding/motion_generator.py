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
        return (gamma * normalized_motion) + beta




class MotionGenerator(nn.Module):
    def __init__(self, num_joints=75, in_channels=3, latent_dim=64, emotion_dim=64):
        super().__init__()


        self.input_proj = nn.Conv2d(in_channels, latent_dim, kernel_size=1)


        self.stgcn1 = STGCN_Block(latent_dim, latent_dim)
        self.adain1 = EmotionAdaIN(emotion_dim, latent_dim)

        self.stgcn2 = STGCN_Block(latent_dim, latent_dim)
        self.adain2 = EmotionAdaIN(emotion_dim, latent_dim)

        self.stgcn3 = STGCN_Block(latent_dim, latent_dim)


        self.output_proj = nn.Conv2d(latent_dim, in_channels, kernel_size=1)

    def forward(self, content_code, emotion_vector, adj_matrix, style_multiplier=1.0):
        """
        content_code:    [Batch, 64, Time, 75] (From Content Encoder)
        emotion_vector:  [Batch, 64]           (From Audio)
        adj_matrix:      [75, 75]
        original_motion: [Batch, 3, Time, 75]  (For the final residual connection)
        """
        amplified_emotion = emotion_vector * style_multiplier


        x = self.stgcn1(content_code, adj_matrix)
        x = self.adain1(x, amplified_emotion)


        x = self.stgcn2(x, adj_matrix)
        x = self.adain2(x, amplified_emotion)


        x = self.stgcn3(x, adj_matrix)


        stylized_motion = self.output_proj(x)

        return stylized_motion


if __name__ == "__main__":

    print("Initializing Motion Generator Pipeline...")


    batch_size = 1
    channels = 3
    frames = 128
    joints = 75

    dummy_motion = torch.randn(batch_size, channels, frames, joints)
    dummy_emotion = torch.randn(batch_size, 64)
    dummy_adj = torch.randn(joints, joints)


    model = MotionGenerator(num_joints=joints)


    output_motion = model(dummy_motion, dummy_emotion, dummy_adj)

    print(f"Input Shape:  {dummy_motion.shape}")
    print(f"Output Shape: {output_motion.shape}")

    if dummy_motion.shape == output_motion.shape:
        print("SUCCESS: The Generator cleanly ingested the emotion and output a valid 3D skeleton tensor!")
