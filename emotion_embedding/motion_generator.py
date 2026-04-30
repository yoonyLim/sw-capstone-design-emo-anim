import torch
import torch.nn as nn

# ==========================================
# 1. THE BUILDING BLOCKS
# ==========================================
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

# ==========================================
# 2. THE FINAL ASSEMBLY
# ==========================================
class MotionGenerator(nn.Module):
    def __init__(self, num_joints=75, in_channels=3, latent_dim=64, emotion_dim=64):
        super().__init__()
        
        # 1. Content Projection: Bumps the 3 channels (X,Y,Z) up to 64 for deep learning
        self.input_proj = nn.Conv2d(in_channels, latent_dim, kernel_size=1)
        
        # 2. The Stylization Stack (ST-GCN + AdaIN)
        self.stgcn1 = STGCN_Block(latent_dim, latent_dim)
        self.adain1 = EmotionAdaIN(emotion_dim, latent_dim)
        
        self.stgcn2 = STGCN_Block(latent_dim, latent_dim)
        self.adain2 = EmotionAdaIN(emotion_dim, latent_dim)
        
        self.stgcn3 = STGCN_Block(latent_dim, latent_dim)
        
        # 3. Output Projection: Squashes the 64 channels back down to 3 (X,Y,Z)
        self.output_proj = nn.Conv2d(latent_dim, in_channels, kernel_size=1)

    def forward(self, content_code, emotion_vector, adj_matrix, original_motion):
        """
        content_code:    [Batch, 64, Time, 75] (From Content Encoder)
        emotion_vector:  [Batch, 64]           (From Audio)
        adj_matrix:      [75, 75]
        original_motion: [Batch, 3, Time, 75]  (For the final residual connection)
        """
        
        # Step 1: Inject the Emotion (Layer 1)
        x = self.stgcn1(content_code, adj_matrix)
        x = self.adain1(x, emotion_vector)
        
        # Step 2: Inject the Emotion (Layer 2)
        x = self.stgcn2(x, adj_matrix)
        x = self.adain2(x, emotion_vector)
        
        # Step 3: Final smoothing pass
        x = self.stgcn3(x, adj_matrix)
        
        # Step 4: Decode back to rotations
        emotion_delta = self.output_proj(x)
        
        # Step 5: THE RESIDUAL CONNECTION
        stylized_motion = original_motion + emotion_delta
        
        return stylized_motion


if __name__ == "__main__":
    # --- IGNITION TEST ---
    print("Initializing Motion Generator Pipeline...")
    
    # 1. Setup Dummy Data (Mimicking a 128-frame window from your parser)
    batch_size = 1
    channels = 3      # X, Y, Z rotations
    frames = 128      # 2.1 seconds of 60fps animation
    joints = 75       # From your BVH parser
    
    dummy_motion = torch.randn(batch_size, channels, frames, joints)
    dummy_emotion = torch.randn(batch_size, 64) # From your AST .onnx file
    dummy_adj = torch.randn(joints, joints)     # From your BVH parser
    
    # 2. Build the Model
    model = MotionGenerator(num_joints=joints)
    
    # 3. Push the data through
    output_motion = model(dummy_motion, dummy_emotion, dummy_adj)
    
    print(f"Input Shape:  {dummy_motion.shape}")
    print(f"Output Shape: {output_motion.shape}")
    
    if dummy_motion.shape == output_motion.shape:
        print("SUCCESS: The Generator cleanly ingested the emotion and output a valid 3D skeleton tensor!")