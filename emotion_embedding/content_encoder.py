import torch
import torch.nn as nn
from motion_model import STGCN_Block

class ContentEncoder(nn.Module):
    def __init__(self, num_joints=75, in_channels=3, latent_dim=64):
        super().__init__()
        
        # 1. Input Projection
        self.input_proj = nn.Conv2d(in_channels, latent_dim, kernel_size=1)
        
        # 2. ST-GCN + Instance Normalization Stack
        self.stgcn1 = STGCN_Block(latent_dim, latent_dim)
        # InstanceNorm strips the variance/mean (the "style") across the spatial dimension
        self.in1 = nn.InstanceNorm2d(latent_dim, affine=False) 
        
        self.stgcn2 = STGCN_Block(latent_dim, latent_dim)
        self.in2 = nn.InstanceNorm2d(latent_dim, affine=False)
        
        # 3. Final ST-GCN Pass
        self.stgcn3 = STGCN_Block(latent_dim, latent_dim)

    def forward(self, motion_features, adj_matrix):
        """
        motion_features: [Batch, 3, Time, 75]
        adj_matrix:      [75, 75]
        """
        x = self.input_proj(motion_features)
        
        # Pass 1: Extract and Strip Style
        x = self.stgcn1(x, adj_matrix)
        x = self.in1(x)
        
        # Pass 2: Extract deeper relations and Strip Style
        x = self.stgcn2(x, adj_matrix)
        x = self.in2(x)
        
        # Final Pass: The pure "Content Code"
        content_code = self.stgcn3(x, adj_matrix)
        
        return content_code