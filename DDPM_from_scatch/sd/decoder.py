import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)

class VAE_ResidualBlock(nn.Moudle):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    
    def forward(self, x: torch.Tensor)  -> torch.Tensor:
        # x : (Batch_size, In_channels), Height, Width)
        
        residue = x
        
        # (Batch_size, In_channels), Height, Width) -> (Batch_size, In_channels), Height, Width)
        x = self.groupnorm_1(x) 
        
        # (Batch_size, In_channels), Height, Width) -> (Batch_size, In_channels), Height, Width)
        x = F.silu(x)
        
        # (Batch_size, In_channels), Height, Width) -> (Batch_size, In_channels), Height, Width)
        x = self.conv_1(x)
        
        # (Batch_size, In_channels), Height, Width) -> (Batch_size, In_channels), Height, Width)
        x = self.groupnorm_2(x)
        
        # (Batch_size, In_channels), Height, Width) -> (Batch_size, In_channels), Height, Width)
        x = F.silu(x)
        
        x = self.conv_2(x)
        
        # If in_channel and out_channel are different, then self.residual_layer must be added. 
        return x + self.residual_layer(residue)
        
        
        
        