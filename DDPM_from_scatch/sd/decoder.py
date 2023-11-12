import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        # GroupNorm 관련 내용 -> Notion 참고
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (Batch_size, Features, Height, Width)
        
        residue = x
        
        n, c, h, w = x.shape
        
        # (Batch_size, Features, Height, Width) ->  (Batch_size, Features, Height * Width)
        x = x.view(n, c, h * w)
        
        # (Batch_size, Features, Height * Width) -> (Batch_size, Height * Width, Features)
        # 여기서는 ViT에서처럼 각 pixel간의 relate를 계산하는 Attention을 사용함.
        # 기존의 token embedding간의 attention을 계산하는 것과 같은 매커니즘.
        x = x.transpose(-1, -2) 
        
        # (Batch_size, Height * Width, Features) -> (Batch_size, Height * Width, Features)
        x = self.attention(x)
        # (Batch_size, Height * Width, Features) -> (Batch_size, Features, Height * Width)
        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))
        
        return x + residue
        
        
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
        
        
        
        