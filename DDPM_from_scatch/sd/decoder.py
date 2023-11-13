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
        

class VAE_Decoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            
            nn.Conv2d(4, 512, kernel_size=3, padding=3),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            
            # TODO : nn.Upsample -> 고전 방식의 up-scaling방식이며,  nearest neighbor와 bilinear, bicubic 인터폴레이션 등이 이에 해당
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 356, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128), # 128개의 channel을 32개의 Group으로 나눔으로 나눈뒤에 Normalization을 함.
            
            nn.SiLU(),
            
            # (Batch_size, 128, Height, Width) -> (Batch_size, 3, Height, Width), Orignal Image Resolution
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x : (Batch_Size, 4, Height / 8, Width / 8) it means "Encoder output shape"
        
        x /= 0.18125
        
        for module in self:
            x = module(x)
            
        # (Batch_size, 3, Height, Width)
        return x