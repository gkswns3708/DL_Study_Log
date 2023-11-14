import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x : (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        # (1, 1280)
        return x

class SwitchSequential(nn.Sequential):
    def forward(self, x : torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context) # compute the cross attention between our latents and the prompt
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
            

def UNET(nn.Module):
    
    def __init__(self):
        
        self.encoders = nn.Module([
            # (Batch_Size, 4, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_residualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> # (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_residualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_residualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_residualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_residualBlock(1280, 1280)),
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            
            UNET_AttentionBlock(8, 160),
            
            UNET_ResidualBlock(1280, 1280),
        )
        
        self.decoderrs = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_residualBlock(2560, 1280)),
            
            SwitchSequential(UNET_residualBlock(2560, 1280)),
            
            SwitchSequential(UNET_residualBlock(2560, 1280), Upsample(1280)),
            
            SwitchSequential(UNET_residualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_residualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            SwitchSequential(UNET_residualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_residualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_residualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            SwitchSequential(UNET_residualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])
        
        

class Diffusion(nn.Module):
    
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET() # TODO : UNET의 역할 -> 얼마나 noise가 있는지를 predict하는 것.
        self.final = UNET_OutputLayer(320, 4)
        
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.tensor):
        # latent : from VAE output (Batch_size, 4, Height / 8, Width / 8)
        # context : from CLIP output (Batch_Size, Seq_Len, Dim(=768))
        # time : (1, 320)
        
        # TODO : time은 왜 batch 개념이 없는가.
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # self.final은 Unet의 output을 Unet의 input과 size를 동일하게 하는 역할을 함(Channel까지).
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)
        
        # (Bathc, 4, Height / 8, Width / 8)
        return output