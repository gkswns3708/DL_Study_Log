import torch 
from torch import nn
from torch.nn import function as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    # TODO : nn.Sequential에 대한 이해.
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Height=512, Width=512) -> (Batch_size, 128, Height=512, Width=512)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
        )
        