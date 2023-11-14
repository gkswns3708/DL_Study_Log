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