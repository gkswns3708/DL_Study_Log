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
            
            # (Batch_size, 128, Hieght, Width) -> (Batch_size, 128, Hieght, Width)
            VAE_ResidualBlock(128, 128), # Combination for Convolution and Normalization layer
            
            # (Batch_size, 128, Hieght, Width) -> (Batch_size, 128, Hieght, Width)
            VAE_ResidualBlock(128, 128), # Combination for Convolution and Normalization layer
            
            # 이전 layer의 output이 128channel이므로 input channel이 128
            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kenel_size=3, stride=2, padding=0),

            # (Batch_size, 128, Hieght / 2, Width / 2) -> (Batch_size, 256, Hieght / 2, Width / 2)
            VAE_ResidualBlock(128, 256), # Combination for Convolution and Normalization layer
        
            # (Batch_size, 256, Hieght / 2, Width / 2) -> (Batch_size, 256, Hieght / 2, Width / 2)
            VAE_ResidualBlock(256, 256), # Combination for Convolution and Normalization layer
            
            # (Batch_size, 256, Hieght / 2, Width / 2) -> (Batch_size, 256, Hieght / 4, Width / 4)
            nn.Conv2d(256, 256, kenel_size=3, stride=2, padding=0),
            
            # (Batch_size, 256, Hieght / 4, Width / 4) -> (Batch_size, 512, Hieght / 4, Width / 4)
            VAE_ResidualBlock(256, 512), # Combination for Convolution and Normalization layer
        
            # (Batch_size, 512, Hieght / 4, Width / 4) -> (Batch_size, 512, Hieght / 4, Width / 4)
            VAE_ResidualBlock(512, 512), # Combination for Convolution and Normalization layer
            
            # (Batch_size, 512, Hieght / 4, Width / 4) -> (Batch_size, 512, Hieght / 8, Width / 8)
            nn.Conv2d(512, 512, kenel_size=3, stride=2, padding=0),
            
            # (Batch_size, 512, Hieght / 8, Width / 8) -> (Batch_size, 512, Hieght / 8, Width / 8)
            VAE_ResidualBlock(512, 512), # Combination for Convolution and Normalization layer
            
            # (Batch_size, 512, Hieght / 8, Width / 8) -> (Batch_size, 512, Hieght / 8, Width / 8)
            VAE_ResidualBlock(512, 512), # Combination for Convolution and Normalization layer
            
            # (Batch_size, 512, Hieght / 8, Width / 8) -> (Batch_size, 512, Hieght / 8, Width / 8)
            VAE_ResidualBlock(512, 512), # Combination for Convolution and Normalization layer
            
            # (Batch_size, 512, Hieght / 8, Width / 8) -> (Batch_size, 512, Hieght / 8, Width / 8)
            VAE_AttentionBlock(512), # Self-Attention over each-pixel 
            
            # (Batch_size, 512, Hieght / 8, Width / 8) -> (Batch_size, 512, Hieght / 8, Width / 8)
            VAE_ResidualBlock(512, 512), # Combination for Convolution and Normalization layer
            
            # (Batch_size, 512, Hieght / 8, Width / 8) -> (Batch_size, 512, Hieght / 8, Width / 8)
            # TODO : Group Normalization 동작 방식 이해하기.
            nn.GroupNorm(32, 512),
            
            # (Batch_size, 512, Hieght / 8, Width / 8) -> (Batch_size, 512, Hieght / 8, Width / 8)
            # there is no particular reason for useing SiLU Activation Function.
            # The paper said that the SiLU Activation Function has better performance.
            nn.SiLU(), # derived from the sigmoid linear unit x / (1 + e^{-x})
            
            # (Batch_size, 512, Hieght / 8, Width / 8) -> (Batch_size, 8, Hieght / 8, Width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # (Batch_size, 8, Hieght / 8, Width / 8) -> (Batch_size, 8, Hieght / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    
    # VAE에서 중요한 것은 Latent Space
    # VAE에서 encoding하는 것은 단순히 image의 정보 뿐만 아니라, distribution 그자체, Multivariate Gaussian Distribution
    # 그래서 Decoder는 방금 우리가 학습했던 Multivariate Guassian Distribution에서 sampling을 하면 어떤 image를 생성시킬 수 있는 것.
    # Latent Vector가 이러한 Multivariate Gaussian Distribution속에서 생성되는 것을 학습하기 위해서는 이러한 Distribution의 Mean과 Variance(정확히는 Log variance)를 학습한다는 것.
    # 
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor():
        # x : (Batch, Channel=3, Height=512, Width=512)
        # noise : (Batch_size, Out_Channels, Height / 8, Width / 8) # noise의 크기는 output의 크기(shape)와 같다 
        
        for module in self:
            # 중간의 image의 크기를 / 2하는 convoltuion layer이다. 
            if getattr(module, 'stride', None) == (2, 2):  
                # (Padding_Left, Padding Right, Padding_Up, Padding_Bottom), 1인 곳에만 padding을 달아준다는 뜻.
                # TODO : 왜 오른쪽과 아래에만 padding을 달아주는 것인가.
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        # (Batch_size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_size, 4, Height / 8, Width / 8)
        # Encoder의 Output을 2등분으로 쪼개서 Mean과 Variance을 예측하도록 하는 듯.
        mean, log_variance = torch.chunk(x, 2, dim=1) 
        
        # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 4, Height / 8, Width / 8)
        # 크기를 -30에서 20사이로 조절함. -30이하면 -30, 20이상이면 20, 그 외에는 본연의 값을 유지합니다. 
        log_variance  = torch.clamp(log_variance, -30, 20)
        
        # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        
        # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt() 
        
        # Z = N(0, 1) -> N(mean, variance) = X?
        # X = mean + stdev * z 
        x = mean + stdev * noise
        
        # Scale the output by a constant -> There is no historical reason. But without this constant, performance is lower than it exists.
        x *= 0.18125
        
        return x
        
        