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
    
class UNET_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        # time_embedding의 크기가 1280이므로 n_time = 1280
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            # residual connection시에 in_channel과 out_channel이 같으면 그냥 연결하면 됨.
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature : latent -> (Batch_Size, In_channels, Height, Width)
        # time : (1, 1280)
        
        residue = feature
        
        feature = self.groupnorm_feature(feature)
        
        feature = F.silu(feature)
        
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        
        time = self.linear_time(time)
        
        # Time embedding에는 Batch와 Channel에 대한 dimension이 없으므로 unsqueeze를 2번함.
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merged(merged)
        
        merged = F.silu(merged)
        
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
        
class UNET_AttentionBlock(nn.Module):
    
    def __init__(self, n_head: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_head * n_embed
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim(=768))
        # Input에 대해서 Transformer에서 했던 것처럼 latent(x)에 normalization을 적용 후 convolution함, 물론 Transformer에는 convolutio이 없긴 함 ㅋ
        
        residue_long = x
        
        x = self.groupnorm(x)
        
        x = self.conv_input(x)
        
        n, c, h, w= x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n,c , h * w))
        # cross attention을 적용하기 위해 transpose를 함.
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width,  Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self Attention with skip connection
        residue_short = x
        
        x = self.layernorm_1(x)
        self.attention_1(x) # 이 파트는 일반적인 Transformer Figure(MultiHead Attention)에서 보는 것처럼 진행됨.
        x += residue_short
        
        residue_short = x
        
        # Normalization + Cross Attention with skip connection
        x = self.layernorm_2(x)
        
        # Cross Attention
        self.attention_2(x, context) # 이 파트는 Cross Attention을 계산함. TODO: 이 파트가 명확하게 그림이 잘 안 그려짐, 다시 제대로 생각해봐야 할 듯.
        
        x += residue_short
        
        reidue_short = x
        
        # Normalization + FF(Feed Forward) with GeGLU and skip connection
        
        x = self.layernorm_3(x)
        
        # TODO: 실제 Stable diffusion에서 아래와 같이 구현되어 있다고 함. 확인해봐야 할 듯.
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        
        # element wise multiplication
        # 그리고 이러한 연산을 하는 이유는 이러한 application에서 더 성능이 좋았기 때문임. 다른 이유는 없음.
        x = x * F.gelu(gate) 
        
        x = self.linear_geglu_2(x)
        
        x += residue_short
        
        # (Batch_Size, Width * Height, Features) -> (Batch_Size, Features , Width * Height)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features , Width * Height) -> (Batch_Size, Features , Width, Height)
        x = x.view((n, c, h, w))
        
        return self.conv_output(x) + residue_long
        
        
        

class Upsample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) ->  (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
        

class UNET(nn.Module):
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
        

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.grouopnorm = nn.roupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, 320, Height / 8, Width / 8)
        
        x = self.groupnorm(x)
        
        x = F.silu(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
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