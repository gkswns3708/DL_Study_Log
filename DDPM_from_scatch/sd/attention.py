import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) # W_{Q, K, V}, 이후에 Q, K, V로 분리될 예정.
        self.out_proj = nn.Linear(d_embed, d_embed) # W_0 Weight
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x : torch.Tensor, causal_mask=False):
        # x : (Batch_size, Seq_Len, Dim)
        
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, Dim * 3) -> 3 tensors of shape (Batch_size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(x, dim=-1)
        
        # TODO : 왜 transpose (1, 2)를 하는 것인가?
        # (Batch_size, Seq_len, Dim) -> (Batch_size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)
        
        # (Batch_Size, H, Seq_Len, Seq_len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask Where the upper triangle (above the principal diagonal() is made up of 1)
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_size, H, Seq_Len, Seq_Len) @ (Batch_size, H, Seq_Len, Dim / H) -> (Batch_size, H, Seq_Len, Dim / H)
        output = weight @ v
        
        # (Batch_size, H, Seq_Len, Dim / H) -> (Batch_size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        # (Batch_Size, Seq_Len, Dim)
        return output
        
class CrossAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x : (latent): (Batch_Size, Seq_Len_Q, Dim_Q)
        # y : (contet) : (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
        
        input_shape = x.shape
        batch_Size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_Size, -1, self.n_heads, self.d_head)
        
        # Multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqert(self.d_head)
        
        # 여기서는 Masking과정을 하지 않는데, 그 이유는 단순하게도 image와 prompt간의 attention을 계산하기에, next word를 보면 안되는 그럼 개념이 없다.
        weight = F.softmax(weight, dim = -1)
        
        output = weight @ v
        
        output = output.transpose(1, 2).contiguous()
        
        output = output.view(input_shape)
        
        output = self.out_proj(output)
        
        return output