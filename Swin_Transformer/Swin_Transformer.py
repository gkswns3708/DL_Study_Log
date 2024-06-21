# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) 
    # 위에서 x.permute를 0, 1, 3, 2, 4, 5로 하는 이유는 H // 7, W // 7이 7 x 7 개만큼을 차원을 통해 남기기 위해서 이렇게 구현함.
    # windows.shape -> (B * (H / window_size) * (W / window_size), window_size, window_size, C) 
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# WindowAttention은 Window내의 Multi-Head Attention을 의미한다.
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim # embed_dim은 96이다. layer가 0일 때는 96, 1일 때는 192, 2일 때는 384, 3일 때는 768이다.
        self.window_size = window_size  # Wh, Ww, 보통은 (7, 7)이다.
        self.num_heads = num_heads # num_heads는 [3, 6, 12, 24]이다.
        head_dim = dim // num_heads # 각 head가 내뱉어야 하는 dim은 dim // num_heads로 계산된다.
        # 해당 코드의 역할은 query 값을 scaling하여 분산을 줄여주어 학습의 안정성을 주고, 원래 기본 Attention Mechanism에서 사용되는 것이다.
        self.scale = qk_scale or head_dim ** -0.5 # 아래의 q = q * self.scale이라는 코드에서 사용된다.

        # define a parameter table of relative position bias
        # X, Y축을 따라서 [-M + 1, M - 1] 범위에 있기 때문에 해당 크기를 가지는 (2M - 1) * (2M - 1) 크기의 테이블을 만들어준다.
        # 이거는 bi-cubic interpolation을 통해 다른 window 크기의 fine-tuning을 위한 모델을 초기화 하는 데에도 사용할 수 잇습니다.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # 아래 내용을 이해하기 위한 링크 : https://www.youtube.com/watch?v=Ws2RAh_VDyU
        coords_h = torch.arange(self.window_size[0]) # Wh -> [0, 1, 2, 3, 4, 5, 6]
        coords_w = torch.arange(self.window_size[1]) # Ww -> [0, 1, 2, 3, 4, 5, 6]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww -> [[0, 0], [0, ]]
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 아래 연산은 각각의 좌표에 대해서 상대적인 좌표를 계산하는 연산이다.
        # 그래서 총 2 x 49 x 49의 텐서로, 이는 각  위치 쌍 간의 행과 열 차이를 나타냅니다.
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # permute & contiguous를 통해 차원 축을 맞추고,
        # 해당 연산 이후 shape이 (window_size, window_size, 2)인데, relative_coords[(Y * X) + X] = 49개의 [dx, dy]로 표현됩니다.
        # 즉, 특정 위치 Y, X가 정해지면 window_size * window_size갯수 만큼의 상대 위치를 표현할 수 있습니다.
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 양수로 연산하기 위해 M - 1을 더해줌(최소값이 -(M - 1)이 되도록))
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # TODO : 이거 왜 하는건지 아는 사람?
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 가장 마지막 축을 기준으로 합쳐서 최종 상대 위치 index를 계산합니다. 결과적으로 49 x 49 행렬이 생성되며
        # 이 행렬은 윈도우 내 각 토큰 쌍 간의 상대적 위치를 고유하게 식별하는 인덱스를 제공합니다.
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # TODO: self.register_buffer의 역할은 무엇인가?
        # Answer : register_buffer는 모델의 state_dict에 저장되지만, 학습 중에 업데이트되지 않는 매개변수를 저장하는 데 사용됩니다.
        # self.register_buffer nn.Module에 내장된 class method입니다.
        self.register_buffer("relative_position_index", relative_position_index)

        # Attention 계산을 위한 dim -> 3 * dim으로 확장하는 연산(이후에는 3개로 나누어서 q, k, v로 사용됨)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.relative_position_bias_table을 std=0.02를 기준으로 truncation 한다고 생각함.
        # TODO : 왜 .02인지는 의문
        trunc_normal_(self.relative_position_bias_table, std=.02) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # self.qkv(x).shape -> (B, N, 3 * C) -> (B, N, 3, num_heads, C // num_heads) -> (3(q, k, v), B, num_heads, N, C // num_heads)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale # head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) # @는 단순히 내적(inner product)을 의미하고, 연산에 사용되는 행렬만을 transpose하기 위해 (-2, -1)을 사용합니다.
        # relative_position_index는 각 토큰 쌍 간의 상대적 위치를 고유하게 식별하는 인덱스를 제공합니다.
        # self.relative_position_index.view(-1)의 shape은 49이며, 
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


# 즉, SwinTransformerBlock자체는 Figure 3-(b) 중에서 1개의 구조를 가지는 Class이다.
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        # 예상되는 dim은 768, input_resolution은 (56, 56), num_heads는 12, window_size는 7, shift_size는 0 or window_size // 2, mlp_ratio는 4.0
        self.dim = dim # embed_dim은 96이다. layer가 0일 때는 96, 1일 때는 192, 2일 때는 384, 3일 때는 768이다.
        self.input_resolution = input_resolution # patches_resolution은 (56, 56), (28, 28), (14, 14), (7, 7)이다.
        self.num_heads = num_heads  # num_heads는 [3, 6, 12, 24]이다. 이것은 Swin Transformer Block내의 Multi-Head Attention의 num_heads를 나타낸다.
        self.window_size = window_size # window_size는 7이다.
        self.shift_size = shift_size # 여기서 짝수 번째는 0 이고 홀수 번째는 window_size // 2만큼 shifting을 한다.
        self.mlp_ratio = mlp_ratio # mlp_ratio는 4.0이다.
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim) # 이게 Figure 3-(b)에서 첫 번째 LayerNorm을 의미한다.
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, #
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.shit_size > 0이라는 뜻은 SW-MSA라는 것이고, SW-MSA를 할 때 notion의 그림과 같이 Masking 과정이 추가됩니다.
        # window_size // 2
        if self.shift_size > 0: 
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution # patches_resolution은 (56, 56), (28, 28), (14, 14), (7, 7)이다.
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size), # [0, 49)
                        slice(-self.window_size, -self.shift_size), # [49 -> 53)
                        slice(-self.shift_size, None)) # [53 -> 56)
            w_slices = (slice(0, -self.window_size), # [0, 49)
                        slice(-self.window_size, -self.shift_size), # [49 -> 53)
                        slice(-self.shift_size, None)) # [53 -> 56)
            # TODO: cnt는 왜 저장함?
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # nW, window_size, window_size
            # (nW, 1, window_size * window_size) - (nW, window_size * window_size, 1)의 연산으로, 최종적으로
            # (nW, window_size * window_size, window_size * window_size)형태의 attn_mask로 계산되고
            # 이는 각 윈도우 내의 모든 토큰 쌍간의 상대적 위치 차이를 나타냅니다.
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) 
            # Masking을 위해 -100을 넣는 이유는, softmax를 통해 확률을 구할 때, 0이 되는 것을 방지하기 위함이다.
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        # self.fused_window_process는 윈도우 기반 어텐션 메커니즘의 특정 처리 단계들을 통합(fuse)하여 실행하는 것을 의미합니다.
        # fused_window_process가 True이면, WindowProcess를 사용하고, False이면 WindowPartition을 사용합니다.
        # 아마 일반적인 실험에서는 False를 사용할 듯 하다.
        # 그리고 libaray가 잘 install 되어 있으면 사용될 듯 합니다.
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x) # layer_norm
        x = x.view(B, H, W, C) # B H W C의 값은 1, 56, 56, 768

        # cyclic shift
        # 아래에서 사용되는 nW는 H x W / window_size^2 이다.
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        # TODO : 이미 .view를 하기 전에 shape이 완성되어 있을텐데, 왜 아래 코드가 필요하지?
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        # W-MSA와 SW-MSA의 차이점은 shift_size가 0이냐 아니냐의 차이이다.
        # W-MSA는 mask가 없지만, 
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim # Patch Embedding의 예상 shape은 (B, 56 x 56, 96)이다. 고로 dim은 96, 192, 384, 768이다.
        self.input_resolution = input_resolution # (56, 56), (28, 28), (14, 14), (7, 7)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 # 여기서 짝수 번째는 0 이고 홀수 번째는 window_size // 2만큼 shifting을 한다.
                                 shift_size=0 if (i % 2 == 0) else window_size // 2, 
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process) for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # img_size = (img_size, img_size)
        patch_size = to_2tuple(patch_size) # patch_size = (patch_size , patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] # 각 패치의 resolution
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # 여기서 self.proj는 Conv2d를 kernel_size를 patch_size, stride를 path_size로 했기 때문에
        # 각각의 patch를 96길이를 가지는 embedding으로 변환하는 연산임.
        # 그래서 224 X 224의 image에 4 x 4 kernel을 이용해 연산을 했기 때문에 56 x 56이 생기고 output_channel이 96이므로
        # (B, 3, 224, 224) -> (B, 96, 56, 56) shape으로 변환됨.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 여기서 shape이 flatten(2)와 transpose(1, 2)를 거치면서 
        # (B, 3, 224, 224) -> (B, 96, 56, 56) -> (B, 96, 56 x 56) -> (B, 56 x 56, 96)이 됨.
        # 즉, 한 이미지에 대한 embedding의 크기는 1 x 56 x 56 x 96으로써 설명됨.
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02) # 평균이 0이고 분산이 1이고, -2 ~ 2 사이의 값만 사용하도록 truncation함.

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), # embed_dim은 96이다. i_layer가 0일 때는 96, 1일 때는 192, 2일 때는 384, 3일 때는 768이다.
                               input_resolution=(patches_resolution[0] // (2 ** i_layer), # patches_resolution은 (56, 56), (28, 28), (14, 14), (7, 7)이다.
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer], # depths는 [2, 2, 6, 2]이다. 이것은 Swin Transformer Block을 몇 번 반복할 것인지 나타낸다.
                               num_heads=num_heads[i_layer], # num_heads는 [3, 6, 12, 24]이다. 이것은 Swin Transformer Block내의 Multi-Head Attention의 num_heads를 나타낸다.
                               # TODO : W-MSA(Window Multi-Head Self Attention)의 num_heads와 SW-MSA()
                               window_size=window_size, # window_size는 7이다.
                               mlp_ratio=self.mlp_ratio, # mlp_ratio는 4.0이다.
                               qkv_bias=qkv_bias, qk_scale=qk_scale,  
                               drop=drop_rate, attn_drop=attn_drop_rate, 
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, # downsampling을 위한 method는 PatchMerging 방식이다.
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops