from mmseg.models.builder import MODELS
import einops
import torch
from torch import concat
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor


class DWConv(nn.Module):
    def __init__(self, H, W, dim=768, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=dim,
        )
        self.h = H
        self.w = W

    def forward(self, x):
        is_3d = len(x.shape) == 3
        if is_3d:
            B, N, C = x.shape
            x = x.transpose(1, 2).view(B, C, self.h, self.w).contiguous()
        x = self.dwconv(x)
        if is_3d:
            x = x.flatten(2).transpose(1, 2)

        return x



class ConvResBlock(nn.Module):
    def __init__(self, hidden_dims,up=4) -> None:
        super().__init__()
        self.down = nn.Conv2d(1024,hidden_dims,1)
        self.conv1 = nn.Conv2d(hidden_dims,hidden_dims,3,1,1)
        self.up = nn.UpsamplingNearest2d(scale_factor=up)
    def forward(self,x):
        x = self.up(x)
        identity = x
        x = self.conv1(self.down(x)) + identity
        return x



class TemporalExpert(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        hidden_dims: int,
        scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.scale_init = scale_init
        self.hidden_dims = hidden_dims

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.down = nn.Sequential(nn.Conv2d(embed_dims, hidden_dims, 1))
        self.up = nn.Sequential(nn.Conv2d(hidden_dims, embed_dims, 1))

        self.x2token = nn.MultiheadAttention(
            hidden_dims, num_heads=8, dropout=0.0, batch_first=True
        )
        self.token2x = nn.MultiheadAttention(
            hidden_dims, num_heads=8, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dims)
        # self.norm2 = nn.LayerNorm(hidden_dims)
        # self.norm3 = nn.LayerNorm(hidden_dims)

        learnable_tokens_a = nn.Parameter(torch.empty([1, 20, 64])).cuda()
        learnable_tokens_b = nn.Parameter(torch.empty([1, 64, self.hidden_dims])).cuda()
        val = math.sqrt(
            6.0 / float(3 * reduce(mul, (32, 32), 1) + (self.hidden_dims * 32) ** 0.5)
        )
        nn.init.uniform_(learnable_tokens_a.data, -val, val)
        nn.init.uniform_(learnable_tokens_b.data, -val, val)
        self.tokens = learnable_tokens_a @ learnable_tokens_b  # 1,20,512

        self.scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(
        self,
        feats: Tensor,
    ) -> Tensor:

        cls_token, feats = torch.tensor_split(feats, [1], dim=1)
        # feats: bt,N,512  past_feats: T,hidden_dims,32,32 l_shadow: b,L,512 l_bg: b,L,512  coarse_mask: 5,1,7,7
        t = 5
        b = feats.shape[0] // t
        N = feats.shape[1]
        feats = feats.reshape(b * t, 32, 32, -1).permute(0, 3, 1, 2)  # bt, c, h, w
        feats = self.down(feats)
        temproral_feat = self.pool(feats).reshape(b, t, -1)  # 1, 5, 512
        tokens = self.x2token(self.tokens, temproral_feat, temproral_feat)[
            0
        ]  # 1, 20, 512
        tokens = self.norm1(tokens)
        feats = (
            feats.reshape(b, t, self.hidden_dims, 32, 32)
            .permute(0, 3, 4, 1, 2)
            .flatten(0, 2)  
        )  # bN, 5, 512
        tokens = tokens.repeat(N, 1, 1)  # bN, 20, 512
        temproral_feat = self.token2x(feats, tokens, tokens)[0]  # bN, 5, 512
        temproral_feat = (
            temproral_feat.reshape(b, 32, 32, t, -1)
            .permute(0, 3, 4, 1, 2)
            .flatten(0, 1)
        )
        temproral_feat = self.up(temproral_feat).flatten(2).permute(0, 2, 1)
        # print(f"temproral_feat:{temproral_feat.shape},cls_token:{cls_token.shape}")
        temproral_feat = torch.cat([cls_token, temproral_feat], dim=1)
        return temproral_feat * self.scale

