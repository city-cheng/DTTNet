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
from mmcv.cnn import ConvModule

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


@MODELS.register_module()
class SemanticAdapterDecouple(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        hidden_dims: int,
        num_frames: int,
        scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.scale_init = scale_init
        self.hidden_dims = hidden_dims

        self.shadow2x = nn.MultiheadAttention(
            hidden_dims, num_heads=8, dropout=0.0, batch_first=True
        )
        # self.norm1 = nn.LayerNorm(hidden_dims)
        self.bg2x = nn.MultiheadAttention(
            hidden_dims, num_heads=8, dropout=0.0, batch_first=True
        )

        self.down = nn.Sequential(
            nn.Linear(embed_dims, self.hidden_dims),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dims, self.hidden_dims),
            # nn.LayerNorm(self.hidden_dims),
        )
        # self.down2 = nn.Linear(embed_dims, hidden_dims)
        self.up = nn.Sequential(
            nn.Linear(self.hidden_dims, embed_dims),
            # nn.ReLU(),
            # nn.Linear(self.embed_dims, self.embed_dims),
            # nn.LayerNorm(self.hidden_dims),
        )

        self.sem_conv1 = ConvModule(
            in_channels=self.hidden_dims,
            out_channels=self.hidden_dims,
            kernel_size=3,
            padding=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type="BN", requires_grad=True),
        )
        self.sem_conv3 = nn.Conv2d(self.hidden_dims,1,1)
        # self.mlp_token2feat = nn.Linear(self.hidden_dims, self.hidden_dims)
        # self.mlp_delta_f = nn.Linear(self.hidden_dims, self.hidden_dims)

        self.shadow_scale = nn.Parameter(torch.tensor(self.scale_init))
        self.bg_scale = nn.Parameter(torch.tensor(self.scale_init))

    def forward(
        self,
        feats: Tensor,
        l_shadow: Tensor,
        l_bg: Tensor,
        has_cls_token=True,
    ) -> Tensor:

        # feats: bt,N,1024  past_feats: T,hidden_dims,32,32 l_shadow: b,L,512 l_bg: b,L,512  coarse_mask: 5,1,7,7
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=1)

        identity = feats
        feats = self.down(feats)  # feats_: bt,N,hidden_dims

        t = 5
        b = feats.shape[0] // t
        N = feats.shape[1]
        # shadow_feat = feats * coarse_mask  # bt,N,512
        feats = feats.reshape(b, t, N, -1).flatten(1, 2)  # B,TN,hidden_dims

        shadow_feat = self.shadow2x(feats, l_shadow, l_shadow)[0]  #  b,L,1024
        # shadow_feat = self.norm1(shadow_feat)

        bg_feat = self.bg2x(feats, l_bg, l_bg)[0]  # B,TN,hidden_dims
        # bg_feat = self.norm2(bg_feat)

        shadow_feat = shadow_feat.reshape(b, t, N, -1).flatten(0, 1)  # BT,N,hidden_dims
        bg_feat = bg_feat.reshape(b, t, N, -1).flatten(0, 1)  # BT,N,hidden_dims

        semantic_feat = self.sem_conv1(
            shadow_feat.reshape(b*t,32,32,-1).permute(0,3,1,2)
        )
        body = self.sem_conv3(semantic_feat)

        # body = self.sem_conv3(shadow_feat.reshape(b*t,32,32,-1).permute(0,3,1,2))

        # feats = identity + self.up(self.shadow_scale * shadow_feat + self.bg_scale * bg_feat)  # feats: bt,N,1024
        feats = identity + self.up(self.shadow_scale * semantic_feat.flatten(2).permute(0,2,1) +  self.bg_scale * bg_feat)  # feats: bt,N,1024

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=1)
        return feats,body


