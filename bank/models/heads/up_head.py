import einops
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import concat

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.models.utils import resize

from mmseg.models.builder import MODELS

from mmseg.models.decode_heads.decode_head import BaseDecodeHead


from mmseg.models.utils import *

from .boudary_adapter import BoundaryAttn

from .dysample import DySample

# from bank.models.heads import ShadowGRU


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, H, W, dim=768, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )
        self.h = H
        self.w = W

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.h, self.w).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(
        self,
        H,
        W,
        in_features,
        hidden_features=256,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.out_features = out_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(H, W, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, self.out_features, H, W)
        return x


@MODELS.register_module()
class UpHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, decoder_params, **kwargs):
        super(UpHead, self).__init__(input_transform="multiple_select", **kwargs)

        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

        embedding_dim = decoder_params["embed_dim"]
        self.embeding_dim = embedding_dim
        dySample_style = decoder_params["dySample_style"]
        dyscope = decoder_params["dyscope"]
        kernel_size = decoder_params["kernel_size"]
        hidden_dims = decoder_params["hidden_dims"]

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.up_convs = nn.ModuleList()
        self.up_convs_bound = nn.ModuleList()
        H, W = 32, 32
        for _ in range(3):
            self.up_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size - 1) // 2,
                        norm_cfg=self.norm_cfg,
                        # act_cfg=self.act_cfg,
                    ),
                    # DySample(
                    #     in_channels=embedding_dim,
                    #     scale=2,
                    #     style=dySample_style,
                    #     dyscope=dyscope,
                    # ),
                    nn.Upsample(
                        scale_factor=2,
                        mode='bilinear',
                    ),
                )
            )
            # self.up_convs_bound.append(
            #     nn.Sequential(
            #         ConvModule(
            #             in_channels=512,
            #             out_channels=512,
            #             kernel_size=kernel_size,
            #             stride=1,
            #             padding=int(kernel_size - 1) // 2,
            #             norm_cfg=self.norm_cfg,
            #             # act_cfg=self.act_cfg,
            #         ),
            #         DySample(
            #             in_channels=512,
            #             scale=2,
            #             style=dySample_style,
            #             dyscope=dyscope,
            #         ),
            #     )
            # )
            H, W = H * 2, W * 2
        # self.simam = simam_module(embedding_dim)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # self.linear_pred = nn.Conv2d(hidden_dims, self.num_classes, kernel_size=1)
        # self.down = nn.Conv2d(embedding_dim, hidden_dims, kernel_size=3, padding=1)

        # self.bound_head = nn.Conv2d(512, self.num_classes, kernel_size=1)
        # self.bound_fuse = BoundaryFuse(768, 512)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type="BN", requires_grad=True),
        )

        # self.boundary_attn = BoundaryAttn(hidden_dims, patch_size=4)




    def forward(self, x_i=None,inputs=None,ops="feat"):
        if ops=="feat":
            return self.get_decode_feats(inputs)
        elif ops=="mask":
            return self.get_masks(x_i)
        else:
            raise NotImplementedError
    def get_decode_feats(self, inputs, l_shadow=None, l_bg=None, gt=None):
        # print(f"inputs:  {inputs[0].shape},{len(inputs)}")
        # self.gru.clear_state()
        x = self._transform_inputs(inputs)  # len=4,
        c1, c2, c3, c4 = x  # B,C,H,W
        B, _, H, W = c4.shape
        C = self.embeding_dim
        t = 5

        c4 = (
            self.linear_c4(c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        )
        # c4 = resize(c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        c3 = (
            self.linear_c3(c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        )
        # c3 = resize(c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        c2 = (
            self.linear_c2(c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        )
        # c2 = resize(c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        c1 = (
            self.linear_c1(c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2], c1.shape[3])
        )

        _c = self.linear_fuse(
            torch.cat([c4, c3, c2, c1], dim=1)
        )  # BT,embedding_dim,H,W

        x = self.dropout(_c)  #  BT,embedding_dim,H,W
        return x # BT,embedding_dim,H,W

    def get_masks(self, x_i):
        # print(f"x_i: {x_i.shape}")
        # if self.bounds is not None:
        #     x_i = self.bound_fuse(x_i, self.bounds, self.bound_feats)

        # x_bound_i = self.down(x_i)  # 1,512,H,W
        for i, up_conv in enumerate(self.up_convs):
            x_i = up_conv(x_i)
        # for i, up_conv in enumerate(self.up_convs_bound):
        #     x_bound_i = up_conv(x_bound_i)

        fine_mask_feats_i = self.dropout(x_i) # 1,embedding_dim,H,W

        logits_i = self.linear_pred(fine_mask_feats_i)

        # return logits_i, bounds_i, fine_bound_feats_i
        return logits_i
        # return logits


class BoundaryFuse(nn.Module):
    def __init__(self, embed_dims, hidden_dims):
        super(BoundaryFuse, self).__init__()
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims

        # self.encoder = BoundaryFeatureEncoder(hidden_dims)

        # self.brb = BAFM_BRB()

        self.down = nn.Sequential(
            nn.Linear(embed_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
        )
        self.down2 = nn.Sequential(
            nn.Linear(embed_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
        )
        self.linear_a1 = nn.Linear(hidden_dims, hidden_dims)
        self.linear_a2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear_b = nn.Linear(hidden_dims, hidden_dims)

        self.up = nn.Sequential(
            nn.Linear(self.hidden_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def forward(self, feat, bounds, bound_feats):
        # feat: 1, 768,32,32   bounds: T, 1, 32, 32  bound_feats: [(1, N, 768),(1, N, 768)]
        # print(f"**************feat: {feat.shape}, bounds: {bounds.shape}, bound_feats: {bound_feats[0].shape}")
        _, C, H, W = feat.shape
        x = feat.flatten(2, 3).permute(0, 2, 1)  # x: 1, 1024, 768
        # print(f"**************x: {x.shape}")
        x = self.down(x).squeeze(0)  # x:1024, 512
        bound_feats = torch.concat(bound_feats, dim=1) # bound_feats: 1, TN, 768
        bound_feats = self.down2(bound_feats)  # bound_feats: 1, TN, 512
        mask1 = bounds.sum(dim=0) > 0
        mask2 = ~mask1
        x = self.process_attention(x, bound_feats, mask1, mask2)

        x = self.up(x)
        x = x.permute(1, 0).reshape(1, C, H, W)
        x = feat + x

        return x  # x:1,768,32,32

    def masked_attention(self, q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def process_attention(self, A, B, mask1, mask2):
        """
        Args:
            A: shape [1024, 512]
            B: shape [1, TN, 512]
            mask1/mask2: shape [1, 32, 32]
        Returns:
            output: shape [B, 512, 32, 32]
        """
        # B_batch, _, _ = A.shape
        # print(f"**************A: {A.shape}, B: {B.shape}, mask1: {mask1.shape}, mask2: {mask2.shape}")

        # 展平空间维度并转置通道
        # B = B.flatten(0, 1)  # [T*N, 512]
        B = B.squeeze(0)  # [T*N, 512]
        mask1 = mask1.view(-1).bool()  # [1024]
        mask2 = mask2.view(-1).bool()

        # 初始化输出张量
        output = torch.zeros_like(A)

        # 提取当前样本的掩码区域
        a1 = A[mask1]  # [L1, 512]
        a2 = A[mask2]  # [L2, 512]
        b = B  # [T*N, 512]
        a1 = self.linear_a1(a1)
        a2 = self.linear_a2(a2)
        b = self.linear_b(b)

        # 第一级交叉注意力
        if a1.shape[0] > 0:
            c1 = self.masked_attention(a1, b, b)  # [L1, 512]
            output[mask1] = c1
        else:
            c1 = torch.zeros(0, 512, device=A.device)

        # 第二级交叉注意力
        if a2.shape[0] > 0 and c1.shape[0] > 0:
            c2 = self.masked_attention(a2, c1, c1)  # [L2, 512]
            output[mask2] = c2

        return output
