import einops
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmseg.models.utils import resize

from mmseg.models.builder import MODELS

from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from mmseg.models.utils import *

from .dysample import DySample
from torch.nn.functional import sigmoid



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


@MODELS.register_module()
class UpPromptHeadDecouple(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, decoder_params, **kwargs):
        super(UpPromptHeadDecouple, self).__init__(
            input_transform="multiple_select", **kwargs
        )

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

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.up_convs_detail = nn.ModuleList()
        H, W = 32, 32
        for _ in range(2):
            self.up_convs_detail.append(
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
                    nn.Upsample(scale_factor=2),
                )
            )
            H, W = H * 2, W * 2
        self.up_convs_body = nn.ModuleList()
        H, W = 32, 32
        for _ in range(2):
            self.up_convs_body.append(
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
                    nn.Upsample(scale_factor=2),
                )
            )
            H, W = H * 2, W * 2

        self.linear_pred = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim, 3, 1, 1),
            nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1),
        )

        self.body_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.detail_head = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type="BN", requires_grad=True),
        )

        # self.gru = MODELS.build(decoder_params["neck"])

    def forward(self, inputs, l_shadow=None, l_bg=None, gt=None):
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
        
        x = self.dropout(_c)

        for i, up_conv in enumerate(self.up_convs_body):
            x_body = up_conv(x)

        for i, up_conv in enumerate(self.up_convs_detail):
            x_detail = up_conv(x)

        x_body = self.dropout(x_body)
        x_detail = self.dropout(x_detail)
        logits = self.linear_pred(torch.cat([x_body, x_detail], dim=1))
        body = self.body_pred(x_body)
        detail = self.detail_head(x_detail)
        

        return logits, body, detail
        # return logits
