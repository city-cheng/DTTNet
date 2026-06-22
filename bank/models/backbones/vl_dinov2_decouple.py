from mmseg.models.builder import BACKBONES, MODELS

from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import torch.nn.functional as F
import torch
from torch import nn
from utils.diff_utils import *
from .moe_adapter import TemporalExpert

class LinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank,bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_features, rank) * std_dev)
        self.B = nn.Parameter(torch.randn(rank, out_features) * std_dev)
    def forward(self, x):
        return 0.1 * (x @ self.A @ self.B)


@BACKBONES.register_module()
class VLDinoVisionTransformerDecouple(DinoVisionTransformer):
    def __init__(
        self,
        adapter_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adapters = nn.ModuleList([MODELS.build(adapter_config) for i in range(4)])
        self.temporal_adapters = nn.ModuleList(
            [TemporalExpert(1024, 256) for i in range(20)]
        )



    def forward_features(
        self,
        x,
        l_shadow,
        l_bg,
        gt=None,
        masks=None,
    ):
        # x: (1, C, 32, 32)  feats: [(T, C, 32, 32),...] bound_feats: (T, C, 128, 128) coarse_mask: (5,1,7,7) l_shadow: (5,20,1024)
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size

        x = self.prepare_tokens_with_masks(x, masks)

        outs = []
        i = 0
        j = 0
        bodys = []
        # bounds = []
        for idx, blk in enumerate(self.blocks):
            if idx in self.out_indices:
                x = blk(x)
                x,body = self.adapters[i].forward(
                    x,
                    l_shadow,
                    l_bg,
                    # idx,
                    # coarse_mask,
                    # feats[i],
                    has_cls_token=True,
                )  # 1,N,1024
                i += 1
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
                bodys.append(body)
            else:
                x = x + self.temporal_adapters[j](x)
                x = blk(x)
                j += 1


        # return outs,bounds
        return outs,bodys  # [(1,1024,32,32),...]

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(
            self,
            [
                "adapters",
                "loras"
            ],
        )
        set_train(
            self,
            [
                "adapters",
                "loras"
            ],
        )

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "adapter" not in k and "loras" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
