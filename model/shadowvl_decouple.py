import einops
import torch
from torch import concat
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import BACKBONES, MODELS

from bank.models.backbones.utils import set_requires_grad
from utils.diff_utils import *

import open_clip

import einops



class ShadowVL(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.backbone = BACKBONES.build(config["backbone"])
        self.decode_head = MODELS.build(config["decode_head"])
        self.vl_match = MODELS.build(config["vl_match"])

        self.clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="/root/models/laionCLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin",
        )


        self.clip_model.cuda()
        self.clip_model.eval()
        self.clip_model.visual.output_tokens = True
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.config = config
        self.num_frames = config["num_frames"]
        self.batch_size = config["batch_size"]

        self.backbone.init_weights()
        self.decode_head.init_weights()


        set_requires_grad(
            self,
            [
                "vl_match",
                "decode_head",
                "adapters",
                "loras",
            ],
            False,
        )


    def forward(self, images, gt=None, text=None, is_train=True, clear_memory=True):
        # images:b,t,3,h,w       gt: b,t,1,h,w
        b, t, _, h, w = images.shape
        clip = images.flatten(0, 1)  # bt,3,h,w
        if gt != None:
            gt = gt.flatten(0, 1)  # bt,1,h,w
        # text = text["shadow"]
        # print(f"text:{text}")
        clip_input = F.interpolate(clip, size=(224, 224), mode="bilinear")

        _, tokens = self.clip_model.encode_image(clip_input)
        text_features = self.encode_text(self.clip_model, text)
        l_shadow, l_bg = text_features[0].unsqueeze(0), text_features[1].unsqueeze(0)
        # print(f"tokens: {tokens.shape}, l_shadow:{l_shadow.shape},l_bg:{l_bg.shape}")

        l_shadow, l_bg = self.vl_match(tokens, l_shadow, l_bg)

        clip_features,bodys = self.backbone.forward_features(
            clip, l_shadow, l_bg, gt=gt
        )  # bt,c,h,w

        coarse_logits,details = self.decode_head(clip_features, l_shadow)  # bt,1,h,w

        return coarse_logits,bodys,details

    def encode_text(self, model, text):
        text = self.tokenizer([text["shadow"][0], text["gray"][0]]).cuda()
        cast_dtype = model.transformer.get_cast_dtype()

        x = model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + model.positional_embedding.to(cast_dtype)
        x = model.transformer(x, attn_mask=model.attn_mask)
        x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return x

    def update_memory(self, feat_1, feat_2, feat_3, feat_4, bound_feat, bound=None):
        """bounds: T, 1, 128, 128  bound_feats: T, 512, 128, 128
        bound: 1, 1, 128, 128  bound_feat: 1, 512, 128, 128

        """
        if self.bound_feats is None:
            self.bound_feats = bound_feat
            self.feats_1 = feat_1
            self.feats_2 = feat_2
            self.feats_3 = feat_3
            self.feats_4 = feat_4
        self.bound_feats = concat([self.bound_feats, bound_feat], dim=0)
        self.feats_1 = concat([self.feats_1, feat_1], dim=0)
        self.feats_2 = concat([self.feats_2, feat_2], dim=0)
        self.feats_3 = concat([self.feats_3, feat_3], dim=0)
        self.feats_4 = concat([self.feats_4, feat_4], dim=0)
        # print(f"feats_1:{self.feats_1.shape},feats_2:{self.feats_2.shape},feats_3:{self.feats_3.shape},feats_4:{self.feats_4.shape}")

        if self.bound_feats.shape[0] > 3:
            self.bound_feats = self.bound_feats[1:]
            self.feats_1 = self.feats_1[1:]
            self.feats_2 = self.feats_2[1:]
            self.feats_3 = self.feats_3[1:]
            self.feats_4 = self.feats_4[1:]

    def clear_memory(self):
        self.bound_feats = None
        self.feats_1 = None
        self.feats_2 = None
        self.feats_3 = None
        self.feats_4 = None
