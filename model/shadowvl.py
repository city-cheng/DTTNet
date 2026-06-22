import einops
import torch
from torch import concat
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import BACKBONES, MODELS

from bank.models.backbones.utils import set_requires_grad
from utils.diff_utils import *

import open_clip

# from clip.clip import build_model, tokenize
import einops


class Projection(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x


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

        clip_features = self.backbone.forward_features(
            clip, l_shadow, l_bg, gt=gt
        )  # bt,c,h,w

        coarse_logits = self.decode_head(clip_features, l_shadow)  # bt,1,h,w

        return coarse_logits

    def encode_decode_batch(self, clip, l_shadow, coarse_mask=None, l_bg=None, gt=None):
        clip_features = self.backbone.forward_features(
            clip, l_shadow, coarse_mask, gt=gt
        )  # bt,c,h,w
        x = self.decode_head(inputs=clip_features, ops="feat")  # bt,1,h,w
        coarse_logits = self.decode_head(x_i=x, ops="mask")
        return coarse_logits

    def encode_decode_with_memory(
        self, clip, l_shadow, l_bg, gt=None, clear_memory=True
    ):
        # clip: bt,c,h,w
        logits = []
        bounds = []
        if clear_memory:
            self.clear_memory()
        for i in range(clip.shape[0]):
            clip_i = clip[i : i + 1]  # 1, c, h, w
            clip_features_i = self.backbone.forward_features(
                clip_i,
                l_shadow,
                l_bg,
                [self.feats_1, self.feats_2, self.feats_3, self.feats_4],
                self.bound_feats,
                gt=gt,
            )  # 1,c,h,w
            x_i = self.decode_head(
                inputs=clip_features_i, ops="feat"
            )  # 1,embedding_dim,H,W

            logits_i, bounds_i, fine_bound_feats_i = self.decode_head(
                x_i=x_i, ops="mask"
            )
            # bounds.append(bounds_i)
            logits.append(logits_i)
            self.update_memory(
                clip_features_i[0],
                clip_features_i[1],
                clip_features_i[2],
                clip_features_i[3],
                fine_bound_feats_i,
            )
        # bounds = torch.concat(bounds, dim=0)
        logits = torch.concat(logits, dim=0)
        # return logits, bounds
        return logits, None

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

    def contrastive_loss(self, x, mask, l_shadow, temperature=0.07):
        """
        x: 特征图，形状为 (B, N, C) -> (5, 196, 512)
        mask: 阴影掩码，形状为 (B, 1, H, W) -> (5, 1, 14, 14)
        l_shadow: 阴影相关文本描述，形状为 (1, 512)
        temperature: 对比损失的温度系数
        """
        batch_size = x.size(0)
        # 将掩码展平为 (B, H*W)
        mask_flat = mask.view(batch_size, -1)  # (5, 196)

        # 收集阴影和非阴影区域的特征
        shadow_features = []
        non_shadow_features = []
        for i in range(batch_size):
            current_mask = mask_flat[i]  # (196)
            current_features = x[i]  # (196, 512)

            # 提取阴影特征
            shadow_mask = current_mask == 1
            shadow_feat = current_features[shadow_mask]  # (K_i, 512)
            shadow_features.append(shadow_feat)

            # 提取非阴影特征
            non_shadow_mask = current_mask == 0
            non_shadow_feat = current_features[non_shadow_mask]  # (196-K_i, 512)
            non_shadow_features.append(non_shadow_feat)

        # 拼接所有样本的特征
        x_shadow = torch.cat(shadow_features, dim=0)  # (total_shadow, 512)
        x_non_shadow = torch.cat(non_shadow_features, dim=0)  # (total_non_shadow, 512)

        # 处理无阴影或无非阴影的情况
        if x_shadow.size(0) == 0 or x_non_shadow.size(0) == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # 归一化特征
        l_shadow_norm = F.normalize(l_shadow, p=2, dim=1)  # (1, 512)
        x_shadow_norm = F.normalize(x_shadow, p=2, dim=1)  # (total_shadow, 512)
        x_non_shadow_norm = F.normalize(
            x_non_shadow, p=2, dim=1
        )  # (total_non_shadow, 512)

        # 计算相似度 (余弦相似度)
        s_pos = torch.mm(l_shadow_norm, x_shadow_norm.T)  # (1, total_shadow)
        s_neg = torch.mm(l_shadow_norm, x_non_shadow_norm.T)  # (1, total_non_shadow)

        # 计算对比损失
        sum_neg = torch.exp(s_neg / temperature).sum()
        numerators = torch.exp(s_pos / temperature)  # (1, total_shadow)
        denominators = numerators + sum_neg  # 分母 = 正样本相似度 + 负样本总相似度
        # print(f"sum_neg: {sum_neg}, numerators: {numerators}, denominators: {denominators}")

        # 计算每个正样本的损失并取平均
        losses = -torch.log(numerators / (denominators + 1e-8))  # 防止除零
        total_loss = losses.mean()

        return total_loss
