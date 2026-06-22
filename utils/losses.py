import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


from collections import defaultdict


class BBCEWithLogitLoss(nn.Module):
    """
    Balanced BCEWithLogitLoss
    """

    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1.0 - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss


class OrthoLoss(nn.Module):
    def __init__(self):
        super(OrthoLoss, self).__init__()

    def forward(self, pred, target):
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        pred_ = pred
        target_ = target
        ortho_loss = 0
        dim = pred.shape[1]
        for i in range(pred.shape[0]):
            # ortho_loss += torch.mean(torch.abs(pred_[i:i+1,:].mm(target_[i:i+1,:].t()))/dim)
            ortho_loss += torch.mean(
                (pred_[i : i + 1, :].mm(target_[i : i + 1, :].t())).pow(2) / dim
            )

        ortho_loss /= pred.shape[0]
        return ortho_loss


class EdgeLoss(nn.Module):
    def __init__(self, apply_sigmoid=True):
        super(EdgeLoss, self).__init__()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, edge_shadmask, edge_noshad):
        if self.apply_sigmoid:
            edge_shadmask = F.sigmoid(edge_shadmask)
            edge_noshad = F.sigmoid(edge_noshad)
        # can't for backward gradient computation
        # edge_shadmask[edge_shadmask >= 0.5] = 1
        # edge_shadmask[edge_shadmask < 0.5] =0
        # edge_noshad[edge_noshad >= 0.5] = 1
        # edge_noshad[edge_noshad < 0.5] = 0
        edge = edge_shadmask + edge_noshad
        # print(edge)
        edge[edge < 1] = 0
        edge[edge >= 1] = 1
        numerator = torch.sum(edge)
        denominator = torch.sum(1.0 - edge)
        loss = numerator / denominator
        return loss


class DiffLoss_2(nn.Module):
    def __init__(self):
        super(DiffLoss_2, self).__init__()
        self.l1 = nn.L1Loss()
        self.ortho = OrthoLoss()

    def forward(self, img, noshad, label, mask):
        # batch_size = noshad.size(0)
        mask = (mask > 0.5).type(torch.int64)
        # mask = torch.sigmoid(mask)
        img_noshad = img * (1 - mask)
        noshad_noshad = noshad * (1 - label)

        loss = self.l1(img_noshad, noshad_noshad)
        # loss += 0.01*self.ortho(img_shad, noshad_shad)
        return loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def gram_matrix(self, y):
        """Returns the gram matrix of y (used to compute style loss)"""
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)  # C和w*h转置
        gram = features.bmm(features_t) / (c * h * w)  # bmm 将features与features_t相乘
        return gram

    def forward(self, img, noshad):
        g_noshad = self.gram_matrix(img)
        g_shad = self.gram_matrix(noshad)

        loss = 0
        for i in range(g_shad.shape[0]):
            loss += torch.mean(torch.abs(g_noshad[i : i + 1, :] - g_shad[i : i + 1, :]))
        return loss


class ZeroLoss(nn.Module):
    def __init__(self):
        super(ZeroLoss, self).__init__()

    def forward(self, target):
        zero_loss = torch.mean(torch.abs(target))
        return zero_loss


class SoftDiceLoss(nn.Module):
    def __init__(self, activation="sigmoid"):
        super(SoftDiceLoss, self).__init__()
        if activation is None or activation == "none":
            self.activation_fn = lambda x: x
        elif activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation == "softmax2d":
            self.activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError(
                "Activation implemented for sigmoid and softmax2d 激活函数的操作"
            )

    def forward(
        self,
        pred,
        gt,
        smooth=1,
    ):
        r"""computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
        """

        pred = self.activation_fn(pred)

        N = gt.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)

        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        dice = (2 * intersection + smooth) / (unionset + smooth)
        dice = dice.sum() / N
        loss = 1 - dice
        return loss


class BootstrappedBCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.binary_cross_entropy_with_logits(input, target), 1.0

        raw_loss = F.binary_cross_entropy_with_logits(
            input, target, reduction="none"
        ).view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * (
                (self.end_warm - it) / (self.end_warm - self.start_warm)
            )
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


def isnan(x):
    return x != x


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def scotch_loss(out_1, out_2, temperature=0.1, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """

    # print(f'out1:{out_1.shape},out2:{out_2.shape}')

    out_1_dist = out_1
    out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size]
    # neg: [2 * batch_size]

    cov = torch.mm(out, out_dist.t().contiguous())
    # print(f'out:{out.shape},out_dist:{out_dist.shape},cov:{cov.shape}')
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # positive similarity: using out_1 with itself, so only consider the first batch_size elements
    pos = torch.exp(torch.sum(out_1 * out_1, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + pos + eps)).mean()

    return loss


class XMemLossComputer:
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss()
        self.bootstrap_bce = BootstrappedBCE(config["start_warm"], config["end_warm"])
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def compute(self, logit, gt, walks=None, it=None):
        """
        Args:
            logit: B,1,H,W
            pred: B,1,H,W  0/1 masks
            gt: B,1,H,W    0/1 masks
        """
        losses = defaultdict(int)

        losses["total_loss"] = 0
        if self.config["dice_loss"]:
            losses["dice_loss"] = 0
            losses["dice_loss"] += self.dice_loss(logit, gt)
            losses["total_loss"] += self.config["dice_weight"] * losses["dice_loss"]

        if self.config["bce_loss"]:
            losses["bce_loss"] = 0
            losses["bce_loss"] += self.bce_loss(logit, gt)
            losses["total_loss"] += self.config["bce_weight"] * losses["bce_loss"]

        if self.config["bootstrap_bce_loss"] and it != None:
            losses["bootstrap_bce_loss"] = 0
            bbce_loss, p = self.bootstrap_bce(logit, gt, it)
            losses["bootstrap_bce_loss"] += bbce_loss
            losses["total_loss"] += (
                self.config["bootstrap_bce_weight"] * losses["bootstrap_bce_loss"]
            )
        if walks != None and self.config["walk_loss"]:
            losses["walk_loss"] = 0
            for name, (A, target) in walks.items():
                logits = torch.log(A + 1e-20).flatten(0, -2)
                loss = self.ce_loss(logits, target).mean()
                # acc = (torch.argmax(logits, dim=-1) == target).float().mean()
                losses["walk_loss"] += loss
                # losses['walk_acc'] = acc
                losses["total_loss"] += (
                    self.config["walk_loss_weight"] * losses["walk_loss"]
                )

        return losses


class SDDNetLossComputer:
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.loss_edge = EdgeLoss()
        self.loss_l1 = nn.L1Loss()
        self.loss_l2 = nn.MSELoss()
        self.loss_ortho = OrthoLoss()
        self.loss_diff = DiffLoss_2()
        self.loss_style = StyleLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_zero = ZeroLoss()
        self.loss_cos = nn.CosineEmbeddingLoss()
        # self.loss_fn = BootstrappedBCE(config["start_warm"], config["end_warm"])
        self.loss_fn = BBCEWithLogitLoss()

    def compute(self, images, pred: dict, gt, it=None):
        """
        Args:
            logit: B,1,H,W
            pred: B,1,H,W  0/1 masks
            gt: B,1,H,W    0/1 masks
        """
        losses = defaultdict(int)
        loss1 = self.loss_l1(pred["logits_shadimg"], images)
        # loss2,p = self.loss_fn(pred["logits_shadmask"], gt,it)
        loss2 = self.loss_fn(pred["logits_shadmask"], gt)
        loss3 = self.loss_diff(
            images, pred["logits_noshad"], gt, pred["logits_shadmask"]
        )
        # loss4 = self.loss_cos(
        #     pred["f_low_shad"],
        #     pred["f_high_shad"],
        #     torch.ones([pred["f_low_shad"].size()[0]]).cuda(),
        # )
        loss5 = self.loss_cos(
            pred["f_low_shad"],
            pred["f_low_feat"],
            torch.ones([pred["f_low_shad"].size()[0]]).cuda(),
        )
        loss6 = self.loss_cos(
            pred["f_high_shad"],
            pred["f_high_feat"],
            torch.ones([pred["f_low_shad"].size()[0]]).cuda(),
        )
        loss7 = self.loss_ortho(pred["f_low_noshad"], pred["f_low_shad"])
        loss8 = self.loss_ortho(pred["f_high_noshad"], pred["f_high_shad"])

        loss_mask = loss2
        loss_shadimg = 0.2 * loss1
        loss_noshad = 0.2 * loss3
        loss_filter = 0.2 * (0.5 * loss5 + 0.5 * loss6 + 0.01 * loss7 + 0.01 * loss8)
        # print(f'loss5:{loss5},loss6:{loss6},loss7:{loss7},loss8:{loss8}')
        # print(f'pred["f_low_feat"]:{pred["f_low_feat"]},pred["f_high_feat"]:{pred["f_high_feat"]}')
        loss_total = loss_mask + loss_shadimg + loss_noshad + loss_filter

        losses["loss_mask"] = 0
        losses["loss_mask"] += loss_mask
        losses["loss_shadimg"] = 0
        losses["loss_shadimg"] += loss_shadimg
        losses["loss_noshad"] = 0
        losses["loss_noshad"] += loss_noshad
        losses["loss_filter"] = 0
        losses["loss_filter"] += loss_filter
        losses["total_loss"] = 0
        losses["total_loss"] += loss_total

        return losses


def margin_loss(sim_matrix, query_sim_matrix, m=0.5):

    l = 0.5 - abs(sim_matrix - query_sim_matrix)
    l = l.view(-1)
    g = torch.zeros(l.size()).to(l.device)

    margin = torch.max(g, l)
    # print(margin.max())
    return margin.max()


class SASLossComputer:
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss()
        # self.bootstrap_bce = BootstrappedBCE(config["start_warm"], config["end_warm"])
        self.hinge_loss = lovasz_hinge
        self.scotch_loss = scotch_loss
        self.margin_loss = margin_loss
        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)

    def compute(
        self,
        logits,
        gt,
        coarse_mask=None,
        x_bound_gt=None,
        preds=None,
        scotch_loss_input=None,
        it=None,
    ):
        """
        Args:
            logit: B,1,H,W
            pred: B,1,H,W  0/1 masks
            gt: B,1,H,W    0/1 masks
        """
        losses = defaultdict(int)
        # coarse_gt  = F.interpolate(gt,size=32,mode="nearest")

        # gt = gt.flatten(0,1)

        mean_aux_pred = 0
        if preds != None:
            for pred in preds:
                mean_aux_pred += torch.nn.functional.interpolate(
                    pred,
                    size=self.config["scale"],
                    mode="bilinear",
                    align_corners=False,
                )  # b*t,1,h,w
            mean_aux_pred /= len(preds)

        losses["total_loss"] = 0
        if self.config["dice_loss"]:
            losses["dice_loss"] = 0
            losses["coarse_mask_dice_loss"] = 0
            if coarse_mask is not None:
                losses["coarse_mask_dice_loss"] += self.dice_loss(
                    coarse_mask, x_bound_gt
                )
            losses["dice_loss"] += self.dice_loss(logits, gt)
            losses["total_loss"] += (
                self.config["dice_weight"] * losses["dice_loss"]
                + 0.5 * losses["coarse_mask_dice_loss"]
            )

        if self.config["bce_loss"]:
            losses["coarse_mask_bce_loss"] = 0
            losses["pred_bce_loss"] = 0
            losses["bce_loss"] = 0
            losses["bce_loss"] += self.bce_loss(logits, gt)
            if coarse_mask is not None:
                losses["coarse_mask_bce_loss"] += self.bce_loss(coarse_mask, x_bound_gt)
            if preds is not None:
                losses["pred_bce_loss"] += self.bce_loss(mean_aux_pred, x_bound_gt)
            losses["total_loss"] += (
                self.config["bce_weight"] * losses["bce_loss"]
                + 0.5 * losses["coarse_mask_bce_loss"]
                + 0.5 * losses["pred_bce_loss"]
            )

        if self.config["hinge_loss"]:
            losses["hinge_loss"] = 0
            hinge_loss = self.hinge_loss(logits.squeeze(1), gt.squeeze(1))
            losses["hinge_loss"] += hinge_loss
            losses["total_loss"] += (
                losses["hinge_loss"] * self.config["hinge_loss_weight"]
            )

        return losses


def contrast_loss(slots, pos_feats, neg_feats, t=0.1, eps=1e-6):
    # fore_slots back_slots : b,num_slots,c
    # fore_feats back_feats : b,c  or  b,n,c
    fore_slots = slots.flatten(0, 1)  # b*num_slots,c
    if len(pos_feats.shape) == 3:
        # b,n,c ->bn,c
        pos_feats = pos_feats.flatten(0, 1)
        neg_feats = neg_feats.flatten(0, 1)
    n = pos_feats.shape[0]
    feats = torch.concat([pos_feats, neg_feats], dim=0)  # 2*n,c

    pos_neg_attn = (fore_slots @ feats.t()) / t  # b*num_slots,2*n
    pos_neg = torch.logsumexp(pos_neg_attn, dim=1)  # b*num_slots,

    pos_attn = pos_neg_attn[:, :n]  # b*num_slots,n
    pos = torch.logsumexp(pos_attn, dim=1)  # b*num_slots,

    loss = -(pos - pos_neg).mean()
    return loss


class SASLossComputer_decouple:
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss()
        self.mse_loss1 = nn.MSELoss(reduce=True)
        self.mse_loss2 = nn.MSELoss(reduce=True)

    def compute(
        self,
        logits,
        gt,
        body_gt=None,
        detail_gt=None,
        body=None,
        detail=None,
    ):
        """
        Args:
            logit: B,1,H,W
            pred: B,1,H,W  0/1 masks
            gt: B,1,H,W    0/1 masks
        """
        losses = defaultdict(int)
        # coarse_gt  = F.interpolate(gt,size=32,mode="nearest")

        # gt = gt.flatten(0,1)

        losses["total_loss"] = 0
        if self.config["dice_loss"]:
            losses["dice_loss"] = 0
            losses["coarse_mask_dice_loss"] = 0
            if detail is not None:
                losses["coarse_mask_dice_loss"] += self.dice_loss(
                    detail, detail_gt
                )
            losses["dice_loss"] += self.dice_loss(logits, gt)
            losses["total_loss"] += (
                self.config["dice_weight"] * losses["dice_loss"]
                + 0.3 * losses["coarse_mask_dice_loss"]
            )

        if self.config["bce_loss"]:
            losses["coarse_mask_bce_loss"] = 0
            losses["bce_loss"] = 0
            losses["bce_loss"] += self.bce_loss(logits, gt)
            if detail is not None:
                losses["coarse_mask_bce_loss"] += self.bce_loss(detail, detail_gt)
            losses["total_loss"] += (
                self.config["bce_weight"] * losses["bce_loss"]
                + 0.3 * losses["coarse_mask_bce_loss"]
            )

        losses["body_loss"] = 0
        body_loss = 0
        for bo in body:
            body_loss += self.mse_loss1(bo, body_gt)
        losses["body_loss"] += body_loss
        losses["total_loss"] += 0.3 * losses["body_loss"]

        return losses
