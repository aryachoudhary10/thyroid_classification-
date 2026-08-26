"""Multi-region diagnostic evidence encoder.

Produces one embedding per (frame, evidence-type) pair:

    e[b, t, k] in R^d      k in {core, margin, peri, global}

Two modes, both driven by the same region maps:

``roi_pool`` (default)
    ONE backbone pass per frame. Region embeddings are obtained by
    coverage-weighted pooling of the layer3 (14x14) and layer4 (7x7) feature
    maps. Cost is *lower* than RCAF's two-branch design while yielding four
    evidence streams instead of two, which is what makes the full DER-MIL
    trainable on a single Colab GPU.

``masked_input``
    K backbone passes per frame over region-masked images, i.e. the literal
    x (*) m formulation generalised to K regions. Faithful but K times the
    compute; kept for a like-for-like ablation against RCAF.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .backbone import (Backbone, flatten_bag, gather_frames, scatter_frames,
                       unflatten_bag, valid_index)

EPS = 1e-6


def coverage_pool(fmap: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Coverage-weighted spatial pooling.

    fmap    : (N, C, H, W)
    weights : (N, K, Hw, Ww)  non-negative region coverage in [0, 1]
    returns : (N, K, C)
    """
    n, c, h, w = fmap.shape
    if weights.shape[-2:] != (h, w):
        weights = F.adaptive_avg_pool2d(weights, (h, w))
    fmap = fmap.float()
    weights = weights.float()
    num = torch.einsum("nchw,nkhw->nkc", fmap, weights)
    den = weights.sum(dim=(2, 3)).clamp_min(EPS).unsqueeze(-1)
    pooled = num / den
    # A region that vanished at this resolution (e.g. an all-zero mask) gets a
    # deterministic zero vector rather than a garbage value.
    alive = (weights.sum(dim=(2, 3)) > EPS).float().unsqueeze(-1)
    return pooled * alive


class EvidenceEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.regions: Sequence[str] = tuple(cfg.regions)
        self.K = len(self.regions)
        self.mode = cfg.evidence_mode
        self.backbone = Backbone(cfg.backbone, cfg.pretrained)
        d = cfg.embed_dim

        if self.mode == "roi_pool":
            in_dim = self.backbone.dim_c4 + self.backbone.dim_c5
        elif self.mode == "masked_input":
            in_dim = self.backbone.out_dim
        else:
            raise KeyError("unknown evidence_mode: " + str(self.mode))

        # One projection head per evidence type: the same spatial features mean
        # different things inside the lesion, on its boundary, and around it.
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d),
                          nn.ReLU(inplace=True), nn.Dropout(cfg.dropout))
            for _ in range(self.K)])
        # Explicit evidence-type identity, so cross-view matching can stay
        # type-aware even after the projection heads.
        self.type_embed = nn.Parameter(torch.zeros(self.K, d))
        nn.init.normal_(self.type_embed, std=0.02)

    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns evidence tensor (B, T, K, D)."""
        b, t = batch["image"].shape[:2]
        idx = valid_index(batch["valid"])
        img = gather_frames(batch["image"], idx)              # (N, 3, H, W)
        reg = gather_frames(batch["regions"], idx)            # (N, K, R, R)

        if self.mode == "roi_pool":
            maps = self.backbone.forward_maps(img)
            p4 = coverage_pool(maps["c4"], reg)               # (N, K, C4)
            p5 = coverage_pool(maps["c5"], reg)               # (N, K, C5)
            feats = torch.cat([p4, p5], dim=-1)               # (N, K, C4+C5)
        else:
            size = img.shape[-1]
            reg_full = F.interpolate(reg, size=(size, size), mode="nearest")
            chunks = []
            for k in range(self.K):
                xk = img * reg_full[:, k:k + 1]
                chunks.append(self.backbone.forward_vec(xk))
            feats = torch.stack(chunks, dim=1)                # (N, K, C5)

        out = torch.stack(
            [self.heads[k](feats[:, k]) for k in range(self.K)], dim=1)   # (N, K, D)
        out = out + self.type_embed.unsqueeze(0)
        return scatter_frames(out, idx, b, t)                 # (B, T, K, D)

    def set_stage(self, stage: int) -> None:
        self.backbone.set_stage(stage)
