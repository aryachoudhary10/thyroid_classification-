"""Fair patient-level baselines (paper Table 5).

Every baseline shares the backbone, embedding width, bag construction, two-stage
schedule and checkpoint-selection logic with RCAF and DER-MIL, so differences
are attributable to architecture rather than to protocol.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ..config import ModelConfig
from .attn_mil import AttnMIL, masked_max, masked_mean
from .backbone import (Backbone, flatten_bag, gather_frames, scatter_frames,
                       unflatten_bag, valid_index)


# --------------------------------------------------------------------------- #
def lesion_bboxes(lesion: torch.Tensor, pad: float = 0.10) -> torch.Tensor:
    """(N, 1, H, W) binary masks -> (N, 5) roi_align boxes [idx, x1, y1, x2, y2].

    Frames with an empty mask fall back to the full frame.
    """
    n, _c, h, w = lesion.shape
    m = (lesion[:, 0] > 0.5)
    boxes = torch.zeros(n, 5, device=lesion.device, dtype=torch.float32)
    boxes[:, 0] = torch.arange(n, device=lesion.device, dtype=torch.float32)

    rows = m.any(dim=2)              # (N, H)
    cols = m.any(dim=1)              # (N, W)
    idx_h = torch.arange(h, device=lesion.device).float()
    idx_w = torch.arange(w, device=lesion.device).float()

    big = torch.tensor(float(max(h, w)), device=lesion.device)
    y1 = torch.where(rows, idx_h.unsqueeze(0), big).min(dim=1).values
    y2 = torch.where(rows, idx_h.unsqueeze(0), torch.zeros_like(big)).max(dim=1).values
    x1 = torch.where(cols, idx_w.unsqueeze(0), big).min(dim=1).values
    x2 = torch.where(cols, idx_w.unsqueeze(0), torch.zeros_like(big)).max(dim=1).values

    empty = ~m.flatten(1).any(dim=1)
    y1 = torch.where(empty, torch.zeros_like(y1), y1)
    x1 = torch.where(empty, torch.zeros_like(x1), x1)
    y2 = torch.where(empty, torch.full_like(y2, h - 1.0), y2)
    x2 = torch.where(empty, torch.full_like(x2, w - 1.0), x2)

    ph = (y2 - y1).clamp_min(1.0) * pad
    pw = (x2 - x1).clamp_min(1.0) * pad
    boxes[:, 1] = (x1 - pw).clamp(0, w - 1)
    boxes[:, 2] = (y1 - ph).clamp(0, h - 1)
    boxes[:, 3] = (x2 + pw).clamp(1, w)
    boxes[:, 4] = (y2 + ph).clamp(1, h)
    return boxes


def crop_lesion(image: torch.Tensor, lesion: torch.Tensor, out: int) -> torch.Tensor:
    boxes = lesion_bboxes(lesion)
    return roi_align(image, boxes, output_size=(out, out), spatial_scale=1.0,
                     sampling_ratio=2, aligned=True)


# --------------------------------------------------------------------------- #
class _SingleBranchMIL(nn.Module):
    """Shared skeleton for image-only and lesion-only AttnMIL baselines."""

    def __init__(self, cfg: ModelConfig, mode: str):
        super().__init__()
        assert mode in ("image", "lesion_crop", "lesion_mult")
        self.cfg, self.mode = cfg, mode
        self.backbone = Backbone(cfg.backbone, cfg.pretrained)
        d = cfg.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.out_dim, d), nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout))
        self.mil = AttnMIL(d, cfg.attn_dim, gated=True, dropout=cfg.dropout)
        self.head = nn.Linear(d, 1)

    def _frames(self, batch, idx) -> torch.Tensor:
        x = gather_frames(batch["image"], idx)
        if self.mode == "lesion_crop":
            x = crop_lesion(x, gather_frames(batch["lesion"], idx),
                            self.cfg_image_size(batch))
        elif self.mode == "lesion_mult":
            x = x * gather_frames(batch["lesion"], idx)
        return x

    @staticmethod
    def cfg_image_size(batch) -> int:
        return int(batch["image"].shape[-1])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        b, t = batch["image"].shape[:2]
        idx = valid_index(batch["valid"])
        z = self.proj(self.backbone.forward_vec(self._frames(batch, idx)))
        z = scatter_frames(z, idx, b, t)
        bag, alpha = self.mil(z, batch["valid"])
        return {"logit": self.head(bag).squeeze(-1), "alpha": alpha,
                "bag": bag, "frame_emb": z}

    def set_stage(self, stage: int) -> None:
        self.backbone.set_stage(stage)


class ImageOnlyMIL(_SingleBranchMIL):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg, "image")


class LesionCropMIL(_SingleBranchMIL):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg, "lesion_crop")


# --------------------------------------------------------------------------- #
class TransformerBagMIL(nn.Module):
    """Sequence-level bag model. No positional encoding -> permutation invariant."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg.backbone, cfg.pretrained)
        d = cfg.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.out_dim, d), nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.tf_heads, dim_feedforward=2 * d,
            dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.tf_layers)
        self.mil = AttnMIL(d, cfg.attn_dim, gated=True, dropout=cfg.dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        b, t = batch["image"].shape[:2]
        idx = valid_index(batch["valid"])
        z = self.proj(self.backbone.forward_vec(gather_frames(batch["image"], idx)))
        z = scatter_frames(z, idx, b, t)
        pad = batch["valid"] < 0.5
        # A fully padded row would produce NaNs; bag construction guarantees at
        # least one valid frame, but guard anyway.
        pad = pad & (~pad.all(dim=1, keepdim=True))
        z = self.encoder(z, src_key_padding_mask=pad)
        z = torch.nan_to_num(z)
        bag, alpha = self.mil(z, batch["valid"])
        return {"logit": self.head(bag).squeeze(-1), "alpha": alpha,
                "bag": bag, "frame_emb": z}

    def set_stage(self, stage: int) -> None:
        self.backbone.set_stage(stage)


# --------------------------------------------------------------------------- #
class PoolingMIL(nn.Module):
    """Naive reference: frame-level logits combined by mean or max pooling."""

    def __init__(self, cfg: ModelConfig, pool: str = "mean"):
        super().__init__()
        assert pool in ("mean", "max")
        self.pool = pool
        self.cfg = cfg
        self.backbone = Backbone(cfg.backbone, cfg.pretrained)
        d = cfg.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.out_dim, d), nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout))
        self.head = nn.Linear(d, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        b, t = batch["image"].shape[:2]
        idx = valid_index(batch["valid"])
        z = self.proj(self.backbone.forward_vec(gather_frames(batch["image"], idx)))
        z = scatter_frames(z, idx, b, t)
        frame_logits = self.head(z).squeeze(-1)                 # (B, T)
        valid = batch["valid"]
        if self.pool == "mean":
            logit = masked_mean(frame_logits.unsqueeze(-1), valid).squeeze(-1)
        else:
            logit = masked_max(frame_logits.unsqueeze(-1), valid).squeeze(-1)
        alpha = valid / valid.sum(1, keepdim=True).clamp_min(1e-8)
        return {"logit": logit, "alpha": alpha, "bag": z.mean(1), "frame_emb": z}

    def set_stage(self, stage: int) -> None:
        self.backbone.set_stage(stage)
