"""RCAF -- Region-Aware Context-Aware Fusion (reference implementation).

Faithful to the published equations:

    (2)  x_les = x (*) m,           x_ctx = x
    (3)  z_les = phi(x_les),        z_ctx = phi(x_ctx)        (shared phi)
    (4)  g     = sigmoid(Wg [z_les || z_ctx])
    (5)  z     = g (*) z_les + (1 - g) (*) z_ctx
    (6/7) AttnMIL aggregation over valid frames
    (8)  y_hat = sigmoid(Wo z + bo)

THIS FILE IS THE PRESERVED BASELINE. DER-MIL lives in ``der_mil.py`` and does
not import from here, so the baseline can never drift while the new model is
being developed.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import ModelConfig
from .attn_mil import AttnMIL
from .backbone import (Backbone, flatten_bag, gather_frames, scatter_frames,
                       unflatten_bag, valid_index)


class RCAF(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg.backbone, cfg.pretrained)
        d_in = self.backbone.out_dim
        d = cfg.embed_dim

        self.proj = nn.Sequential(
            nn.Linear(d_in, d), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout))

        self.gated = cfg.rcaf_gated
        if self.gated:
            self.gate = nn.Linear(2 * d, d)            # Eq. (4)
        else:
            # "RCAF no-gating": fixed (input-independent) linear combination.
            self.fuse = nn.Linear(2 * d, d)

        self.mil = AttnMIL(d, cfg.attn_dim, gated=True, dropout=cfg.dropout)
        self.head = nn.Linear(d, 1)                    # Eq. (8)

    # ------------------------------------------------------------------ #
    def encode_frames(self, image: torch.Tensor, lesion: torch.Tensor,
                      valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t = image.shape[:2]
        # Only real frames reach the encoder; padded slots are restored as zeros.
        idx = valid_index(valid)
        x_ctx = gather_frames(image, idx)              # (N, 3, H, W)
        x_les = x_ctx * gather_frames(lesion, idx)     # Eq. (2)

        # One batched pass over the concatenated branches keeps phi shared and
        # halves kernel-launch overhead versus two separate calls.
        both = torch.cat([x_les, x_ctx], 0)
        feat = self.proj(self.backbone.forward_vec(both))
        z_les, z_ctx = feat.chunk(2, dim=0)            # Eq. (3)

        if self.gated:
            g = torch.sigmoid(self.gate(torch.cat([z_les, z_ctx], -1)))   # Eq. (4)
            z = g * z_les + (1.0 - g) * z_ctx                              # Eq. (5)
        else:
            g = torch.full_like(z_les, 0.5)
            z = self.fuse(torch.cat([z_les, z_ctx], -1))

        return {
            "z": scatter_frames(z, idx, b, t),
            "gate_les": scatter_frames(g, idx, b, t),
            "z_les": scatter_frames(z_les, idx, b, t),
            "z_ctx": scatter_frames(z_ctx, idx, b, t),
        }

    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valid = batch["valid"]
        enc = self.encode_frames(batch["image"], batch["lesion"], valid)
        bag, alpha = self.mil(enc["z"], valid)
        logit = self.head(bag).squeeze(-1)
        gate_les = enc["gate_les"].mean(-1)            # (B, T) mean gate per frame
        return {
            "logit": logit,
            "alpha": alpha,
            "bag": bag,
            "frame_emb": enc["z"],
            "gate_les": gate_les,
            "gate_ctx": 1.0 - gate_les,
        }

    # ------------------------------------------------------------------ #
    def set_stage(self, stage: int) -> None:
        self.backbone.set_stage(stage)

    def head_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("backbone."):
                yield p


# --------------------------------------------------------------------------- #
class MaskChannelMIL(nn.Module):
    """Diagnostic baseline: naive image--mask channel concatenation.

    Included only to reproduce the shortcut-sensitivity experiment (Table 12).
    It is deliberately NOT part of the fair baseline ladder.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg.backbone, cfg.pretrained)
        # widen conv1 to 4 channels, replicating the pretrained RGB filters
        old = self.backbone.stem[0]
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride,
                        old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight.mean(1, keepdim=True)
        self.backbone.stem[0] = new

        d = cfg.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.out_dim, d), nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout))
        self.mil = AttnMIL(d, cfg.attn_dim, gated=True, dropout=cfg.dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        b, t = batch["image"].shape[:2]
        idx = valid_index(batch["valid"])
        x = torch.cat([batch["image"], batch["lesion"]], dim=2)   # 4 channels
        z = self.proj(self.backbone.forward_vec(gather_frames(x, idx)))
        z = scatter_frames(z, idx, b, t)
        bag, alpha = self.mil(z, batch["valid"])
        return {"logit": self.head(bag).squeeze(-1), "alpha": alpha,
                "bag": bag, "frame_emb": z}

    def set_stage(self, stage: int) -> None:
        self.backbone.set_stage(stage)
        # conv1 was rebuilt, so it must stay trainable for the extra channel
        for p in self.backbone.stem[0].parameters():
            p.requires_grad_(True)
