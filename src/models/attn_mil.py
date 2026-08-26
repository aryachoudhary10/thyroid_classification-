"""Attention-based multiple instance learning (Ilse et al., 2018).

Equation (6) of the paper, with padded frames excluded from the softmax via the
frame-validity mask. Everything here is permutation invariant.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e4          # fp16-safe masking constant


class AttnMIL(nn.Module):
    """alpha_t = softmax_t( w^T tanh(V z_t) ),  z = sum_t alpha_t z_t"""

    def __init__(self, in_dim: int, attn_dim: int = 128, gated: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid()) if gated else None
        self.w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def scores(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T) unnormalised attention logits."""
        h = self.V(self.drop(z))
        if self.U is not None:
            h = h * self.U(z)
        return self.w(h).squeeze(-1)

    def forward(self, z: torch.Tensor, valid: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z     : (B, T, D)
        valid : (B, T) in {0, 1}
        bias  : optional (B, T) additive log-space modulation (DER-MIL uses
                log-reliability here so the modulation stays inside the softmax)
        Returns (bag embedding (B, D), attention weights (B, T)).
        """
        logits = self.scores(z)
        if bias is not None:
            logits = logits + bias
        logits = logits.masked_fill(valid < 0.5, NEG_INF)
        alpha = torch.softmax(logits, dim=1) * (valid > 0.5).float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        bag = torch.bmm(alpha.unsqueeze(1), z).squeeze(1)
        return bag, alpha


# --------------------------------------------------------------------------- #
def masked_mean(x: torch.Tensor, valid: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Mean over ``dim`` honouring a (B, T) validity mask."""
    while valid.dim() < x.dim():
        valid = valid.unsqueeze(-1)
    s = (x * valid).sum(dim=dim)
    n = valid.sum(dim=dim).clamp_min(1e-8)
    return s / n


def masked_max(x: torch.Tensor, valid: torch.Tensor, dim: int = 1) -> torch.Tensor:
    while valid.dim() < x.dim():
        valid = valid.unsqueeze(-1)
    return (x.masked_fill(valid < 0.5, NEG_INF)).max(dim=dim).values
