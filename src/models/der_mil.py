"""DER-MIL -- Diagnostic Evidence Reliability Multiple Instance Learning.

Pipeline
--------
    frames + masks
        -> multi-region evidence tokens        e[b, t, k]        (K per frame)
        -> importance / support / contradiction / uncertainty
        -> evidence reliability                R[b, t, k]
        -> reliability-aware evidence pooling  h[b, t]           (level 1)
        -> reliability-aware frame attention   alpha[b, t]       (level 2)
        -> patient embedding -> malignancy logit

Reliability enters both aggregation levels in log-space, i.e. *inside* the
softmax, so it re-normalises the competition between evidence rather than
rescaling an already-normalised distribution. That keeps the whole model a
proper weighted average and keeps gradients well behaved.

Setting ``use_reliability=False`` yields the "multi-region MIL without
reliability" ablation: identical encoder and capacity, plain importance
attention. That is the comparison that isolates the contribution claimed here.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .attn_mil import AttnMIL, NEG_INF
from .evidence_encoder import EvidenceEncoder
from .reliability import EvidenceReliability

EPS = 1e-6


class DERMIL(nn.Module):
    def __init__(self, cfg: ModelConfig, use_reliability: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_reliability = use_reliability
        d = cfg.embed_dim

        self.encoder = EvidenceEncoder(cfg)
        self.K = self.encoder.K

        self.reliability = EvidenceReliability(
            d,
            use_support=cfg.use_support and use_reliability,
            use_contradiction=cfg.use_contradiction and use_reliability,
            use_uncertainty=cfg.use_uncertainty and use_reliability,
            reliability_fn=cfg.reliability_fn,
            signed_graph=cfg.signed_graph,
            graph_layers=cfg.graph_layers,
            dropout=cfg.dropout,
            alpha=cfg.rel_alpha_init, beta=cfg.rel_beta_init, gamma=cfg.rel_gamma_init,
            mc_samples=cfg.mc_dropout_samples,
        )

        self.frame_norm = nn.LayerNorm(d)
        self.mil = AttnMIL(d, cfg.attn_dim, gated=True, dropout=cfg.dropout)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, 1))

    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valid = batch["valid"]                                  # (B, T)
        e = self.encoder(batch)                                 # (B, T, K, D)
        rel = self.reliability(e, valid)

        I, R = rel["I"], rel["R"]
        e_ctx = rel["evidence"]

        # ---- level 1: evidence weights inside each frame ------------------
        ev_logits = I
        if self.use_reliability:
            ev_logits = ev_logits + torch.log(R.clamp_min(EPS))
        w_ev = torch.softmax(ev_logits, dim=2)                   # (B, T, K)
        h = torch.einsum("btk,btkd->btd", w_ev, e_ctx)           # (B, T, D)
        h = self.frame_norm(h)

        # ---- level 2: reliability-modulated frame attention ---------------
        frame_R = (w_ev * R).sum(dim=2)                          # (B, T)
        bias = torch.log(frame_R.clamp_min(EPS)) if self.use_reliability else None
        bag, alpha = self.mil(h, valid, bias=bias)
        logit = self.head(bag).squeeze(-1)

        # joint (frame, evidence) contribution, sums to 1 over valid entries
        w_joint = alpha.unsqueeze(-1) * w_ev

        out = {
            "logit": logit,
            "alpha": alpha,
            "bag": bag,
            "frame_emb": h,
            "evidence_emb": e_ctx,
            "w_evidence": w_ev,
            "w_joint": w_joint,
            "frame_reliability": frame_R,
            "I": I, "S": rel["S"], "D": rel["D"], "U": rel["U"], "R": R,
            "has_peers": rel["has_peers"],
        }
        for key in ("A_pos", "A_neg", "cos"):
            if key in rel:
                out[key] = rel[key]
        return out

    # ------------------------------------------------------------------ #
    def set_stage(self, stage: int) -> None:
        self.encoder.set_stage(stage)

    def reliability_coefficients(self) -> Dict[str, float]:
        return self.reliability.fuse.coefficients()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_with_suppression(self, batch: Dict[str, torch.Tensor],
                                 region_idx: Optional[int] = None,
                                 frame_idx: Optional[int] = None,
                                 token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Counterfactual forward pass with evidence removed at the token level.

        ``token`` is an optional (B, T, K) binary keep-mask, letting the
        counterfactual analysis suppress an arbitrary set of evidence tokens
        without touching pixels -- which isolates the aggregation mechanism from
        the encoder.
        """
        valid = batch["valid"]
        e = self.encoder(batch)
        rel = self.reliability(e, valid)
        I, R, e_ctx = rel["I"], rel["R"], rel["evidence"]

        keep = torch.ones_like(I)
        if region_idx is not None:
            keep[:, :, region_idx] = 0.0
        if frame_idx is not None:
            keep[:, frame_idx, :] = 0.0
        if token is not None:
            keep = keep * token
        # never empty a frame completely
        keep = torch.where(keep.sum(-1, keepdim=True) > 0.5, keep, torch.ones_like(keep))

        ev_logits = I
        if self.use_reliability:
            ev_logits = ev_logits + torch.log(R.clamp_min(EPS))
        ev_logits = ev_logits.masked_fill(keep < 0.5, NEG_INF)
        w_ev = torch.softmax(ev_logits, dim=2)
        h = self.frame_norm(torch.einsum("btk,btkd->btd", w_ev, e_ctx))

        frame_R = (w_ev * R).sum(dim=2)
        bias = torch.log(frame_R.clamp_min(EPS)) if self.use_reliability else None
        bag, _alpha = self.mil(h, valid, bias=bias)
        return self.head(bag).squeeze(-1)
