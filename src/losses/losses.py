"""Training objectives.

L = L_cls + l1 * L_consistency + l2 * L_reliability [+ l3 * L_counterfactual]

Two design decisions worth stating explicitly, because both are easy to get
wrong in a way that silently invalidates the reliability claim:

1. The consistency term is weighted by the *detached* support matrix A+. If the
   weights carried gradient, the cheapest way to minimise the loss would be to
   drive all support to zero, and the model would learn to declare that no view
   ever corroborates any other. Detaching makes consistency a statement about
   representations, not about the relation head.

2. Consistency is applied only where the model has already judged two views to
   be supportive. Different ultrasound views legitimately carry complementary
   information, so forcing all cross-view embeddings together would destroy the
   very signal the contradiction channel exists to detect.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LossConfig

EPS = 1e-6


# --------------------------------------------------------------------------- #
def bce_loss(logit: torch.Tensor, target: torch.Tensor,
             pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)


def focal_loss(logit: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
               label_smoothing: float = 0.0,
               pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Focal loss with label smoothing, used for TN5000 domain adaptation."""
    if label_smoothing > 0:
        target = target * (1 - label_smoothing) + 0.5 * label_smoothing
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none",
                                            pos_weight=pos_weight)
    p_t = p * target + (1 - p) * (1 - target)
    return (ce * (1.0 - p_t).clamp_min(EPS).pow(gamma)).mean()


# --------------------------------------------------------------------------- #
def cross_view_consistency(out: Dict[str, torch.Tensor],
                           valid: torch.Tensor) -> torch.Tensor:
    """Support-weighted agreement between same-type evidence across views.

    Also penalises tokens that are simultaneously strongly supportive and
    strongly contradictory, which would make the two channels uninterpretable.
    """
    if "A_pos" not in out or "evidence_emb" not in out:
        return torch.zeros((), device=valid.device)

    e = out["evidence_emb"]                        # (B, T, K, D)
    a_pos = out["A_pos"].detach()                  # (B, K, T, T)
    b, t, k, _d = e.shape
    if t < 2 or a_pos.numel() == 0:
        return torch.zeros((), device=e.device)

    ek = e.permute(0, 2, 1, 3)                     # (B, K, T, D)
    en = F.normalize(ek, dim=-1)
    cos = torch.einsum("bkid,bkjd->bkij", en, en)  # (B, K, T, T)

    w = a_pos
    denom = w.sum().clamp_min(EPS)
    l_align = (w * (1.0 - cos)).sum() / denom

    if "A_neg" in out and out["A_neg"].numel():
        a_neg = out["A_neg"]
        l_excl = (out["A_pos"] * a_neg).mean()
    else:
        l_excl = torch.zeros((), device=e.device)
    return l_align + l_excl


def reliability_regularizer(out: Dict[str, torch.Tensor], valid: torch.Tensor,
                            target_mean: float = 0.5,
                            min_std: float = 0.05) -> torch.Tensor:
    """Keep the reliability distribution from collapsing to all-high or all-low."""
    if "R" not in out:
        return torch.zeros((), device=valid.device)
    R = out["R"]                                   # (B, T, K)
    m = valid.unsqueeze(-1).expand_as(R)
    n = m.sum().clamp_min(1.0)
    mean = (R * m).sum() / n
    var = ((R - mean) ** 2 * m).sum() / n
    std = var.clamp_min(0.0).sqrt()
    return (mean - target_mean) ** 2 + F.relu(min_std - std) ** 2 * 10.0


def counterfactual_ranking(model, batch: Dict[str, torch.Tensor],
                           out: Dict[str, torch.Tensor],
                           n_pairs: int = 4, margin: float = 0.05) -> torch.Tensor:
    """Rank-align reliability with actual predictive influence.

    For token pairs (a, b) with R_a > R_b, suppressing a should move the
    prediction at least ``margin`` more than suppressing b.
    """
    if not hasattr(model, "predict_with_suppression") or "R" not in out:
        return torch.zeros((), device=batch["valid"].device)

    R = out["R"].detach()
    valid = batch["valid"]
    b, t, k = R.shape
    device = R.device
    base = out["logit"].detach()

    flat = (R * valid.unsqueeze(-1)).reshape(b, t * k)
    losses = []
    for _ in range(n_pairs):
        i = torch.randint(0, t * k, (b,), device=device)
        j = torch.randint(0, t * k, (b,), device=device)
        keep_i = torch.ones(b, t * k, device=device)
        keep_j = torch.ones(b, t * k, device=device)
        keep_i.scatter_(1, i.unsqueeze(1), 0.0)
        keep_j.scatter_(1, j.unsqueeze(1), 0.0)
        li = model.predict_with_suppression(batch, token=keep_i.view(b, t, k))
        lj = model.predict_with_suppression(batch, token=keep_j.view(b, t, k))
        d_i = (base - li).abs()
        d_j = (base - lj).abs()
        r_i = flat.gather(1, i.unsqueeze(1)).squeeze(1)
        r_j = flat.gather(1, j.unsqueeze(1)).squeeze(1)
        sign = torch.sign(r_i - r_j)
        losses.append(F.relu(margin - sign * (d_i - d_j)).mean())
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


# --------------------------------------------------------------------------- #
class TotalLoss(nn.Module):
    def __init__(self, cfg: LossConfig, pos_weight: Optional[float] = None,
                 use_focal: bool = False, focal_gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight)) if pos_weight is not None else torch.tensor(1.0))

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                model=None) -> Dict[str, torch.Tensor]:
        y = batch["label"]
        pw = self.pos_weight if float(self.pos_weight) != 1.0 else None
        if self.use_focal:
            l_cls = focal_loss(out["logit"], y, self.focal_gamma,
                               self.label_smoothing, pw)
        else:
            l_cls = bce_loss(out["logit"], y, pw)

        parts = {"cls": l_cls}
        total = l_cls
        valid = batch["valid"]

        if self.cfg.lambda_consistency > 0:
            l_cons = cross_view_consistency(out, valid)
            parts["consistency"] = l_cons
            total = total + self.cfg.lambda_consistency * l_cons

        if self.cfg.lambda_reliability > 0:
            l_rel = reliability_regularizer(out, valid)
            parts["reliability"] = l_rel
            total = total + self.cfg.lambda_reliability * l_rel

        if self.cfg.lambda_counterfactual > 0 and model is not None:
            l_cf = counterfactual_ranking(model, batch, out,
                                          self.cfg.cf_pairs_per_bag, self.cfg.cf_margin)
            parts["counterfactual"] = l_cf
            total = total + self.cfg.lambda_counterfactual * l_cf

        parts["total"] = total
        return parts
