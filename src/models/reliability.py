"""Diagnostic Evidence Reliability -- the core contribution of DER-MIL.

For every evidence token e[b, t, k] the model estimates four quantities:

    I  importance     how predictive this evidence claims to be
    S  support        how strongly *other views* corroborate it
    D  contradiction  how strongly other reliable views disagree with it
    U  uncertainty    how confident the model is in the observation itself

and combines them into a reliability score

    R = f(I, S, D, U)  in (0, 1)

Design notes that matter:

* Support and contradiction are learned SEPARATELY and are not two ends of one
  axis. Two views can be dissimilar because they are complementary (different
  orientation, different zoom) rather than because they disagree diagnostically.
  A single signed similarity cannot express that; a positive channel and a
  negative channel can.
* Peer influence is weighted by peer confidence (1 - U_j), so a blurred frame
  cannot manufacture support or contradiction for a clean one.
* Patients with a single frame have no cross-view evidence at all. Rather than
  penalising them, S and D are defined as 0 and a learned no-peer offset enters
  the reliability function together with an explicit ``has_peers`` indicator.
* Every operation is masked by frame validity and is permutation invariant.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# --------------------------------------------------------------------------- #
class ImportanceHead(nn.Module):
    """Per-token diagnostic importance logit I[b, t, k]."""

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e).squeeze(-1)                      # (B, T, K)


# --------------------------------------------------------------------------- #
class UncertaintyHead(nn.Module):
    """Heteroscedastic evidence uncertainty U[b, t, k] in (0, 1).

    Trained implicitly: U scales the reliability that gates the prediction, so
    tokens the classifier cannot use are pushed towards high U by the task loss.
    ``mc_samples`` > 0 additionally injects dropout-based epistemic uncertainty
    at evaluation time.
    """

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, -1.0)             # start optimistic

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(self.body(e))).squeeze(-1)

    @torch.no_grad()
    def mc_uncertainty(self, e: torch.Tensor, samples: int) -> torch.Tensor:
        """Predictive spread of the evidence embedding under dropout."""
        was_training = self.training
        self.train(True)                                    # enable dropout only
        vals = torch.stack([self.forward(e) for _ in range(samples)], 0)
        self.train(was_training)
        return (vals.mean(0) + vals.std(0)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
class CrossViewRelation(nn.Module):
    """Signed support / contradiction between the same evidence type across views."""

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.0,
                 signed: bool = True, layers: int = 1):
        super().__init__()
        self.signed = signed
        self.layers = max(int(layers), 0)
        rel_in = 4 * dim
        self.pos = nn.Sequential(
            nn.Linear(rel_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.neg = nn.Sequential(
            nn.Linear(rel_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1)) if signed else None
        self.msg_pos = nn.Linear(dim, dim)
        self.msg_neg = nn.Linear(dim, dim) if signed else None
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pair_features(e: torch.Tensor) -> torch.Tensor:
        """(B, K, T, D) -> (B, K, T, T, 4D) symmetric pairwise descriptor."""
        b, k, t, d = e.shape
        ei = e.unsqueeze(3).expand(b, k, t, t, d)
        ej = e.unsqueeze(2).expand(b, k, t, t, d)
        return torch.cat([ei, ej, (ei - ej).abs(), ei * ej], dim=-1)

    def forward(self, e: torch.Tensor, valid: torch.Tensor,
                conf: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        e     : (B, T, K, D)
        valid : (B, T)
        conf  : (B, T, K) peer confidence in [0, 1] (typically 1 - U)
        """
        b, t, k, d = e.shape
        ek = e.permute(0, 2, 1, 3).contiguous()             # (B, K, T, D)

        vi = valid.view(b, 1, t, 1)
        vj = valid.view(b, 1, 1, t)
        eye = torch.eye(t, device=e.device, dtype=e.dtype).view(1, 1, t, t)
        pair_mask = (vi * vj) * (1.0 - eye)                 # (B, 1, T, T), self excluded
        pair_mask = pair_mask.expand(b, k, t, t)

        n_peers = pair_mask.sum(-1)                         # (B, K, T)
        has_peers = (n_peers > 0.5).float()

        if t == 1 or float(pair_mask.sum()) == 0.0:
            zeros = torch.zeros(b, t, k, device=e.device, dtype=e.dtype)
            return {"support": zeros, "contradiction": zeros,
                    "has_peers": zeros, "evidence": e,
                    "A_pos": torch.zeros(b, k, t, t, device=e.device, dtype=e.dtype),
                    "A_neg": torch.zeros(b, k, t, t, device=e.device, dtype=e.dtype),
                    "cos": torch.zeros(b, k, t, t, device=e.device, dtype=e.dtype)}

        feats = self._pair_features(ek)
        a_pos = torch.sigmoid(self.pos(feats).squeeze(-1)) * pair_mask
        a_neg = (torch.sigmoid(self.neg(feats).squeeze(-1)) * pair_mask
                 if self.signed else torch.zeros_like(a_pos))

        # Peer confidence weighting: an unreliable view cannot vouch for, or
        # argue against, another view.
        if conf is not None:
            cj = conf.permute(0, 2, 1).unsqueeze(2)         # (B, K, 1, T)
            a_pos = a_pos * cj
            a_neg = a_neg * cj

        denom = n_peers.clamp_min(1.0).unsqueeze(-1)
        support = (a_pos.sum(-1) / denom.squeeze(-1)) * has_peers          # (B, K, T)
        contra = (a_neg.sum(-1) / denom.squeeze(-1)) * has_peers

        # --- signed message passing -------------------------------------- #
        h = ek
        for _ in range(self.layers):
            w_pos = a_pos / denom
            m_pos = torch.einsum("bkij,bkjd->bkid", w_pos, h)
            upd = self.msg_pos(m_pos)
            if self.signed and self.msg_neg is not None:
                w_neg = a_neg / denom
                m_neg = torch.einsum("bkij,bkjd->bkid", w_neg, h)
                upd = upd - F.softplus(self.lam) * self.msg_neg(m_neg)
            h = self.norm(h + upd)

        return {
            "support": support.permute(0, 2, 1).contiguous(),        # (B, T, K)
            "contradiction": contra.permute(0, 2, 1).contiguous(),
            "has_peers": has_peers.permute(0, 2, 1).contiguous(),
            "evidence": h.permute(0, 2, 1, 3).contiguous(),           # (B, T, K, D)
            "A_pos": a_pos, "A_neg": a_neg,
            "cos": F.cosine_similarity(ek.unsqueeze(3), ek.unsqueeze(2), dim=-1) * pair_mask,
        }


# --------------------------------------------------------------------------- #
class ReliabilityHead(nn.Module):
    """R = f(I, S, D, U) in (0, 1).

    ``linear`` gives the interpretable, sign-constrained form
        R = sigmoid( I + a*S - b*D - c*U + bias )     a, b, c >= 0
    ``mlp`` lets the interaction be learned, at the cost of interpretability.
    Both are exposed so the paper can report whether the extra flexibility pays.
    """

    def __init__(self, kind: str = "mlp", hidden: int = 32,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.kind = kind
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.raw_beta = nn.Parameter(torch.tensor(float(beta)))
        self.raw_gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.bias = nn.Parameter(torch.zeros(1))
        self.no_peer = nn.Parameter(torch.zeros(1))
        if kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(5, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 1))
        elif kind != "linear":
            raise KeyError("reliability_fn must be 'mlp' or 'linear'")

    def coefficients(self) -> Dict[str, float]:
        return {"alpha": float(F.softplus(self.raw_alpha)),
                "beta": float(F.softplus(self.raw_beta)),
                "gamma": float(F.softplus(self.raw_gamma))}

    def forward(self, I: torch.Tensor, S: torch.Tensor, D: torch.Tensor,
                U: torch.Tensor, has_peers: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self.raw_alpha)
        b = F.softplus(self.raw_beta)
        c = F.softplus(self.raw_gamma)
        i_norm = torch.tanh(I)                      # keep the scale bounded
        if self.kind == "linear":
            z = i_norm + a * S - b * D - c * U + self.bias
            z = z + (1.0 - has_peers) * self.no_peer
        else:
            feats = torch.stack([i_norm, S, D, U, has_peers], dim=-1)
            z = self.net(feats).squeeze(-1) + self.bias
        return torch.sigmoid(z)


# --------------------------------------------------------------------------- #
class EvidenceReliability(nn.Module):
    """Bundles importance, cross-view relation, uncertainty and the fusion rule."""

    def __init__(self, dim: int, *, use_support: bool = True,
                 use_contradiction: bool = True, use_uncertainty: bool = True,
                 reliability_fn: str = "mlp", signed_graph: bool = True,
                 graph_layers: int = 1, dropout: float = 0.0,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 mc_samples: int = 0):
        super().__init__()
        self.use_support = use_support
        self.use_contradiction = use_contradiction
        self.use_uncertainty = use_uncertainty
        self.mc_samples = mc_samples

        self.importance = ImportanceHead(dim, dropout=dropout)
        self.uncertainty = UncertaintyHead(dim, dropout=max(dropout, 0.2)) \
            if use_uncertainty else None
        self.relation = CrossViewRelation(
            dim, dropout=dropout,
            signed=signed_graph and use_contradiction,
            layers=graph_layers) if (use_support or use_contradiction) else None
        self.fuse = ReliabilityHead(reliability_fn, alpha=alpha, beta=beta, gamma=gamma)

    # ------------------------------------------------------------------ #
    def forward(self, e: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, k, _d = e.shape
        zeros = torch.zeros(b, t, k, device=e.device, dtype=e.dtype)

        if self.uncertainty is not None:
            U = self.uncertainty(e)
            if (not self.training) and self.mc_samples > 0:
                U = self.uncertainty.mc_uncertainty(e, self.mc_samples)
        else:
            U = zeros

        conf = (1.0 - U).clamp(0.0, 1.0)
        rel = self.relation(e, valid, conf) if self.relation is not None else None

        if rel is not None:
            e_ctx = rel["evidence"]
            S = rel["support"] if self.use_support else zeros
            D = rel["contradiction"] if self.use_contradiction else zeros
            has_peers = rel["has_peers"]
        else:
            e_ctx, S, D = e, zeros, zeros
            has_peers = (valid.sum(1, keepdim=True) > 1.5).float().unsqueeze(-1).expand(b, t, k)

        I = self.importance(e_ctx)
        R = self.fuse(I, S, D, U, has_peers)

        vm = valid.unsqueeze(-1)                              # (B, T, 1)
        out = {"evidence": e_ctx, "I": I * vm, "S": S * vm, "D": D * vm,
               "U": U * vm, "R": R * vm, "has_peers": has_peers * vm}
        if rel is not None:
            out["A_pos"] = rel["A_pos"]
            out["A_neg"] = rel["A_neg"]
            out["cos"] = rel["cos"]
        return out
