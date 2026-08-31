"""Arm 7: lesion descriptors as a controlled clinical vocabulary.

TN5000 ships no radiology reports and no structured clinical fields -- its
TIRADS, age and sex columns are empty -- so there is no text to pair with an
image. The text is therefore *derived*: geometry and echotexture statistics are
computed from the frame and its lesion mask, quantised into ordinal clinical
phrases, and encoded as a sentence over a closed vocabulary.

Be precise about what this is. It is a vision-language architecture over a
controlled vocabulary, in the same family as structured-report models -- it is
NOT a pretrained medical language model, and it must not be written up as one.
The advantage is that it works identically on ThyroidXL and TN5000, needs no
external weights, and every phrase is traceable to a measurable quantity. The
limitation is that the vocabulary can only express what the descriptors measure.

The descriptors are the ones ultrasound risk-stratification systems actually
key on: size relative to field, taller-than-wide orientation, margin
irregularity, echogenicity relative to surrounding tissue, and internal
heterogeneity.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

# Ordinal phrase banks. Each descriptor contributes exactly one phrase, so a
# sentence is a fixed-length sequence and the "tokenizer" is a bucketize.
PHRASE_BANKS: Dict[str, Tuple[str, ...]] = {
    "size":      ("punctate", "small", "moderately sized", "large", "very large"),
    "shape":     ("wider-than-tall", "round", "taller-than-wide"),
    "margin":    ("smooth margins", "mildly lobulated margins", "irregular margins"),
    "echo":      ("markedly hypoechoic", "hypoechoic", "isoechoic", "hyperechoic"),
    "texture":   ("homogeneous echotexture", "mildly heterogeneous echotexture",
                  "markedly heterogeneous echotexture"),
    "contrast":  ("indistinct from surrounding tissue", "moderately distinct",
                  "sharply distinct from surrounding tissue"),
}
DESCRIPTOR_ORDER: Tuple[str, ...] = ("size", "shape", "margin", "echo",
                                     "texture", "contrast")

# Provisional bin edges. These are placeholders: descriptor scales depend on the
# imaging characteristics of the cohort, and edges that are wrong collapse the
# vocabulary onto one or two phrases, which silently turns the text branch into
# a constant. Call ``calibrate_bin_edges`` on the real ThyroidXL manifest and
# ``set_bin_edges`` with the result before training the arm; the edges are then
# frozen and travel with the run.
BIN_EDGES: Dict[str, Tuple[float, ...]] = {
    "size":     (0.01, 0.04, 0.10, 0.22),
    "shape":    (0.90, 1.10),
    "margin":   (1.15, 1.45),
    "echo":     (0.70, 0.92, 1.08),
    "texture":  (0.14, 0.24),
    "contrast": (0.06, 0.16),
}

VOCAB_SIZES: Tuple[int, ...] = tuple(len(PHRASE_BANKS[k]) for k in DESCRIPTOR_ORDER)
VOCAB_TOTAL: int = sum(VOCAB_SIZES)
N_DESCRIPTORS: int = len(DESCRIPTOR_ORDER)


def set_bin_edges(edges: Dict[str, Tuple[float, ...]]) -> None:
    """Freeze quantised phrase boundaries for the rest of the run."""
    for k, v in edges.items():
        if k in BIN_EDGES:
            expect = len(PHRASE_BANKS[k]) - 1
            if len(v) != expect:
                raise ValueError("%s needs %d edges, got %d" % (k, expect, len(v)))
            BIN_EDGES[k] = tuple(float(x) for x in v)


def bin_edges_snapshot() -> Dict[str, Tuple[float, ...]]:
    return {k: tuple(v) for k, v in BIN_EDGES.items()}


# --------------------------------------------------------------------------- #
@torch.no_grad()
def frame_descriptors(image: torch.Tensor, lesion: torch.Tensor
                      ) -> Dict[str, torch.Tensor]:
    """Continuous descriptors for a batch of frames.

    ``image``  (N, C, H, W) in normalised space -- only relative statistics are
               used, so the normalisation constant cancels.
    ``lesion`` (N, 1, H, W) binary mask.

    Returns one (N,) tensor per descriptor name.
    """
    n, _c, h, w = image.shape
    g = image.mean(dim=1, keepdim=True)                       # (N,1,H,W) grey
    m = (lesion > 0.5).float()
    area = m.sum(dim=(1, 2, 3))                               # (N,)
    has = (area > EPS).float()
    a_safe = area.clamp_min(1.0)

    # ---- size: fraction of the field occupied ---------------------------- #
    size = area / float(h * w)

    # ---- shape: second moments give an orientation-free aspect ratio ------ #
    ys = torch.arange(h, device=image.device, dtype=image.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=image.device, dtype=image.dtype).view(1, 1, 1, w)
    cy = (m * ys).sum(dim=(1, 2, 3)) / a_safe
    cx = (m * xs).sum(dim=(1, 2, 3)) / a_safe
    vy = (m * (ys - cy.view(-1, 1, 1, 1)) ** 2).sum(dim=(1, 2, 3)) / a_safe
    vx = (m * (xs - cx.view(-1, 1, 1, 1)) ** 2).sum(dim=(1, 2, 3)) / a_safe
    shape = (vy.clamp_min(EPS).sqrt()) / (vx.clamp_min(EPS).sqrt() + EPS)

    # ---- margin: perimeter against that of a circle of equal area --------- #
    # A 4-neighbour gradient counts boundary pixels without a contour trace.
    dy = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().sum(dim=(1, 2, 3))
    dx = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().sum(dim=(1, 2, 3))
    perim = dy + dx
    circ_perim = 2.0 * torch.pi * (area.clamp_min(EPS) / torch.pi).sqrt()
    margin = perim / (circ_perim + EPS)

    # ---- echogenicity and texture: inside versus outside ------------------ #
    out_m = 1.0 - m
    out_a = out_m.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mu_in = (g * m).sum(dim=(1, 2, 3)) / a_safe
    mu_out = (g * out_m).sum(dim=(1, 2, 3)) / out_a
    var_in = ((g - mu_in.view(-1, 1, 1, 1)) ** 2 * m).sum(dim=(1, 2, 3)) / a_safe
    sd_in = var_in.clamp_min(0).sqrt()
    spread = g.std(dim=(1, 2, 3)).clamp_min(EPS)

    echo = (mu_in + EPS) / (mu_out + EPS)          # <1 hypoechoic, >1 hyperechoic
    texture = sd_in / spread                       # internal heterogeneity
    contrast = (mu_in - mu_out).abs() / spread     # boundary conspicuity

    # Frames with no lesion get neutral mid-bin values rather than zeros, so an
    # empty mask reads as "no finding" instead of as an extreme measurement.
    def _fill(v: torch.Tensor, neutral: float) -> torch.Tensor:
        return v * has + neutral * (1.0 - has)

    return {"size": _fill(size, 0.02), "shape": _fill(shape, 1.0),
            "margin": _fill(margin, 1.0), "echo": _fill(echo, 1.0),
            "texture": _fill(texture, 0.10), "contrast": _fill(contrast, 0.05)}


def descriptors_to_tokens(desc: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Quantise descriptors into phrase ids, offset into one shared vocabulary."""
    toks: List[torch.Tensor] = []
    offset = 0
    for name in DESCRIPTOR_ORDER:
        edges = torch.tensor(BIN_EDGES[name], device=desc[name].device,
                             dtype=desc[name].dtype)
        idx = torch.bucketize(desc[name], edges)
        toks.append(idx.clamp(0, len(PHRASE_BANKS[name]) - 1) + offset)
        offset += len(PHRASE_BANKS[name])
    return torch.stack(toks, dim=1).long()                    # (N, N_DESCRIPTORS)


def render_sentence(tokens: torch.Tensor) -> List[str]:
    """Human-readable sentences, for figures and for auditing the vocabulary."""
    flat: List[str] = []
    banks = [PHRASE_BANKS[k] for k in DESCRIPTOR_ORDER]
    offsets, o = [], 0
    for k in DESCRIPTOR_ORDER:
        offsets.append(o)
        o += len(PHRASE_BANKS[k])
    for row in tokens.tolist():
        words = [banks[i][row[i] - offsets[i]] for i in range(len(row))]
        flat.append("A %s thyroid nodule, %s, with %s, %s, %s, %s."
                    % (words[0], words[1], words[2], words[3], words[4], words[5]))
    return flat


# --------------------------------------------------------------------------- #
class DescriptorTextEncoder(nn.Module):
    """Embeds a fixed-length phrase sequence into one text vector.

    A sum over learned phrase embeddings with a positional term per descriptor
    slot -- the sequence has six positions with fixed semantics, so attention
    over it would model nothing a linear map cannot.
    """

    def __init__(self, dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_TOTAL, dim)
        self.slot = nn.Parameter(torch.zeros(N_DESCRIPTORS, dim))
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(N, N_DESCRIPTORS) ids -> (N, dim)."""
        e = self.emb(tokens) + self.slot.unsqueeze(0)
        return self.drop(self.norm(e.mean(dim=1)))


class VisionLanguageFusion(nn.Module):
    """Gated fusion of a frame embedding with its derived-text embedding.

    The gate is conditioned on both streams, so the model can fall back to
    vision alone when the descriptors are uninformative -- which is what should
    happen when the mask is poor, and is why this is a gate rather than a
    concatenation.
    """

    def __init__(self, vis_dim: int, txt_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.text = DescriptorTextEncoder(txt_dim, dropout)
        self.proj = nn.Linear(txt_dim, vis_dim)
        self.gate = nn.Sequential(
            nn.Linear(vis_dim * 2, vis_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(vis_dim)

    def forward(self, vis: torch.Tensor, image: torch.Tensor,
                lesion: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``vis`` (B, T, D); ``image`` / ``lesion`` (B, T, C, H, W)."""
        b, t, d = vis.shape
        img = image.reshape(b * t, *image.shape[2:])
        les = lesion.reshape(b * t, *lesion.shape[2:])
        desc = frame_descriptors(img, les)
        tokens = descriptors_to_tokens(desc)
        txt = self.proj(self.text(tokens)).reshape(b, t, d)
        g = self.gate(torch.cat([vis, txt], dim=-1))
        return {"fused": self.norm(vis + g * txt), "text_emb": txt,
                "text_gate": g.mean(dim=-1), "tokens": tokens.reshape(b, t, -1)}


# --------------------------------------------------------------------------- #
@torch.no_grad()
def calibrate_bin_edges(cfg, manifest, registry=None, max_frames: int = 4000
                        ) -> Dict[str, Tuple[float, ...]]:
    """Fit phrase boundaries to the empirical descriptor distribution.

    Edges are placed at evenly spaced quantiles, so every phrase in every bank
    is used by roughly the same number of frames. That is the property the
    vocabulary needs: a bank whose phrases are never emitted contributes a dead
    embedding, and one emitted for everything contributes a constant.

    Fitted on the development cohort only and returned frozen.
    """
    import numpy as np
    import os
    from ..data.splits import test_frame
    from ..engine.cv import make_dataset
    from ..utils.common import banner, log, save_json

    key = "vl/bins/%s" % cfg.run.run_name
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, "descriptor_bins.json")
    if registry is not None and registry.is_done(key) and os.path.exists(out):
        from ..utils.common import load_json
        edges = {k: tuple(v) for k, v in load_json(out, {}).items()}
        set_bin_edges(edges)
        log("SKIP  " + key)
        return edges

    banner("DESCRIPTOR BIN CALIBRATION")
    test_ids = set(test_frame(manifest)["patient_id"].astype(str))
    dev = manifest[~manifest["patient_id"].astype(str).isin(test_ids)]
    dev = (dev if len(dev) else manifest).reset_index(drop=True)

    reqs = {"need_regions": False, "region_res": 28, "regions": ("core",)}
    ds = make_dataset(cfg, dev, reqs, train=False)
    acc: Dict[str, list] = {k: [] for k in DESCRIPTOR_ORDER}
    seen = 0
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.optim.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers)
    for batch in loader:
        img, les, valid = batch["image"], batch["lesion"], batch["valid"]
        b, t = valid.shape
        keep = valid.reshape(-1) > 0.5
        d = frame_descriptors(img.reshape(b * t, *img.shape[2:])[keep],
                              les.reshape(b * t, *les.shape[2:])[keep])
        for k, v in d.items():
            acc[k].append(v.cpu().numpy())
        seen += int(keep.sum())
        if seen >= max_frames:
            break
    del loader, ds

    edges: Dict[str, Tuple[float, ...]] = {}
    for k in DESCRIPTOR_ORDER:
        v = np.concatenate(acc[k]) if acc[k] else np.zeros(1)
        n_edge = len(PHRASE_BANKS[k]) - 1
        qs = [(i + 1) / (n_edge + 1) for i in range(n_edge)]
        e = np.quantile(v, qs)
        # Strictly increasing, or bucketize silently merges phrases.
        for i in range(1, len(e)):
            if e[i] <= e[i - 1]:
                e[i] = e[i - 1] + 1e-6
        edges[k] = tuple(float(x) for x in e)
        log("  %-9s n=%5d  edges %s" % (k, len(v),
            ", ".join("%.4f" % x for x in edges[k])))

    set_bin_edges(edges)
    save_json({k: list(v) for k, v in edges.items()}, out)
    if registry is not None:
        registry.mark_done(key, {"json": out, "n_frames": seen})
    return edges
