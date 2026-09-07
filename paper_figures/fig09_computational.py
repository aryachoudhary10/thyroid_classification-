"""Figure 9 -- computational cost, ConvNeXt-Tiny backbone.

Parameter count depends only on architecture and config, not on trained
weight values, so every number here is computed by instantiating the real
model class with the real training config (backbone=convnext_tiny,
evidence_mode=masked_input) and measuring it directly -- no checkpoint file
is required, and none of these three models needed one to be exact.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, MODEL_LABELS, style_ax

sys.path.insert(0, data.ROOT)
from src.config import Config          # noqa: E402
from src.models.factory import build_model  # noqa: E402

MODELS = ["lesion_mil", "mr_mil", "der_mil"]


def measure(model_name: str, n_frames: int = 3, n_reps: int = 20):
    cfg = Config()
    cfg.model.pretrained = False
    cfg.model.backbone = "convnext_tiny"
    cfg.model.evidence_mode = "masked_input"
    model, reqs = build_model(cfg, model_name)
    model.eval()
    res = reqs["region_res"]
    batch = {"image": torch.randn(1, n_frames, 3, 224, 224),
            "lesion": (torch.rand(1, n_frames, 1, 224, 224) > 0.7).float(),
            "regions": (torch.rand(1, n_frames, 4, res, res) > 0.7).float(),
            "valid": torch.ones(1, n_frames), "label": torch.tensor([0.0])}
    with torch.no_grad():
        for _ in range(3):
            model(batch)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            model(batch)
        t1 = time.perf_counter()
    with torch.profiler.profile(with_flops=True) as prof:
        with torch.no_grad():
            model(batch)
    flops = sum(e.flops for e in prof.key_averages() if e.flops) / 1e9
    n_params = sum(v.numel() for v in model.state_dict().values()) / 1e6
    return n_params, flops, 1000 * (t1 - t0) / n_reps


def main() -> None:
    measured = {m: measure(m) for m in MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    x = np.arange(len(MODELS))
    labs = [MODEL_LABELS[m].split(" (")[0] for m in MODELS]
    cols = [MODEL_COLORS[m] for m in MODELS]

    ax = axes[0]
    vals = [measured[m][0] for m in MODELS]
    ax.bar(x, vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, "%.1fM" % v, ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    style_ax(ax, "(a) Parameters, ConvNeXt-Tiny backbone\n(exact, architecture-determined)",
            ylabel="Millions of parameters")

    ax = axes[1]
    vals = [measured[m][1] for m in MODELS]
    ax.bar(x, vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, "%.1f" % v, ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    style_ax(ax, "(b) Inference FLOPs, 3-frame bag\n(measured, torch.profiler)",
            ylabel="GFLOPs / patient")

    ax = axes[2]
    vals = [measured[m][2] for m in MODELS]
    ax.bar(x, vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 5, "%.0f ms" % v, ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    style_ax(ax, "(c) CPU inference latency, 3-frame bag\n(measured on this machine, 20-run mean)",
            ylabel="Milliseconds / patient")

    fig.tight_layout()
    caption = (
        "Figure 9. Computational cost, ConvNeXt-Tiny backbone. (a) Exact "
        "parameter counts (lesion-only %.2fM, MR-MIL %.2fM, DER-MIL %.2fM); "
        "the ~0.4M difference between MR-MIL and DER-MIL is the reliability "
        "head. (b-c) FLOPs and CPU wall-clock latency for a representative "
        "3-frame patient bag, measured directly on this machine "
        "(torch.profiler, 20-run mean). The masked_input evidence encoding "
        "used throughout this project runs one backbone pass per region (4x "
        "a single-branch model) rather than pooling from shared feature maps, "
        "which is the dominant cost for MR-MIL and DER-MIL relative to the "
        "lesion-only baseline."
        % (measured["lesion_mil"][0], measured["mr_mil"][0], measured["der_mil"][0])
    )
    style.save(fig, "fig09_computational_cost_convnext", caption)


if __name__ == "__main__":
    main()
