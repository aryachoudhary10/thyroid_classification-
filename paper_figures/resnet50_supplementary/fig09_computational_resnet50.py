"""Figure 9 -- computational cost.

Parameter counts are exact: computed by loading each model's real saved
checkpoint state_dict and summing tensor sizes, no architecture code required.
FLOPs and CPU latency require instantiating the actual nn.Module and running a
forward pass, which is only possible for mr_mil and der_mil -- the RCAF class
was intentionally removed from this codebase after its checkpoint had already
been produced, so only its parameter count (from the surviving weights) is
available here, not a live FLOPs/latency measurement. That gap is reported
rather than filled with an estimate.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PAPER_FIG_OUTDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, MODEL_LABELS, style_ax

import sys
sys.path.insert(0, data.ROOT)
from src.config import Config          # noqa: E402
from src.models.factory import build_model  # noqa: E402


def count_params(ckpt_path: str) -> float:
    sd = torch.load(data.require(ckpt_path), map_location="cpu", weights_only=False)
    return sum(v.numel() for v in sd["model"].values()) / 1e6


def measure(model_name: str, n_frames: int = 3, n_reps: int = 20):
    cfg = Config()
    cfg.model.pretrained = False
    cfg.model.backbone = "resnet50"
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
    params_ckpt = {
        "rcaf": count_params(data.CKPT["rcaf"]),
        "mr_mil": count_params(data.CKPT["mr_mil"]),
        "der_mil": count_params(data.CKPT["der_mil"]),
    }
    measured = {m: measure(m) for m in ("mr_mil", "der_mil")}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    models_all = ["rcaf", "mr_mil", "der_mil"]
    models_live = ["mr_mil", "der_mil"]

    ax = axes[0]
    x = np.arange(len(models_all))
    ax.bar(x, [params_ckpt[m] for m in models_all],
          color=[MODEL_COLORS[m] for m in models_all], width=0.55)
    for i, m in enumerate(models_all):
        ax.text(i, params_ckpt[m] + 0.3, "%.1fM" % params_ckpt[m], ha="center",
               fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m].split(" (")[0] for m in models_all],
                                          fontsize=9)
    style_ax(ax, "(a) Parameters\n(exact, from saved checkpoints)", ylabel="Millions of parameters")

    ax = axes[1]
    x2 = np.arange(len(models_live))
    vals = [measured[m][1] for m in models_live]
    ax.bar(x2, vals, color=[MODEL_COLORS[m] for m in models_live], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, "%.1f" % v, ha="center", fontsize=9, color=INK)
    ax.set_xticks(x2); ax.set_xticklabels([MODEL_LABELS[m].split(" (")[0] for m in models_live],
                                           fontsize=9.5)
    style_ax(ax, "(b) Inference FLOPs, 3-frame bag\n(measured, torch.profiler)",
            ylabel="GFLOPs / patient")

    ax = axes[2]
    vals2 = [measured[m][2] for m in models_live]
    ax.bar(x2, vals2, color=[MODEL_COLORS[m] for m in models_live], width=0.5)
    for i, v in enumerate(vals2):
        ax.text(i, v + 5, "%.0f ms" % v, ha="center", fontsize=9, color=INK)
    ax.set_xticks(x2); ax.set_xticklabels([MODEL_LABELS[m].split(" (")[0] for m in models_live],
                                           fontsize=9.5)
    style_ax(ax, "(c) CPU inference latency, 3-frame bag\n(measured on this machine, 20-run mean)",
            ylabel="Milliseconds / patient")

    fig.tight_layout()
    caption = (
        "Figure 9. Computational cost. (a) Exact parameter counts, computed by "
        "loading each model's real saved checkpoint and summing tensor sizes "
        "(RCAF %.2fM, MR-MIL %.2fM, DER-MIL %.2fM); the ~0.4M difference "
        "between MR-MIL and DER-MIL is the reliability head. (b-c) FLOPs and "
        "CPU wall-clock latency for a representative 3-frame patient bag, "
        "measured directly on this machine (torch.profiler, 20-run mean); "
        "RCAF is omitted from (b-c) because its model class was removed from "
        "the codebase after its checkpoint was produced, so it can no longer "
        "be instantiated for a live forward-pass measurement -- only the "
        "parameter count, read from the surviving weights, remains available. "
        "The masked_input evidence encoding used here runs one backbone pass "
        "per region (4x a single-branch model) rather than pooling from shared "
        "feature maps, which is the dominant cost for both models shown."
        % (params_ckpt["rcaf"], params_ckpt["mr_mil"], params_ckpt["der_mil"])
    )
    style.save(fig, "fig09_computational_cost_resnet50", caption)


if __name__ == "__main__":
    main()
