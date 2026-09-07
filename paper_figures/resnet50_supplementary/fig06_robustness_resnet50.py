"""Figure 6 -- robustness checks (DER-MIL, ResNet-50, ThyroidXL test).

Four independent stress tests, all from real evaluation runs on the frozen
test predictor: segmentation-quality degradation, frame-order permutation,
within-patient mask-identity shuffling, and removal of the frame the model
itself calls most/least reliable.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PAPER_FIG_OUTDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, NEUTRAL, style_ax

MASK_ORDER = ["clean", "dilate5", "erode5", "erode15", "dilate15", "zeros"]
MASK_LABELS = {"clean": "Clean\n(ground truth)", "dilate5": "Dilate\n5px",
              "erode5": "Erode\n5px", "erode15": "Erode\n15px",
              "dilate15": "Dilate\n15px", "zeros": "Zeroed\nmask"}


def main() -> None:
    mq = pd.read_csv(data.require(data.MASK_QUALITY)).set_index("condition").loc[MASK_ORDER].reset_index()
    fr = pd.read_csv(data.require(data.FRAME_REMOVAL))
    sc = pd.read_csv(data.require(data.SHORTCUT)).set_index("model")
    perm = data.load_json(data.PERMUTATION)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))

    # ---- (a) mask-quality degradation sweep ----------------------------------- #
    ax = axes[0, 0]
    x = np.arange(len(MASK_ORDER))
    ax.plot(x, mq["roc_auc"], marker="o", color=MODEL_COLORS["der_mil"],
           linewidth=2, markersize=7)
    for i, v in enumerate(mq["roc_auc"]):
        ax.annotate("%.3f" % v, (i, v), textcoords="offset points", xytext=(0, 8),
                   ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([MASK_LABELS[c] for c in MASK_ORDER], fontsize=8.5)
    ax.axhline(mq.loc[0, "roc_auc"], color=NEUTRAL, linestyle=(0, (4, 3)), linewidth=1)
    style_ax(ax, "(a) Segmentation-quality degradation, no retraining\nDER-MIL, ThyroidXL test",
            ylabel="ROC-AUC")

    # ---- (b) frame removal by the model's own reliability ranking ------------- #
    ax = axes[0, 1]
    order = ["highest_reliability", "random", "lowest_reliability"]
    lab = {"highest_reliability": "Remove\nhighest-R frame", "random": "Remove\nrandom frame",
          "lowest_reliability": "Remove\nlowest-R frame"}
    fr2 = fr.set_index("removed").loc[order].reset_index()
    xb = np.arange(len(order))
    ax.bar(xb, fr2["roc_auc_after"], color=[MODEL_COLORS["der_mil"], NEUTRAL,
                                            MODEL_COLORS["mr_mil"]], width=0.55)
    for i, v in enumerate(fr2["roc_auc_after"]):
        ax.text(i, v + 0.0004, "%.4f" % v, ha="center", fontsize=9, color=INK)
    ax.set_xticks(xb); ax.set_xticklabels([lab[o] for o in order], fontsize=9)
    ax.set_ylim(fr2["roc_auc_after"].min() - 0.002, fr2["roc_auc_after"].max() + 0.002)
    style_ax(ax, "(b) One frame removed, by DER-MIL's own\nreliability ranking (ThyroidXL test)",
            ylabel="ROC-AUC after removal")

    # ---- (c) within-patient mask-identity shuffle (shortcut probe) ------------ #
    ax = axes[1, 0]
    models = ["rcaf", "mr_mil", "der_mil"]
    xb = np.arange(len(models))
    w = 0.35
    ax.bar(xb - w / 2, [sc.loc[m, "roc_auc"] for m in models], w,
          color=[MODEL_COLORS[m] for m in models], label="Correct masks")
    ax.bar(xb + w / 2, [sc.loc[m, "roc_auc_shuffled"] for m in models], w,
          color=[MODEL_COLORS[m] for m in models], alpha=0.45,
          label="Within-patient shuffled masks")
    for i, m in enumerate(models):
        d = sc.loc[m, "d_roc_auc"]
        ax.text(i, max(sc.loc[m, "roc_auc"], sc.loc[m, "roc_auc_shuffled"]) + 0.001,
               "Δ=%+.4f" % d, ha="center", fontsize=8, color=INK_MUTED)
    ax.set_xticks(xb); ax.set_xticklabels(["RCAF", "MR-MIL", "DER-MIL"], fontsize=9.5)
    ax.set_ylim(0.98, 1.0)
    style_ax(ax, "(c) Within-patient mask-identity shuffle\n(shortcut-reliance probe)",
            ylabel="ROC-AUC")
    ax.legend(fontsize=8, loc="lower left")

    # ---- (d) permutation invariance + summary stat card ------------------------ #
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("(d) Frame-order permutation invariance (DER-MIL)", fontsize=10.5,
                color=INK, loc="left")
    rows = [
        ("Patients tested", "%d" % perm["n"]),
        ("Mean |Δ predicted probability|", "%.2e" % perm["mean_abs_delta_p"]),
        ("95th-percentile |Δp|", "%.2e" % perm["p95_abs_delta_p"]),
        ("Max |Δp| across all patients", "%.2e" % perm["max_abs_delta_p"]),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.78 - i * 0.18
        ax.text(0.0, y, k, fontsize=10, color=INK_MUTED, transform=ax.transAxes)
        ax.text(1.0, y, v, fontsize=10, color=INK, transform=ax.transAxes, ha="right")
    ax.text(0.0, 0.02, "Shuffling the order in which a patient's frames are fed\n"
                      "to the model changes its prediction by less than 1e-6:\n"
                      "the aggregation stage is exactly permutation invariant\n"
                      "by construction, not just empirically stable.",
           fontsize=8.5, color=INK_MUTED, transform=ax.transAxes, va="bottom")

    fig.tight_layout()
    caption = (
        "Figure 6. Robustness of DER-MIL on the ThyroidXL held-out test cohort, "
        "all evaluated on the single frozen predictor with no retraining. "
        "(a) ROC-AUC under progressively degraded segmentation masks, from "
        "ground truth to complete mask removal (delta from clean: %.4f at "
        "5px dilation to %.4f with zeroed masks). (b) Effect of removing one "
        "frame per patient chosen by the model's own reliability score: "
        "removing the frame DER-MIL itself scores as most reliable costs the "
        "most ranking performance (ROC-AUC %.4f after removal), more than "
        "removing a random frame (%.4f) or the frame it scores as least "
        "reliable (%.4f) -- the direction an informative ranking should "
        "produce. The mean per-patient probability shift does not follow the "
        "same order (%.4f, %.4f, %.4f respectively), so ranking impact and "
        "probability-magnitude impact are not interchangeable readings of "
        "this test. (c) Within-patient shuffling of which mask is paired with "
        "which frame -- a shortcut-reliance probe -- changes DER-MIL's ROC-AUC "
        "by only %+.4f, comparable to RCAF (%+.4f) and MR-MIL (%+.4f). "
        "(d) Frame-order permutation invariance, confirmed on %d test patients."
        % (mq.loc[mq.condition == "dilate5", "delta"].item(),
           mq.loc[mq.condition == "zeros", "delta"].item(),
           fr2.loc[fr2.removed == "highest_reliability", "roc_auc_after"].item(),
           fr2.loc[fr2.removed == "random", "roc_auc_after"].item(),
           fr2.loc[fr2.removed == "lowest_reliability", "roc_auc_after"].item(),
           fr2.loc[fr2.removed == "highest_reliability", "mean_abs_delta_p"].item(),
           fr2.loc[fr2.removed == "random", "mean_abs_delta_p"].item(),
           fr2.loc[fr2.removed == "lowest_reliability", "mean_abs_delta_p"].item(),
           sc.loc["der_mil", "d_roc_auc"], sc.loc["rcaf", "d_roc_auc"],
           sc.loc["mr_mil", "d_roc_auc"], perm["n"])
    )
    style.save(fig, "fig06_robustness_resnet50", caption)


if __name__ == "__main__":
    main()
