"""Figure 8 -- TN5000 external (cross-domain) validation, ConvNeXt-Tiny.

Only DER-MIL was adapted and evaluated on TN5000 with the ConvNeXt-Tiny
backbone -- MR-MIL and lesion-only were not run there. TN5000 also ships zero
pixel-level masks (0 of 5000 images), so the "pixel" mask-adaptation arm
silently falls back to the same bounding-box rectangles as the "bbox" arm:
these are two independent training runs of an IDENTICAL configuration, not
two different mask types, and are shown that way here -- as a direct,
measured estimate of session-to-session training variance, not as a
supervised-vs-alternative-mask comparison.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, NEUTRAL, style_ax


def main() -> None:
    res = pd.read_csv(data.require(data.CONVNEXT_TN5000)).set_index("mask_type")
    bbox = data.load_predictions(data.CONVNEXT_TN5000_PRED["bbox"], pcol="p")
    pixel = data.load_predictions(data.CONVNEXT_TN5000_PRED["pixel"], pcol="p")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

    # ---- (a) headline external result, with CI --------------------------------#
    ax = axes[0]
    pt = res.loc["bbox", "roc_auc"]
    lo, hi = [float(x) for x in res.loc["bbox", "roc_auc_ci"].strip("[]").split("-")]
    ax.barh([0], [pt], xerr=[[pt - lo], [hi - pt]], color=MODEL_COLORS["der_mil"],
           height=0.4, error_kw=dict(ecolor=INK_MUTED, capsize=4, linewidth=1.4))
    ax.text(pt + 0.01, 0, "%.3f [%.3f, %.3f]" % (pt, lo, hi), va="center", fontsize=10,
           color=INK)
    ax.set_yticks([0]); ax.set_yticklabels(["DER-MIL\n(ConvNeXt-Tiny)"], fontsize=10)
    ax.set_xlim(0.80, 1.0)
    style_ax(ax, "(a) TN5000 external validation\nbbox-mask adaptation, n=250 (class-balanced)",
            xlabel="ROC-AUC")

    # ---- (b) same-config run-to-run variance -----------------------------------#
    ax = axes[1]
    dl = data.delong(bbox, pixel)
    labels = ["Session 1\n(\"bbox\")", "Session 2\n(\"pixel\", fell back to\nidentical bbox input)"]
    vals = [res.loc["bbox", "roc_auc"], res.loc["pixel", "roc_auc"]]
    bars = ax.bar([0, 1], vals, color=MODEL_COLORS["der_mil"], width=0.5)
    bars[1].set_alpha(0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, "%.4f" % v, ha="center", fontsize=10, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.85, 1.0)
    style_ax(ax, "(b) Two runs, identical config & input\n(TN5000 ships 0 pixel masks of 5000)",
            ylabel="ROC-AUC")
    ax.text(0.5, 0.05, "spread = %.4f AUC (training\nstochasticity only; DeLong\np=%.3f treating them as paired)"
           % (abs(vals[0] - vals[1]), dl["p_value"]),
           ha="center", fontsize=8, color=INK_MUTED, transform=ax.transAxes)

    fig.tight_layout()
    caption = (
        "Figure 8. External (cross-domain) validation on TN5000, ConvNeXt-Tiny "
        "backbone, DER-MIL only -- MR-MIL and lesion-only were not evaluated "
        "on TN5000 with this backbone. (a) ROC-AUC on the class-balanced "
        "250-image evaluation subset with 95%% bootstrap CI, adapted with "
        "bounding-box masks. (b) TN5000 ships pixel-level masks for 0 of 5000 "
        "images, so the nominal \"pixel-mask\" adaptation run received "
        "identical bounding-box input to the \"bbox\" run; the two sessions "
        "are therefore two independent trainings of the same configuration, "
        "and their %.4f-point spread is a direct, measured estimate of "
        "training-stochasticity noise on this 250-image subset rather than "
        "an effect of mask type."
        % abs(vals[0] - vals[1])
    )
    style.save(fig, "fig08_external_tn5000_convnext", caption)


if __name__ == "__main__":
    main()
