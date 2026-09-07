"""Figure 8 -- TN5000 external (cross-domain) validation, ConvNeXt-Tiny.

DER-MIL and MR-MIL were both adapted and evaluated on TN5000 with the
ConvNeXt-Tiny backbone (lesion-only was not). TN5000 also ships zero
pixel-level masks (0 of 5000 images), so the "pixel" mask-adaptation arm
silently falls back to the same bounding-box rectangles as the "bbox" arm for
both models: these are pairs of independent training runs of an IDENTICAL
configuration, not two different mask types, and are shown that way here.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, MODEL_LABELS, style_ax

MODELS = ["mr_mil", "der_mil"]


def main() -> None:
    res = {m: pd.read_csv(data.require(data.CONVNEXT_TN5000_ALL[m])).set_index("mask_type")
          for m in MODELS}
    preds = {(m, v): data.load_predictions(data.CONVNEXT_TN5000_PRED_ALL[(m, v)], pcol="p")
            for m in MODELS for v in ("bbox", "pixel")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- (a) headline external comparison, bbox arm, with CI ------------------ #
    ax = axes[0]
    y = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        pt = res[m].loc["bbox", "roc_auc"]
        lo, hi = [float(x) for x in res[m].loc["bbox", "roc_auc_ci"].strip("[]").split("-")]
        ax.barh([i], [pt], xerr=[[pt - lo], [hi - pt]], color=MODEL_COLORS[m],
               height=0.5, error_kw=dict(ecolor=INK_MUTED, capsize=4, linewidth=1.4))
        ax.text(hi + 0.005, i, "%.3f [%.3f, %.3f]" % (pt, lo, hi), va="center", fontsize=9.5,
               color=INK)
    ax.set_yticks(y); ax.set_yticklabels([MODEL_LABELS[m].split(" (")[0] for m in MODELS], fontsize=10)
    ax.set_xlim(0.85, 1.0)
    dl = data.delong(preds[("der_mil", "bbox")], preds[("mr_mil", "bbox")])
    style_ax(ax, "(a) TN5000 external validation, ConvNeXt-Tiny\nbbox-mask adaptation, n=250 (class-balanced)",
            xlabel="ROC-AUC")
    ax.text(0.02, -0.22, "Paired DeLong: diff=%+.4f, p=%.3f (n.s.)" % (dl["diff"], dl["p_value"]),
           transform=ax.transAxes, fontsize=8.5, color=INK_MUTED, clip_on=False)

    # ---- (b) same-config run-to-run variance, both models ---------------------- #
    ax = axes[1]
    x = np.arange(len(MODELS))
    w = 0.35
    bbox_vals = [res[m].loc["bbox", "roc_auc"] for m in MODELS]
    pixel_vals = [res[m].loc["pixel", "roc_auc"] for m in MODELS]
    ax.bar(x - w / 2, bbox_vals, w, color=[MODEL_COLORS[m] for m in MODELS],
          label="Session 1 (\"bbox\")")
    bars2 = ax.bar(x + w / 2, pixel_vals, w, color=[MODEL_COLORS[m] for m in MODELS],
                   label="Session 2 (\"pixel\", identical input)")
    for b in bars2:
        b.set_alpha(0.55)
    for i, (bv, pv) in enumerate(zip(bbox_vals, pixel_vals)):
        ax.text(i - w / 2, bv + 0.003, "%.3f" % bv, ha="center", fontsize=8, color=INK)
        ax.text(i + w / 2, pv + 0.003, "%.3f" % pv, ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m].split(" (")[0] for m in MODELS], fontsize=10)
    ax.set_ylim(0.9, 1.0)
    style_ax(ax, "(b) Two runs, identical config & input, per model\n(TN5000 ships 0 pixel masks of 5000)",
            ylabel="ROC-AUC")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    spread = {m: abs(res[m].loc["bbox", "roc_auc"] - res[m].loc["pixel", "roc_auc"]) for m in MODELS}
    caption = (
        "Figure 8. External (cross-domain) validation on TN5000, ConvNeXt-"
        "Tiny backbone. (a) ROC-AUC on the class-balanced 250-image "
        "evaluation subset with 95%% bootstrap CI, bounding-box mask "
        "adaptation, for both DER-MIL and MR-MIL. Unlike the ResNet-50 "
        "result (where DER-MIL significantly outperformed MR-MIL on both "
        "mask arms, p=0.044 and p=0.031), the two are not significantly "
        "different on ConvNeXt-Tiny (diff=%+.4f, paired DeLong p=%.3f on "
        "the identical 250 images) and MR-MIL nominally leads. (b) TN5000 "
        "ships pixel-level masks for 0 of 5000 images, so the nominal "
        "\"pixel-mask\" adaptation run received identical bounding-box "
        "input to the \"bbox\" run for both models; each pair is therefore "
        "two independent trainings of the same configuration, with spreads "
        "of %.4f (DER-MIL) and %.4f (MR-MIL) AUC from training stochasticity "
        "alone. The DER-MIL-over-MR-MIL external advantage seen on ResNet-50 "
        "does not replicate on this backbone."
        % (dl["diff"], dl["p_value"], spread["der_mil"], spread["mr_mil"])
    )
    style.save(fig, "fig08_external_tn5000_convnext", caption)


if __name__ == "__main__":
    main()
