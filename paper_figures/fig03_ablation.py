"""Figure 3 -- ablation study: what does each architectural component add?

Two independent ablations, both backed by real predictions:
  (a-b) lesion-only -> multi-region (MR-MIL) -> +reliability (DER-MIL),
        on ConvNeXt-Tiny, mirroring the base paper's Table 11 structure.
  (c)   reliability fusion rule: mlp (unconstrained) vs linear (sign-
        constrained a*S - b*D - c*U) vs off (MR-MIL), same backbone.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, NEUTRAL, style_ax

LADDER = ["lesion_mil", "mr_mil", "der_mil"]
LADDER_LABELS = {"lesion_mil": "Lesion-only\n+ AttnMIL", "mr_mil": "+ Multi-region\nevidence (MR-MIL)",
                 "der_mil": "+ Reliability\n(DER-MIL)"}


def main() -> None:
    ab = pd.read_csv(data.require(data.CONVNEXT_ABLATION))
    ab = ab.set_index("model").loc[LADDER].reset_index()

    preds = {m: data.load_predictions(data.CONVNEXT_PRED[m], pcol="p_raw")
            for m in LADDER}
    preds_lin = data.load_predictions(data.DER_MIL_LINEAR_PRED, pcol="p_raw")
    coeffs = data.load_json(data.DER_MIL_LINEAR_COEFFS)["der_mil"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # ---- (a) ladder: ROC-AUC and F1 ------------------------------------------ #
    ax = axes[0]
    x = np.arange(len(LADDER))
    w = 0.35
    ax.bar(x - w / 2, ab["roc_auc"], w, color=[MODEL_COLORS[m] for m in LADDER],
          label="ROC-AUC")
    ax.bar(x + w / 2, ab["f1"], w, color=[MODEL_COLORS[m] for m in LADDER],
          alpha=0.45, label="F1 (raw @ 0.5)")
    for i, m in enumerate(LADDER):
        ax.text(i - w / 2, ab.loc[i, "roc_auc"] + 0.004, "%.3f" % ab.loc[i, "roc_auc"],
               ha="center", fontsize=8.5, color=INK)
        ax.text(i + w / 2, ab.loc[i, "f1"] + 0.004, "%.3f" % ab.loc[i, "f1"],
               ha="center", fontsize=8.5, color=INK_MUTED)
    ax.set_xticks(x); ax.set_xticklabels([LADDER_LABELS[m] for m in LADDER], fontsize=9)
    ax.set_ylim(0.80, 1.02)
    style_ax(ax, "(a) Ablation ladder -- ConvNeXt-Tiny\nThyroidXL test, n=739")
    ax.legend(fontsize=8.5, loc="lower right")

    # ---- (b) is the mr_mil->der_mil step significant? ------------------------- #
    ax = axes[1]
    dl = data.delong(preds["der_mil"], preds["mr_mil"])
    labels = ["MR-MIL\n(no reliability)", "DER-MIL\n(+ reliability)"]
    vals = [ab.loc[ab.model == "mr_mil", "roc_auc"].item(),
           ab.loc[ab.model == "der_mil", "roc_auc"].item()]
    cols = [MODEL_COLORS["mr_mil"], MODEL_COLORS["der_mil"]]
    ax.bar([0, 1], vals, color=cols, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0006, "%.4f" % v, ha="center", fontsize=9, color=INK)
    ax.plot([0, 0, 1, 1], [max(vals) + 0.0025, max(vals) + 0.004,
                          max(vals) + 0.004, max(vals) + 0.0025],
           color=INK_MUTED, linewidth=1)
    ax.text(0.5, max(vals) + 0.0045, "DeLong p=%.3f (n.s.)" % dl["p_value"]
           if dl["p_value"] >= 0.05 else "DeLong p=%.3f *" % dl["p_value"],
           ha="center", fontsize=9, color=INK_MUTED)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(min(vals) - 0.003, max(vals) + 0.008)
    style_ax(ax, "(b) Does reliability help?\nPaired test, identical patients")

    # ---- (c) fusion-rule probe: mlp vs linear vs off -------------------------- #
    ax = axes[2]
    rows = [("mr_mil\n(reliability off)", ab.loc[ab.model == "mr_mil", "roc_auc"].item(),
            MODEL_COLORS["mr_mil"]),
           ("der_mil\n(mlp fusion)", ab.loc[ab.model == "der_mil", "roc_auc"].item(),
            MODEL_COLORS["der_mil"]),
           ("der_mil\n(linear fusion)", data.metrics_from_predictions(preds_lin)["roc_auc"],
            MODEL_COLORS["der_mil_linear"])]
    xb = np.arange(len(rows))
    ax.bar(xb, [r[1] for r in rows], color=[r[2] for r in rows], width=0.55)
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 0.0006, "%.4f" % r[1], ha="center", fontsize=9, color=INK)
    ax.set_xticks(xb); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(min(r[1] for r in rows) - 0.003, max(r[1] for r in rows) + 0.006)
    style_ax(ax, "(c) Fusion-rule probe\nR = sigmoid(a*S - b*D - c*U + tanh(I))")
    ax.text(0.02, 0.03,
           "learned a=%.3f, b=%.3f, c=%.3f\n(init: a=b=c=1.313)" %
           (coeffs["alpha"], coeffs["beta"], coeffs["gamma"]),
           transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")

    fig.tight_layout()
    caption = (
        "Figure 3. Ablation study on ConvNeXt-Tiny (ThyroidXL test, n=739). "
        "(a) Progressive addition of components, mirroring the structure of the "
        "published RCAF ablation: lesion-only evidence with attention-based MIL "
        "(ROC-AUC %.4f), adding margin/peri/global regions (MR-MIL, %.4f), then "
        "adding the reliability mechanism (DER-MIL, %.4f). The largest gain "
        "comes from multi-region evidence; reliability's contribution to "
        "discrimination is small. (b) Paired DeLong test between MR-MIL and "
        "DER-MIL on the identical 739 patients (diff=%+.4f, z=%.2f, p=%.4f): "
        "not statistically significant. (c) Replacing the unconstrained MLP "
        "fusion rule with the sign-constrained linear form R = "
        "sigmoid(tanh(I)+aS-bD-cU) does not recover the gap; the learned "
        "coefficients a, b, c remain within 3%% of their softplus(1.0) "
        "initialisation, indicating the fusion parameters received little "
        "training signal under this task loss."
        % (ab.loc[ab.model == "lesion_mil", "roc_auc"].item(),
           ab.loc[ab.model == "mr_mil", "roc_auc"].item(),
           ab.loc[ab.model == "der_mil", "roc_auc"].item(),
           dl["diff"], dl["z"], dl["p_value"])
    )
    style.save(fig, "fig03_ablation_ladder", caption)


if __name__ == "__main__":
    main()
