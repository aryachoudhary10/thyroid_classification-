"""Figure 7 -- statistical uncertainty and significance.

ConvNeXt-Tiny backbone. Per-fold OOF ROC-AUC is recomputed directly from each
fold's held-out logits, not read from a pre-aggregated summary. Bootstrap CIs
and DeLong tests are computed live from raw test-set predictions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, MODEL_LABELS, NEUTRAL, style_ax

MODELS = ["lesion_mil", "mr_mil", "der_mil"]


def per_fold_auc(oof_path: str) -> np.ndarray:
    d = pd.read_csv(data.require(oof_path))
    return np.array([roc_auc_score(g.label, g.logit) for _, g in d.groupby("fold")])


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    for i, m in enumerate(MODELS):
        folds = per_fold_auc(data.CONVNEXT_OOF[m])
        jitter = (np.random.RandomState(0).rand(len(folds)) - 0.5) * 0.12
        ax.scatter(np.full(len(folds), i) + jitter, folds, s=45,
                  color=MODEL_COLORS[m], edgecolor=INK, linewidth=0.5, zorder=3)
        ax.scatter([i], [folds.mean()], marker="_", s=900, linewidth=2.5,
                  color=INK, zorder=4)
        ax.text(i + 0.28, folds.mean(), "%.4f ± %.4f" % (folds.mean(), folds.std()),
               fontsize=8, color=INK_MUTED, va="center")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m].split(" (")[0] for m in MODELS], fontsize=9.5)
    ax.set_xlim(-0.5, len(MODELS) - 0.15)
    style_ax(ax, "(a) 5-fold out-of-fold ROC-AUC, ConvNeXt-Tiny\nThyroidXL development cohort, n=3354",
            ylabel="ROC-AUC (per fold)")

    ax = axes[1]
    preds = {m: data.load_predictions(data.CONVNEXT_PRED[m], pcol="p_raw") for m in MODELS}
    rows = []
    for m in MODELS:
        ci = data.ci_from_predictions(preds[m])
        pt, lo, hi = ci["roc_auc"]
        rows.append((m, pt, lo, hi))
    y = np.arange(len(rows))
    for i, (m, pt, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=MODEL_COLORS[m], linewidth=2.5)
        ax.scatter([pt], [i], color=MODEL_COLORS[m], s=90, zorder=3,
                  edgecolor=INK, linewidth=0.6)
        ax.text(hi + 0.006, i, "%.4f [%.3f, %.3f]" % (pt, lo, hi), fontsize=8,
               va="center", color=INK)
    ax.set_yticks(y); ax.set_yticklabels([MODEL_LABELS[m].split(" (")[0] for m in MODELS],
                                          fontsize=9.5)
    ax.set_xlim(0.90, 1.03)
    style_ax(ax, "(b) Test-set ROC-AUC, 95%% bootstrap CI\nConvNeXt-Tiny (2000 resamples, n=739)",
            xlabel="ROC-AUC")

    fig.tight_layout()
    dl = data.delong(preds["der_mil"], preds["mr_mil"])
    fold_means = {m: per_fold_auc(data.CONVNEXT_OOF[m]) for m in MODELS}
    caption = (
        "Figure 7. Statistical uncertainty, ConvNeXt-Tiny backbone. "
        "(a) Out-of-fold ROC-AUC on each of the 5 grouped patient-level "
        "development folds (der_mil %.4f±%.4f, mr_mil %.4f±%.4f, lesion-only "
        "%.4f±%.4f; mean±SD). (b) Test-set ROC-AUC with 95%% bootstrap "
        "confidence intervals from the single-shot evaluation on the "
        "untouched test cohort. Paired DeLong test on identical patients, "
        "DER-MIL vs MR-MIL: diff=%+.4f, z=%.2f, p=%.4f (not significant)."
        % (fold_means["der_mil"].mean(), fold_means["der_mil"].std(),
           fold_means["mr_mil"].mean(), fold_means["mr_mil"].std(),
           fold_means["lesion_mil"].mean(), fold_means["lesion_mil"].std(),
           dl["diff"], dl["z"], dl["p_value"])
    )
    style.save(fig, "fig07_significance_convnext", caption)


if __name__ == "__main__":
    main()
