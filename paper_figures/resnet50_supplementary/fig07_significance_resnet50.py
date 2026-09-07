"""Figure 7 -- statistical uncertainty and significance.

Per-fold OOF ROC-AUC is recomputed directly from each fold's held-out logits
(oof.csv), not read from a pre-aggregated summary. Bootstrap CIs and DeLong
tests are computed live from the raw test-set predictions via
src/eval/metrics.py, the same functions the training pipeline itself uses.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PAPER_FIG_OUTDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, MODEL_LABELS, NEUTRAL, style_ax

MODELS = ["rcaf", "mr_mil", "der_mil"]


def per_fold_auc(oof_path: str) -> np.ndarray:
    d = pd.read_csv(data.require(oof_path))
    return np.array([roc_auc_score(g.label, g.logit) for _, g in d.groupby("fold")])


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # ---- (a) per-fold OOF ROC-AUC, development cohort ------------------------- #
    ax = axes[0]
    for i, m in enumerate(MODELS):
        folds = per_fold_auc(data.RESNET50_OOF[m])
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
    style_ax(ax, "(a) 5-fold out-of-fold ROC-AUC\nThyroidXL development cohort, n=3354",
            ylabel="ROC-AUC (per fold)")

    # ---- (b) test-set forest plot with pairwise DeLong annotations ------------ #
    ax = axes[1]
    preds = {m: data.load_predictions(data.RESNET50[m]) for m in MODELS}
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
        ax.text(hi + 0.002, i, "%.4f [%.3f, %.3f]" % (pt, lo, hi), fontsize=8,
               va="center", color=INK)
    ax.set_yticks(y); ax.set_yticklabels([MODEL_LABELS[m].split(" (")[0] for m in MODELS],
                                          fontsize=9.5)
    ax.set_xlim(0.975, 1.02)
    style_ax(ax, "(b) Test-set ROC-AUC, 95%% bootstrap CI\n(2000 resamples, n=739)",
            xlabel="ROC-AUC")

    fig.tight_layout()
    dl = {}
    for a, b in (("der_mil", "rcaf"), ("der_mil", "mr_mil")):
        dl["%s_vs_%s" % (a, b)] = data.delong(preds[a], preds[b])
    fold_means = {m: per_fold_auc(data.RESNET50_OOF[m]) for m in MODELS}
    caption = (
        "Figure 7. Statistical uncertainty across the model comparison. "
        "(a) Out-of-fold ROC-AUC on each of the 5 grouped patient-level "
        "development folds (der_mil %.4f±%.4f, mr_mil %.4f±%.4f, rcaf "
        "%.4f±%.4f; mean±SD), showing the fold-to-fold spread underlying the "
        "pooled development estimate. (b) Test-set ROC-AUC with 95%% bootstrap "
        "confidence intervals; the single-shot evaluation on the untouched "
        "test cohort (n=739). Paired DeLong tests on identical patients: "
        "DER-MIL vs RCAF diff=%+.4f, z=%.2f, p=%.4f (significant); DER-MIL vs "
        "MR-MIL diff=%+.4f, z=%.2f, p=%.4f (not significant)."
        % (fold_means["der_mil"].mean(), fold_means["der_mil"].std(),
           fold_means["mr_mil"].mean(), fold_means["mr_mil"].std(),
           fold_means["rcaf"].mean(), fold_means["rcaf"].std(),
           dl["der_mil_vs_rcaf"]["diff"], dl["der_mil_vs_rcaf"]["z"],
           dl["der_mil_vs_rcaf"]["p_value"],
           dl["der_mil_vs_mr_mil"]["diff"], dl["der_mil_vs_mr_mil"]["z"],
           dl["der_mil_vs_mr_mil"]["p_value"])
    )
    style.save(fig, "fig07_significance_resnet50", caption)


if __name__ == "__main__":
    main()
