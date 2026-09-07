"""Figure 5 -- probability calibration on the ThyroidXL test cohort.

Reliability diagrams are computed directly from raw predicted probabilities
(p_raw, from the actual test_predictions.csv of each model) binned against
observed empirical accuracy; ECE bars come from calibration_table.csv, which
the pipeline produces from the same predictions via temperature scaling fit
on development-set OOF only (never on the test cohort).
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
from style import INK, INK_MUTED, MODEL_COLORS, MODEL_LABELS, NEUTRAL, style_ax

MODELS = ["rcaf", "mr_mil", "der_mil"]


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mean_p, acc, n = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        mean_p.append(p[m].mean())
        acc.append(y[m].mean())
        n.append(int(m.sum()))
    return np.array(mean_p), np.array(acc), np.array(n)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # ---- (a) reliability diagrams, raw probabilities, all three models -------- #
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle=(0, (4, 3)), linewidth=1,
           label="Perfect calibration")
    for m in MODELS:
        d = data.load_predictions(data.RESNET50[m], pcol="p_raw")
        mp, acc, n = reliability_bins(d.label.to_numpy(), d.p.to_numpy())
        ax.plot(mp, acc, marker="o", markersize=5, linewidth=1.8,
               color=MODEL_COLORS[m], label=MODEL_LABELS[m].split(" (")[0])
    style_ax(ax, "(a) Reliability diagram -- raw probabilities\nThyroidXL test, n=739, 10 bins",
            xlabel="Mean predicted probability", ylabel="Empirical malignancy rate")
    ax.legend(fontsize=8.5, loc="upper left")

    # ---- (b) ECE before/after prior-shift correction, all three models -------- #
    ax = axes[1]
    rows = []
    for m in MODELS:
        cal = pd.read_csv(data.require(data.RESNET50_CAL[m]))
        test = cal[cal.cohort == "Test set"]
        rows.append((m, test[test.condition.str.contains("Temp")]["ece"].iloc[0],
                    test[test.condition.str.contains("Prior")]["ece"].iloc[0]))
    x = np.arange(len(rows))
    w = 0.35
    ax.bar(x - w / 2, [r[1] for r in rows], w, color=[MODEL_COLORS[r[0]] for r in rows],
          alpha=0.55, label="Temperature scaling only")
    ax.bar(x + w / 2, [r[2] for r in rows], w, color=[MODEL_COLORS[r[0]] for r in rows],
          label="+ prior-shift correction")
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r[1] + 0.002, "%.3f" % r[1], ha="center", fontsize=8, color=INK_MUTED)
        ax.text(i + w / 2, r[2] + 0.002, "%.3f" % r[2], ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[r[0]].split(" (")[0] for r in rows],
                                          fontsize=9)
    style_ax(ax, "(b) Expected calibration error, ThyroidXL test",
            ylabel="ECE (lower is better)")
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    caption = (
        "Figure 5. Probability calibration on the ThyroidXL held-out test "
        "cohort. (a) Reliability diagrams built directly from each model's raw "
        "predicted probabilities (10 equal-width bins), plotting mean predicted "
        "probability against the observed malignancy rate in that bin. "
        "(b) Expected calibration error (ECE) on the test cohort, comparing "
        "development-only temperature scaling against the same calibration "
        "with an additional prior-shift correction for the change in "
        "malignancy prevalence between development (26.1%%) and test (47.8%%). "
        "The calibration temperature and correction are both fit on "
        "out-of-fold development predictions only and applied unchanged to the "
        "test cohort."
    )
    style.save(fig, "fig05_calibration_resnet50", caption)


if __name__ == "__main__":
    main()
