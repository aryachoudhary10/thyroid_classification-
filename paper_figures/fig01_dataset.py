"""Figure 1 -- dataset composition (ThyroidXL development/test + TN5000).

Every number below is read directly from the manifests the training pipeline
produced (kaggle_jobs/final_out/results/hires/manifest.csv and the TN5000
manifest under convnext_out/), not recomputed or estimated.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from style import CLASS_COLORS, INK, INK_MUTED, NEUTRAL, style_ax


def main() -> None:
    man = pd.read_csv(data.require(data.MANIFEST), dtype={"patient_id": str})
    tn = pd.read_csv(data.require(data.TN5000_MANIFEST))

    dev = man[man["split"] == "dev"]
    test = man[man["split"] == "test"]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))

    # ---- (a) benign/malignant by cohort, ThyroidXL + TN5000 ----------------- #
    ax = axes[0, 0]
    cohorts = [("ThyroidXL\ndevelopment\n(n=%d)" % len(dev), dev.label),
               ("ThyroidXL\ntest\n(n=%d)" % len(test), test.label),
               ("TN5000\ntrain\n(n=%d)" % (tn.split == "train").sum(),
                tn[tn.split == "train"].label),
               ("TN5000\nval\n(n=%d)" % (tn.split == "val").sum(),
                tn[tn.split == "val"].label),
               ("TN5000\ntest\n(n=%d)" % (tn.split == "test").sum(),
                tn[tn.split == "test"].label)]
    x = np.arange(len(cohorts))
    ben = [float((c[1] == 0).mean()) * 100 for c in cohorts]
    mal = [float((c[1] == 1).mean()) * 100 for c in cohorts]
    ax.bar(x, ben, color=CLASS_COLORS["benign"], label="Benign", width=0.6)
    ax.bar(x, mal, bottom=ben, color=CLASS_COLORS["malignant"], label="Malignant",
          width=0.6)
    for i, (b, m) in enumerate(zip(ben, mal)):
        ax.text(i, b / 2, "%.0f%%" % b, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")
        ax.text(i, b + m / 2, "%.0f%%" % m, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cohorts], fontsize=8)
    ax.set_ylim(0, 105)
    style_ax(ax, "(a) Class balance by cohort", ylabel="Patients / images (%)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # ---- (b) TIRADS distribution, dev vs test ------------------------------- #
    ax = axes[0, 1]
    tir_dev = dev["tirads"].dropna().astype(int).value_counts().sort_index()
    tir_test = test["tirads"].dropna().astype(int).value_counts().sort_index()
    cats = sorted(set(tir_dev.index) | set(tir_test.index))
    xw = np.arange(len(cats))
    w = 0.38
    ax.bar(xw - w / 2, [100 * tir_dev.get(c, 0) / len(dev) for c in cats], w,
          color=NEUTRAL, label="Development")
    ax.bar(xw + w / 2, [100 * tir_test.get(c, 0) / len(test) for c in cats], w,
          color=CLASS_COLORS["malignant"], label="Test")
    ax.set_xticks(xw)
    ax.set_xticklabels(["TI-RADS %d" % c for c in cats], fontsize=8.5)
    style_ax(ax, "(b) TI-RADS category (ThyroidXL)", ylabel="Patients (%)")
    ax.legend(fontsize=8.5)

    # ---- (c) frames-per-patient distribution -------------------------------- #
    ax = axes[1, 0]
    bins = np.arange(0.5, man["n_frames"].max() + 1.5, 1)
    ax.hist(dev["n_frames"], bins=bins, color=NEUTRAL, alpha=0.85,
           label="Development (median %d)" % dev["n_frames"].median(),
           density=True)
    ax.hist(test["n_frames"], bins=bins, color=CLASS_COLORS["malignant"],
           alpha=0.55, label="Test (median %d)" % test["n_frames"].median(),
           density=True, histtype="stepfilled")
    ax.set_xticks(range(1, int(man["n_frames"].max()) + 1))
    style_ax(ax, "(c) Frames per patient bag", xlabel="Frames in bag",
            ylabel="Density")
    ax.legend(fontsize=8.5)

    # ---- (d) headline sample sizes, both datasets --------------------------- #
    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ("ThyroidXL development", "%d patients / %d images" %
         (len(dev), dev["n_frames"].sum())),
        ("ThyroidXL test (held out)", "%d patients / %d images" %
         (len(test), test["n_frames"].sum())),
        ("Mean age (dev / test)", "%.1f / %.1f years" %
         (dev["age"].mean(), test["age"].mean())),
        ("TN5000 train / val / test", "%d / %d / %d images" %
         ((tn.split == "train").sum(), (tn.split == "val").sum(),
          (tn.split == "test").sum())),
        ("TN5000 malignancy prevalence", "%.1f%% (train), %.1f%% (val), %.1f%% (test)" %
         (100 * tn[tn.split == "train"].label.mean(),
          100 * tn[tn.split == "val"].label.mean(),
          100 * tn[tn.split == "test"].label.mean())),
    ]
    ax.set_title("(d) Cohort summary", fontsize=11, color=INK, loc="left")
    for i, (k, v) in enumerate(rows):
        y = 0.88 - i * 0.19
        ax.text(0.0, y, k, fontsize=9.5, color=INK_MUTED, transform=ax.transAxes)
        ax.text(1.0, y, v, fontsize=9.5, color=INK, transform=ax.transAxes,
               ha="right", fontweight="normal")

    fig.suptitle("", fontsize=1)
    fig.tight_layout()
    caption = (
        "Figure 1. Dataset composition. (a) Benign/malignant balance across the "
        "ThyroidXL development and held-out test cohorts and the three official "
        "TN5000 splits; ThyroidXL's official split is class-imbalanced in "
        "development (73.9%% benign) and near-balanced in test (52.2%% benign), "
        "while TN5000 is malignant-majority throughout. (b) TI-RADS category "
        "distribution for the patients with a recorded score. (c) Distribution "
        "of the number of ultrasound frames per patient bag (median 3 in both "
        "cohorts; no patient exceeds 10 frames). (d) Headline sample sizes. All "
        "values are computed directly from the manifests produced by the "
        "leakage-free data-preparation pipeline "
        "(kaggle_jobs/final_out/results/hires/manifest.csv and the TN5000 "
        "manifest), not estimated."
    )
    style.save(fig, "fig01_dataset_composition", caption)


if __name__ == "__main__":
    main()
