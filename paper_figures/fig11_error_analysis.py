"""Figure 11 -- error analysis (DER-MIL, ResNet-50, ThyroidXL test).

TP/TN/FP/FN counts and their breakdown by TI-RADS category and frames-per-bag
come from a direct join of the model's real test predictions
(test_predictions.csv) against the real patient manifest on patient_id -- no
representative example IMAGES are shown here, because no raw ThyroidXL/TN5000
pixel data exists anywhere in this local environment (everything was trained
and evaluated on Kaggle; only result tables and a handful of checkpoints were
ever fetched back). That gap is reported explicitly rather than illustrated
with a placeholder or a stand-in image from elsewhere.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from style import CLASS_COLORS, INK, INK_MUTED, MODEL_COLORS, NEUTRAL, style_ax

OUTCOME_COLORS = {"TP": "#0072B2", "TN": "#56B4E9", "FP": "#D55E00", "FN": "#9B4F96"}


def main() -> None:
    pred = pd.read_csv(data.require(data.RESNET50["der_mil"]), dtype={"patient_id": str})
    pred["patient_id"] = pred["patient_id"].map(data.norm_pid)
    man = pd.read_csv(data.require(data.MANIFEST), dtype={"patient_id": str})
    man["patient_id"] = man["patient_id"].map(data.norm_pid)
    m = pred.merge(man[["patient_id", "tirads", "n_frames"]], on="patient_id", how="left")

    m["pred"] = (m["p_raw"] >= 0.5).astype(int)
    m["outcome"] = "TN"
    m.loc[(m.label == 1) & (m.pred == 1), "outcome"] = "TP"
    m.loc[(m.label == 0) & (m.pred == 1), "outcome"] = "FP"
    m.loc[(m.label == 1) & (m.pred == 0), "outcome"] = "FN"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # ---- (a) outcome counts ----------------------------------------------------#
    ax = axes[0]
    order = ["TP", "TN", "FP", "FN"]
    counts = m["outcome"].value_counts().reindex(order)
    ax.bar(range(4), counts.values, color=[OUTCOME_COLORS[o] for o in order], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 8, "%d" % v, ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(order, fontsize=11)
    style_ax(ax, "(a) Outcome counts, threshold 0.5\nThyroidXL test, n=739", ylabel="Patients")

    # ---- (b) false negatives by TI-RADS, vs all malignant patients ------------- #
    ax = axes[1]
    mal = m[m.label == 1].copy()
    mal["tirads"] = mal["tirads"].fillna(-1).astype(int)
    cats = sorted(c for c in mal["tirads"].unique() if c > 0)
    fn_frac = [100 * ((mal.tirads == c) & (mal.outcome == "FN")).sum() / max((mal.tirads == c).sum(), 1)
              for c in cats]
    n_per_cat = [int((mal.tirads == c).sum()) for c in cats]
    ax.bar(range(len(cats)), fn_frac, color=OUTCOME_COLORS["FN"], width=0.55)
    for i, (f, n) in enumerate(zip(fn_frac, n_per_cat)):
        ax.text(i, f + 1, "%.0f%%\n(n=%d)" % (f, n), ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(["TI-RADS %d" % c for c in cats], fontsize=9)
    style_ax(ax, "(b) False-negative rate by TI-RADS\n(among malignant test patients)",
            ylabel="False-negative rate (%)")

    # ---- (c) outcome by frames-per-bag ------------------------------------------#
    ax = axes[2]
    for o in order:
        sub = m[m.outcome == o]
        ax.scatter(sub["n_frames"] + np.random.RandomState(hash(o) % 2**31).uniform(-0.15, 0.15, len(sub)),
                  [o] * len(sub), s=18, alpha=0.55, color=OUTCOME_COLORS[o])
    style_ax(ax, "(c) Outcome vs frames available per patient\n(jittered for visibility)",
            xlabel="Frames in bag")
    ax.set_yticks(range(4)); ax.set_yticklabels(order[::-1] if False else order)

    fig.tight_layout()
    n_fn = int((m.outcome == "FN").sum())
    n_mal = int((m.label == 1).sum())
    hi_tirads_fn = int(((m.outcome == "FN") & (m.tirads >= 4)).sum())
    caption = (
        "Figure 11. Error analysis, DER-MIL on the ThyroidXL held-out test "
        "cohort (n=739, threshold 0.5), from a direct join of the model's raw "
        "predictions with the patient manifest on patient_id. (a) Outcome "
        "counts (TP=%d, TN=%d, FP=%d, FN=%d), matching the confusion matrix in "
        "Figure 2. (b) False-negative rate among the %d malignant test "
        "patients, broken down by TI-RADS category; %d of the %d false "
        "negatives (%.0f%%) occur in TI-RADS 4-5 nodules, i.e. the model's "
        "misses are concentrated in cases the clinical scoring system also "
        "flags as higher-risk rather than in unambiguous low-risk nodules. "
        "(c) Outcome against the number of frames available for that patient's "
        "bag; no image or embedding data was available locally to illustrate "
        "individual TP/FP/FN/TN cases with representative frames -- ThyroidXL "
        "and TN5000 pixel data reside only on Kaggle, where all training and "
        "evaluation ran, and were never downloaded to this environment."
        % (counts["TP"], counts["TN"], counts["FP"], counts["FN"], n_mal,
           hi_tirads_fn, n_fn, 100 * hi_tirads_fn / max(n_fn, 1))
    )
    style.save(fig, "fig11_error_analysis", caption)


if __name__ == "__main__":
    main()
