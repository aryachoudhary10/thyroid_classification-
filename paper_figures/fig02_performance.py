"""Figure 2 -- patient-level discrimination on the ThyroidXL held-out test set.

ROC and PR curves, confusion matrices, and the baseline comparison bar chart
are all computed directly from raw prediction CSVs (patient_id, label, p_raw)
via sklearn -- nothing here is a pre-rendered number from a report.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve)

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, MODEL_LABELS, NEUTRAL, style_ax

MODELS = ["rcaf", "lesion_mil", "mr_mil", "der_mil"]
# lesion_mil (ResNet-50) test predictions were never produced in the hires
# namespace ablation was ConvNeXt-only; ResNet-50 headline comparison uses the
# three models that were actually run there.
MODELS_RESNET50 = ["rcaf", "mr_mil", "der_mil"]


def main() -> None:
    preds = {m: data.load_predictions(data.RESNET50[m]) for m in MODELS_RESNET50}

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])

    # ---- (a) ROC curves ------------------------------------------------------ #
    ax = fig.add_subplot(gs[0, 0])
    for m in MODELS_RESNET50:
        d = preds[m]
        fpr, tpr, _ = roc_curve(d.label, d.p)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], linewidth=2.0,
               label="%s (AUC %.3f)" % (MODEL_LABELS[m].split(" (")[0], a))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle=(0, (4, 3)), linewidth=1)
    style_ax(ax, "(a) ROC -- ThyroidXL test (n=739)", "False positive rate",
            "True positive rate")
    ax.legend(fontsize=8, loc="lower right")

    # ---- (b) PR curves --------------------------------------------------------#
    ax = fig.add_subplot(gs[0, 1])
    prevalence = float(preds["der_mil"].label.mean())
    for m in MODELS_RESNET50:
        d = preds[m]
        prec, rec, _ = precision_recall_curve(d.label, d.p)
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=MODEL_COLORS[m], linewidth=2.0,
               label="%s (AP %.3f)" % (MODEL_LABELS[m].split(" (")[0], ap))
    ax.axhline(prevalence, color=NEUTRAL, linestyle=(0, (4, 3)), linewidth=1,
              label="prevalence (%.2f)" % prevalence)
    style_ax(ax, "(b) Precision--recall -- ThyroidXL test", "Recall", "Precision")
    ax.legend(fontsize=8, loc="lower left")

    # ---- (c) baseline comparison bar chart, with 95% bootstrap CI ------------ #
    ax = fig.add_subplot(gs[0, 2])
    ys, cis, cols, labs = [], [], [], []
    for m in MODELS_RESNET50:
        ci = data.ci_from_predictions(preds[m])
        pt, lo, hi = ci["roc_auc"]
        ys.append(pt); cis.append((pt - lo, hi - pt))
        cols.append(MODEL_COLORS[m]); labs.append(MODEL_LABELS[m].split(" (")[0])
    yb = np.arange(len(MODELS_RESNET50))
    ax.barh(yb, ys, xerr=np.array(cis).T, color=cols, height=0.55,
           error_kw=dict(ecolor=INK_MUTED, capsize=3, linewidth=1.2))
    for i, v in enumerate(ys):
        ax.text(v + 0.006, i, "%.3f" % v, va="center", fontsize=9, color=INK)
    ax.set_yticks(yb); ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlim(0.95, 1.0)
    style_ax(ax, "(c) ROC-AUC, 95% bootstrap CI", xlabel="ROC-AUC")

    # ---- (d-f) confusion matrices -------------------------------------------- #
    for i, m in enumerate(MODELS_RESNET50):
        ax = fig.add_subplot(gs[1, i])
        d = preds[m]
        cm = confusion_matrix(d.label, (d.p >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for r in range(2):
            for c in range(2):
                frac = cm[r, c] / max(cm.max(), 1)
                ax.text(c, r, format(int(cm[r, c]), ","), ha="center", va="center",
                       fontsize=14, color="white" if frac > 0.55 else INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred.\nbenign", "Pred.\nmalignant"],
                                                   fontsize=8.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True\nbenign", "True\nmalignant"],
                                                   fontsize=8.5)
        ax.set_title("(%s) %s" % ("def"[i], MODEL_LABELS[m].split(" (")[0]),
                    fontsize=10.5, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(False)
        ax.tick_params(length=0)

    fig.tight_layout()
    dl_rcaf = data.delong(preds["der_mil"], preds["rcaf"])
    dl_mrmil = data.delong(preds["der_mil"], preds["mr_mil"])
    caption = (
        "Figure 2. Patient-level discrimination on the ThyroidXL held-out test "
        "cohort (n=739), evaluated once under the frozen leakage-free protocol. "
        "(a) ROC and (b) precision-recall curves computed directly from raw "
        "patient-level probabilities for RCAF (published two-branch gated-fusion "
        "baseline, reimplemented under this pipeline), MR-MIL (multi-region "
        "evidence, reliability disabled) and DER-MIL (proposed). (c) ROC-AUC "
        "with 95%% bootstrap confidence intervals (2000 resamples). (d-f) "
        "Confusion matrices at the raw 0.5 probability threshold. Paired DeLong "
        "test on identical patients: DER-MIL vs RCAF diff=%+.4f, "
        "z=%.2f, p=%.4f; DER-MIL vs MR-MIL diff=%+.4f, z=%.2f, p=%.4f."
        % (dl_rcaf["diff"], dl_rcaf["z"], dl_rcaf["p_value"],
           dl_mrmil["diff"], dl_mrmil["z"], dl_mrmil["p_value"])
    )
    style.save(fig, "fig02_performance_resnet50", caption)


if __name__ == "__main__":
    main()
