"""Figure 8 -- TN5000 external (cross-domain) validation.

TN5000 is a Chinese thyroid ultrasound dataset, acquired under different
scanners and a different population prior, used only for cross-domain
generalisation -- never for training. Panel (c) is included deliberately: two
independent sessions ran the identical bbox-mask adaptation config for DER-MIL
and produced 0.914 and 0.934 ROC-AUC. That spread is real, measured, and
larger than several of the model-vs-model gaps discussed elsewhere in this
project, so it is shown rather than only the more favourable run.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import style
from style import INK, INK_MUTED, MODEL_COLORS, MODEL_LABELS, NEUTRAL, style_ax

ARM_LABELS = {"bbox": "Bbox masks\n(supervised)", "unet": "U-Net masks\n(supervised)",
             "zero_shot": "Zero-shot\n(no adaptation)", "upl": "+ Pseudo-\nlabeling",
             "upl_tent": "+ Pseudo-label\n+ TENT", "retrieval_k5": "Retrieval\nbags (k=5)"}
ARM_ORDER = ["bbox", "unet", "zero_shot", "upl", "upl_tent", "retrieval_k5"]


def main() -> None:
    ladder = pd.read_csv(data.require(data.TN5000_ARM_LADDER))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    # ---- (a) full arm ladder, der_mil vs mr_mil, ResNet-50 --------------------- #
    ax = axes[0]
    x = np.arange(len(ARM_ORDER))
    w = 0.35
    for j, m in enumerate(("der_mil", "mr_mil")):
        sub = ladder[ladder.model == m].set_index("arm").loc[ARM_ORDER]
        off = -w / 2 if m == "der_mil" else w / 2
        ax.bar(x + off, sub["auc"], w, color=MODEL_COLORS[m],
              label=MODEL_LABELS[m].split(" (")[0])
    ax.axvline(1.5, color=NEUTRAL, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(0.75, 1.0, "supervised", ha="center", fontsize=8, color=INK_MUTED,
           transform=ax.get_xaxis_transform())
    ax.text(3.75, 1.0, "label-free (no TN5000 labels used)", ha="center", fontsize=8,
           color=INK_MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=8)
    style_ax(ax, "(a) TN5000 adaptation-arm ladder, ResNet-50\n(n=250, class-balanced eval subset)",
            ylabel="ROC-AUC")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_ylim(0.6, 1.0)

    # ---- (b) all label-free arms make things worse than zero-shot ------------ #
    ax = axes[1]
    for j, m in enumerate(("der_mil", "mr_mil")):
        sub = ladder[(ladder.model == m) & (ladder.target_labels == "no")]
        sub = sub.set_index("arm").loc[["zero_shot", "upl", "upl_tent", "retrieval_k5"]]
        ax.plot(range(4), sub["auc"], marker="o", color=MODEL_COLORS[m],
               linewidth=2, markersize=7, label=MODEL_LABELS[m].split(" (")[0])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Zero-shot", "+ Pseudo-\nlabel", "+ TENT", "Retrieval\nbags"], fontsize=9)
    style_ax(ax, "(b) Label-free adaptation arms\n(no TN5000 label ever used)",
            ylabel="ROC-AUC")
    ax.legend(fontsize=9)

    # ---- (c) same-config run-to-run variance, measured directly --------------- #
    ax = axes[2]
    runs = [("Run 1\n(kaggle_jobs/final_out)", 0.913984, MODEL_COLORS["der_mil"]),
           ("Run 2\n(kaggle_jobs/report_out)", ladder.loc[(ladder.model == "der_mil") &
                                                          (ladder.arm == "bbox"), "auc"].item(),
            MODEL_COLORS["der_mil"])]
    x2 = np.arange(2)
    ax.bar(x2, [r[1] for r in runs], color=[r[2] for r in runs], width=0.5, alpha=[1.0, 0.55][0])
    for i, r in enumerate(runs):
        ax.text(i, r[1] + 0.003, "%.4f" % r[1], ha="center", fontsize=10, color=INK)
    ax.set_xticks(x2); ax.set_xticklabels([r[0] for r in runs], fontsize=9)
    spread = abs(runs[1][1] - runs[0][1])
    ax.set_ylim(0.85, 0.98)
    style_ax(ax, "(c) Identical config, two sessions\nDER-MIL, bbox masks, same 250 images",
            ylabel="ROC-AUC")
    ax.text(0.5, 0.06, "spread = %.4f AUC\n(training stochasticity only --\nno code or data changed)"
           % spread, ha="center", fontsize=8.5, color=INK_MUTED, transform=ax.transAxes)

    fig.tight_layout()
    bbox_a = data.load_predictions(data.TN5000_PRED[("der_mil", "bbox")], pcol="p")
    bbox_b = data.load_predictions(data.TN5000_PRED[("mr_mil", "bbox")], pcol="p")
    unet_a = data.load_predictions(data.TN5000_PRED[("der_mil", "unet")], pcol="p")
    unet_b = data.load_predictions(data.TN5000_PRED[("mr_mil", "unet")], pcol="p")
    dl_bbox = data.delong(bbox_a, bbox_b)
    dl_unet = data.delong(unet_a, unet_b)
    caption = (
        "Figure 8. External (cross-domain) validation on TN5000, evaluated on a "
        "class-balanced 250-image subset of the official validation split, "
        "never used for training. (a) Full adaptation-arm ladder for DER-MIL "
        "and MR-MIL: supervised domain adaptation with bounding-box or "
        "U-Net-predicted masks, versus three label-free arms that never see a "
        "TN5000 label. (b) DER-MIL is significantly ahead of MR-MIL on both "
        "supervised arms (paired DeLong on the identical 250 images: bbox "
        "diff=%+.4f p=%.4f, U-Net-mask diff=%+.4f p=%.4f) but every label-free "
        "arm underperforms "
        "plain zero-shot transfer for both models, indicating the pseudo-"
        "labeling and test-time-adaptation arms are not currently beneficial "
        "on this domain shift. (c) The identical bbox-adaptation configuration "
        "run in two independent Kaggle sessions produced ROC-AUC values %.4f "
        "apart, which is training stochasticity alone and is comparable in "
        "magnitude to several of the model-vs-model differences reported in "
        "this project -- external comparisons on this 250-image subset should "
        "be read with that noise floor in mind."
        % (dl_bbox["diff"], dl_bbox["p_value"], dl_unet["diff"], dl_unet["p_value"], spread)
    )
    style.save(fig, "fig08_external_tn5000", caption)


if __name__ == "__main__":
    main()
