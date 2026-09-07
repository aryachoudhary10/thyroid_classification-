"""Figure 10 -- pipeline / architecture schematic.

This is a diagram of the ACTUAL implemented pipeline (src/models/der_mil.py,
reliability.py, backbone.py, attn_mil.py), not a rendering of any measured
data. Box labels, stage order, and the four evidence regions are read directly
from the module structure; no numeric values are shown here.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import style
from style import INK, INK_MUTED, REGION_COLORS


def box(ax, x, y, w, h, text, color, fontsize=9, textcolor=None):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.2, edgecolor=INK, facecolor=color)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
           color=textcolor or INK, wrap=True)
    return r


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
               arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, linewidth=1.4,
                              shrinkA=2, shrinkB=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 8.2))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # Input
    box(ax, 0.3, 6.7, 2.0, 1.0, "Ultrasound\nframe x[t]\n+ lesion mask m[t]", "#EDEDEA")
    arrow(ax, 2.3, 7.2, 3.1, 7.2)

    # Region construction
    box(ax, 3.1, 6.7, 2.1, 1.0, "4 evidence regions\ncore / margin\nperi / global", "#EDEDEA")
    arrow(ax, 5.2, 7.2, 6.0, 7.2)

    # Shared backbone
    box(ax, 6.0, 6.7, 2.1, 1.0, "Shared CNN backbone\n(ConvNeXt-Tiny /\nswappable)", "#DDE7F0")
    arrow(ax, 8.1, 7.2, 8.9, 7.2)

    box(ax, 8.9, 6.7, 2.4, 1.0, "Evidence embeddings\ne[t,k], k in\n{core,margin,peri,global}", "#EDEDEA")

    arrow(ax, 10.1, 6.7, 10.1, 6.0)

    # Reliability head, expanded
    ry = 3.8
    box(ax, 6.7, ry + 1.55, 6.9, 0.5, "Reliability head   R = f(I, S, D, U)", "#F5EDF5",
       fontsize=10)
    box(ax, 6.8, ry + 0.85, 1.45, 0.55, "Importance I\n(MLP on e)",
       REGION_COLORS["core"], 8, "white")
    box(ax, 8.4, ry + 0.85, 1.65, 0.55, "Support S\n(cross-frame,\nsame region)",
       REGION_COLORS["margin"], 8, "white")
    box(ax, 10.2, ry + 0.85, 1.65, 0.55, "Contradiction D\n(cross-frame,\nsame region)",
       REGION_COLORS["peri"], 8, "white")
    box(ax, 12.0, ry + 0.85, 1.5, 0.55, "Uncertainty U\n(learned,\nimplicit)",
       REGION_COLORS["global"], 8, "white")
    for x0 in (7.5, 9.2, 11.0, 12.75):
        arrow(ax, x0, ry + 0.85, x0, ry + 0.55)
    box(ax, 8.7, ry - 0.25 + 0.55, 3.0, 0.5,
       "R[t,k] in (0,1) per (frame, region) token", "#F5EDF5", fontsize=8.5)

    arrow(ax, 10.1, ry + 0.55, 10.1, ry - 0.2)

    # Region-weighted fusion
    box(ax, 8.5, ry - 1.05, 3.2, 0.75,
       "Region-weighted fusion\nsoftmax(I + log R) over k -> h[t]", "#DDE7F0", fontsize=8.5)
    arrow(ax, 10.1, ry - 1.05, 10.1, ry - 1.55)

    # AttnMIL
    box(ax, 8.7, ry - 2.35, 2.8, 0.75,
       "Attention-based MIL\nalpha[t] biased by mean R[t]", "#DDE7F0", fontsize=8.5)
    arrow(ax, 10.1, ry - 2.35, 10.1, ry - 2.9)

    box(ax, 9.0, ry - 3.6, 2.2, 0.65, "Patient embedding\n-> P(malignant)", "#EDEDEA", fontsize=8.5)

    # Frame bag. Routed as an explicit L-shaped connector (not a single arc)
    # so it is guaranteed not to cross any other box: it uses the actual clear
    # gap between the top of the reliability-head bar (ry+2.05 = 5.85) and the
    # bottom of the backbone/embeddings row (y=6.7), and a diagonal final leg
    # that stays right of the backbone box's right edge (x=8.1) for every y in
    # that gap, entering the embeddings box at a point distinct from the
    # backbone->embeddings arrow above it.
    box(ax, 0.4, 5.95, 2.4, 0.6, "Frame 1 .. T\n(median 3, max 10\nframes per patient)",
       "#EDEDEA", 8.5)
    ax.plot([2.8, 8.5], [6.25, 6.25], color=INK_MUTED, linewidth=1.4, zorder=1)
    arrow(ax, 8.5, 6.25, 8.9, 6.85)
    ax.text(4.0, 6.5, "each frame -> 4 region tokens, looped over T",
           fontsize=8, color=INK_MUTED, ha="center")

    # MR-MIL note
    box(ax, 0.4, 3.9, 3.6, 1.3,
       "MR-MIL ablation: identical pipeline\nwith the reliability head disabled\n"
       "(R forced uniform) -- isolates the\ncontribution of multi-region evidence\n"
       "alone from the reliability mechanism.", "#F2F2EF", fontsize=8.3)

    ax.set_title("DER-MIL patient-level pipeline (schematic of the implemented architecture)",
                fontsize=12, color=INK, loc="left")

    caption = (
        "Figure 10. Schematic of the implemented DER-MIL pipeline "
        "(src/models/der_mil.py, reliability.py, backbone.py, attn_mil.py). "
        "Each ultrasound frame is decomposed into four evidence regions "
        "(core, margin, peri, global) that share one CNN backbone. The "
        "reliability head computes an importance score I from the evidence "
        "embedding, a support score S and contradiction score D by comparing "
        "the same region across a patient's other frames, and an uncertainty "
        "score U learned implicitly through its effect on the final "
        "prediction; these combine into a reliability score R per (frame, "
        "region) token. R weights the four regions when fusing them into a "
        "per-frame embedding, and separately biases the attention-based "
        "multiple-instance-learning aggregation across frames into one "
        "patient-level malignancy probability. The MR-MIL ablation used "
        "throughout this project is the identical pipeline with the "
        "reliability head disabled. This is a structural diagram of the code, "
        "not a rendering of measured values."
    )
    style.save(fig, "fig10_architecture_schematic", caption)


if __name__ == "__main__":
    main()
