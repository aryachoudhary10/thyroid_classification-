"""Shared publication style for every figure in paper_figures/.

Palette is inherited unchanged from src/viz/explain.py so any figure produced
there and any figure produced here read as one system. Colours are validated
for colour-vision deficiency and are assigned in FIXED order, never cycled:
core/margin/peri/global always get the same hue, benign/malignant always get
the same hue, model identities (RCAF / lesion-only / MR-MIL / DER-MIL) always
get the same hue across every figure in the set.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --------------------------------------------------------------------------- #
# Palette (identical values to src/viz/explain.py -- do not diverge)
REGION_COLORS = {
    "core": "#0072B2",      # blue
    "margin": "#009E73",    # green
    "peri": "#D55E00",      # vermillion
    "global": "#9B4F96",    # purple
}
CLASS_COLORS = {"benign": "#0072B2", "malignant": "#D55E00"}
SUPPORT_COLOR = "#009E73"
CONTRADICTION_COLOR = "#D55E00"
NEUTRAL = "#8A8A85"
INK = "#22221F"
INK_MUTED = "#6B6B66"
GRID = "#E3E3DE"

# Fixed model identity colours, used in every comparison figure across the set.
MODEL_COLORS = {
    "rcaf": "#8A8A85",          # neutral grey -- published baseline
    "image_mil": "#56B4E9",     # sky blue
    "lesion_mil": "#E69F00",    # orange
    "mr_mil": "#009E73",        # green -- reliability-off ablation
    "der_mil": "#D55E00",       # vermillion -- proposed model
    "der_mil_linear": "#CC79A7",  # pink -- constrained-fusion probe
}
MODEL_LABELS = {
    "rcaf": "RCAF (published baseline)",
    "image_mil": "Image-only + AttnMIL",
    "lesion_mil": "Lesion-only + AttnMIL",
    "mr_mil": "MR-MIL (multi-region, no reliability)",
    "der_mil": "DER-MIL (proposed)",
    "der_mil_linear": "DER-MIL (linear fusion)",
}

# Output root: PAPER_FIG_OUTDIR lets a script in a subfolder (e.g.
# resnet50_supplementary/) redirect its own figures without touching the
# primary output/ directory. Set it before importing this module.
FIGDIR = os.environ.get(
    "PAPER_FIG_OUTDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
PNG_DIR = os.path.join(FIGDIR, "png")
PDF_DIR = os.path.join(FIGDIR, "pdf")
SVG_DIR = os.path.join(FIGDIR, "svg")
CAP_DIR = os.path.join(FIGDIR, "captions")
for _d in (PNG_DIR, PDF_DIR, SVG_DIR, CAP_DIR):
    os.makedirs(_d, exist_ok=True)


def apply_rcparams() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "none",
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,      # embed as real text, not paths, in the PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


apply_rcparams()


def style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)


def save(fig, name: str, caption: Optional[str] = None, dpi: int = 300) -> str:
    """Save PNG (raster, dpi) + PDF + SVG (vector) under output/, plus a caption."""
    png = os.path.join(PNG_DIR, name + ".png")
    pdf = os.path.join(PDF_DIR, name + ".pdf")
    svg = os.path.join(SVG_DIR, name + ".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if caption is not None:
        with open(os.path.join(CAP_DIR, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(caption.strip() + "\n")
    print("  wrote %-32s -> %s / .pdf / .svg" % (name, png))
    return png
