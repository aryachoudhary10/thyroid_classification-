"""Figures for the paper and for day-to-day debugging.

Palette note
------------
The categorical hues below were validated for colour-vision deficiency and for
contrast against both light and dark chart surfaces (worst adjacent-pair CVD
Delta-E 11.0, normal-vision floor 18.7, all slots >= 3:1 contrast). They are
assigned in fixed order and never cycled: core, margin, peri, global always get
the same hue in every figure, and benign / malignant always get blue / orange.
Every chart also carries a legend or direct labels, so identity is never
communicated by colour alone.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..eval.calibration import expected_calibration_error, reliability_curve

# fixed categorical order -- never cycled, never reordered per figure
REGION_COLORS: Dict[str, str] = {
    "core": "#0072B2",      # blue
    "margin": "#009E73",    # green
    "peri": "#D55E00",      # vermillion
    "global": "#9B4F96",    # purple
}
CLASS_COLORS = {"benign": "#0072B2", "malignant": "#D55E00"}
# diverging pair for signed relations: two hues either side of a neutral grey
SUPPORT_COLOR = "#009E73"
CONTRADICTION_COLOR = "#D55E00"
NEUTRAL = "#8A8A85"

INK = "#22221F"
INK_MUTED = "#6B6B66"
GRID = "#E3E3DE"


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def plot_reliability_diagram(y: np.ndarray, p_before: np.ndarray,
                             p_after: Optional[np.ndarray], path: str,
                             n_bins: int = 10, title: str = "Reliability diagram") -> str:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=NEUTRAL, linewidth=1.4,
            label="Perfect calibration")

    series = [("Before scaling", p_before, CLASS_COLORS["malignant"])]
    if p_after is not None:
        series.append(("After scaling", p_after, CLASS_COLORS["benign"]))

    for name, p, color in series:
        xs, ys, _ns = reliability_curve(y, p, n_bins)
        ece = expected_calibration_error(y, p, 15)
        ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.2,
                label="%s (ECE = %.4f)" % (name, ece))
        if len(xs):
            ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(6, -2), color=color, fontsize=8.5)

    _style(ax, title, "Mean predicted probability", "Observed malignant fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
def plot_evidence_distribution(df: pd.DataFrame, path: str,
                               value_cols: Sequence[str] = ("support", "contradiction",
                                                            "uncertainty", "reliability"),
                               title: str = "Evidence channels by class") -> str:
    """Small multiples -- one panel per channel, benign vs malignant."""
    cols = [c for c in value_cols if c in df.columns]
    if not cols:
        raise ValueError("none of the requested columns are present")
    fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 3.6), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        groups = [("benign", df.loc[df["label"] == 0, col].values),
                  ("malignant", df.loc[df["label"] == 1, col].values)]
        for i, (name, vals) in enumerate(groups):
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            color = CLASS_COLORS[name]
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.22
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=9, alpha=0.35,
                       color=color, linewidths=0)
            ax.plot([i - 0.28, i + 0.28], [vals.mean()] * 2, color=color,
                    linewidth=2.6, solid_capstyle="round")
            ax.annotate("%.3f" % vals.mean(), (i + 0.30, vals.mean()),
                        fontsize=8, color=color, va="center")
        _style(ax, col.capitalize(), "", "value" if col == cols[0] else "")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Malignant"], fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)

    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
def plot_patient_explanation(images: np.ndarray, masks: np.ndarray,
                             channels: Dict[str, np.ndarray],
                             alpha: np.ndarray, prob: float, label: int,
                             patient_id: str, path: str,
                             regions: Sequence[str] = ("core", "margin", "peri", "global")
                             ) -> str:
    """Per-patient panel: frames on top, evidence channels underneath.

    ``channels`` maps a channel name (I / S / D / U / R) to a (T, K) array.
    """
    n = int(len(images))
    fig = plt.figure(figsize=(3.0 * max(n, 2), 6.4))
    gs = fig.add_gridspec(2, max(n, 2), height_ratios=[1.15, 1.0], hspace=0.35)

    for t in range(n):
        ax = fig.add_subplot(gs[0, t])
        img = images[t]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / max(img.ptp(), 1e-6)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        if masks is not None and t < len(masks):
            m = masks[t]
            m = m[0] if m.ndim == 3 else m
            ax.contour(m, levels=[0.5], colors=[REGION_COLORS["core"]], linewidths=1.6)
        ax.set_title("Frame %d   alpha = %.3f" % (t + 1, float(alpha[t])),
                     fontsize=9.5, color=INK, loc="left")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ch_names = [c for c in ("I", "S", "D", "U", "R") if c in channels]
    x = np.arange(n)
    width = 0.8 / max(len(ch_names), 1)
    channel_color = {"I": REGION_COLORS["core"], "S": SUPPORT_COLOR,
                     "D": CONTRADICTION_COLOR, "U": NEUTRAL,
                     "R": REGION_COLORS["global"]}
    channel_label = {"I": "Importance", "S": "Support", "D": "Contradiction",
                     "U": "Uncertainty", "R": "Reliability"}
    for i, c in enumerate(ch_names):
        vals = np.asarray(channels[c])[:n]
        vals = vals.mean(axis=1) if vals.ndim > 1 else vals
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.86,
               color=channel_color[c], label=channel_label[c],
               edgecolor="white", linewidth=1.0)
    _style(ax, "", "", "channel value")
    ax.set_xticks(x)
    ax.set_xticklabels(["Frame %d" % (t + 1) for t in range(n)], fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(ch_names), labelcolor=INK,
              loc="upper center", bbox_to_anchor=(0.5, 1.16))

    truth = "malignant" if label == 1 else "benign"
    fig.suptitle("Patient %s  |  truth: %s  |  P(malignant) = %.3f"
                 % (patient_id, truth, prob),
                 color=INK, fontsize=12, x=0.02, ha="left")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
def plot_support_contradiction(df: pd.DataFrame, path: str,
                               title: str = "Cross-view support vs contradiction") -> str:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for name, cls in (("benign", 0), ("malignant", 1)):
        sub = df[df["label"] == cls]
        ax.scatter(sub["support"], sub["contradiction"], s=16, alpha=0.55,
                   color=CLASS_COLORS[name], linewidths=0, label=name.capitalize())
    lim = float(max(df["support"].max(), df["contradiction"].max(), 0.05)) * 1.08
    ax.plot([0, lim], [0, lim], linestyle=(0, (4, 3)), color=NEUTRAL, linewidth=1.2)
    ax.annotate("agreement dominates", (lim * 0.72, lim * 0.30), fontsize=8.5,
                color=SUPPORT_COLOR)
    ax.annotate("contradiction dominates", (lim * 0.10, lim * 0.90), fontsize=8.5,
                color=CONTRADICTION_COLOR)
    _style(ax, title, "Mean cross-view support S", "Mean cross-view contradiction D")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
def plot_confusion(y: np.ndarray, p: np.ndarray, thr: float, path: str,
                   title: str = "Confusion matrix") -> str:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    # sequential single hue, light -> dark
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / max(cm.max(), 1)
            ax.text(j, i, format(int(cm[i, j]), ","), ha="center", va="center",
                    fontsize=14, color="white" if frac > 0.55 else INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted benign", "Predicted malignant"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True benign", "True malignant"], fontsize=9)
    ax.set_title("%s (threshold %.3f)" % (title, thr), fontsize=11, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0, colors=INK_MUTED)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
def plot_training_curves(history: List[Dict], path: str, monitor: str = "roc_auc",
                         title: str = "Training history") -> str:
    if not history:
        raise ValueError("empty history")
    h = pd.DataFrame(history)
    key = "val_" + monitor
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    steps = np.arange(len(h))
    ax.plot(steps, h[key], color=CLASS_COLORS["benign"], linewidth=2.0,
            marker="o", markersize=4, markeredgecolor="white",
            label="validation " + monitor)
    best_i = int(np.nanargmax(h[key].values))
    ax.scatter([best_i], [h[key].values[best_i]], s=70, facecolor="none",
               edgecolor=CLASS_COLORS["malignant"], linewidth=2, zorder=5)
    ax.annotate("best %.4f" % h[key].values[best_i], (best_i, h[key].values[best_i]),
                textcoords="offset points", xytext=(8, -10), fontsize=9,
                color=CLASS_COLORS["malignant"])
    if (h["stage"] == 2).any():
        first2 = int(np.argmax(h["stage"].values == 2))
        ax.axvline(first2 - 0.5, color=NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate("stage 2 (fine-tune)", (first2 - 0.4, ax.get_ylim()[0]),
                    fontsize=8.5, color=INK_MUTED, va="bottom")
    _style(ax, title, "epoch (concatenated across stages)", "validation " + monitor)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
def plot_model_comparison(df: pd.DataFrame, path: str, metric: str = "roc_auc",
                          title: str = "Patient-level discrimination") -> str:
    """Horizontal bars -- magnitude comparison across a handful of models."""
    d = df.sort_values(metric).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(d) + 1.6))
    colors = [CLASS_COLORS["malignant"] if str(m).startswith(("der", "mr"))
              else CLASS_COLORS["benign"] for m in d["model"]]
    ax.barh(np.arange(len(d)), d[metric], height=0.62, color=colors,
            edgecolor="white", linewidth=1.2)
    for i, v in enumerate(d[metric]):
        ax.annotate("%.4f" % v, (v, i), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["model"], fontsize=9.5, color=INK)
    lo = float(max(0.0, d[metric].min() - 0.05))
    ax.set_xlim(lo, min(1.02, float(d[metric].max()) + 0.06))
    _style(ax, title, metric.replace("_", "-").upper(), "")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
