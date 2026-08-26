"""Patient-level discrimination and operating-point metrics.

Includes a DeLong test for correlated ROC curves (the paper uses it for the
head-to-head AUC comparison) and stratified bootstrap confidence intervals.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)


# --------------------------------------------------------------------------- #
def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return {"threshold": float(thr), "sensitivity": float(sens),
            "specificity": float(spec), "precision": float(prec), "f1": float(f1),
            "accuracy": float((tp + tn) / max(len(y), 1)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def all_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out: Dict[str, float] = {}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out.update(threshold_metrics(y, p, thr))
    out["brier"] = float(np.mean((p - y) ** 2))
    out["n"] = int(len(y))
    out["prevalence"] = float(y.mean()) if len(y) else float("nan")
    return out


# --------------------------------------------------------------------------- #
def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def sensitivity_constrained_threshold(y: np.ndarray, p: np.ndarray,
                                      target: float = 0.90) -> float:
    """Largest threshold whose sensitivity still meets ``target``."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return float(np.min(p))
    # roc_curve returns thresholds in decreasing order; pick the strictest that works
    return float(thr[ok[np.argmax(thr[ok])]] if len(ok) else 0.5)


# --------------------------------------------------------------------------- #
def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000,
                 thr: float = 0.5, seed: int = 0,
                 keys: Sequence[str] = ("roc_auc", "pr_auc", "sensitivity",
                                        "specificity", "precision", "f1"),
                 stratified: bool = True) -> Dict[str, Tuple[float, float, float]]:
    """Return {metric: (point, lo, hi)} with 95% percentile intervals."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    point = all_metrics(y, p, thr)
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    samples: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        if stratified and len(idx_pos) and len(idx_neg):
            bi = np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                                 rng.choice(idx_neg, len(idx_neg), replace=True)])
        else:
            bi = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[bi])) < 2:
            continue
        m = all_metrics(y[bi], p[bi], thr)
        for k in keys:
            samples[k].append(m[k])

    out: Dict[str, Tuple[float, float, float]] = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            out[k] = (point.get(k, float("nan")), float("nan"), float("nan"))
        else:
            out[k] = (point[k], float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5)))
    return out


def format_ci(ci: Dict[str, Tuple[float, float, float]], digits: int = 3) -> Dict[str, str]:
    f = "%%.%df [%%.%df, %%.%df]" % (digits, digits, digits)
    return {k: (f % v) for k, v in ci.items()}


# --------------------------------------------------------------------------- #
# DeLong test for two correlated ROC curves
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and z[j + 1] == z[i]:
            j += 1
        t[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(preds: np.ndarray, n_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    m, n = n_pos, preds.shape[1] - n_pos
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = preds.shape[0]

    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])

    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    s = s01 / m + s10 / n
    return auc, np.atleast_2d(s)


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
    """Two-sided DeLong test for AUC(p1) vs AUC(p2) on the same cohort."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    y_s = y[order]
    n_pos = int(y_s.sum())
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    auc, s = _fast_delong(preds, n_pos)
    diff = auc[0] - auc[1]
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return {"auc1": float(auc[0]), "auc2": float(auc[1]), "diff": float(diff),
                "z": float("nan"), "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return {"auc1": float(auc[0]), "auc2": float(auc[1]), "diff": float(diff),
            "z": float(z), "p_value": float(p)}


# --------------------------------------------------------------------------- #
def bootstrap_auc_difference(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                             n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        bi = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[bi])) < 2:
            continue
        diffs.append(roc_auc_score(y[bi], np.asarray(p1)[bi])
                     - roc_auc_score(y[bi], np.asarray(p2)[bi]))
    d = np.asarray(diffs)
    return {"mean_diff": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p_gt_0": float((d > 0).mean())}


def paired_wilcoxon(fold_a: Sequence[float], fold_b: Sequence[float]) -> Dict[str, float]:
    """One-sided paired Wilcoxon signed-rank over per-fold scores."""
    a, b = np.asarray(fold_a, float), np.asarray(fold_b, float)
    if len(a) != len(b) or len(a) < 3 or np.allclose(a, b):
        return {"W": float("nan"), "p_value": float("nan"),
                "cohens_d": float("nan"), "wins": float("nan")}
    try:
        res = stats.wilcoxon(a, b, alternative="greater")
        W, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        W, p = float("nan"), float("nan")
    d = a - b
    cohen = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf")
    return {"W": W, "p_value": p, "cohens_d": cohen, "wins": float((a > b).mean())}
