"""Probability calibration, fitted development-only and frozen before test.

Temperature scaling (Eq. 10) sharpens or softens probabilities without changing
the ranking. It does NOT correct a prior shift, so the one-step prevalence
correction of Eq. (11) is provided separately and reported as its own row.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    max_iter: int = 300, lr: float = 0.02) -> float:
    """Optimise a single scalar T on out-of-fold development predictions."""
    z = torch.tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels, dtype=np.float64), dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.05, 20.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / t, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:                                          # noqa: BLE001
        return 1.0
    return float(torch.exp(log_t).clamp(0.05, 20.0).item())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / max(T, 1e-6)))


def prior_shift_logits(logits: np.ndarray, T: float, p_dev: float,
                       p_test: float) -> np.ndarray:
    """Eq. (11): z~ = z/T + logit(p_test) - logit(p_dev)."""
    z = np.asarray(logits, dtype=float) / max(T, 1e-6)
    adj = np.log(p_test / (1 - p_test)) - np.log(p_dev / (1 - p_dev))
    return z + adj


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# --------------------------------------------------------------------------- #
def expected_calibration_error(y: np.ndarray, p: np.ndarray,
                               n_bins: int = 15) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    out = {
        "ece": expected_calibration_error(y, p, n_bins),
        "brier": float(np.mean((p - y) ** 2)),
    }
    for cls, name in ((0, "benign"), (1, "malignant")):
        m = y == cls
        out["ece_" + name] = (expected_calibration_error(y[m], p[m], n_bins)
                              if m.sum() > 0 else float("nan"))
    return out


def reliability_curve(y: np.ndarray, p: np.ndarray,
                      n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean predicted probability, empirical accuracy, bin counts)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ns.append(int(m.sum()))
    return np.asarray(xs), np.asarray(ys), np.asarray(ns)


# --------------------------------------------------------------------------- #
class CalibrationBundle:
    """Everything fitted on development data and frozen before the test set."""

    def __init__(self, temperature: float, thresholds: Dict[str, float],
                 p_dev: float):
        self.temperature = float(temperature)
        self.thresholds = dict(thresholds)
        self.p_dev = float(p_dev)

    def to_dict(self) -> Dict[str, object]:
        return {"temperature": self.temperature, "thresholds": self.thresholds,
                "p_dev": self.p_dev}

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "CalibrationBundle":
        return CalibrationBundle(float(d["temperature"]),          # type: ignore[arg-type]
                                 dict(d["thresholds"]),            # type: ignore[arg-type]
                                 float(d["p_dev"]))                # type: ignore[arg-type]

    def probabilities(self, logits: np.ndarray, p_test: Optional[float] = None,
                      apply_prior_shift: bool = False) -> np.ndarray:
        if apply_prior_shift and p_test is not None:
            return sigmoid(prior_shift_logits(logits, self.temperature,
                                              self.p_dev, p_test))
        return apply_temperature(logits, self.temperature)
