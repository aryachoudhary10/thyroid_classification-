"""Multi-region diagnostic evidence maps and mask perturbations.

Given a binary lesion mask M we derive four anatomically motivated evidence
regions used by DER-MIL:

    core   = erode(M)                 lesion interior / internal texture
    margin = dilate(M) - erode(M)     boundary band, margin irregularity
    peri   = dilate(M, k) - M         perinodular halo / surrounding tissue
    global = 1                        whole frame, broad anatomical context

The same module supplies the mask-degradation operators used by the robustness
experiments (Table 13 of the paper) so training and stress-testing share one
implementation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REGION_ORDER = ("core", "margin", "peri", "global")


# --------------------------------------------------------------------------- #
def _kernel(px: int) -> np.ndarray:
    px = max(int(px), 1)
    if px % 2 == 0:
        px += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


def binarize(mask: np.ndarray, thr: float = 0.5) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    return (m > thr).astype(np.uint8)


def bbox_to_mask(boxes: Sequence[Tuple[int, int, int, int]],
                 out_hw: Tuple[int, int],
                 src_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Rasterise Pascal-VOC boxes into a binary mask at ``out_hw``."""
    h, w = out_hw
    m = np.zeros((h, w), np.uint8)
    if not boxes:
        return m
    sh, sw = src_hw if src_hw else (h, w)
    sy, sx = h / float(max(sh, 1)), w / float(max(sw, 1))
    for (x1, y1, x2, y2) in boxes:
        xa, xb = int(round(x1 * sx)), int(round(x2 * sx))
        ya, yb = int(round(y1 * sy)), int(round(y2 * sy))
        xa, xb = max(0, min(xa, w - 1)), max(0, min(xb, w))
        ya, yb = max(0, min(ya, h - 1)), max(0, min(yb, h))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = 1
    return m


# --------------------------------------------------------------------------- #
def make_regions(mask: np.ndarray,
                 margin_inner_px: int = 6,
                 margin_outer_px: int = 6,
                 peri_px: int = 22,
                 names: Sequence[str] = REGION_ORDER,
                 fallback_full: bool = True) -> np.ndarray:
    """Return a float32 array of shape (K, H, W) with values in [0, 1].

    An empty mask degrades gracefully: lesion-derived regions become all-zero
    (or the full frame if ``fallback_full``), which is exactly the behaviour the
    "complete mask removal" robustness condition needs.
    """
    m = binarize(mask)
    h, w = m.shape[:2]
    empty = m.sum() == 0

    if empty:
        core = np.zeros_like(m)
        margin = np.zeros_like(m)
        peri = np.zeros_like(m)
        if fallback_full:
            # No lesion evidence at all -- keep the maps zero so downstream
            # pooling yields the learned "no evidence" embedding.
            pass
    else:
        eroded = cv2.erode(m, _kernel(margin_inner_px), iterations=1)
        dilated = cv2.dilate(m, _kernel(margin_outer_px), iterations=1)
        core = eroded if eroded.sum() > 0 else m
        margin = np.clip(dilated.astype(np.int16) - eroded.astype(np.int16), 0, 1).astype(np.uint8)
        peri_out = cv2.dilate(m, _kernel(peri_px), iterations=1)
        peri = np.clip(peri_out.astype(np.int16) - dilated.astype(np.int16), 0, 1).astype(np.uint8)
        if margin.sum() == 0:
            margin = m
        if peri.sum() == 0:
            peri = np.clip(cv2.dilate(m, _kernel(peri_px)) - m, 0, 1).astype(np.uint8)

    table = {
        "core": core,
        "margin": margin,
        "peri": peri,
        "global": np.ones((h, w), np.uint8),
        "lesion": m,
    }
    return np.stack([table[n].astype(np.float32) for n in names], 0)


def resize_regions(regions: np.ndarray, size: int) -> np.ndarray:
    """Area-resize (K, H, W) region maps -- fractional coverage is preserved."""
    if regions.shape[-1] == size and regions.shape[-2] == size:
        return regions
    out = np.zeros((regions.shape[0], size, size), np.float32)
    for i in range(regions.shape[0]):
        out[i] = cv2.resize(regions[i], (size, size), interpolation=cv2.INTER_AREA)
    return out


# --------------------------------------------------------------------------- #
# Mask degradation (paper Table 13) + extensions used by the reliability tests
# --------------------------------------------------------------------------- #
MASK_CONDITIONS: Dict[str, Dict] = {
    "clean":        {"op": "none"},
    "dilate5":      {"op": "dilate", "px": 5, "iters": 1},
    "erode5":       {"op": "erode", "px": 5, "iters": 1},
    "erode15":      {"op": "erode", "px": 15, "iters": 2},
    "dilate15":     {"op": "dilate", "px": 15, "iters": 2},
    "zeros":        {"op": "zeros"},
    "shift10":      {"op": "shift", "px": 10},
    "boxify":       {"op": "boxify"},
}


def degrade_mask(mask: np.ndarray, condition: str) -> np.ndarray:
    spec = MASK_CONDITIONS.get(condition)
    if spec is None:
        raise KeyError("unknown mask condition: " + str(condition))
    m = binarize(mask)
    op = spec["op"]
    if op == "none":
        return m
    if op == "zeros":
        return np.zeros_like(m)
    if op == "dilate":
        return cv2.dilate(m, _kernel(spec["px"]), iterations=spec.get("iters", 1))
    if op == "erode":
        return cv2.erode(m, _kernel(spec["px"]), iterations=spec.get("iters", 1))
    if op == "shift":
        d = int(spec["px"])
        M = np.float32([[1, 0, d], [0, 1, d]])
        return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)
    if op == "boxify":
        ys, xs = np.nonzero(m)
        out = np.zeros_like(m)
        if len(ys):
            out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
        return out
    raise KeyError(op)


# --------------------------------------------------------------------------- #
# Frame corruption (Section 24 of the spec: artificial low-quality views)
# --------------------------------------------------------------------------- #
CORRUPTIONS = ("blur", "noise", "contrast", "occlusion", "downres")


def corrupt_frame(img: np.ndarray, kind: str, severity: float = 1.0,
                  rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """``img`` is HxWx3 float32 in [0, 1]. Returns a corrupted copy."""
    rng = rng or np.random.RandomState(0)
    x = img.copy()
    s = float(np.clip(severity, 0.0, 1.0))
    if kind == "blur":
        k = int(3 + round(12 * s)) | 1
        x = cv2.GaussianBlur(x, (k, k), 0)
    elif kind == "noise":
        x = x + rng.normal(0, 0.02 + 0.18 * s, x.shape).astype(np.float32)
    elif kind == "contrast":
        mean = x.mean()
        x = (x - mean) * (1.0 - 0.85 * s) + mean
    elif kind == "occlusion":
        h, w = x.shape[:2]
        bh, bw = int(h * (0.15 + 0.35 * s)), int(w * (0.15 + 0.35 * s))
        y0 = rng.randint(0, max(h - bh, 1))
        x0 = rng.randint(0, max(w - bw, 1))
        x[y0:y0 + bh, x0:x0 + bw] = float(x.mean())
    elif kind == "downres":
        h, w = x.shape[:2]
        f = max(1, int(2 + round(6 * s)))
        small = cv2.resize(x, (max(w // f, 2), max(h // f, 2)), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        raise KeyError("unknown corruption: " + str(kind))
    return np.clip(x, 0.0, 1.0).astype(np.float32)
