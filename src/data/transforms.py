"""Joint image/mask transforms.

Geometry must be applied identically to the frame and its lesion mask, so the
augmentation pipeline is written explicitly rather than composed from two
independent torchvision pipelines.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import DataConfig


# --------------------------------------------------------------------------- #
def _affine(img: np.ndarray, mask: Optional[np.ndarray], angle: float,
            tx: float, ty: float, scale: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx * w
    M[1, 2] += ty * h
    out_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out_mask = None
    if mask is not None:
        out_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out_img, out_mask


class JointTransform:
    """Train-time augmentation shared by every model in the comparison."""

    def __init__(self, cfg: DataConfig, train: bool):
        self.cfg = cfg
        self.train = train

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray],
                 rng: np.random.RandomState) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        c = self.cfg
        if not self.train:
            return img, mask

        if rng.rand() < c.aug_hflip:
            img = img[:, ::-1].copy()
            if mask is not None:
                mask = mask[:, ::-1].copy()

        angle = rng.uniform(-c.aug_rotate_deg, c.aug_rotate_deg)
        tx = rng.uniform(-c.aug_translate, c.aug_translate)
        ty = rng.uniform(-c.aug_translate, c.aug_translate)
        scale = rng.uniform(c.aug_scale[0], c.aug_scale[1])
        if abs(angle) > 1e-3 or abs(tx) > 1e-4 or abs(ty) > 1e-4 or abs(scale - 1) > 1e-4:
            img, mask = _affine(img, mask, angle, tx, ty, scale)

        if c.aug_brightness > 0:
            img = img + rng.uniform(-c.aug_brightness, c.aug_brightness)
        if c.aug_contrast > 0:
            m = float(img.mean())
            img = (img - m) * (1.0 + rng.uniform(-c.aug_contrast, c.aug_contrast)) + m
        return np.clip(img, 0.0, 1.0).astype(np.float32), mask


# --------------------------------------------------------------------------- #
def tta_views(n: int = 8) -> List[Tuple[bool, float, float]]:
    """Deterministic (hflip, rotation_deg, scale) views for test-time augmentation."""
    combos: List[Tuple[bool, float, float]] = []
    for flip in (False, True):
        for ang in (-5.0, 5.0):
            for sc in (1.0, 1.05):
                combos.append((flip, ang, sc))
    return combos[:n] if n <= len(combos) else combos


def apply_tta(img: np.ndarray, mask: Optional[np.ndarray],
              view: Tuple[bool, float, float]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    flip, ang, sc = view
    if flip:
        img = img[:, ::-1].copy()
        if mask is not None:
            mask = mask[:, ::-1].copy()
    if abs(ang) > 1e-3 or abs(sc - 1.0) > 1e-4:
        img, mask = _affine(img, mask, ang, 0.0, 0.0, sc)
    return img, mask


# --------------------------------------------------------------------------- #
def normalize_chw(img_hwc: np.ndarray, mean, std) -> np.ndarray:
    """HxWx3 in [0,1] -> 3xHxW normalised float32."""
    x = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)
    m = np.asarray(mean, np.float32).reshape(3, 1, 1)
    s = np.asarray(std, np.float32).reshape(3, 1, 1)
    return (x - m) / s
