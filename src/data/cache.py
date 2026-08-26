"""One-time image cache at the training resolution.

ThyroidXL ships 532x727 PNGs. The dataset resizes every frame to 224 on the CPU
at load time, which the calibration run showed costs roughly 60% of wall-clock:
~82 s per epoch against ~33 s of GPU work. That decode is repeated for every
epoch, every fold and every model -- the same pixels, thousands of times.

Caching the resized frames once removes that cost for the entire study. It is
scientifically inert: the pipeline already resizes to 224 with INTER_LINEAR
(images) and INTER_NEAREST (masks), so a cached frame is the identical array
the loader would have produced.

The cache is resumable (existing files are skipped), verified by count, and
lives wherever you point it -- on Kaggle that is /kaggle/working, so it is
carried between chained sessions along with the checkpoints.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ..utils.common import human_bytes, log

cv2.setNumThreads(0)


# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str, kind: str, src: str) -> str:
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(cache_dir, kind, stem + ".png")


def _resize_one(args) -> Tuple[bool, Optional[str]]:
    src, dst, size, is_mask = args
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True, None
    try:
        if is_mask:
            a = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if a is None:
                return False, src
            a = cv2.resize(a, (size, size), interpolation=cv2.INTER_NEAREST)
            # keep it strictly binary so downstream morphology is unambiguous
            a = ((a > 127).astype(np.uint8)) * 255
        else:
            a = cv2.imread(src, cv2.IMREAD_COLOR)
            if a is None:
                return False, src
            a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
        tmp = dst + ".tmp.png"
        ok = cv2.imwrite(tmp, a)
        if not ok:
            return False, src
        os.replace(tmp, dst)
        return True, None
    except Exception:                                          # noqa: BLE001
        return False, src


# --------------------------------------------------------------------------- #
def build_resize_cache(manifest: pd.DataFrame, cache_dir: str, size: int = 224,
                       workers: int = 8) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Materialise every frame and mask at ``size``; return a rewritten manifest.

    Frames whose mask is missing keep ``None`` so the dataset's own fallback
    still applies.
    """
    os.makedirs(os.path.join(cache_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "msk"), exist_ok=True)

    jobs: List[Tuple[str, str, int, bool]] = []
    new_images: List[List[str]] = []
    new_masks: List[List[Optional[str]]] = []

    for _i, row in manifest.iterrows():
        imgs, msks = [], []
        for src in row["image_paths"]:
            dst = _cache_path(cache_dir, "img", src)
            jobs.append((src, dst, size, False))
            imgs.append(dst)
        for src in row["mask_paths"]:
            if isinstance(src, str) and src:
                dst = _cache_path(cache_dir, "msk", src)
                jobs.append((src, dst, size, True))
                msks.append(dst)
            else:
                msks.append(None)
        new_images.append(imgs)
        new_masks.append(msks)

    todo = [j for j in jobs if not (os.path.exists(j[1]) and os.path.getsize(j[1]) > 0)]
    log("resize cache: %d files total, %d already cached, %d to build"
        % (len(jobs), len(jobs) - len(todo), len(todo)))

    failures: List[str] = []
    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ok, src in ex.map(_resize_one, todo, chunksize=32):
                done += 1
                if not ok:
                    failures.append(src)
                if done % 2000 == 0:
                    log("  cached %d/%d" % (done, len(todo)))

    missing = [j[1] for j in jobs
               if not (os.path.exists(j[1]) and os.path.getsize(j[1]) > 0)]
    if missing:
        raise RuntimeError(
            "resize cache incomplete: %d of %d files missing (e.g. %s). "
            "Refusing to train on a partial cache."
            % (len(missing), len(jobs), missing[:3]))

    out = manifest.copy()
    out["image_paths"] = new_images
    out["mask_paths"] = new_masks

    total = sum(os.path.getsize(j[1]) for j in jobs)
    stats = {"files": len(jobs), "built": len(todo), "failed": len(failures),
             "bytes": total}
    log("resize cache: ready -- %d files, %s on disk%s"
        % (len(jobs), human_bytes(total),
           (", %d read failures" % len(failures)) if failures else ""))
    return out, stats


# --------------------------------------------------------------------------- #
def verify_cache(manifest: pd.DataFrame, sample: int = 40,
                 size: int = 224) -> Dict[str, object]:
    """Spot-check that cached frames really are ``size`` and masks stayed binary."""
    rng = np.random.RandomState(0)
    rows = manifest.sample(min(sample, len(manifest)), random_state=0)
    bad_shape, bad_mask, checked = [], [], 0
    for _i, row in rows.iterrows():
        for p in row["image_paths"][:1]:
            a = cv2.imread(p, cv2.IMREAD_COLOR)
            checked += 1
            if a is None or a.shape[0] != size or a.shape[1] != size:
                bad_shape.append(p)
        for p in row["mask_paths"][:1]:
            if not isinstance(p, str):
                continue
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape[0] != size:
                bad_shape.append(p)
            elif not set(np.unique(m).tolist()) <= {0, 255}:
                bad_mask.append(p)
    res = {"checked": checked, "bad_shape": len(bad_shape), "bad_mask": len(bad_mask)}
    log("cache verify: %s" % res)
    if bad_shape or bad_mask:
        raise RuntimeError("cache verification failed: " + str(res))
    return res
