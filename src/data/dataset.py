"""Patient-bag dataset.

One item == one patient == a padded tensor of Tmax frames plus a validity mask.
Padded slots carry zeros and are excluded everywhere downstream (attention,
support, contradiction, uncertainty, pooling, losses).

The same dataset object serves training, evaluation and every robustness
experiment: interventions (mask degradation, frame corruption, within-patient
mask permutation, evidence suppression, frame dropout, TTA views) are applied
inside ``__getitem__`` and are configured through the ``Intervention``
dataclass, so nothing about the model or the loop has to change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from ..utils.common import log
from .discovery import parse_voc_bbox
from .regions import (bbox_to_mask, binarize, corrupt_frame, degrade_mask,
                      make_regions, resize_regions)
from .transforms import JointTransform, apply_tta, normalize_chw, tta_views

cv2.setNumThreads(0)          # avoid thread thrash inside DataLoader workers


# --------------------------------------------------------------------------- #
@dataclass
class Intervention:
    """Declarative description of a robustness / counterfactual perturbation."""
    mask_condition: str = "clean"          # see regions.MASK_CONDITIONS
    permute_masks: bool = False            # shuffle masks across a patient's frames
    corrupt_kind: Optional[str] = None     # blur | noise | contrast | occlusion | downres
    corrupt_severity: float = 1.0
    corrupt_n_frames: int = 1              # how many frames of the bag to corrupt
    corrupt_frame_idx: Optional[int] = None
    drop_frame_idx: Optional[int] = None   # remove one frame from the bag
    keep_frames: Optional[Sequence[int]] = None
    suppress_regions: Tuple[str, ...] = ()  # zero these evidence maps
    permute_frame_order: bool = False
    tta_view: Optional[int] = None         # index into transforms.tta_views()
    seed: int = 0

    def is_identity(self) -> bool:
        return (self.mask_condition == "clean" and not self.permute_masks
                and self.corrupt_kind is None and self.drop_frame_idx is None
                and self.keep_frames is None and not self.suppress_regions
                and not self.permute_frame_order and self.tta_view is None)


# --------------------------------------------------------------------------- #
class PatientBagDataset(Dataset):
    """Yields dicts of tensors; shapes are fixed so the default collate works."""

    def __init__(self,
                 manifest: pd.DataFrame,
                 cfg: DataConfig,
                 train: bool = False,
                 need_regions: bool = True,
                 region_res: Optional[int] = 28,
                 regions: Sequence[str] = ("core", "margin", "peri", "global"),
                 intervention: Optional[Intervention] = None,
                 epoch_seed: int = 0):
        self.man = manifest.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.need_regions = need_regions
        self.regions = tuple(regions)
        # masked_input mode needs full-resolution region maps
        self.region_res = region_res if region_res else cfg.image_size
        self.tf = JointTransform(cfg, train)
        self.intervention = intervention or Intervention()
        self.epoch_seed = epoch_seed
        self._tta = tta_views(8)

    def __len__(self) -> int:
        return len(self.man)

    def set_epoch(self, epoch: int) -> None:
        self.epoch_seed = epoch

    # ------------------------------------------------------------------ #
    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.cfg.image_size, self.cfg.image_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.image_size, self.cfg.image_size),
                         interpolation=cv2.INTER_LINEAR)
        return (img.astype(np.float32) / 255.0)

    def _read_mask(self, mask_path: Optional[str], xml_path: Optional[str]) -> np.ndarray:
        size = self.cfg.image_size
        if mask_path and os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                return binarize(m)
        if xml_path and os.path.exists(xml_path):
            boxes, src_hw_wh, _names = parse_voc_bbox(xml_path)
            src_hw = (src_hw_wh[1], src_hw_wh[0]) if src_hw_wh else None
            return bbox_to_mask(boxes, (size, size), src_hw)
        # No annotation: fall back to a centred ellipse covering the mid-field.
        m = np.zeros((size, size), np.uint8)
        cv2.ellipse(m, (size // 2, size // 2), (size // 4, size // 5), 0, 0, 360, 1, -1)
        return m

    # ------------------------------------------------------------------ #
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.man.iloc[index]
        iv = self.intervention
        base_seed = (hash((str(row["patient_id"]), int(self.epoch_seed), int(iv.seed)))
                     % (2 ** 31 - 1))
        rng = np.random.RandomState(base_seed)

        paths: List[str] = [p for p in row["image_paths"] if p]
        mpaths: List[Optional[str]] = list(row["mask_paths"])[:len(paths)]
        xpaths: List[Optional[str]] = list(row["bbox_xmls"])[:len(paths)]
        while len(mpaths) < len(paths):
            mpaths.append(None)
        while len(xpaths) < len(paths):
            xpaths.append(None)

        order = list(range(len(paths)))
        if iv.keep_frames is not None:
            order = [i for i in order if i in set(iv.keep_frames)]
        if iv.drop_frame_idx is not None:
            order = [i for i in order if i != iv.drop_frame_idx]
        if iv.permute_frame_order:
            order = list(rng.permutation(order))
        if not order:                       # never hand back an empty bag
            order = [0]

        t_max = self.cfg.t_max
        order = order[:t_max]
        n = len(order)

        size = self.cfg.image_size
        K = len(self.regions)
        R = self.region_res

        images = np.zeros((t_max, 3, size, size), np.float32)
        lesion = np.zeros((t_max, 1, size, size), np.float32)
        regmaps = np.zeros((t_max, K, R, R), np.float32)
        valid = np.zeros((t_max,), np.float32)

        # which frames get corrupted
        corrupt_set: set = set()
        if iv.corrupt_kind is not None:
            if iv.corrupt_frame_idx is not None:
                corrupt_set = {iv.corrupt_frame_idx}
            else:
                k = min(iv.corrupt_n_frames, n)
                corrupt_set = set(rng.choice(n, size=k, replace=False).tolist())

        raw_masks: List[np.ndarray] = []
        raw_imgs: List[np.ndarray] = []
        for slot, fi in enumerate(order):
            img = self._read_image(paths[fi])
            msk = self._read_mask(mpaths[fi], xpaths[fi])
            raw_imgs.append(img)
            raw_masks.append(msk)

        # within-patient mask permutation (shortcut-sensitivity diagnostic):
        # images untouched, masks shuffled across this patient's valid frames.
        if iv.permute_masks and n > 1:
            perm = rng.permutation(n)
            if np.all(perm == np.arange(n)):
                perm = np.roll(np.arange(n), 1)
            raw_masks = [raw_masks[i] for i in perm]

        for slot in range(n):
            img, msk = raw_imgs[slot], raw_masks[slot]

            if iv.mask_condition != "clean":
                msk = degrade_mask(msk, iv.mask_condition)
            if slot in corrupt_set:
                img = corrupt_frame(img, iv.corrupt_kind, iv.corrupt_severity, rng)
            if iv.tta_view is not None:
                img, msk = apply_tta(img, msk, self._tta[iv.tta_view % len(self._tta)])
            img, msk = self.tf(img, msk, rng)
            msk = binarize(msk) if msk is not None else np.zeros((size, size), np.uint8)

            images[slot] = normalize_chw(img, self.cfg.normalize_mean, self.cfg.normalize_std)
            lesion[slot, 0] = msk.astype(np.float32)
            if self.need_regions:
                rm = make_regions(msk,
                                  self.cfg.margin_inner_px,
                                  self.cfg.margin_outer_px,
                                  self.cfg.peri_px,
                                  names=self.regions)
                rm = resize_regions(rm, R)
                for ri, rname in enumerate(self.regions):
                    if rname in iv.suppress_regions:
                        rm[ri] = 0.0
                regmaps[slot] = rm
            valid[slot] = 1.0

        return {
            "image": torch.from_numpy(images),
            "lesion": torch.from_numpy(lesion),
            "regions": torch.from_numpy(regmaps),
            "valid": torch.from_numpy(valid),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "index": torch.tensor(int(index), dtype=torch.long),
            "n_frames": torch.tensor(int(n), dtype=torch.long),
        }

    # ------------------------------------------------------------------ #
    def patient_ids(self) -> List[str]:
        return list(self.man["patient_id"].astype(str))

    def labels(self) -> np.ndarray:
        return self.man["label"].to_numpy().astype(int)


# --------------------------------------------------------------------------- #
def make_loader(dataset: PatientBagDataset,
                batch_size: int,
                shuffle: bool,
                cfg: DataConfig,
                seed: int = 0,
                drop_last: bool = False) -> torch.utils.data.DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        generator=g if shuffle else None,
        persistent_workers=cfg.num_workers > 0,
    )
