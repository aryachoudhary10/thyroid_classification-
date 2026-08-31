"""Arm 2 of the external validation: U-Net-generated pixel masks for TN5000.

TN5000 ships Pascal VOC bounding boxes and no pixel masks, so the bbox arm is
the only one the dataset supports directly. The source paper's headline external
number was produced with U-Net-generated masks, so reproducing that arm means
generating them the same way:

    1. train a segmenter on ThyroidXL, where every frame has a ground-truth
       pixel mask, using the development cohort only;
    2. run it over TN5000 and cache one PNG mask per image;
    3. point a copy of the TN5000 manifest at those PNGs and run the ordinary
       domain-adaptation protocol against it.

The segmenter never sees a TN5000 label and is never trained on TN5000, so this
adds no target-domain supervision to the classification protocol. It does add a
dependency: the masks are predictions, and their quality bounds the arm. The
mean Dice on a held-out ThyroidXL split is reported so that bound is explicit
rather than assumed.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.manifest import load_manifest, save_manifest
from ..data.regions import binarize
from ..data.splits import test_frame
from ..data.transforms import normalize_chw
from ..models.unet import ResUNet, dice_bce_loss, dice_score
from ..utils.checkpoint import CheckpointManager, StageRegistry
from ..utils.common import banner, log, save_json


# --------------------------------------------------------------------------- #
class FrameSegDataset(torch.utils.data.Dataset):
    """Flat (image, mask) frames pulled out of a patient-level manifest.

    ``require_mask`` keeps only frames that actually have a pixel mask on disk,
    which is what training needs; inference sets it False and returns the frame
    index so predictions can be written back against the right image path.
    """

    def __init__(self, man: pd.DataFrame, cfg: Config, train: bool = False,
                 require_mask: bool = True):
        self.cfg = cfg
        self.train = train
        self.size = cfg.data.image_size
        rows: List[Tuple[str, Optional[str]]] = []
        for _i, r in man.iterrows():
            paths = [p for p in r["image_paths"] if p]
            mpaths = list(r["mask_paths"])[:len(paths)]
            while len(mpaths) < len(paths):
                mpaths.append(None)
            for ip, mp in zip(paths, mpaths):
                has = isinstance(mp, str) and mp and os.path.exists(mp)
                if require_mask and not has:
                    continue
                rows.append((ip, mp if has else None))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, path: str, gray: bool = False) -> Optional[np.ndarray]:
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        im = cv2.imread(path, flag)
        if im is None:
            return None
        interp = cv2.INTER_NEAREST if gray else cv2.INTER_LINEAR
        return cv2.resize(im, (self.size, self.size), interpolation=interp)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ip, mp = self.rows[i]
        img = self._read(ip)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if mp:
            m = self._read(mp, gray=True)
            msk = binarize(m) if m is not None else np.zeros((self.size, self.size), np.uint8)
        else:
            msk = np.zeros((self.size, self.size), np.uint8)

        if self.train:
            if np.random.rand() < self.cfg.data.aug_hflip:
                img = img[:, ::-1].copy()
                msk = msk[:, ::-1].copy()
            if np.random.rand() < 0.5:
                f = 1.0 + np.random.uniform(-self.cfg.data.aug_brightness,
                                            self.cfg.data.aug_brightness)
                img = np.clip(img * f, 0.0, 1.0)

        x = normalize_chw(img, self.cfg.data.normalize_mean, self.cfg.data.normalize_std)
        return {"image": torch.from_numpy(x),
                "mask": torch.from_numpy(msk.astype(np.float32))[None],
                "index": torch.tensor(i, dtype=torch.long)}


# --------------------------------------------------------------------------- #
def train_segmenter(cfg: Config, manifest: pd.DataFrame, registry: StageRegistry,
                    val_frac: float = 0.1) -> Dict[str, Any]:
    """Train the ThyroidXL segmenter. Resumable; returns the best checkpoint."""
    key = "seg/train/%s" % cfg.run.run_name
    run = "%s/segmenter" % cfg.run.run_name
    cm = CheckpointManager(cfg.run.ckpt_root, run)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, "segmenter.json")

    if registry.is_done(key) and cm.has_best():
        log("SKIP  " + key)
        return {"ckpt": cm.best_path, **registry.artifacts(key)}

    banner("SEGMENTER TRAINING (ThyroidXL development cohort)")
    sc = cfg.seg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Development cohort only -- the test cohort stays untouched, and TN5000 is
    # never involved in training the segmenter.
    test_ids = set(test_frame(manifest)["patient_id"].astype(str))
    dev = manifest[~manifest["patient_id"].astype(str).isin(test_ids)].reset_index(drop=True)
    if dev.empty:
        dev = manifest.reset_index(drop=True)

    rng = np.random.RandomState(cfg.run.seed)
    perm = rng.permutation(len(dev))
    n_val = max(int(round(val_frac * len(dev))), 1)
    va = dev.iloc[perm[:n_val]].reset_index(drop=True)
    tr = dev.iloc[perm[n_val:]].reset_index(drop=True)

    tr_ds = FrameSegDataset(tr, cfg, train=True, require_mask=True)
    va_ds = FrameSegDataset(va, cfg, train=False, require_mask=True)
    log("  frames: %d train / %d val (patients %d / %d)"
        % (len(tr_ds), len(va_ds), len(tr), len(va)))
    if len(tr_ds) == 0:
        log("  no ground-truth pixel masks available -- cannot train a segmenter")
        return {"skipped": True, "reason": "no pixel masks in the manifest"}

    model = ResUNet(sc.backbone, cfg.model.pretrained)
    start_epoch, best = 0, -math.inf
    if cm.has_last():
        ck = cm.load_last(model=model, map_location="cpu")
        start_epoch = int(ck.get("epoch", 0))
        best = float(ck.get("best_metric", -math.inf))
        log("  resuming segmenter at epoch %d (best Dice %.4f)" % (start_epoch, best))
    model.to(device)

    amp = cfg.optim.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    opt = torch.optim.AdamW(model.parameters(), lr=sc.lr,
                            weight_decay=cfg.optim.weight_decay)
    if cm.has_last() and start_epoch > 0:
        cm.load(cm.last_path, optimizer=opt, scaler=scaler,
                map_location=str(device), restore_rng=False)

    for epoch in range(start_epoch, sc.epochs):
        model.set_encoder_trainable(epoch >= sc.freeze_encoder_epochs)
        for g in opt.param_groups:
            g["lr"] = sc.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(sc.epochs, 1)))

        model.train()
        loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=sc.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, drop_last=False)
        losses = []
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = dice_bce_loss(model(x), y, sc.dice_weight)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        del loader

        model.eval()
        dices: List[float] = []
        vl = torch.utils.data.DataLoader(va_ds, batch_size=sc.batch_size,
                                         shuffle=False, num_workers=cfg.data.num_workers)
        with torch.no_grad():
            for batch in vl:
                x = batch["image"].to(device)
                y = batch["mask"].to(device)
                dices.extend(dice_score(model(x), y).cpu().numpy().tolist())
        del vl
        d = float(np.mean(dices)) if dices else float("nan")
        improved = (not cm.has_best()) or (d == d and d > best + 1e-6)
        if d == d and d > best:
            best = d
        log("  epoch %02d  loss %.4f  val Dice %.4f%s"
            % (epoch, float(np.mean(losses)) if losses else float("nan"), d,
               "  *" if improved else ""))
        cm.save(model, opt, scaler, stage=1, epoch=epoch + 1, global_step=0,
                step_in_epoch=0, best_metric=best, best_stage=1, best_epoch=epoch,
                epochs_no_improve=0, extra={"kind": "segmenter"}, is_best=improved)

    res = {"ckpt": cm.best_path, "val_dice": best, "backbone": sc.backbone,
           "n_train_frames": len(tr_ds), "n_val_frames": len(va_ds)}
    save_json(res, out)
    registry.mark_done(key, {"json": out, "val_dice": best, "ckpt": cm.best_path})
    log("  segmenter done | held-out ThyroidXL Dice %.4f" % best)
    del model, tr_ds, va_ds
    torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_masks(cfg: Config, seg_ckpt: str, man: pd.DataFrame,
                  registry: StageRegistry, tag: str = "tn5000",
                  min_area_frac: float = 0.001) -> pd.DataFrame:
    """Write one predicted PNG mask per image and return a repointed manifest.

    A prediction that collapses to (near) nothing is discarded and the frame's
    ``mask_paths`` entry is left as None, so the dataset falls back to the VOC
    box rather than handing the model an empty lesion branch. Empty masks are a
    known failure mode of cross-domain segmentation and silently feeding them
    through would corrupt the arm.
    """
    key = "seg/predict/%s/%s" % (cfg.run.run_name, tag)
    mask_dir = os.path.join(cfg.run.results_root, cfg.run.run_name,
                            "%s_unet_masks" % tag)
    out_csv = os.path.join(cfg.run.results_root, cfg.run.run_name,
                           "%s_manifest_unet.csv" % tag)
    if registry.is_done(key) and os.path.exists(out_csv):
        log("SKIP  " + key)
        return load_manifest(out_csv)

    banner("SEGMENTER INFERENCE  |  %s" % tag)
    os.makedirs(mask_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(cfg.seg.backbone, pretrained=False)
    state = torch.load(seg_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    ds = FrameSegDataset(man, cfg, train=False, require_mask=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.seg.batch_size,
                                         shuffle=False, num_workers=cfg.data.num_workers)
    size = cfg.data.image_size
    min_area = max(int(min_area_frac * size * size), 1)
    written: Dict[str, str] = {}
    n_empty = 0
    areas: List[float] = []

    for batch in loader:
        x = batch["image"].to(device)
        prob = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        for j, gi in enumerate(batch["index"].numpy().tolist()):
            ip, _mp = ds.rows[gi]
            m = (prob[j] > cfg.seg.threshold).astype(np.uint8)
            if int(m.sum()) < min_area:
                n_empty += 1
                continue
            areas.append(float(m.mean()))
            stem = os.path.splitext(os.path.basename(ip))[0]
            path = os.path.join(mask_dir, "%s.png" % stem)
            cv2.imwrite(path, m * 255)
            written[ip] = path
    del loader, ds

    def repoint(row) -> List[Optional[str]]:
        return [written.get(p) for p in row["image_paths"]]

    out = man.copy()
    out["mask_paths"] = out.apply(repoint, axis=1)
    save_manifest(out, out_csv)

    total = sum(len(r) for r in out["image_paths"])
    log("  wrote %d masks for %d frames (%d rejected as empty, %.1f%%)"
        % (len(written), total, n_empty, 100.0 * n_empty / max(total, 1)))
    if areas:
        log("  predicted lesion area: mean %.3f of frame, median %.3f"
            % (float(np.mean(areas)), float(np.median(areas))))
    registry.mark_done(key, {"csv": out_csv, "dir": mask_dir,
                             "n_masks": len(written), "n_empty": n_empty})
    del model
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
def unet_mask_manifest(cfg: Config, manifest: pd.DataFrame,
                       tn_manifest: pd.DataFrame,
                       registry: StageRegistry) -> Optional[pd.DataFrame]:
    """Train the segmenter on ThyroidXL, then repoint TN5000 at its predictions.

    Returns None when no ground-truth masks are available to train on, so the
    caller can skip the arm rather than silently evaluate on ellipse fallbacks.
    """
    seg = train_segmenter(cfg, manifest, registry)
    if seg.get("skipped") or not seg.get("ckpt"):
        log("  U-Net arm unavailable: " + str(seg.get("reason", "no checkpoint")))
        return None
    return predict_masks(cfg, seg["ckpt"], tn_manifest, registry, tag="tn5000")
