"""Setup B: external domain-adapted validation on TN5000.

TN5000 is a Chinese thyroid ultrasound dataset acquired on different scanners
under a different population prior, and it is image-level rather than
patient-level. Each image therefore becomes a bag of one, which is the honest
mapping -- padding a single frame into a fake multi-view bag would fabricate
cross-view evidence that does not exist.

Protocol, applied identically to the preserved RCAF baseline and to DER-MIL:

    1. start from the ThyroidXL-trained checkpoint
    2. domain-adapt on the TN5000 training split with progressive backbone
       unfreezing, focal loss with label smoothing, cosine LR + 5-epoch warmup
    3. evaluate ONCE on a class-balanced held-out subset (125 malignant +
       125 benign) with 8-view deterministic TTA
    4. repeat with bounding-box masks derived from the VOC XML annotations, to
       show how the models behave when only weak localisation is available

The evaluation subset is carved out first and is excluded from adaptation,
early stopping, model selection and threshold tuning.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.dataset import Intervention, PatientBagDataset
from ..data.discovery import LayoutOverride, build_records, label_from_voc
from ..data.manifest import (build_patient_manifest, load_manifest, mask_coverage,
                             save_manifest)
from ..data.tn5000 import (build_tn5000_manifest as _adapter_manifest,
                           looks_like_tn5000, official_eval_subset, split_summary)
from ..data.splits import class_pos_weight
from ..engine.cv import make_dataset
from ..engine.trainer import move_batch, predict_dataset
from ..eval.calibration import sigmoid
from ..eval.metrics import all_metrics, bootstrap_ci, delong_test
from ..losses.losses import focal_loss
from ..models.factory import build_model
from ..utils.checkpoint import CheckpointManager, StageRegistry
from ..utils.common import banner, load_json, log, save_json


# --------------------------------------------------------------------------- #
def build_tn5000_manifest(cfg: Config, registry: StageRegistry,
                          override: Optional[LayoutOverride] = None) -> pd.DataFrame:
    key = "tn5000/manifest/%s" % cfg.run.run_name
    path = os.path.join(cfg.run.results_root, cfg.run.run_name, "tn5000_manifest.csv")
    if registry.is_done(key) and os.path.exists(path):
        log("SKIP  " + key)
        return load_manifest(path)

    banner("TN5000 MANIFEST")
    root = cfg.data.tn5000_root
    if not root:
        raise ValueError("cfg.data.tn5000_root is not set")

    if override is None and looks_like_tn5000(root):
        man = _adapter_manifest(root)
        summ = split_summary(man)
        if not summ.empty:
            log("official split balance:" + chr(10) + summ.to_string(index=False))
    else:
        log("TN5000 adapter not applicable -- using generic discovery")
        rec = build_records(root, override, drop_unlabeled=False)
        if rec["label"].isna().any() or rec["label"].nunique(dropna=True) < 2:
            labs = []
            for _i, r in rec.iterrows():
                v = r.get("label")
                if pd.isna(v) and isinstance(r.get("bbox_xml"), str):
                    v = label_from_voc(r["bbox_xml"])
                labs.append(v)
            rec["label"] = labs
            rec = rec[~rec["label"].isna()].copy()
            rec["label"] = rec["label"].astype(int)
        rec["patient_id"] = rec["stem"].astype(str) if "stem" in rec else rec["image_path"]
        man = build_patient_manifest(rec, t_max=1)

    log("TN5000: %d images, %.1f%% malignant" % (len(man), 100 * man["label"].mean()))
    log("mask coverage: " + str(mask_coverage(man)))

    save_manifest(man, path)
    registry.mark_done(key, {"csv": path, "n": len(man)})
    return man


def carve_evaluation_subset(man: pd.DataFrame, per_class: int = 125,
                            seed: int = 1337) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Class-balanced held-out subset, removed from everything else."""
    rng = np.random.RandomState(seed)
    # Honour an official validation split if one exists; otherwise sample.
    pool = man[man["split"] == "val"] if (man["split"] == "val").sum() >= 2 * per_class else man

    picks = []
    for cls in (1, 0):
        idx = np.array(pool.index[pool["label"] == cls].to_numpy(), copy=True)
        rng.shuffle(idx)
        take = min(per_class, len(idx))
        if take < per_class:
            log("TN5000: WARNING only %d of %d requested class-%d cases available"
                % (take, per_class, cls))
        picks.append(idx[:take])
    eval_idx = np.concatenate(picks)
    eval_df = man.loc[eval_idx].reset_index(drop=True)
    train_df = man.drop(index=eval_idx).reset_index(drop=True)
    log("TN5000 split: adapt on %d images, evaluate on %d (%d malignant / %d benign)"
        % (len(train_df), len(eval_df), int((eval_df['label'] == 1).sum()),
           int((eval_df['label'] == 0).sum())))
    return train_df, eval_df


def force_bbox_masks(man: pd.DataFrame) -> pd.DataFrame:
    """Robustness arm: ignore pixel masks and use VOC bounding boxes instead."""
    out = man.copy()
    out["mask_paths"] = out["mask_paths"].map(lambda lst: [None] * len(lst))
    return out


# --------------------------------------------------------------------------- #
def _lr_at(epoch: int, cfg: Config) -> float:
    e = cfg.external
    if epoch < e.warmup_epochs:
        return e.lr * float(epoch + 1) / float(max(e.warmup_epochs, 1))
    prog = (epoch - e.warmup_epochs) / max(e.epochs - e.warmup_epochs, 1)
    return e.lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def _unfreeze_level(epoch: int, cfg: Config) -> int:
    return sum(1 for m in cfg.external.unfreeze_epochs if epoch >= m)


def _set_unfreeze(model: torch.nn.Module, level: int) -> None:
    bb = getattr(model, "backbone", None)
    if bb is None:
        bb = getattr(getattr(model, "encoder", None), "backbone", None)
    if bb is not None and hasattr(bb, "set_progressive_unfreeze"):
        bb.set_progressive_unfreeze(level)


def domain_adapt(cfg: Config, model_name: str, source_ckpt: str,
                 train_df: pd.DataFrame, val_df: pd.DataFrame,
                 registry: StageRegistry, tag: str) -> str:
    """Resumable TN5000 fine-tuning. Returns the best checkpoint path."""
    key = "tn5000/adapt/%s/%s/%s" % (cfg.run.run_name, model_name, tag)
    run = "%s/%s/tn5000_%s" % (cfg.run.run_name, model_name, tag)
    cm = CheckpointManager(cfg.run.ckpt_root, run)

    if registry.is_done(key) and cm.has_best():
        log("SKIP  " + key)
        return cm.best_path

    banner("TN5000 DOMAIN ADAPTATION  |  %s  (%s masks)" % (model_name, tag))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reqs = build_model(cfg, model_name)

    start_epoch, best = 0, -math.inf
    if cm.has_last():
        ck = cm.load_last(model=model, map_location="cpu")
        start_epoch = int(ck.get("epoch", 0))
        best = float(ck.get("best_metric", -math.inf))
        log("  resuming TN5000 adaptation at epoch %d (best %.4f)" % (start_epoch, best))
    else:
        src = torch.load(source_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(src["model"], strict=False)
        log("  loaded ThyroidXL weights (%d missing, %d unexpected)"
            % (len(missing), len(unexpected)))
    model.to(device)

    tr_ds = make_dataset(cfg, train_df, reqs, train=True)
    va_ds = make_dataset(cfg, val_df, reqs, train=False)
    pw = class_pos_weight(train_df)
    amp = cfg.optim.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for epoch in range(start_epoch, cfg.external.epochs):
        level = _unfreeze_level(epoch, cfg)
        _set_unfreeze(model, level)
        lr = _lr_at(epoch, cfg)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.optim.weight_decay)
        if cm.has_last() and epoch == start_epoch and start_epoch > 0:
            cm.load(cm.last_path, optimizer=opt, scaler=scaler,
                    map_location=str(device), restore_rng=False)

        model.train()
        loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=cfg.external.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, drop_last=False)
        losses = []
        pw_t = torch.tensor(pw, device=device)
        for batch in loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(batch)
                loss = focal_loss(out["logit"], batch["label"],
                                  cfg.external.focal_gamma,
                                  cfg.external.label_smoothing, pw_t)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        del loader

        logits, labels, _ = predict_dataset(model, va_ds, cfg, device,
                                            batch_size=cfg.external.batch_size)
        auc = all_metrics(labels, sigmoid(logits))["roc_auc"]
        finite = auc == auc
        # Always materialise a best checkpoint on the first epoch: a monitoring
        # split that happens to be single-class must not leave us with none.
        improved = (not cm.has_best()) or (finite and auc > best + 1e-6)
        if finite and auc > best:
            best = auc
        log("  epoch %02d  lr %.2e  unfreeze %d  loss %.4f  val AUC %.4f%s"
            % (epoch, lr, level, float(np.mean(losses)) if losses else float("nan"),
               auc, "  *" if improved else ""))
        cm.save(model, opt, scaler, stage=1, epoch=epoch + 1, global_step=0,
                step_in_epoch=0, best_metric=best, best_stage=1, best_epoch=epoch,
                epochs_no_improve=0, extra={"tag": tag}, is_best=improved)

    registry.mark_done(key, {"ckpt": cm.best_path, "best_val_auc": best})
    del model, tr_ds, va_ds
    torch.cuda.empty_cache()
    return cm.best_path


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_with_tta(cfg: Config, model_name: str, ckpt: str,
                      eval_df: pd.DataFrame, n_views: int = 8) -> Dict[str, Any]:
    """Deterministic n-view TTA; probabilities are averaged across views."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reqs = build_model(cfg, model_name)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    probs, labels = [], None
    for v in range(max(n_views, 1)):
        ds = make_dataset(cfg, eval_df, reqs, train=False,
                          intervention=Intervention(tta_view=v))
        lg, y, _e = predict_dataset(model, ds, cfg, device,
                                    batch_size=cfg.external.batch_size)
        probs.append(sigmoid(lg))
        labels = y
        del ds
    p = np.mean(np.vstack(probs), axis=0)
    del model
    torch.cuda.empty_cache()
    return {"p": p, "y": labels}


def external_validation(cfg: Config, model_name: str, source_ckpt: str,
                        tn_manifest: pd.DataFrame, registry: StageRegistry,
                        mask_variants: Tuple[str, ...] = ("pixel", "bbox")
                        ) -> pd.DataFrame:
    """Adapt + evaluate one model on TN5000 under each mask variant."""
    key = "tn5000/eval/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "tn5000_results.csv")
    pred_dir = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name)
    if registry.is_done(key) and os.path.exists(out):
        log("SKIP  " + key)
        return pd.read_csv(out)

    rows = []
    for variant in mask_variants:
        man = tn_manifest if variant == "pixel" else force_bbox_masks(tn_manifest)
        if (man["split"] == "val").sum() > 0:
            adapt_df, eval_df = official_eval_subset(
                man, cfg.external.eval_subset_per_class, cfg.run.seed)
        else:
            adapt_df, eval_df = carve_evaluation_subset(
                man, cfg.external.eval_subset_per_class, cfg.run.seed)
        # Small internal split for epoch monitoring, disjoint from the eval
        # subset and stratified so ROC-AUC stays defined.
        val_idx = []
        for cls in sorted(adapt_df["label"].unique()):
            pool_idx = adapt_df.index[adapt_df["label"] == cls]
            take = min(max(int(round(0.12 * len(pool_idx))), 2), max(len(pool_idx) - 1, 0))
            if take > 0:
                val_idx.extend(pd.Index(pool_idx).to_series()
                               .sample(take, random_state=cfg.run.seed).tolist())
        val_df = adapt_df.loc[val_idx].reset_index(drop=True)
        fit_df = adapt_df.drop(index=val_idx).reset_index(drop=True)
        if val_df["label"].nunique() < 2:
            log("  WARNING monitoring split is single-class; "
                "epoch selection falls back to the last epoch")

        ckpt = domain_adapt(cfg, model_name, source_ckpt, fit_df, val_df,
                            registry, variant)
        res = evaluate_with_tta(cfg, model_name, ckpt, eval_df, cfg.eval.tta_views)
        ci = bootstrap_ci(res["y"], res["p"], cfg.eval.bootstrap_external,
                          0.5, seed=cfg.run.seed)
        row = {"model": model_name, "mask_type": variant,
               "n_eval": int(len(res["y"]))}
        for k, (pt, lo, hi) in ci.items():
            row[k] = pt
            row[k + "_ci"] = "[%.3f-%.3f]" % (lo, hi)
        rows.append(row)
        log("  %s / %s masks: AUC %.3f %s"
            % (model_name, variant, row["roc_auc"], row["roc_auc_ci"]))

        os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame({"patient_id": eval_df["patient_id"].astype(str).values,
                      "label": res["y"].astype(int), "p": res["p"]}).to_csv(
            os.path.join(pred_dir, "tn5000_predictions_%s.csv" % variant), index=False)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    registry.mark_done(key, {"csv": out})
    return df


# --------------------------------------------------------------------------- #
def compare_external(cfg: Config, model_names: List[str],
                     variant: str = "pixel") -> Dict[str, Any]:
    """Head-to-head DeLong test on the shared TN5000 evaluation subset."""
    preds = {}
    for name in model_names:
        p = os.path.join(cfg.run.results_root, cfg.run.run_name, name,
                         "tn5000_predictions_%s.csv" % variant)
        if os.path.exists(p):
            preds[name] = pd.read_csv(p)
    if len(preds) < 2:
        return {}
    names = list(preds)
    a, b = preds[names[0]], preds[names[1]]
    merged = a.merge(b, on="patient_id", suffixes=("_a", "_b"))
    if merged.empty:
        return {}
    res = delong_test(merged["label_a"].values, merged["p_a"].values,
                      merged["p_b"].values)
    res["model_a"], res["model_b"], res["mask_type"] = names[0], names[1], variant
    log("TN5000 %s: %s AUC %.4f vs %s AUC %.4f  (DeLong z=%.2f, p=%.4g)"
        % (variant, names[0], res["auc1"], names[1], res["auc2"],
           res["z"], res["p_value"]))
    return res
