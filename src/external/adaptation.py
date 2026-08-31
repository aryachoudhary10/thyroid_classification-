"""Label-free adaptation arms for the TN5000 external validation.

The finished bbox arm fine-tunes on TN5000 *labels*, which is the source paper's
protocol but also its weakest claim: it says the model can be taught a new
domain, not that it generalises to one. These two arms use no target labels at
all, so they answer the question a reviewer actually asks.

    Arm 4  uncertainty-aware pseudo-labeling
           The source model scores the adaptation pool under TTA. Predictions
           that are both confident and *stable across views* become training
           targets; everything else is discarded. Confidence alone is not
           enough -- a miscalibrated model is confidently wrong, and on this
           dataset we have already seen a model assign a mean probability of
           0.97 while clearing 3 of 125 benign cases. View-variance is the part
           that catches that, because a genuinely wrong prediction tends to be
           unstable under flips and small rotations.

    Arm 5  test-time adaptation (TENT)
           Only the normalisation layers' affine parameters are updated, by
           minimising prediction entropy on the evaluation data itself. No
           labels, no training pool, one pass. Classical TENT assumes
           BatchNorm; the ConvNeXt and Swin trunks from arm 3 use LayerNorm, so
           both are collected.

Neither arm ever reads ``eval_df["label"]`` except to score the final result.
"""
from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import Config
from ..data.dataset import Intervention
from ..engine.cv import make_dataset
from ..engine.trainer import move_batch, predict_dataset
from ..eval.calibration import sigmoid
from ..eval.metrics import all_metrics, bootstrap_ci
from ..losses.losses import focal_loss
from ..models.factory import build_model
from ..utils.checkpoint import CheckpointManager, StageRegistry
from ..utils.common import banner, log, save_json


# --------------------------------------------------------------------------- #
@torch.no_grad()
def tta_statistics(cfg: Config, model: nn.Module, reqs: Dict[str, object],
                   df: pd.DataFrame, device: torch.device, n_views: int = 8
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and across-view standard deviation of the predicted probability.

    The standard deviation is the uncertainty signal: a prediction that survives
    eight deterministic geometric views unchanged is a different object from one
    that swings, even when both have the same mean.
    """
    probs: List[np.ndarray] = []
    labels = None
    for v in range(max(n_views, 1)):
        ds = make_dataset(cfg, df, reqs, train=False,
                          intervention=Intervention(tta_view=v))
        lg, y, _e = predict_dataset(model, ds, cfg, device,
                                    batch_size=cfg.external.batch_size)
        probs.append(sigmoid(lg))
        labels = y
        del ds
    p = np.vstack(probs)
    return p.mean(axis=0), p.std(axis=0), labels


def select_pseudo_labels(mean_p: np.ndarray, std_p: np.ndarray,
                         conf: float = 0.85, max_std: float = 0.10,
                         balance: bool = True, seed: int = 0
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Indices and hard targets for the samples worth training on.

    ``balance`` equalises the two pseudo-classes by subsampling the larger one.
    Without it the arm inherits the source model's prior: on TN5000 the source
    models skew heavily malignant, so an unbalanced pool would reinforce exactly
    the bias that makes the zero-shot specificity collapse.
    """
    keep_pos = (mean_p >= conf) & (std_p <= max_std)
    keep_neg = (mean_p <= 1.0 - conf) & (std_p <= max_std)
    idx_pos = np.where(keep_pos)[0]
    idx_neg = np.where(keep_neg)[0]

    if balance and len(idx_pos) and len(idx_neg):
        rng = np.random.RandomState(seed)
        n = min(len(idx_pos), len(idx_neg))
        idx_pos = rng.choice(idx_pos, n, replace=False)
        idx_neg = rng.choice(idx_neg, n, replace=False)

    idx = np.concatenate([idx_pos, idx_neg])
    y = np.concatenate([np.ones(len(idx_pos)), np.zeros(len(idx_neg))])
    order = np.argsort(idx)
    return idx[order], y[order].astype(np.float32)


# --------------------------------------------------------------------------- #
def pseudo_label_adapt(cfg: Config, model_name: str, source_ckpt: str,
                       adapt_df: pd.DataFrame, registry: StageRegistry,
                       tag: str = "upl") -> Tuple[str, Dict[str, Any]]:
    """Arm 4. Self-train on the adaptation pool using no target labels."""
    key = "tn5000/upl/%s/%s/%s" % (cfg.run.run_name, model_name, tag)
    run = "%s/%s/tn5000_%s" % (cfg.run.run_name, model_name, tag)
    cm = CheckpointManager(cfg.run.ckpt_root, run)
    if registry.is_done(key):
        art = dict(registry.artifacts(key))
        if art.get("degenerate"):
            log("SKIP  %s  (no usable pseudo-labels; source weights stand)" % key)
            return source_ckpt, art
        if cm.has_best():
            log("SKIP  " + key)
            return cm.best_path, art

    banner("TN5000 UNCERTAINTY-AWARE PSEUDO-LABELING  |  %s" % model_name)
    ac = cfg.adapt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reqs = build_model(cfg, model_name)
    src = torch.load(source_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(src["model"], strict=False)
    model.to(device)

    # The pool carries real labels; they are dropped here so nothing downstream
    # can read them by accident.
    pool = adapt_df.reset_index(drop=True)
    history: List[Dict[str, Any]] = []
    amp = cfg.optim.amp and device.type == "cuda"

    for rnd in range(ac.upl_rounds):
        model.eval()
        mean_p, std_p, _y = tta_statistics(cfg, model, reqs, pool, device,
                                           cfg.eval.tta_views)
        idx, pseudo = select_pseudo_labels(mean_p, std_p, ac.upl_confidence,
                                           ac.upl_max_std, True, cfg.run.seed + rnd)
        if len(idx) < ac.upl_min_samples or len(np.unique(pseudo)) < 2:
            log("  round %d: only %d usable pseudo-labels (need >= %d, both "
                "classes) -- stopping" % (rnd, len(idx), ac.upl_min_samples))
            break

        sub = pool.iloc[idx].reset_index(drop=True).copy()
        true = sub["label"].to_numpy().astype(float)
        sub["label"] = pseudo
        agree = float((true == pseudo).mean())
        log("  round %d: %d/%d kept (%.1f%%) | %d pos %d neg | agreement with "
            "held-back truth %.3f"
            % (rnd, len(idx), len(pool), 100.0 * len(idx) / max(len(pool), 1),
               int(pseudo.sum()), int(len(pseudo) - pseudo.sum()), agree))
        history.append({"round": rnd, "n_kept": int(len(idx)),
                        "frac_kept": float(len(idx) / max(len(pool), 1)),
                        "pseudo_label_accuracy": agree})

        ds = make_dataset(cfg, sub, reqs, train=True)
        model.train()
        bb = getattr(model, "backbone", None) or getattr(
            getattr(model, "encoder", None), "backbone", None)
        if bb is not None and hasattr(bb, "set_progressive_unfreeze"):
            bb.set_progressive_unfreeze(1 + rnd)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=ac.upl_lr,
                                weight_decay=cfg.optim.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        for ep in range(ac.upl_epochs):
            loader = torch.utils.data.DataLoader(
                ds, batch_size=cfg.external.batch_size, shuffle=True,
                num_workers=cfg.data.num_workers)
            losses = []
            for batch in loader:
                batch = move_batch(batch, device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    out = model(batch)
                    loss = focal_loss(out["logit"], batch["label"],
                                      cfg.external.focal_gamma,
                                      cfg.external.label_smoothing, None)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            log("    epoch %d  loss %.4f" % (ep, float(np.mean(losses)) if losses else float("nan")))
            del loader
        del ds
        cm.save(model, opt, scaler, stage=1, epoch=rnd + 1, global_step=0,
                step_in_epoch=0, best_metric=float(rnd), best_stage=1,
                best_epoch=rnd, epochs_no_improve=0,
                extra={"tag": tag, "history": history}, is_best=True)

    del model
    torch.cuda.empty_cache()

    if not history:
        # Every candidate failed the confidence or stability gate, so no model
        # was ever fitted. Returning the source checkpoint keeps the ladder
        # scoreable, and the flag makes the rung report itself as degenerate
        # instead of silently duplicating the zero-shot row.
        log("  pseudo-labeling produced nothing usable -- the arm degenerates "
            "to the source model")
        info = {"rounds": [], "used_target_labels": False, "degenerate": True}
        registry.mark_done(key, {"degenerate": True, "rounds": 0})
        return source_ckpt, info

    info = {"rounds": history, "used_target_labels": False, "degenerate": False}
    registry.mark_done(key, {"ckpt": cm.best_path, "rounds": len(history)})
    return cm.best_path, info


# --------------------------------------------------------------------------- #
def _norm_affine_params(model: nn.Module) -> List[nn.Parameter]:
    """Affine parameters of every normalisation layer, and nothing else."""
    out: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,
                          nn.GroupNorm, nn.InstanceNorm2d)):
            for p in (m.weight, m.bias):
                if p is not None:
                    p.requires_grad_(True)
                    out.append(p)
    return out


def _binary_entropy(logit: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logit).clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def tent_adapt(cfg: Config, model_name: str, ckpt: str, eval_df: pd.DataFrame,
               n_views: int = 8) -> Dict[str, Any]:
    """Arm 5. Entropy-minimising test-time adaptation, no labels anywhere.

    Adaptation happens on a copy of the model that is discarded afterwards, and
    the evaluation labels are never touched -- the objective is the entropy of
    the model's own predictions.
    """
    ac = cfg.adapt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reqs = build_model(cfg, model_name)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    for p in model.parameters():
        p.requires_grad_(False)
    params = _norm_affine_params(model)
    if not params:
        log("  no normalisation layers to adapt -- TENT is a no-op here")
        params = []

    # Normalisation layers in train mode so batch statistics come from the
    # target domain; everything else stays in eval mode.
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            m.train()

    opt = torch.optim.Adam(params, lr=ac.tent_lr) if params else None
    ds = make_dataset(cfg, eval_df, reqs, train=False)
    for step in range(ac.tent_steps if opt else 0):
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.external.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers)
        ents = []
        for batch in loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = _binary_entropy(model(batch)["logit"])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            ents.append(float(loss.detach().cpu()))
        log("  TENT step %d  mean entropy %.4f"
            % (step, float(np.mean(ents)) if ents else float("nan")))
        del loader
    del ds

    model.eval()
    mean_p, std_p, labels = tta_statistics(cfg, model, reqs, eval_df, device, n_views)
    del model
    torch.cuda.empty_cache()
    return {"p": mean_p, "std": std_p, "y": labels}


# --------------------------------------------------------------------------- #
def run_label_free_arms(cfg: Config, model_name: str, source_ckpt: str,
                        adapt_df: pd.DataFrame, eval_df: pd.DataFrame,
                        registry: StageRegistry) -> pd.DataFrame:
    """Arms 4 and 5 as a ladder, each added on top of the previous one.

    Reported separately rather than as one number, so a gain can be attributed
    to pseudo-labeling or to test-time adaptation instead of to "the method".
    """
    key = "tn5000/labelfree/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "tn5000_labelfree.csv")
    if registry.is_done(key) and os.path.exists(out):
        log("SKIP  " + key)
        return pd.read_csv(out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, Any]] = []
    preds: Dict[str, np.ndarray] = {}

    def score(name: str, res: Dict[str, Any], note: str) -> None:
        met = all_metrics(res["y"], res["p"])
        ci = bootstrap_ci(res["y"], res["p"], cfg.eval.bootstrap_external, 0.5,
                          seed=cfg.run.seed)
        row: Dict[str, Any] = {"model": model_name, "arm": name,
                               "target_labels": note,
                               "n_eval": int(len(res["y"]))}
        for k, (pt, lo, hi) in ci.items():
            row[k] = pt
            row[k + "_ci"] = "[%.3f-%.3f]" % (lo, hi)
        row["brier"] = met["brier"]
        rows.append(row)
        preds[name] = res["p"]
        log("  %-22s AUC %.4f %s" % (name, row["roc_auc"], row["roc_auc_ci"]))

    # ---- baseline rung: source model, no adaptation of any kind ------------ #
    banner("TN5000 LABEL-FREE LADDER  |  %s" % model_name)
    model, reqs = build_model(cfg, model_name)
    st = torch.load(source_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    model.to(device).eval()
    mp, sp, y = tta_statistics(cfg, model, reqs, eval_df, device, cfg.eval.tta_views)
    del model
    torch.cuda.empty_cache()
    score("zero_shot", {"p": mp, "std": sp, "y": y}, "none")

    # ---- rung 2: + uncertainty-aware pseudo-labeling ----------------------- #
    upl_ckpt, upl_info = pseudo_label_adapt(cfg, model_name, source_ckpt,
                                            adapt_df, registry)
    degenerate = bool(upl_info.get("degenerate"))
    model, reqs = build_model(cfg, model_name)
    st = torch.load(upl_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    model.to(device).eval()
    mp, sp, y = tta_statistics(cfg, model, reqs, eval_df, device, cfg.eval.tta_views)
    del model
    torch.cuda.empty_cache()
    score("upl" + (" (degenerate)" if degenerate else ""),
          {"p": mp, "std": sp, "y": y}, "none")

    # ---- rung 3: + test-time adaptation on top of the pseudo-labelled model  #
    res = tent_adapt(cfg, model_name, upl_ckpt, eval_df, cfg.eval.tta_views)
    score(("tent" if degenerate else "upl_tent"), res, "none")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    save_json({"pseudo_label_rounds": upl_info.get("rounds", [])},
              os.path.join(os.path.dirname(out), "tn5000_upl_rounds.json"))
    pd.DataFrame({"patient_id": eval_df["patient_id"].astype(str).values,
                  "label": y.astype(int), **preds}).to_csv(
        os.path.join(os.path.dirname(out), "tn5000_labelfree_predictions.csv"),
        index=False)
    registry.mark_done(key, {"csv": out})
    return df
