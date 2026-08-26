"""Robustness, shortcut and reliability-validation experiments.

Everything here runs on the official test cohort with frozen weights -- no
retraining, no threshold re-selection. Each experiment is cached by the stage
registry so a runtime restart never repeats a completed sweep.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.dataset import Intervention
from ..data.regions import CORRUPTIONS, MASK_CONDITIONS
from ..data.splits import test_frame
from ..engine.cv import predict_with_checkpoints
from ..eval.calibration import sigmoid
from ..eval.metrics import all_metrics
from ..utils.checkpoint import StageRegistry
from ..utils.common import banner, log, save_json


# --------------------------------------------------------------------------- #
def _eval(cfg: Config, frame: pd.DataFrame, model_name: str, ckpts: List[str],
          iv: Optional[Intervention], thr: float = 0.5) -> Dict[str, float]:
    logits, labels, _ = predict_with_checkpoints(cfg, frame, model_name, ckpts,
                                                 intervention=iv)
    return all_metrics(labels, sigmoid(logits), thr)


def _cached(registry: StageRegistry, key: str, path: str):
    if registry.is_done(key) and os.path.exists(path):
        log("SKIP  " + key)
        return pd.read_csv(path)
    return None


# --------------------------------------------------------------------------- #
def mask_quality_sweep(cfg: Config, manifest: pd.DataFrame, model_name: str,
                       ckpts: List[str], registry: StageRegistry,
                       conditions: Sequence[str] = ("clean", "dilate5", "erode5",
                                                    "erode15", "dilate15", "zeros")
                       ) -> pd.DataFrame:
    """Table 13: ROC-AUC under progressive segmentation degradation."""
    key = "robust/mask/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "mask_quality.csv")
    cached = _cached(registry, key, out)
    if cached is not None:
        return cached

    banner("MASK QUALITY SENSITIVITY  |  %s" % model_name)
    te = test_frame(manifest)
    rows = []
    base = None
    for cond in conditions:
        m = _eval(cfg, te, model_name, ckpts, Intervention(mask_condition=cond))
        if cond == "clean":
            base = m["roc_auc"]
        rows.append({"condition": cond,
                     "description": str(MASK_CONDITIONS[cond]),
                     "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "f1": m["f1"],
                     "delta": (m["roc_auc"] - base) if base is not None else 0.0})
        log("  %-10s ROC-AUC %.4f  (delta %+.4f)"
            % (cond, m["roc_auc"], rows[-1]["delta"]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    registry.mark_done(key, {"csv": out})
    return df


# --------------------------------------------------------------------------- #
def shortcut_sensitivity(cfg: Config, manifest: pd.DataFrame,
                         models: Dict[str, List[str]], registry: StageRegistry
                         ) -> pd.DataFrame:
    """Table 12: within-patient mask permutation, images untouched.

    A model that exploits mask--image alignment as a shortcut collapses here.
    A model that uses masks as structured anatomical guidance barely moves.
    """
    key = "robust/shortcut/%s" % cfg.run.run_name
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, "shortcut.csv")
    cached = _cached(registry, key, out)
    if cached is not None:
        return cached

    banner("SHORTCUT SENSITIVITY (within-patient mask permutation)")
    te = test_frame(manifest)
    # Permutation only bites when a patient has >1 frame.
    multi = te[te["n_frames"] > 1].reset_index(drop=True)
    log("  %d/%d test patients have more than one frame" % (len(multi), len(te)))

    rows = []
    for name, ckpts in models.items():
        std = _eval(cfg, multi, name, ckpts, None)
        shf = _eval(cfg, multi, name, ckpts, Intervention(permute_masks=True, seed=7))
        rows.append({"model": name,
                     "roc_auc": std["roc_auc"], "roc_auc_shuffled": shf["roc_auc"],
                     "d_roc_auc": shf["roc_auc"] - std["roc_auc"],
                     "pr_auc": std["pr_auc"], "pr_auc_shuffled": shf["pr_auc"],
                     "d_pr_auc": shf["pr_auc"] - std["pr_auc"],
                     "f1": std["f1"], "f1_shuffled": shf["f1"],
                     "d_f1": shf["f1"] - std["f1"]})
        log("  %-14s %.4f -> %.4f  (delta %+.4f)"
            % (name, std["roc_auc"], shf["roc_auc"], rows[-1]["d_roc_auc"]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    registry.mark_done(key, {"csv": out})
    return df


# --------------------------------------------------------------------------- #
def frame_corruption_study(cfg: Config, manifest: pd.DataFrame,
                           models: Dict[str, List[str]], registry: StageRegistry,
                           kinds: Sequence[str] = CORRUPTIONS,
                           severities: Sequence[float] = (0.5, 1.0)) -> pd.DataFrame:
    """RQ4: does uncertainty-aware reliability absorb a corrupted view?

    One frame per multi-frame patient is degraded; images from the other views
    are untouched. A model that cannot down-weight the bad view degrades more.
    """
    key = "robust/corrupt/%s" % cfg.run.run_name
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, "frame_corruption.csv")
    cached = _cached(registry, key, out)
    if cached is not None:
        return cached

    banner("FRAME CORRUPTION ROBUSTNESS")
    te = test_frame(manifest)
    multi = te[te["n_frames"] > 1].reset_index(drop=True)
    rows = []
    for name, ckpts in models.items():
        base = _eval(cfg, multi, name, ckpts, None)
        rows.append({"model": name, "corruption": "none", "severity": 0.0,
                     "roc_auc": base["roc_auc"], "delta": 0.0, "f1": base["f1"]})
        for kind in kinds:
            for sev in severities:
                iv = Intervention(corrupt_kind=kind, corrupt_severity=sev,
                                  corrupt_n_frames=1, seed=11)
                m = _eval(cfg, multi, name, ckpts, iv)
                rows.append({"model": name, "corruption": kind, "severity": sev,
                             "roc_auc": m["roc_auc"],
                             "delta": m["roc_auc"] - base["roc_auc"], "f1": m["f1"]})
                log("  %-14s %-10s sev %.1f  ROC-AUC %.4f (%+.4f)"
                    % (name, kind, sev, m["roc_auc"], rows[-1]["delta"]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    registry.mark_done(key, {"csv": out})
    return df


# --------------------------------------------------------------------------- #
def corruption_reliability_response(cfg: Config, manifest: pd.DataFrame,
                                    model_name: str, ckpts: List[str],
                                    registry: StageRegistry,
                                    kind: str = "blur", severity: float = 1.0
                                    ) -> Dict[str, Any]:
    """Does the corrupted frame actually receive lower reliability?

    This is the experiment that decides whether the reliability head means what
    the paper claims it means, independently of any accuracy number.
    """
    key = "robust/relresponse/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "reliability_response.json")
    if registry.is_done(key) and os.path.exists(out):
        log("SKIP  " + key)
        import json
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)

    banner("RELIABILITY RESPONSE TO A CORRUPTED VIEW  |  %s" % model_name)
    te = test_frame(manifest)
    multi = te[te["n_frames"] > 1].reset_index(drop=True)
    collect = ("frame_reliability", "U", "alpha")

    _lc, _yc, ex_clean = predict_with_checkpoints(
        cfg, multi, model_name, ckpts[:1], None, collect=collect)
    iv = Intervention(corrupt_kind=kind, corrupt_severity=severity,
                      corrupt_n_frames=1, corrupt_frame_idx=0, seed=11)
    _ld, _yd, ex_dirty = predict_with_checkpoints(
        cfg, multi, model_name, ckpts[:1], iv, collect=collect)

    res: Dict[str, Any] = {"model": model_name, "corruption": kind,
                           "severity": severity, "n_patients": int(len(multi))}
    if "frame_reliability" in ex_clean and ex_clean["frame_reliability"].size:
        rc = ex_clean["frame_reliability"]
        rd = ex_dirty["frame_reliability"]
        res["reliability_frame0_clean"] = float(np.mean(rc[:, 0]))
        res["reliability_frame0_corrupted"] = float(np.mean(rd[:, 0]))
        res["reliability_others_clean"] = float(np.mean(rc[:, 1:]))
        res["reliability_others_corrupted"] = float(np.mean(rd[:, 1:]))
        res["reliability_drop_on_corrupted"] = float(np.mean(rc[:, 0] - rd[:, 0]))
        res["reliability_drop_on_clean_peers"] = float(np.mean(rc[:, 1:] - rd[:, 1:]))
    if "U" in ex_clean and ex_clean["U"].size:
        uc, ud = ex_clean["U"], ex_dirty["U"]
        res["uncertainty_frame0_clean"] = float(np.mean(uc[:, 0]))
        res["uncertainty_frame0_corrupted"] = float(np.mean(ud[:, 0]))
        res["uncertainty_rise_on_corrupted"] = float(np.mean(ud[:, 0] - uc[:, 0]))
    if "alpha" in ex_clean and ex_clean["alpha"].size:
        res["attention_frame0_clean"] = float(np.mean(ex_clean["alpha"][:, 0]))
        res["attention_frame0_corrupted"] = float(np.mean(ex_dirty["alpha"][:, 0]))

    for k, v in res.items():
        if isinstance(v, float):
            log("  %-38s %.5f" % (k, v))
    save_json(res, out)
    registry.mark_done(key, {"json": out})
    return res


# --------------------------------------------------------------------------- #
def permutation_invariance(cfg: Config, manifest: pd.DataFrame, model_name: str,
                           ckpts: List[str], registry: StageRegistry,
                           n_repeats: int = 3) -> Dict[str, float]:
    """Patient predictions must not depend on the arbitrary order of frames."""
    key = "robust/perm/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "permutation_invariance.json")
    if registry.is_done(key) and os.path.exists(out):
        log("SKIP  " + key)
        import json
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)

    te = test_frame(manifest)
    multi = te[te["n_frames"] > 1].reset_index(drop=True)
    base, _y, _e = predict_with_checkpoints(cfg, multi, model_name, ckpts[:1], None)
    deltas = []
    for r in range(n_repeats):
        p, _y2, _e2 = predict_with_checkpoints(
            cfg, multi, model_name, ckpts[:1],
            Intervention(permute_frame_order=True, seed=100 + r))
        deltas.append(np.abs(sigmoid(p) - sigmoid(base)))
    d = np.concatenate(deltas)
    res = {"model": model_name, "mean_abs_delta_p": float(d.mean()),
           "max_abs_delta_p": float(d.max()),
           "p95_abs_delta_p": float(np.percentile(d, 95)),
           "n": int(len(multi))}
    log("permutation invariance %s: mean |dp| = %.2e, max = %.2e"
        % (model_name, res["mean_abs_delta_p"], res["max_abs_delta_p"]))
    save_json(res, out)
    registry.mark_done(key, {"json": out})
    return res


# --------------------------------------------------------------------------- #
def frame_removal_study(cfg: Config, manifest: pd.DataFrame, model_name: str,
                        ckpts: List[str], registry: StageRegistry) -> pd.DataFrame:
    """Remove the most / least reliable frame and measure the prediction shift.

    If reliability is meaningful, deleting the most reliable view must move the
    prediction more than deleting the least reliable one.
    """
    key = "robust/framedrop/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "frame_removal.csv")
    cached = _cached(registry, key, out)
    if cached is not None:
        return cached

    banner("FRAME REMOVAL  |  %s" % model_name)
    te = test_frame(manifest)
    multi = te[te["n_frames"] > 1].reset_index(drop=True)
    base_logit, labels, ex = predict_with_checkpoints(
        cfg, multi, model_name, ckpts[:1], None,
        collect=("frame_reliability", "alpha"))

    score = ex.get("frame_reliability")
    score_name = "reliability"
    if score is None or not getattr(score, "size", 0):
        score = ex.get("alpha")
        score_name = "attention"
    if score is None or not getattr(score, "size", 0):
        log("  model exposes neither reliability nor attention -- using random order")
        score = np.random.RandomState(0).rand(len(multi), int(multi["n_frames"].max()))

    valid_n = multi["n_frames"].values
    masked = np.where(np.arange(score.shape[1])[None, :] < valid_n[:, None],
                      score, -np.inf)
    hi = np.argmax(masked, axis=1)
    lo = np.argmin(np.where(np.isfinite(masked), masked, np.inf), axis=1)
    rng = np.random.RandomState(3)
    rnd = np.array([rng.randint(0, max(int(n), 1)) for n in valid_n])

    base_p = sigmoid(base_logit)
    rows = []
    for label, idx in (("highest_" + score_name, hi),
                       ("lowest_" + score_name, lo),
                       ("random", rnd)):
        deltas, aucs = [], []
        # Interventions are per-patient, so evaluate group-by-group on the index.
        for j in np.unique(idx):
            sub = multi[idx == j].reset_index(drop=True)
            if not len(sub):
                continue
            lg, _y, _e = predict_with_checkpoints(
                cfg, sub, model_name, ckpts[:1],
                Intervention(drop_frame_idx=int(j)))
            deltas.append(np.abs(sigmoid(lg) - base_p[idx == j]))
            aucs.append((sigmoid(lg), _y))
        d = np.concatenate(deltas) if deltas else np.zeros(1)
        p_all = np.concatenate([a for a, _b in aucs])
        y_all = np.concatenate([b for _a, b in aucs])
        m = all_metrics(y_all, p_all)
        rows.append({"removed": label, "mean_abs_delta_p": float(d.mean()),
                     "median_abs_delta_p": float(np.median(d)),
                     "roc_auc_after": m["roc_auc"]})
        log("  drop %-24s mean |dp| = %.4f | ROC-AUC after = %.4f"
            % (label, rows[-1]["mean_abs_delta_p"], m["roc_auc"]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    registry.mark_done(key, {"csv": out})
    return df
