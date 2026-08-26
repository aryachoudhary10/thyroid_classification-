"""Leakage-free development protocol and single-shot test evaluation.

Order of operations, enforced by the stage registry:

    1. train fold k on dev-minus-fold-k, validate on fold k      (k = 0..4)
    2. concatenate held-out predictions            -> OOF
    3. fit temperature + select thresholds ON OOF ONLY
    4. build the final predictor
    5. evaluate ONCE on the untouched test cohort with the frozen calibration

Nothing in steps 1-3 ever sees a test patient. Step 5 runs exactly once per
model and its outputs are cached, so re-running the notebook cannot silently
turn the test set into a validation set.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.dataset import Intervention, PatientBagDataset
from ..data.splits import class_pos_weight, fold_frames, prevalence, test_frame
from ..eval.calibration import (CalibrationBundle, calibration_report,
                                fit_temperature, apply_temperature, sigmoid,
                                prior_shift_logits)
from ..eval.metrics import (all_metrics, bootstrap_ci, sensitivity_constrained_threshold,
                            youden_threshold)
from ..models.factory import build_model
from ..utils.checkpoint import CheckpointManager, StageRegistry
from ..utils.common import banner, load_json, log, save_json
from .trainer import Trainer, predict_dataset


# --------------------------------------------------------------------------- #
def make_dataset(cfg: Config, frame: pd.DataFrame, reqs: Dict[str, Any],
                 train: bool, intervention: Optional[Intervention] = None
                 ) -> PatientBagDataset:
    return PatientBagDataset(
        frame, cfg.data, train=train,
        need_regions=bool(reqs["need_regions"]),
        region_res=int(reqs["region_res"]),
        regions=tuple(reqs["regions"]),
        intervention=intervention)


def run_name_for(cfg: Config, model_name: str, suffix: str) -> str:
    return "%s/%s/%s" % (cfg.run.run_name, model_name, suffix)


# --------------------------------------------------------------------------- #
def train_fold(cfg: Config, manifest: pd.DataFrame, model_name: str,
               fold: int, registry: StageRegistry) -> Dict[str, Any]:
    key = "cv/%s/%s/fold%d" % (cfg.run.run_name, model_name, fold)
    rn = run_name_for(cfg, model_name, "fold%d" % fold)
    out_path = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                            "oof_fold%d.csv" % fold)

    if registry.is_done(key) and os.path.exists(out_path):
        log("SKIP  " + key)
        return {"oof_csv": out_path}

    tr_df, va_df = fold_frames(manifest, fold)
    if cfg.run.debug_subset:
        tr_df = tr_df.head(cfg.run.debug_subset)
        va_df = va_df.head(max(cfg.run.debug_subset // 4, 8))

    model, reqs = build_model(cfg, model_name)
    train_ds = make_dataset(cfg, tr_df, reqs, train=True)
    val_ds = make_dataset(cfg, va_df, reqs, train=False)

    trainer = Trainer(cfg, model, train_ds, val_ds, rn,
                      pos_weight=class_pos_weight(tr_df))
    summary = trainer.fit()

    logits, labels, _ = trainer.predict(val_ds, use_best=True)
    df = pd.DataFrame({"patient_id": va_df["patient_id"].astype(str).values,
                       "label": labels.astype(int), "logit": logits,
                       "fold": fold})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    fold_auc = all_metrics(labels, sigmoid(logits))["roc_auc"]
    log("fold %d %s: held-out ROC-AUC = %.4f" % (fold, model_name, fold_auc))

    registry.mark_done(key, {"oof_csv": out_path, "roc_auc": fold_auc,
                             "best_val": summary["best_metric"],
                             "ckpt": summary["best_path"]})
    del trainer, model, train_ds, val_ds
    torch.cuda.empty_cache()
    return {"oof_csv": out_path, "roc_auc": fold_auc}


def run_cross_validation(cfg: Config, manifest: pd.DataFrame, model_name: str,
                         registry: StageRegistry) -> pd.DataFrame:
    banner("CROSS-VALIDATION  |  %s" % model_name)
    frames = []
    for k in range(cfg.eval.n_folds):
        art = train_fold(cfg, manifest, model_name, k, registry)
        frames.append(pd.read_csv(art["oof_csv"]))
    oof = pd.concat(frames, ignore_index=True)
    oof["patient_id"] = oof["patient_id"].astype(str)

    path = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name, "oof.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    oof.to_csv(path, index=False)

    per_fold = [all_metrics(g["label"].values, sigmoid(g["logit"].values))["roc_auc"]
                for _k, g in oof.groupby("fold")]
    pooled = all_metrics(oof["label"].values, sigmoid(oof["logit"].values))
    log("%s: pooled OOF ROC-AUC %.4f | per-fold mean %.4f +/- %.4f"
        % (model_name, pooled["roc_auc"], float(np.mean(per_fold)), float(np.std(per_fold))))
    save_json({"per_fold_roc_auc": per_fold, "pooled": pooled},
              os.path.join(os.path.dirname(path), "oof_summary.json"))
    return oof


# --------------------------------------------------------------------------- #
def fit_calibration(cfg: Config, oof: pd.DataFrame) -> CalibrationBundle:
    """Temperature + operating thresholds, estimated on OOF predictions only."""
    y = oof["label"].values.astype(int)
    z = oof["logit"].values.astype(float)

    T = fit_temperature(z, y)
    p_cal = apply_temperature(z, T)

    thresholds = {
        "raw": 0.5,
        "youden": youden_threshold(y, p_cal),
        "sens%d" % int(100 * cfg.eval.sensitivity_target):
            sensitivity_constrained_threshold(y, p_cal, cfg.eval.sensitivity_target),
    }
    bundle = CalibrationBundle(T, thresholds, prevalence(oof))
    log("calibration: T = %.4f | thresholds %s" % (T, {k: round(v, 4)
                                                       for k, v in thresholds.items()}))
    if T <= 0.06 or T >= 19.0:
        log("  WARNING temperature hit its clamp -- the out-of-fold logits are "
            "degenerate (too few patients, or perfectly separable). Treat the "
            "calibrated probabilities as unreliable.")
    return bundle


def calibration_table(oof: pd.DataFrame, test: pd.DataFrame,
                      bundle: CalibrationBundle, n_bins: int = 15) -> pd.DataFrame:
    """Reproduces Table 9: ECE / Brier before and after each correction."""
    rows = []
    y_d, z_d = oof["label"].values.astype(int), oof["logit"].values.astype(float)
    rows.append({"cohort": "Dev (OOF)", "condition": "Before temp. scaling",
                 **calibration_report(y_d, sigmoid(z_d), n_bins)})
    rows.append({"cohort": "Dev (OOF)",
                 "condition": "After temp. scaling (T=%.3f)" % bundle.temperature,
                 **calibration_report(y_d, apply_temperature(z_d, bundle.temperature), n_bins)})

    if test is not None and len(test):
        y_t, z_t = test["label"].values.astype(int), test["logit"].values.astype(float)
        rows.append({"cohort": "Test set", "condition": "Temp. scaling only",
                     **calibration_report(y_t, apply_temperature(z_t, bundle.temperature), n_bins)})
        p_test = float(y_t.mean())
        shifted = sigmoid(prior_shift_logits(z_t, bundle.temperature, bundle.p_dev, p_test))
        rows.append({"cohort": "Test set", "condition": "+ Prior-shift correction",
                     **calibration_report(y_t, shifted, n_bins)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def build_final_predictor(cfg: Config, manifest: pd.DataFrame, model_name: str,
                          registry: StageRegistry, mode: str = "ensemble"
                          ) -> List[str]:
    """Return the checkpoint path(s) that constitute the final model.

    ``ensemble`` (default) averages the logits of the five fold checkpoints.
    It costs no additional GPU time, uses only development data, and keeps the
    leakage-free guarantee intact.
    ``refit`` retrains once on 100% of the development cohort for the median
    number of epochs the folds needed -- closer to the wording in the paper,
    but one extra full training run per model.
    ``best_fold`` picks the single fold checkpoint with the highest validation
    ROC-AUC; used for external domain adaptation where a single set of weights
    is required.
    """
    paths = []
    for k in range(cfg.eval.n_folds):
        art = registry.artifacts("cv/%s/%s/fold%d" % (cfg.run.run_name, model_name, k))
        if art.get("ckpt") and os.path.exists(art["ckpt"]):
            paths.append(art["ckpt"])

    if mode == "ensemble":
        return paths
    if mode == "best_fold":
        scored = []
        for k in range(cfg.eval.n_folds):
            art = registry.artifacts("cv/%s/%s/fold%d" % (cfg.run.run_name, model_name, k))
            if art.get("ckpt"):
                scored.append((art.get("best_val", -1), art["ckpt"]))
        scored.sort(reverse=True)
        return [scored[0][1]] if scored else paths[:1]

    if mode == "refit":
        key = "refit/%s/%s" % (cfg.run.run_name, model_name)
        rn = run_name_for(cfg, model_name, "final")
        cm = CheckpointManager(cfg.run.ckpt_root, rn)
        if registry.is_done(key) and cm.has_best():
            return [cm.best_path]
        if manifest is None or manifest.empty:
            log("  refit checkpoint not available yet -- falling back to the "
                "fold ensemble for this call")
            return paths
        dev = manifest[manifest["split"] == "dev"].reset_index(drop=True)
        holdout = dev.sample(frac=0.10, random_state=cfg.run.seed)
        tr = dev.drop(holdout.index).reset_index(drop=True)
        model, reqs = build_model(cfg, model_name)
        trainer = Trainer(cfg, model, make_dataset(cfg, tr, reqs, True),
                          make_dataset(cfg, holdout.reset_index(drop=True), reqs, False),
                          rn, pos_weight=class_pos_weight(tr))
        trainer.fit()
        registry.mark_done(key, {"ckpt": trainer.ckpt.best_path})
        del trainer, model
        torch.cuda.empty_cache()
        return [cm.best_path]

    raise KeyError("unknown final_mode: " + str(mode))


@torch.no_grad()
def predict_with_checkpoints(cfg: Config, frame: pd.DataFrame, model_name: str,
                             ckpt_paths: List[str],
                             intervention: Optional[Intervention] = None,
                             collect: Tuple[str, ...] = ()
                             ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Average the logits of one or more checkpoints of the same architecture."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reqs = build_model(cfg, model_name)
    ds = make_dataset(cfg, frame, reqs, train=False, intervention=intervention)

    all_logits, labels, extras = [], None, {}
    for path in ckpt_paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        logits, y, ex = predict_dataset(model, ds, cfg, device, collect=collect)
        all_logits.append(logits)
        labels = y
        if not extras:
            extras = ex
    del model, ds
    torch.cuda.empty_cache()
    return np.mean(np.vstack(all_logits), axis=0), labels, extras


# --------------------------------------------------------------------------- #
def evaluate_test(cfg: Config, manifest: pd.DataFrame, model_name: str,
                  bundle: CalibrationBundle, registry: StageRegistry,
                  final_mode: str = "ensemble") -> Dict[str, Any]:
    """Single-shot evaluation on the untouched official test cohort."""
    key = "test/%s/%s" % (cfg.run.run_name, model_name)
    out_dir = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name)
    pred_path = os.path.join(out_dir, "test_predictions.csv")

    if registry.is_done(key) and os.path.exists(pred_path):
        log("SKIP  " + key + "  (test set already consumed for this model)")
        return {"predictions": pred_path, **registry.artifacts(key)}

    banner("SINGLE-SHOT TEST EVALUATION  |  %s" % model_name)
    te = test_frame(manifest)
    ckpts = build_final_predictor(cfg, manifest, model_name, registry, final_mode)
    if not ckpts:
        raise RuntimeError("no fold checkpoints found for " + model_name)
    log("final predictor: %s over %d checkpoint(s)" % (final_mode, len(ckpts)))

    logits, labels, _ = predict_with_checkpoints(cfg, te, model_name, ckpts)
    p_raw = sigmoid(logits)
    p_cal = apply_temperature(logits, bundle.temperature)

    rows = {"patient_id": te["patient_id"].astype(str).values,
            "label": labels.astype(int), "logit": logits,
            "p_raw": p_raw, "p_cal": p_cal}
    p_test = float(labels.mean())
    if cfg.eval.prior_shift_correction:
        rows["p_shift"] = sigmoid(prior_shift_logits(logits, bundle.temperature,
                                                     bundle.p_dev, p_test))
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    results: Dict[str, Any] = {"model": model_name, "n_test": int(len(labels)),
                               "final_mode": final_mode}
    results["raw@0.5"] = all_metrics(labels, p_raw, 0.5)
    for name, thr in bundle.thresholds.items():
        if name == "raw":
            continue
        results["cal@" + name] = all_metrics(labels, p_cal, thr)
    results["ci"] = {k: list(v) for k, v in
                     bootstrap_ci(labels, p_raw, cfg.eval.bootstrap_test, 0.5,
                                  seed=cfg.run.seed).items()}
    save_json(results, os.path.join(out_dir, "test_results.json"))
    log("%s test ROC-AUC = %.4f  (PR-AUC %.4f, F1@0.5 %.4f)"
        % (model_name, results["raw@0.5"]["roc_auc"], results["raw@0.5"]["pr_auc"],
           results["raw@0.5"]["f1"]))

    registry.mark_done(key, {"predictions": pred_path,
                             "roc_auc": results["raw@0.5"]["roc_auc"],
                             "ckpts": ckpts})
    return {"predictions": pred_path, **results}


# --------------------------------------------------------------------------- #
def full_protocol(cfg: Config, manifest: pd.DataFrame, model_name: str,
                  registry: StageRegistry, final_mode: str = "ensemble"
                  ) -> Dict[str, Any]:
    """CV -> calibration -> single-shot test, all resumable."""
    oof = run_cross_validation(cfg, manifest, model_name, registry)
    bundle = fit_calibration(cfg, oof)
    out_dir = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name)
    save_json(bundle.to_dict(), os.path.join(out_dir, "calibration.json"))
    res = evaluate_test(cfg, manifest, model_name, bundle, registry, final_mode)
    return {"oof": oof, "calibration": bundle, "test": res}
