"""Paper-ready comparison tables and statistical head-to-heads."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..eval.calibration import sigmoid
from ..eval.metrics import (all_metrics, bootstrap_auc_difference, bootstrap_ci,
                            delong_test, paired_wilcoxon)
from ..utils.common import load_json, log, save_json


# --------------------------------------------------------------------------- #
def _oof_path(cfg: Config, model: str) -> str:
    return os.path.join(cfg.run.results_root, cfg.run.run_name, model, "oof.csv")


def _test_path(cfg: Config, model: str) -> str:
    return os.path.join(cfg.run.results_root, cfg.run.run_name, model,
                        "test_predictions.csv")


def load_oof(cfg: Config, model: str) -> Optional[pd.DataFrame]:
    p = _oof_path(cfg, model)
    return pd.read_csv(p) if os.path.exists(p) else None


def load_test(cfg: Config, model: str) -> Optional[pd.DataFrame]:
    p = _test_path(cfg, model)
    return pd.read_csv(p) if os.path.exists(p) else None


# --------------------------------------------------------------------------- #
def per_fold_auc(oof: pd.DataFrame) -> List[float]:
    return [all_metrics(g["label"].values, sigmoid(g["logit"].values))["roc_auc"]
            for _k, g in oof.groupby("fold")]


def development_table(cfg: Config, models: Sequence[str]) -> pd.DataFrame:
    """Table 6 analogue: pooled OOF metrics with per-fold mean +/- std."""
    rows = []
    for m in models:
        oof = load_oof(cfg, m)
        if oof is None:
            continue
        p = sigmoid(oof["logit"].values)
        y = oof["label"].values.astype(int)
        met = all_metrics(y, p, 0.5)
        folds = per_fold_auc(oof)
        rows.append({
            "model": m,
            "roc_auc_pooled": met["roc_auc"],
            "roc_auc_fold_mean": float(np.mean(folds)),
            "roc_auc_fold_std": float(np.std(folds)),
            "pr_auc": met["pr_auc"], "sensitivity": met["sensitivity"],
            "specificity": met["specificity"], "precision": met["precision"],
            "f1": met["f1"], "brier": met["brier"], "n": met["n"],
        })
    return pd.DataFrame(rows).sort_values("roc_auc_pooled", ascending=False)


def test_table(cfg: Config, models: Sequence[str], prob_col: str = "p_raw",
               thr: float = 0.5, with_ci: bool = True) -> pd.DataFrame:
    """Independent-test metrics with bootstrap confidence intervals."""
    rows = []
    for m in models:
        te = load_test(cfg, m)
        if te is None:
            continue
        y = te["label"].values.astype(int)
        p = te[prob_col].values
        row: Dict[str, Any] = {"model": m, **all_metrics(y, p, thr)}
        if with_ci:
            ci = bootstrap_ci(y, p, cfg.eval.bootstrap_test, thr, seed=cfg.run.seed)
            for k, (pt, lo, hi) in ci.items():
                row[k + "_ci"] = "[%.3f, %.3f]" % (lo, hi)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


# --------------------------------------------------------------------------- #
def head_to_head(cfg: Config, proposed: str, baselines: Sequence[str]) -> pd.DataFrame:
    """Per-fold Wilcoxon + bootstrapped OOF AUC gap + DeLong on the test set."""
    rows = []
    oof_p = load_oof(cfg, proposed)
    te_p = load_test(cfg, proposed)
    if oof_p is None:
        return pd.DataFrame()
    folds_p = per_fold_auc(oof_p)

    for b in baselines:
        if b == proposed:
            continue
        oof_b = load_oof(cfg, b)
        if oof_b is None:
            continue
        folds_b = per_fold_auc(oof_b)
        w = paired_wilcoxon(folds_p, folds_b)

        merged = oof_p.merge(oof_b, on="patient_id", suffixes=("_p", "_b"))
        boot = bootstrap_auc_difference(
            merged["label_p"].values, sigmoid(merged["logit_p"].values),
            sigmoid(merged["logit_b"].values), n_boot=1000, seed=cfg.run.seed)

        row = {"baseline": b,
               "oof_auc_proposed": float(np.mean(folds_p)),
               "oof_auc_baseline": float(np.mean(folds_b)),
               "wilcoxon_W": w["W"], "wilcoxon_p": w["p_value"],
               "cohens_d": w["cohens_d"], "folds_won": w["wins"],
               "boot_mean_diff": boot["mean_diff"],
               "boot_ci": "[%.4f, %.4f]" % (boot["lo"], boot["hi"])}

        te_b = load_test(cfg, b)
        if te_p is not None and te_b is not None:
            mt = te_p.merge(te_b, on="patient_id", suffixes=("_p", "_b"))
            if len(mt):
                dl = delong_test(mt["label_p"].values, mt["p_raw_p"].values,
                                 mt["p_raw_b"].values)
                row.update({"test_auc_proposed": dl["auc1"],
                            "test_auc_baseline": dl["auc2"],
                            "delong_z": dl["z"], "delong_p": dl["p_value"]})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def threshold_table(cfg: Config, model: str) -> pd.DataFrame:
    """Table 10 analogue: behaviour at each development-selected threshold."""
    te = load_test(cfg, model)
    cal = load_json(os.path.join(cfg.run.results_root, cfg.run.run_name, model,
                                 "calibration.json"), {})
    if te is None or not cal:
        return pd.DataFrame()
    y = te["label"].values.astype(int)
    rows = []
    rows.append({"operating_point": "Raw @ 0.5", **all_metrics(y, te["p_raw"].values, 0.5)})
    for name, thr in cal.get("thresholds", {}).items():
        if name == "raw":
            continue
        rows.append({"operating_point": "Calibrated @ " + name,
                     **all_metrics(y, te["p_cal"].values, float(thr))})
    if "p_shift" in te.columns:
        rows.append({"operating_point": "Prior-shifted @ 0.5",
                     **all_metrics(y, te["p_shift"].values, 0.5)})
    cols = ["operating_point", "threshold", "sensitivity", "specificity",
            "precision", "f1", "roc_auc", "brier", "tp", "fp", "tn", "fn"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def tirads_comparison(cfg: Config, manifest: pd.DataFrame, model: str,
                      cutoff: int = 4) -> pd.DataFrame:
    """Model predictions versus TIRADS >= cutoff on the same test cohort."""
    te = load_test(cfg, model)
    if te is None or "tirads" not in manifest.columns:
        return pd.DataFrame()
    m = manifest[manifest["split"] == "test"][["patient_id", "tirads"]].copy()
    m["patient_id"] = m["patient_id"].astype(str)
    te["patient_id"] = te["patient_id"].astype(str)
    j = te.merge(m, on="patient_id", how="left").dropna(subset=["tirads"])
    if j.empty:
        return pd.DataFrame()
    y = j["label"].values.astype(int)
    rows = [{"method": "TIRADS >= %d" % cutoff,
             **all_metrics(y, (j["tirads"].values >= cutoff).astype(float), 0.5)},
            {"method": model + " @ 0.5", **all_metrics(y, j["p_raw"].values, 0.5)}]
    keep = ["method", "sensitivity", "specificity", "precision", "f1", "n"]
    return pd.DataFrame(rows)[keep]


# --------------------------------------------------------------------------- #
def write_all_tables(cfg: Config, models: Sequence[str], proposed: str,
                     manifest: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    out_dir = os.path.join(cfg.run.results_root, cfg.run.run_name, "_tables")
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}

    def dump(name: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        p = os.path.join(out_dir, name + ".csv")
        df.to_csv(p, index=False)
        written[name] = p
        log("\n== " + name + " ==\n" + df.to_string(index=False))

    dump("development_comparison", development_table(cfg, models))
    dump("test_comparison", test_table(cfg, models))
    dump("head_to_head", head_to_head(cfg, proposed, models))
    dump("thresholds_" + proposed, threshold_table(cfg, proposed))
    if manifest is not None:
        dump("tirads_" + proposed, tirads_comparison(cfg, manifest, proposed))
    save_json(written, os.path.join(out_dir, "index.json"))
    return written
