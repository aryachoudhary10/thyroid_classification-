"""Verified paths into kaggle_jobs/ and small helpers shared by every figure.

Every path below was checked to exist before being written here (see the audit
commands run at the start of this task). If a path goes missing later, loading
fails loudly rather than silently falling back to a different file -- a wrong
substitution is exactly the kind of error this project has hit before.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KJ = os.path.join(ROOT, "kaggle_jobs")
sys.path.insert(0, ROOT)
from src.eval.metrics import all_metrics, bootstrap_ci, delong_test  # noqa: E402

# --------------------------------------------------------------------------- #
# ResNet-50, hires namespace -- the controlled comparison against RCAF
HIRES = os.path.join(KJ, "final_out", "results", "hires")
RESNET50 = {
    "rcaf": os.path.join(HIRES, "rcaf", "test_predictions.csv"),
    "mr_mil": os.path.join(HIRES, "mr_mil", "test_predictions.csv"),
    "der_mil": os.path.join(HIRES, "der_mil", "test_predictions.csv"),
}
RESNET50_OOF = {
    "rcaf": os.path.join(HIRES, "rcaf", "oof.csv"),
    "mr_mil": os.path.join(HIRES, "mr_mil", "oof.csv"),
    "der_mil": os.path.join(HIRES, "der_mil", "oof.csv"),
}
RESNET50_CAL = {
    "rcaf": os.path.join(HIRES, "rcaf", "calibration_table.csv"),
    "mr_mil": os.path.join(HIRES, "mr_mil", "calibration_table.csv"),
    "der_mil": os.path.join(HIRES, "der_mil", "calibration_table.csv"),
}
TEST_COMPARISON_RESNET50 = os.path.join(HIRES, "_tables", "test_comparison.csv")
HEAD_TO_HEAD_RESNET50 = os.path.join(HIRES, "_tables", "head_to_head.csv")
COHORT_STATS = os.path.join(HIRES, "cohort_stats.csv")
MANIFEST = os.path.join(HIRES, "manifest.csv")
SHORTCUT = os.path.join(HIRES, "shortcut.csv")

RELIABILITY_TOKENS = {
    "der_mil": os.path.join(KJ, "train", "out", "results", "hires", "der_mil",
                            "reliability_influence_tokens.csv"),
    "mr_mil": os.path.join(KJ, "train", "out", "results", "hires", "mr_mil",
                           "reliability_influence_tokens.csv"),
}
RELIABILITY_JSON = {
    "der_mil": os.path.join(KJ, "train", "out", "results", "hires", "der_mil",
                            "reliability_influence.json"),
    "mr_mil": os.path.join(KJ, "train", "out", "results", "hires", "mr_mil",
                           "reliability_influence.json"),
}
EVIDENCE_ABLATION = os.path.join(HIRES, "der_mil", "evidence_ablation.csv")
FRAME_REMOVAL = os.path.join(HIRES, "der_mil", "frame_removal.csv")
MASK_QUALITY = os.path.join(HIRES, "der_mil", "mask_quality.csv")
PERMUTATION = os.path.join(HIRES, "der_mil", "permutation_invariance.json")

# TN5000, ResNet-50
TN5000_RESNET50 = {
    "rcaf": os.path.join(HIRES, "rcaf", "tn5000_results.csv"),
    "mr_mil": os.path.join(HIRES, "mr_mil", "tn5000_results.csv"),
    "der_mil": os.path.join(HIRES, "der_mil", "tn5000_results.csv"),
}
TN5000_LABELFREE = {
    "der_mil": os.path.join(KJ, "report_out", "report",
                            "hires__der_mil__tn5000_labelfree.csv"),
    "mr_mil": os.path.join(KJ, "report_out", "report",
                           "hires__mr_mil__tn5000_labelfree.csv"),
}
TN5000_ARM_LADDER = os.path.join(KJ, "report_out", "report",
                                 "_tables__tn5000_arm_ladder.csv")
TN5000_PRED = {
    ("der_mil", "bbox"): os.path.join(KJ, "report_out", "report",
                                      "hires__der_mil__tn5000_predictions_bbox.csv"),
    ("der_mil", "unet"): os.path.join(KJ, "report_out", "report",
                                      "hires__der_mil__tn5000_predictions_unet.csv"),
    ("mr_mil", "bbox"): os.path.join(KJ, "report_out", "report",
                                     "hires__mr_mil__tn5000_predictions_bbox.csv"),
    ("mr_mil", "unet"): os.path.join(KJ, "report_out", "report",
                                     "hires__mr_mil__tn5000_predictions_unet.csv"),
}

# ConvNeXt-Tiny ablation ladder (lesion_mil -> mr_mil -> der_mil)
CONVNEXT_ABLATION = os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                                 "_tables", "ablation_convnexttiny.csv")
CONVNEXT_PRED = {
    "lesion_mil": os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                               "lesion_mil", "test_predictions.csv"),
    "mr_mil": os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                           "mr_mil", "test_predictions.csv"),
    "der_mil": os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                            "der_mil", "test_predictions.csv"),
}
CONVNEXT_TN5000 = os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                               "der_mil", "tn5000_results.csv")
CONVNEXT_TN5000_PRED = {
    "bbox": os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                         "der_mil", "tn5000_predictions_bbox.csv"),
    "pixel": os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                          "der_mil", "tn5000_predictions_pixel.csv"),
}
TN5000_MANIFEST = os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                               "tn5000_manifest.csv")
CONVNEXT_OOF = {
    m: os.path.join(KJ, "convnext_out", "results", "convnexttiny", m, "oof.csv")
    for m in ("lesion_mil", "mr_mil", "der_mil")
}
CONVNEXT_TN5000_ALL = {
    m: os.path.join(KJ, "convnext_out", "results", "convnexttiny", m, "tn5000_results.csv")
    for m in ("mr_mil", "der_mil")
}
CONVNEXT_TN5000_PRED_ALL = {
    (m, variant): os.path.join(KJ, "convnext_out", "results", "convnexttiny", m,
                               "tn5000_predictions_%s.csv" % variant)
    for m in ("mr_mil", "der_mil") for variant in ("bbox", "pixel")
}
CONVNEXT_EVIDENCE_ABLATION = os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                                          "der_mil", "evidence_ablation.csv")
CONVNEXT_RELIABILITY_JSON = {
    m: os.path.join(KJ, "convnext_out", "results", "convnexttiny", m,
                    "reliability_influence.json")
    for m in ("mr_mil", "der_mil")
}
CONVNEXT_RELIABILITY_TOKENS = {
    m: os.path.join(KJ, "convnext_out", "results", "convnexttiny", m,
                    "reliability_influence_tokens.csv")
    for m in ("mr_mil", "der_mil")
}
CONVNEXT_MASK_QUALITY = os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                                     "der_mil", "mask_quality.csv")
CONVNEXT_FRAME_REMOVAL = os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                                      "der_mil", "frame_removal.csv")
CONVNEXT_PERMUTATION = os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                                    "der_mil", "permutation_invariance.json")
CONVNEXT_SHORTCUT = os.path.join(KJ, "convnext_out", "results", "convnexttiny",
                                 "shortcut.csv")
CONVNEXT_CAL = {
    m: os.path.join(KJ, "convnext_out", "results", "convnexttiny", m,
                    "calibration_table.csv")
    for m in ("lesion_mil", "mr_mil", "der_mil")
}

# Reliability-fusion probe: mlp (default, above) vs linear (constrained form)
DER_MIL_LINEAR_PRED = os.path.join(KJ, "linrel_status", "status",
                                   "convnextlinrel__der_mil__test_predictions.csv")
DER_MIL_LINEAR_COEFFS = os.path.join(KJ, "linrel_status", "status",
                                     "convnextlinrel__reliability_coefficients.json")

# Checkpoints for the computational-cost figure (real state dicts, ResNet-50)
CKPT = {
    "rcaf": os.path.join(KJ, "rcaf_ckpt", "checkpoints", "main", "rcaf",
                         "final", "best.pt"),
    "mr_mil": os.path.join(KJ, "train", "out_hires", "checkpoints", "hires",
                           "mr_mil", "final", "best.pt"),
    "der_mil": os.path.join(KJ, "train", "out_hires", "checkpoints", "hires",
                            "der_mil", "final", "best.pt"),
}


# --------------------------------------------------------------------------- #
def require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "expected real result file is missing: %s\n"
            "This figure needs it -- either the file moved, or it was never "
            "produced. Not substituting a different file silently." % path)
    return path


def norm_pid(s) -> str:
    """Patient ids are inconsistently zero-padded across runs (e.g. '00000126'
    in one CSV, '126' in another from the same session). Normalise to int-string
    before any join or pairing."""
    try:
        return str(int(str(s)))
    except (TypeError, ValueError):
        return str(s)


def load_predictions(path: str, pcol: str = "p_raw") -> pd.DataFrame:
    df = pd.read_csv(require(path), dtype={"patient_id": str})
    df["patient_id"] = df["patient_id"].map(norm_pid)
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={pcol: "p"}) if pcol != "p" else df
    if "p" not in df.columns:
        raise KeyError("%s has no column '%s'" % (path, pcol))
    return df[["patient_id", "label", "p"]].dropna()


def paired(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join two prediction frames on patient_id, asserting labels agree."""
    m = df_a.merge(df_b, on="patient_id", suffixes=("_a", "_b"))
    if not (m["label_a"] == m["label_b"]).all():
        raise ValueError("label mismatch on shared patients -- do not pair these")
    return m


def metrics_from_predictions(df: pd.DataFrame) -> Dict[str, float]:
    return all_metrics(df["label"].to_numpy(), df["p"].to_numpy())


def ci_from_predictions(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0
                        ) -> Dict[str, Tuple[float, float, float]]:
    return bootstrap_ci(df["label"].to_numpy(), df["p"].to_numpy(), n_boot,
                        0.5, seed=seed)


def delong(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, float]:
    m = paired(df_a, df_b)
    return delong_test(m["label_a"].to_numpy(), m["p_a"].to_numpy(),
                       m["p_b"].to_numpy())


def load_json(path: str) -> dict:
    return json.load(open(require(path), encoding="utf-8"))
