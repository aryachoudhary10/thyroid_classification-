"""Leakage-free grouped stratified cross-validation at the patient level.

The manifest already holds one row per patient, so a stratified K-fold over
manifest rows is group-safe by construction. We still route through explicit
group bookkeeping and assert the invariant, because a silent regression here
invalidates every number the project produces.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..utils.common import log


def make_folds(manifest: pd.DataFrame,
               n_folds: int = 5,
               seed: int = 1337) -> pd.DataFrame:
    """Add a ``fold`` column to the development cohort (test rows get -1)."""
    man = manifest.copy()
    man["fold"] = -1
    dev = man[man["split"] == "dev"]
    if dev.empty:
        raise ValueError("development cohort is empty")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = dev["label"].to_numpy()
    for k, (_tr, va) in enumerate(skf.split(np.zeros(len(dev)), y)):
        man.loc[dev.index[va], "fold"] = k

    counts = man[man["split"] == "dev"].groupby("fold")["label"].agg(["size", "mean"])
    log("folds:\n" + counts.to_string())
    validate_folds(man)
    return man


def validate_folds(manifest: pd.DataFrame) -> None:
    dev = manifest[manifest["split"] == "dev"]
    if (dev["fold"] < 0).any():
        raise AssertionError("some development patients were not assigned a fold")
    test = manifest[manifest["split"] == "test"]
    if len(test) and (test["fold"] >= 0).any():
        raise AssertionError("test patients must never receive a fold id")

    # A patient id must live in exactly one fold.
    per_pid = dev.groupby("patient_id")["fold"].nunique()
    if (per_pid > 1).any():
        raise AssertionError("patient appears in multiple folds -- grouped split violated")

    overlap = set(dev["patient_id"]) & set(test["patient_id"])
    if overlap:
        raise AssertionError("dev/test patient overlap: " + str(list(overlap)[:5]))
    log("split validation: OK (dev=%d, test=%d, folds=%d)"
        % (len(dev), len(test), dev["fold"].nunique()))


def fold_frames(manifest: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) for one development fold."""
    dev = manifest[manifest["split"] == "dev"]
    tr = dev[dev["fold"] != fold].reset_index(drop=True)
    va = dev[dev["fold"] == fold].reset_index(drop=True)
    assert not set(tr["patient_id"]) & set(va["patient_id"]), "fold leakage"
    return tr, va


def test_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[manifest["split"] == "test"].reset_index(drop=True)


def class_pos_weight(train_df: pd.DataFrame) -> float:
    """pos_weight for BCEWithLogits, matching the paper's class weighting."""
    n_pos = float((train_df["label"] == 1).sum())
    n_neg = float((train_df["label"] == 0).sum())
    if n_pos <= 0:
        return 1.0
    return max(n_neg / n_pos, 1e-3)


def prevalence(df: pd.DataFrame) -> float:
    return float((df["label"] == 1).mean())
