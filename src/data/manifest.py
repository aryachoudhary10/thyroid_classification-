"""Patient-level manifest construction and cohort statistics.

Turns the flat per-image records from ``discovery`` into the bag structure the
paper works with: one patient == one bag of up to Tmax frames sharing a single
histopathology label.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.common import log, save_json


# --------------------------------------------------------------------------- #
def build_patient_manifest(records: pd.DataFrame,
                           t_max: int = 10,
                           truncation: str = "first",
                           seed: int = 0) -> pd.DataFrame:
    """Collapse per-image records into one row per patient.

    ``truncation`` controls what happens if a patient has more than ``t_max``
    frames. The ThyroidXL cohort never triggers it (max is 10); other datasets
    might, so the policy is explicit rather than silent.
    """
    if records.empty:
        raise ValueError("no records to build a manifest from")

    df = records.copy()
    # A patient must have exactly one label; majority-vote and warn otherwise.
    grp = df.groupby("patient_id")
    inconsistent = grp["label"].nunique()
    bad = inconsistent[inconsistent > 1].index.tolist()
    if bad:
        log("manifest: WARNING %d patients have mixed image labels; using majority"
            % len(bad))

    rng = np.random.RandomState(seed)
    rows: List[Dict] = []
    for pid, g in grp:
        g = g.sort_values("image_path")
        if len(g) > t_max:
            if truncation == "random":
                g = g.iloc[rng.permutation(len(g))[:t_max]]
            else:
                g = g.iloc[:t_max]
        label = int(round(float(g["label"].mean())))
        splits = [s for s in g["split"].tolist() if isinstance(s, str)]
        split = max(set(splits), key=splits.count) if splits else None
        rows.append({
            "patient_id": str(pid),
            "label": label,
            "split": split,
            "n_frames": len(g),
            "image_paths": list(g["image_path"]),
            "mask_paths": [m if isinstance(m, str) else None for m in g["mask_path"]],
            "bbox_xmls": [x if isinstance(x, str) else None for x in g["bbox_xml"]],
            "tirads": float(np.nanmax(g["tirads"].values)) if g["tirads"].notna().any() else np.nan,
            "age": float(g["age"].iloc[0]) if g["age"].notna().any() else np.nan,
            "sex": g["sex"].iloc[0] if g["sex"].notna().any() else None,
        })

    man = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    log("manifest: %d patients, %d frames, max frames/patient = %d"
        % (len(man), int(man["n_frames"].sum()), int(man["n_frames"].max())))
    return man


# --------------------------------------------------------------------------- #
def ensure_official_split(man: pd.DataFrame,
                          test_fraction: float = 0.18,
                          seed: int = 1337,
                          stratify: bool = True) -> pd.DataFrame:
    """Guarantee every patient carries a dev/test assignment.

    If the mirror ships the official ThyroidXL split we keep it untouched.
    Otherwise we create a held-out cohort ONCE, at the patient level, and cache
    it in the manifest so it never drifts between runs.
    """
    man = man.copy()
    has_split = man["split"].isin(["dev", "test"]).sum()
    if has_split >= 0.9 * len(man):
        man["split"] = man["split"].fillna("dev")
        man.loc[~man["split"].isin(["dev", "test"]), "split"] = "dev"
        log("manifest: using the split shipped with the dataset "
            "(dev=%d, test=%d)" % ((man['split'] == 'dev').sum(),
                                   (man['split'] == 'test').sum()))
        return man

    log("manifest: no official split found -- creating a frozen patient-level "
        "held-out test cohort (%.0f%%)" % (100 * test_fraction))
    rng = np.random.RandomState(seed)
    assign = pd.Series("dev", index=man.index)
    groups = [man.index[man["label"] == c] for c in (0, 1)] if stratify else [man.index]
    for idx in groups:
        idx = np.array(idx)
        rng.shuffle(idx)
        n_test = int(round(test_fraction * len(idx)))
        assign.loc[idx[:n_test]] = "test"
    man["split"] = assign.values
    return man


# --------------------------------------------------------------------------- #
def cohort_stats(man: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the layout of Table 1 in the paper."""
    out: Dict[str, Dict[str, object]] = {}
    for split in ("dev", "test"):
        s = man[man["split"] == split]
        if s.empty:
            continue
        n = len(s)
        col: Dict[str, object] = {
            "Number of patients": n,
            "Number of images": int(s["n_frames"].sum()),
            "Frames per patient (min-max)": "%d-%d" % (s["n_frames"].min(), s["n_frames"].max()),
            "Mean frames per patient": round(float(s["n_frames"].mean()), 2),
            "Median frames per patient": int(s["n_frames"].median()),
            "Benign patients, n (%)": "%d (%.1f%%)" % ((s["label"] == 0).sum(),
                                                       100 * (s["label"] == 0).mean()),
            "Malignant patients, n (%)": "%d (%.1f%%)" % ((s["label"] == 1).sum(),
                                                          100 * (s["label"] == 1).mean()),
        }
        for t in (1, 2, 3, 4, 5):
            col["TIRADS %d, n" % t] = int((s["tirads"] == t).sum())
        if s["age"].notna().any():
            col["Mean age, years"] = "%.1f +/- %.1f (range %d-%d)" % (
                s["age"].mean(), s["age"].std(), s["age"].min(), s["age"].max())
        out["Development set" if split == "dev" else "Test set"] = col
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
def save_manifest(man: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ser = man.copy()
    for c in ("image_paths", "mask_paths", "bbox_xmls"):
        ser[c] = ser[c].map(lambda v: "|".join("" if x is None else str(x) for x in v))
    ser.to_csv(path, index=False)


def load_manifest(path: str) -> pd.DataFrame:
    man = pd.read_csv(path)
    for c in ("image_paths", "mask_paths", "bbox_xmls"):
        man[c] = man[c].fillna("").map(
            lambda s: [x if x else None for x in str(s).split("|")] if s else [])
    man["patient_id"] = man["patient_id"].astype(str)
    return man


# --------------------------------------------------------------------------- #
def mask_coverage(man: pd.DataFrame) -> Dict[str, float]:
    """How many frames actually have a usable lesion mask."""
    total = pix = box = none = 0
    for _, r in man.iterrows():
        for m, x in zip(r["mask_paths"], r["bbox_xmls"]):
            total += 1
            if m:
                pix += 1
            elif x:
                box += 1
            else:
                none += 1
    return {"frames": total,
            "pixel_mask": pix / max(total, 1),
            "bbox_only": box / max(total, 1),
            "no_mask": none / max(total, 1)}


def assert_no_patient_leakage(man: pd.DataFrame) -> None:
    """Hard guarantee: no patient id appears in more than one split."""
    counts = man.groupby("patient_id")["split"].nunique()
    offenders = counts[counts > 1]
    if len(offenders):
        raise AssertionError("patient-level leakage: %d patients span multiple splits (%s)"
                             % (len(offenders), list(offenders.index[:5])))
    if man["patient_id"].duplicated().any():
        raise AssertionError("duplicate patient rows in manifest")
    log("leakage check: OK -- %d unique patients, no split overlap" % len(man))
