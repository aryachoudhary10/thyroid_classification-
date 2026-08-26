"""Purpose-built ThyroidXL reader.

Generic discovery gets the images, masks, patient ids and train/test split
right, but it cannot parse ``stats/id2info_eng.json`` -- that file is a dict
keyed by patient, not a table, so the heuristics would silently produce garbage
labels. This module reads the dataset's own official files instead.

Verified layout (confirmed by running against the real dataset):

    ThyroidXL/
      train/images   9541 .png     test/images   2094 .png
      train/masks    9541 .png     test/masks    2094 .png
      stats/train_patients.txt  3354 ids
      stats/test_patients.txt    739 ids
      stats/id2info_eng.json    4093 patients

Filenames are ``<patient>_<hash>_<frame>.png``; masks reuse the identical stem.

Label encoding, resolved by brute-forcing every code subset against Table 1 --
``{1, 2}`` was the only one that reproduced both 877/3354 and 353/739:

    1 = Malignant (Histopathology)   708
    2 = Malignant (FNAC)             522
    3 = Benign (FNAC)               2769
    4 = Benign (Histopathology)       94
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.common import log

MALIGNANT_CODES = (1, 2)
BENIGN_CODES = (3, 4)
CONCLUSION_TEXT = {
    1: "Malignant (Histopathology)",
    2: "Malignant (FNAC)",
    3: "Benign (FNAC)",
    4: "Benign (Histopathology)",
}
# gender code 2 == female (3650 patients == the paper's 89.2%)
FEMALE_CODE = 2

# Paper Table 1, used as a hard regression check.
TABLE1 = {
    "dev_patients": 3354, "test_patients": 739,
    "dev_images": 9541, "test_images": 2094,
    "dev_malignant": 877, "test_malignant": 353,
    "dev_tirads": {1: 24, 2: 328, 3: 1090, 4: 1024, 5: 888},
    "test_tirads": {1: 0, 2: 45, 3: 182, 4: 215, 5: 297},
}


# --------------------------------------------------------------------------- #
def find_root(root: str) -> Optional[str]:
    """Locate the ThyroidXL directory anywhere under ``root``."""
    if not root or not os.path.isdir(root):
        return None
    if os.path.isdir(os.path.join(root, "stats")) and \
            os.path.isfile(os.path.join(root, "stats", "train_patients.txt")):
        return root
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            cand = os.path.join(dp, d)
            if os.path.isfile(os.path.join(cand, "stats", "train_patients.txt")):
                return cand
    return None


def looks_like_thyroidxl(root: str) -> bool:
    return find_root(root) is not None


def _read_ids(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _clean_age(value) -> float:
    """The shipped `age` field mixes real ages with birth years (max 2005).

    The paper reports 48.1 +/- 12.7 over 12-94, so the published statistics were
    computed on cleaned data. Anything outside a plausible human range becomes
    NaN rather than being silently averaged in.
    """
    try:
        a = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return a if 1.0 <= a <= 120.0 else float("nan")


# --------------------------------------------------------------------------- #
def build_thyroidxl_manifest(root: str, t_max: int = 10,
                             strict: bool = True) -> pd.DataFrame:
    """One row per patient, built from the dataset's own official files."""
    base = find_root(root)
    if base is None:
        raise FileNotFoundError(
            "no ThyroidXL layout under " + str(root)
            + " (expected stats/train_patients.txt)")
    log("ThyroidXL adapter: root = " + base)

    stats = os.path.join(base, "stats")
    train_ids = _read_ids(os.path.join(stats, "train_patients.txt"))
    test_ids = _read_ids(os.path.join(stats, "test_patients.txt"))
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError("official splits overlap on %d patients" % len(overlap))

    with open(os.path.join(stats, "id2info_eng.json"), encoding="utf-8") as fh:
        info: Dict[str, dict] = json.load(fh)
    log("ThyroidXL adapter: %d train / %d test patients, %d in id2info"
        % (len(train_ids), len(test_ids), len(info)))

    rows: List[Dict] = []
    missing_info, missing_files, truncated = 0, 0, 0

    for split, ids, sub in (("dev", train_ids, "train"), ("test", test_ids, "test")):
        img_dir = os.path.join(base, sub, "images")
        msk_dir = os.path.join(base, sub, "masks")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError("missing image directory: " + img_dir)

        for pid in ids:
            rec = info.get(pid)
            if rec is None:
                missing_info += 1
                continue

            names = list(rec.get("images") or [])
            if not names:                       # fall back to the filesystem
                names = sorted(f for f in os.listdir(img_dir)
                               if f.startswith(pid + "_"))
            names = sorted(names)
            if len(names) > t_max:
                names = names[:t_max]
                truncated += 1

            images, masks = [], []
            for n in names:
                ip = os.path.join(img_dir, n)
                if not os.path.exists(ip):
                    missing_files += 1
                    continue
                mp = os.path.join(msk_dir, n)
                images.append(ip)
                masks.append(mp if os.path.exists(mp) else None)
            if not images:
                missing_files += 1
                continue

            code = rec.get("conclusion")
            if code in MALIGNANT_CODES:
                label = 1
            elif code in BENIGN_CODES:
                label = 0
            else:
                # An unknown code must never be guessed into a class.
                log("ThyroidXL adapter: unknown conclusion code %r for patient %s"
                    % (code, pid))
                continue

            nodule = rec.get("nodule_1") or {}
            tirads = nodule.get("TIRADS")
            rows.append({
                "patient_id": str(pid),
                "label": label,
                "split": split,
                "n_frames": len(images),
                "image_paths": images,
                "mask_paths": masks,
                "bbox_xmls": [None] * len(images),
                "tirads": float(tirads) if tirads is not None else np.nan,
                "age": _clean_age(rec.get("age")),
                "sex": ("F" if rec.get("gender") == FEMALE_CODE else "M")
                       if rec.get("gender") is not None else None,
                "conclusion_code": code,
                "conclusion_text": CONCLUSION_TEXT.get(code, str(code)),
            })

    man = pd.DataFrame(rows).sort_values(["split", "patient_id"]).reset_index(drop=True)
    if missing_info or missing_files or truncated:
        log("ThyroidXL adapter: %d patients absent from id2info, %d with missing "
            "files, %d truncated to Tmax" % (missing_info, missing_files, truncated))
    log("ThyroidXL adapter: %d patients, %d frames"
        % (len(man), int(man["n_frames"].sum())))

    if strict:
        assert_table1(man)
    return man


# --------------------------------------------------------------------------- #
def assert_table1(man: pd.DataFrame, strict: bool = True) -> Dict[str, object]:
    """Regression check: does the manifest reproduce Table 1 of the paper?

    This is the cheapest guard the project has against a silent data-layer
    regression. Every patient-level number depends on it, so a mismatch raises
    rather than warning.
    """
    dev = man[man["split"] == "dev"]
    test = man[man["split"] == "test"]
    got = {
        "dev_patients": len(dev), "test_patients": len(test),
        "dev_images": int(dev["n_frames"].sum()), "test_images": int(test["n_frames"].sum()),
        "dev_malignant": int((dev["label"] == 1).sum()),
        "test_malignant": int((test["label"] == 1).sum()),
        "dev_tirads": {int(k): int(v) for k, v in
                       dev["tirads"].dropna().astype(int).value_counts().items()},
        "test_tirads": {int(k): int(v) for k, v in
                        test["tirads"].dropna().astype(int).value_counts().items()},
    }

    problems: List[str] = []
    for key in ("dev_patients", "test_patients", "dev_images", "test_images",
                "dev_malignant", "test_malignant"):
        if got[key] != TABLE1[key]:
            problems.append("%s: expected %d, got %d" % (key, TABLE1[key], got[key]))
    for key in ("dev_tirads", "test_tirads"):
        want = {k: v for k, v in TABLE1[key].items() if v > 0}
        have = {k: v for k, v in got[key].items() if v > 0}
        if want != have:
            problems.append("%s: expected %s, got %s" % (key, want, have))

    if problems:
        msg = ("ThyroidXL manifest does NOT reproduce Table 1:\n  "
               + "\n  ".join(problems))
        if strict:
            raise AssertionError(msg)
        log("WARNING " + msg)
    else:
        log("ThyroidXL adapter: Table 1 reproduced exactly "
            "(%d/%d patients, %d/%d images, %d/%d malignant, TIRADS match)"
            % (got["dev_patients"], got["test_patients"],
               got["dev_images"], got["test_images"],
               got["dev_malignant"], got["test_malignant"]))
    return got


# --------------------------------------------------------------------------- #
def label_provenance(man: pd.DataFrame) -> pd.DataFrame:
    """How many labels come from histopathology vs FNAC.

    The paper describes labels as histopathology-derived; in the shipped data a
    large majority are FNAC-confirmed. Worth reporting rather than restating the
    paper's wording.
    """
    if "conclusion_text" not in man.columns:
        return pd.DataFrame()
    tab = (man.groupby(["split", "conclusion_text"])
              .size().rename("n").reset_index())
    tab["source"] = np.where(tab["conclusion_text"].str.contains("Histopathology"),
                             "histopathology", "FNAC")
    return tab
