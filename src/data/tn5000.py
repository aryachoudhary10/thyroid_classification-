"""Purpose-built TN5000 reader.

Verified layout:

    Main data/
      JPEGImages/     5000 .jpg
      Annotations/    5000 .xml     Pascal VOC, exactly one object per image
      ImageSets/Main/ train 3500 | val 500 | test 1000 | trainval 4000

Class identity is the one thing the data cannot tell you: the VOC ``<name>``
field is the literal string "0" or "1", and the authors' own repo declares
``classes = ('0', '1')`` with no semantics. It is resolved from the source
publication (Zhang et al., Sci Data 2025), which reports **3,572 malignant and
1,428 benign**; the mirror counts 3574 class-1 and 1426 class-0, so

    class "1" = malignant,  class "0" = benign

(the 2-image difference between mirror and paper is noted, not corrected).

The official ``val`` split holds 375 malignant + 125 benign, which is exactly
how the RCAF paper arrived at its "125 malignant + 125 benign" balanced subset:
take all 125 benign and match 125 malignant.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.common import log

MALIGNANT_CLASS = "1"
BENIGN_CLASS = "0"
# Reported in the source paper; used only as a sanity check, never to relabel.
PAPER_COUNTS = {"malignant": 3572, "benign": 1428}


# --------------------------------------------------------------------------- #
def find_root(root: str) -> Optional[str]:
    """Locate the VOC-style TN5000 directory anywhere under ``root``."""
    if not root or not os.path.isdir(root):
        return None

    def ok(d: str) -> bool:
        return (os.path.isdir(os.path.join(d, "JPEGImages"))
                and os.path.isdir(os.path.join(d, "Annotations")))

    if ok(root):
        return root
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            cand = os.path.join(dp, d)
            if ok(cand):
                return cand
    return None


def looks_like_tn5000(root: str) -> bool:
    return find_root(root) is not None


def _read_ids(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _voc_class(xml_path: str) -> Optional[str]:
    try:
        root = ET.parse(xml_path).getroot()
        obj = root.find("object")
        if obj is None:
            return None
        return (obj.findtext("name") or "").strip()
    except Exception:                                          # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
def build_tn5000_manifest(root: str, verify: bool = True) -> pd.DataFrame:
    """One row per image (a bag of one), carrying the official split."""
    base = find_root(root)
    if base is None:
        raise FileNotFoundError("no TN5000 VOC layout under " + str(root))
    log("TN5000 adapter: root = " + base)

    img_dir = os.path.join(base, "JPEGImages")
    ann_dir = os.path.join(base, "Annotations")
    sets_dir = os.path.join(base, "ImageSets", "Main")

    # official splits, if present
    split_of: Dict[str, str] = {}
    if os.path.isdir(sets_dir):
        for name, tag in (("train.txt", "train"), ("val.txt", "val"),
                          ("test.txt", "test")):
            p = os.path.join(sets_dir, name)
            if os.path.exists(p):
                ids = _read_ids(p)
                for i in ids:
                    split_of[i] = tag
                log("TN5000 adapter: official %-5s split = %d ids" % (tag, len(ids)))
    else:
        log("TN5000 adapter: no ImageSets/Main -- splits will be created")

    rows: List[Dict] = []
    unknown = 0
    for fn in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        xml = os.path.join(ann_dir, stem + ".xml")
        cls = _voc_class(xml) if os.path.exists(xml) else None
        if cls == MALIGNANT_CLASS:
            label = 1
        elif cls == BENIGN_CLASS:
            label = 0
        else:
            unknown += 1
            continue
        rows.append({
            "patient_id": stem,
            "label": label,
            "split": split_of.get(stem),
            "n_frames": 1,
            "image_paths": [os.path.join(img_dir, fn)],
            "mask_paths": [None],                 # no pixel masks in TN5000
            "bbox_xmls": [xml if os.path.exists(xml) else None],
            "tirads": np.nan,
            "age": np.nan,
            "sex": None,
        })

    man = pd.DataFrame(rows).reset_index(drop=True)
    if unknown:
        log("TN5000 adapter: %d images had no usable VOC class and were dropped"
            % unknown)
    n_mal = int((man["label"] == 1).sum())
    n_ben = int((man["label"] == 0).sum())
    log("TN5000 adapter: %d images -- %d malignant / %d benign"
        % (len(man), n_mal, n_ben))

    if verify:
        d_mal = abs(n_mal - PAPER_COUNTS["malignant"])
        d_ben = abs(n_ben - PAPER_COUNTS["benign"])
        if d_mal > 20 or d_ben > 20:
            log("TN5000 adapter: WARNING counts differ from the source paper "
                "(%d/%d vs %d/%d). Check the class mapping before trusting any "
                "external-validation number."
                % (n_mal, n_ben, PAPER_COUNTS["malignant"], PAPER_COUNTS["benign"]))
        else:
            log("TN5000 adapter: class mapping consistent with the source paper "
                "(paper %d/%d, mirror %d/%d)"
                % (PAPER_COUNTS["malignant"], PAPER_COUNTS["benign"], n_mal, n_ben))
    return man


# --------------------------------------------------------------------------- #
def official_eval_subset(man: pd.DataFrame, per_class: int = 125,
                         seed: int = 1337) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Class-balanced evaluation subset drawn from the official ``val`` split.

    Returns (adaptation pool, evaluation subset). The evaluation subset is
    removed from the pool entirely -- it is never seen by training, early
    stopping, model selection or threshold tuning.
    """
    rng = np.random.RandomState(seed)
    has_official = (man["split"] == "val").sum() > 0
    pool = man[man["split"] == "val"] if has_official else man
    if has_official:
        log("TN5000 adapter: drawing the evaluation subset from the official "
            "val split (%d images)" % len(pool))
    else:
        log("TN5000 adapter: no official val split -- sampling from the full set")

    picks: List[int] = []
    for cls in (1, 0):
        idx = np.array(pool.index[pool["label"] == cls].to_numpy(), copy=True)
        rng.shuffle(idx)
        take = min(per_class, len(idx))
        if take < per_class:
            log("TN5000 adapter: WARNING only %d of %d class-%d cases available"
                % (take, per_class, cls))
        picks.extend(idx[:take].tolist())

    eval_df = man.loc[picks].reset_index(drop=True)
    adapt_df = man.drop(index=picks).reset_index(drop=True)
    # Never adapt on the official test split either -- keep it untouched.
    if (adapt_df["split"] == "test").any():
        n = int((adapt_df["split"] == "test").sum())
        adapt_df = adapt_df[adapt_df["split"] != "test"].reset_index(drop=True)
        log("TN5000 adapter: withheld %d official test images from adaptation" % n)

    log("TN5000 adapter: adapt on %d images, evaluate on %d (%d malignant / %d benign)"
        % (len(adapt_df), len(eval_df),
           int((eval_df["label"] == 1).sum()), int((eval_df["label"] == 0).sum())))
    return adapt_df, eval_df


def split_summary(man: pd.DataFrame) -> pd.DataFrame:
    """Per-split class balance, for the record."""
    if man.empty:
        return pd.DataFrame()
    t = (man.assign(split=man["split"].fillna("unassigned"))
            .groupby(["split", "label"]).size().unstack(fill_value=0))
    t.columns = ["benign" if c == 0 else "malignant" for c in t.columns]
    t["total"] = t.sum(axis=1)
    return t.reset_index()
