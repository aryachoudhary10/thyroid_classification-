"""Filesystem discovery for the ThyroidXL and TN5000 Kaggle mirrors.

The exact folder layout of a Kaggle mirror is not guaranteed, so nothing here
hard-codes paths. Instead we:

  1. walk the download root and index every image / mask / table / annotation,
  2. score candidate directories with name heuristics,
  3. pair masks to images by filename stem,
  4. pull patient id / label / split / TIRADS from any CSV that has usable
     columns, falling back to folder-name and filename inference.

Always run ``describe_root`` first and eyeball the printed tree. If discovery
guesses wrong, pass an explicit ``LayoutOverride`` -- no code edits needed.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.common import log

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TAB_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".json"}

MASK_HINTS = ("mask", "seg", "label_img", "labels_img", "gt", "groundtruth",
              "ground_truth", "annotation_mask", "binary")
IMAGE_HINTS = ("image", "img", "frame", "ultrasound", "us", "scan", "data", "jpg", "png")
BENIGN_HINTS = ("benign", "b_", "negative", "normal", "0")
MALIGNANT_HINTS = ("malignant", "malign", "m_", "cancer", "positive", "1")

PATIENT_COL_PAT = re.compile(r"(patient|case|subject|pid|study|exam)", re.I)
IMAGE_COL_PAT = re.compile(r"(image|file|img|path|name|frame)", re.I)
LABEL_COL_PAT = re.compile(r"(label|class|diagnos|malign|benign|target|pathol|cancer)", re.I)
SPLIT_COL_PAT = re.compile(r"(split|subset|fold|partition|set)$", re.I)
TIRADS_COL_PAT = re.compile(r"(tirads|ti_rads|tr\b|birads)", re.I)
AGE_COL_PAT = re.compile(r"^age", re.I)
SEX_COL_PAT = re.compile(r"(sex|gender)", re.I)


# --------------------------------------------------------------------------- #
@dataclass
class LayoutOverride:
    """Manual escape hatch when the heuristics guess wrong."""
    image_dirs: Optional[Sequence[str]] = None
    mask_dirs: Optional[Sequence[str]] = None
    table_path: Optional[str] = None
    patient_col: Optional[str] = None
    image_col: Optional[str] = None
    label_col: Optional[str] = None
    split_col: Optional[str] = None
    tirads_col: Optional[str] = None
    # regex with a single capture group applied to the image stem
    patient_regex: Optional[str] = None
    positive_values: Sequence[str] = field(
        default_factory=lambda: ("1", "malignant", "malign", "positive", "cancer",
                                 "yes", "true", "m"))


# --------------------------------------------------------------------------- #
def _walk_index(root: str) -> Dict[str, List[str]]:
    images, masks_dirs, tables, xmls, others = [], set(), [], [], []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            full = os.path.join(dirpath, fn)
            if ext in IMG_EXT:
                images.append(full)
            elif ext in TAB_EXT:
                tables.append(full)
            elif ext == ".xml":
                xmls.append(full)
            elif ext == ".txt":
                others.append(full)
    return {"images": images, "tables": tables, "xmls": xmls, "txts": others}


def describe_root(root: str, max_dirs: int = 60, max_examples: int = 3) -> str:
    """Human-readable summary of a downloaded dataset -- print this first."""
    idx = _walk_index(root)
    by_dir: Dict[str, List[str]] = defaultdict(list)
    for p in idx["images"]:
        by_dir[os.path.dirname(p)].append(p)

    lines = ["ROOT: " + root,
             "images: %d   tables: %d   xml: %d   txt: %d"
             % (len(idx["images"]), len(idx["tables"]), len(idx["xmls"]), len(idx["txts"])),
             "",
             "--- image directories (count) ---"]
    for d, files in sorted(by_dir.items(), key=lambda kv: -len(kv[1]))[:max_dirs]:
        rel = os.path.relpath(d, root)
        lines.append("  %-70s %6d" % (rel[:70], len(files)))
        for f in files[:max_examples]:
            lines.append("        e.g. " + os.path.basename(f))

    if idx["tables"]:
        lines += ["", "--- tables ---"]
        for t in idx["tables"][:20]:
            lines.append("  " + os.path.relpath(t, root))
            try:
                df = _read_table(t, nrows=3)
                lines.append("      cols: " + ", ".join(map(str, df.columns[:25])))
            except Exception as exc:                           # noqa: BLE001
                lines.append("      (unreadable: " + str(exc)[:80] + ")")

    if idx["xmls"]:
        lines += ["", "--- xml annotations: %d (e.g. %s) ---"
                  % (len(idx["xmls"]), os.path.relpath(idx["xmls"][0], root))]
    return "\n".join(lines)


def _read_table(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=nrows)
    if ext == ".json":
        df = pd.read_json(path)
        return df.head(nrows) if nrows else df
    sep = "\t" if ext == ".tsv" else ","
    return pd.read_csv(path, sep=sep, nrows=nrows)


# --------------------------------------------------------------------------- #
def _dir_score(path: str, hints: Sequence[str]) -> int:
    low = path.lower().replace("\\", "/")
    return sum(1 for h in hints if h in low)


def _is_mask_dir(path: str) -> bool:
    low = os.path.basename(path).lower()
    return any(h in low for h in MASK_HINTS)


def _split_image_mask_dirs(image_paths: Sequence[str]) -> Tuple[List[str], List[str]]:
    dirs = sorted({os.path.dirname(p) for p in image_paths})
    mask_dirs = [d for d in dirs if _is_mask_dir(d)]
    img_dirs = [d for d in dirs if d not in mask_dirs]
    return img_dirs, mask_dirs


def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _pair_masks(image_paths: Sequence[str], mask_paths: Sequence[str]) -> Dict[str, str]:
    """Map image path -> mask path by filename stem, tolerating common suffixes."""
    by_stem: Dict[str, str] = {}
    for m in mask_paths:
        s = _stem(m)
        by_stem.setdefault(s, m)
        for suf in ("_mask", "-mask", "_seg", "-seg", "_gt", "-gt", "_label", "_m"):
            if s.endswith(suf):
                by_stem.setdefault(s[: -len(suf)], m)
    out: Dict[str, str] = {}
    for img in image_paths:
        s = _stem(img)
        cand = by_stem.get(s)
        if cand is None:
            for suf in ("_mask", "_seg", "_gt", "_label"):
                cand = by_stem.get(s + suf)
                if cand is not None:
                    break
        if cand is not None:
            out[img] = cand
    return out


# --------------------------------------------------------------------------- #
def _pick_col(df: pd.DataFrame, pattern: re.Pattern, explicit: Optional[str]) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    for c in df.columns:
        if pattern.search(str(c)):
            return c
    return None


def _to_binary_label(series: pd.Series, positives: Sequence[str]) -> pd.Series:
    pos = {str(p).strip().lower() for p in positives}

    def conv(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in pos:
            return 1.0
        if s in ("0", "benign", "negative", "no", "false", "b"):
            return 0.0
        try:
            f = float(s)
            return 1.0 if f >= 0.5 else 0.0
        except ValueError:
            return np.nan

    return series.map(conv)


def _infer_patient_from_stem(stem: str, regex: Optional[str]) -> str:
    if regex:
        m = re.search(regex, stem)
        if m:
            return m.group(1)
    # common patterns: 1234_1.png / patient_0012_frame3 / case12-2
    m = re.match(r"^([A-Za-z]*[_-]?\d+)", stem)
    if m:
        return m.group(1)
    return stem


def _label_from_path(path: str) -> Optional[int]:
    low = path.lower().replace("\\", "/")
    parts = low.split("/")
    for part in reversed(parts[:-1]):
        if any(h == part or h in part for h in ("malignant", "malign", "cancer")):
            return 1
        if any(h == part or h in part for h in ("benign",)):
            return 0
    return None


def _split_from_path(path: str) -> Optional[str]:
    low = path.lower().replace("\\", "/")
    for part in low.split("/"):
        if part in ("test", "testing", "val", "valid", "validation"):
            return "test" if part.startswith("test") else "val"
        if part in ("train", "training", "dev", "development"):
            return "dev"
    return None


# --------------------------------------------------------------------------- #
def build_records(root: str,
                  override: Optional[LayoutOverride] = None,
                  require_mask: bool = False,
                  drop_unlabeled: bool = True) -> pd.DataFrame:
    """Return a per-image DataFrame.

    Columns: image_path, mask_path, bbox_xml, patient_id, label, split,
             tirads, age, sex, source.
    """
    override = override or LayoutOverride()
    idx = _walk_index(root)
    if not idx["images"]:
        raise RuntimeError("no images found under " + root)

    if override.image_dirs is not None:
        img_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                    for d in override.image_dirs]
        mask_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                     for d in (override.mask_dirs or [])]
        images = [p for p in idx["images"]
                  if any(os.path.dirname(p).startswith(d) for d in img_dirs)]
        masks = [p for p in idx["images"]
                 if any(os.path.dirname(p).startswith(d) for d in mask_dirs)]
    else:
        img_dirs, mask_dirs = _split_image_mask_dirs(idx["images"])
        images = [p for p in idx["images"] if os.path.dirname(p) in set(img_dirs)]
        masks = [p for p in idx["images"] if os.path.dirname(p) in set(mask_dirs)]

    log("discovery: %d images, %d mask candidates (%d img dirs, %d mask dirs)"
        % (len(images), len(masks), len(set(os.path.dirname(p) for p in images)),
           len(set(os.path.dirname(p) for p in masks))))

    mask_map = _pair_masks(images, masks) if masks else {}
    xml_map: Dict[str, str] = {}
    for x in idx["xmls"]:
        xml_map.setdefault(_stem(x), x)

    rows = []
    for p in images:
        stem = _stem(p)
        rows.append({
            "image_path": p,
            "mask_path": mask_map.get(p),
            "bbox_xml": xml_map.get(stem),
            "stem": stem,
            "patient_id": None,
            "label": _label_from_path(p),
            "split": _split_from_path(p),
            "tirads": np.nan,
            "age": np.nan,
            "sex": None,
        })
    df = pd.DataFrame(rows)

    # ---- enrich from a metadata table, if one is usable ------------------ #
    table = _choose_table(idx["tables"], override)
    if table is not None:
        df = _merge_table(df, table, override)

    # ---- patient id fallback -------------------------------------------- #
    missing_pid = df["patient_id"].isna()
    if missing_pid.any():
        df.loc[missing_pid, "patient_id"] = df.loc[missing_pid, "stem"].map(
            lambda s: _infer_patient_from_stem(s, override.patient_regex))
    df["patient_id"] = df["patient_id"].astype(str)

    # ---- label sanity ---------------------------------------------------- #
    n_missing_label = int(df["label"].isna().sum())
    if n_missing_label and drop_unlabeled:
        log("discovery: WARNING %d/%d images have no label; they will be dropped"
            % (n_missing_label, len(df)))
        df = df[~df["label"].isna()].copy()
    elif n_missing_label:
        log("discovery: %d/%d images have no label yet -- left for the caller "
            "to resolve (e.g. from VOC object names)" % (n_missing_label, len(df)))
    if not df["label"].isna().any():
        df["label"] = df["label"].astype(int)

    if require_mask:
        n_no_mask = int(df["mask_path"].isna().sum())
        if n_no_mask:
            log("discovery: %d images without a pixel mask (bbox/xml fallback will apply)"
                % n_no_mask)

    df["source"] = os.path.basename(root.rstrip("/\\"))
    return df.reset_index(drop=True)


def _choose_table(tables: Sequence[str], override: LayoutOverride) -> Optional[pd.DataFrame]:
    if override.table_path:
        return _read_table(override.table_path)
    best, best_score = None, 0
    for t in tables:
        try:
            df = _read_table(t)
        except Exception:                                      # noqa: BLE001
            continue
        if df.empty:
            continue
        score = 0
        cols = list(df.columns)
        score += 2 * any(IMAGE_COL_PAT.search(str(c)) for c in cols)
        score += 2 * any(LABEL_COL_PAT.search(str(c)) for c in cols)
        score += 1 * any(PATIENT_COL_PAT.search(str(c)) for c in cols)
        score += 1 * any(SPLIT_COL_PAT.search(str(c)) for c in cols)
        score += 1 * any(TIRADS_COL_PAT.search(str(c)) for c in cols)
        score += min(len(df) // 1000, 3)
        if score > best_score:
            best, best_score = df, score
    if best is not None:
        log("discovery: using metadata table with columns " + ", ".join(map(str, best.columns[:20])))
    return best


def _merge_table(df: pd.DataFrame, table: pd.DataFrame,
                 override: LayoutOverride) -> pd.DataFrame:
    img_col = _pick_col(table, IMAGE_COL_PAT, override.image_col)
    pid_col = _pick_col(table, PATIENT_COL_PAT, override.patient_col)
    lab_col = _pick_col(table, LABEL_COL_PAT, override.label_col)
    spl_col = _pick_col(table, SPLIT_COL_PAT, override.split_col)
    tir_col = _pick_col(table, TIRADS_COL_PAT, override.tirads_col)
    age_col = _pick_col(table, AGE_COL_PAT, None)
    sex_col = _pick_col(table, SEX_COL_PAT, None)

    if img_col is None and pid_col is None:
        log("discovery: metadata table has no joinable key -- ignoring it")
        return df

    tab = table.copy()
    if img_col is not None:
        tab["_key"] = tab[img_col].astype(str).map(lambda s: _stem(str(s)))
        left_key = "stem"
    else:
        tab["_key"] = tab[pid_col].astype(str)
        left_key = "patient_id"

    keep = {"_key": "_key"}
    if pid_col is not None:
        keep[pid_col] = "_pid"
    if lab_col is not None:
        keep[lab_col] = "_label"
    if spl_col is not None:
        keep[spl_col] = "_split"
    if tir_col is not None:
        keep[tir_col] = "_tirads"
    if age_col is not None:
        keep[age_col] = "_age"
    if sex_col is not None:
        keep[sex_col] = "_sex"
    tab = tab[list(keep)].rename(columns=keep).drop_duplicates("_key")

    if left_key == "patient_id":
        df["_join"] = df["patient_id"].astype(str)
    else:
        df["_join"] = df["stem"].astype(str)
    out = df.merge(tab, left_on="_join", right_on="_key", how="left")

    if "_pid" in out:
        out["patient_id"] = out["_pid"].where(out["_pid"].notna(), out["patient_id"])
    if "_label" in out:
        lab = _to_binary_label(out["_label"], override.positive_values)
        out["label"] = lab.where(lab.notna(), out["label"])
    if "_split" in out:
        norm = out["_split"].astype(str).str.lower().map(
            lambda s: "test" if s.startswith("test") else
                      ("dev" if s.startswith(("train", "dev")) else
                       ("val" if s.startswith("val") else None)))
        out["split"] = norm.where(norm.notna(), out["split"])
    if "_tirads" in out:
        out["tirads"] = pd.to_numeric(
            out["_tirads"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
    if "_age" in out:
        out["age"] = pd.to_numeric(out["_age"], errors="coerce")
    if "_sex" in out:
        out["sex"] = out["_sex"]

    drop = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop)


# --------------------------------------------------------------------------- #
def parse_voc_bbox(xml_path: str) -> Tuple[List[Tuple[int, int, int, int]],
                                           Optional[Tuple[int, int]],
                                           List[str]]:
    """Return (boxes, (width, height), names) from a Pascal VOC XML file."""
    boxes, names = [], []
    size = None
    try:
        root = ET.parse(xml_path).getroot()
        s = root.find("size")
        if s is not None:
            w = s.findtext("width")
            h = s.findtext("height")
            if w and h:
                size = (int(float(w)), int(float(h)))
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                boxes.append((int(float(bb.findtext("xmin"))),
                              int(float(bb.findtext("ymin"))),
                              int(float(bb.findtext("xmax"))),
                              int(float(bb.findtext("ymax")))))
                names.append((obj.findtext("name") or "").strip().lower())
            except (TypeError, ValueError):
                continue
    except Exception as exc:                                   # noqa: BLE001
        log("voc parse failed for " + xml_path + ": " + str(exc)[:100])
    return boxes, size, names


def label_from_voc(xml_path: str) -> Optional[int]:
    """TN5000 encodes benign/malignant in the VOC object <name> field."""
    _boxes, _size, names = parse_voc_bbox(xml_path)
    if not names:
        return None
    joined = " ".join(names)
    if any(h in joined for h in ("malign", "cancer", "positive")):
        return 1
    if "benign" in joined:
        return 0
    if joined.strip() in ("1", "m"):
        return 1
    if joined.strip() in ("0", "b"):
        return 0
    return None
