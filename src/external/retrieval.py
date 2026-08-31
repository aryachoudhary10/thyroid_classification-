"""Arm 6: retrieval pseudo-bags, so multiple-instance reasoning is not inert.

TN5000 is an image-level dataset -- one image per case. Mapped honestly onto a
patient-level model, every bag holds a single frame, which means attention MIL,
cross-view support, contradiction and the whole reliability mechanism have
nothing to operate on. That is why DER-MIL and its reliability-disabled ablation
score identically on this dataset: not because the mechanism fails, but because
the dataset never asks it a question.

This arm supplies the missing structure by retrieval. Each evaluation image
becomes the query of a bag completed with its nearest neighbours in the source
model's embedding space, so the bag holds several genuinely related pieces of
evidence and the aggregation machinery has something to weigh.

Two honesty constraints are built in, because this arm is easy to do wrongly.

* Neighbours are drawn from the **adaptation pool only**, never from the
  evaluation subset. Retrieving evaluation images into each other's bags would
  let the held-out set inform itself, which is transductive leakage across the
  very cases being scored.
* The label of a bag is the **query's** label, and neighbour labels are never
  read -- not for construction, not for weighting.

What this arm is NOT: the neighbours are different patients, so the bag holds
cross-*case* evidence rather than several views of one nodule. That is a
different construct from the ThyroidXL bags the model was trained on, and the
number it produces should be reported as retrieval-augmented classification
rather than as the same patient-level task.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.manifest import load_manifest, save_manifest
from ..engine.cv import make_dataset
from ..engine.trainer import predict_dataset
from ..models.factory import build_model
from ..utils.checkpoint import StageRegistry
from ..utils.common import banner, log


# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_frames(cfg: Config, model_name: str, ckpt: str, df: pd.DataFrame,
                 device: Optional[torch.device] = None) -> np.ndarray:
    """One L2-normalised embedding per row, taken from the model's frame stage.

    Rows are single-frame bags here, so frame 0 is the whole case.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reqs = build_model(cfg, model_name)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    ds = make_dataset(cfg, df, reqs, train=False)
    _lg, _y, ex = predict_dataset(model, ds, cfg, device, collect=("frame_emb",),
                                  batch_size=cfg.external.batch_size)
    del model, ds
    torch.cuda.empty_cache()

    emb = ex.get("frame_emb")
    if emb is None or not getattr(emb, "size", 0):
        raise RuntimeError("model %s exposes no frame_emb; retrieval needs one"
                           % model_name)
    emb = emb[:, 0, :] if emb.ndim == 3 else emb           # (N, D)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norm, 1e-8)


def _neighbours(q: np.ndarray, pool: np.ndarray, k: int,
                block: int = 512) -> np.ndarray:
    """Indices of the k most cosine-similar pool rows for each query row."""
    out = np.zeros((len(q), k), dtype=np.int64)
    for s in range(0, len(q), block):
        sim = q[s:s + block] @ pool.T                      # both are L2-normalised
        out[s:s + block] = np.argsort(-sim, axis=1)[:, :k]
    return out


# --------------------------------------------------------------------------- #
def build_retrieval_bags(cfg: Config, model_name: str, ckpt: str,
                         adapt_df: pd.DataFrame, eval_df: pd.DataFrame,
                         registry: StageRegistry, bag_size: Optional[int] = None
                         ) -> pd.DataFrame:
    """Turn each evaluation image into a bag of itself plus its neighbours."""
    k = int(bag_size or cfg.adapt.retrieval_bag_size)
    key = "tn5000/retrieval/%s/%s/k%d" % (cfg.run.run_name, model_name, k)
    out_csv = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                           "tn5000_retrieval_bags_k%d.csv" % k)
    if registry.is_done(key) and os.path.exists(out_csv):
        log("SKIP  " + key)
        return load_manifest(out_csv)

    banner("TN5000 RETRIEVAL PSEUDO-BAGS  |  %s  (bag size %d)" % (model_name, k))
    pool = adapt_df.reset_index(drop=True)
    qry = eval_df.reset_index(drop=True)

    e_pool = embed_frames(cfg, model_name, ckpt, pool)
    e_qry = embed_frames(cfg, model_name, ckpt, qry)
    log("  embedded %d pool / %d query images (dim %d)"
        % (len(e_pool), len(e_qry), e_pool.shape[1]))

    nbr = _neighbours(e_qry, e_pool, max(k - 1, 0))

    rows: List[Dict[str, Any]] = []
    purity: List[float] = []
    for i, r in qry.iterrows():
        imgs = [p for p in r["image_paths"] if p][:1]
        msks = list(r["mask_paths"])[:1] or [None]
        xmls = list(r["bbox_xmls"])[:1] or [None]
        for j in nbr[i]:
            nr = pool.iloc[int(j)]
            ip = [p for p in nr["image_paths"] if p]
            if not ip:
                continue
            imgs.append(ip[0])
            msks.append((list(nr["mask_paths"]) or [None])[0])
            xmls.append((list(nr["bbox_xmls"]) or [None])[0])
        # Diagnostic only -- neighbour labels never enter the bag or its label.
        nb_lbl = pool.iloc[[int(j) for j in nbr[i]]]["label"].to_numpy()
        if len(nb_lbl):
            purity.append(float((nb_lbl == r["label"]).mean()))

        rows.append({"patient_id": str(r["patient_id"]), "label": int(r["label"]),
                     "split": "val", "n_frames": len(imgs),
                     "image_paths": imgs, "mask_paths": msks, "bbox_xmls": xmls,
                     "tirads": r.get("tirads"), "age": r.get("age"),
                     "sex": r.get("sex")})

    bags = pd.DataFrame(rows)
    save_manifest(bags, out_csv)
    mean_pure = float(np.mean(purity)) if purity else float("nan")
    log("  built %d bags of %d | mean neighbour label purity %.3f "
        "(0.5 = retrieval carries no class signal)" % (len(bags), k, mean_pure))
    registry.mark_done(key, {"csv": out_csv, "bag_size": k,
                             "neighbour_purity": mean_pure})
    return bags


# --------------------------------------------------------------------------- #
def evaluate_retrieval_bags(cfg: Config, model_name: str, ckpt: str,
                            bags: pd.DataFrame, n_views: int = 8) -> Dict[str, Any]:
    """Score the retrieval bags with TTA, exactly as the other arms are scored."""
    from ..data.dataset import Intervention
    from ..eval.calibration import sigmoid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reqs = build_model(cfg, model_name)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    probs, labels = [], None
    with torch.no_grad():
        for v in range(max(n_views, 1)):
            ds = make_dataset(cfg, bags, reqs, train=False,
                              intervention=Intervention(tta_view=v))
            lg, y, _e = predict_dataset(model, ds, cfg, device,
                                        batch_size=cfg.external.batch_size)
            probs.append(sigmoid(lg))
            labels = y
            del ds
    del model
    torch.cuda.empty_cache()
    return {"p": np.mean(np.vstack(probs), axis=0), "y": labels}
