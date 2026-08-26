"""Counterfactual evidence validation.

Attention weights are not explanations. The question this module answers is
falsifiable: does the reliability the model assigns to a piece of evidence
predict how much the prediction actually moves when that evidence is removed?

Two levels of intervention:

* pixel level  -- the evidence region is zeroed in the input, so the encoder
                  never sees it. Tests the whole pipeline.
* token level  -- the evidence embedding is masked out of the aggregation only.
                  Isolates the aggregation mechanism from the encoder.

The headline statistic is the rank correlation between R[t, k] and
|p(x) - p(x without evidence k of frame t)| computed per patient.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ..config import Config
from ..data.dataset import Intervention
from ..data.splits import test_frame
from ..engine.cv import make_dataset, predict_with_checkpoints
from ..engine.trainer import move_batch
from ..eval.calibration import sigmoid
from ..eval.metrics import all_metrics
from ..models.factory import build_model
from ..utils.checkpoint import StageRegistry
from ..utils.common import banner, load_json, log, save_json


# --------------------------------------------------------------------------- #
def evidence_ablation(cfg: Config, manifest: pd.DataFrame, model_name: str,
                      ckpts: List[str], registry: StageRegistry,
                      regions: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Suppress one evidence region at a time, in pixel space, and re-evaluate."""
    key = "cf/evidence/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "evidence_ablation.csv")
    if registry.is_done(key) and os.path.exists(out):
        log("SKIP  " + key)
        return pd.read_csv(out)

    banner("EVIDENCE ABLATION (pixel level)  |  %s" % model_name)
    regions = tuple(regions or cfg.model.regions)
    te = test_frame(manifest)
    base_logit, labels, _ = predict_with_checkpoints(cfg, te, model_name, ckpts, None)
    base_p = sigmoid(base_logit)
    base_m = all_metrics(labels, base_p)

    rows = [{"suppressed": "none", "roc_auc": base_m["roc_auc"],
             "delta_roc_auc": 0.0, "mean_abs_delta_p": 0.0, "f1": base_m["f1"]}]
    for r in regions:
        lg, y, _e = predict_with_checkpoints(
            cfg, te, model_name, ckpts, Intervention(suppress_regions=(r,)))
        p = sigmoid(lg)
        m = all_metrics(y, p)
        rows.append({"suppressed": r, "roc_auc": m["roc_auc"],
                     "delta_roc_auc": m["roc_auc"] - base_m["roc_auc"],
                     "mean_abs_delta_p": float(np.mean(np.abs(p - base_p))),
                     "f1": m["f1"]})
        log("  suppress %-8s ROC-AUC %.4f (%+.4f) | mean |dp| %.4f"
            % (r, m["roc_auc"], rows[-1]["delta_roc_auc"], rows[-1]["mean_abs_delta_p"]))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    registry.mark_done(key, {"csv": out})
    return df


# --------------------------------------------------------------------------- #
@torch.no_grad()
def reliability_influence_correlation(cfg: Config, manifest: pd.DataFrame,
                                      model_name: str, ckpt: str,
                                      registry: StageRegistry,
                                      max_patients: int = 400) -> Dict[str, Any]:
    """Token-level counterfactual: correlate R[t,k] with |delta p| when removed.

    Returns Spearman and Kendall correlations computed WITHIN each patient and
    then averaged, which is the right unit of analysis -- comparing reliability
    across patients would confound it with case difficulty.
    """
    key = "cf/influence/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "reliability_influence.json")
    if registry.is_done(key) and os.path.exists(out):
        log("SKIP  " + key)
        return load_json(out, {})

    banner("RELIABILITY vs COUNTERFACTUAL INFLUENCE  |  %s" % model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reqs = build_model(cfg, model_name)
    if not hasattr(model, "predict_with_suppression"):
        log("  model has no token-level suppression hook -- skipping")
        return {"skipped": True}

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    te = test_frame(manifest)
    if max_patients and len(te) > max_patients:
        te = te.sample(max_patients, random_state=cfg.run.seed).reset_index(drop=True)
    ds = make_dataset(cfg, te, reqs, train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.optim.batch_size,
                                         shuffle=False, num_workers=cfg.data.num_workers)

    spearman, kendall, records = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out_full = model(batch)
        base = out_full["logit"].float()
        R = out_full["R"].float()                       # (B, T, K)
        valid = batch["valid"]
        b, t, k = R.shape

        infl = torch.zeros_like(R)
        for ti in range(t):
            for ki in range(k):
                keep = torch.ones(b, t, k, device=device)
                keep[:, ti, ki] = 0.0
                lg = model.predict_with_suppression(batch, token=keep).float()
                infl[:, ti, ki] = (torch.sigmoid(base) - torch.sigmoid(lg)).abs()

        Rn = R.cpu().numpy()
        In = infl.cpu().numpy()
        Vn = valid.cpu().numpy()
        for i in range(b):
            mask = Vn[i] > 0.5
            r_i = Rn[i][mask].ravel()
            i_i = In[i][mask].ravel()
            if len(r_i) < 3 or np.allclose(r_i, r_i[0]):
                continue
            sp = stats.spearmanr(r_i, i_i).correlation
            kd = stats.kendalltau(r_i, i_i).correlation
            if np.isfinite(sp):
                spearman.append(float(sp))
            if np.isfinite(kd):
                kendall.append(float(kd))
            records.append({"mean_R": float(r_i.mean()),
                            "mean_influence": float(i_i.mean())})

    res: Dict[str, Any] = {
        "model": model_name,
        "n_patients_analysed": len(spearman),
        "mean_within_patient_spearman": float(np.mean(spearman)) if spearman else float("nan"),
        "median_within_patient_spearman": float(np.median(spearman)) if spearman else float("nan"),
        "frac_positive_spearman": float(np.mean(np.asarray(spearman) > 0)) if spearman else float("nan"),
        "mean_within_patient_kendall": float(np.mean(kendall)) if kendall else float("nan"),
    }
    if spearman:
        t_stat, p_val = stats.ttest_1samp(spearman, 0.0)
        res["ttest_t"] = float(t_stat)
        res["ttest_p"] = float(p_val)
    for kk, vv in res.items():
        if isinstance(vv, float):
            log("  %-38s %.4f" % (kk, vv))
    save_json(res, out)
    registry.mark_done(key, {"json": out})
    del model, ds, loader
    torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------- #
@torch.no_grad()
def agreement_contradiction_cases(cfg: Config, manifest: pd.DataFrame,
                                  model_name: str, ckpt: str,
                                  registry: StageRegistry,
                                  top_n: int = 12) -> pd.DataFrame:
    """Rank test patients by cross-view agreement vs contradiction.

    Produces the case list behind the qualitative agreement / contradiction
    analyses: the patients where views corroborate each other most strongly,
    and those where the contradiction channel fires hardest.
    """
    key = "cf/cases/%s/%s" % (cfg.run.run_name, model_name)
    out = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                       "agreement_cases.csv")
    if registry.is_done(key) and os.path.exists(out):
        log("SKIP  " + key)
        return pd.read_csv(out)

    te = test_frame(manifest)
    multi = te[te["n_frames"] > 1].reset_index(drop=True)
    logits, labels, ex = predict_with_checkpoints(
        cfg, multi, model_name, [ckpt], None,
        collect=("S", "D", "U", "R", "alpha", "frame_reliability"))

    if "S" not in ex or not getattr(ex["S"], "size", 0):
        log("  model exposes no support/contradiction channels -- skipping")
        return pd.DataFrame()

    n_frames = multi["n_frames"].values
    mask = (np.arange(ex["S"].shape[1])[None, :] < n_frames[:, None])[:, :, None]

    def mmean(a):
        m = np.broadcast_to(mask, a.shape)
        return (a * m).sum(axis=(1, 2)) / np.maximum(m.sum(axis=(1, 2)), 1)

    df = pd.DataFrame({
        "patient_id": multi["patient_id"].astype(str).values,
        "label": labels.astype(int),
        "n_frames": n_frames,
        "p": sigmoid(logits),
        "support": mmean(ex["S"]),
        "contradiction": mmean(ex["D"]),
        "uncertainty": mmean(ex["U"]),
        "reliability": mmean(ex["R"]),
    })
    df["net_agreement"] = df["support"] - df["contradiction"]
    df["correct"] = ((df["p"] >= 0.5).astype(int) == df["label"]).astype(int)
    df = df.sort_values("net_agreement", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    log("  strongest agreement cases:")
    log(df.head(min(top_n, len(df)))[
        ["patient_id", "label", "p", "support", "contradiction", "correct"]].to_string())
    log("  strongest contradiction cases:")
    log(df.tail(min(top_n, len(df)))[
        ["patient_id", "label", "p", "support", "contradiction", "correct"]].to_string())

    hi = df[df["net_agreement"] > df["net_agreement"].median()]
    lo = df[df["net_agreement"] <= df["net_agreement"].median()]
    log("  accuracy | high agreement %.3f (n=%d) vs low agreement %.3f (n=%d)"
        % (hi["correct"].mean(), len(hi), lo["correct"].mean(), len(lo)))
    registry.mark_done(key, {"csv": out})
    return df
