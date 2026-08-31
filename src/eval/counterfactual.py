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

    The claim under test is about *views*: can the evidence in one frame be
    trusted, given what the other frames say. So the correlation is computed
    WITHIN each patient and WITHIN each region, across frames -- the region is
    held fixed and the view varies. Patients with fewer than three valid frames
    are excluded, because a rank correlation over two points is +-1 by
    construction and carries no information.

    Regions must not be pooled into the same ranking. Regions differ in
    influence by orders of magnitude (see ``evidence_ablation``: core and margin
    move the prediction ~100x more than peri and global), so a correlation taken
    over frames and regions together is dominated by the region axis and says
    nothing about cross-view reliability. A single-frame bag with K regions even
    satisfies a >=3-token guard while contributing zero cross-view content. That
    pooled statistic is still reported, under ``pooled_tokens``, but it is a
    diagnostic and not the test.

    ``region_profile`` reports mean R and mean influence per region, which is
    what reveals the confound directly, and every raw (frame, region, R,
    influence) row is written to CSV so none of this needs a second GPU pass.
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

    min_frames = max(int(getattr(cfg.eval, "influence_min_frames", 3)), 3)
    cfg_regions = tuple(getattr(cfg.model, "regions", ()) or cfg.data.region_names)
    pids = te["patient_id"].astype(str).tolist()

    rows: List[Dict[str, Any]] = []
    xview_sp: Dict[str, List[float]] = {}
    xview_kd: Dict[str, List[float]] = {}
    pooled_sp: List[float] = []
    pooled_kd: List[float] = []
    n_pat_xview = 0

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

        names = (list(cfg_regions) if len(cfg_regions) == k
                 else ["region%d" % j for j in range(k)])
        Rn = R.cpu().numpy()
        In = infl.cpu().numpy()
        Vn = valid.cpu().numpy()
        Ix = batch["index"].cpu().numpy()

        for i in range(b):
            vmask = Vn[i] > 0.5                         # (T,) frame validity
            Rv = Rn[i][vmask]                           # (F, K)
            Iv = In[i][vmask]
            f = int(Rv.shape[0])
            gi = int(Ix[i])
            pid = pids[gi] if 0 <= gi < len(pids) else str(gi)

            for fi in range(f):
                for ki in range(k):
                    rows.append({"patient_id": pid, "n_valid_frames": f,
                                 "frame": fi, "region": names[ki],
                                 "R": float(Rv[fi, ki]),
                                 "influence": float(Iv[fi, ki])})

            # ---- primary: across views, one region at a time ---------------
            if f >= min_frames:
                counted = False
                for ki in range(k):
                    r_k, i_k = Rv[:, ki], Iv[:, ki]
                    if np.allclose(r_k, r_k[0]) or np.allclose(i_k, i_k[0]):
                        continue
                    sp = stats.spearmanr(r_k, i_k).correlation
                    kd = stats.kendalltau(r_k, i_k).correlation
                    if np.isfinite(sp):
                        xview_sp.setdefault(names[ki], []).append(float(sp))
                        counted = True
                    if np.isfinite(kd):
                        xview_kd.setdefault(names[ki], []).append(float(kd))
                if counted:
                    n_pat_xview += 1

            # ---- diagnostic: the old statistic, regions pooled in ----------
            r_i, i_i = Rv.ravel(), Iv.ravel()
            if len(r_i) >= 3 and not np.allclose(r_i, r_i[0]):
                sp = stats.spearmanr(r_i, i_i).correlation
                kd = stats.kendalltau(r_i, i_i).correlation
                if np.isfinite(sp):
                    pooled_sp.append(float(sp))
                if np.isfinite(kd):
                    pooled_kd.append(float(kd))

    raw = pd.DataFrame(rows)
    raw_path = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name,
                            "reliability_influence_tokens.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    raw.to_csv(raw_path, index=False)

    all_sp = [v for lst in xview_sp.values() for v in lst]
    all_kd = [v for lst in xview_kd.values() for v in lst]

    def _blk(sp: List[float], kd: List[float]) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "n": len(sp),
            "mean_spearman": float(np.mean(sp)) if sp else float("nan"),
            "median_spearman": float(np.median(sp)) if sp else float("nan"),
            "frac_positive": float(np.mean(np.asarray(sp) > 0)) if sp else float("nan"),
            "mean_kendall": float(np.mean(kd)) if kd else float("nan"),
        }
        if len(sp) > 1:
            t_stat, p_val = stats.ttest_1samp(sp, 0.0)
            d["ttest_t"] = float(t_stat)
            d["ttest_p"] = float(p_val)
        return d

    cross = _blk(all_sp, all_kd)
    cross["n_patients"] = n_pat_xview
    cross["min_frames"] = min_frames
    cross["per_region"] = {n: _blk(v, xview_kd.get(n, []))
                           for n, v in sorted(xview_sp.items())}

    pooled = _blk(pooled_sp, pooled_kd)
    pooled["note"] = ("frames and regions ranked together; dominated by the "
                      "region axis whenever regions differ systematically in "
                      "influence, and satisfiable by a single-frame bag -- "
                      "a confound diagnostic, not a test of the cross-view claim")

    profile: Dict[str, Any] = {}
    if len(raw):
        agg = raw.groupby("region")[["R", "influence"]].mean()
        profile = {n: {"mean_R": float(r["R"]), "mean_influence": float(r["influence"])}
                   for n, r in agg.iterrows()}

    res: Dict[str, Any] = {
        "model": model_name,
        "cross_view": cross,
        "region_profile": profile,
        "pooled_tokens": pooled,
        "tokens_csv": raw_path,
        # Legacy keys, unchanged in meaning: these have always been the pooled
        # statistic. Read `cross_view` for the claim.
        "n_patients_analysed": pooled["n"],
        "mean_within_patient_spearman": pooled["mean_spearman"],
        "median_within_patient_spearman": pooled["median_spearman"],
        "frac_positive_spearman": pooled["frac_positive"],
        "mean_within_patient_kendall": pooled["mean_kendall"],
    }

    log("  CROSS-VIEW  (per region, across frames; bags with >= %d frames)" % min_frames)
    log("    patients %d | correlations %d" % (n_pat_xview, cross["n"]))
    log("    mean rho %+.4f | median %+.4f | frac positive %.3f | p %.3g"
        % (cross["mean_spearman"], cross["median_spearman"],
           cross["frac_positive"], cross.get("ttest_p", float("nan"))))
    for n, d in cross["per_region"].items():
        log("      %-8s n=%4d  mean rho %+.4f  frac+ %.3f"
            % (n, d["n"], d["mean_spearman"], d["frac_positive"]))
    log("  REGION PROFILE  (is influence merely a property of the region?)")
    for n, d in profile.items():
        log("      %-8s mean R %.4f  mean |dp| %.5f" % (n, d["mean_R"], d["mean_influence"]))
    log("  POOLED TOKENS  (legacy, region-confounded)")
    log("    mean rho %+.4f | frac positive %.3f | n %d"
        % (pooled["mean_spearman"], pooled["frac_positive"], pooled["n"]))
    log("  raw tokens -> %s (%d rows)" % (raw_path, len(raw)))

    save_json(res, out)
    registry.mark_done(key, {"json": out, "tokens_csv": raw_path})
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
