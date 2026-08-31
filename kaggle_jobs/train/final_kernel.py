"""Finishing run: calibration, robustness on the final models, and TN5000.

Everything here operates on checkpoints that already exist. Only the TN5000
domain adaptation trains anything, and that is a separate model by design.

    1. regenerate OOF + test predictions          (inference)
    2. fit temperature + thresholds on OOF ONLY   (development-only)
    3. robustness + counterfactual suite          (inference)
    4. TN5000 external validation, zero-shot AND domain-adapted

Checkpoints arrive from the chained kernel output and from any attached
checkpoint dataset. They are merged into a single writable tree so every
downstream call sees one consistent namespace.

The external-validation arms are opt-in via DERMIL_ARMS, because they differ by
orders of magnitude in cost. See the arm table in the launch instructions.
"""
import os
import shutil
import sys
import time

T0 = time.time()
BUDGET_S = float(os.environ.get("DERMIL_BUDGET_HOURS", "8.5")) * 3600 - 900


def elapsed():
    return time.time() - T0


def left():
    return BUDGET_S - elapsed()


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


# --------------------------------------------------------------------------- #
def ensure_torch():
    if os.environ.get("DERMIL_TORCH_FIXED"):
        return
    try:
        import torch as _t
    except Exception:
        return
    if not _t.cuda.is_available():
        return
    p = _t.cuda.get_device_properties(0)
    sm = "sm_%d%d" % (p.major, p.minor)
    if sm in _t.cuda.get_arch_list():
        return
    print("installing PyTorch with %s support ..." % sm, flush=True)
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.5.1", "torchvision==0.20.1",
                        "--index-url", "https://download.pytorch.org/whl/cu121"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stderr or "")[-1500:])
        raise RuntimeError("torch install failed")
    os.environ["DERMIL_TORCH_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


ensure_torch()

INPUT, WORK = "/kaggle/input", "/kaggle/working"
CODE = os.environ.get("DERMIL_CODE")
if CODE and CODE not in sys.path:
    sys.path.insert(0, CODE)

RUN = os.environ.get("DERMIL_RUN", "hires")
DER_MODELS = [m.strip() for m in
              os.environ.get("DERMIL_MODELS", "der_mil,mr_mil").split(",") if m.strip()]
# RCAF has been removed from the project; a baseline is opt-in now.
BASELINE = os.environ.get("DERMIL_BASELINE", "").strip()
ALL_MODELS = DER_MODELS + ([BASELINE] if BASELINE else [])


def find_dir(root, *needles):
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            p = os.path.join(dp, d)
            if all(n.lower() in p.lower() for n in needles):
                return p
    return None


THYROID = find_dir(INPUT, "thyroidxl")
TN5000 = find_dir(INPUT, "main data") or find_dir(INPUT, "main-data")
print("thyroid :", THYROID)
print("tn5000  :", TN5000)

# ---- merge every checkpoint source into one namespace ---------------------- #
CKPT = "/tmp/ckpt"
os.makedirs(CKPT, exist_ok=True)
found = {}
for dp, dn, _fn in os.walk(INPUT):
    for ns in dn:
        for m in ALL_MODELS:
            for sub in ("fold0", "final"):
                src = os.path.join(dp, ns, m, sub, "best.pt")
                if os.path.isfile(src) and m not in found:
                    found[m] = os.path.join(dp, ns, m)
for m, src in found.items():
    dst = os.path.join(CKPT, RUN, m)
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)
    print("  %-9s <- %s" % (m, src))
missing = [m for m in ALL_MODELS if m not in found]
if missing:
    print("  WARNING no checkpoints for:", missing)
ALL_MODELS = [m for m in ALL_MODELS if m in found]
DER_MODELS = [m for m in DER_MODELS if m in found]

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import torch                                                    # noqa: E402
from src import pipeline                                        # noqa: E402
from src.config import Config                                   # noqa: E402
from src.data.splits import fold_frames, prevalence, test_frame  # noqa: E402
from src.engine.cv import (calibration_table, fit_calibration,   # noqa: E402
                           predict_with_checkpoints)
from src.eval.calibration import apply_temperature, sigmoid      # noqa: E402
from src.eval.metrics import all_metrics, bootstrap_ci, delong_test  # noqa: E402
from src.eval.reporting import threshold_table                   # noqa: E402
from src.eval.robustness import (frame_removal_study, mask_quality_sweep,  # noqa: E402
                                 permutation_invariance, shortcut_sensitivity)
from src.eval.counterfactual import (evidence_ablation,          # noqa: E402
                                     reliability_influence_correlation)
from src.utils.checkpoint import StageRegistry                   # noqa: E402
from src.utils.common import log, save_json, set_seed            # noqa: E402

RESULTS = os.path.join(WORK, "results")
cfg = Config()
cfg.data.thyroidxl_root = THYROID
cfg.data.tn5000_root = TN5000
cfg.data.num_workers = 4
cfg.run.run_name = RUN
cfg.run.ckpt_root = os.path.join(WORK, "registry")
cfg.run.results_root = RESULTS
cfg.model.evidence_mode = os.environ.get("DERMIL_EVIDENCE_MODE", "masked_input")
if os.environ.get("DERMIL_REGIONS"):
    cfg.model.regions = tuple(r.strip() for r in os.environ["DERMIL_REGIONS"].split(","))
print("config: evidence_mode=%s regions=%s" % (cfg.model.evidence_mode, cfg.model.regions))

set_seed(cfg.run.seed)
registry = StageRegistry(cfg.run.ckpt_root)
manifest = pipeline.prepare_thyroidxl(cfg, registry)
manifest = pipeline.cache_images(cfg, manifest, registry, "/tmp/dermil_cache")


def cks(model, sub="final"):
    p = os.path.join(CKPT, RUN, model, sub, "best.pt")
    return [p] if os.path.exists(p) else []


# =========================================================================== #
banner("1-2. PREDICTIONS AND DEVELOPMENT-ONLY CALIBRATION")
summary = {}
for model in ALL_MODELS:
    out_dir = os.path.join(RESULTS, RUN, model)
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for k in range(cfg.eval.n_folds):
        cp = cks(model, "fold%d" % k)
        if not cp:
            continue
        _tr, va = fold_frames(manifest, k)
        lg, y, _e = predict_with_checkpoints(cfg, va, model, cp)
        frames.append(pd.DataFrame({"patient_id": va["patient_id"].astype(str).values,
                                    "label": y.astype(int), "logit": lg, "fold": k}))
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(os.path.join(out_dir, "oof.csv"), index=False)
    pooled = all_metrics(oof["label"].values, sigmoid(oof["logit"].values))
    log("%-9s pooled OOF ROC-AUC %.4f" % (model, pooled["roc_auc"]))

    # calibration fitted on OOF only, then frozen
    bundle = fit_calibration(cfg, oof)
    save_json(bundle.to_dict(), os.path.join(out_dir, "calibration.json"))

    te = test_frame(manifest)
    lg, y, _e = predict_with_checkpoints(cfg, te, model, cks(model, "final") or cks(model, "fold0"))
    p_cal = apply_temperature(lg, bundle.temperature)
    pd.DataFrame({"patient_id": te["patient_id"].astype(str).values,
                  "label": y.astype(int), "logit": lg, "p_raw": sigmoid(lg),
                  "p_cal": p_cal}).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False)

    ct = calibration_table(oof, pd.read_csv(os.path.join(out_dir, "test_predictions.csv")),
                           bundle, cfg.eval.ece_bins)
    ct.to_csv(os.path.join(out_dir, "calibration_table.csv"), index=False)
    tt = threshold_table(cfg, model)
    tt.to_csv(os.path.join(out_dir, "threshold_table.csv"), index=False)
    log("\ncalibration (%s)\n%s" % (model, ct.to_string(index=False)))
    log("\noperating points (%s)\n%s" % (model, tt.to_string(index=False)))

    tm = all_metrics(y, sigmoid(lg))
    summary[model] = {"oof": pooled, "test": tm,
                      "temperature": bundle.temperature,
                      "thresholds": bundle.thresholds}

banner("PAIRED DeLong ON THE TEST COHORT")
for i in range(len(ALL_MODELS)):
    for j in range(i + 1, len(ALL_MODELS)):
        a, b = ALL_MODELS[i], ALL_MODELS[j]
        ta = pd.read_csv(os.path.join(RESULTS, RUN, a, "test_predictions.csv"))
        tb = pd.read_csv(os.path.join(RESULTS, RUN, b, "test_predictions.csv"))
        m = ta.merge(tb, on="patient_id", suffixes=("_a", "_b"))
        d = delong_test(m["label_a"].values, m["p_raw_a"].values, m["p_raw_b"].values)
        print("  %-9s %.4f vs %-9s %.4f | diff %+.4f | z=%+.2f p=%.4f"
              % (a, d["auc1"], b, d["auc2"], d["diff"], d["z"], d["p_value"]))

# =========================================================================== #
banner("3. ROBUSTNESS AND COUNTERFACTUAL SUITE (final models)")
ck_map = {m: cks(m, "final") or cks(m, "fold0") for m in ALL_MODELS}
try:
    shortcut_sensitivity(cfg, manifest, ck_map, registry)
except Exception:
    import traceback; traceback.print_exc()
for m in ALL_MODELS:
    for fn in (mask_quality_sweep, permutation_invariance, frame_removal_study):
        if left() < 600:
            log("budget low -- skipping remaining robustness")
            break
        try:
            fn(cfg, manifest, m, ck_map[m], registry)
        except Exception:
            import traceback; traceback.print_exc()
for m in DER_MODELS:
    if left() < 900:
        break
    try:
        evidence_ablation(cfg, manifest, m, ck_map[m], registry)
        reliability_influence_correlation(cfg, manifest, m, ck_map[m][0], registry)
    except Exception:
        import traceback; traceback.print_exc()

# =========================================================================== #
banner("4. TN5000 EXTERNAL VALIDATION")
if TN5000 and left() > 1800:
    from src.external.tn5000 import (build_tn5000_manifest, evaluate_with_tta,
                                     external_validation, force_bbox_masks)
    try:
        tn = build_tn5000_manifest(cfg, registry)

        # ---- zero-shot: no adaptation at all. A far stronger generalisation
        # claim than the adapted number, and it costs only inference.
        from src.data.tn5000 import official_eval_subset
        _adapt, ev = official_eval_subset(tn, cfg.external.eval_subset_per_class,
                                          cfg.run.seed)
        banner("TN5000 ZERO-SHOT (no domain adaptation)")
        zs = {}
        for m in ALL_MODELS:
            r = evaluate_with_tta(cfg, m, ck_map[m][0], ev, cfg.eval.tta_views)
            met = all_metrics(r["y"], r["p"])
            ci = bootstrap_ci(r["y"], r["p"], 1000, 0.5, seed=cfg.run.seed)
            zs[m] = {"metrics": met, "ci": {k: list(v) for k, v in ci.items()}}
            log("  %-9s zero-shot AUC %.4f [%.4f, %.4f] | F1 %.4f"
                % (m, met["roc_auc"], ci["roc_auc"][1], ci["roc_auc"][2], met["f1"]))
            pd.DataFrame({"patient_id": ev["patient_id"].astype(str).values,
                          "label": r["y"].astype(int), "p": r["p"]}).to_csv(
                os.path.join(RESULTS, RUN, m, "tn5000_zeroshot.csv"), index=False)
        save_json(zs, os.path.join(RESULTS, RUN, "tn5000_zeroshot.json"))

        # ---- domain-adapted, matching the source paper's protocol ----------
        ARMS = [a.strip() for a in
                os.environ.get("DERMIL_ARMS", "bbox").split(",") if a.strip()]
        log("arms requested: %s" % ", ".join(ARMS))

        variants = tuple(v for v in ("bbox", "unet") if v in ARMS)
        unet_man = None
        if "unet" in variants:
            from src.external.segmentation import unet_mask_manifest
            unet_man = unet_mask_manifest(cfg, manifest, tn, registry)

        for m in ALL_MODELS:
            if left() < 2400:
                log("budget low -- stopping before %s adaptation" % m)
                break
            banner("TN5000 DOMAIN ADAPTATION: " + m)
            if variants:
                df = external_validation(cfg, m, ck_map[m][0], tn, registry,
                                         mask_variants=variants,
                                         unet_manifest=unet_man)
                log(chr(10) + df.to_string(index=False))

            if "labelfree" in ARMS and left() > 1800:
                from src.external.adaptation import run_label_free_arms
                a_df, e_df = official_eval_subset(
                    force_bbox_masks(tn), cfg.external.eval_subset_per_class,
                    cfg.run.seed)
                log(chr(10) + run_label_free_arms(
                    cfg, m, ck_map[m][0], a_df, e_df, registry).to_string(index=False))

            if "retrieval" in ARMS and left() > 900:
                from src.external.retrieval import (build_retrieval_bags,
                                                    evaluate_retrieval_bags)
                a_df, e_df = official_eval_subset(
                    force_bbox_masks(tn), cfg.external.eval_subset_per_class,
                    cfg.run.seed)
                bags = build_retrieval_bags(cfg, m, ck_map[m][0], a_df, e_df, registry)
                r = evaluate_retrieval_bags(cfg, m, ck_map[m][0], bags,
                                            cfg.eval.tta_views)
                met = all_metrics(r["y"], r["p"])
                log("  retrieval bags k=%d: AUC %.4f | F1 %.4f"
                    % (cfg.adapt.retrieval_bag_size, met["roc_auc"], met["f1"]))
                pd.DataFrame({"patient_id": bags["patient_id"].astype(str).values,
                              "label": r["y"].astype(int), "p": r["p"]}).to_csv(
                    os.path.join(RESULTS, RUN, m,
                                 "tn5000_retrieval_predictions.csv"), index=False)
    except Exception:
        import traceback; traceback.print_exc()
else:
    log("TN5000 skipped (dataset missing or budget exhausted)")

# =========================================================================== #
banner("REPORT")
try:
    pipeline.build_report(cfg, manifest, ALL_MODELS,
                          DER_MODELS[0] if DER_MODELS else ALL_MODELS[0])
except Exception:
    import traceback; traceback.print_exc()

save_json(summary, os.path.join(RESULTS, RUN, "final_summary.json"))
banner("DONE in %.2f h" % (elapsed() / 3600))
print("registry:\n" + registry.summary())
n = sum(len(f) for _d, _s, f in os.walk(WORK))
print("output files:", n)
