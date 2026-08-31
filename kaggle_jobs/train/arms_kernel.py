"""TN5000 external-validation arms 2, 4, 5 and 6.

Inference and light adaptation only -- no model is trained from scratch here, so
this runs off the existing ThyroidXL checkpoints and does not depend on which
model ends up being the proposal.

    arm 2  U-Net pixel masks      segmenter trained on ThyroidXL, applied to
                                  TN5000; the arm the source paper's headline
                                  0.914 was produced with
    arm 4  pseudo-labeling        self-training gated on confidence AND
                                  across-view stability; no target labels
    arm 5  TENT                    entropy minimisation on normalisation
                                  affine params; no target labels
    arm 6  retrieval bags          neighbours from the adaptation pool complete
                                  each query into a real bag, so MIL is not
                                  inert on a one-image-per-case dataset

Deliberately NOT here: arm 3 (foundation encoders) and arm 7 (der_mil_vl) both
retrain models from scratch and should wait until the proposal is settled.

Sections 1-3 of the finishing run (predictions, calibration, robustness) are
already done and are not repeated.
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
MODELS = [m.strip() for m in
          os.environ.get("DERMIL_MODELS", "der_mil,mr_mil").split(",") if m.strip()]
ARMS = [a.strip() for a in
        os.environ.get("DERMIL_ARMS", "bbox,unet,labelfree,retrieval").split(",")
        if a.strip()]


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
print("arms    :", ", ".join(ARMS))

CKPT = "/tmp/ckpt"
os.makedirs(CKPT, exist_ok=True)
found = {}
for dp, dn, _fn in os.walk(INPUT):
    for ns in dn:
        for m in MODELS:
            for sub in ("final", "fold0"):
                src = os.path.join(dp, ns, m, sub, "best.pt")
                if os.path.isfile(src) and m not in found:
                    found[m] = os.path.join(dp, ns, m)
for m, src in found.items():
    dst = os.path.join(CKPT, RUN, m)
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)
    print("  %-9s <- %s" % (m, src))
MODELS = [m for m in MODELS if m in found]
if not MODELS:
    raise SystemExit("no checkpoints found -- chain from the training kernel")

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
from src import pipeline                                        # noqa: E402
from src.config import Config                                   # noqa: E402
from src.data.tn5000 import official_eval_subset                # noqa: E402
from src.eval.metrics import all_metrics, bootstrap_ci, delong_test  # noqa: E402
from src.external.tn5000 import (build_tn5000_manifest, external_validation,  # noqa: E402
                                 force_bbox_masks)
from src.utils.checkpoint import StageRegistry                  # noqa: E402
from src.utils.common import log, save_json, set_seed           # noqa: E402

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


def ck(model):
    for sub in ("final", "fold0"):
        p = os.path.join(CKPT, RUN, model, sub, "best.pt")
        if os.path.exists(p):
            return p
    return None


# =========================================================================== #
banner("TN5000 MANIFEST")
tn = build_tn5000_manifest(cfg, registry)

# ---- arm 2: train the segmenter once, share it across models -------------- #
unet_man = None
if "unet" in ARMS and left() > 3600:
    from src.external.segmentation import unet_mask_manifest
    try:
        unet_man = unet_mask_manifest(cfg, manifest, tn, registry)
    except Exception:
        import traceback
        traceback.print_exc()
    if unet_man is None:
        log("U-Net arm unavailable -- continuing without it")
        ARMS = [a for a in ARMS if a != "unet"]

variants = tuple(v for v in ("bbox", "unet") if v in ARMS)

# =========================================================================== #
summary = {}
for m in MODELS:
    c = ck(m)
    banner("MODEL: %s" % m)

    if variants and left() > 2400:
        try:
            df = external_validation(cfg, m, c, tn, registry,
                                     mask_variants=variants, unet_manifest=unet_man)
            log("\n" + df.to_string(index=False))
            summary.setdefault(m, {})["mask_arms"] = df.to_dict("records")
        except Exception:
            import traceback
            traceback.print_exc()

    if "labelfree" in ARMS and left() > 2400:
        try:
            from src.external.adaptation import run_label_free_arms
            a_df, e_df = official_eval_subset(
                force_bbox_masks(tn), cfg.external.eval_subset_per_class, cfg.run.seed)
            df = run_label_free_arms(cfg, m, c, a_df, e_df, registry)
            log("\n" + df.to_string(index=False))
            summary.setdefault(m, {})["label_free"] = df.to_dict("records")
        except Exception:
            import traceback
            traceback.print_exc()

    if "retrieval" in ARMS and left() > 1200:
        try:
            from src.external.retrieval import (build_retrieval_bags,
                                                evaluate_retrieval_bags)
            a_df, e_df = official_eval_subset(
                force_bbox_masks(tn), cfg.external.eval_subset_per_class, cfg.run.seed)
            bags = build_retrieval_bags(cfg, m, c, a_df, e_df, registry)
            r = evaluate_retrieval_bags(cfg, m, c, bags, cfg.eval.tta_views)
            met = all_metrics(r["y"], r["p"])
            ci = bootstrap_ci(r["y"], r["p"], cfg.eval.bootstrap_external, 0.5,
                              seed=cfg.run.seed)
            log("  retrieval bags k=%d: AUC %.4f [%.3f-%.3f] | F1 %.4f"
                % (cfg.adapt.retrieval_bag_size, met["roc_auc"],
                   ci["roc_auc"][1], ci["roc_auc"][2], met["f1"]))
            summary.setdefault(m, {})["retrieval"] = {
                "auc": met["roc_auc"], "f1": met["f1"],
                "ci": [ci["roc_auc"][1], ci["roc_auc"][2]],
                "bag_size": cfg.adapt.retrieval_bag_size}
            pd.DataFrame({"patient_id": bags["patient_id"].astype(str).values,
                          "label": r["y"].astype(int), "p": r["p"]}).to_csv(
                os.path.join(RESULTS, RUN, m,
                             "tn5000_retrieval_predictions.csv"), index=False)
        except Exception:
            import traceback
            traceback.print_exc()

# =========================================================================== #
banner("ARM LADDER SUMMARY")
rows = []
for m, blocks in summary.items():
    for rec in blocks.get("mask_arms", []):
        rows.append({"model": m, "arm": rec.get("mask_type"),
                     "target_labels": "yes",
                     "auc": rec.get("roc_auc"), "ci": rec.get("roc_auc_ci")})
    for rec in blocks.get("label_free", []):
        rows.append({"model": m, "arm": rec.get("arm"), "target_labels": "no",
                     "auc": rec.get("roc_auc"), "ci": rec.get("roc_auc_ci")})
    if "retrieval" in blocks:
        b = blocks["retrieval"]
        rows.append({"model": m, "arm": "retrieval_k%d" % b["bag_size"],
                     "target_labels": "no",
                     "auc": b["auc"],
                     "ci": "[%.3f-%.3f]" % (b["ci"][0], b["ci"][1])})
if rows:
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    out = os.path.join(RESULTS, RUN, "_tables")
    os.makedirs(out, exist_ok=True)
    tbl.to_csv(os.path.join(out, "tn5000_arm_ladder.csv"), index=False)

save_json(summary, os.path.join(RESULTS, RUN, "tn5000_arms_summary.json"))
print("""
Reading the ladder
------------------
`target_labels = yes` arms fine-tune on TN5000 labels. They measure whether the
model can be TAUGHT the domain.

`target_labels = no` arms (zero_shot, upl, upl_tent, retrieval) never see a
TN5000 label. They measure whether it GENERALISES to the domain, which is the
stronger claim and the one the source paper does not report.

An `upl (degenerate)` row means every candidate failed the confidence or
stability gate, so no pseudo-labels were fitted and that rung equals zero-shot.
""")
banner("DONE in %.2f h" % (elapsed() / 3600))
print("registry:\n" + registry.summary())
