"""Inference-only recovery of OOF and test predictions.

Two training sessions produced correct checkpoints but their `results/`
directory was dropped from the kernel output: the 224 px cache was written
under /kaggle/working, and its 23k files overflowed the saved output. The
model weights survived; only the derived predictions were lost.

Retraining to recover them would cost ~6 h. This kernel instead reloads each
fold's best checkpoint and re-runs inference, which costs minutes:

    fold k best.pt  -> predict on fold k's held-out patients  -> OOF
    refit best.pt   -> predict on the untouched test cohort   -> test

It writes only small CSVs, so the output stays tiny. Nothing is trained, so no
result can shift as a side effect of running it.
"""
import os
import sys
import time

T0 = time.time()


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


# --------------------------------------------------------------------------- #
def ensure_torch_supports_gpu():
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
        print((r.stderr or "")[-2000:])
        raise RuntimeError("torch install failed")
    os.environ["DERMIL_TORCH_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


ensure_torch_supports_gpu()

INPUT, WORK = "/kaggle/input", "/kaggle/working"
CODE = os.environ.get("DERMIL_CODE")
if CODE and CODE not in sys.path:
    sys.path.insert(0, CODE)


def find_dir(root, *needles):
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            p = os.path.join(dp, d)
            if all(n.lower() in p.lower() for n in needles):
                return p
    return None


THYROID = find_dir(INPUT, "thyroidxl")
print("thyroid :", THYROID)

# ---- locate a checkpoint root --------------------------------------------- #
# Two shapes must both work:
#   a) a chained kernel output ->  <root>/checkpoints/<run>/<model>/fold0/best.pt
#   b) an uploaded dataset     ->  <root>/<run>/<model>/fold0/best.pt
# (Kaggle strips the top-level folder when it extracts an uploaded zip, so the
# "checkpoints/" prefix is present in one case and absent in the other.)
RUN_ENV = os.environ.get("DERMIL_RUN", "hires")
MODELS_ENV = [m.strip() for m in
              os.environ.get("DERMIL_MODELS", "der_mil,mr_mil").split(",") if m.strip()]


def find_ckpt_root():
    probes = ["fold0", "final"]
    for dp, _dn, _fn in os.walk(INPUT):
        for m in MODELS_ENV:
            for sub in probes:
                if os.path.isfile(os.path.join(dp, RUN_ENV, m, sub, "best.pt")):
                    return dp
    return None


CKPT_SRC = find_ckpt_root()
if CKPT_SRC is None:
    print("searched under", INPUT)
    for dp, dn, fn in os.walk(INPUT):
        if dp.count(os.sep) <= 4:
            print("  ", dp, "dirs:", dn[:6])
    raise RuntimeError("no checkpoints found for run=%s models=%s"
                       % (RUN_ENV, MODELS_ENV))
CKPT = os.path.join(WORK, "registry")
os.makedirs(CKPT, exist_ok=True)
print("reading checkpoints from", CKPT_SRC)

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import torch                                                    # noqa: E402
from src import pipeline                                        # noqa: E402
from src.config import Config                                   # noqa: E402
from src.data.splits import fold_frames, test_frame             # noqa: E402
from src.engine.cv import predict_with_checkpoints              # noqa: E402
from src.eval.calibration import sigmoid                        # noqa: E402
from src.eval.metrics import all_metrics, bootstrap_ci, delong_test  # noqa: E402
from src.utils.checkpoint import StageRegistry                  # noqa: E402
from src.utils.common import log, save_json, set_seed           # noqa: E402

RUN, MODELS = RUN_ENV, MODELS_ENV
RESULTS = os.path.join(WORK, "results")

cfg = Config()
cfg.data.thyroidxl_root = THYROID
cfg.data.num_workers = 4
cfg.run.run_name = RUN
cfg.run.ckpt_root = CKPT
cfg.run.results_root = RESULTS
if os.environ.get("DERMIL_EVIDENCE_MODE"):
    cfg.model.evidence_mode = os.environ["DERMIL_EVIDENCE_MODE"]
if os.environ.get("DERMIL_REGIONS"):
    cfg.model.regions = tuple(r.strip() for r in os.environ["DERMIL_REGIONS"].split(","))
if os.environ.get("DERMIL_EVIDENCE_FUSION"):
    cfg.model.evidence_fusion = os.environ["DERMIL_EVIDENCE_FUSION"]
print("config: evidence_mode=%s regions=%s fusion=%s"
      % (cfg.model.evidence_mode, cfg.model.regions, cfg.model.evidence_fusion))

set_seed(cfg.run.seed)
registry = StageRegistry(CKPT)
manifest = pipeline.prepare_thyroidxl(cfg, registry)
manifest = pipeline.cache_images(cfg, manifest, registry, "/tmp/dermil_cache")

# --------------------------------------------------------------------------- #
def ckpt_path(model, sub):
    p = os.path.join(CKPT_SRC, RUN, model, sub, "best.pt")
    return p if os.path.exists(p) else None


summary = {}
for model in MODELS:
    banner("REGENERATING PREDICTIONS: " + model)
    out_dir = os.path.join(RESULTS, RUN, model)
    os.makedirs(out_dir, exist_ok=True)

    # ---- out-of-fold ------------------------------------------------------ #
    frames = []
    for k in range(cfg.eval.n_folds):
        cp = ckpt_path(model, "fold%d" % k)
        if cp is None:
            log("  fold%d checkpoint missing -- skipped" % k)
            continue
        _tr, va = fold_frames(manifest, k)
        lg, y, _e = predict_with_checkpoints(cfg, va, model, [cp])
        frames.append(pd.DataFrame({"patient_id": va["patient_id"].astype(str).values,
                                    "label": y.astype(int), "logit": lg, "fold": k}))
        log("  fold%d: %d patients, ROC-AUC %.4f"
            % (k, len(y), all_metrics(y, sigmoid(lg))["roc_auc"]))
    if not frames:
        continue
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(os.path.join(out_dir, "oof.csv"), index=False)

    per_fold = [all_metrics(g["label"].values, sigmoid(g["logit"].values))["roc_auc"]
                for _k, g in oof.groupby("fold")]
    pooled = all_metrics(oof["label"].values, sigmoid(oof["logit"].values))
    log("  POOLED OOF ROC-AUC %.4f | per-fold mean %.4f +/- %.4f"
        % (pooled["roc_auc"], float(np.mean(per_fold)), float(np.std(per_fold))))

    # ---- single-shot test with the refit model ---------------------------- #
    cp = ckpt_path(model, "final") or ckpt_path(model, "fold0")
    te = test_frame(manifest)
    lg, y, _e = predict_with_checkpoints(cfg, te, model, [cp])
    pd.DataFrame({"patient_id": te["patient_id"].astype(str).values,
                  "label": y.astype(int), "logit": lg,
                  "p_raw": sigmoid(lg)}).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False)
    tm = all_metrics(y, sigmoid(lg))
    ci = bootstrap_ci(y, sigmoid(lg), 2000, 0.5, seed=cfg.run.seed)
    log("  TEST ROC-AUC %.4f [%.4f, %.4f] | PR-AUC %.4f | F1 %.4f"
        % (tm["roc_auc"], ci["roc_auc"][1], ci["roc_auc"][2], tm["pr_auc"], tm["f1"]))

    summary[model] = {"oof_pooled": pooled, "oof_per_fold": per_fold, "test": tm,
                      "test_ci": {k: list(v) for k, v in ci.items()}}

# --------------------------------------------------------------------------- #
banner("HEAD TO HEAD")
names = [m for m in MODELS if m in summary]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        oa = pd.read_csv(os.path.join(RESULTS, RUN, a, "oof.csv"))
        ob = pd.read_csv(os.path.join(RESULTS, RUN, b, "oof.csv"))
        m = oa.merge(ob, on="patient_id", suffixes=("_a", "_b"))
        d = delong_test(m["label_a"].values, sigmoid(m["logit_a"].values),
                        sigmoid(m["logit_b"].values))
        print("OOF   %-9s %.4f vs %-9s %.4f | diff %+.4f | DeLong z=%.2f p=%.4f"
              % (a, d["auc1"], b, d["auc2"], d["diff"], d["z"], d["p_value"]))
        ta = pd.read_csv(os.path.join(RESULTS, RUN, a, "test_predictions.csv"))
        tb = pd.read_csv(os.path.join(RESULTS, RUN, b, "test_predictions.csv"))
        mt = ta.merge(tb, on="patient_id", suffixes=("_a", "_b"))
        dt = delong_test(mt["label_a"].values, mt["p_raw_a"].values, mt["p_raw_b"].values)
        print("TEST  %-9s %.4f vs %-9s %.4f | diff %+.4f | DeLong z=%.2f p=%.4f"
              % (a, dt["auc1"], b, dt["auc2"], dt["diff"], dt["z"], dt["p_value"]))

save_json(summary, os.path.join(RESULTS, RUN, "verify_summary.json"))
banner("DONE in %.1f min" % ((time.time() - T0) / 60))
n = sum(len(f) for _d, _s, f in os.walk(WORK))
print("output files:", n)
