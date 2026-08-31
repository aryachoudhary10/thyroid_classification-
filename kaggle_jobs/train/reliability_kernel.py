"""Decisive re-run: does reliability predict counterfactual influence?

Short kernel, one question. The previous run reported a within-patient Spearman
of -0.75 for DER-MIL with 0 of 400 patients positive, which read as a
falsification of the whole reliability claim. That statistic pooled frames and
regions into a single ranking, and regions differ in influence by two orders of
magnitude (core/margin ~0.11 mean |dp|, peri/global ~0.001), so it was dominated
by the region axis and never tested the cross-view claim at all. A single-frame
bag with K=4 regions even satisfied its >=3-token guard.

The corrected statistic holds the region fixed and varies the frame, which is
what the claim actually says. This kernel re-runs only that stage.

Everything here is inference on existing checkpoints. No training. Expect ~15
minutes, and read `cross_view` in the output, not `pooled_tokens`.
"""
import os
import shutil
import sys
import time

T0 = time.time()


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
MAX_PATIENTS = int(os.environ.get("DERMIL_INFLUENCE_PATIENTS", "700"))


def find_dir(root, *needles):
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            p = os.path.join(dp, d)
            if all(n.lower() in p.lower() for n in needles):
                return p
    return None


THYROID = find_dir(INPUT, "thyroidxl")
print("thyroid :", THYROID)

# ---- merge every checkpoint source into one namespace ---------------------- #
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
missing = [m for m in MODELS if m not in found]
if missing:
    print("  WARNING no checkpoints for:", missing)
MODELS = [m for m in MODELS if m in found]
if not MODELS:
    raise SystemExit("no checkpoints found -- attach the checkpoint dataset")

import json                                                      # noqa: E402
from src import pipeline                                         # noqa: E402
from src.config import Config                                    # noqa: E402
from src.eval.counterfactual import reliability_influence_correlation  # noqa: E402
from src.utils.checkpoint import StageRegistry                   # noqa: E402
from src.utils.common import log, save_json, set_seed            # noqa: E402

RESULTS = os.path.join(WORK, "results")
cfg = Config()
cfg.data.thyroidxl_root = THYROID
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
    return p if os.path.exists(p) else None


# =========================================================================== #
banner("RELIABILITY vs COUNTERFACTUAL INFLUENCE  (corrected)")
out = {}
for m in MODELS:
    ck = cks(m) or cks(m, "fold0")
    if not ck:
        print("  skipping %s -- no checkpoint" % m)
        continue
    # A previous registry may mark this done with the OLD statistic; force it.
    registry.reset("cf/influence/%s/%s" % (RUN, m))
    stale = os.path.join(RESULTS, RUN, m, "reliability_influence.json")
    if os.path.exists(stale):
        os.remove(stale)
    try:
        out[m] = reliability_influence_correlation(cfg, manifest, m, ck, registry,
                                                   max_patients=MAX_PATIENTS)
    except Exception:
        import traceback
        traceback.print_exc()

# =========================================================================== #
banner("VERDICT")
print("%-10s %10s %10s %8s %10s   %s"
      % ("model", "cross-rho", "frac+", "n", "p", "pooled-rho (old, confounded)"))
for m, r in out.items():
    cv = r.get("cross_view", {})
    pt = r.get("pooled_tokens", {})
    print("%-10s %10.4f %10.3f %8d %10.3g   %.4f"
          % (m, cv.get("mean_spearman", float("nan")),
             cv.get("frac_positive", float("nan")), cv.get("n", 0),
             cv.get("ttest_p", float("nan")), pt.get("mean_spearman", float("nan"))))

print("\nregion profile (is influence merely a property of the region?)")
for m, r in out.items():
    for name, d in (r.get("region_profile") or {}).items():
        print("  %-10s %-8s mean R %.4f  mean |dp| %.5f"
              % (m, name, d["mean_R"], d["mean_influence"]))

print("""
How to read this
----------------
cross-rho > 0 with p < 0.05 and frac+ well above 0.5
    reliability predicts influence across views. DER-MIL's central claim holds
    and DER-MIL is the proposed model; MR-MIL is its ablation.

cross-rho ~ 0
    the mechanism does not order evidence usefully. MR-MIL becomes the honest
    proposal and the reliability head becomes a reported negative result.

cross-rho < 0
    the mechanism is inverted. Worth one look at the reliability head's sign
    before concluding, then report as a negative result.

If the region profile shows high mean R where mean |dp| is near zero, that also
confirms the old -0.75 was the region confound rather than a model failure.
""")
save_json(out, os.path.join(RESULTS, RUN, "reliability_influence_corrected.json"))
banner("DONE in %.1f min" % ((time.time() - T0) / 60.0))
print("registry:\n" + registry.summary())
