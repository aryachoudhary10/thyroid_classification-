"""DER-MIL training kernel for Kaggle.

Session chaining
----------------
A Kaggle GPU session is capped (~12 h) and cannot be resumed in place. The
pattern used here:

    version N   writes checkpoints + results into /kaggle/working
                -> that becomes the kernel's OUTPUT
    version N+1 attaches the previous version's output via `kernel_sources`
                -> copies it back into /kaggle/working and continues

`StageRegistry` then reports every finished fold as SKIP, and the interrupted
fold resumes from `last.pt`. Re-pushing the same kernel is therefore the resume
mechanism: no GPU work is ever repeated.

A wall-clock budget stops training between folds while there is still time to
flush the output, because a session killed mid-write loses the whole session.
"""
import os
import shutil
import sys
import time

T0 = time.time()

# ---- budget: stop cleanly well before Kaggle kills the session -------------- #
BUDGET_HOURS = float(os.environ.get("DERMIL_BUDGET_HOURS", "8.5"))
BUDGET_S = BUDGET_HOURS * 3600
RESERVE_S = 12 * 60          # leave time to write outputs


def elapsed():
    return time.time() - T0


def time_left():
    return BUDGET_S - RESERVE_S - elapsed()


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)



# --------------------------------------------------------------------------- #
# 0. GPU / PyTorch compatibility guard
#
# Kaggle allocates a Tesla P100 to API-pushed kernels and the accelerator cannot
# be chosen through the API (every machine_shape token normalises to "Gpu").
# The P100 is Pascal (sm_60), and the preinstalled PyTorch 2.10 build only ships
# kernels for sm_70+, so the very first CUDA op dies with
# "no kernel image is available for execution on the device".
#
# Fix: detect the mismatch, install a build that still supports sm_60, and
# re-exec this script so the new torch is actually the one in memory.
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
    arches = _t.cuda.get_arch_list()
    print("GPU: %s (%s) | torch %s supports %s" % (p.name, sm, _t.__version__, arches))
    if sm in arches:
        return

    print("INCOMPATIBLE: installing a PyTorch build with %s support ..." % sm,
          flush=True)
    import subprocess
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.5.1", "torchvision==0.20.1",
                        "--index-url", "https://download.pytorch.org/whl/cu121"],
                       capture_output=True, text=True)
    print("pip rc=%d in %.1f min" % (r.returncode, (time.time() - t0) / 60), flush=True)
    if r.returncode != 0:
        print((r.stderr or "")[-2000:])
        raise RuntimeError("could not install a GPU-compatible PyTorch")
    os.environ["DERMIL_TORCH_FIXED"] = "1"
    print("re-executing with the new PyTorch ...", flush=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)


ensure_torch_supports_gpu()

# --------------------------------------------------------------------------- #
# 1. locate inputs
# --------------------------------------------------------------------------- #
INPUT = "/kaggle/input"
WORK = "/kaggle/working"


def find_dir(root, *needles):
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            p = os.path.join(dp, d)
            if all(n.lower() in p.lower() for n in needles):
                return p
    return None


def find_code_root():
    """The dermil-code dataset: a directory containing src/config.py."""
    for dp, dn, _fn in os.walk(INPUT):
        if "src" in dn and os.path.isfile(os.path.join(dp, "src", "config.py")):
            return dp
    return None


# The bootstrap header of a self-contained kernel sets DERMIL_CODE; otherwise
# fall back to a `dermil-code` dataset attached as an input.
CODE = os.environ.get("DERMIL_CODE") or find_code_root()
if CODE is None:
    raise RuntimeError("no source tree found (need src/config.py)")
if CODE not in sys.path:
    sys.path.insert(0, CODE)
print("code    :", CODE)

THYROID = find_dir(INPUT, "thyroidxl")
TN5000 = find_dir(INPUT, "main data") or find_dir(INPUT, "main-data")
print("thyroid :", THYROID)
print("tn5000  :", TN5000)

# --------------------------------------------------------------------------- #
# 2. restore the previous session's checkpoints
# --------------------------------------------------------------------------- #
CKPT = os.path.join(WORK, "checkpoints")
RESULTS = os.path.join(WORK, "results")

prev = None
for dp, dn, _fn in os.walk(INPUT):
    if "checkpoints" in dn and os.path.isfile(
            os.path.join(dp, "checkpoints", "registry.json")):
        prev = dp
        break

if prev:
    banner("RESUMING FROM A PREVIOUS SESSION: " + prev)
    for sub in ("checkpoints", "results"):
        src = os.path.join(prev, sub)
        dst = os.path.join(WORK, sub)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            n = sum(len(f) for _d, _s, f in os.walk(dst))
            print("  restored %-12s %5d files" % (sub, n))
else:
    banner("FRESH RUN (no previous checkpoints attached)")
os.makedirs(CKPT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# --------------------------------------------------------------------------- #
# 3. configure
# --------------------------------------------------------------------------- #
import torch                                                  # noqa: E402
from src.config import Config                                 # noqa: E402
from src import pipeline                                      # noqa: E402
from src.engine.cv import train_fold                          # noqa: E402
from src.utils.checkpoint import StageRegistry                # noqa: E402
from src.utils.common import log, set_seed                    # noqa: E402

print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

MODELS = os.environ.get(
    "DERMIL_MODELS", "rcaf,mr_mil,der_mil,lesion_mil,image_mil").split(",")
MODELS = [m.strip() for m in MODELS if m.strip()]
FINAL_MODE = os.environ.get("DERMIL_FINAL_MODE", "refit")
RUN_EXTERNAL = os.environ.get("DERMIL_EXTERNAL", "1") == "1"
RUN_ROBUST = os.environ.get("DERMIL_ROBUST", "1") == "1"

cfg = Config()
# The calibration/main run used the default roi_pool encoder, which pools region
# embeddings from the 14x14 / 7x7 feature maps. Measured lesion area is 6.75% of
# the frame, so the margin band lands under one feature-map cell -- setting
# masked_input runs each region masked at full 224 resolution instead (K
# backbone passes per frame, ~4x the compute).
EVIDENCE_MODE = os.environ.get("DERMIL_EVIDENCE_MODE", "")
# regions "lesion,margin,peri,global" makes the evidence set a strict superset
# of RCAF's two branches: lesion == x*M, global == x.
REGIONS = os.environ.get("DERMIL_REGIONS", "")
EVIDENCE_FUSION = os.environ.get("DERMIL_EVIDENCE_FUSION", "")
# Backbone sweep (arm 3). Checkpoints are backbone-shaped, so a swap needs its
# own DERMIL_RUN -- otherwise the registry reports the ResNet-50 folds as done,
# skips training, and then tries to load those weights into the new trunk.
BACKBONE = os.environ.get("DERMIL_BACKBONE", "")
cfg.data.thyroidxl_root = THYROID
cfg.data.tn5000_root = TN5000
cfg.data.num_workers = 4
cfg.run.run_name = os.environ.get("DERMIL_RUN", "main")
cfg.run.ckpt_root = CKPT
cfg.run.results_root = RESULTS
cfg.run.save_every_steps = 100          # frequent, because sessions die abruptly
if os.environ.get("DERMIL_DEBUG"):
    cfg.run.debug_subset = int(os.environ["DERMIL_DEBUG"])
    cfg.optim.stage1_epochs = 1
    cfg.optim.stage2_epochs = 1
    cfg.eval.n_folds = 2

if EVIDENCE_MODE:
    cfg.model.evidence_mode = EVIDENCE_MODE
    print("evidence_mode override -> %s" % EVIDENCE_MODE)
if REGIONS:
    cfg.model.regions = tuple(r.strip() for r in REGIONS.split(",") if r.strip())
    print("regions override -> %s" % (cfg.model.regions,))
if EVIDENCE_FUSION:
    cfg.model.evidence_fusion = EVIDENCE_FUSION
    print("evidence_fusion override -> %s" % EVIDENCE_FUSION)
if BACKBONE:
    if BACKBONE != "resnet50" and cfg.run.run_name in ("main", "hires"):
        raise SystemExit(
            "refusing to run backbone '%s' under DERMIL_RUN=%s: those namespaces "
            "hold ResNet-50 checkpoints and the registry would skip training and "
            "then load incompatible weights. Use e.g. DERMIL_RUN=%s."
            % (BACKBONE, cfg.run.run_name, BACKBONE.replace("_", "")))
    cfg.model.backbone = BACKBONE
    print("backbone override -> %s" % BACKBONE)

set_seed(cfg.run.seed)
registry = StageRegistry(cfg.run.ckpt_root)
print("\nregistry on entry:\n" + registry.summary())
print("\nmodels:", MODELS, "| final_mode:", FINAL_MODE,
      "| budget: %.1f h" % BUDGET_HOURS)

# --------------------------------------------------------------------------- #
# 4. data
# --------------------------------------------------------------------------- #
banner("DATA")
manifest = pipeline.prepare_thyroidxl(cfg, registry)

# One-time resize cache. The calibration run measured ~60% of epoch wall-clock
# going into PNG decode; caching at 224 removes it for every later fold/model.
# The cache lives in /kaggle/temp, NOT /kaggle/working: everything under
# working becomes the kernel output, and carrying ~600 MB of derived pixels
# between sessions costs more than the ~3 min it takes to rebuild them.
# cache_images detects the missing files and rebuilds automatically.
USE_CACHE = os.environ.get("DERMIL_CACHE", "1") == "1"
if USE_CACHE:
    # NOT /kaggle/working: everything there becomes the kernel output, and the
    # 23k cache files overflowed it twice, silently dropping results/ from the
    # saved output. /kaggle/temp does not exist on Kaggle images; /tmp does.
    scratch = "/tmp"
    CACHE_DIR = os.path.join(scratch, "dermil_cache")
    t_cache = time.time()
    manifest = pipeline.cache_images(cfg, manifest, registry, CACHE_DIR)
    log("resize cache ready in %.1f min" % ((time.time() - t_cache) / 60))

# --------------------------------------------------------------------------- #
# 5. fold-by-fold training under a wall-clock budget
# --------------------------------------------------------------------------- #
banner("TRAINING")
fold_times = []
stopped_early = False

# A calibration session trains a single fold to measure the real cost before
# committing the full budget.
MAX_FOLDS = int(os.environ.get("DERMIL_MAX_FOLDS", "0")) or cfg.eval.n_folds

for model in MODELS:
    for fold in range(min(MAX_FOLDS, cfg.eval.n_folds)):
        key = "cv/%s/%s/fold%d" % (cfg.run.run_name, model, fold)
        if registry.is_done(key):
            continue
        # Only start a fold we can plausibly finish; otherwise the session dies
        # mid-fold and the partial epoch has to be replayed next time.
        need = max(fold_times) if fold_times else 0.0
        if time_left() < need * 1.15 and fold_times:
            log("budget: %.2f h left, a fold takes up to %.2f h -- stopping here"
                % (time_left() / 3600, need / 3600))
            stopped_early = True
            break
        t0 = time.time()
        train_fold(cfg, manifest, model, fold, registry)
        fold_times.append(time.time() - t0)
        log("fold wall-clock %.1f min | elapsed %.2f h | budget left %.2f h"
            % (fold_times[-1] / 60, elapsed() / 3600, time_left() / 3600))
        if time_left() <= 0:
            stopped_early = True
            break
    if stopped_early:
        break

# --------------------------------------------------------------------------- #
# 6. calibration + single-shot test, only for models whose folds are complete
# --------------------------------------------------------------------------- #
def folds_done(model):
    return all(registry.is_done("cv/%s/%s/fold%d" % (cfg.run.run_name, model, f))
               for f in range(cfg.eval.n_folds))


ready = [m for m in MODELS if folds_done(m)]
print("\nmodels with all folds complete:", ready)

for model in ready:
    if time_left() <= 0:
        stopped_early = True
        break
    try:
        pipeline.run_model(cfg, manifest, registry, model, FINAL_MODE)
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        print("run_model(%s) failed:" % model)
        traceback.print_exc()

# --------------------------------------------------------------------------- #
# 7. robustness + external, only once the core models are done
# --------------------------------------------------------------------------- #
if RUN_ROBUST and not stopped_early and time_left() > 1800 and len(ready) >= 2:
    try:
        pipeline.run_robustness(cfg, manifest, registry, ready,
                                proposed="der_mil" if "der_mil" in ready else ready[0],
                                final_mode=FINAL_MODE)
    except Exception:                                          # noqa: BLE001
        import traceback
        traceback.print_exc()

if RUN_EXTERNAL and not stopped_early and time_left() > 1800 and TN5000:
    ext = [m for m in ("rcaf", "der_mil") if m in ready]
    if ext:
        try:
            pipeline.run_external(cfg, registry, ext)
        except Exception:                                      # noqa: BLE001
            import traceback
            traceback.print_exc()

# --------------------------------------------------------------------------- #
# 8. report + handover
# --------------------------------------------------------------------------- #
if ready:
    try:
        pipeline.build_report(cfg, manifest, ready,
                              "der_mil" if "der_mil" in ready else ready[0])
    except Exception:                                          # noqa: BLE001
        import traceback
        traceback.print_exc()

banner("SESSION SUMMARY")
print("elapsed        : %.2f h of %.1f h budget" % (elapsed() / 3600, BUDGET_HOURS))
print("folds trained  : %d this session" % len(fold_times))
print("stopped early  :", stopped_early)
print("\nregistry on exit:\n" + registry.summary())

remaining = [(m, f) for m in MODELS for f in range(cfg.eval.n_folds)
             if not registry.is_done("cv/%s/%s/fold%d" % (cfg.run.run_name, m, f))]
print("\nfolds still to do: %d" % len(remaining))
for m, f in remaining[:20]:
    print("   %s fold%d" % (m, f))
if remaining:
    print("\n>>> Push this kernel again to continue from here. <<<")
else:
    print("\n>>> All folds complete. <<<")

n_files = sum(len(f) for _d, _s, f in os.walk(WORK))
size = sum(os.path.getsize(os.path.join(d, f))
           for d, _s, fs in os.walk(WORK) for f in fs)
print("\noutput: %d files, %.2f GB" % (n_files, size / 1e9))
