"""Closes the ConvNeXt-Tiny gap: reliability mechanism + robustness suite.

These two analyses were only ever run on the ResNet-50 checkpoint. This
kernel is pure inference on the already-trained ConvNeXt-Tiny checkpoints
(lesion_mil, mr_mil, der_mil; final_mode="refit" resolves to the existing
refit checkpoint, never triggers training) -- it runs:

    1. the full robustness suite via pipeline.run_robustness(): mask-quality
       degradation, frame-order permutation, frame removal, within-patient
       mask shuffle (shortcut probe), for all three models; plus
       evidence-region ablation, corruption/reliability response, the
       reliability<->influence counterfactual test, and agreement/
       contradiction case mining, for der_mil (the proposed model).
    2. the SAME reliability<->influence counterfactual test for mr_mil too
       (pipeline.run_robustness only runs it for the proposed model), so the
       region-profile comparison panel in fig04 has both models on ConvNeXt,
       matching what exists for ResNet-50.
"""
import os
import shutil
import sys
import time

T0 = time.time()


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


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
        print((r.stderr or "")[-1500:])
        raise RuntimeError("torch install failed")
    os.environ["DERMIL_TORCH_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


ensure_torch_supports_gpu()

INPUT, WORK = "/kaggle/input", "/kaggle/working"
CODE = os.environ.get("DERMIL_CODE")
if CODE and CODE not in sys.path:
    sys.path.insert(0, CODE)

RUN = os.environ.get("DERMIL_RUN", "convnexttiny")
MODELS = [m.strip() for m in
          os.environ.get("DERMIL_MODELS", "lesion_mil,mr_mil,der_mil").split(",")
          if m.strip()]
PROPOSED = os.environ.get("DERMIL_PROPOSED", "der_mil")
INFLUENCE_PATIENTS = int(os.environ.get("DERMIL_INFLUENCE_PATIENTS", "700"))


def find_dir(root, *needles):
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            p = os.path.join(dp, d)
            if all(n.lower() in p.lower() for n in needles):
                return p
    return None


THYROID = find_dir(INPUT, "thyroidxl")
print("thyroid :", THYROID)

# ---- restore checkpoints from EVERY chained source, not just the first ----- #
# der_mil (trained by choudhary15/dermil-convnext, a solo run) and
# lesion_mil/mr_mil (trained by choudhary15/dermil-convnext-ablation) live in
# two Kaggle kernel lineages that were never chained to each other on Kaggle's
# side -- only merged locally on disk via separate fetches. Restoring from
# only one of them silently drops the other model's checkpoint, which is
# exactly what happened the first time this kernel ran: der_mil's checkpoint
# resolved to nothing, so it silently fell out of `ready` and the "proposed"
# role fell back to whichever model was first in the list.
#
# Each source has its OWN registry.json; a plain copytree of a second source
# on top of the first would overwrite (not merge) that file, silently
# forgetting every stage the first source had recorded. So the checkpoint/
# results FILES are copied from every source (safe: they live in disjoint
# per-model subdirectories), but every source's registry.json is loaded and
# merged by key, and the merged dict is the one written back.
CKPT = os.path.join(WORK, "checkpoints")
RESULTS = os.path.join(WORK, "results")
os.makedirs(CKPT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

import json  # noqa: E402

sources = []
for dp, dn, _fn in os.walk(INPUT):
    if "checkpoints" in dn and os.path.isfile(
            os.path.join(dp, "checkpoints", "registry.json")):
        sources.append(dp)
if not sources:
    raise SystemExit("no chained checkpoint source found -- push with --chain "
                     "from a session that has trained checkpoints attached")

banner("RESTORING CHECKPOINTS FROM %d SOURCE(S)" % len(sources))
merged_registry = {}
for prev in sources:
    print("  source: " + prev)
    for sub in ("checkpoints", "results"):
        src = os.path.join(prev, sub)
        dst = os.path.join(WORK, sub)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            n = sum(len(f) for _d, _s, f in os.walk(dst))
            print("    copied %-12s (running total %5d files under %s)" % (sub, n, dst))
    reg_path = os.path.join(prev, "checkpoints", "registry.json")
    try:
        this_reg = json.load(open(reg_path, encoding="utf-8"))
        overlap = set(this_reg) & set(merged_registry)
        merged_registry.update(this_reg)
        print("    registry: +%d keys (%d overlapping, overwritten)"
             % (len(this_reg), len(overlap)))
    except Exception as e:
        print("    could not read %s: %s" % (reg_path, e))

merged_path = os.path.join(CKPT, "registry.json")
json.dump(merged_registry, open(merged_path, "w", encoding="utf-8"), indent=2)
print("  merged registry written to %s (%d total keys)"
     % (merged_path, len(merged_registry)))
for want in ("refit/convnexttiny/lesion_mil", "refit/convnexttiny/mr_mil",
            "refit/convnexttiny/der_mil"):
    print("    %-32s %s" % (want, "present" if want in merged_registry else "MISSING"))

import torch                                                     # noqa: E402
from src import pipeline                                         # noqa: E402
from src.config import Config                                    # noqa: E402
from src.eval.counterfactual import reliability_influence_correlation  # noqa: E402
from src.pipeline import checkpoints_for                         # noqa: E402
from src.utils.checkpoint import StageRegistry                   # noqa: E402
from src.utils.common import log, save_json, set_seed            # noqa: E402

print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
     "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

cfg = Config()
cfg.data.thyroidxl_root = THYROID
cfg.data.num_workers = 4
cfg.run.run_name = RUN
cfg.run.ckpt_root = CKPT
cfg.run.results_root = RESULTS
cfg.model.backbone = "convnext_tiny"
cfg.model.evidence_mode = os.environ.get("DERMIL_EVIDENCE_MODE", "masked_input")
if os.environ.get("DERMIL_REGIONS"):
    cfg.model.regions = tuple(r.strip() for r in os.environ["DERMIL_REGIONS"].split(","))
print("config: backbone=%s evidence_mode=%s regions=%s"
     % (cfg.model.backbone, cfg.model.evidence_mode, cfg.model.regions))

set_seed(cfg.run.seed)
registry = StageRegistry(cfg.run.ckpt_root)
manifest = pipeline.prepare_thyroidxl(cfg, registry)
manifest = pipeline.cache_images(cfg, manifest, registry, "/tmp/dermil_cache")

ready = [m for m in MODELS if checkpoints_for(cfg, registry, m, "refit")]
print("models with a usable refit checkpoint:", ready)
if not ready:
    raise SystemExit("no checkpoints resolved for any of %s under run '%s'" % (MODELS, RUN))

# =========================================================================== #
banner("ROBUSTNESS SUITE (mask quality, permutation, frame removal, shortcut) "
      "+ mechanism validation for the proposed model (%s)" % PROPOSED)
try:
    pipeline.run_robustness(cfg, manifest, registry, ready,
                            proposed=PROPOSED if PROPOSED in ready else ready[0],
                            final_mode="refit")
except Exception:
    import traceback
    traceback.print_exc()

# =========================================================================== #
# pipeline.run_robustness only runs the reliability<->influence test for the
# proposed model. Run it for every OTHER model too that exposes a reliability
# head (mr_mil included, since it is the same DERMIL class with reliability
# disabled), so the region-profile comparison has more than one model.
banner("RELIABILITY <-> INFLUENCE, remaining models")
for m in ready:
    if m == PROPOSED:
        continue
    ck = checkpoints_for(cfg, registry, m, "refit")
    if not ck:
        continue
    key = "cf/influence/%s/%s" % (RUN, m)
    stale = os.path.join(RESULTS, RUN, m, "reliability_influence.json")
    registry.reset(key)
    if os.path.exists(stale):
        os.remove(stale)
    try:
        reliability_influence_correlation(cfg, manifest, m, ck[0], registry,
                                          max_patients=INFLUENCE_PATIENTS)
    except Exception:
        import traceback
        traceback.print_exc()

save_json({"models_covered": ready, "proposed": PROPOSED, "run": RUN},
         os.path.join(RESULTS, RUN, "convnext_mechanism_summary.json"))
banner("DONE in %.2f h" % ((time.time() - T0) / 3600))
print("registry:\n" + registry.summary())
