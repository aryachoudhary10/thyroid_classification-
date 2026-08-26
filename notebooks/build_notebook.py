"""Generates DER_MIL_Colab.ipynb. Run: python notebooks/build_notebook.py"""
from __future__ import annotations

import json
import os

MD = "markdown"
CODE = "code"


def cell(kind, src):
    src = src.strip("\n").split("\n")
    src = [ln + "\n" for ln in src[:-1]] + [src[-1]]
    base = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == CODE:
        base["outputs"] = []
        base["execution_count"] = None
    return base


CELLS = [
(MD, """
# DER-MIL: Diagnostic Evidence Reliability Learning for patient-level thyroid ultrasound

This notebook runs the whole study end to end:

| stage | what it does |
|---|---|
| 1 | download ThyroidXL + TN5000 from Kaggle and **inspect their layout** |
| 2 | build the patient-level manifest, grouped stratified 5-fold split, leakage assertions |
| 3 | train the fair baselines, the preserved **RCAF** baseline, and the proposed **DER-MIL** |
| 4 | development-only calibration + threshold selection, then a single-shot test evaluation |
| 5 | ablation ladder (importance -> +support -> +contradiction -> +uncertainty) |
| 6 | robustness: mask degradation, shortcut permutation, frame corruption, frame removal, permutation invariance |
| 7 | counterfactual validation: does reliability predict actual influence? |
| 8 | external domain-adapted validation on TN5000 for **both RCAF and DER-MIL** |
| 9 | paper-ready tables and figures |

## Resuming after a Colab disconnect

**Every stage is checkpointed.** Put `CKPT_ROOT` on Google Drive (cell 2) and, after a
runtime restart, just re-run the cells top to bottom. Finished stages print `SKIP`; the
interrupted training run resumes from its last saved batch. You never repeat GPU work.
"""),

(CODE, """
#@title 1. Install dependencies
!pip -q install kagglehub opencv-python-headless
import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
"""),

(MD, """
## 2. Project code + a checkpoint directory that survives restarts

Upload the project folder (the one containing `src/`) to your Drive, then point
`PROJECT_DIR` at it. `CKPT_ROOT` **must** live on Drive - that is what makes a
runtime restart free.
"""),

(CODE, """
#@title 2. Mount Drive, locate the code, set checkpoint roots
import os, sys

USE_DRIVE = True  #@param {type:"boolean"}
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive'
else:
    DRIVE = '/content'

PROJECT_DIR = os.path.join(DRIVE, 'dermil')     #@param {type:"string"}
CKPT_ROOT   = os.path.join(DRIVE, 'dermil_checkpoints')
RESULTS_ROOT= os.path.join(DRIVE, 'dermil_results')

assert os.path.isdir(os.path.join(PROJECT_DIR, 'src')), (
    "Could not find src/ under %s -- upload the project folder there first." % PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
os.makedirs(CKPT_ROOT, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)
print("code    :", PROJECT_DIR)
print("ckpts   :", CKPT_ROOT)
print("results :", RESULTS_ROOT)
"""),

(CODE, """
#@title 3. Download both datasets from Kaggle
from src.config import Config
from src import pipeline

cfg = Config()
cfg.run.ckpt_root    = CKPT_ROOT
cfg.run.results_root = RESULTS_ROOT
cfg.run.run_name     = 'main'

cfg = pipeline.download_datasets(cfg)
print(cfg.data.thyroidxl_root)
print(cfg.data.tn5000_root)
"""),

(MD, """
## 4. Inspect the dataset layout - **read this output before going further**

Nothing about the Kaggle folder structure is hard-coded. Discovery walks the download,
guesses which directories hold images vs masks, and picks a metadata table. Check that
the counts and column names below look right.

If a guess is wrong, do **not** edit any source file - pass a `LayoutOverride` to
`prepare_thyroidxl` in the next cell:

```python
from src.data.discovery import LayoutOverride
ov = LayoutOverride(image_dirs=['images'], mask_dirs=['masks'],
                    table_path='/root/.cache/.../metadata.csv',
                    patient_col='patient_id', image_col='image_name',
                    label_col='pathology', split_col='split')
man = pipeline.prepare_thyroidxl(cfg, registry, override=ov, force=True)
```
"""),

(CODE, """
#@title 4. Inspect
pipeline.inspect_datasets(cfg)
"""),

(CODE, """
#@title 5. Build the patient manifest, folds, and leakage checks
from src.utils.checkpoint import StageRegistry
registry = StageRegistry(cfg.run.ckpt_root)

OVERRIDE = None   # <- put a LayoutOverride here if discovery guessed wrong
manifest = pipeline.prepare_thyroidxl(cfg, registry, override=OVERRIDE)
manifest.head()
"""),

(MD, """
## 6. Smoke run first (strongly recommended)

Ten minutes now beats discovering a shape bug three hours into training.
This trains DER-MIL on a small subset with a tiny schedule, in a **separate**
run namespace so it cannot contaminate the real results.
"""),

(CODE, """
#@title 6. Optional smoke run
RUN_SMOKE = True  #@param {type:"boolean"}
if RUN_SMOKE:
    smoke = cfg.clone(**{'run.run_name': 'smoke', 'run.debug_subset': 120,
                         'optim.stage1_epochs': 1, 'optim.stage2_epochs': 1,
                         'eval.n_folds': 2, 'eval.bootstrap_test': 50})
    smoke_reg = StageRegistry(smoke.run.ckpt_root)
    smoke_man = pipeline.prepare_thyroidxl(smoke, smoke_reg)
    pipeline.run_models(smoke, smoke_man, smoke_reg, ['der_mil'])
    print('smoke run OK')
"""),

(MD, """
## 7. The real training run

`MODELS` is the fair comparison ladder. Order matters only for your patience -
each model is independently checkpointed, so you can stop after any of them and
resume later.

* `mean_pool`, `max_pool` - naive aggregation references
* `image_mil`, `lesion_mil`, `transformer_bag` - fair patient-level baselines
* `rcaf` - **the preserved published baseline**
* `mr_mil` - multi-region evidence, *no reliability*: the ablation that isolates the contribution
* `der_mil` - the proposed model

`FINAL_MODE` decides what "the final model" means for the single-shot test:

* `refit` - retrain once on the full development cohort (matches the paper's wording, one extra run per model)
* `ensemble` - average the five fold checkpoints (free, but the temperature fitted on
  single-model out-of-fold logits transfers only approximately to averaged logits, so
  discrimination is unaffected while calibration numbers are slightly pessimistic)
"""),

(CODE, """
#@title 7. Train everything
MODELS = ['mean_pool', 'max_pool', 'image_mil', 'lesion_mil',
          'transformer_bag', 'rcaf', 'mr_mil', 'der_mil']   #@param
FINAL_MODE = 'refit'  #@param ['refit', 'ensemble']

results = pipeline.run_models(cfg, manifest, registry, MODELS, final_mode=FINAL_MODE)
print(registry.summary())
"""),

(CODE, """
#@title 8. Ablation ladder for the reliability components
ABLATIONS = ['rcaf_nogate', 'der_i', 'der_is', 'der_isd', 'der_iu']
pipeline.run_models(cfg, manifest, registry, ABLATIONS, final_mode=FINAL_MODE)

# diagnostic shortcut baseline -- NOT part of the fair ladder
pipeline.run_models(cfg, manifest, registry, ['mask_channel'], final_mode=FINAL_MODE)
"""),

(CODE, """
#@title 9. Robustness, shortcut and counterfactual suite
ROBUST_MODELS = ['rcaf', 'mr_mil', 'der_mil', 'mask_channel']
rob = pipeline.run_robustness(cfg, manifest, registry, ROBUST_MODELS,
                              proposed='der_mil', final_mode=FINAL_MODE)
for k, v in rob.items():
    print('\\n===', k, '===')
    print(v if not isinstance(v, dict) else list(v))
"""),

(MD, """
## 10. External validation on TN5000

Both the preserved RCAF baseline **and** DER-MIL are domain-adapted from their
ThyroidXL checkpoints and evaluated on the same class-balanced held-out subset, with
8-view TTA. The `bbox` arm repeats everything using only bounding-box masks derived
from the VOC XML annotations.
"""),

(CODE, """
#@title 10. TN5000
external = pipeline.run_external(cfg, registry,
                                 model_names=['rcaf', 'der_mil'],
                                 mask_variants=('pixel', 'bbox'))
external
"""),

(CODE, """
#@title 11. Tables
tables = pipeline.build_report(cfg, manifest, MODELS + ABLATIONS, proposed='der_mil')
tables
"""),

(CODE, """
#@title 12. Figures
figs = pipeline.make_figures(cfg, manifest, registry, MODELS, proposed='der_mil')
from IPython.display import Image, display
for f in figs:
    display(Image(f))
"""),

(MD, """
## 13. Managing the checkpoint registry

```python
print(registry.summary())          # what is finished

registry.reset('cv/main/der_mil/fold2')   # redo one fold
registry.reset_prefix('robust/')          # redo every robustness experiment
registry.reset_prefix('test/')            # release the test set (think first!)
```

Resetting a registry key makes the pipeline rerun that stage. It does **not** delete the
model checkpoint, so a reset training stage resumes from `last.pt` rather than starting
over; delete `CKPT_ROOT/main/<model>/fold<k>/` if you want a genuinely fresh run.

**The test set.** `test/...` keys exist to stop you from repeatedly evaluating on the
held-out cohort and quietly turning it into a validation set. Only reset them if the
model genuinely changed.
"""),

(CODE, """
#@title 13. Registry status
print(registry.summary())
"""),
]


def main():
    nb = {
        "cells": [cell(k, s) for k, s in CELLS],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "DER_MIL_Colab.ipynb")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", out, "(%d cells)" % len(nb["cells"]))


if __name__ == "__main__":
    main()
