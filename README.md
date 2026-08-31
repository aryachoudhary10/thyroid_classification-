# DER-MIL — Diagnostic Evidence Reliability Learning for patient-level thyroid ultrasound

Patient-level thyroid ultrasound malignancy prediction. The task and the leakage-free
protocol follow Sherif, Elsayed & Deif, *Sci Rep* 2026 (`s41598-026-61342-8`); the model
replaces "which evidence is important?" with a stronger, falsifiable question:

> **Can the diagnostic evidence in one ultrasound view be trusted, given that other views
> support it, contradict it, or are themselves uncertain?**

```
Patient bag of frames + lesion masks
        │
        ├── multi-region evidence tokens    e[t,k]   k ∈ {core, margin, peri, global}
        │
        ├── Importance      I[t,k]     how predictive this evidence claims to be
        ├── Support         S[t,k]     how strongly other views corroborate it
        ├── Contradiction   D[t,k]     how strongly other reliable views disagree
        ├── Uncertainty     U[t,k]     confidence in the observation itself
        │
        └── Reliability     R = f(I, S, D, U) ∈ (0,1)
                    │
                    ├── level 1: reliability-weighted evidence pooling  → h[t]
                    ├── level 2: reliability-modulated frame attention  → α[t]
                    └── patient embedding → P(malignant)
```

The contribution is the **reliability mechanism**, not the use of a graph. The signed
message passing is a tool that earns its place only if the ablation says so — and the
ablation is built in.

---

## Quick start (Colab)

1. Upload this folder to Drive (e.g. `MyDrive/dermil`).
2. Open [notebooks/DER_MIL_Colab.ipynb](notebooks/DER_MIL_Colab.ipynb).
3. Run the cells top to bottom.

Datasets are pulled with `kagglehub`:

```python
kagglehub.dataset_download("safaaqaisi/thyroidxl")        # ThyroidXL  (Setup A)
kagglehub.dataset_download("abdullahelafifi/main-data")   # TN5000     (Setup B)
```

### Local / CLI

```bash
pip install -r requirements.txt
python experiments/smoke_test.py          # synthetic end-to-end check (~2 min, CPU)
python experiments/run_all.py --stage all --download
python experiments/run_all.py --status    # what is finished
```

---

## Checkpointing and resume — read this

Two independent mechanisms, both rooted at `cfg.run.ckpt_root`. **Put that on Drive.**

**`StageRegistry`** (`registry.json`) — a ledger of finished pipeline stages:
`data/thyroidxl`, `cv/main/der_mil/fold3`, `test/main/der_mil`, `tn5000/adapt/main/der_mil/pixel`, …
A completed stage prints `SKIP` and is never recomputed.

**`CheckpointManager`** (`last.pt` / `best.pt` per run) — model, optimiser, AMP scaler,
stage, epoch, step-in-epoch, best metric, patience counter, RNG state. Written atomically
(`.tmp` + rename), every epoch **and** every `save_every_steps` batches.

**After a Colab disconnect, re-run the same cells.** Finished folds skip; the interrupted
one replays its shuffled epoch, skips the batches already applied, and continues.

```python
print(registry.summary())                # what is done
registry.reset('cv/main/der_mil/fold2')  # redo one fold (resumes from last.pt)
registry.reset_prefix('robust/')         # redo the robustness suite
```

Resetting a key does **not** delete weights. For a genuinely fresh run, also delete
`CKPT_ROOT/main/<model>/fold<k>/`.

---

## Dataset discovery

No Kaggle path is hard-coded. [src/data/discovery.py](src/data/discovery.py) walks the
download, scores directories by name, pairs masks to images by filename stem, and picks a
metadata table by column heuristics. It falls back through pixel masks → VOC XML boxes →
folder-name labels → filename-derived patient ids.

**Always run `pipeline.inspect_datasets(cfg)` first and read the output.** If a guess is
wrong, pass a `LayoutOverride` — never edit source:

```python
from src.data.discovery import LayoutOverride
ov = LayoutOverride(image_dirs=['images'], mask_dirs=['masks'],
                    table_path='.../metadata.csv',
                    patient_col='patient_id', image_col='image_name',
                    label_col='pathology', split_col='split')
manifest = pipeline.prepare_thyroidxl(cfg, registry, override=ov, force=True)
```

If the mirror ships no official split, one frozen patient-level test cohort is created
**once** and cached in the manifest, so it never drifts between runs.

---

## Leakage-free protocol

1. Grouped stratified 5-fold CV over **patients** in the development cohort.
2. Held-out fold predictions concatenated → out-of-fold (OOF) set.
3. Temperature scaling **and** both operating thresholds (Youden, Sensitivity ≥ 0.90)
   fitted on OOF only.
4. Final predictor built (see below).
5. **One** evaluation on the untouched test cohort with the frozen calibration.

Asserted, not assumed: `assert_no_patient_leakage`, `validate_folds`, and a per-fold
`assert not set(train_pid) & set(val_pid)`. The `test/...` registry keys exist to stop you
from silently turning the held-out cohort into a validation set by re-running a cell.

### `final_mode`

| mode | cost | note |
|---|---|---|
| `refit` (default) | +1 training run per model | retrain on the full dev cohort — matches the paper's wording |
| `ensemble` | free | average the five fold checkpoints. Discrimination is fine; **calibration is approximate**, because the temperature was fitted on single-model OOF logits and averaged logits have a different scale |
| `best_fold` | free | single highest-validation fold; used as the source for TN5000 adaptation |

---

## Models

| name | what it is |
|---|---|
| `mean_pool`, `max_pool` | naive frame-logit pooling |
| `image_mil` | image-only encoder + AttnMIL |
| `lesion_mil` | lesion-crop + AttnMIL |
| `transformer_bag` | sequence bag model (no positional encoding → permutation invariant) |
| `mask_channel` | naive image–mask concatenation — **diagnostic only**, probes shortcut behaviour |
| `mr_mil` | multi-region evidence, **reliability disabled** — the ablation that isolates the claim |
| `der_i` / `der_is` / `der_isd` / `der_iu` | the reliability ladder |
| **`der_mil`** | **the proposed model** |

### Evidence encoding: an efficiency decision worth knowing

`evidence_mode="roi_pool"` (default) runs the backbone **once per frame** and derives the
four region embeddings by coverage-weighted pooling of the layer3 (14×14) and layer4 (7×7)
feature maps. That is *cheaper* than a two-branch lesion/context design while producing four
evidence streams instead of two — which is what makes full DER-MIL trainable on one Colab GPU.

`evidence_mode="masked_input"` instead runs K masked forward passes per frame, i.e. the
literal `x ⊙ m` formulation generalised to K regions. Faithful, K× the compute, available
for a like-for-like ablation. Coverage pooling is coarser for thin margin rings; widen
`margin_inner_px` / `margin_outer_px` if that matters for your resolution.

---

## Experiments

| experiment | question | module |
|---|---|---|
| Fair comparison + head-to-head | does DER-MIL beat the patient-level baselines? | `eval/reporting.py` |
| Ablation ladder | does each of I/S/D/U contribute? | `models/factory.py` |
| Calibration + threshold transfer | do the probabilities mean anything? | `eval/calibration.py` |
| Mask degradation (dilate/erode/zeros) | robustness to imperfect segmentation | `eval/robustness.py` |
| **Within-patient mask permutation** | is the model shortcut-prone? | `eval/robustness.py` |
| **Frame corruption** | does uncertainty absorb a bad view? | `eval/robustness.py` |
| **Reliability response** | does the *corrupted* frame actually lose reliability? | `eval/robustness.py` |
| **Frame removal** | does removing the most reliable view move the prediction most? | `eval/robustness.py` |
| Permutation invariance | is the prediction independent of frame order? | `eval/robustness.py` |
| **Evidence ablation** | which region actually drives the decision? | `eval/counterfactual.py` |
| **Reliability ↔ influence correlation** | *the headline test:* does R predict counterfactual influence? | `eval/counterfactual.py` |
| Agreement / contradiction cases | is the model right more often when views agree? | `eval/counterfactual.py` |
| TIRADS comparison | versus the clinical standard | `eval/reporting.py` |
| **TN5000 external validation** | cross-domain, multi-arm domain adaptation | `external/tn5000.py` |

The reliability↔influence correlation is the one that can falsify the paper's central
claim, so it is computed **within each patient** and then averaged — comparing reliability
across patients would confound it with case difficulty.

---

## Two design decisions that keep the claim honest

**Consistency is weighted by a detached `A+`.** If the support weights carried gradient,
the cheapest way to minimise the consistency loss would be to drive all support to zero —
the model would learn that no view ever corroborates another. Detaching makes it a
statement about representations, not about the relation head. Consistency is also applied
*only* where the model already judged two views supportive: different views legitimately
carry complementary information, and forcing all cross-view embeddings together would
destroy the very signal the contradiction channel exists to detect.

**Support and contradiction are separate channels, not two ends of one axis.** Two views
can be dissimilar because they are complementary rather than because they disagree. A
signed similarity cannot express that; a positive channel and a negative channel can.
Peer influence is additionally weighted by peer confidence `(1 − U)`, so a blurred frame
cannot manufacture support or contradiction for a clean one. Patients with a single frame
have no cross-view evidence: `S = D = 0` and a learned no-peer offset enters the
reliability function with an explicit `has_peers` indicator, so they are not penalised.

---

## What is verified vs what is not

`python experiments/smoke_test.py` builds miniature ThyroidXL- and TN5000-shaped datasets
and runs the entire pipeline on CPU. It passes, and it asserts:

- every registered model constructs and produces finite logits;
- DER-MIL is **permutation invariant** (max |Δlogit| under frame shuffling < 1e-3);
- **padded frames receive exactly zero attention**;
- a training run interrupted mid-way **resumes** correctly;
- re-running any finished stage is a no-op.

It proves the plumbing, not the science. The accuracy, reliability-correlation and
robustness numbers only mean something once the models are trained on the real datasets.
Nothing in this repository has been run against actual ThyroidXL or TN5000 data yet — the
first real run is yours, and the discovery output in step 4 is where to look if anything
is mis-mapped.

---

## Layout

```
src/
  config.py              every experiment knob, serialised next to each checkpoint
  pipeline.py            high-level resumable API (what the notebook calls)
  data/     discovery, manifest, regions, transforms, dataset, splits
  models/   backbone, attn_mil, baselines,
            evidence_encoder, reliability, der_mil, factory
  losses/   BCE / focal, support-weighted consistency, reliability regulariser,
            counterfactual ranking
  engine/   trainer (two-stage, mid-epoch resume), cv (protocol)
  eval/     metrics (+DeLong, bootstrap), calibration, robustness,
            counterfactual, reporting
  external/ tn5000 (domain adaptation + TTA)
  viz/      figures (CVD-validated palette)
experiments/  run_all.py, smoke_test.py
notebooks/    DER_MIL_Colab.ipynb
```
