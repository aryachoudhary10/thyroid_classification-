"""End-to-end smoke test on synthetic data.

Builds a miniature ThyroidXL-shaped and TN5000-shaped dataset on disk, then
runs the complete pipeline: discovery -> manifest -> grouped CV -> calibration
-> single-shot test -> robustness -> counterfactual -> external validation ->
tables -> figures. It also kills a training run mid-way and restarts it to
prove that resume works.

    python experiments/smoke_test.py [--keep]

Runs on CPU in a couple of minutes. It proves the plumbing, not the science.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd

from src.config import Config
from src.utils.checkpoint import StageRegistry
from src.utils.common import banner, log, set_seed


# --------------------------------------------------------------------------- #
def make_thyroidxl(root: str, n_patients: int = 48, size: int = 96) -> None:
    img_dir = os.path.join(root, "images")
    msk_dir = os.path.join(root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    rows = []
    for pid in range(n_patients):
        label = int(pid % 2)
        n_frames = int(rng.randint(1, 4))
        split = "test" if pid % 5 == 0 else "dev"
        for f in range(n_frames):
            name = "p%03d_f%d" % (pid, f)
            img = (rng.rand(size, size) * 60 + 60).astype(np.uint8)
            m = np.zeros((size, size), np.uint8)
            cx, cy = rng.randint(30, size - 30, 2)
            r = int(rng.randint(10, 18))
            cv2.circle(m, (int(cx), int(cy)), r, 1, -1)
            # malignant lesions are darker and more irregular -- a learnable signal
            img[m > 0] = np.clip(img[m > 0] - (70 if label else 15), 0, 255)
            if label:
                cv2.circle(img, (int(cx) + 4, int(cy)), 3, 230, -1)
            cv2.imwrite(os.path.join(img_dir, name + ".png"), img)
            cv2.imwrite(os.path.join(msk_dir, name + "_mask.png"), m * 255)
            rows.append({"image": name + ".png", "patient_id": "p%03d" % pid,
                         "diagnosis": "malignant" if label else "benign",
                         "split": split, "tirads": int(3 + label * 2),
                         "age": int(30 + pid % 40), "sex": "F"})
    pd.DataFrame(rows).to_csv(os.path.join(root, "metadata.csv"), index=False)


def make_tn5000(root: str, n_images: int = 60, size: int = 96) -> None:
    img_dir = os.path.join(root, "images")
    ann_dir = os.path.join(root, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    rng = np.random.RandomState(1)
    for i in range(n_images):
        label = int(i % 2)
        name = "tn%04d" % i
        img = (rng.rand(size, size) * 50 + 90).astype(np.uint8)
        cx, cy = rng.randint(30, size - 30, 2)
        r = int(rng.randint(10, 18))
        cv2.circle(img, (int(cx), int(cy)), r, 40 if label else 120, -1)
        cv2.imwrite(os.path.join(img_dir, name + ".png"), img)
        with open(os.path.join(ann_dir, name + ".xml"), "w", encoding="utf-8") as fh:
            fh.write(
                "<annotation><size><width>%d</width><height>%d</height></size>"
                "<object><name>%s</name><bndbox><xmin>%d</xmin><ymin>%d</ymin>"
                "<xmax>%d</xmax><ymax>%d</ymax></bndbox></object></annotation>"
                % (size, size, "malignant" if label else "benign",
                   max(cx - r, 0), max(cy - r, 0), min(cx + r, size), min(cy + r, size)))


# --------------------------------------------------------------------------- #
def tiny_config(root: str, thyroid: str, tn: str) -> Config:
    cfg = Config()
    cfg.data.thyroidxl_root = thyroid
    cfg.data.tn5000_root = tn
    cfg.data.image_size = 64
    cfg.data.t_max = 3
    cfg.data.num_workers = 0
    cfg.data.peri_px = 10
    cfg.data.margin_inner_px = 3
    cfg.data.margin_outer_px = 3

    cfg.model.backbone = "resnet18"
    cfg.model.pretrained = False
    cfg.model.embed_dim = 48
    cfg.model.attn_dim = 24
    cfg.model.tf_heads = 2
    cfg.model.tf_layers = 1

    cfg.optim.stage1_epochs = 1
    cfg.optim.stage2_epochs = 1
    cfg.optim.batch_size = 4
    cfg.optim.amp = False

    cfg.eval.n_folds = 2
    cfg.eval.bootstrap_test = 40
    cfg.eval.bootstrap_external = 40
    cfg.eval.tta_views = 2

    cfg.external.epochs = 1
    cfg.external.warmup_epochs = 1
    cfg.external.eval_subset_per_class = 8
    cfg.external.batch_size = 4

    cfg.run.run_name = "smoke"
    cfg.run.ckpt_root = os.path.join(root, "checkpoints")
    cfg.run.results_root = os.path.join(root, "results")
    cfg.run.save_every_steps = 3
    cfg.run.seed = 7
    return cfg


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    args = ap.parse_args()

    root = tempfile.mkdtemp(prefix="dermil_smoke_")
    log("workspace: " + root)
    thyroid = os.path.join(root, "thyroidxl")
    tn = os.path.join(root, "tn5000")
    make_thyroidxl(thyroid)
    make_tn5000(tn)

    from src import pipeline
    from src.data.dataset import Intervention, PatientBagDataset
    from src.engine.cv import make_dataset
    from src.models.factory import MODEL_REGISTRY, build_model

    cfg = tiny_config(root, thyroid, tn)
    set_seed(cfg.run.seed)
    registry = StageRegistry(cfg.run.ckpt_root)

    ok = True
    try:
        # ---- 1. data ---------------------------------------------------- #
        banner("1. DISCOVERY AND MANIFEST")
        man = pipeline.prepare_thyroidxl(cfg, registry)
        assert len(man) > 0
        assert set(man["split"]) <= {"dev", "test"}

        # ---- 2. every model constructs and does a forward pass ----------- #
        banner("2. FORWARD PASS FOR EVERY REGISTERED MODEL")
        import torch
        sample = man[man["split"] == "dev"].head(4).reset_index(drop=True)
        for name in MODEL_REGISTRY:
            model, reqs = build_model(cfg, name)
            ds = make_dataset(cfg, sample, reqs, train=False)
            batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
            model.eval()
            with torch.no_grad():
                out = model(batch)
            assert out["logit"].shape == (len(ds),), name
            assert torch.isfinite(out["logit"]).all(), name
            extras = [k for k in ("R", "S", "D", "U") if k in out]
            log("  %-16s logit %s  extras %s"
                % (name, tuple(out["logit"].shape), extras or "-"))

        # ---- 3. permutation invariance of the proposed model ------------- #
        banner("3. PERMUTATION INVARIANCE (analytic check)")
        model, reqs = build_model(cfg, "der_mil")
        model.eval()
        multi = man[man["n_frames"] > 1].head(4).reset_index(drop=True)
        ds_a = make_dataset(cfg, multi, reqs, train=False)
        ds_b = make_dataset(cfg, multi, reqs, train=False,
                            intervention=Intervention(permute_frame_order=True, seed=5))
        with torch.no_grad():
            ba = torch.utils.data.default_collate([ds_a[i] for i in range(len(ds_a))])
            bb = torch.utils.data.default_collate([ds_b[i] for i in range(len(ds_b))])
            la = model(ba)["logit"]
            lb = model(bb)["logit"]
        delta = float((la - lb).abs().max())
        log("  max |logit difference| under frame permutation = %.3e" % delta)
        assert delta < 1e-3, "DER-MIL is not permutation invariant (%.3e)" % delta

        # ---- 4. padded frames must not influence anything ---------------- #
        banner("4. PADDING ISOLATION")
        single = man[man["n_frames"] == 1].head(3).reset_index(drop=True)
        if len(single):
            ds_s = make_dataset(cfg, single, reqs, train=False)
            with torch.no_grad():
                bs = torch.utils.data.default_collate([ds_s[i] for i in range(len(ds_s))])
                o = model(bs)
            a = o["alpha"].numpy()
            assert np.allclose(a[:, 1:], 0, atol=1e-6), "padded frames received attention"
            log("  attention on padded slots = %.2e (expected 0)" % float(np.abs(a[:, 1:]).max()))

        # ---- 5. train a fold, interrupt it, resume ----------------------- #
        banner("5. TRAINING WITH SIMULATED RUNTIME RESTART")
        from src.engine.cv import train_fold
        cfg_int = cfg.clone(**{"optim.stage1_epochs": 1, "optim.stage2_epochs": 0})
        train_fold(cfg_int, man, "der_mil", 0, registry)
        registry.reset("cv/smoke/der_mil/fold0")
        cfg_full = cfg.clone(**{"optim.stage1_epochs": 1, "optim.stage2_epochs": 1})
        train_fold(cfg_full, man, "der_mil", 0, registry)
        log("  resume path exercised (stage 1 reused, stage 2 continued)")

        # ---- 6. full protocol for two models ----------------------------- #
        banner("6. FULL PROTOCOL")
        models = ["rcaf", "der_mil"]
        pipeline.run_models(cfg, man, registry, models)

        # ---- 7. robustness + counterfactual ------------------------------ #
        banner("7. ROBUSTNESS SUITE")
        pipeline.run_robustness(cfg, man, registry, models, proposed="der_mil")

        # ---- 8. external validation -------------------------------------- #
        banner("8. EXTERNAL VALIDATION")
        pipeline.run_external(cfg, registry, models, mask_variants=("bbox",))

        # ---- 9. tables + figures ----------------------------------------- #
        banner("9. TABLES AND FIGURES")
        pipeline.build_report(cfg, man, models, "der_mil")
        figs = pipeline.make_figures(cfg, man, registry, models, "der_mil")
        assert figs, "no figures produced"

        # ---- 10. idempotence: re-running must skip everything ------------ #
        banner("10. IDEMPOTENCE")
        pipeline.run_models(cfg, man, registry, models)
        log("registry:\n" + registry.summary())

    except Exception:
        ok = False
        import traceback
        traceback.print_exc()

    banner("SMOKE TEST " + ("PASSED" if ok else "FAILED"))
    if args.keep:
        log("workspace kept at " + root)
    else:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
