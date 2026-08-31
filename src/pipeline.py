"""High-level, resumable pipeline. This is the API the Colab notebook calls.

Every step is guarded by the stage registry, so the correct way to recover from
a Colab disconnect is simply to re-run the same cell: finished stages print
SKIP, the interrupted one resumes from its last checkpoint.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data.discovery import LayoutOverride, build_records, describe_root
from .data.manifest import (assert_no_patient_leakage, build_patient_manifest,
                            cohort_stats, ensure_official_split, load_manifest,
                            mask_coverage, save_manifest)
from .data.cache import build_resize_cache, verify_cache
from .data.splits import make_folds, validate_folds
from .data.thyroidxl import (assert_table1, build_thyroidxl_manifest,
                             label_provenance, looks_like_thyroidxl)
from .engine.cv import (build_final_predictor, fit_calibration, full_protocol,
                        calibration_table)
from .eval import reporting
from .eval.metrics import all_metrics
from .eval.counterfactual import (agreement_contradiction_cases, evidence_ablation,
                                  reliability_influence_correlation)
from .eval.robustness import (corruption_reliability_response, frame_corruption_study,
                              frame_removal_study, mask_quality_sweep,
                              permutation_invariance, shortcut_sensitivity)
from .utils.checkpoint import StageRegistry
from .utils.common import banner, in_colab, log, save_json, set_seed


# --------------------------------------------------------------------------- #
# 0. datasets
# --------------------------------------------------------------------------- #
def download_datasets(cfg: Config, thyroidxl: bool = True, tn5000: bool = True
                      ) -> Config:
    import kagglehub
    if thyroidxl and not cfg.data.thyroidxl_root:
        p = kagglehub.dataset_download("safaaqaisi/thyroidxl")
        cfg.data.thyroidxl_root = p
        log("ThyroidXL -> " + p)
    if tn5000 and not cfg.data.tn5000_root:
        p = kagglehub.dataset_download("abdullahelafifi/main-data")
        cfg.data.tn5000_root = p
        log("TN5000 -> " + p)
    return cfg


def inspect_datasets(cfg: Config) -> None:
    """ALWAYS run this once and read the output before training anything."""
    for name, root in (("ThyroidXL", cfg.data.thyroidxl_root),
                       ("TN5000", cfg.data.tn5000_root)):
        if not root:
            continue
        banner("DATASET LAYOUT: " + name)
        print(describe_root(root))


# --------------------------------------------------------------------------- #
# 1. manifest + folds
# --------------------------------------------------------------------------- #
def prepare_thyroidxl(cfg: Config, registry: StageRegistry,
                      override: Optional[LayoutOverride] = None,
                      force: bool = False) -> pd.DataFrame:
    key = "data/thyroidxl/%s" % cfg.run.run_name
    path = os.path.join(cfg.run.results_root, cfg.run.run_name, "manifest.csv")

    if registry.is_done(key) and os.path.exists(path) and not force:
        log("SKIP  " + key)
        man = load_manifest(path)
    else:
        banner("BUILDING THYROIDXL PATIENT MANIFEST")
        root = cfg.data.thyroidxl_root
        # Prefer the dataset's own official files; fall back to generic
        # discovery only for other mirrors or synthetic test fixtures.
        if override is None and looks_like_thyroidxl(root):
            man = build_thyroidxl_manifest(root, t_max=cfg.data.t_max, strict=True)
            prov = label_provenance(man)
            if not prov.empty:
                log("label provenance:" + chr(10) + prov.to_string(index=False))
        else:
            log("ThyroidXL adapter not applicable -- using generic discovery")
            rec = build_records(root, override, require_mask=True)
            man = build_patient_manifest(rec, t_max=cfg.data.t_max)
            man = ensure_official_split(man, seed=cfg.run.seed)
        assert_no_patient_leakage(man)
        man = make_folds(man, cfg.eval.n_folds, cfg.run.seed)
        save_manifest(man, path)
        registry.mark_done(key, {"csv": path, "n_patients": len(man)})

    validate_folds(man)
    log("mask coverage: " + str(mask_coverage(man)))
    stats = cohort_stats(man)
    print(stats.to_string())
    stats.to_csv(os.path.join(os.path.dirname(path), "cohort_stats.csv"))
    if cfg.run.debug_subset:
        log("DEBUG MODE: restricting to %d development patients" % cfg.run.debug_subset)
    return man


def cache_images(cfg: Config, manifest: pd.DataFrame, registry: StageRegistry,
                 cache_dir: str, force: bool = False) -> pd.DataFrame:
    """Materialise every frame at the training resolution, once.

    The calibration run measured ~60% of epoch wall-clock going into decoding
    532x727 PNGs and resizing them to 224 -- repeated for every epoch, fold and
    model. Doing it once is bit-identical and removes that cost from the whole
    study. Returns a manifest whose paths point at the cache.
    """
    key = "data/cache/%s" % cfg.run.run_name
    cached_csv = os.path.join(cfg.run.results_root, cfg.run.run_name,
                              "manifest_cached.csv")

    if registry.is_done(key) and os.path.exists(cached_csv) and not force:
        log("SKIP  " + key)
        man = load_manifest(cached_csv)
        # A chained session may have restored the manifest but not the pixels.
        first = man.iloc[0]["image_paths"][0]
        if os.path.exists(first):
            verify_cache(man, size=cfg.data.image_size)
            return man
        log("cache manifest present but files are missing -- rebuilding")

    banner("BUILDING RESIZE CACHE (%d px)" % cfg.data.image_size)
    man, stats = build_resize_cache(manifest, cache_dir, cfg.data.image_size,
                                    workers=max(cfg.data.num_workers * 2, 8))
    verify_cache(man, size=cfg.data.image_size)
    save_manifest(man, cached_csv)
    registry.mark_done(key, {"csv": cached_csv, **stats})
    return man


# --------------------------------------------------------------------------- #
# 2. training + protocol
# --------------------------------------------------------------------------- #
def run_model(cfg: Config, manifest: pd.DataFrame, registry: StageRegistry,
              model_name: str, final_mode: str = "ensemble") -> Dict[str, Any]:
    set_seed(cfg.run.seed)
    res = full_protocol(cfg, manifest, model_name, registry, final_mode)
    out_dir = os.path.join(cfg.run.results_root, cfg.run.run_name, model_name)
    ct = calibration_table(res["oof"],
                           reporting.load_test(cfg, model_name),
                           res["calibration"], cfg.eval.ece_bins)
    ct.to_csv(os.path.join(out_dir, "calibration_table.csv"), index=False)
    log("\n== calibration (%s) ==\n" % model_name + ct.to_string(index=False))
    return res


def run_models(cfg: Config, manifest: pd.DataFrame, registry: StageRegistry,
               model_names: Sequence[str], final_mode: str = "ensemble"
               ) -> Dict[str, Any]:
    # Descriptor models quantise measurements into phrases; the boundaries must
    # be fitted before any training so they are identical across every fold and
    # every dataset the run touches.
    if any(str(m).endswith("_vl") for m in model_names):
        from .models.descriptors import calibrate_bin_edges
        calibrate_bin_edges(cfg, manifest, registry)

    out = {}
    for name in model_names:
        banner("MODEL: " + name)
        out[name] = run_model(cfg, manifest, registry, name, final_mode)
        torch.cuda.empty_cache()
    return out


def checkpoints_for(cfg: Config, registry: StageRegistry, model_name: str,
                    mode: str = "ensemble") -> List[str]:
    """Checkpoints of the final predictor, read straight from the registry.

    The mode is passed through unchanged so the robustness suite interrogates
    the SAME predictor the test table reports. "refit" resolves to the
    already-trained refit checkpoint when one exists and falls back to the fold
    ensemble otherwise, so this never triggers training as a side effect.
    """
    return build_final_predictor(cfg, pd.DataFrame(), model_name, registry, mode)


# --------------------------------------------------------------------------- #
# 3. robustness + counterfactual suite
# --------------------------------------------------------------------------- #
def run_robustness(cfg: Config, manifest: pd.DataFrame, registry: StageRegistry,
                   model_names: Sequence[str], proposed: str = "der_mil",
                   final_mode: str = "ensemble") -> Dict[str, Any]:
    """Robustness must interrogate the SAME predictor the test table reports."""
    banner("ROBUSTNESS AND COUNTERFACTUAL SUITE")
    ckpts = {m: checkpoints_for(cfg, registry, m, final_mode) for m in model_names}
    ckpts = {m: c for m, c in ckpts.items() if c}
    results: Dict[str, Any] = {}

    # shortcut sensitivity across every model that consumes masks
    results["shortcut"] = shortcut_sensitivity(cfg, manifest, ckpts, registry)

    # per-model sweeps
    for m, c in ckpts.items():
        results.setdefault("mask_quality", {})[m] = mask_quality_sweep(
            cfg, manifest, m, c, registry)
        results.setdefault("permutation", {})[m] = permutation_invariance(
            cfg, manifest, m, c, registry)
        results.setdefault("frame_removal", {})[m] = frame_removal_study(
            cfg, manifest, m, c, registry)

    results["frame_corruption"] = frame_corruption_study(cfg, manifest, ckpts, registry)

    # reliability-specific validation, only meaningful for the proposed family
    if proposed in ckpts:
        c = ckpts[proposed]
        results["evidence_ablation"] = evidence_ablation(
            cfg, manifest, proposed, c, registry)
        results["reliability_response"] = corruption_reliability_response(
            cfg, manifest, proposed, c, registry)
        results["reliability_influence"] = reliability_influence_correlation(
            cfg, manifest, proposed, c[0], registry)
        results["agreement_cases"] = agreement_contradiction_cases(
            cfg, manifest, proposed, c[0], registry)
    return results


# --------------------------------------------------------------------------- #
# 4. external validation
# --------------------------------------------------------------------------- #
def run_external(cfg: Config, registry: StageRegistry,
                 model_names: Sequence[str] = ("mr_mil", "der_mil"),
                 override: Optional[LayoutOverride] = None,
                 mask_variants: Tuple[str, ...] = ("pixel", "bbox"),
                 manifest: Optional[pd.DataFrame] = None,
                 label_free: bool = False,
                 retrieval: bool = False) -> pd.DataFrame:
    """Run the TN5000 arms for each model.

    The ``unet`` arm needs a segmenter trained on ThyroidXL, so it also needs
    the ThyroidXL ``manifest``. Asking for ``unet`` without one is a hard error
    rather than a silent downgrade to boxes -- a mask arm that quietly evaluates
    something other than what it claims is worse than one that does not run.
    """
    from .external.tn5000 import (build_tn5000_manifest, compare_external,
                                  external_validation, force_bbox_masks)
    from .external.segmentation import unet_mask_manifest
    from .external.adaptation import run_label_free_arms
    from .external.retrieval import build_retrieval_bags, evaluate_retrieval_bags
    from .data.tn5000 import official_eval_subset
    from .models.descriptors import calibrate_bin_edges

    if any(str(m).endswith("_vl") for m in model_names):
        if manifest is None:
            raise ValueError("descriptor models need the ThyroidXL manifest to "
                             "calibrate their phrase boundaries; pass manifest=...")
        calibrate_bin_edges(cfg, manifest, registry)
    banner("EXTERNAL VALIDATION ON TN5000 (Setup B)")
    tn = build_tn5000_manifest(cfg, registry, override)

    unet_man = None
    if "unet" in mask_variants:
        if manifest is None:
            raise ValueError("the 'unet' mask arm needs the ThyroidXL manifest "
                             "to train a segmenter; pass manifest=...")
        unet_man = unet_mask_manifest(cfg, manifest, tn, registry)

    frames = []
    for name in model_names:
        src = checkpoints_for(cfg, registry, name, "best_fold")
        if not src:
            log("  no ThyroidXL checkpoint for %s -- skipping" % name)
            continue
        df = external_validation(cfg, name, src[0], tn, registry, mask_variants,
                                 unet_manifest=unet_man)
        frames.append(df)

        if label_free:
            # Same carve as the supervised arms, so every rung is scored on the
            # identical held-out subset and the comparison stays paired.
            base = force_bbox_masks(tn) if "bbox" in mask_variants else tn
            adapt_df, eval_df = official_eval_subset(
                base, cfg.external.eval_subset_per_class, cfg.run.seed)
            try:
                run_label_free_arms(cfg, name, src[0], adapt_df, eval_df, registry)
            except Exception:
                import traceback
                traceback.print_exc()

        if retrieval:
            base = force_bbox_masks(tn) if "bbox" in mask_variants else tn
            adapt_df, eval_df = official_eval_subset(
                base, cfg.external.eval_subset_per_class, cfg.run.seed)
            try:
                bags = build_retrieval_bags(cfg, name, src[0], adapt_df,
                                            eval_df, registry)
                r = evaluate_retrieval_bags(cfg, name, src[0], bags,
                                            cfg.eval.tta_views)
                met = all_metrics(r["y"], r["p"])
                log("  retrieval bags (k=%d): AUC %.4f | F1 %.4f"
                    % (cfg.adapt.retrieval_bag_size, met["roc_auc"], met["f1"]))
                pd.DataFrame({"patient_id": bags["patient_id"].astype(str).values,
                              "label": r["y"].astype(int), "p": r["p"]}).to_csv(
                    os.path.join(cfg.run.results_root, cfg.run.run_name, name,
                                 "tn5000_retrieval_predictions.csv"), index=False)
            except Exception:
                import traceback
                traceback.print_exc()

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(cfg.run.results_root, cfg.run.run_name, "_tables",
                        "tn5000_external.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    log("\n== TN5000 external validation ==\n" + out.to_string(index=False))

    for v in mask_variants:
        cmp_res = compare_external(cfg, list(model_names), v)
        if cmp_res:
            save_json(cmp_res, os.path.join(os.path.dirname(path),
                                            "tn5000_delong_%s.json" % v))
    return out


# --------------------------------------------------------------------------- #
# 5. reporting
# --------------------------------------------------------------------------- #
def build_report(cfg: Config, manifest: pd.DataFrame, models: Sequence[str],
                 proposed: str = "der_mil") -> Dict[str, str]:
    banner("TABLES")
    return reporting.write_all_tables(cfg, models, proposed, manifest)


def make_figures(cfg: Config, manifest: pd.DataFrame, registry: StageRegistry,
                 models: Sequence[str], proposed: str = "der_mil") -> List[str]:
    from .viz.explain import (plot_confusion, plot_evidence_distribution,
                              plot_model_comparison, plot_reliability_diagram,
                              plot_support_contradiction)
    from .eval.calibration import apply_temperature, sigmoid
    from .utils.common import load_json

    fig_dir = os.path.join(cfg.run.results_root, cfg.run.run_name, "_figures")
    paths: List[str] = []

    tt = reporting.test_table(cfg, models, with_ci=False)
    if not tt.empty:
        paths.append(plot_model_comparison(
            tt[["model", "roc_auc"]], os.path.join(fig_dir, "model_comparison.png")))

    oof = reporting.load_oof(cfg, proposed)
    cal = load_json(os.path.join(cfg.run.results_root, cfg.run.run_name, proposed,
                                 "calibration.json"), {})
    if oof is not None and cal:
        y = oof["label"].values.astype(int)
        z = oof["logit"].values
        paths.append(plot_reliability_diagram(
            y, sigmoid(z), apply_temperature(z, float(cal["temperature"])),
            os.path.join(fig_dir, "reliability_diagram.png"),
            title="Reliability of %s (out-of-fold development)" % proposed))

    te = reporting.load_test(cfg, proposed)
    if te is not None:
        paths.append(plot_confusion(
            te["label"].values, te["p_raw"].values, 0.5,
            os.path.join(fig_dir, "confusion_%s.png" % proposed),
            title="%s on the official test cohort" % proposed))

    cases_path = os.path.join(cfg.run.results_root, cfg.run.run_name, proposed,
                              "agreement_cases.csv")
    if os.path.exists(cases_path):
        cases = pd.read_csv(cases_path)
        if not cases.empty:
            paths.append(plot_evidence_distribution(
                cases, os.path.join(fig_dir, "evidence_channels.png")))
            paths.append(plot_support_contradiction(
                cases, os.path.join(fig_dir, "support_vs_contradiction.png")))

    log("figures written: " + ", ".join(os.path.basename(p) for p in paths))
    return paths
