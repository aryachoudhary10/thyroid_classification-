"""Command-line driver for the whole study (same code path as the notebook).

    python experiments/run_all.py --stage all
    python experiments/run_all.py --stage train --models mr_mil der_mil
    python experiments/run_all.py --stage external
    python experiments/run_all.py --status
    python experiments/run_all.py --reset-prefix robust/

Every stage is resumable: re-running the same command after an interruption
skips what is finished and continues the rest.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline
from src.config import Config
from src.models.factory import ABLATION_LADDER, ALL_MAIN
from src.utils.checkpoint import StageRegistry
from src.utils.common import banner, log, set_seed


def build_config(args) -> Config:
    cfg = Config()
    cfg.run.run_name = args.run_name
    cfg.run.ckpt_root = args.ckpt_root
    cfg.run.results_root = args.results_root
    cfg.run.seed = args.seed
    cfg.run.debug_subset = args.debug_subset
    cfg.data.thyroidxl_root = args.thyroidxl_root
    cfg.data.tn5000_root = args.tn5000_root
    cfg.data.num_workers = args.workers
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.epochs:
        cfg.optim.stage1_epochs = args.epochs
        cfg.optim.stage2_epochs = args.epochs
    if args.batch_size:
        cfg.optim.batch_size = args.batch_size
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "train", "ablation", "robust",
                             "external", "report"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--proposed", default="der_mil")
    ap.add_argument("--final-mode", default="refit", choices=["refit", "ensemble"])
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--ckpt-root", default="./checkpoints")
    ap.add_argument("--results-root", default="./results")
    ap.add_argument("--thyroidxl-root", default=None)
    ap.add_argument("--tn5000-root", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--debug-subset", type=int, default=0)
    ap.add_argument("--download", action="store_true",
                    help="fetch the datasets from Kaggle if roots are unset")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", default=None, help="reset one registry key")
    ap.add_argument("--reset-prefix", default=None, help="reset every key with this prefix")
    args = ap.parse_args()

    cfg = build_config(args)
    registry = StageRegistry(cfg.run.ckpt_root)

    if args.status:
        print(registry.summary())
        return 0
    if args.reset:
        registry.reset(args.reset)
        log("reset " + args.reset)
        return 0
    if args.reset_prefix:
        n = registry.reset_prefix(args.reset_prefix)
        log("reset %d keys under %s" % (n, args.reset_prefix))
        return 0

    if args.download or not cfg.data.thyroidxl_root:
        cfg = pipeline.download_datasets(cfg)

    set_seed(cfg.run.seed)
    models = args.models or ALL_MAIN
    stage = args.stage

    if stage in ("all", "data"):
        pipeline.inspect_datasets(cfg)
    manifest = pipeline.prepare_thyroidxl(cfg, registry)
    if stage == "data":
        return 0

    if stage in ("all", "train"):
        pipeline.run_models(cfg, manifest, registry, models, args.final_mode)
    if stage in ("all", "ablation"):
        extra = [m for m in ABLATION_LADDER + ["mask_channel"] if m not in models]
        pipeline.run_models(cfg, manifest, registry, extra, args.final_mode)
    if stage in ("all", "robust"):
        rob_models = [m for m in models + ["mask_channel"]
                      if m in ("mr_mil", "der_mil", "mask_channel",
                               "lesion_mil", "image_mil")]
        pipeline.run_robustness(cfg, manifest, registry, rob_models,
                                args.proposed, args.final_mode)
    if stage in ("all", "external"):
        pipeline.run_external(cfg, registry, sorted({"mr_mil", args.proposed}))
    if stage in ("all", "report"):
        all_models = sorted(set(models) | set(ABLATION_LADDER))
        pipeline.build_report(cfg, manifest, all_models, args.proposed)
        pipeline.make_figures(cfg, manifest, registry, models, args.proposed)

    banner("DONE")
    print(registry.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
