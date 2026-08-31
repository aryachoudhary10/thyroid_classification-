"""Central configuration for the DER-MIL project.

Every knob that changes an experiment lives here so a run is fully reproducible
from a single serialised dict, which is stored next to each checkpoint.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    # Filled in at runtime by kagglehub + discovery.
    thyroidxl_root: Optional[str] = None
    tn5000_root: Optional[str] = None

    image_size: int = 224
    t_max: int = 10                       # paper: Tmax = 10, no patient exceeds it
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ---- region generation (DER-MIL) ---------------------------------------
    # Ring widths in pixels at image_size resolution.
    margin_inner_px: int = 6              # erode(M) -> inner edge of the ring
    margin_outer_px: int = 6              # dilate(M) -> outer edge of the ring
    peri_px: int = 22                     # dilate(M, peri) minus M
    region_names: Tuple[str, ...] = ("core", "margin", "peri", "global")

    # ---- augmentation ------------------------------------------------------
    aug_hflip: float = 0.5
    aug_rotate_deg: float = 8.0
    aug_translate: float = 0.05
    aug_scale: Tuple[float, float] = (0.92, 1.08)
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15

    num_workers: int = 2
    pin_memory: bool = True


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    name: str = "der_mil"                 # see models/factory.py MODEL_REGISTRY
    backbone: str = "resnet50"
    pretrained: bool = True
    embed_dim: int = 256
    attn_dim: int = 128                   # paper: d_attn = d_embed / 2
    dropout: float = 0.25

    # ---- arm 7: derived-descriptor vision-language fusion -------------
    use_descriptors: bool = False         # der_mil_vl turns this on
    descriptor_dim: int = 128

    # ---- DER-MIL evidence encoder -----------------------------------------
    # roi_pool     : ONE backbone pass per frame; region embeddings come from
    #                masked pooling of the spatial feature map. Colab friendly.
    # masked_input : K backbone passes per frame on region-masked images.
    #                Faithful to the published elementwise-product formulation but
    #                K times the compute.
    evidence_mode: str = "roi_pool"
    regions: Tuple[str, ...] = ("core", "margin", "peri", "global")
    # How the K evidence streams are combined into one frame embedding.
    #   softmax : one scalar weight per region (K numbers)
    #   gated   : one weight per region PER DIMENSION (K x D), which is what
    #             makes this a strict generalisation of the published gated
    #             fusion (Sherif et al., Sci Rep 2026), which gates 256
    #             dimensions independently, so a scalar mixture
    #             is strictly less expressive despite having more streams.
    evidence_fusion: str = "softmax"

    # ---- DER-MIL reliability components (each independently switchable) ----
    use_support: bool = True
    use_contradiction: bool = True
    use_uncertainty: bool = True
    reliability_fn: str = "mlp"           # "mlp" | "linear"
    rel_alpha_init: float = 1.0
    rel_beta_init: float = 1.0
    rel_gamma_init: float = 1.0
    signed_graph: bool = True             # separate A+ / A- message passing
    graph_layers: int = 1
    mc_dropout_samples: int = 0           # >0 enables MC-dropout uncertainty at eval

    # ---- transformer-bag baseline -----------------------------------------
    tf_layers: int = 2
    tf_heads: int = 4


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    pos_weight_from_data: bool = True
    lambda_consistency: float = 0.05      # support-weighted cross-view consistency
    lambda_reliability: float = 0.01      # anti-collapse regulariser
    lambda_counterfactual: float = 0.0    # 0 -> evaluation-only (recommended first)
    cf_margin: float = 0.05
    cf_pairs_per_bag: int = 4


# --------------------------------------------------------------------------- #
# Optimisation (two-stage schedule, paper Table 2)
# --------------------------------------------------------------------------- #
@dataclass
class OptimConfig:
    stage1_lr: float = 1e-3               # frozen backbone
    stage1_epochs: int = 10
    stage1_patience: int = 3
    stage2_lr: float = 1e-4               # fine-tune layer3 / layer4
    stage2_epochs: int = 15
    stage2_patience: int = 4
    weight_decay: float = 1e-4
    batch_size: int = 8                   # bags per batch
    grad_clip: float = 5.0
    amp: bool = True
    monitor: str = "roc_auc"


# --------------------------------------------------------------------------- #
# Evaluation / protocol
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    n_folds: int = 5
    bootstrap_test: int = 2000
    bootstrap_external: int = 1000
    ece_bins: int = 15
    sensitivity_target: float = 0.90
    prior_shift_correction: bool = True
    tta_views: int = 8                    # external validation


# --------------------------------------------------------------------------- #
# External domain adaptation (TN5000, Setup B)
# --------------------------------------------------------------------------- #
@dataclass
class ExternalConfig:
    enabled: bool = True
    epochs: int = 12
    lr: float = 5e-5
    warmup_epochs: int = 5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    unfreeze_epochs: Tuple[int, ...] = (0, 3, 6)   # progressive backbone unfreezing
    eval_subset_per_class: int = 125               # 125 malignant + 125 benign
    mask_source: str = "auto"                      # "pixel" | "bbox" | "auto"
    batch_size: int = 16


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
@dataclass
class SegConfig:
    """U-Net segmenter for the TN5000 pixel-mask arm.

    Trained on ThyroidXL ground-truth masks only; never sees TN5000 labels.
    """
    backbone: str = "resnet34"            # lighter than the classifier trunk
    epochs: int = 14
    lr: float = 3e-4
    batch_size: int = 16
    dice_weight: float = 0.5              # loss = 0.5*BCE + 0.5*softDice
    freeze_encoder_epochs: int = 2        # warm the decoder before unfreezing
    threshold: float = 0.5                # probability -> binary mask


@dataclass
class AdaptConfig:
    """Label-free TN5000 adaptation arms (4 and 5). No target labels are used."""
    # ---- arm 4: uncertainty-aware pseudo-labeling --------------------------
    upl_rounds: int = 2
    upl_epochs: int = 2
    upl_lr: float = 5e-5
    upl_confidence: float = 0.85          # |p - 0.5| gate on the TTA mean
    upl_max_std: float = 0.10             # across-view stability gate
    upl_min_samples: int = 8              # below this the arm is degenerate

    # ---- arm 5: test-time adaptation (TENT) --------------------------------
    tent_steps: int = 1                   # passes over the evaluation set
    tent_lr: float = 1e-3                 # normalisation affine params only

    # ---- arm 6: retrieval pseudo-bags --------------------------------------
    retrieval_bag_size: int = 5           # query + 4 nearest neighbours


@dataclass
class RunConfig:
    run_name: str = "der_mil"
    ckpt_root: str = "./checkpoints"
    results_root: str = "./results"
    seed: int = 1337
    device: str = "cuda"
    save_every_steps: int = 200           # in-epoch checkpoint cadence
    exact_resume: bool = True             # skip completed batches when resuming
    debug_subset: int = 0                 # >0 -> keep only N patients (smoke test)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    external: ExternalConfig = field(default_factory=ExternalConfig)
    seg: SegConfig = field(default_factory=SegConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        def build(cls, sub):
            names = {f.name for f in dataclasses.fields(cls)}
            kwargs = {}
            for k, v in (sub or {}).items():
                if k not in names:
                    continue
                kwargs[k] = tuple(v) if isinstance(v, list) else v
            return cls(**kwargs)

        return Config(
            data=build(DataConfig, d.get("data")),
            model=build(ModelConfig, d.get("model")),
            loss=build(LossConfig, d.get("loss")),
            optim=build(OptimConfig, d.get("optim")),
            eval=build(EvalConfig, d.get("eval")),
            external=build(ExternalConfig, d.get("external")),
            run=build(RunConfig, d.get("run")),
        )

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return Config.from_dict(json.load(fh))

    def clone(self, **overrides) -> "Config":
        """Clone with dotted overrides, e.g. clone(**{"model.name": "der_mil"})."""
        d = self.to_dict()
        for key, value in overrides.items():
            section, _, leaf = key.partition(".")
            if not leaf:
                raise KeyError("override keys must be dotted, got " + key)
            if section not in d:
                raise KeyError("unknown config section: " + section)
            d[section][leaf] = value
        return Config.from_dict(d)


def variant(base: Config, name: str, **overrides) -> Config:
    """Named experiment variant, given its own checkpoint namespace."""
    cfg = base.clone(**overrides) if overrides else Config.from_dict(base.to_dict())
    cfg.run.run_name = name
    return cfg
