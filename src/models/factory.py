"""Model registry.

Each entry maps an experiment name to (constructor, dataset requirements). The
dataset requirements tell the loader whether region maps are needed and at what
resolution, so a baseline never pays for tensors it does not use.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch.nn as nn

from ..config import Config, ModelConfig
from .baselines import (ImageOnlyMIL, LesionCropMIL, MaskChannelMIL, PoolingMIL,
                        TransformerBagMIL)
from .der_mil import DERMIL


# --------------------------------------------------------------------------- #
def _needs(regions: bool, res: int = 28) -> Dict[str, object]:
    return {"need_regions": regions, "region_res": res}


MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    # ---- naive references ------------------------------------------------ #
    "mean_pool": {"build": lambda c: PoolingMIL(c.model, "mean"),
                  "data": _needs(False), "group": "naive"},
    "max_pool": {"build": lambda c: PoolingMIL(c.model, "max"),
                 "data": _needs(False), "group": "naive"},

    # ---- fair patient-level baselines ------------------------------------ #
    "image_mil": {"build": lambda c: ImageOnlyMIL(c.model),
                  "data": _needs(False), "group": "fair"},
    "lesion_mil": {"build": lambda c: LesionCropMIL(c.model),
                   "data": _needs(False), "group": "fair"},
    "transformer_bag": {"build": lambda c: TransformerBagMIL(c.model),
                        "data": _needs(False), "group": "fair"},

    # ---- shortcut diagnostics (NOT fair baselines) ----------------------- #
    "mask_channel": {"build": lambda c: MaskChannelMIL(c.model),
                     "data": _needs(False), "group": "diagnostic"},

    # ---- proposed model and its ablation ladder -------------------------- #
    "mr_mil": {"build": lambda c: DERMIL(c.model, use_reliability=False),
               "data": _needs(True), "group": "proposed"},
    "der_i": {"build": lambda c: DERMIL(_flags(c.model, sup=False, con=False, unc=False)),
              "data": _needs(True), "group": "proposed"},
    "der_is": {"build": lambda c: DERMIL(_flags(c.model, sup=True, con=False, unc=False)),
               "data": _needs(True), "group": "proposed"},
    "der_isd": {"build": lambda c: DERMIL(_flags(c.model, sup=True, con=True, unc=False)),
                "data": _needs(True), "group": "proposed"},
    "der_iu": {"build": lambda c: DERMIL(_flags(c.model, sup=False, con=False, unc=True)),
               "data": _needs(True), "group": "proposed"},
    "der_mil": {"build": lambda c: DERMIL(_flags(c.model, sup=True, con=True, unc=True)),
                "data": _needs(True), "group": "proposed"},

    # ---- arm 7: same model, plus derived-descriptor text ----------------- #
    "der_mil_vl": {"build": lambda c: DERMIL(
        _with(_flags(c.model, sup=True, con=True, unc=True), use_descriptors=True)),
        "data": _needs(True), "group": "proposed"},
}


def _with(mc: ModelConfig, **overrides) -> ModelConfig:
    """Copy a ModelConfig with a few fields replaced."""
    d = dict(vars(mc))
    d.update(overrides)
    return ModelConfig(**d)


def _flags(mc: ModelConfig, sup: bool, con: bool, unc: bool) -> ModelConfig:
    return _with(mc, use_support=sup, use_contradiction=con, use_uncertainty=unc)


# --------------------------------------------------------------------------- #
def build_model(cfg: Config, name: str = None) -> Tuple[nn.Module, Dict[str, object]]:
    name = name or cfg.model.name
    if name not in MODEL_REGISTRY:
        raise KeyError("unknown model '%s'. Available: %s"
                       % (name, ", ".join(sorted(MODEL_REGISTRY))))
    entry = MODEL_REGISTRY[name]
    model = entry["build"](cfg)                                # type: ignore[operator]
    reqs = dict(entry["data"])                                 # type: ignore[arg-type]
    # masked_input evidence needs full-resolution region maps
    if name.startswith(("der_", "mr_")) and cfg.model.evidence_mode == "masked_input":
        reqs["region_res"] = cfg.data.image_size
    reqs["regions"] = tuple(cfg.model.regions) if reqs["need_regions"] else ("core",)
    return model, reqs


def model_group(name: str) -> str:
    return str(MODEL_REGISTRY.get(name, {}).get("group", "other"))


FAIR_BASELINES = [n for n, e in MODEL_REGISTRY.items() if e["group"] == "fair"]
ABLATION_LADDER = ["lesion_mil", "mr_mil", "der_i", "der_is",
                   "der_isd", "der_iu", "der_mil"]
ALL_MAIN = ["mean_pool", "max_pool", "image_mil", "lesion_mil",
            "transformer_bag", "mr_mil", "der_mil"]
