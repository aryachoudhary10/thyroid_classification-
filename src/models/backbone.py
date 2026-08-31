"""Shared CNN encoder.

A single backbone instance is shared by every branch of every model so that
performance differences are attributable to the fusion / reliability design
rather than to representational capacity (paper, Methods).

``forward_maps`` exposes the intermediate spatial maps that DER-MIL needs for
region pooling; ``forward_vec`` is the plain global-pooled path used by the
simpler baselines.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision


# Encoders whose stages are discovered by probing rather than hard-coded.
# Arm 3 of the external-validation ladder swaps the ResNet trunk for one of
# these; every model keeps the same ``Backbone`` interface, so the swap is a
# config change rather than a code change.
FOUNDATION_BACKBONES = ("convnext_tiny", "convnext_small", "convnext_base",
                        "efficientnet_v2_s", "efficientnet_v2_m",
                        "swin_t", "swin_s", "swin_v2_t", "swin_v2_s")


BACKBONE_DIMS: Dict[str, Dict[str, int]] = {
    "resnet18": {"c4": 256, "c5": 512},
    "resnet34": {"c4": 256, "c5": 512},
    "resnet50": {"c4": 1024, "c5": 2048},
    "resnet101": {"c4": 1024, "c5": 2048},
}


def _local_weights(name: str) -> "Optional[str]":
    """Find a pretrained checkpoint on disk, for offline environments.

    Kaggle kernels on an unverified account have no network, so torchvision
    cannot fetch ImageNet weights. Set DERMIL_WEIGHTS_DIR to a directory (e.g.
    an attached weights dataset) holding files such as
    ``resnet50-0676ba61.pth``; the first file whose name starts with the
    backbone name is used.
    """
    root = os.environ.get("DERMIL_WEIGHTS_DIR")
    if not root or not os.path.isdir(root):
        return None
    for dirpath, _dn, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith(name) and fn.endswith((".pth", ".pt")):
                return os.path.join(dirpath, fn)
    return None


def _load_tv(name: str, pretrained: bool) -> nn.Module:
    """Load any torchvision classifier, preferring a local checkpoint offline."""
    fn = getattr(torchvision.models, name)
    if not pretrained:
        try:
            return fn(weights=None)
        except TypeError:                                      # very old torchvision
            return fn(pretrained=False)

    local = _local_weights(name)
    if local:
        try:
            net = fn(weights=None)
        except TypeError:
            net = fn(pretrained=False)
        state = torch.load(local, map_location="cpu", weights_only=True)
        missing, unexpected = net.load_state_dict(state, strict=False)
        print("backbone: loaded local ImageNet weights from %s "
              "(%d missing, %d unexpected)" % (local, len(missing), len(unexpected)))
        return net

    try:
        return fn(weights="IMAGENET1K_V1")
    except TypeError:                                          # very old torchvision
        return fn(pretrained=True)
    except Exception as exc:                                   # noqa: BLE001
        # Offline and no local copy: fail loudly rather than silently training
        # a randomly initialised backbone, which would look like a bad result.
        raise RuntimeError(
            "could not obtain pretrained weights for " + name + " (" + str(exc)[:120]
            + "). Either enable internet, or set DERMIL_WEIGHTS_DIR to a "
            "directory containing e.g. resnet50-0676ba61.pth, or set "
            "cfg.model.pretrained=False deliberately.") from exc


def _probe_stages(features: nn.Module, in_size: Tuple[int, int] = (128, 96)
                  ) -> List[Dict[str, object]]:
    """Run a dummy pass to learn each stage's stride, width and memory layout.

    Probed rather than tabulated because torchvision reorganises these module
    trees between releases, and because Swin emits NHWC while ConvNeXt and
    EfficientNet emit NCHW. The dummy input is deliberately non-square so the
    two spatial axes can be told apart from the channel axis.
    """
    h0, w0 = in_size
    x = torch.zeros(1, 3, h0, w0)
    out: List[Dict[str, object]] = []
    with torch.no_grad():
        for i, mod in enumerate(features):
            x = mod(x)
            if x.dim() != 4:
                continue
            # NHWC iff dims 1,2 hold the spatial axes in the input aspect ratio
            nhwc = (x.shape[1] * w0 == x.shape[2] * h0) and x.shape[1] != x.shape[3]
            h = x.shape[1] if nhwc else x.shape[2]
            c = x.shape[3] if nhwc else x.shape[1]
            out.append({"index": i, "stride": max(h0 // max(int(h), 1), 1),
                        "channels": int(c), "nhwc": bool(nhwc)})
    return out


def _to_nchw(x: torch.Tensor, nhwc: bool) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous() if nhwc else x


class Backbone(nn.Module):
    """ResNet trunk with a two-stage fine-tuning schedule."""

    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.name = name
        if name in BACKBONE_DIMS:
            self.kind = "resnet"
            net = _load_tv(name, pretrained)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.dim_c4 = BACKBONE_DIMS[name]["c4"]
            self.dim_c5 = BACKBONE_DIMS[name]["c5"]
        elif name in FOUNDATION_BACKBONES:
            self.kind = "foundation"
            net = _load_tv(name, pretrained)
            self.features = net.features
            stages = _probe_stages(self.features)
            s16 = [d for d in stages if d["stride"] == 16]
            s32 = [d for d in stages if d["stride"] >= 32]
            if not s16 or not s32:
                raise RuntimeError("could not locate stride-16/32 stages in " + name)
            self._t4, self._t5 = s16[-1], s32[-1]
            self.dim_c4 = int(self._t4["channels"])
            self.dim_c5 = int(self._t5["channels"])
            # Stage boundaries for progressive unfreezing, coarse-to-fine.
            self._bounds = [int(self._t5["index"]), int(self._t4["index"])]
        else:
            raise KeyError("unsupported backbone: " + name + ". Known: "
                           + ", ".join(sorted(set(BACKBONE_DIMS) | set(FOUNDATION_BACKBONES))))
        self.out_dim = self.dim_c5

    # ------------------------------------------------------------------ #
    def forward_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Spatial feature maps at stride 16 (``c4``) and stride 32 (``c5``).

        Always returned NCHW, so region pooling is layout-agnostic even when the
        encoder is a Swin variant that works internally in NHWC.
        """
        if self.kind == "resnet":
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            c4 = self.layer3(x)      # (B, dim_c4, H/16, W/16)  -> 14x14 @ 224
            c5 = self.layer4(c4)     # (B, dim_c5, H/32, W/32)  ->  7x7 @ 224
            return {"c4": c4, "c5": c5}

        c4 = c5 = None
        for i, mod in enumerate(self.features):
            x = mod(x)
            if i == self._t4["index"]:
                c4 = _to_nchw(x, bool(self._t4["nhwc"]))
            if i == self._t5["index"]:
                c5 = _to_nchw(x, bool(self._t5["nhwc"]))
                break
        return {"c4": c4, "c5": c5}

    def forward_vec(self, x: torch.Tensor) -> torch.Tensor:
        c5 = self.forward_maps(x)["c5"]
        return torch.flatten(nn.functional.adaptive_avg_pool2d(c5, 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_vec(x)

    # ------------------------------------------------------------------ #
    def _blocks(self) -> List[nn.Module]:
        """Trainable groups, deepest first -- the unfreezing order."""
        if self.kind == "resnet":
            return [self.layer4, self.layer3, self.layer2]
        # Foundation trunks: everything from the stride-32 tap, then from the
        # stride-16 tap, then the remainder.
        b32, b16 = self._bounds
        deep = nn.ModuleList([m for i, m in enumerate(self.features) if i > b16])
        mid = nn.ModuleList([m for i, m in enumerate(self.features)
                             if b32 - (b32 - b16) <= i <= b16])
        shallow = nn.ModuleList([m for i, m in enumerate(self.features) if i < b16])
        return [deep, mid, shallow]

    def set_stage(self, stage: int) -> None:
        """stage 1: everything frozen. stage 2: the two deepest groups train."""
        for p in self.parameters():
            p.requires_grad_(False)
        if stage >= 2:
            for m in self._blocks()[:2]:
                for p in m.parameters():
                    p.requires_grad_(True)

    def set_progressive_unfreeze(self, level: int) -> None:
        """TN5000 adaptation: 0 -> head only, then one more group per level."""
        for p in self.parameters():
            p.requires_grad_(False)
        for m in self._blocks()[:max(int(level), 0)]:
            for p in m.parameters():
                p.requires_grad_(True)

    def train(self, mode: bool = True):
        """Keep frozen BatchNorm in eval mode so its statistics do not drift."""
        super().train(mode)
        if not mode:
            return self
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                trainable = any(p.requires_grad for p in module.parameters())
                if not trainable:
                    module.eval()
        return self


# --------------------------------------------------------------------------- #
def flatten_bag(x: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) -> (B*T, C, H, W)"""
    b, t = x.shape[:2]
    return x.reshape(b * t, *x.shape[2:])


def unflatten_bag(x: torch.Tensor, b: int, t: int) -> torch.Tensor:
    """(B*T, D) -> (B, T, D)"""
    return x.reshape(b, t, *x.shape[1:])


# --------------------------------------------------------------------------- #
# Valid-frame compaction.
#
# Bags are padded to Tmax=10 but average 2.84 real frames, so running the
# backbone over every slot wastes roughly 3.5x the compute and memory on
# all-zero images. These helpers gather the valid frames into a dense batch
# before the encoder and scatter the embeddings back afterwards; padded slots
# end up as exact zeros, which is stricter than the masking that follows.
# --------------------------------------------------------------------------- #
def valid_index(valid: torch.Tensor) -> torch.Tensor:
    """(B, T) validity mask -> flat indices of the valid slots."""
    return (valid.reshape(-1) > 0.5).nonzero(as_tuple=True)[0]


def gather_frames(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) -> (N, C, H, W) keeping only the valid slots."""
    return x.reshape(-1, *x.shape[2:]).index_select(0, idx)


def scatter_frames(feat: torch.Tensor, idx: torch.Tensor, b: int, t: int) -> torch.Tensor:
    """(N, ...) -> (B, T, ...), padded slots filled with zeros."""
    out = feat.new_zeros(b * t, *feat.shape[1:])
    out.index_copy_(0, idx, feat)
    return out.reshape(b, t, *feat.shape[1:])
