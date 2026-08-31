"""ResNet-encoder U-Net for thyroid nodule segmentation.

This exists for one purpose: TN5000 ships bounding boxes only, so the pixel-mask
arm of the external validation needs masks that do not exist in the dataset. The
segmenter is trained on ThyroidXL, where ground-truth pixel masks are available
for every frame, and then applied to TN5000. That is the same construction the
source paper used for its headline external number, which is what makes the two
directly comparable.

The segmenter is never trained on TN5000 and never sees a TN5000 label, so it
introduces no target-domain supervision into the classification protocol.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import _load_tv


# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    """Two 3x3 convolutions with BN and ReLU."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _Up(nn.Module):
    """Upsample to the skip's resolution, concatenate, then convolve.

    Bilinear upsampling rather than a transposed convolution: ultrasound masks
    are blobs, and transposed convolutions put checkerboard artefacts on blob
    boundaries, which is exactly where mask quality matters for a lesion branch.
    """

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.block = _Block(c_in + c_skip, c_out)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else [s * 2 for s in x.shape[-2:]]
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


# --------------------------------------------------------------------------- #
class ResUNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet encoder.

    Encoder channel widths are probed with a dummy forward pass rather than
    hard-coded, so resnet18/34/50/101 all work without a lookup table.
    """

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True,
                 decoder_channels: List[int] = None):
        super().__init__()
        net = _load_tv(backbone, pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # H/2
        self.pool = net.maxpool                                   # H/4
        self.layer1 = net.layer1                                  # H/4
        self.layer2 = net.layer2                                  # H/8
        self.layer3 = net.layer3                                  # H/16
        self.layer4 = net.layer4                                  # H/32

        with torch.no_grad():
            d = torch.zeros(1, 3, 64, 64)
            e0 = self.stem(d)
            e1 = self.layer1(self.pool(e0))
            e2 = self.layer2(e1)
            e3 = self.layer3(e2)
            e4 = self.layer4(e3)
        chans = [t.shape[1] for t in (e0, e1, e2, e3, e4)]

        dc = decoder_channels or [256, 128, 64, 48]
        self.up3 = _Up(chans[4], chans[3], dc[0])
        self.up2 = _Up(dc[0], chans[2], dc[1])
        self.up1 = _Up(dc[1], chans[1], dc[2])
        self.up0 = _Up(dc[2], chans[0], dc[3])
        self.head = nn.Conv2d(dc[3], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, 1, H, W) logits."""
        h, w = x.shape[-2:]
        e0 = self.stem(x)
        e1 = self.layer1(self.pool(e0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        d = self.up3(e4, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        d = self.up0(d, e0)
        logit = self.head(d)
        if logit.shape[-2:] != (h, w):
            logit = F.interpolate(logit, size=(h, w), mode="bilinear",
                                  align_corners=False)
        return logit

    def set_encoder_trainable(self, flag: bool) -> None:
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            for p in m.parameters():
                p.requires_grad_(flag)


# --------------------------------------------------------------------------- #
def dice_bce_loss(logit: torch.Tensor, target: torch.Tensor,
                  dice_weight: float = 0.5, eps: float = 1.0) -> torch.Tensor:
    """BCE plus soft Dice.

    Thyroid nodules cover a small fraction of the frame, so BCE alone drifts
    toward predicting background everywhere. Dice supplies the overlap term that
    keeps small lesions from being optimised away.
    """
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = 1.0 - (num / den).mean()
    return (1.0 - dice_weight) * bce + dice_weight * dice


@torch.no_grad()
def dice_score(logit: torch.Tensor, target: torch.Tensor,
               thr: float = 0.5, eps: float = 1.0) -> torch.Tensor:
    """Hard Dice per sample, for monitoring."""
    p = (torch.sigmoid(logit) > thr).float()
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return num / den
