"""Regenerate every ResNet-50 / RCAF supplementary figure from scratch.

Run:  python make_all.py
Writes to ./output/ (this folder), never touching the primary ConvNeXt-Tiny
figures in ../output/.
"""
from __future__ import annotations

import importlib
import time

MODULES = [
    "fig02_performance_resnet50", "fig04_reliability_mechanism_resnet50",
    "fig05_calibration_resnet50", "fig06_robustness_resnet50",
    "fig07_significance_resnet50", "fig08_external_resnet50",
    "fig09_computational_resnet50", "fig11_error_analysis_resnet50",
]


def main() -> None:
    for name in MODULES:
        t0 = time.time()
        print("=" * 70)
        print(name)
        print("=" * 70)
        mod = importlib.import_module(name)
        mod.main()
        print("  (%.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
