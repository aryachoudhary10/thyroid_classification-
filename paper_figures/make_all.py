"""Regenerate every figure in this set from scratch.

Run:  python make_all.py
Requires: matplotlib, numpy, pandas, scipy, scikit-learn, torch (fig09 only).
All data is read from ../kaggle_jobs/**; nothing is downloaded or fabricated.
"""
from __future__ import annotations

import importlib
import time

MODULES = [
    "fig01_dataset", "fig02_performance", "fig03_ablation",
    "fig05_calibration", "fig07_significance",
    "fig08_external", "fig09_computational", "fig10_architecture",
    "fig11_error_analysis",
]
# Note: no fig04 (reliability mechanism) or fig06 (robustness) here -- those
# analyses were only ever run on the ResNet-50 baseline and have no ConvNeXt-
# Tiny equivalent to build. See resnet50_supplementary/ for both.


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
