"""Small shared helpers: seeding, device, JSON IO, logging."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"])
                            else state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    except Exception as exc:                                  # noqa: BLE001
        log("WARN: could not restore RNG state (" + str(exc) + ")")


def get_device(pref: str = "cuda") -> torch.device:
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    stamp = "[%7.1fs] " % elapsed
    print(stamp + str(msg), flush=True)


def banner(msg: str) -> None:
    line = "=" * 78
    print("\n" + line + "\n" + msg + "\n" + line, flush=True)


# --------------------------------------------------------------------------- #
def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                          # noqa: BLE001
        return default


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return "%3.1f%s" % (n, unit)
        n /= 1024.0
    return "%.1fPB" % n


def in_colab() -> bool:
    return "google.colab" in sys.modules
