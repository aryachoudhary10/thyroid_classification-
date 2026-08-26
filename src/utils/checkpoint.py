"""Crash-safe checkpointing and a stage registry for resumable pipelines.

Two independent mechanisms:

1. ``CheckpointManager`` -- per-training-run state (model, optimiser, scaler,
   epoch, global step, best metric, RNG). Written atomically so a runtime kill
   in the middle of a save cannot corrupt the file.

2. ``StageRegistry`` -- a coarse-grained "what is already finished" ledger for
   the whole pipeline (manifest built? fold 3 trained? TN5000 adapted?).
   Re-running any driver script skips completed stages instead of redoing GPU
   work.

Both are keyed off ``ckpt_root``. Point that at Google Drive in Colab and a
runtime restart costs you nothing but the time to re-mount.
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Any, Callable, Dict, Optional

import torch

from .common import load_json, log, rng_state, load_rng_state, save_json


# --------------------------------------------------------------------------- #
def atomic_torch_save(obj: Any, path: str) -> None:
    """Write to ``path.tmp`` then rename -- rename is atomic on one filesystem."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
class StageRegistry:
    """A JSON ledger of completed pipeline stages."""

    def __init__(self, ckpt_root: str):
        self.root = ckpt_root
        self.path = os.path.join(ckpt_root, "registry.json")
        os.makedirs(ckpt_root, exist_ok=True)
        self._data: Dict[str, Any] = load_json(self.path, default={}) or {}

    # -- queries ---------------------------------------------------------- #
    def is_done(self, key: str) -> bool:
        return bool(self._data.get(key, {}).get("done", False))

    def get(self, key: str) -> Dict[str, Any]:
        return dict(self._data.get(key, {}))

    def artifacts(self, key: str) -> Dict[str, Any]:
        return dict(self._data.get(key, {}).get("artifacts", {}))

    # -- mutation --------------------------------------------------------- #
    def mark_done(self, key: str, artifacts: Optional[Dict[str, Any]] = None) -> None:
        self._data[key] = {
            "done": True,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifacts": artifacts or {},
        }
        save_json(self._data, self.path)

    def reset(self, key: str) -> None:
        self._data.pop(key, None)
        save_json(self._data, self.path)

    def reset_prefix(self, prefix: str) -> int:
        keys = [k for k in self._data if k.startswith(prefix)]
        for k in keys:
            self._data.pop(k, None)
        save_json(self._data, self.path)
        return len(keys)

    def summary(self) -> str:
        if not self._data:
            return "(registry empty -- nothing completed yet)"
        rows = sorted(self._data.items())
        width = max(len(k) for k, _ in rows)
        out = []
        for k, v in rows:
            flag = "DONE" if v.get("done") else "    "
            out.append("  [" + flag + "] " + k.ljust(width) + "  " + str(v.get("finished_at", "")))
        return "\n".join(out)

    # -- convenience ------------------------------------------------------ #
    def run_once(self, key: str, fn: Callable[[], Optional[Dict[str, Any]]],
                 force: bool = False) -> Dict[str, Any]:
        """Execute ``fn`` unless ``key`` is already complete."""
        if self.is_done(key) and not force:
            log("SKIP  " + key + "  (already complete)")
            return self.artifacts(key)
        log("RUN   " + key)
        artifacts = fn() or {}
        self.mark_done(key, artifacts)
        return artifacts


# --------------------------------------------------------------------------- #
class CheckpointManager:
    """Per-run model/optimiser state with best + last + optional epoch history."""

    def __init__(self, ckpt_root: str, run_name: str, keep_epoch_copies: bool = False):
        self.dir = os.path.join(ckpt_root, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.last_path = os.path.join(self.dir, "last.pt")
        self.best_path = os.path.join(self.dir, "best.pt")
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.keep_epoch_copies = keep_epoch_copies

    # -- save ------------------------------------------------------------- #
    def save(self,
             model: torch.nn.Module,
             optimizer: Optional[torch.optim.Optimizer],
             scaler: Optional[Any],
             *,
             stage: int,
             epoch: int,
             global_step: int,
             step_in_epoch: int,
             best_metric: float,
             best_stage: int,
             best_epoch: int,
             epochs_no_improve: int,
             extra: Optional[Dict[str, Any]] = None,
             is_best: bool = False,
             store_rng: bool = True) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "stage": stage,
            "epoch": epoch,
            "global_step": global_step,
            "step_in_epoch": step_in_epoch,
            "best_metric": best_metric,
            "best_stage": best_stage,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "extra": extra or {},
            "rng": rng_state() if store_rng else None,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_torch_save(payload, self.last_path)
        if is_best:
            # Copy rather than re-serialise: cheaper and guaranteed identical.
            shutil.copyfile(self.last_path, self.best_path + ".tmp")
            os.replace(self.best_path + ".tmp", self.best_path)
        if self.keep_epoch_copies:
            shutil.copyfile(self.last_path,
                            os.path.join(self.dir, "epoch_%03d.pt" % epoch))
        save_json({k: v for k, v in payload.items()
                   if k in ("stage", "epoch", "global_step", "step_in_epoch",
                            "best_metric", "best_stage", "best_epoch",
                            "epochs_no_improve", "saved_at")},
                  self.meta_path)

    # -- load ------------------------------------------------------------- #
    def has_last(self) -> bool:
        return os.path.exists(self.last_path)

    def has_best(self) -> bool:
        return os.path.exists(self.best_path)

    def load(self, path: str, model: Optional[torch.nn.Module] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scaler: Optional[Any] = None,
             map_location: str = "cpu",
             restore_rng: bool = True,
             strict: bool = True) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if model is not None:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)  # type: ignore[misc]
            if not strict and (missing or unexpected):
                log("  state_dict: %d missing, %d unexpected keys"
                    % (len(missing), len(unexpected)))
        if optimizer is not None and ckpt.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as exc:                          # noqa: BLE001
                log("  WARN: optimizer state not restored (" + str(exc) + ")")
        if scaler is not None and ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:                                  # noqa: BLE001
                pass
        if restore_rng:
            load_rng_state(ckpt.get("rng"))
        return ckpt

    def load_last(self, **kw) -> Optional[Dict[str, Any]]:
        if not self.has_last():
            return None
        return self.load(self.last_path, **kw)

    def load_best(self, **kw) -> Optional[Dict[str, Any]]:
        if not self.has_best():
            return None
        return self.load(self.best_path, **kw)

    def clear(self) -> None:
        for p in (self.last_path, self.best_path, self.meta_path):
            if os.path.exists(p):
                os.remove(p)
