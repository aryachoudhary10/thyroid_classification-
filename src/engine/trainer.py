"""Two-stage trainer with crash-safe, mid-epoch resume.

Stage 1 -- backbone frozen, only fusion / reliability / MIL / head are trained.
Stage 2 -- layer3 and layer4 unfrozen and fine-tuned jointly.

Both stages early-stop on validation ROC-AUC and the best checkpoint is chosen
on the highest validation ROC-AUC observed across BOTH stages, matching the
protocol applied to every model in the comparison.

Resume semantics
----------------
``last.pt`` is rewritten every ``save_every_steps`` batches and at every epoch
boundary. It stores stage, epoch, step-in-epoch, optimiser, AMP scaler, best
metric, patience counter and RNG state. Calling ``fit()`` again after a runtime
restart continues from exactly where it stopped; with ``exact_resume`` the
partially finished epoch replays its shuffled order and skips the batches that
were already applied.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import PatientBagDataset
from ..eval.metrics import all_metrics
from ..losses.losses import TotalLoss
from ..utils.checkpoint import CheckpointManager
from ..utils.common import count_params, log


# --------------------------------------------------------------------------- #
def move_batch(batch: Dict[str, torch.Tensor], device: torch.device,
               keys: Tuple[str, ...] = ("image", "lesion", "regions", "valid", "label")
               ) -> Dict[str, torch.Tensor]:
    out = dict(batch)
    for k in keys:
        if k in out and torch.is_tensor(out[k]):
            out[k] = out[k].to(device, non_blocking=True)
    return out


class Trainer:
    def __init__(self,
                 cfg: Config,
                 model: nn.Module,
                 train_ds: PatientBagDataset,
                 val_ds: PatientBagDataset,
                 run_name: str,
                 pos_weight: Optional[float] = None,
                 use_focal: bool = False,
                 device: Optional[torch.device] = None):
        self.cfg = cfg
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = TotalLoss(
            cfg.loss, pos_weight if cfg.loss.pos_weight_from_data else None,
            use_focal=use_focal,
            focal_gamma=cfg.external.focal_gamma,
            label_smoothing=cfg.external.label_smoothing if use_focal else 0.0,
        ).to(self.device)

        self.ckpt = CheckpointManager(cfg.run.ckpt_root, run_name)
        self.run_name = run_name
        self.amp = cfg.optim.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def _stage_spec(self, stage: int) -> Tuple[float, int, int]:
        o = self.cfg.optim
        if stage == 1:
            return o.stage1_lr, o.stage1_epochs, o.stage1_patience
        return o.stage2_lr, o.stage2_epochs, o.stage2_patience

    def _build_optimizer(self, stage: int) -> torch.optim.Optimizer:
        if hasattr(self.model, "set_stage"):
            self.model.set_stage(stage)
        lr, _e, _p = self._stage_spec(stage)
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=self.cfg.optim.weight_decay)

    def _loader(self, ds: PatientBagDataset, shuffle: bool, epoch: int) -> DataLoader:
        c = self.cfg
        g = torch.Generator()
        g.manual_seed(c.run.seed * 1000 + epoch)
        return DataLoader(ds, batch_size=c.optim.batch_size, shuffle=shuffle,
                          num_workers=c.data.num_workers,
                          pin_memory=c.data.pin_memory and self.device.type == "cuda",
                          generator=g if shuffle else None,
                          persistent_workers=False,
                          drop_last=False)

    # ------------------------------------------------------------------ #
    def _sync(self, st):
        self._best = st["best_metric"]
        self._best_stage = st["best_stage"]
        self._best_epoch = st["best_epoch"]
        self._no_improve = st["epochs_no_improve"]

    def fit(self) -> Dict[str, Any]:
        st = {"stage": 1, "epoch": 0, "global_step": 0, "step_in_epoch": 0,
              "best_metric": -math.inf, "best_stage": 1, "best_epoch": -1,
              "epochs_no_improve": 0, "done": False}

        if self.ckpt.has_last():
            ck = self.ckpt.load_last(model=self.model, map_location=str(self.device))
            if ck is not None:
                for k in ("stage", "epoch", "global_step", "step_in_epoch",
                          "best_metric", "best_stage", "best_epoch", "epochs_no_improve"):
                    if ck.get(k) is not None:
                        st[k] = ck[k]
                extra = ck.get("extra") or {}
                self.history = extra.get("history", [])
                st["done"] = bool(extra.get("done", False))
                log("resume %s: stage %d, epoch %d, step %d, best %.4f%s"
                    % (self.run_name, st["stage"], st["epoch"], st["step_in_epoch"],
                       st["best_metric"], "  (already finished)" if st["done"] else ""))
        self._sync(st)
        if st["done"]:
            return self._finish(st)

        p = count_params(self.model)
        log("%s: %s parameters" % (self.run_name, format(p["total"], ",")))

        resume_stage, resume_epoch = int(st["stage"]), int(st["epoch"])
        for stage in (1, 2):
            if stage < resume_stage:
                continue
            lr, max_epochs, patience = self._stage_spec(stage)
            if max_epochs <= 0:
                continue

            optimizer = self._build_optimizer(stage)
            if stage == resume_stage and self.ckpt.has_last() and (
                    resume_epoch > 0 or st["step_in_epoch"] > 0):
                # Optimiser state is only meaningful inside the stage that wrote it.
                self.ckpt.load(self.ckpt.last_path, optimizer=optimizer,
                               scaler=self.scaler, map_location=str(self.device),
                               restore_rng=False)
                epoch = resume_epoch
                skip = int(st["step_in_epoch"])
            else:
                epoch, skip = 0, 0
                st["epochs_no_improve"] = 0
            st["stage"] = stage

            log("--- %s stage %d (lr=%.1e, <=%d epochs, patience %d) ---"
                % (self.run_name, stage, lr, max_epochs, patience))

            while epoch < max_epochs:
                st["epoch"] = epoch
                self._sync(st)
                tr = self._train_epoch(optimizer, stage, epoch,
                                       skip_batches=skip if self.cfg.run.exact_resume else 0,
                                       global_step=st["global_step"])
                skip = 0
                st["global_step"] = tr["global_step"]
                st["step_in_epoch"] = 0

                va = self.evaluate(self.val_ds)
                score = va.get(self.cfg.optim.monitor, float("nan"))
                improved = (score == score) and score > st["best_metric"] + 1e-6
                if improved:
                    st["best_metric"] = float(score)
                    st["best_stage"] = stage
                    st["best_epoch"] = epoch
                    st["epochs_no_improve"] = 0
                else:
                    st["epochs_no_improve"] += 1

                self.history.append({
                    "stage": stage, "epoch": epoch, "train_loss": tr["loss"],
                    "val_" + self.cfg.optim.monitor: float(score),
                    "val_pr_auc": va.get("pr_auc"), "val_f1": va.get("f1"),
                    "best": st["best_metric"], "lr": lr, "seconds": tr["seconds"]})
                log("  s%d e%02d  loss %.4f | val %s %.4f | best %.4f%s"
                    % (stage, epoch, tr["loss"], self.cfg.optim.monitor, score,
                       st["best_metric"], "  *" if improved else ""))

                epoch += 1
                st["epoch"] = epoch
                self._sync(st)
                self._save(optimizer, st, is_best=improved)

                if st["epochs_no_improve"] >= patience:
                    log("  early stop after %d epochs without improvement" % patience)
                    break

            st["epoch"] = 0
            st["step_in_epoch"] = 0
            st["epochs_no_improve"] = 0
            resume_epoch = 0

        st["done"] = True
        st["stage"] = 2
        self._sync(st)
        self._save(None, st, is_best=False)
        log("%s: finished. best val %s = %.4f (stage %d, epoch %d)"
            % (self.run_name, self.cfg.optim.monitor, st["best_metric"],
               st["best_stage"], st["best_epoch"]))
        return self._finish(st)

    # ------------------------------------------------------------------ #
    def _train_epoch(self, optimizer, stage: int, epoch: int, skip_batches: int,
                     global_step: int) -> Dict[str, Any]:
        self.model.train()
        loader = self._loader(self.train_ds, shuffle=True, epoch=epoch)
        self.train_ds.set_epoch(epoch)
        t0 = time.time()
        losses: List[float] = []
        n_batches = len(loader)
        if skip_batches:
            log("  replaying epoch %d, skipping %d/%d already-applied batches"
                % (epoch, skip_batches, n_batches))

        for i, batch in enumerate(loader):
            if i < skip_batches:
                continue
            batch = move_batch(batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                out = self.model(batch)
                parts = self.criterion(out, batch, model=self.model)
                loss = parts["total"]

            if not torch.isfinite(loss):
                log("  WARN non-finite loss at step %d -- batch skipped" % i)
                continue

            self.scaler.scale(loss).backward()
            if self.cfg.optim.grad_clip > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.optim.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()

            losses.append(float(loss.detach().cpu()))
            global_step += 1

            if (self.cfg.run.save_every_steps > 0
                    and global_step % self.cfg.run.save_every_steps == 0):
                self._save(optimizer,
                           {"stage": stage, "epoch": epoch, "global_step": global_step,
                            "step_in_epoch": i + 1, "best_metric": self._best,
                            "best_stage": self._best_stage, "best_epoch": self._best_epoch,
                            "epochs_no_improve": self._no_improve},
                           is_best=False)

        del loader
        return {"loss": float(np.mean(losses)) if losses else float("nan"),
                "global_step": global_step, "seconds": time.time() - t0}

    # ------------------------------------------------------------------ #
    _best = -math.inf
    _best_stage = 1
    _best_epoch = -1
    _no_improve = 0

    def _save(self, optimizer, state: Dict[str, Any], is_best: bool) -> None:
        self._best = state["best_metric"]
        self._best_stage = state["best_stage"]
        self._best_epoch = state["best_epoch"]
        self._no_improve = state["epochs_no_improve"]
        self.ckpt.save(self.model, optimizer, self.scaler,
                       stage=state["stage"], epoch=state["epoch"],
                       global_step=state["global_step"],
                       step_in_epoch=state["step_in_epoch"],
                       best_metric=state["best_metric"],
                       best_stage=state["best_stage"],
                       best_epoch=state["best_epoch"],
                       epochs_no_improve=state["epochs_no_improve"],
                       extra={"history": self.history, "done": bool(state.get("done", False)),
                              "run_name": self.run_name},
                       is_best=is_best)

    def _finish(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"best_metric": state["best_metric"], "best_stage": state["best_stage"],
                "best_epoch": state["best_epoch"], "history": self.history,
                "best_path": self.ckpt.best_path, "last_path": self.ckpt.last_path}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, ds: PatientBagDataset, use_best: bool = False) -> Dict[str, float]:
        if use_best and self.ckpt.has_best():
            self.ckpt.load_best(model=self.model, map_location=str(self.device),
                                restore_rng=False)
        logits, labels, _ = self.predict(ds)
        p = 1.0 / (1.0 + np.exp(-logits))
        return all_metrics(labels, p, 0.5)

    @torch.no_grad()
    def predict(self, ds: PatientBagDataset, collect: Tuple[str, ...] = (),
                use_best: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Returns (logits, labels, extras) in dataset order."""
        if use_best and self.ckpt.has_best():
            self.ckpt.load_best(model=self.model, map_location=str(self.device),
                                restore_rng=False)
        self.model.eval()
        loader = self._loader(ds, shuffle=False, epoch=0)
        logits, labels = [], []
        extras: Dict[str, List[np.ndarray]] = {k: [] for k in collect}
        for batch in loader:
            batch = move_batch(batch, self.device)
            with torch.amp.autocast("cuda", enabled=self.amp):
                out = self.model(batch)
            logits.append(out["logit"].float().detach().cpu().numpy())
            labels.append(batch["label"].detach().cpu().numpy())
            for k in collect:
                if k in out and torch.is_tensor(out[k]):
                    extras[k].append(out[k].float().detach().cpu().numpy())
        del loader
        L = np.concatenate(logits) if logits else np.zeros(0)
        Y = np.concatenate(labels) if labels else np.zeros(0)
        E = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in extras.items()}
        return L, Y, E


# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_dataset(model: nn.Module, ds: PatientBagDataset, cfg: Config,
                    device: torch.device, collect: Tuple[str, ...] = (),
                    batch_size: Optional[int] = None
                    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Standalone inference helper for robustness / counterfactual passes."""
    model.eval().to(device)
    loader = DataLoader(ds, batch_size=batch_size or cfg.optim.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers,
                        pin_memory=cfg.data.pin_memory and device.type == "cuda")
    logits, labels = [], []
    extras: Dict[str, List[np.ndarray]] = {k: [] for k in collect}
    amp = cfg.optim.amp and device.type == "cuda"
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(batch)
        logits.append(out["logit"].float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        for k in collect:
            if k in out and torch.is_tensor(out[k]):
                extras[k].append(out[k].float().cpu().numpy())
    del loader
    return (np.concatenate(logits), np.concatenate(labels),
            {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in extras.items()})
