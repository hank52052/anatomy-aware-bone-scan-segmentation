"""Generic training loop for bone-scan stages (Stage 2 anatomy / Stage 3 lesion).

For simplicity we do NOT use PyTorch Lightning — the loop is plain torch, which
keeps everything in a single file with no extra dependency surface. The loop
exposes hooks for logging history (for plotting), checkpointing, and validation.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import CompositeLoss


# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-3              # 3e-3 caused divergence at ep 6; reverted
    weight_decay: float = 5e-5
    warmup_epochs: int = 2        # shorter — 5 epochs of low LR was wasting time
    grad_clip: float = 1.0        # tightened from 5.0 to prevent loss spikes
    seed: int = 0                 # reproducibility
    amp: bool = False    # disabled by default — losses can become unstable in fp16; enable explicitly after verifying
    amp_dtype: str = "fp16"  # "bf16" keeps the fp32 exponent range, so no GradScaler is needed
    val_every: int = 1
    save_dir: Path = Path("runs/stage3")
    device: str = "cuda"
    # Early stopping — 2026-05-19 relaxed after overfit diagnostic showed model
    # needs 50-80 epochs to converge (early-stop kept cutting at ep 10-20 prematurely)
    early_stop: bool = True
    es_patience: int = 25        # was 8 — model val_dice can oscillate for 20+ epochs while LR decays
    es_min_delta: float = 1e-3
    es_metric: str = "val_lesion_dice"
    es_min_epochs: int = 40      # was 10 — must train at least 40 epochs to let LR cosine decay complete
    # curriculum (Stage 3): start with alpha=1.0 (GT mask) then anneal to 0 (prediction)
    curriculum_start_alpha: float = 1.0
    curriculum_end_alpha: float = 0.0
    curriculum_warm_epochs: int = 5      # full GT for this many
    curriculum_anneal_epochs: int = 15   # linear anneal over this many
    # which loss tasks active
    use_anatomy_task: bool = True
    use_lesion_task: bool = True


def cosine_warmup_lr(epoch: int, total: int, warmup: int, base_lr: float,
                     min_lr_ratio: float = 0.05) -> float:
    """Linear warmup + cosine decay to base_lr * min_lr_ratio."""
    if epoch < warmup:
        # smoother sub-linear warmup to avoid divergence at high base LR
        return base_lr * (epoch + 1) / max(warmup, 1)
    if total - warmup <= 0:
        return base_lr
    t = (epoch - warmup) / (total - warmup)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def curriculum_alpha(epoch: int, cfg: TrainConfig) -> float:
    """1.0 (full GT) → 0.0 (full prediction) per curriculum schedule."""
    if epoch < cfg.curriculum_warm_epochs:
        return cfg.curriculum_start_alpha
    rel = (epoch - cfg.curriculum_warm_epochs) / max(cfg.curriculum_anneal_epochs, 1)
    if rel >= 1.0:
        return cfg.curriculum_end_alpha
    return cfg.curriculum_start_alpha + (cfg.curriculum_end_alpha - cfg.curriculum_start_alpha) * rel


# ---------------------------------------------------------------------------
def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _dice_per_view(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """logits (B, C, H, W), target (B, H, W) — binary FG vs background.

    NOTE: only counts samples where GT actually has foreground. Cases with
    empty GT (no lesion present) used to return Dice=1.0 via 0/0+eps which
    silently inflated the metric — now they are skipped.
    """
    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits[:, 0]) > 0.5).float()
    else:
        pred = (logits.argmax(dim=1) > 0).float()
    t = (target > 0).float()
    gt_has_fg = (t.sum(dim=(1, 2)) > 0)
    if not gt_has_fg.any():
        return float("nan")
    inter = (pred[gt_has_fg] * t[gt_has_fg]).sum(dim=(1, 2))
    denom = pred[gt_has_fg].sum(dim=(1, 2)) + t[gt_has_fg].sum(dim=(1, 2))
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.mean())


@torch.no_grad()
def evaluate(model, loader, device, return_preds: bool = False):
    """Compute per-sample lesion + anatomy Dice on the entire val set.

    Only samples whose GT mask has foreground contribute to the Dice average,
    so non-meta / non-lesion-labeled patients don't inflate the metric.
    """
    model.eval()
    lesion_dice_list, anatomy_dice_list = [], []
    preds = []
    eps = 1e-6
    for batch in loader:
        batch = _to_device(batch, device)
        x_ant, x_post = batch["image"][:, 0:1], batch["image"][:, 1:2]
        out = model(x_ant, x_post,
                    anatomy_gt_ant=None, anatomy_gt_post=None,
                    curriculum_alpha=0.0)
        for view, idx in [("ant", 0), ("post", 1)]:
            l_l = out[f"lesion_logits_{view}"]
            les_t = batch["lesion_mask"][:, idx]
            # per-sample lesion Dice
            pred_les = (l_l.argmax(dim=1) > 0).float()
            t_les = (les_t > 0).float()
            for b in range(pred_les.shape[0]):
                if t_les[b].sum() > 0 and int(batch["has_lesion_mask"][b]) == 1:
                    inter = (pred_les[b] * t_les[b]).sum()
                    denom = pred_les[b].sum() + t_les[b].sum()
                    d = (2 * inter + eps) / (denom + eps)
                    lesion_dice_list.append(float(d.cpu()))

            ana_t = batch["anatomy_mask"][:, idx]
            if int(batch["has_anatomy_mask"][0]) == 1:
                a_l = out[f"anatomy_logits_{view}"]
                pred_ana = (a_l.argmax(dim=1) > 0).float()
                t_ana = (ana_t > 0).float()
                for b in range(pred_ana.shape[0]):
                    if t_ana[b].sum() > 0 and int(batch["has_anatomy_mask"][b]) == 1:
                        inter = (pred_ana[b] * t_ana[b]).sum()
                        denom = pred_ana[b].sum() + t_ana[b].sum()
                        d = (2 * inter + eps) / (denom + eps)
                        anatomy_dice_list.append(float(d.cpu()))

            if return_preds:
                preds.append({
                    "patient_id": batch["patient_id"], "view": view,
                    "lesion_logits": l_l.detach().cpu(),
                    "anatomy_logits": out[f"anatomy_logits_{view}"].detach().cpu(),
                    "lesion_mask": les_t.detach().cpu(),
                    "anatomy_mask": ana_t.detach().cpu(),
                })
    import numpy as _np
    out_metrics = {
        "lesion_dice": float(_np.mean(lesion_dice_list)) if lesion_dice_list else float("nan"),
        "anatomy_dice": float(_np.mean(anatomy_dice_list)) if anatomy_dice_list else float("nan"),
        "n_lesion": len(lesion_dice_list),
        "n_anatomy": len(anatomy_dice_list),
    }
    if return_preds:
        return out_metrics, preds
    return out_metrics


# ---------------------------------------------------------------------------
def fit(model, train_loader: DataLoader, val_loader: DataLoader | None,
        cfg: TrainConfig, loss_fn: CompositeLoss,
        on_epoch_end=None) -> dict:
    """Plain torch training loop. Returns history dict."""
    # Reproducibility: same seed → same model behaviour run-to-run for diagnosis
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.amp_dtype == "fp16")
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    cfg.save_dir = Path(cfg.save_dir)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    history = {"epoch": [], "lr": [], "alpha": [],
               "train_loss": [], "train_les_dice_loss": [], "train_ana_dice_loss": [],
               "train_les_focal": [], "train_ana_cons": [],
               "val_lesion_dice": [], "val_anatomy_dice": [],
               "epoch_seconds": [], "early_stopped": []}
    best_val = -1.0
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(cfg.epochs):
        lr = cosine_warmup_lr(epoch, cfg.epochs, cfg.warmup_epochs, cfg.lr)
        for g in optim.param_groups:
            g["lr"] = lr
        alpha = curriculum_alpha(epoch, cfg)
        model.train()
        t0 = time.time()
        sums = {"total": 0.0, "les_dice": 0.0, "ana_dice": 0.0, "les_focal": 0.0, "ana_cons": 0.0}
        n_steps = 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            x_ant = batch["image"][:, 0:1]
            x_post = batch["image"][:, 1:2]
            gt_a_ant = batch["anatomy_mask"][:, 0:1].float() if cfg.use_anatomy_task else None
            gt_a_post = batch["anatomy_mask"][:, 1:2].float() if cfg.use_anatomy_task else None
            with torch.amp.autocast("cuda", enabled=cfg.amp, dtype=amp_dtype):
                out = model(x_ant, x_post,
                            anatomy_gt_ant=gt_a_ant,
                            anatomy_gt_post=gt_a_post,
                            curriculum_alpha=alpha)
                terms = loss_fn(out, batch["lesion_mask"], batch["anatomy_mask"],
                                has_lesion_mask=batch.get("has_lesion_mask"),
                                has_anatomy_mask=batch.get("has_anatomy_mask"),
                                use_anatomy=cfg.use_anatomy_task,
                                use_lesion=cfg.use_lesion_task)
                loss = terms["total"]
            optim.zero_grad(set_to_none=True)
            # Guard: if every sample in batch was masked out by has_lesion_mask /
            # has_anatomy_mask, loss can degenerate to a 0 constant with no grad_fn.
            # Skip backward in that case (no-op step).
            if not (loss.requires_grad and loss.grad_fn is not None):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            for k in sums:
                v = terms.get(k, None)
                if isinstance(v, torch.Tensor):
                    sums[k] += float(v.detach().cpu())
            n_steps += 1
        n_steps = max(n_steps, 1)

        # logging
        history["epoch"].append(epoch)
        history["lr"].append(lr)
        history["alpha"].append(alpha)
        history["train_loss"].append(sums["total"] / n_steps)
        history["train_les_dice_loss"].append(sums["les_dice"] / n_steps)
        history["train_ana_dice_loss"].append(sums["ana_dice"] / n_steps)
        history["train_les_focal"].append(sums["les_focal"] / n_steps)
        history["train_ana_cons"].append(sums["ana_cons"] / n_steps)

        val_les = val_ana = float("nan")
        if val_loader is not None and (epoch % cfg.val_every == 0):
            m = evaluate(model, val_loader, device)
            val_les = m["lesion_dice"]
            val_ana = m["anatomy_dice"]
        history["val_lesion_dice"].append(val_les)
        history["val_anatomy_dice"].append(val_ana)
        history["epoch_seconds"].append(time.time() - t0)

        # checkpoint
        # monitor metric (higher is better)
        if cfg.es_metric == "val_lesion_dice":
            score = val_les if not np.isnan(val_les) else -1.0
        elif cfg.es_metric == "val_anatomy_dice":
            score = val_ana if not np.isnan(val_ana) else -1.0
        else:
            score = val_les if not np.isnan(val_les) else -1.0
        if score > best_val + cfg.es_min_delta:
            best_val = score
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "val_lesion_dice": val_les,
                "val_anatomy_dice": val_ana,
            }, cfg.save_dir / "best.pt")
        else:
            epochs_since_improve += 1
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
            "history": history,
        }, cfg.save_dir / "last.pt")

        # decide early stop
        will_stop = (cfg.early_stop
                     and epoch >= cfg.es_min_epochs
                     and epochs_since_improve >= cfg.es_patience)
        history["early_stopped"].append(int(will_stop))

        print(f"[ep {epoch:3d}] lr={lr:.5f} α={alpha:.2f}  "
              f"loss={sums['total']/n_steps:.4f}  "
              f"val_lesion_dice={val_les:.4f}  val_anatomy_dice={val_ana:.4f}  "
              f"best={best_val:.4f}@ep{best_epoch}  noImp={epochs_since_improve}/{cfg.es_patience}  "
              f"({history['epoch_seconds'][-1]:.1f}s)", flush=True)

        if on_epoch_end is not None:
            on_epoch_end(epoch, history)

        if will_stop:
            print(f"[early stop] no improvement for {cfg.es_patience} epochs "
                  f"(best={best_val:.4f}@ep{best_epoch}). Stopping at epoch {epoch}.", flush=True)
            break

    # final history dump
    (cfg.save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
