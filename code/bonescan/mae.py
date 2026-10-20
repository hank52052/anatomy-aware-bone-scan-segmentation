"""Stage 1 — Masked Autoencoder (MAE) pretraining on the union of available images.

This is a lightweight MAE: we apply patch-level random masking on the input
(both ANT and POST), and train an encoder + small reconstruction decoder to
reconstruct the masked patches. The encoder weights are then loaded as the
initialization for Stage 2/3.

Key choices:
  - Patch size 16x16 — matches BS-80K wholeBody 256×1024 (16×64 patch grid).
  - Mask ratio 0.6 (60% of patches masked) — standard MAE setting.
  - Loss: MSE on masked patches only (normalized by patch std).
    Code release note: run_mae below uses the plain mean squared error over the masked pixels, without
    per-patch normalisation.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import DEFAULT_OUT, load_manifest
from .models import AnchorBSModel, BackboneConfig


# ---------------------------------------------------------------------------
class SingleViewMAEDataset(Dataset):
    """Each row = one (image_path, source). Returns single-channel 2D image (1, H, W)."""

    def __init__(self, manifest_df: pd.DataFrame, out_dir: Path = DEFAULT_OUT,
                 augment: bool = True):
        self.df = manifest_df[manifest_df.image_path != ""].reset_index(drop=True)
        self.out = Path(out_dir)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p = self.out / r.image_path
        try:
            arr = np.load(str(p)).astype(np.float32)
        except Exception:
            # fall back: load jpg from BS-80K crop catalog (image_path may be relative to data/)
            from PIL import Image
            full = Path(os.environ.get("BONESCAN_ROOT", ".")) / "data" / r.image_path
            im = Image.open(full).convert("L")
            arr = np.asarray(im, dtype=np.float32) / 255.0
        # ensure 2D
        if arr.ndim != 2:
            arr = arr.squeeze()
        # Resize/pad to 1024x256 if needed (BS-80K crops have variable sizes)
        if arr.shape != (1024, 256):
            from PIL import Image
            im = Image.fromarray((arr * 65535).clip(0, 65535).astype(np.uint16), mode="I;16")
            im = im.resize((256, 1024), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 65535.0
        # very light intensity augmentation
        if self.augment:
            g = np.random.uniform(0.9, 1.1)
            arr = np.clip(np.power(arr, g), 0.0, 1.0)
        return torch.from_numpy(arr).unsqueeze(0).float(), r.source


# ---------------------------------------------------------------------------
def random_patch_mask(B: int, H: int, W: int, patch: int = 16,
                       mask_ratio: float = 0.6, device: str = "cuda") -> torch.Tensor:
    """Return a (B, 1, H, W) float mask in {0, 1}, 1 = unmasked, 0 = masked."""
    nh, nw = H // patch, W // patch
    n_total = nh * nw
    n_keep = int(round(n_total * (1.0 - mask_ratio)))
    keep_idx = torch.zeros(B, n_total, device=device)
    for b in range(B):
        perm = torch.randperm(n_total, device=device)
        keep_idx[b, perm[:n_keep]] = 1.0
    keep_idx = keep_idx.view(B, 1, nh, nw)
    keep = F.interpolate(keep_idx, scale_factor=patch, mode="nearest")
    return keep


@dataclass
class MAEConfig:
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 5e-5
    warmup_epochs: int = 3
    mask_ratio: float = 0.6
    patch_size: int = 16
    save_dir: Path = Path("runs/stage1_mae")
    device: str = "cuda"


def cosine_warmup(epoch, total, warmup, base, min_ratio=0.01):
    if epoch < warmup:
        return base * (epoch + 1) / max(warmup, 1)
    if total - warmup <= 0:
        return base
    t = (epoch - warmup) / (total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


# ---------------------------------------------------------------------------
def run_mae(manifest_filter: list[str] | None = None, **kwargs) -> dict:
    """Run MAE pretraining on the union of selected sources.

    manifest_filter: list of source names to include. None → all single-channel sources.
    """
    cfg = MAEConfig(**kwargs)
    cfg.save_dir = Path(cfg.save_dir)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # re-resolve DEFAULT_OUT at call time (callers may monkey-patch
    # bonescan.data.DEFAULT_OUT for remote use)
    import bonescan.data as _bd
    df = load_manifest(_bd.DEFAULT_OUT)
    if manifest_filter is None:
        # include everything that has an image_path saved as .npy in standardized/images
        # exclude bs80k_crop (paths point to jpg in temp/, smaller resolution — optional)
        df = df[df.source.isin([
            "bs80k_wb", "new_mb", "new_mp", "new_bb", "new_bp",
            "old_breast", "old_prostate",
            "meta_breast_meta", "meta_breast_non-meta",
            "meta_prostate_meta", "meta_prostate_non-meta",
        ])]
    else:
        df = df[df.source.isin(manifest_filter)]
    print(f"MAE pretraining on {len(df)} images from {df.source.nunique()} sources")

    ds = SingleViewMAEDataset(df, out_dir=_bd.DEFAULT_OUT)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=(cfg.num_workers > 0))

    # Build model with reconstruction decoder (single-view)
    model = AnchorBSModel(
        BackboneConfig(),
        num_anatomy_classes=11,
        num_lesion_classes=2,
        enable_hbs=False, enable_cvfm=False,
        enable_recon=True,
    )
    device = torch.device(cfg.device)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"epoch": [], "lr": [], "loss": [], "epoch_seconds": []}
    for epoch in range(cfg.epochs):
        lr = cosine_warmup(epoch, cfg.epochs, cfg.warmup_epochs, cfg.lr)
        for g in optim.param_groups:
            g["lr"] = lr
        model.train()
        total = n = 0.0
        t0 = time.time()
        for x, _src in loader:
            x = x.to(device, non_blocking=True)
            mask = random_patch_mask(x.shape[0], x.shape[2], x.shape[3],
                                     patch=cfg.patch_size, mask_ratio=cfg.mask_ratio,
                                     device=device)
            x_masked = x * mask  # set masked pixels to 0
            # Use the model in "ant view only" mode for SSL: pass x_masked as ANT, zeros as POST
            x_post = torch.zeros_like(x_masked)
            out = model(x_masked, x_post, anatomy_gt_ant=None, anatomy_gt_post=None,
                        curriculum_alpha=0.0)
            recon = out["recon_ant"]   # (B, 1, H, W)
            # MSE only on masked pixels (1 - mask)
            unmasked = (1.0 - mask)
            diff = (recon - x).pow(2) * unmasked
            denom = unmasked.sum().clamp(min=1.0)
            loss = diff.sum() / denom
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total += float(loss.detach().cpu()); n += 1
        history["epoch"].append(epoch); history["lr"].append(lr)
        history["loss"].append(total / max(n, 1))
        history["epoch_seconds"].append(time.time() - t0)
        print(f"[MAE ep {epoch:3d}] lr={lr:.5f} loss={total/max(n,1):.5f} "
              f"({history['epoch_seconds'][-1]:.1f}s)", flush=True)

        # checkpoint every 10 epochs and at end
        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            torch.save({"model": model.state_dict(), "epoch": epoch, "history": history},
                       cfg.save_dir / f"ckpt_ep{epoch:03d}.pt")

    torch.save({"model": model.state_dict(), "epoch": cfg.epochs - 1,
                "history": history}, cfg.save_dir / "final.pt")
    (cfg.save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"save_dir": str(cfg.save_dir), "history": history}
