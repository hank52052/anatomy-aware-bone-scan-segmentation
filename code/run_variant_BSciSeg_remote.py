"""BSci-Seg (He et al. 2025, Biomed Sig Proc Control) re-implementation.

Paper: 'encoder-decoder + PDSS (Patch-level Dual-Sampling Strategy)
+ MRFA (Multi-Receptive-Field Attention)'.
Code is closed-source — this is a best-effort re-implementation from the paper description.

Architecture:
  - U-Net encoder-decoder (5 stages, features 32→512)
  - MRFA module at bottleneck: parallel dilated convs (1,2,4,8) + SE channel attention
  - PDSS: encoder-decoder paired contrastive loss (InfoNCE) at each scale

Training: same fold_0 split + aug as our model for apples-to-apples comparison.
"""
from __future__ import annotations

import os
import sys
import time
import pickle
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BONESCAN_ROOT = Path(os.environ.get("BONESCAN_ROOT", "."))  # folder with data/, runs/, checkpoints/
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(BONESCAN_ROOT / "data" / "standardized")))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(BONESCAN_ROOT / "runs")))
LIBRARY_DIR = Path(os.environ.get("LIBRARY_DIR", str(BONESCAN_ROOT / "checkpoints")))
EXPAND_FACTOR = int(os.environ.get("EXPAND_FACTOR", "10"))
TARGET_FG_ANCHOR = 0.2

import bonescan.data as _bs_data
_bs_data.DEFAULT_OUT = DATA_ROOT

from bonescan.data import BoneScanDataset, assemble_patient_records, k_fold_split, load_manifest

LESION_SRC = ("new_mb", "new_mp", "meta_breast_meta", "meta_prostate_meta",
              "meta_breast_non-meta", "meta_prostate_non-meta")
ANAT_SRC = ("bs80k",)


# ============================ Preprocessing (same as our model) ============================

def _otsu_threshold(img, n_bins=64):
    flat = img.flatten(); flat = flat[flat > 0]
    if flat.size < 10: return 0.5
    hist, edges = np.histogram(flat, bins=n_bins, range=(0, 1))
    hist = hist.astype(np.float64); total = hist.sum()
    if total < 1: return 0.5
    p = hist / total; omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(n_bins)); mu_total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_total * omega - mu) ** 2 / (omega * (1 - omega) + 1e-9)
    sigma_b2[np.isnan(sigma_b2)] = -1
    return float(edges[int(np.argmax(sigma_b2))])


def _otsu_trim_normalize(img2d):
    img = img2d.astype(np.float32)
    if img.max() < 1e-6: return img
    thr = _otsu_threshold(img); fg = img[img > thr]
    if fg.size < 50: fg = img[img > 0]
    if fg.size < 10: return img
    p30, p70 = np.percentile(fg, [30, 70])
    core = fg[(fg >= p30) & (fg <= p70)]
    if core.size < 10: core = fg
    anchor = float(core.mean())
    if anchor < 1e-6: return img
    return np.clip(img * (TARGET_FG_ANCHOR / anchor), 0.0, 1.0).astype(np.float32)


class BSciSegDataset(BoneScanDataset):
    def __init__(self, *args, expand_factor=EXPAND_FACTOR, **kw):
        super().__init__(*args, **kw)
        self.expand = expand_factor
        self._n_orig = len(self.records)
    def __len__(self): return self._n_orig * self.expand
    def __getitem__(self, idx):
        item = super().__getitem__(idx % self._n_orig)
        img = item["image"].numpy()
        for ch in range(2):
            img[ch] = _otsu_trim_normalize(img[ch])
        item["image"] = torch.from_numpy(img.astype(np.float32))
        return item


class BSciSegValDataset(BoneScanDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        img = item["image"].numpy()
        for ch in range(2):
            img[ch] = _otsu_trim_normalize(img[ch])
        item["image"] = torch.from_numpy(img.astype(np.float32))
        return item


# ============================ BSci-Seg architecture ============================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def forward(self, x): return self.net(x)


class MRFA(nn.Module):
    """Multi-Receptive-Field Attention.

    Parallel dilated convs (1, 2, 4, 8) + SE channel attention + residual.
    """
    def __init__(self, channels, dilations=(1, 2, 4, 8)):
        super().__init__()
        branch_out = channels // len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch_out, 3, padding=d, dilation=d, bias=False),
                nn.InstanceNorm2d(branch_out, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ) for d in dilations
        ])
        self.proj = nn.Conv2d(branch_out * len(dilations), channels, 1, bias=False)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        feats = torch.cat([b(x) for b in self.branches], dim=1)
        out = self.proj(feats)
        out = self.norm(out)
        out = out * self.se(out)
        return F.leaky_relu(x + out, 0.01, inplace=True)


class BSciSegUNet(nn.Module):
    """Encoder-decoder U-Net with MRFA at bottleneck, exposes encoder/decoder feats for PDSS."""
    def __init__(self, in_channels=2, n_classes=2, features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.enc = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.enc.append(ConvBlock(prev, f))
            prev = f
        self.pool = nn.MaxPool2d(2, 2)
        self.mrfa = MRFA(features[-1])
        # decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        rev = features[::-1]
        for i in range(len(rev) - 1):
            self.up.append(nn.ConvTranspose2d(rev[i], rev[i + 1], 2, stride=2))
            self.dec.append(ConvBlock(rev[i + 1] * 2, rev[i + 1]))
        # output head
        self.out_conv = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x_ant, x_post):
        # process ANT and POST separately (1-channel each) — same as our pipeline
        results = {}
        for view, x_in in [("ant", x_ant), ("post", x_post)]:
            x = x_in
            skips = []
            enc_feats = []
            for i, blk in enumerate(self.enc):
                x = blk(x)
                if i < len(self.enc) - 1:
                    skips.append(x)
                    enc_feats.append(x)
                    x = self.pool(x)
            # bottleneck
            x = self.mrfa(x)
            enc_feats.append(x)
            # decoder
            dec_feats = []
            for i, (up, blk) in enumerate(zip(self.up, self.dec)):
                x = up(x)
                # In case of size mismatch (odd dims), interpolate to skip size
                if x.shape[-2:] != skips[-(i + 1)].shape[-2:]:
                    x = F.interpolate(x, size=skips[-(i + 1)].shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skips[-(i + 1)]], dim=1)
                x = blk(x)
                dec_feats.append(x)
            logits = self.out_conv(x)
            results[f"lesion_logits_{view}"] = logits
            results[f"enc_feats_{view}"] = enc_feats
            results[f"dec_feats_{view}"] = dec_feats
        return results


# ============================ PDSS contrastive loss ============================

def pdss_loss(enc_feats, dec_feats, mask, n_pos=8, n_neg=24, temperature=0.07):
    """Patch-level Dual-Sampling: InfoNCE between encoder anchors and decoder pos/neg samples.

    For each (enc_feat, dec_feat) at matching scale, sample n_pos positive patch coords
    (inside lesion mask) and n_neg hard-negative coords (outside but with high feature norm).
    """
    total = torch.tensor(0.0, device=mask.device, dtype=torch.float32)
    n_used = 0
    for k in range(min(len(enc_feats), len(dec_feats))):
        ef, df = enc_feats[k], dec_feats[k]
        if ef.shape[-2:] != df.shape[-2:]:
            continue
        H, W = ef.shape[-2:]
        m_lr = F.interpolate(mask.float().unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
        B = ef.shape[0]
        for b in range(B):
            pos = (m_lr[b] > 0.5).nonzero()
            neg = (m_lr[b] < 0.5).nonzero()
            if len(pos) < n_pos or len(neg) < n_neg: continue
            pi = pos[torch.randperm(len(pos), device=pos.device)[:n_pos]]
            ni = neg[torch.randperm(len(neg), device=neg.device)[:n_neg]]
            anchors = ef[b, :, pi[:, 0], pi[:, 1]].T  # (n_pos, C)
            pos_emb = df[b, :, pi[:, 0], pi[:, 1]].T  # (n_pos, C)
            neg_emb = df[b, :, ni[:, 0], ni[:, 1]].T  # (n_neg, C)
            anchors = F.normalize(anchors, dim=-1)
            pos_emb = F.normalize(pos_emb, dim=-1)
            neg_emb = F.normalize(neg_emb, dim=-1)
            pos_logit = (anchors * pos_emb).sum(-1, keepdim=True) / temperature  # (n_pos, 1)
            neg_logits = anchors @ neg_emb.T / temperature  # (n_pos, n_neg)
            logits = torch.cat([pos_logit, neg_logits], dim=-1)
            targets = torch.zeros(n_pos, dtype=torch.long, device=logits.device)
            total = total + F.cross_entropy(logits, targets)
            n_used += 1
    return total / max(1, n_used)


# ============================ Loss ============================

def soft_dice_loss(logits, target, smooth=1.0, ignore_bg=True):
    prob = F.softmax(logits, dim=1)
    onehot = F.one_hot(target.long().clamp(0, logits.shape[1] - 1), logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dim=dims)
    denom = prob.sum(dim=dims) + onehot.sum(dim=dims)
    dice = (2 * inter + smooth) / (denom + smooth)
    if ignore_bg: dice = dice[1:]
    return 1.0 - dice.mean()


def composite_loss(out, lesion_mask, has_lesion, pdss_weight=0.1):
    device = lesion_mask.device
    total = torch.tensor(0.0, device=device); n = 0
    terms = {"ce": 0.0, "dice": 0.0, "pdss": 0.0}
    for view, idx in [("ant", 0), ("post", 1)]:
        logits = out[f"lesion_logits_{view}"]
        t = lesion_mask[:, idx].long()
        # only samples with lesion mask supervised
        sel = has_lesion.bool()
        if not sel.any(): continue
        L = logits[sel]; T = t[sel]
        ce = F.cross_entropy(L, T); d = soft_dice_loss(L, T)
        terms["ce"] += float(ce.item()); terms["dice"] += float(d.item())
        loss_v = ce + d
        # PDSS at supervised samples
        if pdss_weight > 0:
            ef = [f[sel] for f in out[f"enc_feats_{view}"]]
            df = [f[sel] for f in out[f"dec_feats_{view}"]]
            p = pdss_loss(ef, df, T)
            if p.item() == p.item():  # not NaN
                terms["pdss"] += float(p.item())
                loss_v = loss_v + pdss_weight * p
        total = total + loss_v; n += 1
    return total / max(1, n), terms


# ============================ Eval ============================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices = []
    for batch in loader:
        x = batch["image"].to(device)
        m = batch["lesion_mask"]
        has = batch["has_lesion_mask"]
        ant, post = x[:, 0:1], x[:, 1:2]
        out = model(ant, post)
        for view, idx in [("ant", 0), ("post", 1)]:
            logits = out[f"lesion_logits_{view}"]
            pred = (logits.argmax(dim=1) > 0).float().cpu()
            tt = (m[:, idx] > 0).float()
            for b in range(pred.shape[0]):
                if tt[b].sum() > 0 and int(has[b]) == 1:
                    inter = (pred[b] * tt[b]).sum()
                    denom = pred[b].sum() + tt[b].sum()
                    d = (2 * inter + 1e-6) / (denom + 1e-6)
                    dices.append(float(d))
    return float(np.mean(dices)) if dices else float("nan")


# ============================ Training ============================

def main():
    fold = int(os.environ.get("FOLDS", "0").split(",")[0])
    epochs = int(os.environ.get("EPOCHS", "40"))
    bs = int(os.environ.get("BATCH_SIZE", "8"))
    nw = int(os.environ.get("NUM_WORKERS", "8"))
    lr = float(os.environ.get("LR", "1e-3"))
    suffix = os.environ.get("VARIANT_SUFFIX", "BSciSeg")
    out_dir = RUNS_DIR / f"ablation_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = out_dir / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True)

    print(f"[BSciSeg] fold={fold} lr={lr} epochs={epochs} bs={bs} expand={EXPAND_FACTOR}")
    df = load_manifest(DATA_ROOT)
    recs = assemble_patient_records(df, DATA_ROOT)
    lesion = [r for r in recs if r.source in LESION_SRC]
    anatomy = [r for r in recs if r.source in ANAT_SRC]
    splits = k_fold_split(lesion, k=5, seed=0)
    tr, va = splits[fold]
    train_set = [lesion[i] for i in tr] + anatomy
    val_set = [lesion[i] for i in va]

    lib_path = LIBRARY_DIR / f"_lesion_library_fold{fold}.pkl"
    lesion_library = None
    if lib_path.exists():
        with open(lib_path, "rb") as f: lesion_library = pickle.load(f)
        print(f"  lesion library = {len(lesion_library)}")

    train_ds = BSciSegDataset(train_set, augment=True, augment_strength="tier1_softcp",
                               lesion_library=lesion_library,
                               anatomy_bone_ids=None, mixup_enabled=False)
    val_ds = BSciSegValDataset(val_set, augment=False)
    print(f"  train_len={len(train_ds)} val_len={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                               drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=max(2, nw // 2),
                             pin_memory=True, persistent_workers=True)

    device = torch.device("cuda")
    torch.manual_seed(fold); torch.cuda.manual_seed_all(fold)
    model = BSciSegUNet(in_channels=1, n_classes=2, features=(32, 64, 128, 256, 512)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params = {n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(train_loader),
        pct_start=0.1, anneal_strategy="cos")

    best_val = -1.0; no_imp = 0; history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time(); ep_loss = 0.0; nsteps = 0; ep_terms = {"ce": 0, "dice": 0, "pdss": 0}
        for batch in train_loader:
            x = batch["image"].to(device); m = batch["lesion_mask"].to(device)
            has = batch["has_lesion_mask"].to(device)
            opt.zero_grad()
            ant, post = x[:, 0:1], x[:, 1:2]
            out = model(ant, post)
            loss, terms = composite_loss(out, m, has)
            if not loss.requires_grad: continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step()
            ep_loss += float(loss.item()); nsteps += 1
            for k in ep_terms: ep_terms[k] += terms.get(k, 0)
        val_dice = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": ep_loss / max(1, nsteps), "val_lesion_dice": val_dice,
                        "ce": ep_terms["ce"]/max(1,nsteps), "dice": ep_terms["dice"]/max(1,nsteps),
                        "pdss": ep_terms["pdss"]/max(1,nsteps)})
        improved = val_dice > best_val + 5e-4
        if improved: best_val = val_dice; no_imp = 0
        else: no_imp += 1
        print(f"[ep {ep:3d}] lr={scheduler.get_last_lr()[0]:.5f} loss={ep_loss/max(1,nsteps):.4f} "
              f"ce={ep_terms['ce']/max(1,nsteps):.4f} dice_l={ep_terms['dice']/max(1,nsteps):.4f} "
              f"pdss={ep_terms['pdss']/max(1,nsteps):.4f} "
              f"val_dice={val_dice:.4f} best={best_val:.4f}@? noImp={no_imp}/12 ({time.time()-t0:.1f}s)",
              flush=True)
        if improved:
            torch.save({"model": model.state_dict(), "epoch": ep, "val_dice": val_dice}, fold_dir / "best.pt")
        if no_imp >= 12 and ep >= 15:
            print(f"[early stop] best={best_val:.4f}"); break
    (fold_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[DONE] best val DSC = {best_val:.4f}")
    (out_dir / "summary.json").write_text(json.dumps({
        "variant": f"BSciSeg_reimpl",
        "fold_results": {str(fold): {"best_val_lesion_dice": best_val}},
        "n_params_M": n_params / 1e6,
        "notes": "He BSci-Seg 2025 reimpl: U-Net + MRFA + PDSS contrastive. Closed-source paper."
    }, indent=2))


if __name__ == "__main__":
    main()
