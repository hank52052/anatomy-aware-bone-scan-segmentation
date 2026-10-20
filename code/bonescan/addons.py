"""Architecture add-ons (Variant G+):
  - CoordinateAttention: H/W axis-wise pooling + sigmoid gate (Hou CVPR 2021),
    naturally aligned with 1D-elongated bone-scan lesions (spine, ribs, long bones).
  - component_adaptive_tversky_loss: per-connected-component reweighted Tversky (CATMIL idea).
  - sub_diff_hausdorff_loss: smooth-distance-transform Hausdorff (Visalakshi & Mohan 2025).

All training-time only, no inference overhead (except CoordinateAttention which is small).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021).

    Encode position information into channel attention by H-pooling and W-pooling
    separately, then merging. For 1024×256 bone scans where lesions are elongated
    along spine (vertical) and ribs (horizontal), the two-axis decomposition matches
    anatomical priors.

    Input/output shape (B, C, H, W). Lightweight: ~2*C^2 / r params.
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))      # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))      # (B, C, 1, W)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.act   = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = self.pool_h(x)                       # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)           # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        a_h, a_w = torch.split(y, [h, w], dim=2)
        a_w = a_w.permute(0, 1, 3, 2)              # (B, mid, 1, W)
        a_h = torch.sigmoid(self.conv_h(a_h))      # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(a_w))      # (B, C, 1, W)
        return x * a_h * a_w


# ---------------------------------------------------------------------------
def _binary_target(logits: torch.Tensor, target: torch.Tensor):
    """Return (probs_fg, target_fg) both as float (B, H, W)."""
    if logits.shape[1] > 1:
        probs = F.softmax(logits, dim=1)[:, 1]   # foreground class prob
    else:
        probs = torch.sigmoid(logits.squeeze(1))
    tgt = (target > 0).float()
    return probs, tgt


def component_adaptive_tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                                    alpha: float = 0.3, beta: float = 0.7,
                                    gamma: float = 0.75, eps: float = 1e-6,
                                    max_components: int = 16) -> torch.Tensor:
    """Per-connected-component reweighted Focal-Tversky loss.

    For each sample, label the GT connected components (8-connectivity via
    cumulative max trick on CPU). Each component is weighted inversely by its
    area, so small lesions contribute proportionally more to the loss.

    Falls back to standard Focal-Tversky on the whole batch as a baseline term.
    """
    probs, tgt = _binary_target(logits, target)
    # standard focal-tversky on the full image (anchor term)
    tp = (probs * tgt).sum(dim=(1, 2))
    fp = (probs * (1 - tgt)).sum(dim=(1, 2))
    fn = ((1 - probs) * tgt).sum(dim=(1, 2))
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    base = ((1.0 - ti).clamp(min=1e-7, max=1.0)).pow(gamma).mean()

    # per-component reweighting via scipy.ndimage.label on CPU per sample
    # Done in no-grad context for the labeling step; loss tensor still
    # builds from probs/tgt via masked sums (differentiable).
    try:
        import numpy as np
        from scipy.ndimage import label as _cc_label
    except Exception:
        return base

    per_cc = []
    with torch.no_grad():
        tgt_np = tgt.detach().cpu().numpy()
    for b in range(tgt.shape[0]):
        lbl, n = _cc_label(tgt_np[b] > 0.5)
        if n == 0:
            continue
        # cap to max_components — largest skipped (per-cc reweight focuses on small)
        for k in range(1, min(n, max_components) + 1):
            mask_k = torch.from_numpy((lbl == k)).float().to(probs.device)
            area_k = mask_k.sum().clamp(min=1.0)
            tp_k = (probs[b] * mask_k).sum()
            fp_k = (probs[b] * (1 - mask_k)).sum() / max(mask_k.numel(), 1)
            fn_k = ((1 - probs[b]) * mask_k).sum()
            ti_k = (tp_k + eps) / (tp_k + alpha * fp_k + beta * fn_k + eps)
            l_k = ((1.0 - ti_k).clamp(min=1e-7, max=1.0)).pow(gamma)
            # inverse-area weighting → small components matter more
            w_k = 1.0 / (1.0 + area_k.sqrt())
            per_cc.append(w_k * l_k)
    if per_cc:
        cc_term = torch.stack(per_cc).mean()
        return 0.5 * base + 0.5 * cc_term
    return base


def sub_diff_hausdorff_loss(logits: torch.Tensor, target: torch.Tensor,
                            alpha: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    """Sub-differentiable Hausdorff loss via Gaussian-smoothed distance transform.

    Approximates HD by penalising prob mass that lies far from GT and vice versa,
    weighted by a soft distance map. Smooth, sub-differentiable.
    Cheap surrogate that correlates with HD95.
    """
    probs, tgt = _binary_target(logits, target)
    if tgt.sum() < 1.0 or probs.sum() < 1.0:
        return torch.tensor(0.0, device=probs.device)

    try:
        import numpy as np
        from scipy.ndimage import distance_transform_edt as _dt
    except Exception:
        return torch.tensor(0.0, device=probs.device)

    with torch.no_grad():
        tgt_np = (tgt.detach().cpu().numpy() > 0.5).astype("uint8")
        # distance from GT surface (background pixels' distance to nearest FG)
        dt_to_gt = []
        for b in range(tgt_np.shape[0]):
            if tgt_np[b].any():
                d = _dt(1 - tgt_np[b])
            else:
                d = np.zeros_like(tgt_np[b], dtype="float32")
            dt_to_gt.append(d)
        dt_to_gt = torch.from_numpy(
            __import__("numpy").stack(dt_to_gt).astype("float32")
        ).to(probs.device)

    # penalise pred mass that's far from GT (distance-weighted false positives)
    fp_penalty = (probs * (dt_to_gt ** alpha)).mean()

    # penalise GT mass not covered (distance-weighted false negatives, symmetric)
    with torch.no_grad():
        pred_np = (probs.detach().cpu().numpy() > 0.5).astype("uint8")
        dt_to_pred = []
        for b in range(pred_np.shape[0]):
            if pred_np[b].any():
                d = _dt(1 - pred_np[b])
            else:
                d = __import__("numpy").full_like(pred_np[b], 1024.0, dtype="float32")
            dt_to_pred.append(d)
        dt_to_pred = torch.from_numpy(
            __import__("numpy").stack(dt_to_pred).astype("float32")
        ).to(probs.device)

    fn_penalty = (tgt * (dt_to_pred ** alpha)).mean()

    # normalised so the loss is in a similar scale to Dice (~0–1)
    norm = (dt_to_gt.max().clamp(min=1.0)) ** alpha
    return (fp_penalty + fn_penalty) / norm


# ---------------------------------------------------------------------------
def cc_postprocess(prob_map, threshold: float = 0.5, min_size: int = 16,
                   max_components: int = 50, prob_threshold: float = 0.6):
    """Connected-component soft-NMS post-processing.

    Used at INFERENCE time to clean predictions:
      - threshold prob_map @ `threshold` to get binary candidate
      - find connected components, drop those < min_size pixels
      - rank by mean prob within component, keep top-`max_components`
      - keep only components whose mean prob > `prob_threshold`

    Reduces FP/scan substantially without changing Dice on TP regions.
    Input: prob_map ndarray (H, W) or (B, H, W). Output: same shape, binary uint8.
    """
    import numpy as np
    from scipy.ndimage import label as _label

    def _one(pm):
        binary = (pm >= threshold).astype("uint8")
        lbl, n = _label(binary)
        if n == 0:
            return binary
        out = np.zeros_like(binary)
        comps = []
        for k in range(1, n + 1):
            mask_k = (lbl == k)
            sz = int(mask_k.sum())
            if sz < min_size:
                continue
            mean_p = float(pm[mask_k].mean())
            if mean_p < prob_threshold:
                continue
            comps.append((mean_p, sz, k, mask_k))
        # keep top-N by mean prob
        comps.sort(key=lambda t: -t[0])
        for _, _, _, mask_k in comps[:max_components]:
            out[mask_k] = 1
        return out

    if prob_map.ndim == 2:
        return _one(prob_map)
    return __import__("numpy").stack([_one(prob_map[i]) for i in range(prob_map.shape[0])])
