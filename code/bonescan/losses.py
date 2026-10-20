"""Loss functions for bone-scan multi-task segmentation.

Conventions: predictions are RAW logits (no softmax), targets are integer label maps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 0.3, beta: float = 0.7,
                       gamma_t: float = 0.75, smooth: float = 1.0,
                       fg_class: int = 1) -> torch.Tensor:
    """Focal-Tversky loss for class imbalance (Abraham & Khan 2018).

    Tversky index = TP / (TP + alpha*FP + beta*FN). Higher beta → penalize FN more.
    Focal-Tversky: (1 - TI)^gamma_t to focus on hard examples.

    Numerical safety:
      - clamp (1 - tversky) to [eps, 1.0] before fractional power (negatives → NaN)
      - cast logits to fp32 (AMP safety)
    """
    prob = F.softmax(logits.float(), dim=1)
    pred_fg = prob[:, fg_class].clamp(0.0, 1.0)
    target_fg = (target == fg_class).float()
    tp = (pred_fg * target_fg).sum(dim=(1, 2))
    fp = (pred_fg * (1.0 - target_fg)).sum(dim=(1, 2))
    fn = ((1.0 - pred_fg) * target_fg).sum(dim=(1, 2))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    tversky = tversky.clamp(0.0, 1.0)
    one_minus_tv = (1.0 - tversky).clamp(min=1e-7, max=1.0)
    loss = one_minus_tv.pow(gamma_t)
    return loss.mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   smooth: float = 1.0, ignore_bg: bool = False) -> torch.Tensor:
    """Multi-class Dice loss. logits (B, C, H, W), target (B, H, W) integer."""
    n_class = logits.shape[1]
    prob = F.softmax(logits, dim=1)
    onehot = F.one_hot(target.long().clamp(0, n_class - 1), n_class).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dim=dims)
    denom = prob.sum(dim=dims) + onehot.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    if ignore_bg:
        dice = dice[1:]
    return 1.0 - dice.mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0, alpha: float | None = None) -> torch.Tensor:
    """Multi-class focal loss — numerically stable AMP-safe form."""
    # Use cross_entropy for the log-prob path (handles fp16 stably via internal log-sum-exp)
    target_l = target.long().clamp(0, logits.shape[1] - 1)
    ce = F.cross_entropy(logits.float(), target_l, reduction="none")  # cast to fp32 for safety
    pt = (-ce).exp().clamp(min=1e-8, max=1.0)
    focal_w = (1.0 - pt).pow(gamma)
    loss = focal_w * ce
    if alpha is not None:
        loss = loss * alpha
    return loss.mean()


def boundary_loss(logits: torch.Tensor, target: torch.Tensor,
                  smooth: float = 1.0) -> torch.Tensor:
    """Simple boundary loss using Sobel edges on (binary) target & pred FG prob."""
    if logits.shape[1] < 2:
        return torch.tensor(0.0, device=logits.device)
    pred_fg = F.softmax(logits, dim=1)[:, 1:].sum(dim=1, keepdim=True)  # (B, 1, H, W)
    target_fg = (target > 0).float().unsqueeze(1)
    sobel_x = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]],
                           device=logits.device)
    sobel_y = sobel_x.transpose(-1, -2)
    def edges(x):
        ex = F.conv2d(x, sobel_x, padding=1)
        ey = F.conv2d(x, sobel_y, padding=1)
        return (ex.abs() + ey.abs())
    p_e = edges(pred_fg)
    t_e = edges(target_fg)
    diff = (p_e - t_e).abs()
    # weight by edge magnitude on target
    weight = t_e + smooth
    return (diff * weight).mean() / (weight.mean() + 1e-6)


def anatomy_constraint_loss(lesion_logits: torch.Tensor, anatomy_logits: torch.Tensor) -> torch.Tensor:
    """Penalty for lesion-foreground probability outside the bone (anatomy non-background)."""
    if lesion_logits.shape[1] < 2 or anatomy_logits.shape[1] < 2:
        return torch.tensor(0.0, device=lesion_logits.device)
    p_lesion_fg = F.softmax(lesion_logits, dim=1)[:, 1]                    # (B, H, W)
    p_anat = F.softmax(anatomy_logits, dim=1)
    p_anat_bg = p_anat[:, 0]                                                # (B, H, W)
    # cost = lesion-FG-prob * anatomy-BG-prob → push lesion away from bg
    return (p_lesion_fg * p_anat_bg).mean()


# ---------------------------------------------------------------------------
class CompositeLoss(nn.Module):
    """Weighted composite for lesion + anatomy + optional constraint.

    Inputs (dict from AnchorBSModel.forward):
        lesion_logits_ds_*, anatomy_logits_ds_*
    Targets:
        lesion_mask  (B, 2, H, W)  channel-0 = ANT, channel-1 = POST
        anatomy_mask (B, 2, H, W)
    """

    def __init__(self,
                 w_lesion_dice: float = 1.0,
                 w_lesion_focal: float = 0.5,
                 w_lesion_boundary: float = 0.1,
                 w_lesion_tversky: float = 0.0,
                 w_anatomy_dice: float = 0.5,
                 w_anatomy_ce: float = 0.5,
                 w_anatomy_constraint: float = 0.1,
                 deep_supervision_weights: tuple = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0625),
                 ignore_lesion_bg_dice: bool = False,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7,
                 tversky_gamma: float = 0.75,
                 # NEW (Variant G+):
                 w_lesion_catmil: float = 0.0,
                 w_lesion_sdhl: float = 0.0):
        super().__init__()
        self.w = dict(
            les_dice=w_lesion_dice, les_focal=w_lesion_focal, les_bnd=w_lesion_boundary,
            les_tversky=w_lesion_tversky,
            les_catmil=w_lesion_catmil, les_sdhl=w_lesion_sdhl,
            ana_dice=w_anatomy_dice, ana_ce=w_anatomy_ce, ana_cons=w_anatomy_constraint,
        )
        self.ds_w = list(deep_supervision_weights)
        self.ignore_lesion_bg_dice = ignore_lesion_bg_dice
        self.tversky_kw = dict(alpha=tversky_alpha, beta=tversky_beta, gamma_t=tversky_gamma)

    def _ds_combo(self, ds_list, target, kind, sample_mask: torch.Tensor | None = None):
        """Sum loss over deep-supervision outputs, optionally filtered to samples in sample_mask."""
        total = 0.0
        for k, logits in enumerate(reversed(ds_list)):
            w = self.ds_w[k] if k < len(self.ds_w) else 0.0
            if w == 0.0:
                continue
            if sample_mask is not None:
                if not sample_mask.any():
                    continue
                logits = logits[sample_mask]
                t = target[sample_mask]
            else:
                t = target
            t = F.interpolate(t.unsqueeze(1).float(), size=logits.shape[-2:],
                              mode="nearest").squeeze(1).long()
            if kind == "dice":
                total = total + w * soft_dice_loss(logits, t, ignore_bg=self.ignore_lesion_bg_dice)
            elif kind == "focal":
                total = total + w * focal_loss(logits, t)
            elif kind == "ce":
                total = total + w * F.cross_entropy(logits, t)
            elif kind == "tversky":
                total = total + w * focal_tversky_loss(logits, t, **self.tversky_kw)
        return total

    def forward(self, out: dict, lesion_mask: torch.Tensor, anatomy_mask: torch.Tensor,
                has_lesion_mask: torch.Tensor | None = None,
                has_anatomy_mask: torch.Tensor | None = None,
                use_anatomy: bool = True, use_lesion: bool = True) -> dict:
        """has_lesion_mask, has_anatomy_mask: (B,) int/bool tensors; if None, assume all True."""
        device = next(iter(out.values())).device
        zero = torch.tensor(0.0, device=device)
        terms = dict(les_dice=zero, les_focal=zero, les_bnd=zero,
                     ana_dice=zero, ana_ce=zero, ana_cons=zero)

        les_mask = None if has_lesion_mask is None else has_lesion_mask.bool().to(device)
        ana_mask = None if has_anatomy_mask is None else has_anatomy_mask.bool().to(device)

        if use_lesion:
            terms["les_tversky"] = zero
            terms["les_catmil"] = zero
            terms["les_sdhl"] = zero
            for view, idx in [("ant", 0), ("post", 1)]:
                ds = out[f"lesion_logits_ds_{view}"]
                t = lesion_mask[:, idx]
                if self.w["les_dice"] > 0:
                    terms["les_dice"] = terms["les_dice"] + self.w["les_dice"] * self._ds_combo(ds, t, "dice", sample_mask=les_mask)
                if self.w["les_focal"] > 0:
                    terms["les_focal"] = terms["les_focal"] + self.w["les_focal"] * self._ds_combo(ds, t, "focal", sample_mask=les_mask)
                if self.w.get("les_tversky", 0.0) > 0:
                    terms["les_tversky"] = terms["les_tversky"] + self.w["les_tversky"] * self._ds_combo(ds, t, "tversky", sample_mask=les_mask)
                if self.w["les_bnd"] > 0 and (les_mask is None or les_mask.any()):
                    sel = slice(None) if les_mask is None else les_mask
                    terms["les_bnd"] = terms["les_bnd"] + self.w["les_bnd"] * boundary_loss(ds[-1][sel] if les_mask is not None else ds[-1], t[sel] if les_mask is not None else t)
                # NEW: CATMIL + SDHL on the highest-res prediction only (avoid DS overhead)
                if self.w.get("les_catmil", 0.0) > 0 and (les_mask is None or les_mask.any()):
                    sel = slice(None) if les_mask is None else les_mask
                    logits_hi = ds[-1][sel] if les_mask is not None else ds[-1]
                    t_hi = t[sel] if les_mask is not None else t
                    if logits_hi.shape[0] > 0:
                        from .addons import component_adaptive_tversky_loss
                        terms["les_catmil"] = terms["les_catmil"] + self.w["les_catmil"] * component_adaptive_tversky_loss(
                            logits_hi, t_hi,
                            alpha=self.tversky_kw["alpha"],
                            beta=self.tversky_kw["beta"],
                            gamma=self.tversky_kw["gamma_t"])
                if self.w.get("les_sdhl", 0.0) > 0 and (les_mask is None or les_mask.any()):
                    sel = slice(None) if les_mask is None else les_mask
                    logits_hi = ds[-1][sel] if les_mask is not None else ds[-1]
                    t_hi = t[sel] if les_mask is not None else t
                    if logits_hi.shape[0] > 0:
                        from .addons import sub_diff_hausdorff_loss
                        terms["les_sdhl"] = terms["les_sdhl"] + self.w["les_sdhl"] * sub_diff_hausdorff_loss(
                            logits_hi, t_hi)

        if use_anatomy:
            for view, idx in [("ant", 0), ("post", 1)]:
                ds = out[f"anatomy_logits_ds_{view}"]
                t = anatomy_mask[:, idx]
                if self.w["ana_dice"] > 0:
                    terms["ana_dice"] = terms["ana_dice"] + self.w["ana_dice"] * self._ds_combo(ds, t, "dice", sample_mask=ana_mask)
                if self.w["ana_ce"] > 0:
                    terms["ana_ce"] = terms["ana_ce"] + self.w["ana_ce"] * self._ds_combo(ds, t, "ce", sample_mask=ana_mask)

        if self.w["ana_cons"] > 0 and use_anatomy and use_lesion:
            # Anatomy-constraint only meaningful when both tasks are supervised for the sample
            sel = None
            if les_mask is not None and ana_mask is not None:
                sel = les_mask & ana_mask
            elif les_mask is not None:
                sel = les_mask
            elif ana_mask is not None:
                sel = ana_mask
            for view in ("ant", "post"):
                l_logits = out[f"lesion_logits_{view}"]
                a_logits = out[f"anatomy_logits_{view}"]
                if sel is not None:
                    if not sel.any():
                        continue
                    l_logits = l_logits[sel]
                    a_logits = a_logits[sel]
                terms["ana_cons"] = terms["ana_cons"] + self.w["ana_cons"] * anatomy_constraint_loss(l_logits, a_logits)

        terms["total"] = sum(terms.values())
        return terms
