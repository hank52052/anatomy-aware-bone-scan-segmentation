"""Augmentation for bone-scan images. Both ANT and POST views share the same
spatial augmentation so they remain registered (siamese-friendly).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


STRENGTH_PRESETS = {
    "off":    {"flip_p": 0.0, "rot_deg": 0.0, "scale_pct": 0.0, "gamma_range": (1.0, 1.0), "brightness_pct": 0.0, "poisson_strength": 0.0, "contrast_pct": 0.0, "mult_noise_sigma": 0.0, "softcp_p": 0.0, "mixup_p": 0.0},
    "light":  {"flip_p": 0.0, "rot_deg": 5.0, "scale_pct": 0.05, "gamma_range": (0.9, 1.1), "brightness_pct": 0.05, "poisson_strength": 0.0, "contrast_pct": 0.0, "mult_noise_sigma": 0.0, "softcp_p": 0.0, "mixup_p": 0.0},
    "medium": {"flip_p": 0.0, "rot_deg": 10.0, "scale_pct": 0.10, "gamma_range": (0.8, 1.2), "brightness_pct": 0.10, "poisson_strength": 0.0, "contrast_pct": 0.0, "mult_noise_sigma": 0.0, "softcp_p": 0.0, "mixup_p": 0.0},
    "strong": {"flip_p": 0.5, "rot_deg": 15.0, "scale_pct": 0.15, "gamma_range": (0.7, 1.3), "brightness_pct": 0.15, "poisson_strength": 0.0, "contrast_pct": 0.0, "mult_noise_sigma": 0.0, "softcp_p": 0.0, "mixup_p": 0.0},
    # NEW (Variant H): intensity-only augmentation; NO rotation/scale (those destroy ~10 px lesions);
    # flip_p=0.5 OK because bone-scan anatomy is roughly L/R symmetric;
    # Poisson noise simulates gamma-camera photon statistics.
    "intensity_safe": {"flip_p": 0.5, "rot_deg": 0.0, "scale_pct": 0.0,
                       "gamma_range": (0.85, 1.15), "brightness_pct": 0.03,
                       "poisson_strength": 0.05,
                       "contrast_pct": 0.0, "mult_noise_sigma": 0.0,
                       "softcp_p": 0.0, "mixup_p": 0.0},
    # NEW 2026-05-21 09:55 (Variant J'): tier1 = paired Soft-CP only + gentle intensity.
    # Triggered by local J/fold_0 failure: Tier 2 (Soft-CP + MixUp + strong intensity) caused val osc
    # 0.33-0.52. Tier 1 isolates Soft-CP (user's key ANT-POST paired insight) with safe intensity.
    "tier1_softcp": {"flip_p": 0.5, "rot_deg": 0.0, "scale_pct": 0.0,
                     "gamma_range": (0.85, 1.15), "brightness_pct": 0.03,
                     "poisson_strength": 0.05,
                     "contrast_pct": 0.0,
                     "mult_noise_sigma": 0.0,
                     "softcp_p": 0.5,   # 50% chance to paste 1-3 lesions (gentler than tier2's 60%)
                     "mixup_p": 0.0},   # NO MixUp (was destructive on small dataset)
    # NEW 2026-05-21 (Variant J): tier2 = intensity_safe + stronger intensity + Soft-CP + MixUp.
    # Targets ~4-6x effective data via Soft-CP, with background MixUp for further regularization.
    "tier2_softcp": {"flip_p": 0.5, "rot_deg": 0.0, "scale_pct": 0.0,
                     "gamma_range": (0.70, 1.50), "brightness_pct": 0.10,
                     "poisson_strength": 0.08,
                     "contrast_pct": 0.15,
                     "mult_noise_sigma": 0.05,
                     "softcp_p": 0.6,   # 60% chance to paste 1-3 lesions per image
                     "mixup_p": 0.3},   # 30% chance to MixUp background with another sample
}
# Note: horizontal flip (flip_p) on whole-body bone scans flips L↔R.
# Off by default because POST is itself a horizontal mirror of ANT view.


def _spatial_transform(rng, h, w, rot_deg, scale_pct):
    angle = rng.uniform(-rot_deg, rot_deg)
    scale = 1.0 + rng.uniform(-scale_pct, scale_pct)
    return angle, scale


def _apply_spatial(arr2d, angle, scale, *, mode=Image.BICUBIC):
    """Rotate then scale around the image center, preserving size."""
    h, w = arr2d.shape
    is_float = arr2d.dtype in (np.float32, np.float16, np.float64)
    if is_float:
        # encode to uint16 to use PIL bicubic
        a = (arr2d.astype(np.float32) * 65535.0).clip(0, 65535).astype(np.uint16)
        img = Image.fromarray(a, mode="I;16")
    else:
        img = Image.fromarray(arr2d.astype(np.int32), mode="I")
        mode = Image.NEAREST
    img = img.rotate(angle, resample=mode if not is_float else Image.BICUBIC, fillcolor=0)
    # scale
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BICUBIC if is_float else Image.NEAREST)
    # crop / pad back to (h, w)
    out_img = Image.new(img.mode, (w, h), 0)
    ox = (w - nw) // 2
    oy = (h - nh) // 2
    out_img.paste(img, (ox, oy))
    a = np.asarray(out_img)
    if is_float:
        return (a.astype(np.float32) / 65535.0).astype(arr2d.dtype)
    return a.astype(arr2d.dtype)


def _gamma_brightness(arr2d, gamma_range, brightness_pct, rng):
    g = rng.uniform(*gamma_range)
    a = np.clip(arr2d.astype(np.float32), 0.0, 1.0)
    a = np.power(a, g)
    b = rng.uniform(-brightness_pct, brightness_pct)
    a = np.clip(a + b, 0.0, 1.0)
    return a.astype(arr2d.dtype)


def _contrast(arr2d, contrast_pct, rng):
    """Scale around midpoint 0.5, contrast_pct=0.15 means scale ±15%."""
    if contrast_pct <= 0:
        return arr2d
    s = 1.0 + rng.uniform(-contrast_pct, contrast_pct)
    a = np.clip(arr2d.astype(np.float32), 0.0, 1.0)
    a = 0.5 + (a - 0.5) * s
    return np.clip(a, 0.0, 1.0).astype(arr2d.dtype)


def _mult_noise(arr2d, sigma, rng):
    """Multiplicative Gaussian noise to simulate camera gain fluctuation."""
    if sigma <= 0:
        return arr2d
    a = np.clip(arr2d.astype(np.float32), 0.0, 1.0)
    a = a * (1.0 + rng.normal(0.0, sigma, size=a.shape).astype(np.float32))
    return np.clip(a, 0.0, 1.0).astype(arr2d.dtype)


def _poisson_noise(arr2d, strength, rng):
    """Simulate gamma-camera Poisson statistics by re-sampling.

    Treat normalised image as scaled count rate; sample new counts; rescale.
    strength = 0.05 → noise std ≈ 5% of max intensity, physics-consistent.
    """
    if strength <= 0:
        return arr2d
    a = np.clip(arr2d.astype(np.float32), 0.0, 1.0)
    # higher max counts → less noise. strength*1000 = effective max counts (lower = noisier)
    max_counts = max(50.0, 1.0 / max(strength, 1e-3) * 50)
    scaled = a * max_counts
    # rng.poisson is integer; cast to float and renormalise
    noised = rng.poisson(scaled).astype(np.float32) / max_counts
    return np.clip(noised, 0.0, 1.0).astype(arr2d.dtype)


def _gaussian_feather_mask(h: int, w: int, sigma_pct: float = 0.15) -> np.ndarray:
    """Build 2D Gaussian feathering mask centered on patch — edges fade to 0."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sy, sx = max(1.0, h * sigma_pct), max(1.0, w * sigma_pct)
    g = np.exp(-((yy - cy) ** 2 / (2 * sy * sy) + (xx - cx) ** 2 / (2 * sx * sx)))
    g = g / (g.max() + 1e-8)
    return g.astype(np.float32)


def _soft_cp_paired(ant_img, ant_les, ant_ana, post_img, post_les, post_ana,
                    lesion_library, anatomy_bone_ids, n_paste_range, rng):
    """Soft-CP with paired ANT-POST pasting (real lesion visible
    in BOTH views; artifact usually only one view).

    lesion_library: list of dicts {ant_crop, post_crop, ant_lesion_mask, post_lesion_mask}
                    where ant_crop and post_crop are the SAME lesion seen from ANT/POST views,
                    each a (h, w) float [0,1] array; lesion masks are (h, w) bool.
                    Crops are already x-flipped to align with our flip_x convention (so
                    paste at same (y, x) in both arrays).
    anatomy_bone_ids: set of anatomy label ids considered "bone" (paste target).
    n_paste_range: (min, max) inclusive, e.g. (1, 3) means paste 1 to 3 lesions.

    Returns modified (ant_img, ant_les, ant_ana, post_img, post_les, post_ana).
    """
    if not lesion_library:
        return ant_img, ant_les, ant_ana, post_img, post_les, post_ana
    H, W = ant_img.shape
    n_paste = rng.integers(n_paste_range[0], n_paste_range[1] + 1)

    # Acceptable paste mask: bone-anatomy in ANT
    if anatomy_bone_ids is not None:
        ant_bone = np.isin(ant_ana, list(anatomy_bone_ids))
    else:
        ant_bone = ant_img > 0.1  # fallback: foreground
    valid_ys, valid_xs = np.where(ant_bone)
    if len(valid_ys) == 0:
        return ant_img, ant_les, ant_ana, post_img, post_les, post_ana

    for _ in range(int(n_paste)):
        idx = int(rng.integers(0, len(lesion_library)))
        L = lesion_library[idx]
        ac, pc = L["ant_crop"], L["post_crop"]
        am, pm = L["ant_lesion_mask"], L["post_lesion_mask"]
        ph, pw = ac.shape
        if ph >= H or pw >= W:
            continue
        # random valid bone center
        k = int(rng.integers(0, len(valid_ys)))
        cy, cx = int(valid_ys[k]), int(valid_xs[k])
        y0 = max(0, min(H - ph, cy - ph // 2))
        x0 = max(0, min(W - pw, cx - pw // 2))
        # feather mask
        feather = _gaussian_feather_mask(ph, pw, 0.20)
        # blend ANT
        roi_a = ant_img[y0:y0+ph, x0:x0+pw].astype(np.float32)
        ant_img[y0:y0+ph, x0:x0+pw] = (roi_a * (1 - feather) + ac.astype(np.float32) * feather).astype(ant_img.dtype)
        ant_les[y0:y0+ph, x0:x0+pw] = np.maximum(ant_les[y0:y0+ph, x0:x0+pw], am.astype(ant_les.dtype))
        # blend POST at the MIRRORED x coordinate (flip_x convention)
        post_x0 = max(0, min(W - pw, W - 1 - (x0 + pw)))
        roi_p = post_img[y0:y0+ph, post_x0:post_x0+pw].astype(np.float32)
        post_img[y0:y0+ph, post_x0:post_x0+pw] = (roi_p * (1 - feather) + pc.astype(np.float32) * feather).astype(post_img.dtype)
        post_les[y0:y0+ph, post_x0:post_x0+pw] = np.maximum(post_les[y0:y0+ph, post_x0:post_x0+pw], pm.astype(post_les.dtype))

    return ant_img, ant_les, ant_ana, post_img, post_les, post_ana


def _mixup_background(target_img, target_les, source_img, alpha, rng):
    """Image-level MixUp blending TWO images, but only at background (target_les==0).

    Preserves lesion intensity (full target weight inside lesion region).
    Beta(alpha, alpha) weight; alpha=0.2 gives moderate mixing.
    """
    if alpha <= 0:
        return target_img
    lam = float(rng.beta(alpha, alpha))
    t = target_img.astype(np.float32)
    s = source_img.astype(np.float32)
    lesion_mask = (target_les > 0).astype(np.float32)
    # inside lesion: keep target; outside: linear blend (lam*t + (1-lam)*s)
    blended = lam * t + (1.0 - lam) * s
    out = np.where(lesion_mask > 0, t, blended)
    return np.clip(out, 0.0, 1.0).astype(target_img.dtype)


def random_augment_2view(
    ant_img, ant_les, ant_ana,
    post_img, post_les, post_ana,
    strength: str = "medium",
    seed: int | None = None,
    lesion_library=None,
    anatomy_bone_ids=None,
    mixup_partner=None,  # tuple (ant_img_other, post_img_other) for background mixup
):
    """Apply the SAME spatial transform to ANT and POST. Intensity is independent.

    Tier 2 extras (strength='tier2_softcp'):
      - Paired Soft-CP: paste lesion in BOTH ANT and POST at mirrored x position
        (uses lesion_library and anatomy_bone_ids)
      - Background MixUp with another sample (uses mixup_partner)
      - Strong intensity: gamma, brightness, contrast, multiplicative + Poisson noise
    """
    cfg = STRENGTH_PRESETS[strength]
    rng = np.random.default_rng(seed)

    angle, scale = _spatial_transform(rng, ant_img.shape[0], ant_img.shape[1],
                                      cfg["rot_deg"], cfg["scale_pct"])
    # apply to images
    ant_img = _apply_spatial(ant_img, angle, scale)
    post_img = _apply_spatial(post_img, angle, scale)
    ant_les = _apply_spatial(ant_les, angle, scale)
    post_les = _apply_spatial(post_les, angle, scale)
    ant_ana = _apply_spatial(ant_ana, angle, scale)
    post_ana = _apply_spatial(post_ana, angle, scale)

    # Soft-CP paired pasting (before intensity ops so synthetic lesion goes through same intensity perturbation)
    if cfg.get("softcp_p", 0.0) > 0 and lesion_library and rng.random() < cfg["softcp_p"]:
        ant_img, ant_les, ant_ana, post_img, post_les, post_ana = _soft_cp_paired(
            ant_img, ant_les, ant_ana, post_img, post_les, post_ana,
            lesion_library, anatomy_bone_ids, (1, 3), rng)

    # intensity (independent per view)
    if cfg["gamma_range"] != (1.0, 1.0) or cfg["brightness_pct"] > 0:
        ant_img = _gamma_brightness(ant_img, cfg["gamma_range"], cfg["brightness_pct"], rng)
        post_img = _gamma_brightness(post_img, cfg["gamma_range"], cfg["brightness_pct"], rng)
    # contrast (Tier 2)
    if cfg.get("contrast_pct", 0.0) > 0:
        ant_img = _contrast(ant_img, cfg["contrast_pct"], rng)
        post_img = _contrast(post_img, cfg["contrast_pct"], rng)
    # multiplicative noise (Tier 2)
    if cfg.get("mult_noise_sigma", 0.0) > 0:
        ant_img = _mult_noise(ant_img, cfg["mult_noise_sigma"], rng)
        post_img = _mult_noise(post_img, cfg["mult_noise_sigma"], rng)
    # Poisson noise (gamma-camera physics-consistent)
    if cfg.get("poisson_strength", 0.0) > 0:
        ant_img = _poisson_noise(ant_img, cfg["poisson_strength"], rng)
        post_img = _poisson_noise(post_img, cfg["poisson_strength"], rng)

    # MixUp with another patient's image (background only — preserves lesion in target)
    if cfg.get("mixup_p", 0.0) > 0 and mixup_partner is not None and rng.random() < cfg["mixup_p"]:
        other_ant, other_post = mixup_partner
        ant_img = _mixup_background(ant_img, ant_les, other_ant, 0.2, rng)
        post_img = _mixup_background(post_img, post_les, other_post, 0.2, rng)

    # horizontal flip (rare; usually disabled, but ok for intensity_safe / tier2 presets)
    if rng.random() < cfg["flip_p"]:
        ant_img = np.ascontiguousarray(ant_img[:, ::-1])
        post_img = np.ascontiguousarray(post_img[:, ::-1])
        ant_les = np.ascontiguousarray(ant_les[:, ::-1])
        post_les = np.ascontiguousarray(post_les[:, ::-1])
        ant_ana = np.ascontiguousarray(ant_ana[:, ::-1])
        post_ana = np.ascontiguousarray(post_ana[:, ::-1])
    return (ant_img, ant_les, ant_ana), (post_img, post_les, post_ana)


# ---------------------------------------------------------------------------
def apply_clahe(arr2d: np.ndarray, clip_limit: float = 0.01,
                tile_size: tuple = (16, 8)) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization (skimage).

    Bone-scan images often have low contrast in soft-tissue regions; CLAHE
    boosts local contrast in low-uptake areas without saturating hotspots.
    Apply on per-image normalised array (0-1 float).

    JNM 2022 (PMID 35468579) showed +1.5-2% Dice on bone scan classification.
    """
    try:
        from skimage.exposure import equalize_adapthist
    except ImportError:
        return arr2d
    a = np.clip(arr2d.astype(np.float32), 0.0, 1.0)
    if a.max() < 0.01:
        return arr2d
    out = equalize_adapthist(a, kernel_size=tile_size, clip_limit=clip_limit)
    return out.astype(arr2d.dtype)


def tta_predictions(model, x_ant: "torch.Tensor", x_post: "torch.Tensor",
                    flip_h: bool = True, multi_scale: bool = False,
                    scales: tuple = (0.95, 1.0, 1.05)) -> dict:
    """Test-Time Augmentation: average probability maps across flip + scales.

    Returns dict with averaged ant/post probability tensors (B, C, H, W).
    Boost: +0.01-0.03 Dice on small lesions per literature.
    """
    import torch
    import torch.nn.functional as F

    outs_a = []
    outs_p = []

    def _forward(xa, xp):
        out = model(xa, xp, anatomy_gt_ant=None, anatomy_gt_post=None,
                    curriculum_alpha=0.0)
        pa = F.softmax(out["lesion_logits_ant"], dim=1)
        pp = F.softmax(out["lesion_logits_post"], dim=1)
        return pa, pp

    # base
    with torch.no_grad():
        pa, pp = _forward(x_ant, x_post)
        outs_a.append(pa); outs_p.append(pp)
        # horizontal flip
        if flip_h:
            pa_f, pp_f = _forward(torch.flip(x_ant, dims=[-1]),
                                   torch.flip(x_post, dims=[-1]))
            outs_a.append(torch.flip(pa_f, dims=[-1]))
            outs_p.append(torch.flip(pp_f, dims=[-1]))
        # multi-scale (interpolate input then back)
        if multi_scale:
            for s in scales:
                if s == 1.0:
                    continue
                H = int(x_ant.shape[-2] * s) // 8 * 8
                W = int(x_ant.shape[-1] * s) // 8 * 8
                xa_s = F.interpolate(x_ant, size=(H, W), mode="bilinear", align_corners=False)
                xp_s = F.interpolate(x_post, size=(H, W), mode="bilinear", align_corners=False)
                pa_s, pp_s = _forward(xa_s, xp_s)
                pa_s = F.interpolate(pa_s, size=x_ant.shape[-2:], mode="bilinear", align_corners=False)
                pp_s = F.interpolate(pp_s, size=x_ant.shape[-2:], mode="bilinear", align_corners=False)
                outs_a.append(pa_s); outs_p.append(pp_s)

    avg_a = sum(outs_a) / len(outs_a)
    avg_p = sum(outs_p) / len(outs_p)
    return {"prob_ant": avg_a, "prob_post": avg_p}
