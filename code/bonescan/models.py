"""Segmentation backbone + multi-task heads + Anatomy-Gated HBS + CVFM.

Design (Lite v2):
  - Single backbone: nnU-Net ResEnc-M built via MONAI's DynUNet.
  - Siamese encoder (shared weights) consumes ANT and POST separately.
  - CVFM at the bottleneck fuses the two view feature pyramids.
  - Three decoder heads: anatomy (11-class), lesion (2-class), recon (only used in Stage 1 MAE).
  - HBS module sits *between* anatomy decoder output and lesion decoder, gated by anatomy prediction.
  - At inference, anatomy_pred is the gate. At training-Stage-3, a curriculum
    blends GT anatomy mask (when available) with the prediction.

This module is intentionally self-contained: no Lightning, no training logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
@dataclass
class BackboneConfig:
    in_channels: int = 1                 # we run encoder per-view, single channel
    base_channels: int = 32
    pool_stages: int = 6
    feature_caps: tuple = (32, 64, 128, 256, 320, 320, 320)
    use_residual: bool = True            # ResEnc-M


def _conv_block(c_in, c_out, residual=True):
    layers = [
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(c_out, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(c_out, affine=True),
    ]
    if residual:
        return _ResBlock(c_in, c_out, layers)
    return nn.Sequential(*layers, nn.LeakyReLU(0.01, inplace=True))


class _ResBlock(nn.Module):
    def __init__(self, c_in, c_out, body_layers):
        super().__init__()
        self.body = nn.Sequential(*body_layers)
        self.short = (nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
                      if c_in != c_out else nn.Identity())
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.short(x))


# ---------------------------------------------------------------------------
class Encoder2D(nn.Module):
    """Hierarchical encoder producing a feature pyramid."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        c_in = cfg.in_channels
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        chans = [cfg.base_channels]
        c = cfg.base_channels
        self.stages.append(_conv_block(c_in, c, residual=cfg.use_residual))
        for i in range(cfg.pool_stages):
            c_next = cfg.feature_caps[min(i + 1, len(cfg.feature_caps) - 1)]
            self.downs.append(nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False))
            self.stages.append(_conv_block(c, c_next, residual=cfg.use_residual))
            chans.append(c_next)
            c = c_next
        self.out_channels = chans  # list of channel widths per stage

    def forward(self, x):
        feats = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downs[i - 1](x)
            x = stage(x)
            feats.append(x)
        return feats  # list of (B, C_i, H/2^i, W/2^i)


# ---------------------------------------------------------------------------
class CVFM(nn.Module):
    """Cross-View Fusion Module (lite, CNN variant).

    Inputs: f_ant, f_post  — same shape (B, C, H, W) from the deepest encoder stage.
    Output: fused feature of same shape.

    Implementation: concat along channels, 1x1 reduce, 3x3 mix, SE attention,
    residual add to each view's feature.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_ant, f_post):
        cat = torch.cat([f_ant, f_post], dim=1)
        x = self.reduce(cat)
        x = self.mix(x)
        gate = self.se(x)
        x = x * gate
        # add residually to both views (fused information broadcasts back)
        return f_ant + x, f_post + x


# ---------------------------------------------------------------------------
class CrossViewAttention(nn.Module):
    """Bidirectional cross-view attention at the bottleneck.

    Drop-in alternative to CVFM that follows the standard transformer-style
    cross-attention pattern used widely in cross-view / multi-modal medical
    image segmentation papers (e.g. dual-view bilateral attention modules in
    radiology fusion work).

      Q_a, K_a, V_a from ANT features
      Q_p, K_p, V_p from POST features

      out_a = ANT  +  Proj_a( Attn(Q_a, K_p, V_p) )
      out_p = POST +  Proj_p( Attn(Q_p, K_a, V_a) )

    The bottleneck spatial size is small (~10x8 for a 600x512 input), so
    softmax cost is negligible compared to CVFM.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, (channels, num_heads)
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        # 1x1 projections per view
        self.qa = nn.Conv2d(channels, channels, 1, bias=False)
        self.ka = nn.Conv2d(channels, channels, 1, bias=False)
        self.va = nn.Conv2d(channels, channels, 1, bias=False)
        self.qp = nn.Conv2d(channels, channels, 1, bias=False)
        self.kp = nn.Conv2d(channels, channels, 1, bias=False)
        self.vp = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm_a = nn.GroupNorm(1, channels)
        self.norm_p = nn.GroupNorm(1, channels)
        self.proj_a = nn.Conv2d(channels, channels, 1)
        self.proj_p = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q.shape
        h, d = self.num_heads, self.head_dim
        q = q.reshape(B, h, d, H * W).permute(0, 1, 3, 2)   # (B, h, HW, d)
        k = k.reshape(B, h, d, H * W).permute(0, 1, 3, 2)
        v = v.reshape(B, h, d, H * W).permute(0, 1, 3, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale       # (B, h, HW, HW)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                       # (B, h, HW, d)
        return out.permute(0, 1, 3, 2).reshape(B, C, H, W)

    def forward(self, f_ant: torch.Tensor, f_post: torch.Tensor):
        na = self.norm_a(f_ant)
        np_ = self.norm_p(f_post)
        qa, ka, va = self.qa(na), self.ka(na), self.va(na)
        qp, kp, vp = self.qp(np_), self.kp(np_), self.vp(np_)
        # ANT queries POST keys/values; POST queries ANT keys/values.
        out_a = self.proj_a(self._attend(qa, kp, vp))
        out_p = self.proj_p(self._attend(qp, ka, va))
        return f_ant + out_a, f_post + out_p


# ---------------------------------------------------------------------------
class Decoder2D(nn.Module):
    """Symmetric U-Net decoder with skip connections and deep supervision outputs.

    Optional Coordinate Attention plug-in on every skip (Variant G+ add-on).
    """

    def __init__(self, encoder_channels: list[int], num_classes: int,
                 deep_supervision: bool = True,
                 use_coord_attn: bool = False):
        super().__init__()
        self.ds = deep_supervision
        self.use_coord_attn = use_coord_attn
        enc = list(encoder_channels)            # e.g. [32, 64, 128, 256, 320, 320, 320]
        # Decoder upsamples from deepest (idx -1) to shallowest (idx 0)
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.coord_attns = nn.ModuleList() if use_coord_attn else None
        for i in range(len(enc) - 1, 0, -1):
            in_c, out_c = enc[i], enc[i - 1]
            self.ups.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
            self.blocks.append(_conv_block(out_c * 2, out_c, residual=True))
            self.heads.append(nn.Conv2d(out_c, num_classes, kernel_size=1))
            if use_coord_attn:
                from .addons import CoordinateAttention
                self.coord_attns.append(CoordinateAttention(out_c, reduction=32))

    def forward(self, feats: list[torch.Tensor]):
        x = feats[-1]
        deep_outs = []
        for i, (up, blk, head) in enumerate(zip(self.ups, self.blocks, self.heads)):
            x = up(x)
            skip = feats[-2 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skip], dim=1))
            if self.use_coord_attn:
                x = self.coord_attns[i](x)
            deep_outs.append(head(x))
        # deep_outs[-1] is the highest-resolution prediction (matches input)
        return deep_outs


# ---------------------------------------------------------------------------
class HBSModule(nn.Module):
    """Anatomy-Gated Hierarchical Background Smoothing.

    Adapted from Sun et al. CVPR 2025 (SET) with two key changes:
      - The FG/BG gate is the anatomy *prediction* (or curriculum mix with GT)
        so HBS can run at inference, not training-only.
      - Operates on lesion-decoder skip features rather than detection FPN.

    P_out = P * M + smoothed(P * (1 - M)) + P * (1 - M)
          where smoothed = Conv_expand(ReLU(Conv_reduce(P * (1 - M))))
    """

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 3):
        super().__init__()
        mid = max(channels // reduction, 4)
        pad = kernel_size // 2
        self.reduce = nn.Conv2d(channels, mid, kernel_size=kernel_size, padding=pad, bias=False)
        self.expand = nn.Conv2d(mid, channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, p_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        p_feat: (B, C, H, W)
        mask:   (B, 1, H, W) — float in [0, 1] (binary or soft anatomy probability)
        """
        if mask.shape[-2:] != p_feat.shape[-2:]:
            mask = F.interpolate(mask, size=p_feat.shape[-2:], mode="bilinear", align_corners=False)
        m = mask.clamp(0, 1)
        p_fg = p_feat * m
        p_bg = p_feat * (1.0 - m)
        smoothed = self.expand(self.act(self.reduce(p_bg)))
        return p_fg + smoothed + p_bg


# ---------------------------------------------------------------------------
class AnchorBSModel(nn.Module):
    """Full AnchorBS-Lite model: shared encoder + CVFM + three decoders + HBS.

    Forward signature:
      forward(x_ant, x_post, anatomy_gt=None, curriculum_alpha=0.0, return_intermediate=False)
        x_ant, x_post: (B, 1, H, W)
        anatomy_gt:    (B, 1, H, W) optional, used only during training-Stage-3 curriculum
        curriculum_alpha: float in [0, 1]
            0  → gate = anatomy prediction (inference / late training)
            1  → gate = anatomy_gt          (early training, only when anatomy_gt available)
            interpolated in between
    Returns dict with keys: lesion_logits, anatomy_logits, anatomy_mask_used, recon (optional)
    """

    def __init__(self,
                 backbone_cfg: BackboneConfig | None = None,
                 num_anatomy_classes: int = 11,
                 num_lesion_classes: int = 2,
                 enable_hbs: bool = True,
                 enable_cvfm: bool = True,
                 enable_xattn: bool = False,
                 enable_recon: bool = False,
                 enable_coord_attn: bool = False):
        super().__init__()
        self.cfg = backbone_cfg or BackboneConfig()
        self.encoder = Encoder2D(self.cfg)
        bottleneck_c = self.encoder.out_channels[-1]
        # CVFM and CrossViewAttention are mutually exclusive (either is a
        # cross-view fusion mechanism at the bottleneck); enable at most one.
        assert not (enable_cvfm and enable_xattn), (
            "enable_cvfm and enable_xattn cannot both be True")
        self.cvfm = CVFM(bottleneck_c) if enable_cvfm else None
        self.xattn = CrossViewAttention(bottleneck_c) if enable_xattn else None
        # Coordinate Attention only on the lesion decoder (training-cost & alignment friendly)
        self.anat_decoder = Decoder2D(self.encoder.out_channels, num_anatomy_classes)
        self.lesion_decoder = Decoder2D(self.encoder.out_channels, num_lesion_classes,
                                        use_coord_attn=enable_coord_attn)
        self.hbs = HBSModule(self.encoder.out_channels[-1]) if enable_hbs else None
        self.enable_recon = enable_recon
        self.enable_coord_attn = enable_coord_attn
        if enable_recon:
            self.recon_decoder = Decoder2D(self.encoder.out_channels, self.cfg.in_channels)
        self.num_anatomy_classes = num_anatomy_classes
        self.num_lesion_classes = num_lesion_classes

    # ---- helpers ----
    def encode_views(self, x_ant: torch.Tensor, x_post: torch.Tensor):
        feats_a = self.encoder(x_ant)
        feats_p = self.encoder(x_post)
        if self.cvfm is not None:
            f_a, f_p = self.cvfm(feats_a[-1], feats_p[-1])
            feats_a[-1] = f_a
            feats_p[-1] = f_p
        elif self.xattn is not None:
            f_a, f_p = self.xattn(feats_a[-1], feats_p[-1])
            feats_a[-1] = f_a
            feats_p[-1] = f_p
        return feats_a, feats_p

    def _anatomy_gate(self, anatomy_logits: torch.Tensor,
                      anatomy_gt: torch.Tensor | None,
                      alpha: float) -> torch.Tensor:
        """Build the HBS gate by mixing GT mask (alpha) and prediction (1-alpha).

        Both are converted to a single 'foreground = bone' binary/soft channel.
        Class 0 = background, classes 1..10 = bone/organ regions; we treat all
        non-background classes as foreground (anatomy region).
        """
        pred_prob = anatomy_logits.softmax(dim=1)
        pred_fg = 1.0 - pred_prob[:, 0:1]                 # (B, 1, H, W)
        if anatomy_gt is None or alpha == 0.0:
            return pred_fg
        gt_fg = (anatomy_gt > 0).float()
        if gt_fg.dim() == 3:
            gt_fg = gt_fg.unsqueeze(1)
        return alpha * gt_fg + (1.0 - alpha) * pred_fg

    # ---- forward ----
    def forward(self, x_ant: torch.Tensor, x_post: torch.Tensor,
                anatomy_gt_ant: torch.Tensor | None = None,
                anatomy_gt_post: torch.Tensor | None = None,
                curriculum_alpha: float = 0.0,
                return_intermediate: bool = False) -> dict:
        feats_a, feats_p = self.encode_views(x_ant, x_post)
        anat_out_a = self.anat_decoder(feats_a)
        anat_out_p = self.anat_decoder(feats_p)

        # Build gate for HBS
        if self.hbs is not None:
            gate_a = self._anatomy_gate(anat_out_a[-1], anatomy_gt_ant, curriculum_alpha)
            gate_p = self._anatomy_gate(anat_out_p[-1], anatomy_gt_post, curriculum_alpha)
            # Apply HBS to the bottleneck feature
            feats_a[-1] = self.hbs(feats_a[-1], gate_a)
            feats_p[-1] = self.hbs(feats_p[-1], gate_p)
        else:
            gate_a = gate_p = None

        lesion_out_a = self.lesion_decoder(feats_a)
        lesion_out_p = self.lesion_decoder(feats_p)

        out = {
            "lesion_logits_ant": lesion_out_a[-1],
            "lesion_logits_post": lesion_out_p[-1],
            "anatomy_logits_ant": anat_out_a[-1],
            "anatomy_logits_post": anat_out_p[-1],
            "lesion_logits_ds_ant": lesion_out_a,         # full DS list
            "lesion_logits_ds_post": lesion_out_p,
            "anatomy_logits_ds_ant": anat_out_a,
            "anatomy_logits_ds_post": anat_out_p,
        }
        if self.enable_recon:
            recon_a = self.recon_decoder(feats_a)
            recon_p = self.recon_decoder(feats_p)
            out["recon_ant"] = recon_a[-1]
            out["recon_post"] = recon_p[-1]
        if return_intermediate:
            out["gate_ant"] = gate_a
            out["gate_post"] = gate_p
        return out


# ---------------------------------------------------------------------------
def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def smoke_test():
    """Quick smoke test: build model + run on dummy input."""
    cfg = BackboneConfig()
    m = AnchorBSModel(cfg).cuda() if torch.cuda.is_available() else AnchorBSModel(cfg)
    print(f"AnchorBSModel params: {count_parameters(m) / 1e6:.2f} M")
    H, W = 1024, 256
    x_ant = torch.randn(1, 1, H, W, device=next(m.parameters()).device)
    x_post = torch.randn(1, 1, H, W, device=next(m.parameters()).device)
    out = m(x_ant, x_post)
    print("output shapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s} {tuple(v.shape)}")
        elif isinstance(v, list):
            print(f"  {k:25s} list[len={len(v)}]")


if __name__ == "__main__":
    smoke_test()
