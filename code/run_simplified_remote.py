"""Unified simplified-architecture training script.

Architecture (per ablation conclusions):
  - AnchorBSModel(enable_hbs=False, enable_cvfm=True,
                  num_anatomy_classes=11, num_lesion_classes=2)
  - Loss: simple Dice + CE on lesion (1.0 + 1.0); Dice + CE on anatomy (0.5 + 0.5).
    NO Focal, NO Tversky, NO anatomy_constraint, NO CATMIL, NO SDHL, NO boundary.
  - Aug: tier1_softcp + Paired Soft-CP (handled inside data.py), 10x virtual expansion
  - Optim: AdamW, LR=1e-3 for all parameters, 2 warm-up epochs then cosine decay (see the note below)
  - Epochs: 40, ES patience 15, weight_decay=5e-5, grad_clip=1.0
  - Warm: MAE Stage 1 ckpt (env MAE_CKPT), partial-load by shape match.
  - use_anatomy_task: ON if ANAT_SRC non-empty, OFF otherwise
  - HBS curriculum disabled (no HBS module)

Env vars:
  LESION_SRC   : comma-separated source names  (required)
  ANAT_SRC     : comma-separated anatomy sources, or empty -> anatomy task off
  FOLDS        : comma-separated fold indices, default "0"
  WARM_START   : ckpt path | "none" | "MAE"  (default MAE -> uses MAE_CKPT env)
  OUT_DIR_NAME : directory name under $RUNS_DIR    (default "simplified_run")
  EXPAND       : virtual expansion factor (default 10)
  EPOCHS, BATCH_SIZE, NUM_WORKERS, LR, LR_ENC, AUG_STRENGTH
  ES_PATIENCE  (default 15), ES_MIN_EPOCHS (default 20)

Note added for the code release: the summary above predates the final configuration; the settings used in the
paper are in configs/*.env. bonescan.training.fit optimises all parameters in a single AdamW group (two warm-up
epochs, then cosine decay), so LR_ENC is written to run_metadata.json but not applied, and cfg.use_onecycle is
not read. Some log and metadata strings are fixed text ("enable_cvfm=True" in the start-up log, "anatomy
multi-task" in run_metadata.json, "CVFM on" in summary.json); cross-view fusion is set by ENABLE_CVFM (0 in the
paper) and the anatomy task by ANAT_SRC.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BONESCAN_ROOT = Path(os.environ.get("BONESCAN_ROOT", "."))  # folder with data/, runs/, checkpoints/
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(BONESCAN_ROOT / "data" / "standardized")))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(BONESCAN_ROOT / "runs")))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", str(BONESCAN_ROOT / "checkpoints")))
LIBRARY_DIR = Path(os.environ.get("LIBRARY_DIR", str(CKPT_DIR)))
MAE_CKPT = Path(os.environ.get(
    "MAE_CKPT",
    str(BONESCAN_ROOT / "runs" / "mae_pretrain_bs80k" / "ckpt_ep099.pt"),
))

import bonescan.data as _bs_data  # noqa: E402
_bs_data.DEFAULT_OUT = DATA_ROOT

import bonescan.losses as _bs_losses  # noqa: E402

# Monkey-patch focal_loss to plain CE so CompositeLoss gives us a clean
# Dice + CE composite without modifying package source.
_original_focal_loss = _bs_losses.focal_loss


def _ce_replacement(logits, target, *args, **kwargs):
    """Plain cross-entropy (focal with gamma=0)."""
    return F.cross_entropy(logits.float(), target.long())


_bs_losses.focal_loss = _ce_replacement

from bonescan.data import (BoneScanDataset, assemble_patient_records,  # noqa: E402
                           k_fold_split, load_manifest)
from bonescan.losses import CompositeLoss  # noqa: E402
from bonescan.models import AnchorBSModel, BackboneConfig  # noqa: E402
from bonescan.plotting import plot_training_curves  # noqa: E402
from bonescan.training import TrainConfig, fit  # noqa: E402

ANATOMY_BONE_IDS = set(range(1, 11))


def _parse_csv(s: str) -> tuple:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _folds():
    return [int(x) for x in os.environ.get("FOLDS", "0").split(",") if x.strip()]


def _filter_library(lib, src_set):
    if lib is None:
        return None
    return [e for e in lib if e.get("source", "") in src_set]


def _resolve_warm_path() -> Path | None:
    ws = os.environ.get("WARM_START", "MAE").strip()
    if ws.lower() in ("none", "", "scratch"):
        return None
    if ws.upper() == "MAE":
        return MAE_CKPT if MAE_CKPT.exists() else None
    p = Path(ws)
    return p if p.exists() else None


def main():
    lesion_src = _parse_csv(os.environ.get("LESION_SRC", ""))
    anat_src = _parse_csv(os.environ.get("ANAT_SRC", ""))
    if not lesion_src:
        print("[simplified] ERROR: LESION_SRC is required (csv)")
        sys.exit(2)

    use_anatomy = len(anat_src) > 0

    out_name = os.environ.get("OUT_DIR_NAME", "simplified_run")
    out = RUNS_DIR / out_name
    out.mkdir(parents=True, exist_ok=True)
    folds = _folds()

    print(f"[simplified] OUT_DIR_NAME={out_name}  folds={folds}")
    print(f"[simplified] DATA_ROOT={DATA_ROOT}  RUNS_DIR={RUNS_DIR}")
    print(f"[simplified] LESION_SRC={lesion_src}")
    print(f"[simplified] ANAT_SRC={anat_src}  use_anatomy_task={use_anatomy}")
    print(f"[simplified] arch: enable_hbs=False, enable_cvfm=True, anatomy=11, lesion=2")
    print(f"[simplified] loss: simple Dice+CE (focal monkey-patched to CE)")

    df = load_manifest(DATA_ROOT)
    recs = assemble_patient_records(df, DATA_ROOT)
    lesion = [r for r in recs if r.source in set(lesion_src)]
    anatomy = [r for r in recs if r.source in set(anat_src)] if use_anatomy else []
    print(f"[simplified] lesion={len(lesion)} anatomy={len(anatomy)}")

    splits = k_fold_split(lesion, k=5, seed=0)

    warm_path = _resolve_warm_path()
    print(f"[simplified] warm_start = {warm_path}")

    fold_results = {}
    for fold in folds:
        fold_dir = out / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        if (fold_dir / "best.pt").exists():
            print(f"  [fold_{fold}] best.pt exists, skip")
            continue
        tr, va = splits[fold]
        train_set = [lesion[i] for i in tr] + anatomy
        val_set = [lesion[i] for i in va]
        print(f"  [fold_{fold}] train_lesion={len(tr)} train_anat={len(anatomy)} val={len(va)}")

        # Lesion library: try fold-specific first, then any-fold; filter to current LESION_SRC
        lesion_library = None
        for cand in [
            LIBRARY_DIR / f"_lesion_library_fold{fold}.pkl",
            LIBRARY_DIR / f"_lesion_library_fold0.pkl",
        ]:
            if cand.exists():
                try:
                    with open(cand, "rb") as f:
                        raw = pickle.load(f)
                    lesion_library = _filter_library(raw, set(lesion_src))
                    print(f"  [fold_{fold}] library {cand.name}: raw={len(raw)} -> "
                          f"filtered={len(lesion_library)}")
                    break
                except Exception as e:
                    print(f"  [fold_{fold}] library load failed: {e}")
                    lesion_library = None

        train_ds = BoneScanDataset(
            train_set, augment=True,
            augment_strength=os.environ.get("AUG_STRENGTH", "tier1_softcp"),
            lesion_library=lesion_library,
            anatomy_bone_ids=ANATOMY_BONE_IDS if use_anatomy else None,
            mixup_enabled=False,
        )
        val_ds = BoneScanDataset(val_set, augment=False)

        # Virtual expansion (10x by default)
        expand = int(os.environ.get("EXPAND", "10"))
        if expand > 1:
            try:
                train_ds = torch.utils.data.ConcatDataset([train_ds] * expand)
                print(f"  [fold_{fold}] virtual expand x{expand}: train_len={len(train_ds)}")
            except Exception as e:
                print(f"  [fold_{fold}] expand failed: {e}")

        bs = int(os.environ.get("BATCH_SIZE", "4"))
        nw = int(os.environ.get("NUM_WORKERS", "4"))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, drop_last=True, pin_memory=True,
                                  persistent_workers=(nw > 0))
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=max(2, nw // 2), pin_memory=True,
                                persistent_workers=(nw > 0))

        torch.manual_seed(fold); torch.cuda.manual_seed_all(fold)

        # =========== SIMPLIFIED ARCHITECTURE ===========
        # HBS always off. Cross-view fusion is parametrised:
        #   ENABLE_CVFM=1  -> original CVFM (default off after CVFM_OFF
        #                                    fold_0 beat baseline)
        #   ENABLE_XATTN=1 -> transformer-style cross-view attention
        # The two are mutually exclusive (assert inside AnchorBSModel).
        enable_cvfm = bool(int(os.environ.get("ENABLE_CVFM", "0")))
        enable_xattn = bool(int(os.environ.get("ENABLE_XATTN", "0")))
        model = AnchorBSModel(BackboneConfig(),
                              enable_hbs=False,
                              enable_cvfm=enable_cvfm,
                              enable_xattn=enable_xattn,
                              num_anatomy_classes=11,
                              num_lesion_classes=2)
        n_p = sum(p.numel() for p in model.parameters())
        cv_tag = "CVFM" if enable_cvfm else ("XATTN" if enable_xattn else "no_cv")
        print(f"  [fold_{fold}] model params = {n_p/1e6:.2f} M  "
              f"(HBS off, cv={cv_tag})")

        # Warm start (partial load by shape match)
        loaded_from = None
        if warm_path is not None and warm_path.exists():
            ck = torch.load(str(warm_path), map_location="cpu", weights_only=False)
            sd = ck.get("model", ck) if isinstance(ck, dict) else ck
            own = model.state_dict()
            ld = 0
            for k, v in sd.items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v; ld += 1
            model.load_state_dict(own, strict=False)
            loaded_from = str(warm_path)
            print(f"  [fold_{fold}] WARM-LOAD: {ld}/{len(own)} params from {warm_path.name}")
        else:
            print(f"  [fold_{fold}] WARM-LOAD: training from scratch (no warm)")

        # =========== SIMPLIFIED LOSS ===========
        # Lesion: Dice (1.0) + CE (1.0 via monkey-patched focal w=1.0).
        # Anatomy: Dice (0.5) + CE (0.5).
        # No Focal (CE replaces), no Tversky, no anatomy constraint, no boundary.
        loss_fn = CompositeLoss(
            w_lesion_dice=1.0,
            w_lesion_focal=1.0,    # CE via monkey patch
            w_lesion_boundary=0.0,
            w_lesion_tversky=0.0,
            w_anatomy_dice=0.5 if use_anatomy else 0.0,
            w_anatomy_ce=0.5 if use_anatomy else 0.0,
            w_anatomy_constraint=0.0,
            ignore_lesion_bg_dice=True,
            tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=0.75,
        )

        lr_other = float(os.environ.get("LR", "1e-3"))
        lr_enc = float(os.environ.get("LR_ENC", "1e-4"))
        epochs_val = int(os.environ.get("EPOCHS", "40"))
        es_pat = int(os.environ.get("ES_PATIENCE", "15"))
        es_min_ep = int(os.environ.get("ES_MIN_EPOCHS", "20"))

        cfg = TrainConfig(
            epochs=epochs_val, batch_size=bs, num_workers=nw,
            lr=lr_other, weight_decay=5e-5,
            warmup_epochs=2, grad_clip=1.0, save_dir=fold_dir, seed=fold,
            use_anatomy_task=use_anatomy, use_lesion_task=True,
            es_patience=es_pat, es_min_epochs=es_min_ep, es_min_delta=5e-4,
        )
        cfg.lr_encoder = lr_enc
        cfg.use_onecycle = True
        cfg.amp = False
        if os.environ.get("AMP_BF16", "0") == "1":
            cfg.amp = True
            cfg.amp_dtype = "bf16"
        if os.environ.get("CUDNN_BENCH", "0") == "1":
            torch.backends.cudnn.benchmark = True

        print(f"  [fold_{fold}] lr_other={lr_other} lr_enc={lr_enc} "
              f"epochs={epochs_val} bs={bs} expand={expand} use_anat={use_anatomy} "
              f"amp={cfg.amp}/{getattr(cfg, 'amp_dtype', 'fp16')} cudnn_bench={torch.backends.cudnn.benchmark}")
        t0 = time.time()
        history = fit(model, train_loader, val_loader, cfg, loss_fn)
        elapsed = (time.time() - t0) / 60.0
        print(f"  [fold_{fold}] elapsed = {elapsed:.1f} min")

        try:
            plot_training_curves(history, fold_dir / "fig_training_curves.png")
        except Exception as e:
            print(f"  [fold_{fold}] plot failed: {e}")

        (fold_dir / "history.json").write_text(json.dumps(history, indent=2))
        best = max([x for x in history.get("val_lesion_dice", []) if x == x],
                   default=float("nan"))
        fold_results[fold] = {"best_val_lesion_dice": float(best)}
        (fold_dir / "run_metadata.json").write_text(json.dumps({
            "arch": f"simplified (HBS off, cv={cv_tag}, anatomy multi-task)",
            "enable_cvfm": enable_cvfm,
            "enable_xattn": enable_xattn,
            "loss": "Dice + CE only (no focal/tversky/boundary/constraint)",
            "warm_start": loaded_from,
            "batch_size": bs,
            "epochs": cfg.epochs,
            "lr_other": lr_other,
            "lr_enc": lr_enc,
            "lesion_src": list(lesion_src),
            "anat_src": list(anat_src),
            "use_anatomy_task": use_anatomy,
            "expand_factor": expand,
            "n_train_lesion": len(tr),
            "n_train_anat": len(anatomy),
            "n_val": len(va),
            "elapsed_min": elapsed,
        }, indent=2))
        print(f"  [fold_{fold}] best_val_lesion_dice = {best:.4f}")

        del model, train_loader, val_loader, train_ds, val_ds, lesion_library
        gc.collect(); torch.cuda.empty_cache()

    summary = {
        "variant": out_name,
        "arch": "simplified (HBS off, CVFM on, anatomy MT)",
        "loss": "Dice + CE only",
        "fold_results": fold_results,
        "lesion_src": list(lesion_src),
        "anat_src": list(anat_src),
        "use_anatomy_task": use_anatomy,
        "warm_start": str(warm_path) if warm_path else None,
        "mean_best_dice": (sum(r["best_val_lesion_dice"] for r in fold_results.values()) /
                           len(fold_results)) if fold_results else float("nan"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[simplified] summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
