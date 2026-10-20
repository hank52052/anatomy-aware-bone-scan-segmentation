"""Per-image / per-patient / lesion-level evaluation of the final 5-fold models.

Each method is evaluated the way it was trained:
  ours_*  : AnchorBSModel without CVFM/HBS, BoneScanDataset inputs (no extra normalisation)
  bsciseg : BSci-Seg re-implementation, BSciSegValDataset inputs (Otsu-trim normalisation)
  nnunet  : official nnUNetPredictor (its own z-score, sliding window, mirroring)
All methods: checkpoint chosen on the validation fold by the method's own rule, fixed threshold 0.5,
horizontal-flip TTA for ours/BSci-Seg (nnU-Net uses its default mirroring).

Lesion-free scans: the 60 old-hospital non-metastatic patients are scored inside cross-validation;
the 39 new-hospital metastasis-free patients (new_bb/new_bp, never used for training) are scored by
the model of the fold their patient hash maps to.

Env:
  METHOD      ours_anat | ours_noanat | bsciseg | nnunet | ours_ens | ours_anat_kf
  FOLDS       comma list, default 0,1,2,3,4
  OUT_DIR     default runs/final_eval
  THRESH_GRID 1 = also score every threshold in GRID ({method}_fold{k}_thr.json) and keep the float16
              probability maps ({method}_fold{k}_prob.npz) for the cross-validated threshold analysis
  PP_GRID     1 = also score every post-processing configuration in PP_CONFIGS ({method}_fold{k}_pp.npz),
              computed from the float32 probabilities, for the cross-validated post-processing analysis
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

BONESCAN = Path(os.environ.get("BONESCAN_ROOT", "."))  # folder with data/, runs/ and nnunet/
CODE = Path(__file__).resolve().parents[1]  # the code/ folder (bonescan package and entry scripts)
sys.path.insert(0, str(CODE))
DATA_ROOT = BONESCAN / "data" / "standardized"

import bonescan.data as _bs_data  # noqa: E402

_bs_data.DEFAULT_OUT = DATA_ROOT
from bonescan.data import (BoneScanDataset, assemble_patient_records, k_fold_split,  # noqa: E402
                           load_manifest, patient_id_hash)

LESION_SRC = ("new_mb", "new_mp", "meta_breast_meta", "meta_prostate_meta",
              "meta_breast_non-meta", "meta_prostate_non-meta")
NEG_SRC = ("new_bb", "new_bp")
NEW_HOSPITAL = {"new_mb", "new_mp", "new_bb", "new_bp"}
THRESHOLD = 0.5
GRID = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 ... 0.90, secondary analysis only
# Post-processing grid (secondary analysis, rule fixed 2026-10-20 05:15): threshold t, minimum component size s
# (pixels), minimum component peak probability h, and skeleton filter a (models with an anatomy branch only).
PP_S, PP_H, PP_A = (0, 5, 10, 20, 40), (0.0, 0.7, 0.9), (0, 1)
PP_CONFIGS = [(t, s, h, a) for t in GRID for s in PP_S for h in PP_H for a in PP_A]
PP_FIELDS = ("tp", "fp", "fn", "tn", "gt_area", "pred_area", "n_gt_lesions", "n_gt_detected",
             "n_pred_components", "n_fp_components")
ANATOMY_METHODS = {"ours_anat", "ours_ens", "ours_anat_kf"}
STRUCT = np.ones((3, 3), dtype=bool)  # 8-connectivity
NNUNET_MODEL_DIR = BONESCAN / "nnunet/results/Dataset501_BoneScanLesion/nnUNetTrainer_250epochs__nnUNetPlans__2d"
RUN_DIRS = {"ours_anat": "final_ours_anat", "ours_noanat": "final_ours_noanat"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- models
def _state_dict(ckpt_path: Path) -> dict:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "network_weights"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            return ck[key]
    return ck


def _strict_load(model: torch.nn.Module, sd: dict, tag: str) -> None:
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if "anat" not in k.lower() and "recon" not in k.lower()]
    if missing:
        raise RuntimeError(f"[{tag}] {len(missing)} lesion-path weights missing, e.g. {missing[:5]}")
    print(f"  [{tag}] loaded; missing(non-lesion)={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
          flush=True)


def _bsciseg_module():
    spec = importlib.util.spec_from_file_location("bsciseg_mod", str(CODE / "run_variant_BSciSeg_remote.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KIDNEY_CLASS = 6  # verified from the anatomy masks: two blobs left/right of the spine at ~0.32 of image height


class TwoViewPredictor:
    """Ours and BSci-Seg: (2,H,W) image tensor -> (2,H,W) lesion probability, h-flip TTA.

    kidney_filter=True (our anatomy model only): lesion probability is set to 0 where the anatomy branch's
    flip-averaged prediction is the kidney class. Fixed rule, no tuning.
    """

    def __init__(self, method: str, fold: int, kidney_filter: bool = False, keep_anatomy: bool = False):
        self.kidney_filter = kidney_filter
        self.keep_anatomy = keep_anatomy  # store the flip-averaged anatomy label map of the last call
        self.last_anatomy = None
        if method.startswith("ours"):
            from bonescan.models import AnchorBSModel, BackboneConfig
            self.model = AnchorBSModel(BackboneConfig(), enable_hbs=False, enable_cvfm=False, enable_xattn=False)
            run_name = os.environ.get("RUN_NAME", RUN_DIRS.get(method, method))
            ckpt = BONESCAN / "runs" / run_name / f"fold_{fold}" / "best.pt"
        else:
            mod = _bsciseg_module()
            self.model = mod.BSciSegUNet(in_channels=1, n_classes=2, features=(32, 64, 128, 256, 512))
            ckpt = BONESCAN / "runs" / "ablation_BSciSeg" / f"fold_{fold}" / "best.pt"  # fold_0 links to *_reimpl
        _strict_load(self.model, _state_dict(ckpt), f"{method} fold{fold} {ckpt.name}")
        self.model.to(DEVICE).eval()
        self.ckpt = str(ckpt)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> np.ndarray:
        x = image.unsqueeze(0).to(DEVICE)
        ant, post = x[:, 0:1], x[:, 1:2]
        out = self.model(ant, post)
        flip = self.model(torch.flip(ant, dims=[-1]), torch.flip(post, dims=[-1]))
        probs, anatomy = [], []
        for view in ("ant", "post"):
            key = f"lesion_logits_{view}"
            p = F.softmax(out[key].float(), dim=1)[0, 1]
            pf = torch.flip(F.softmax(flip[key].float(), dim=1)[0, 1], dims=[-1])
            prob = (p + pf) / 2
            if self.kidney_filter or self.keep_anatomy:
                akey = f"anatomy_logits_{view}"
                a = F.softmax(out[akey].float(), dim=1)[0]
                af = torch.flip(F.softmax(flip[akey].float(), dim=1)[0], dims=[-1])
                labels = ((a + af) / 2).argmax(dim=0)
                anatomy.append(labels.to(torch.uint8).cpu().numpy())
                if self.kidney_filter:
                    prob = torch.where(labels == KIDNEY_CLASS, torch.zeros_like(prob), prob)
            probs.append(prob.cpu().numpy())
        self.last_anatomy = np.stack(anatomy, axis=0) if self.keep_anatomy else None
        return np.stack(probs, axis=0)


class EnsemblePredictor:
    """Secondary analysis: mean probability of the anatomy and no-anatomy models of the same fold."""

    def __init__(self, fold: int, keep_anatomy: bool = False):
        self.members = [TwoViewPredictor("ours_anat", fold, keep_anatomy=keep_anatomy),
                        TwoViewPredictor("ours_noanat", fold)]
        self.ckpt = " + ".join(m.ckpt for m in self.members)

    @property
    def last_anatomy(self):
        return self.members[0].last_anatomy  # anatomy branch of the anatomy-trained member

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        return np.mean([m(image) for m in self.members], axis=0)


class NnUNetViewPredictor:
    """Official nnU-Net inference on each raw (unflipped) view; returns (2,H,W) in dataset frame (POST flipped)."""

    def __init__(self, fold: int):
        os.environ.setdefault("nnUNet_raw", str(BONESCAN / "nnunet/raw"))
        os.environ.setdefault("nnUNet_preprocessed", str(BONESCAN / "nnunet/preprocessed"))
        os.environ.setdefault("nnUNet_results", str(BONESCAN / "nnunet/results"))
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                         perform_everything_on_device=True, device=DEVICE,
                                         verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(str(NNUNET_MODEL_DIR), use_folds=(fold,),
                                                            checkpoint_name="checkpoint_best.pth")
        self.ckpt = str(NNUNET_MODEL_DIR / f"fold_{fold}" / "checkpoint_best.pth")

    def __call__(self, rec) -> np.ndarray:
        probs = []
        for view, path in (("ANT", rec.ant_image_path), ("POST", rec.post_image_path)):
            img = np.load(str(path)).astype(np.float32)
            _, prob = self.predictor.predict_single_npy_array(img[None, None], {"spacing": [999.0, 1.0, 1.0]},
                                                              None, None, True)
            p = np.asarray(prob)[1, 0].astype(np.float32)
            probs.append(p[:, ::-1].copy() if view == "POST" else p)
        return np.stack(probs, axis=0)


# ----------------------------------------------------------------------------- metrics
def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask, structure=STRUCT)


def view_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(pred.size - tp - fp - fn)
    gt_lab, n_gt = _components(gt)
    pr_lab, n_pr = _components(pred)
    gt_hit = len(set(np.unique(gt_lab[pred])) - {0}) if n_gt else 0
    pr_true = len(set(np.unique(pr_lab[gt])) - {0}) if n_pr else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "gt_area": int(gt.sum()), "pred_area": int(pred.sum()),
            "n_gt_lesions": int(n_gt), "n_gt_detected": int(gt_hit),
            "n_pred_components": int(n_pr), "n_fp_components": int(n_pr - pr_true)}


def pp_view_metrics(prob: np.ndarray, gt: np.ndarray, skeleton: np.ndarray | None) -> np.ndarray:
    """view_metrics for every configuration in PP_CONFIGS -> int array (len(PP_CONFIGS), len(PP_FIELDS)).

    Components are labelled once per threshold; size, peak-probability and skeleton filters then act on whole
    components. Without a skeleton map the a=1 rows repeat a=0 (the selection never uses them).
    """
    gt = gt.astype(bool)
    gt_lab, n_gt = _components(gt)
    gt_area = int(gt.sum())
    rows = {}
    for t in GRID:
        lab, n = _components(prob >= t)
        ids = np.arange(1, n + 1)
        if n:
            sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            peak = np.asarray(ndimage.maximum(prob, lab, ids), dtype=float)
            overlap = np.bincount(lab[gt], minlength=n + 1)[1:]
            both = (lab > 0) & (gt_lab > 0)
            pairs = np.unique(lab[both].astype(np.int64) * (n_gt + 1) + gt_lab[both])
            pair_pred, pair_gt = pairs // (n_gt + 1), pairs % (n_gt + 1)
            on_skeleton = (np.bincount(lab[skeleton], minlength=n + 1)[1:] > 0) if skeleton is not None else None
        for s in PP_S:
            for h in PP_H:
                for a in PP_A:
                    if n:
                        keep = (sizes >= s) & (peak >= h)
                        if a and on_skeleton is not None:
                            keep &= on_skeleton
                        tp, pred_area = int(overlap[keep].sum()), int(sizes[keep].sum())
                        detected = int(len(np.unique(pair_gt[np.isin(pair_pred, ids[keep])]))) if len(pairs) else 0
                        n_pred, n_fp = int(keep.sum()), int((keep & (overlap == 0)).sum())
                    else:
                        tp = pred_area = detected = n_pred = n_fp = 0
                    fp, fn = pred_area - tp, gt_area - tp
                    rows[(t, s, h, a)] = (tp, fp, fn, int(prob.size - tp - fp - fn), gt_area, pred_area, int(n_gt),
                                          detected, n_pred, n_fp)
    return np.array([rows[c] for c in PP_CONFIGS], dtype=np.int32)


def dice(tp: int, fp: int, fn: int) -> float | None:
    denom = 2 * tp + fp + fn
    return None if (tp + fn) == 0 else (2 * tp / denom if denom else 0.0)


# ----------------------------------------------------------------------------- main
def main() -> None:
    method = os.environ["METHOD"]
    folds = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
    out_dir = Path(os.environ.get("OUT_DIR", str(BONESCAN / "runs/final_eval")))
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_grid = bool(int(os.environ.get("THRESH_GRID", "0")))
    pp_grid = bool(int(os.environ.get("PP_GRID", "0")))
    keep_anatomy = pp_grid and method in ANATOMY_METHODS

    df = load_manifest(DATA_ROOT)
    recs = assemble_patient_records(df, DATA_ROOT)
    lesion = [r for r in recs if r.source in LESION_SRC]
    negatives = [r for r in recs if r.source in NEG_SRC]
    splits = k_fold_split(lesion, k=5, seed=0)

    if method == "bsciseg":
        dataset_cls = _bsciseg_module().BSciSegValDataset
    else:
        dataset_cls = BoneScanDataset

    for fold in folds:
        t0 = time.time()
        if method == "nnunet":
            predictor = NnUNetViewPredictor(fold)
        elif method == "ours_ens":
            predictor = EnsemblePredictor(fold, keep_anatomy=keep_anatomy)
        elif method == "ours_anat_kf":
            predictor = TwoViewPredictor("ours_anat", fold, kidney_filter=True, keep_anatomy=keep_anatomy)
        else:
            predictor = TwoViewPredictor(method, fold, keep_anatomy=keep_anatomy)
        val = [lesion[i] for i in splits[fold][1]]
        neg = [r for r in negatives if patient_id_hash(f"{r.source}_{r.patient_id}", 0) % 5 == fold]
        rows, masks = [], {}
        rows_by_t, probs = {f"{t:.2f}": [] for t in GRID}, {}
        pp_metrics, pp_meta = [], []
        for group, records in (("cv", val), ("neg_new", neg)):
            ds = dataset_cls(records, augment=False)
            for i, rec in enumerate(ds.records):
                item = ds[i]
                prob = predictor(rec) if method == "nnunet" else predictor(item["image"])
                gt = item["lesion_mask"].numpy() > 0
                pred = prob >= THRESHOLD
                for v, view in enumerate(("ANT", "POST")):
                    meta = {"group": group, "fold": fold, "source": rec.source, "patient_id": rec.patient_id,
                            "view": view, "hospital": "new" if rec.source in NEW_HOSPITAL else "old",
                            "has_lesion_mask": bool(rec.has_lesion_mask), "lesion_free_patient":
                            rec.source in NEG_SRC or rec.source.endswith("non-meta")}
                    m = view_metrics(pred[v], gt[v])
                    m.update(meta)
                    m["dice"] = dice(m["tp"], m["fp"], m["fn"])
                    rows.append(m)
                    key = f"{rec.source}|{rec.patient_id}|{view}"
                    masks[key] = np.packbits(pred[v], axis=None)
                    masks[f"{key}|shape"] = np.array(pred[v].shape)
                    if thr_grid:
                        for t in GRID:
                            mt = view_metrics(prob[v] >= t, gt[v])
                            mt.update(meta)
                            mt["dice"] = dice(mt["tp"], mt["fp"], mt["fn"])
                            rows_by_t[f"{t:.2f}"].append(mt)
                        probs[key] = prob[v].astype(np.float16)
                        probs[f"{key}|gt"] = np.packbits(gt[v], axis=None)
                    if pp_grid:
                        skeleton = None
                        if keep_anatomy:
                            labels = predictor.last_anatomy[v]
                            skeleton = (labels != 0) & (labels != KIDNEY_CLASS)
                        pp_metrics.append(pp_view_metrics(prob[v], gt[v], skeleton))
                        pp_meta.append(meta)
        out = {"method": method, "fold": fold, "threshold": THRESHOLD, "checkpoint": predictor.ckpt,
               "n_cv_patients": len(val), "n_neg_new_patients": len(neg), "rows": rows}
        (out_dir / f"{method}_fold{fold}.json").write_text(json.dumps(out))
        np.savez_compressed(out_dir / f"{method}_fold{fold}_pred.npz", **masks)
        if thr_grid:
            (out_dir / f"{method}_fold{fold}_thr.json").write_text(json.dumps(
                {"method": method, "fold": fold, "grid": GRID, "checkpoint": predictor.ckpt, "rows_by_t": rows_by_t}))
            np.savez_compressed(out_dir / f"{method}_fold{fold}_prob.npz", **probs)
        if pp_grid:
            np.savez_compressed(out_dir / f"{method}_fold{fold}_pp.npz", metrics=np.stack(pp_metrics),
                                configs=np.array(PP_CONFIGS, dtype=float), fields=np.array(PP_FIELDS),
                                meta=np.array(json.dumps(pp_meta)), has_skeleton=np.array(keep_anatomy),
                                checkpoint=np.array(predictor.ckpt))
        lesion_imgs = [r["dice"] for r in rows if r["group"] == "cv" and r["has_lesion_mask"] and r["dice"] is not None]
        print(f"[{method} fold{fold}] cv patients={len(val)} neg={len(neg)} lesion images={len(lesion_imgs)} "
              f"DSC_img={np.mean(lesion_imgs):.4f} ({time.time() - t0:.0f}s)", flush=True)
        del predictor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
