"""Convert standardized bone scan dataset → nnUNetv2 format.

Strategy:
- Each patient view (ANT or POST) = one nnUNet case (1-channel 2D image).
- Carry over our k_fold_split(lesion, k=5, seed=0) split-by-patient (ANT + POST of same
  patient go to the same fold) so we can compare apples-to-apples with our model.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BONESCAN_ROOT = Path(os.environ.get("BONESCAN_ROOT", "."))  # folder with data/ and nnunet/
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(BONESCAN_ROOT / "data" / "standardized")))
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(BONESCAN_ROOT / "nnunet" / "raw")))
DATASET_ID = int(os.environ.get("DATASET_ID", "501"))
DATASET_NAME = "BoneScanLesion"

import bonescan.data as _bs_data
_bs_data.DEFAULT_OUT = DATA_ROOT
from bonescan.data import assemble_patient_records, k_fold_split, load_manifest

LESION_SRC = ("new_mb", "new_mp", "meta_breast_meta", "meta_prostate_meta",
              "meta_breast_non-meta", "meta_prostate_non-meta")


def npy_to_nifti(arr2d: np.ndarray, out_path: Path):
    """Save 2D array as a single-slice NIfTI for nnUNet (H, W) → (H, W, 1)."""
    arr = np.asarray(arr2d, dtype=np.float32)
    # nnUNet 2D wants 3D NIfTI with single slice (or whatever, but we go with (1, H, W) layout)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D, got {arr.shape}")
    img = sitk.GetImageFromArray(arr[None, ...])  # (1, H, W)
    img.SetSpacing((1.0, 1.0, 999.0))  # arbitrary; nnUNet 2D ignores z spacing
    sitk.WriteImage(img, str(out_path))


def mask_to_nifti(mask2d: np.ndarray, out_path: Path):
    arr = (np.asarray(mask2d) > 0.5).astype(np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D mask, got {arr.shape}")
    img = sitk.GetImageFromArray(arr[None, ...])
    img.SetSpacing((1.0, 1.0, 999.0))
    sitk.WriteImage(img, str(out_path))


def main():
    print(f"DATA_ROOT={DATA_ROOT}")
    print(f"NNUNET_RAW={NNUNET_RAW}")

    df = load_manifest(DATA_ROOT)
    recs = assemble_patient_records(df, DATA_ROOT)
    lesion = [r for r in recs if r.source in LESION_SRC]
    splits = k_fold_split(lesion, k=5, seed=0)
    print(f"lesion patients = {len(lesion)} (5-fold)")

    ds_dir = NNUNET_RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    img_dir = ds_dir / "imagesTr"
    lbl_dir = ds_dir / "labelsTr"
    if ds_dir.exists():
        print(f"  ds_dir exists, removing: {ds_dir}")
        shutil.rmtree(ds_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    case_idx = 0
    case_to_fold = {}
    # All folds: union of train + val sample idxs is everything; we just need fold per patient.
    # k_fold_split returns (train_idx, val_idx) per fold; the val_idx of each fold is unique.
    pidx_to_fold = {}
    for fi, (tr, va) in enumerate(splits):
        for i in va:
            pidx_to_fold[i] = fi

    for pi, rec in enumerate(lesion):
        fold = pidx_to_fold[pi]
        # rec has .ant_img_path, .post_img_path (already preprocessed npy), .lesion_mask_ant_path, .lesion_mask_post_path
        for view in ["ANT", "POST"]:
            img_path = rec.ant_image_path if view == "ANT" else rec.post_image_path
            msk_path = rec.ant_lesion_mask if view == "ANT" else rec.post_lesion_mask
            if img_path is None or not Path(img_path).exists():
                print(f"  [skip] {rec.patient_id}_{view} missing img"); continue
            img = np.load(img_path)
            msk = np.load(msk_path) if (msk_path and Path(msk_path).exists()) else np.zeros_like(img)
            case_name = f"case_{case_idx:04d}"
            npy_to_nifti(img, img_dir / f"{case_name}_0000.nii.gz")
            mask_to_nifti(msk, lbl_dir / f"{case_name}.nii.gz")
            case_to_fold[case_name] = fold
            case_idx += 1

    # dataset.json (nnUNet v2 schema)
    dataset_json = {
        "channel_names": {"0": "intensity"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": case_idx,
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "Bone scan lesion segmentation (1-channel, ANT and POST as separate cases)",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    # Save case→fold mapping for our custom fold injection later
    (ds_dir / "_case_to_fold.json").write_text(json.dumps(case_to_fold, indent=2))

    print(f"\n[DONE] {case_idx} cases written to {ds_dir}")
    print(f"  imagesTr: {len(list(img_dir.glob('*.nii.gz')))} files")
    print(f"  labelsTr: {len(list(lbl_dir.glob('*.nii.gz')))} files")
    print(f"  fold distribution: ", {f: sum(1 for v in case_to_fold.values() if v == f) for f in range(5)})


if __name__ == "__main__":
    main()
