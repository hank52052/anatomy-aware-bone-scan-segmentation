"""Inject our k_fold split into nnUNet's splits_final.json so fold_0 val set
matches our model's val set exactly (apples-to-apples comparison)."""
import json
import os
from pathlib import Path

DATASET_ID = int(os.environ.get("DATASET_ID", "501"))
BONESCAN_ROOT = Path(os.environ.get("BONESCAN_ROOT", "."))  # folder with nnunet/
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(BONESCAN_ROOT / "nnunet" / "raw")))
NNUNET_PREPROCESSED = Path(os.environ.get("nnUNet_preprocessed", str(BONESCAN_ROOT / "nnunet" / "preprocessed")))

ds_name = f"Dataset{DATASET_ID:03d}_BoneScanLesion"
case_map = json.loads((NNUNET_RAW / ds_name / "_case_to_fold.json").read_text())

# build per-fold val list
n_folds = max(case_map.values()) + 1
folds = [{"train": [], "val": []} for _ in range(n_folds)]
all_cases = sorted(case_map.keys())
for fold in range(n_folds):
    val = sorted([c for c, f in case_map.items() if f == fold])
    train = sorted([c for c, f in case_map.items() if f != fold])
    folds[fold]["train"] = train
    folds[fold]["val"] = val
    print(f"fold {fold}: train={len(train)} val={len(val)}")

out = NNUNET_PREPROCESSED / ds_name / "splits_final.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(folds, indent=2))
print(f"\n[OK] wrote {out}")
