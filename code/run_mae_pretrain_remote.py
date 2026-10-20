"""Run MAE self-supervised pre-training on BS-80K whole-body views and in-house views.

The sources are listed in bonescan.mae.run_mae (manifest_filter=None); the paper used 7,900 single views.
Output: encoder checkpoint that can warm-start downstream lesion-segmentation models.

Modality match (NM bone scan → NM bone scan) is highest possible.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BONESCAN_ROOT = Path(os.environ.get("BONESCAN_ROOT", "."))  # folder with data/ and runs/
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(BONESCAN_ROOT / "data" / "standardized")))
SAVE_DIR = Path(os.environ.get("SAVE_DIR", str(BONESCAN_ROOT / "runs" / "mae_pretrain_bs80k")))

import bonescan.data as _bs_data
_bs_data.DEFAULT_OUT = DATA_ROOT

from bonescan.mae import run_mae


def main():
    print(f"[MAE] data={DATA_ROOT} save_dir={SAVE_DIR}")
    epochs = int(os.environ.get("EPOCHS", "100"))
    bs = int(os.environ.get("BATCH_SIZE", "8"))
    nw = int(os.environ.get("NUM_WORKERS", "8"))
    lr = float(os.environ.get("LR", "1e-3"))
    mask_ratio = float(os.environ.get("MASK_RATIO", "0.6"))
    out = run_mae(
        manifest_filter=None,  # default source list of bonescan.mae.run_mae (7,900 views in the paper)
        epochs=epochs,
        batch_size=bs,
        num_workers=nw,
        lr=lr,
        mask_ratio=mask_ratio,
        save_dir=SAVE_DIR,
    )
    print(f"[MAE done] save_dir={out['save_dir']}")
    print(f"[MAE history] last loss: {out['history']['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
