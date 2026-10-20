"""Dataset, manifest reader, splits for bone-scan project.

Notes on conventions
--------------------
- Images are stored as ``float16`` numpy arrays of shape (H=1024, W=256).
- Model input is two channels: stack ANT and POST of the same patient.
  We assemble these on-the-fly from the manifest (group by patient_id).
- Lesion mask: uint8 0/1 of same shape.
- Anatomy mask: uint8 0..10 of same shape (10 anatomy classes + background).
"""
from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import random_augment_2view

# ---------------------------------------------------------------------------
DEFAULT_OUT = Path(os.environ.get("BONESCAN_ROOT", ".")) / "data" / "standardized"  # entry scripts override this


# ---------------------------------------------------------------------------
@dataclass
class PatientRecord:
    """A patient = ANT image + POST image + optional lesion / anatomy masks."""
    patient_id: str
    source: str
    ant_image_path: Path
    post_image_path: Path
    ant_lesion_mask: Path | None = None
    post_lesion_mask: Path | None = None
    ant_anatomy_mask: Path | None = None
    post_anatomy_mask: Path | None = None
    image_lesion_label: int = -1
    has_lesion_mask: bool = False
    has_anatomy_mask: bool = False
    has_lesion_bbox: bool = False


def load_manifest(out_dir: Path = DEFAULT_OUT) -> pd.DataFrame:
    p = Path(out_dir) / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    return pd.read_csv(p, dtype={"patient_id": str,
                                 "lesion_mask_path": str,
                                 "anatomy_mask_path": str,
                                 "bbox_path": str,
                                 "image_path": str,
                                 "region": str},
                       keep_default_na=False)


# ---------------------------------------------------------------------------
def assemble_patient_records(df: pd.DataFrame, out_dir: Path = DEFAULT_OUT) -> list[PatientRecord]:
    """Group manifest rows by (source, patient_id) → PatientRecord with ANT+POST pair."""
    records: dict[tuple[str, str], dict] = defaultdict(dict)
    for _, r in df.iterrows():
        v = r["view"]
        if v not in ("ANT", "POST"):
            continue  # bs80k_crop has empty view; handled separately
        key = (r["source"], str(r["patient_id"]))
        entry = records[key]
        entry["source"] = r["source"]
        entry["patient_id"] = str(r["patient_id"])
        entry["image_lesion_label"] = max(entry.get("image_lesion_label", -1), int(r["image_lesion_label"]))
        side = v.lower()  # 'ant' / 'post'
        entry[f"{side}_image_path"] = Path(out_dir) / r["image_path"]
        lmp = r["lesion_mask_path"]
        if isinstance(lmp, str) and lmp:
            entry[f"{side}_lesion_mask"] = Path(out_dir) / lmp
        amp = r["anatomy_mask_path"]
        if isinstance(amp, str) and amp:
            entry[f"{side}_anatomy_mask"] = Path(out_dir) / amp
        entry["has_lesion_mask"] = entry.get("has_lesion_mask", False) or bool(r["has_lesion_mask"])
        entry["has_anatomy_mask"] = entry.get("has_anatomy_mask", False) or bool(r["has_anatomy_mask"])
        entry["has_lesion_bbox"] = entry.get("has_lesion_bbox", False) or bool(r["has_lesion_bbox"])

    out = []
    for (src, pid), e in records.items():
        if "ant_image_path" not in e or "post_image_path" not in e:
            continue  # only keep complete ANT+POST pairs
        out.append(PatientRecord(
            patient_id=e["patient_id"],
            source=e["source"],
            ant_image_path=e["ant_image_path"],
            post_image_path=e["post_image_path"],
            ant_lesion_mask=e.get("ant_lesion_mask"),
            post_lesion_mask=e.get("post_lesion_mask"),
            ant_anatomy_mask=e.get("ant_anatomy_mask"),
            post_anatomy_mask=e.get("post_anatomy_mask"),
            image_lesion_label=e["image_lesion_label"],
            has_lesion_mask=bool(e.get("has_lesion_mask", False)),
            has_anatomy_mask=bool(e.get("has_anatomy_mask", False)),
            has_lesion_bbox=bool(e.get("has_lesion_bbox", False)),
        ))
    out.sort(key=lambda r: (r.source, r.patient_id))
    return out


# ---------------------------------------------------------------------------
def patient_id_hash(pid: str, seed: int = 0) -> int:
    h = hashlib.md5(f"{seed}::{pid}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def k_fold_split(records: list[PatientRecord], k: int = 5, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """Patient-level k-fold split. Returns list of (train_idx, val_idx)."""
    folds = [[] for _ in range(k)]
    for i, r in enumerate(records):
        folds[patient_id_hash(f"{r.source}_{r.patient_id}", seed) % k].append(i)
    splits = []
    for k_test in range(k):
        val_idx = folds[k_test]
        train_idx = [i for k_train in range(k) if k_train != k_test for i in folds[k_train]]
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
class BoneScanDataset(Dataset):
    """Per-patient dataset returning (image, lesion_mask, anatomy_mask, meta).

    image: torch.FloatTensor (2, H, W) — channel 0 = ANT, channel 1 = POST
    lesion_mask: torch.LongTensor (2, H, W) or zeros if missing — same channel layout
    anatomy_mask: torch.LongTensor (2, H, W) or zeros if missing
    """

    def __init__(self,
                 records: list[PatientRecord],
                 require_lesion: bool = False,
                 require_anatomy: bool = False,
                 augment: bool = False,
                 augment_strength: str = "medium",
                 lesion_library: list | None = None,
                 anatomy_bone_ids: set | None = None,
                 mixup_enabled: bool = False):
        self.records = [r for r in records
                        if (not require_lesion or r.has_lesion_mask)
                        and (not require_anatomy or r.has_anatomy_mask)]
        self.augment = augment
        self.augment_strength = augment_strength
        # Tier 2 augmentation infrastructure
        self.lesion_library = lesion_library
        self.anatomy_bone_ids = anatomy_bone_ids
        self.mixup_enabled = mixup_enabled

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, path: Path | None, dtype) -> np.ndarray:
        if path is None or not path.exists():
            return None
        return np.load(str(path)).astype(dtype)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        ant = self._load(r.ant_image_path, np.float32)
        post = self._load(r.post_image_path, np.float32)
        ant_les = self._load(r.ant_lesion_mask, np.uint8)
        post_les = self._load(r.post_lesion_mask, np.uint8)
        ant_ana = self._load(r.ant_anatomy_mask, np.uint8)
        post_ana = self._load(r.post_anatomy_mask, np.uint8)

        H, W = ant.shape if ant is not None else (1024, 256)
        zeros = np.zeros((H, W), dtype=np.uint8)
        if ant_les is None: ant_les = zeros.copy()
        if post_les is None: post_les = zeros.copy()
        if ant_ana is None: ant_ana = zeros.copy()
        if post_ana is None: post_ana = zeros.copy()

        # POST horizontal flip — align anatomical L/R with ANT so cross-view attention
        # operates on corresponding anatomy at the same (i,j). Raw acquisition stores
        # ANT|POST stacked horizontally; without flip, same column = opposite body side.
        if post is not None:
            post = post[:, ::-1].copy()
            post_les = post_les[:, ::-1].copy()
            post_ana = post_ana[:, ::-1].copy()

        if self.augment:
            mixup_partner = None
            if self.mixup_enabled and len(self.records) > 1:
                # pick another random record as mixup partner
                jj = (idx + int(np.random.randint(1, len(self.records)))) % len(self.records)
                rj = self.records[jj]
                aj = self._load(rj.ant_image_path, np.float32)
                pj = self._load(rj.post_image_path, np.float32)
                if aj is not None and pj is not None and aj.shape == ant.shape:
                    mixup_partner = (aj, pj)
            (ant, ant_les, ant_ana), (post, post_les, post_ana) = random_augment_2view(
                ant, ant_les, ant_ana, post, post_les, post_ana,
                strength=self.augment_strength,
                lesion_library=self.lesion_library,
                anatomy_bone_ids=self.anatomy_bone_ids,
                mixup_partner=mixup_partner)

        image = np.stack([ant, post], axis=0).astype(np.float32)
        lesion = np.stack([ant_les, post_les], axis=0).astype(np.int64)
        anatomy = np.stack([ant_ana, post_ana], axis=0).astype(np.int64)

        return {
            "image": torch.from_numpy(image),
            "lesion_mask": torch.from_numpy(lesion),
            "anatomy_mask": torch.from_numpy(anatomy),
            "has_lesion_mask": torch.tensor(int(r.has_lesion_mask)),
            "has_anatomy_mask": torch.tensor(int(r.has_anatomy_mask)),
            "image_lesion_label": torch.tensor(int(r.image_lesion_label)),
            "patient_id": r.patient_id,
            "source": r.source,
        }


# ---------------------------------------------------------------------------
def subset_by_source(records: list[PatientRecord], sources: Iterable[str]) -> list[PatientRecord]:
    s = set(sources)
    return [r for r in records if r.source in s]


def filter_lesion_labeled(records: list[PatientRecord]) -> list[PatientRecord]:
    return [r for r in records if r.has_lesion_mask]


def filter_anatomy_labeled(records: list[PatientRecord]) -> list[PatientRecord]:
    return [r for r in records if r.has_anatomy_mask]


def filter_healthy_controls(records: list[PatientRecord]) -> list[PatientRecord]:
    """Patients guaranteed lesion-free: bb, bp, breast-non-meta, prostate-non-meta."""
    keep = {"new_bb", "new_bp", "meta_breast_non-meta", "meta_prostate_non-meta"}
    return [r for r in records if r.source in keep]
