"""Real model outputs for one patient, used as thumbnails in the architecture figure (Figure 2).

Patient: the median case of Figure 4. Saves the anterior and mirrored posterior views, a masked anterior view
(60% of 16x16 patches set to zero, fixed seed), the MAE reconstruction of it, the lesion probability and the
anatomy label map of the network with the anatomy task, the probability of the network without it, and the
final mask of the ensemble (mean probability >= 0.5), all for the fold that scored this patient.

Usage (from the repository root): CUDA_VISIBLE_DEVICES=0 python code/final_eval/make_arch_thumbnails.py <out_npz>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_final as ef  # noqa: E402
from bonescan.mae import random_patch_mask  # noqa: E402
from bonescan.models import AnchorBSModel, BackboneConfig  # noqa: E402

cases = np.load("runs/final_main_cmp/agg/cases.npz")
meta = json.loads(str(cases["c1_meta"]))  # median case of Figure 4
recs = ef.assemble_patient_records(ef.load_manifest(ef.DATA_ROOT), ef.DATA_ROOT)
rec = next(r for r in recs if r.source == meta["source"] and r.patient_id == meta["patient_id"])
fold = meta["fold"]
item = ef.BoneScanDataset([rec], augment=False)[0]
image = item["image"]  # (2, H, W): anterior, mirrored posterior

anat = ef.TwoViewPredictor("ours_anat", fold, keep_anatomy=True)
p_anat = anat(image)
labels = anat.last_anatomy
p_noanat = ef.TwoViewPredictor("ours_noanat", fold)(image)
ensemble = (p_anat + p_noanat) / 2

torch.manual_seed(0)
mae = AnchorBSModel(BackboneConfig(), num_anatomy_classes=11, num_lesion_classes=2, enable_hbs=False,
                    enable_cvfm=False, enable_recon=True)
ck = torch.load("runs/mae_pretrain_bs80k/ckpt_ep099.pt", map_location="cpu", weights_only=False)
state = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
missing = mae.load_state_dict(state, strict=False)
print("MAE load: missing", len(missing.missing_keys), "unexpected", len(missing.unexpected_keys))
mae.to(ef.DEVICE).eval()
x = image[0:1].unsqueeze(0).to(ef.DEVICE)
mask = random_patch_mask(1, x.shape[2], x.shape[3], patch=16, mask_ratio=0.6, device=ef.DEVICE)
with torch.no_grad():
    recon = mae(x * mask, torch.zeros_like(x), anatomy_gt_ant=None, anatomy_gt_post=None, curriculum_alpha=0.0)["recon_ant"]

np.savez_compressed(sys.argv[1], ant=image[0].numpy(), post=image[1].numpy(), mask=mask[0, 0].cpu().numpy(),
                    recon=recon[0, 0].float().cpu().numpy(), p_anat=p_anat, labels=labels, p_noanat=p_noanat,
                    ensemble=ensemble, gt=(item["lesion_mask"].numpy() > 0), meta=np.array(json.dumps(meta)))
print("saved", sys.argv[1], meta, "fold", fold)
