# Anatomy-aware dual-view segmentation of bone metastases on bone scintigraphy with masked autoencoder pre-training and partially labeled multi-task learning

Code for the manuscript of the same title (submitted to *Frontiers in Radiology*). It covers pre-training, training,
the two comparison methods, evaluation, statistics and figures. No data or trained weights are included.

## Overview

The method has two stages. **Stage 1** pre-trains a convolutional encoder as a masked autoencoder (MAE) on 7,900
unlabeled single views from BS-80K and the in-house scans: each view is divided into 16 × 16 patches, 60% of the
patches are set to zero, and a reconstruction decoder restores the view. **Stage 2** trains a dual-view network: one
encoder, shared by the anterior view and the mirrored posterior view, feeds a lesion decoder and an anatomy
(skeletal-region) decoder, both with deep supervision. Training mixes scans with lesion masks and scans with skeletal
labels, and each scan contributes only the loss of the labels it has. The final model averages the lesion
probabilities of two networks, one trained with and one without the anatomy task, each with horizontal-flip
test-time augmentation, and thresholds the mean at 0.5.

## Repository layout

```
configs/                       environment-variable files with the exact settings of each run
  mae_pretrain_bs80k.env       Stage 1 (MAE pre-training)
  final_ours_anat.env          Stage 2, with the anatomy task
  final_ours_noanat.env        Stage 2, without the anatomy task
  final_ours_anat_nomae.env    ablation: anatomy task, no MAE initialisation
  ablation_BSciSeg.env         BSci-Seg re-implementation
code/
  bonescan/                    package: data, augmentation, model, losses, training loop, MAE
  run_mae_pretrain_remote.py   Stage 1 entry point
  run_simplified_remote.py     Stage 2 entry point (all three networks)
  run_variant_BSciSeg_remote.py   BSci-Seg baseline (training; also imported by the evaluation)
  nnunet_convert_remote.py     export to nnU-Net v2 format (one case per view)
  nnunet_inject_splits_remote.py  write our folds into nnU-Net's splits_final.json
  final_eval/                  evaluation, statistics and secondary analyses (Python scripts and run_*.sh)
figures/
  fig1_study_workflow.drawio   Figure 1 source (draw.io)
  make_figure*.py              Figures 2-6 and Supplementary Figure S1
```

The entry scripts keep the names used for the experiments. All scripts read the project folder from the environment
variable `BONESCAN_ROOT` (default: the current directory) and expect `data/`, `checkpoints/`, `runs/` and `nnunet/`
inside it; run the commands below from the repository root, or symlink these folders into it.

## Requirements

- Python 3.11.15, PyTorch 2.12.0.dev20260407+cu128 (nightly build; CUDA 12.8, cuDNN 9.20), nnunetv2 2.7.0. The
  other packages are pinned in `requirements.txt`. Install a CUDA build of PyTorch first, then
  `pip install -r requirements.txt`. The code was not tested with other versions.
- One GPU per training process. The paper runs used bfloat16 autocast (`AMP_BF16=1`), which needs a GPU with
  bfloat16 support.
- Figures: draw.io desktop for Figures 1 and 2 (set `DRAWIO_EXE` to its executable) and the Times New Roman font.
  The figure scripts were run on a separate workstation with Python 3.12.7, numpy 1.26.4, scipy 1.13.1,
  matplotlib 3.9.2 and Pillow 12.3.0.

## Data

- **In-house data.** Scans from two hospitals (Hospital A and Hospital B). They cannot be shared because of patient
  privacy.
- **BS-80K.** Public dataset of Huang et al. (2022): Huang Z, Pu X, Tang G, Ping M, Jiang G, Wang M, Wei X, Ren Y.
  BS-80K: the first large open-access dataset of bone scan images. *Comput Biol Med.* 2022;151(Pt A):106221.
  doi: [10.1016/j.compbiomed.2022.106221](https://doi.org/10.1016/j.compbiomed.2022.106221). Obtain it from its
  authors as described in that publication. Only the whole-body views are used, for MAE pre-training.
- **Not included.** The conversion of the raw scans to the standardized arrays and `manifest.csv`, and the
  construction of the lesion libraries, are not part of this repository. Their formats are given below.

Expected layout:

```
$BONESCAN_ROOT/
  data/standardized/
    manifest.csv                 one row per view
    images/*.npy                 one view per file, float16, 1024 x 256 pixels, intensities in [0, 1]
    masks/lesion/*.npy           uint8 lesion masks (0 = background, 1 = lesion)
    masks/anatomy/*.npy          uint8 skeletal labels (0 = background, 1-10 = regions)
  checkpoints/_lesion_library_fold{0..4}.pkl   lesion crops for paired soft copy-paste (one file per fold)
  runs/                          created by training and evaluation
  nnunet/{raw,preprocessed,results}            created by the nnU-Net steps
```

`manifest.csv` columns read by `code/bonescan/data.py`: `source`, `patient_id`, `view` (`ANT` or `POST`),
`image_path`, `lesion_mask_path`, `anatomy_mask_path` (paths relative to `data/standardized/`, empty when absent),
`image_lesion_label`, `has_lesion_mask`, `has_anatomy_mask`, `has_lesion_bbox`. Anterior and posterior rows of the
same `source` and `patient_id` form one scan; the posterior view is mirrored when loaded.

| `source` | Patients | Role |
|---|---|---|
| `meta_breast_meta`, `meta_breast_non-meta`, `meta_prostate_meta`, `meta_prostate_non-meta` | 90, 10, 50, 50 | Hospital A lesion cohort (lesion masks); `non-meta` = without metastases |
| `new_mb`, `new_mp` | 50, 53 | Hospital B lesion cohort (lesion masks for most patients) |
| `new_bb`, `new_bp` | 22, 17 | Hospital B patients without metastases; scored, never used for supervised training |
| `old_breast`, `old_prostate` | 165, 196 | Hospital A anatomy cohort (skeletal labels only) |
| `bs80k_wb` | 3,247 | BS-80K whole-body views (MAE pre-training only) |

Each lesion library is a pickled list of dicts with `ant_crop` and `post_crop` (float32 crops of the same lesion in
both views, the posterior crop already mirrored), `ant_lesion_mask` and `post_lesion_mask` (uint8), and `source`
(used to filter by `LESION_SRC`); `patient_id`, `lesion_area`, `ant_bbox` and `post_bbox` are stored but not used by
the training code. The library of fold *k* must contain lesions from the training patients of fold *k* only.

Folds: `bonescan.data.k_fold_split` assigns each lesion-cohort patient to one of five folds by an MD5 hash of
`0::{source}_{patient_id}`, so both views of a patient stay in the same fold. Every method uses these folds; nnU-Net
receives them through `splits_final.json`. Patients of `new_bb` and `new_bp` are scored by the model of the fold
their hash maps to.

## Reproducing the experiments

Run from the repository root with the environment activated. Each `configs/*.env` file is loaded into the
environment of one process; `FOLDS` selects the fold.

### 1. Stage 1: MAE pre-training

```bash
mkdir -p runs
( set -a; . configs/mae_pretrain_bs80k.env; set +a
  CUDA_VISIBLE_DEVICES=0 python -u code/run_mae_pretrain_remote.py ) > runs/_MAE_pretrain.log 2>&1
```

This writes `runs/mae_pretrain_bs80k/ckpt_ep{009,...,099}.pt`; Stage 2 loads `ckpt_ep099.pt` (`MAE_CKPT`).

### 2. Stage 2: the three networks (five folds each)

```bash
for cfg in final_ours_anat final_ours_noanat final_ours_anat_nomae; do
  for k in 0 1 2 3 4; do
    ( set -a; . configs/$cfg.env; set +a
      CUDA_VISIBLE_DEVICES=0 FOLDS=$k python -u code/run_simplified_remote.py ) > runs/_${cfg}_f$k.log 2>&1
  done
done
```

Model: `AnchorBSModel(BackboneConfig(), enable_hbs=False, enable_cvfm=False, enable_xattn=False)` with 11 anatomy
classes and 2 lesion classes. Settings (from each run's `run_metadata.json` and launcher):

| Setting | `final_ours_anat` | `final_ours_noanat` | `final_ours_anat_nomae` |
|---|---|---|---|
| Anatomy task (`ANAT_SRC`) | `old_breast,old_prostate` | off (empty) | `old_breast,old_prostate` |
| Encoder initialisation (`WARM_START`) | MAE `ckpt_ep099.pt` | MAE `ckpt_ep099.pt` | none (random) |
| `LR` / `LR_ENC` (see note) | 1e-3 / 1e-4 | 1e-3 / 1e-4 | 1e-3 / 1e-3 |
| Batch size, epochs, passes per epoch (`EXPAND`) | 8, 24, 5 | 8, 24, 5 | 8, 24, 5 |
| Early stopping (patience / minimum epochs) | 8 / 10 | 8 / 10 | 8 / 10 |
| Augmentation, precision | `tier1_softcp`, bf16 | `tier1_softcp`, bf16 | `tier1_softcp`, bf16 |

Lesion sources (`LESION_SRC`) are the six lesion-cohort sources in all three runs. Fixed in the code: AdamW with
weight decay 5e-5, two linear warm-up epochs then cosine decay (`bonescan.training`), gradient clipping at 1.0,
lesion loss Dice + cross-entropy (1 : 1), anatomy loss Dice + cross-entropy (0.5 : 0.5), deep-supervision weights
1, 0.5, 0.25, 0.125, 0.0625, 0.0625, and the checkpoint with the best validation lesion Dice (`best.pt`).

### 3. BSci-Seg re-implementation

```bash
for k in 0 1 2 3 4; do
  ( set -a; . configs/ablation_BSciSeg.env; set +a
    CUDA_VISIBLE_DEVICES=0 FOLDS=$k python -u code/run_variant_BSciSeg_remote.py ) > runs/_BSciSeg_fold$k.log 2>&1
done
```

### 4. nnU-Net v2 (2d configuration, 250 epochs)

```bash
export nnUNet_raw=$PWD/nnunet/raw nnUNet_preprocessed=$PWD/nnunet/preprocessed nnUNet_results=$PWD/nnunet/results
python code/nnunet_convert_remote.py          # Dataset501_BoneScanLesion: one case per view, plus _case_to_fold.json
nnUNetv2_plan_and_preprocess -d 501
python code/nnunet_inject_splits_remote.py    # splits_final.json with the folds above
for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 2d $k -tr nnUNetTrainer_250epochs
done
```

The original runs set `nnUNet_n_proc_DA=5`. The exact planning flags were not recorded; the resulting plans contain
only the `2d` configuration. The evaluation uses `checkpoint_best.pth` of each fold.

### 5. Evaluation and statistics (`code/final_eval`)

Method keys: `ours_anat` and `ours_noanat` (single networks), `ours_ens` (final model: mean probability of the two),
`ours_anat_kf` (anatomy network with a kidney filter, secondary), `bsciseg`, `nnunet`. All methods are scored at a
fixed threshold of 0.5 with each method's own checkpoint rule; ours and BSci-Seg use horizontal-flip test-time
augmentation, nnU-Net its default mirroring.

```bash
export GPU=0
bash code/final_eval/run_all_evals.sh      # all methods and folds -> runs/final_eval (+ agg/)
bash code/final_eval/run_final_main.sh     # main comparison, final model as reference -> runs/final_main_cmp/agg
bash code/final_eval/run_mae_eval.sh       # MAE ablation -> runs/final_eval_nomae, runs/final_mae_cmp/agg
bash code/final_eval/run_cvthr.sh          # threshold grid + cross-validated threshold -> runs/final_eval_thr, runs/final_eval_cvthr
bash code/final_eval/run_ppcv.sh           # post-processing grid + cross-validated choice -> runs/final_eval_pp, runs/final_eval_ppcv
CUDA_VISIBLE_DEVICES=$GPU python code/final_eval/two_stage.py runs/final_eval_2stage
bash code/final_eval/run_two_stage_agg.sh
CUDA_VISIBLE_DEVICES=$GPU python code/final_eval/two_stage_nested.py runs/final_eval_nested
bash code/final_eval/run_nested_agg.sh     # needs runs/final_eval_cvthr
python code/final_eval/patient_auc.py runs/final_eval/agg/patient_auc.json     # needs runs/final_eval_thr
python code/final_eval/extract_cases.py runs/final_eval runs/final_main_cmp/agg/cases.npz
CUDA_VISIBLE_DEVICES=$GPU python code/final_eval/make_arch_thumbnails.py runs/final_main_cmp/agg/arch_thumbs.npz
CUDA_VISIBLE_DEVICES=$GPU python code/final_eval/model_complexity.py runs/final_main_cmp/agg/complexity.json
# exploratory analyses
python code/final_eval/fp_classifier_ceiling.py
python code/final_eval/lesion_size_by_hospital.py runs/final_main_cmp/agg/lesion_size.json
```

`run_thr_grid.sh` scores the threshold grid for selected `method:folds` only. In `runs/final_main_cmp` the final
model is stored under the key `ours_anat` (the reference of the comparison) and the single network with the anatomy
task under `ours_1stage`.

Statistics are computed in `aggregate_results.py` (image- and patient-level Dice, paired Wilcoxon signed-rank tests
of per-patient Dice with Holm adjustment, bootstrap 95% CIs with 5,000 resamples) and `patient_auc.py`
(patient-level AUC with 5,000 bootstrap resamples of patients). Outputs:

| File | Content |
|---|---|
| `runs/final_main_cmp/agg/summary.json` | final model compared with the single networks and the baselines |
| `runs/final_eval/agg/summary.json` | single networks and baselines |
| `runs/final_mae_cmp/agg/summary.json` | MAE ablation |
| `runs/final_eval/agg/patient_auc.json` | patient-level AUC |
| `runs/final_main_cmp/agg/complexity.json` | model size and inference cost |
| `runs/final_eval_cvthr/agg/summary.json`, `runs/final_eval_cvthr/cv_thresholds.json` | cross-validated threshold |
| `runs/final_eval_ppcv/agg/summary.json`, `runs/final_eval_ppcv/pp_choices.json` | cross-validated post-processing |
| `runs/final_eval_2stage_cmp/agg/summary.json`, `runs/final_eval_2stage/two_stage.json` | two-stage pipeline |
| `runs/final_eval_nested/nested.json`, `runs/final_eval_nested/fixed_summary.json` | nested selection of the output pipeline |

### 6. Figures

The figure scripts read `results/` and write `figures/`:

```bash
mkdir -p results/histories
cp runs/final_eval/agg/{per_patient_dsc.csv,lesion_free_fp.csv,patient_auc.json} results/
cp runs/final_main_cmp/agg/{cases.npz,arch_thumbs.npz,lesion_size.json} results/
for run in final_ours_anat final_ours_noanat final_ours_anat_nomae ablation_BSciSeg; do
  for k in 0 1 2 3 4; do cp runs/$run/fold_$k/history.json results/histories/${run}_fold$k.json; done
done
"$DRAWIO_EXE" --export --format png --scale 2 --border 10 \
  --output figures/Figure1_workflow.png figures/fig1_study_workflow.drawio
python figures/make_figure2_drawio.py .            # architecture, with real model outputs as thumbnails
python figures/make_figure3.py .                   # per-patient Dice and lesion-free scans
python figures/make_figure4.py . results/cases.npz # qualitative examples
python figures/make_figure5.py .                   # patient-level ROC curves
python figures/make_figure6.py .                   # lesion size by hospital (exploratory)
python figures/make_figureS1.py .                  # training curves
```

Figures 2 and 4 show patient images; `.gitignore` excludes the figure outputs and `results/`.

## Implementation notes

- Lesion prediction needs only the encoder and the lesion decoder; the anatomy decoder is a training-time head (HBS
  and CVFM are disabled). `model_complexity.py` reports size, operations, time and memory for this lesion path and
  checks that the lesion probability maps are identical to those of the full network used in the evaluation.
- `bonescan.training.fit` optimises all parameters in one AdamW group, so `LR_ENC` is recorded in
  `run_metadata.json` but not applied: the encoder trains with `LR` in every run, MAE-initialised or not.
- The learning rate follows two linear warm-up epochs and a cosine decay towards 5% of `LR` over `EPOCHS` epochs;
  early stopping can end training earlier. `best.pt` is updated when the validation lesion Dice improves by more than 5e-4.
- The curriculum value `α` printed in the training log only acts on the HBS module, which is disabled.
- The MAE loss is the plain mean squared error over the masked pixels.
- The MAE pre-training set contains the in-house views of all cohorts (images only, no labels), including the
  patients of every validation fold and the Hospital B patients without metastases.
- In the BSci-Seg re-implementation the patch-level dual-sampling contrastive term never becomes active (the encoder and
  decoder feature maps are paired in the wrong order), so the baseline is the architecture without that loss.
- The nnU-Net export gives lesion-cohort views without a lesion mask an empty mask.

## Citation

Manuscript under review; citation will be added on publication.

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
