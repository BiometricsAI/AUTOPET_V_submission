# Anatomy-Aware Promptable Segmentation with Online Interactive Training for AUTOPET V

**Team UAM — Universidad Autónoma de Madrid**

This repository contains our submission to the **AUTOPET V** interactive
lesion-segmentation challenge (MICCAI 2026): a promptable, anatomy-aware PET/CT
segmentation model with online interactive training, plus a lightweight tracer
classifier that routes each study to the most appropriate model at inference time.

---

## Overview

To define our proposed model we followed an **incremental methodology**, progressively
incorporating improvements in order to ultimately assemble a complete final system.
So that the model can produce a good initial segmentation and subsequently learn in a
meaningful, positive way from scribble interactions, training is carried out in **two phases**:

- **Phase 1 (pre-training).** Two input channels are provided to the network — the PET
  and CT studies. This phase enables the model to produce an accurate *initial*
  segmentation.
- **Phase 2 (interactive learning).** Prompts (scribbles) are added incrementally at each
  iteration, and the model learns *online* from its own errors at every step.

Our method is built as an **incremental family of models**, where each one extends the
previous with a single, well-motivated change, so the contribution of every design choice
can be evaluated in isolation. All are trained as Phase-1 backbones, and the best one
becomes the pre-trained starting point for the Phase-2 interactive stage.

### Phase-1 backbones (Versions 1–5)

| Version | Key change |
|---|---|
| **V1** | Baseline nnU-Net that stacks PET and CT as input channels. |
| **V2** | Modality-aware design: a Siamese encoder with separate PET, CT and joint decoders. |
| **V3** | Adds organ supervision through a shared lesion–organ head. |
| **V4** | Fuses V2 + V3: two independent modality-specific encoders feeding three task-specialized decoders (organ segmentation from CT, lesion detection from PET, refined lesion segmentation from the fused features). |
| **V5** | System level: since the acquisition tracer is not provided at inference, a lightweight **tracer classifier** routes each study either to the model trained on the combined **FDG+PSMA** data or to a variant trained **exclusively on PSMA**, following evidence that tracer-specific training benefits PSMA cases. |

### Phase-2 interactive stage

The best Phase-1 backbone is used as the pre-trained starting point and fine-tuned with
simulated scribble interactions. Guidance is injected as three extra channels
(`prev_mask`, `dist_fg`, `dist_bg`) built from the interaction points, and the model is
optimized to refine its prediction step by step. Two interactive variants are trained,
differing only in the Gaussian scribble encoding width **σ** (see naming below).

---

## Repository structure

```
.
├── custom_trainers/                     # nnU-Net trainers (Phase 1 and Phase 2)
│   ├── nnUNet3_PreTr.py                 #   Phase 1 backbone (pre-training)
│   ├── nnUNet_Interactive_sigma_10.py   #   Phase 2 interactive trainer, sigma = 10 (FDG+PSMA)
│   ├── nnUNet_Interactive_sigma_5.py    #   Phase 2 interactive trainer, sigma = 5  (PSMA-only)
│   └── simulate_scribbles.py            #   scribble simulation (imported by the trainers)
├── nnUNet_results/                      # model folders with plans.json / dataset.json
│                                        #   (checkpoints downloaded separately, see below)
├── process.py                           # container entry point (inference for one iteration)
├── tracer_classifier.py                 # FDG/PSMA classifier (MIP features + Random Forest)
├── rf_tracer_classifier_ds003.joblib    # trained Random Forest (model + feature order)
├── Dockerfile                           # container definition
├── requirements.txt                     # Python dependencies
├── build.sh                             # build the Docker image
├── test.sh                              # run the container locally on test/input
└── export.sh                            # save the image as a .tar.gz for submission
```

> **Naming convention.** Trainers whose name ends in **`_PreTr`** are **Phase-1**
> (pre-training) models; trainers named **`nnUNet_Interactive_sigma_*`** are the
> **Phase-2** interactive models. The suffix is the scribble Gaussian width σ used at that
> stage (`sigma_10` for the combined FDG+PSMA model, `sigma_5` for the PSMA-only model).

---

## Model weights

The trained checkpoints are **not included in this repository** due to their size. They
are hosted on Google Drive and must be downloaded and placed manually under
`nnUNet_results/`. The `plans.json` and `dataset.json` files are **already versioned in
this repository** inside each model folder, so **only the checkpoints (`.pth`) need to be
downloaded**.

**➡ [Download checkpoints (Google Drive)](https://drive.google.com/file/d/1tlMX7ru_Ra17hgGgKbQ_Vcwsewa1tNOv/view?usp=sharing)**

### Drive structure

On the Drive, weights are organized by experiment. Phase-1 (pre-training) models are in
folders (one checkpoint per fold, named `fold_0_checkpoint_best.pth`,
`fold_1_checkpoint_best.pth`, …), and Phase-2 (interactive) models are flat `.pth` files:

```
<drive_root>/
├── Dataset003_complete_PreTr1/                                        # Phase 1 — V1
├── Dataset003_complete_PreTr2/                                        # Phase 1 — V2
├── Dataset003_complete_PreTr3/                                        # Phase 1 — V3
├── Dataset003_complete_PreTr4/                                        # Phase 1 — V4
├── Dataset005_psma_PreTr3/                                            # Phase 1 — V3 (PSMA-only)
├── Dataset003_complete_Interactive_sigma10_fold_2_checkpoint_best.pth # Phase 2 (FDG+PSMA)
├── Dataset003_complete_Interactive_sigma5_fold_2_checkpoint_best.pth
├── Dataset005_complete_Interactive_sigma10_fold_3_checkpoint_best.pth
└── Dataset005_complete_Interactive_sigma5_fold_3_checkpoint_best.pth  # Phase 2 (PSMA-only)
```

### Where to put each checkpoint

nnU-Net expects each checkpoint inside a `fold_X/` subfolder and renamed to
`checkpoint_best.pth`. The `__nnUNetPlans__3d_fullres` folders (with their
`plans.json` / `dataset.json`) already exist in the repo; you only add the `fold_X/`
subfolder with the checkpoint.

**Phase 2 (interactive) — the two models the container actually uses:**

| Drive file | Destination in `nnUNet_results/` |
|---|---|
| `Dataset003_complete_Interactive_sigma10_fold_2_checkpoint_best.pth` | `Dataset003_complete/nnUNet_Interactive_sigma_10__nnUNetPlans__3d_fullres/fold_2/checkpoint_best.pth` |
| `Dataset005_complete_Interactive_sigma5_fold_3_checkpoint_best.pth` | `Dataset005_psma/nnUNet_Interactive_sigma_5__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth` |

> **Note:** on the Drive the PSMA interactive files are named `Dataset005_complete_...`,
> but the nnU-Net folder is `Dataset005_psma/` — map them as shown above.
>
> The other two interactive checkpoints (`…sigma5…fold_2` of Dataset003 and
> `…sigma10…fold_3` of Dataset005) are ablation variants; include them only to reproduce those experiments.

**Phase 1 (pre-training backbones) — for reproducing the Phase-1 stage:**

| Drive folder | Destination in `nnUNet_results/` |
|---|---|
| `Dataset003_complete_PreTr1/fold_X_checkpoint_best.pth` | `Dataset003_complete/nnUNet1_PreTr__nnUNetPlans__3d_fullres/fold_X/checkpoint_best.pth` |
| `Dataset003_complete_PreTr2/fold_X_checkpoint_best.pth` | `Dataset003_complete/nnUNet2_PreTr__nnUNetPlans__3d_fullres/fold_X/checkpoint_best.pth` |
| `Dataset003_complete_PreTr3/fold_X_checkpoint_best.pth` | `Dataset003_complete/nnUNet3_PreTr__nnUNetPlans__3d_fullres/fold_X/checkpoint_best.pth` |
| `Dataset003_complete_PreTr4/fold_X_checkpoint_best.pth` | `Dataset003_complete/nnUNet4_PreTr__nnUNetPlans__3d_fullres/fold_X/checkpoint_best.pth` |
| `Dataset005_psma_PreTr4/fold_X_checkpoint_best.pth` | `Dataset005_psma/nnUNet4_PreTr__nnUNetPlans__3d_fullres/fold_X/checkpoint_best.pth` |

For each Phase-1 checkpoint, create the matching `fold_X/` subfolder and rename the file to
`checkpoint_best.pth` (e.g. `fold_0_checkpoint_best.pth` → `fold_0/checkpoint_best.pth`).

The final layout the container needs is:

```
nnUNet_results/
├── Dataset003_complete/
│   └── nnUNet_Interactive_sigma_10__nnUNetPlans__3d_fullres/
│       ├── dataset.json          # already in the repo
│       ├── plans.json            # already in the repo
│       └── fold_2/checkpoint_best.pth      # from Drive
└── Dataset005_psma/
    └── nnUNet_Interactive_sigma_5__nnUNetPlans__3d_fullres/
        ├── dataset.json          # already in the repo
        ├── plans.json            # already in the repo
        └── fold_3/checkpoint_best.pth      # from Drive
```

### Requirements

Dependencies are installed inside the container from `requirements.txt`.

