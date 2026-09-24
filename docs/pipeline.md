# Pipeline

TreeSegPy is the upstream half of a two-stage pipeline. This document describes
the end-to-end flow from a raw TLS point cloud to an instance-segmented mesh.

```text
raw TLS LAZ
    │
    ▼  external preprocessing (lidR)
normalised LAZ (decimated, ground-classified, z ∈ [0, 40] m)
    │
    ├──► patch.py :: patch_plot → save_patches  →  .npz patches  (training)
    │                                                    │
    │                                                    ▼
    │                                          train.py + batch.py + model.py
    │                                                    │
    │                                                    ▼
    │                                             best_f1.pt checkpoint
    │
    ▼  predict.py (chunked inference at threshold 0.10)
tree_prob + tree_label LAZ
    │
    ▼  spatial_filter.py (optional, recommended)
filtered tree-point LAZ
    │
    ▼  TopTreeSegR::TTS_pipeline()
instance-segmented mesh
```

## Stages

### 1. Preprocessing (external)

Not part of TreeSegPy. Uses `lidR` in R. Produces a normalised LAZ with an
`intensity` channel and, for training data, a `treeID` field.

### 2. Patch extraction (training only)

`treesegpy/patch.py` provides `patch_plot` and `save_patches`. Patches are
extracted on a regular XY grid with 8 m stride and 1.5 m random jitter, at 6 m
radius, filtered to z ∈ [0, 40] m, and capped at 50,000 points. Each patch is
written as `<plot>_patch<N>.npz`.

### 3. Training

`treesegpy/train.py` splits patches by plot (not by patch), builds batches via
`batch.py`, and trains the KPConv model from `model.py`. See the Training
section of the README for the command and hyperparameters.

### 4. Inference

`treesegpy/predict.py` tiles the plot into full-density 6 m patches with 5 m
stride, runs the model in chunks of up to 200,000 points, and averages
per-point probabilities across overlapping patches. See `inference.md`.

### 5. Spatial post-filter

`treesegpy/spatial_filter.py` applies three sequential stages:

1. **Confidence gate** — keep points with `tree_prob >= 0.20`
2. **Connected-component size** — drop XY-connected islands with fewer than
   500 points
3. **Vertical support** — drop components with no point below 0.70 m

Stage 3 mirrors TopTreeSegR's internal `get_CCMESH` criterion, but is applied
*before* mesh construction, so discarded points never perturb the Delaunay
triangulation of nearby real trees. This is the design rationale for having
the filter at all: a false positive that survives into the mesh can fragment
the trunk ring of a real tree several metres away.

The filter is optional. It is recommended when the classifier's false-positive
rate is expected to be non-trivial, and it is unnecessary when the input is
already clean.

### 6. Handoff to TopTreeSegR

The filtered tree-point LAZ is passed to `TopTreeSegR::TTS_pipeline()`, which
builds the α-complex mesh at α = 0.05 m, computes the Forman gradient,
identifies density-based seed minima below 0.5 m, traces ascending manifolds,
and assigns individual tree IDs. Bayesian Boundary Refinement is available as
an optional stage; the default in the current TopTreeSegR release is
`bbr = FALSE`.