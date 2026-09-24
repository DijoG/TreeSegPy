# TreeSegPy

A KPConv-based point classifier for tree/non-tree segmentation in terrestrial
laser scanning (TLS) forest point clouds. TreeSegPy is the upstream half of a
two-stage pipeline for individual tree segmentation: it assigns each TLS point
a tree/non-tree label, and the filtered cloud is then passed to a mesh-based
topological segmenter (TopTreeSegR) for instance segmentation.

This repository accompanies the manuscript *[will be added after publication]*, which evaluates TreeSegPy
together with TopTreeSegR end to end on five held-out plots from the
TreeScanPL10k dataset.

**Branch note.** The `main` branch reproduces the configuration reported in
the manuscript: a 10-dimensional feature vector and the reference checkpoint
`logs/cw_balanced/best_f1.pt`. An experimental 29-dimensional multi-scale
configuration is preserved on the `multiscale` branch and is not part of the
published results.

## Features

- KPConv-FCNN architecture with a four-level strided encoder and a four-level
  nearest-neighbor upsampling decoder
- Ten-dimensional per-point feature vector combining eight geometric features,
  a bias channel, and normalized height
- Patch-based training and inference with overlapping spherical patches
- Batch inference script for running a trained model over multiple plots
- Threshold sweep utility for selecting the operational decision threshold

## Requirements

- Python 3.10
- PyTorch (with CUDA for GPU inference)
- A compiled KPConv extension. The build script is provided at
  `scripts/build_kpconv.sh`.

Full dependency list: see `scripts/env_frozen.txt`.

## Installation

Clone the repository:
```bash
git clone https://github.com/DijoG/TreeSegPy.git
cd TreeSegPy
```
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/env_frozen.txt
```
Build the KPConv C++/CUDA extensions:
```bash
bash scripts/build_kpconv.sh
```
**Important**: the build must be run in the same environment in which you will
run inference. The compiled .so files are environment-specific.

## Input data

`TreeSegPy` expects normalised LAZ/LAS point clouds with an intensity channel.
Preprocessing follows the sequence used in the accompanying study:
- Spatial decimation to 10,000 points/m²
- Ground classification using the Cloth Simulation Filter
- Height normalization to above-ground elevation using TIN interpolation
- Retention of points within z ∈ [0, 40] m

The z channel must be above-ground elevation, not ellipsoidal height.

## Run inference

Download the reference checkpoint from the GitHub Release:
```bash
mkdir -p logs/cw_balanced
wget https://github.com/DijoG/TreeSegPy/releases/download/v1.0/best_f1.pt \
  -O logs/cw_balanced/best_f1.pt
```
Run inference on a single plot:
```bash 
python -m treesegpy.predict \
  --input path/to/plot.laz \
  --checkpoint logs/cw_balanced/best_f1.pt \
  --output path/to/plot_pred.laz \
  --threshold 0.10
```
The output is a LAZ file with two added fields: `tree_label` (uint8, 0 or 1)
and `tree_prob` (float32, the per-point tree probability). Points with
probability above the threshold are labelled tree.

The threshold of 0.10 is the F1-optimal value reported in the study and is
consistent across all five held-out validation plots, so no per-plot
calibration is required. The classifier is tuned for recall over precision,
because the downstream mesh segmenter tolerates extra tree points better than
missing trunk points; on a pure pine plot, roughly half of all returns are
labelled tree.

## Batch inference

To run inference over a list of plots and produce a threshold sweep for each:
```bash
bash scripts/run_inference_batch.sh
```
Edit the `PLOTS` array in the script before running, or override the input and
output directories with environment variables:
```bash
TREESEGPY_IN_DIR=/my/data TREESEGPY_OUT_DIR=/my/preds \
  bash scripts/run_inference_batch.sh
```
The script invokes `treesegpy.predict` for each plot and then
`scripts/threshold_sweep.py`, which computes precision, recall, and F1 across a
range of probability thresholds and writes them to `threshold_sweep.csv` next
to the predictions. To write the CSV elsewhere, call the sweep manually with a
third argument:
```bash
python scripts/threshold_sweep.py PRED_LAZ GT_LAZ /path/to/output.csv
```
The sweep is diagnostic only — it does not change the operational threshold,
which remains 0.10.

##  Training 
Training operates on pre-extracted `.npz` patches. Generate patches from a
normalised LAZ with `treesegpy.patch.patch_plot` and `save_patches`, then run:

```bash
python -m treesegpy.train \
  --patch-dir ~/projects/TreeSegPy/data/patches_geom \
  --checkpoint-dir ~/projects/TreeSegPy/logs \
  --batch-num 2 \
  --max-epoch 40 \
  --lr 0.01 \
  --val-fraction 0.2 \
  --seed 42 \
  --num-workers 0
```
Command-line arguments:

| Argument | Default | Description |
|---|---|---|
| `--patch-dir` | `~/projects/TreeSegPy/data/patches_geom` | Directory containing `.npz` patch files |
| `--checkpoint-dir` | `~/projects/TreeSegPy/logs` | Output directory for checkpoints and metrics |
| `--batch-num` | 2 | Number of patches per batch |
| `--max-epoch` | 40 | Maximum number of training epochs |
| `--lr` | 0.01 | Initial learning rate |
| `--val-fraction` | 0.2 | Fraction of plots held out for validation |
| `--seed` | 42 | Random seed for plot-level splitting |
| `--num-workers` | 0 | DataLoader workers (0 is required on some systems because the KPConv C++ extensions are not fork-safe) |

Training uses SGD with momentum 0.98 and weight decay 1e-6, a cosine annealing
learning rate schedule from the initial learning rate to 1e-4, and gradient
clipping at norm 100. The loss is cross-entropy with balanced class weights
[0.5, 0.5] for non-tree and tree. Early stopping is triggered after six epochs
without improvement in validation tree-class F1.

Checkpoints are written to the checkpoint directory as `last.pt`, `best.pt`
(lowest validation loss), `best_f1.pt` (highest tree-class F1), and
`epoch_NNN.pt` per epoch. A `metrics.csv` file records per-epoch training and
validation metrics.

## Repository structure

```text
treesegpy/            Core Python package
  config.py             Configuration for the KPConv model architecture
  model.py              Deformable KPConv model and loss
  dataset.py            Patch dataset, batching, and collation
  patch.py              Patch extraction utilities
  io.py                 LAZ/LAS reading and writing
  spatial_filter.py     Spatial filtering utilities
  batch.py              Batch inference helpers
  predict.py            Inference entry point
  train.py              Training entry point
scripts/              Shell and Python utilities
  build_kpconv.sh       Build the KPConv C++/CUDA extension
  run_inference_batch.sh   Batch inference over multiple plots
  threshold_sweep.py    Threshold sweep evaluation
  fix_laz_headers.py    LAZ header repair utility
  preprocess_laz.txt    Preprocessing notes
  env_frozen.txt        Pinned dependency list
kernels/              KPConv kernel point dispositions
src/KPConv-PyTorch/   Vendored KPConv-PyTorch source
```
## Documentation

- `docs/architecture.md` — model architecture and feature vector
- `docs/pipeline.md` — end-to-end flow and spatial post-filter
- `docs/inference.md` — inference modes, runtime, troubleshooting

## Citation

If you use TreeSegPy in your work, please cite:
*[will be added after the paper is published]*

## License

MIT License. See `LICENSE` for details.

## Related work

TopTreeSegR — the downstream mesh-based topological segmenter:
https://github.com/DijoG/TopTreeSegR

TreeScanPL10k — the dataset used in the accompanying study:
Stereńczak, K., Kulicki, M., Kraszewski, B. et al. A Central 
European tree species dataset of annotated terrestrial laser 
scanning point clouds - TreeScanPL10K. Sci Data 13, 1217 (2026). 
https://doi.org/10.1038/s41597-026-07269-1
