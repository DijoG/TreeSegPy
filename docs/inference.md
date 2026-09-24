# Inference

## Single-plot inference

```bash
python -m treesegpy.predict \
  --input path/to/plot.laz \
  --checkpoint logs/cw_balanced/best_f1.pt \
  --output path/to/plot_pred.laz \
  --threshold 0.10
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Normalised input LAZ |
| `--checkpoint` | required | Trained model checkpoint |
| `--output` | None | Output LAZ (optional; if omitted, nothing is written) |
| `--radius` | 6.0 | Patch radius in metres |
| `--stride` | 5.0 | Patch stride in metres |
| `--plot-radius` | 15.0 | Plot radius; patch centres outside this are discarded |
| `--chunk-size` | 200,000 | Points per forward pass |
| `--threshold` | 0.5 | Tree probability threshold |

## How inference works

1. The plot is read with `io.read_laz`, producing `xyz`, `tree_id`, and
   `intensity`.
2. Full-density patches are extracted on a 6 m grid with 5 m stride, filtered
   to points inside the plot circle, at a minimum of 10,000 points per patch.
3. Each patch is processed in chunks of `--chunk-size` points. For each chunk,
   geometric features are computed, the model produces per-point probabilities,
   and the softmax tree-class probability is retained.
4. Per-point probabilities are accumulated across overlapping patches and
   divided by the number of patches that covered each point, giving a single
   averaged tree probability per point.
5. The tree label is assigned by thresholding at `--threshold`.

## Runtime

Inference runs at **full point density** — there is no subsampling step,
unlike training. A typical plot of ~8 million points produces ~20 patches of
~1 million points each.

Measured on the Milicz validation plot (7,920,264 points, 21 patches) on a
CUDA-capable workstation:

- **Total runtime: 1617 s (~27 minutes)**
- Coverage: 100% of input points
- Tree fraction at threshold 0.10: 54.3%

Runtime scales roughly linearly with point count. The `--chunk-size` argument
trades peak GPU memory against the number of forward passes; lowering it
reduces memory at some cost in speed.

## Output fields

The output LAZ preserves all fields of the input and adds:

- `tree_label` (uint8) — 0 for non-tree, 1 for tree, at the chosen threshold
- `tree_prob` (float32) — averaged tree probability per point

## Troubleshooting

- **ImportError on `compute_geometric_features`** — the `patch.py` module on
  your branch does not export that name. On `main` it does; on `multiscale` it
  was renamed to `compute_geometric_features_multiscale`.
- **Shape mismatch on the first conv layer** — the checkpoint's input
  dimension does not match the assembled feature vector. On `main`, both are
  10. On `multiscale`, both are 29. Do not mix checkpoints between branches.
- **Very low tree-point fraction** — check that the input is normalised to
  above-ground elevation and that the `intensity` channel is present.