"""Inference for TreeSegPy: label an entire LAZ with tree/non-tree."""
from pathlib import Path
import time

import numpy as np
import torch

from .config import make_config
from .batch import make_batch
from .io import read_laz, write_laz
from .model import TreeSegModel
from .patch import _grid_centers_xy, compute_geometric_features


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, config=None, device=None):
    if config is None:
        config = make_config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TreeSegModel(config).to(device)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "val_acc": ckpt.get("val_acc"),
        "tree_f1": ckpt.get("tree_f1"),
    }
    return model, config, device, meta


# ---------------------------------------------------------------------------
# Full-density patch extraction (no subsampling)
# ---------------------------------------------------------------------------

def extract_inference_patches_full(xyz, radius=6.0, stride=5.0, plot_radius=15.0):
    """
    Extract overlapping patches at full density. No subsampling.
    Patch centers are kept only if inside the plot circle.
    """
    rng = np.random.default_rng(0)
    all_centers = _grid_centers_xy(
        xyz[:, :2], radius=radius, stride=stride, jitter=0.0, rng=rng,
    )
    centers = [c for c in all_centers
               if np.sqrt(c[0] ** 2 + c[1] ** 2) <= plot_radius - radius * 0.5]

    patches = []
    for cx, cy in centers:
        dx = xyz[:, 0] - cx
        dy = xyz[:, 1] - cy
        d2 = dx * dx + dy * dy
        mask = (d2 <= radius * radius) & (xyz[:, 2] >= 0.0) & (xyz[:, 2] <= 40.0)
        if mask.sum() < 10000:
            continue
        local_idx = np.where(mask)[0]
        patch_xyz = xyz[local_idx].copy()
        patch_xyz[:, 0] -= cx
        patch_xyz[:, 1] -= cy
        patches.append({
            "xyz": patch_xyz.astype(np.float32),
            "center": np.array([cx, cy], dtype=np.float32),
            "local_idx": local_idx,
        })
    return patches


# ---------------------------------------------------------------------------
# Chunked inference on one patch
# ---------------------------------------------------------------------------

def predict_patch(patch_xyz, model, config, device, chunk_size=200_000):
    """
    Run inference on a full-density patch, splitting into chunks.
    Returns one probability per point in patch_xyz.
    """
    n = len(patch_xyz)
    probs = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = patch_xyz[start:end]

        geom = compute_geometric_features(chunk)
        batch = make_batch([{
            "xyz": chunk,
            "tree_id": np.zeros(len(chunk), dtype=np.int32),
            "geom_feats": geom,
        }], config).to(device)

        outputs = None
        with torch.no_grad():
            try:
                outputs = model(batch)
                p = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            except Exception as e:
                print(f"    warning: chunk forward failed ({e}); "
                      f"filling with 0.5", flush=True)
                p = np.full(end - start, 0.5, dtype=np.float32)
        probs[start:end] = p

        # Safe cleanup
        del batch, geom, chunk
        if outputs is not None:
            del outputs
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return probs


# ---------------------------------------------------------------------------
# Full-plot inference with chunked patches
# ---------------------------------------------------------------------------

def predict_plot(laz_path, model, config, device,
                 radius=6.0, stride=5.0, plot_radius=15.0,
                 chunk_size=200_000, verbose=True):
    """
    Full-plot inference with full-density patches and chunked forward passes.
    """
    data = read_laz(laz_path)
    xyz = data["xyz"]
    tree_id_gt = data.get("tree_id", np.zeros(len(xyz), dtype=np.int32))
    n = len(xyz)

    if verbose:
        print(f"  read {n:,} points from {Path(laz_path).name}", flush=True)

    patches = extract_inference_patches_full(
        xyz, radius=radius, stride=stride, plot_radius=plot_radius,
    )
    if verbose:
        print(f"  extracted {len(patches)} full-density patches", flush=True)

    prob_sum = np.zeros(n, dtype=np.float32)
    prob_cnt = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    for pi, p in enumerate(patches):
        probs = predict_patch(p["xyz"], model, config, device,
                              chunk_size=chunk_size)
        np.add.at(prob_sum, p["local_idx"], probs)
        np.add.at(prob_cnt, p["local_idx"], 1)

        if verbose and (pi % 3 == 0 or pi == len(patches) - 1):
            print(f"    patch {pi+1}/{len(patches)} "
                  f"({len(p['xyz']):,} pts, {time.time()-t0:.1f}s)",
                  flush=True)

    tree_prob = np.zeros(n, dtype=np.float32)
    seen = prob_cnt > 0
    tree_prob[seen] = prob_sum[seen] / prob_cnt[seen]
    coverage = float(seen.mean())

    if verbose:
        print(f"  inference done in {time.time()-t0:.1f}s", flush=True)
        print(f"  coverage: {seen.sum():,} / {n:,} points "
              f"({100*coverage:.1f}%)", flush=True)

    return xyz, tree_prob, tree_id_gt, len(patches), coverage


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def predict_laz(laz_path, checkpoint_path, output_path=None,
                radius=6.0, stride=5.0, plot_radius=15.0,
                chunk_size=200_000, threshold=0.5, verbose=True):
    """..."""
    model, config, device, meta = load_model(checkpoint_path)
    if verbose:
        print(f"loaded model from {checkpoint_path}")
        print(f"  epoch {meta['epoch']}, tree F1 {meta['tree_f1']:.4f}, "
              f"val_acc {meta['val_acc']:.4f}")

    xyz, tree_prob, tree_id_gt, n_patches, coverage = predict_plot(
        laz_path, model, config, device,
        radius=radius, stride=stride, plot_radius=plot_radius,
        chunk_size=chunk_size, verbose=verbose,
    )

    tree_label = (tree_prob >= threshold).astype(np.uint8)

    if output_path is not None:
        # Load the original LAZ to preserve its metadata
        import laspy
        original = laspy.read(laz_path)

        write_laz(
            output_path, xyz,
            tree_label=tree_label,
            tree_prob=tree_prob,
            original=original,
        )
        if verbose:
            n_tree = int(tree_label.sum())
            print(f"wrote {output_path}")
            print(f"  {n_tree:,} tree points "
                  f"({100 * n_tree / len(xyz):.1f}%)")

    return {
        "xyz": xyz,
        "tree_prob": tree_prob,
        "tree_label": tree_label,
        "tree_id_gt": tree_id_gt,
        "n_patches": n_patches,
        "coverage": coverage,
        "meta": meta,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--stride", type=float, default=5.0)
    parser.add_argument("--plot-radius", type=float, default=15.0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predict_laz(
        laz_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        radius=args.radius,
        stride=args.stride,
        plot_radius=args.plot_radius,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()