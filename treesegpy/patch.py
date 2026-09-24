"""Patch extraction for TreeSegPy with multi-scale geometric features."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from .io import read_laz, write_laz, list_plots


# Multi-scale neighborhoods for geometric features.
# k=10 ≈ 0.1 m, k=30 ≈ 0.3 m, k=80 ≈ 1.0 m on typical TLS point spacing.
FEATURE_K = [10, 30, 80]

# Radius for height-below-canopy computation
CANOPY_RADIUS = 5.0


def _grid_centers_xy(xy, radius, stride, jitter=0.0, rng=None):
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    xs = np.arange(x_min, x_max + stride, stride)
    ys = np.arange(y_min, y_max + stride, stride)
    centers = []
    for x in xs:
        for y in ys:
            cx, cy = x, y
            if jitter > 0 and rng is not None:
                cx += rng.uniform(-jitter, jitter)
                cy += rng.uniform(-jitter, jitter)
            centers.append((cx, cy))
    return centers


def compute_geometric_features_multiscale(xyz, ks=FEATURE_K):
    """
    Compute geometric features at multiple neighborhood scales.

    Returns (N, 8 * len(ks)) float32 array.
    Columns grouped by scale:
        [linearity, planarity, sphericity, omnivariance,
         eigenentropy, anisotropy, verticality, density_k]
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    N = len(xyz)
    tree = cKDTree(xyz)

    feats_per_scale = []

    for k in ks:
        _, idx = tree.query(xyz, k=k, workers=-1)
        neigh = xyz[idx]
        neigh_mean = neigh.mean(axis=1, keepdims=True)
        centered = neigh - neigh_mean

        cov = np.einsum('nki,nkj->nij', centered, centered) / k
        eigvals = np.linalg.eigvalsh(cov)

        l3, l2, l1 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
        l1 = np.maximum(l1, 1e-12)

        linearity    = (l1 - l2) / l1
        planarity    = (l2 - l3) / l1
        sphericity   = l3 / l1
        omnivariance = np.cbrt(np.maximum(l1 * l2 * l3, 0))
        anisotropy   = (l1 - l3) / l1

        eigsum = np.maximum(l1 + l2 + l3, 1e-12)
        ev = np.stack([l1, l2, l3], axis=1) / eigsum[:, None]
        ev = np.clip(ev, 1e-12, 1)
        eigenentropy = -np.sum(ev * np.log(ev), axis=1)

        verticality = np.zeros(N, dtype=np.float32)
        chunk = 50000
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            _, vecs = np.linalg.eigh(cov[start:end])
            normals = vecs[:, :, 0]
            verticality[start:end] = 1.0 - np.abs(normals[:, 2])

        dist, _ = tree.query(xyz, k=k, workers=-1)
        density_k = dist[:, -1].astype(np.float32)

        scale_feats = np.stack([
            linearity, planarity, sphericity, omnivariance,
            eigenentropy, anisotropy, verticality, density_k,
        ], axis=1).astype(np.float32)

        feats_per_scale.append(scale_feats)

    return np.hstack(feats_per_scale).astype(np.float32)


def compute_height_below_canopy(xyz, radius=CANOPY_RADIUS, max_neighbors=200):
    """
    For each point, z minus the 95th percentile of z within `radius`.
    Vectorized with capped k-NN query.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    N = len(xyz)
    if N < 5:
        return np.zeros(N, dtype=np.float32)

    k = min(max_neighbors, N)
    tree = cKDTree(xyz[:, :2])  # XY only
    dist, idx = tree.query(xyz[:, :2], k=k, workers=-1)

    # Mask out neighbors beyond radius
    valid = dist <= radius

    # Gather z values: (N, k)
    z_all = xyz[:, 2]
    z_neigh = z_all[idx]              # (N, k)

    # Set out-of-radius values to -inf so percentile ignores them
    z_neigh = np.where(valid, z_neigh, -np.inf)

    # 95th percentile per point (uses valid neighbors only)
    z_top = np.percentile(z_neigh, 95, axis=1)
    z_top = np.where(np.isfinite(z_top), z_top, z_all)

    return (z_all - z_top).astype(np.float32)


def extract_patch(xyz, tree_id, intensity, center_xy, radius,
                  z_min=0.0, z_max=40.0,
                  max_points=50_000, rng=None):
    cx, cy = center_xy
    dx = xyz[:, 0] - cx
    dy = xyz[:, 1] - cy
    d2 = dx * dx + dy * dy

    in_xy = d2 <= radius * radius
    in_z  = (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    mask = in_xy & in_z

    if mask.sum() < 1000:
        return None

    patch_xyz       = xyz[mask]
    patch_tree_id   = tree_id[mask]
    patch_intensity = intensity[mask]

    if len(patch_xyz) > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(patch_xyz), size=max_points, replace=False)
        patch_xyz       = patch_xyz[idx]
        patch_tree_id   = patch_tree_id[idx]
        patch_intensity = patch_intensity[idx]

    patch_xyz_local = patch_xyz.copy()
    patch_xyz_local[:, 0] -= cx
    patch_xyz_local[:, 1] -= cy

    geom_feats = compute_geometric_features_multiscale(patch_xyz_local)
    h_below    = compute_height_below_canopy(patch_xyz_local)

    inten_med = np.median(patch_intensity) + 1e-6
    intensity_norm = (patch_intensity / inten_med).astype(np.float32)

    return {
        "xyz":            patch_xyz_local.astype(np.float32),
        "tree_id":        patch_tree_id.astype(np.int32),
        "label":          (patch_tree_id != 0).astype(np.uint8),
        "center":         np.array([cx, cy], dtype=np.float32),
        "geom_feats":     geom_feats,
        "intensity":      patch_intensity.astype(np.float32),
        "intensity_norm": intensity_norm,
        "h_below_canopy": h_below,
    }


def patch_plot(laz_path, radius=6.0, stride_train=8.0, stride_infer=10.0,
               jitter=1.5, max_points=50_000, seed=42):
    rng = np.random.default_rng(seed)
    data = read_laz(laz_path)
    xyz       = data["xyz"]
    tree_id   = data["tree_id"]
    intensity = data["intensity"]

    centers = _grid_centers_xy(xyz[:, :2], radius=radius,
                                stride=stride_train, jitter=jitter, rng=rng)

    patches = []
    for c in centers:
        p = extract_patch(xyz, tree_id, intensity, c, radius,
                          max_points=max_points, rng=rng)
        if p is not None:
            p["plot_name"] = Path(laz_path).stem
            patches.append(p)
    return patches


def save_patches(patches, out_dir, prefix=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, p in enumerate(patches):
        fname = out_dir / f"{prefix}{p['plot_name']}_patch{i:04d}.npz"
        np.savez_compressed(
            fname,
            xyz=p["xyz"],
            tree_id=p["tree_id"],
            label=p["label"],
            center=p["center"],
            geom_feats=p["geom_feats"],
            intensity=p["intensity"],
            intensity_norm=p["intensity_norm"],
            h_below_canopy=p["h_below_canopy"],
        )
        saved.append(fname)
    return saved