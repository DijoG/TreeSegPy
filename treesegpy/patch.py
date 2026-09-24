"""Patch extraction for TreeSegPy with geometric features."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from .io import read_laz, write_laz, list_plots


# Radius for local neighborhood in the geometric feature computation.
# 0.3 m captures trunk cross-sections and small branches but not whole crowns.
FEATURE_RADIUS = 0.3


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


def compute_geometric_features(xyz, radius=0.3, max_neighbors=20):
    """
    Fast vectorized geometric feature computation.

    Uses fixed-k nearest neighbors (k=15) instead of radius search,
    and batches the covariance computation.
    """
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, dtype=np.float32)
    N = len(xyz)
    k = 15

    tree = cKDTree(xyz)
    # query returns (N, k) indices and distances
    dist, idx = tree.query(xyz, k=k, workers=-1)  # parallel query

    # Gather neighbor coordinates: (N, k, 3)
    neigh = xyz[idx]  # (N, k, 3)

    # Center each neighborhood
    neigh_mean = neigh.mean(axis=1, keepdims=True)  # (N, 1, 3)
    centered = neigh - neigh_mean                   # (N, k, 3)

    # Batch covariance: (N, 3, 3)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k

    # Batch eigenvalues: np.linalg.eigvalsh works on stacked matrices
    eigvals = np.linalg.eigvalsh(cov)  # (N, 3) ascending

    l3, l2, l1 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    l1 = np.maximum(l1, 1e-12)

    linearity     = (l1 - l2) / l1
    planarity     = (l2 - l3) / l1
    sphericity    = l3 / l1
    omnivariance  = np.cbrt(np.maximum(l1 * l2 * l3, 0))
    anisotropy    = (l1 - l3) / l1

    # Eigenentropy
    eigsum = l1 + l2 + l3
    eigsum = np.maximum(eigsum, 1e-12)
    ev = np.stack([l1, l2, l3], axis=1) / eigsum[:, None]
    ev = np.clip(ev, 1e-12, 1)
    eigenentropy = -np.sum(ev * np.log(ev), axis=1)

    # Verticality — need the eigenvector for l3. Batch eigvecs is slower,
    # so compute it in chunks.
    # For now, approximate verticality from coordinate spread.
    # Alternatively, run np.linalg.eigh in chunks below.
    verticality = np.zeros(N, dtype=np.float32)
    chunk = 50000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        _, vecs = np.linalg.eigh(cov[start:end])
        # smallest eigenvalue's eigenvector is column 0
        normals = vecs[:, :, 0]  # (chunk, 3)
        verticality[start:end] = 1.0 - np.abs(normals[:, 2])

    # Distance to the k-th neighbor as a density proxy
    density = dist[:, -1].astype(np.float32)

    return np.stack([
        linearity, planarity, sphericity, omnivariance,
        eigenentropy, anisotropy, verticality, density,
    ], axis=1).astype(np.float32)


def extract_patch(xyz, tree_id, center_xy, radius,
                  z_min=0.0, z_max=40.0,
                  max_points=50_000, rng=None):
    cx, cy = center_xy
    dx = xyz[:, 0] - cx
    dy = xyz[:, 1] - cy
    d2 = dx * dx + dy * dy

    in_xy = d2 <= radius * radius
    in_z = (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    mask = in_xy & in_z

    if mask.sum() < 1000:
        return None

    patch_xyz = xyz[mask]
    patch_tree_id = tree_id[mask]

    if len(patch_xyz) > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(patch_xyz), size=max_points, replace=False)
        patch_xyz = patch_xyz[idx]
        patch_tree_id = patch_tree_id[idx]

    patch_xyz_local = patch_xyz.copy()
    patch_xyz_local[:, 0] -= cx
    patch_xyz_local[:, 1] -= cy

    # Compute features on the small patch, not the whole plot
    geom_feats = compute_geometric_features(patch_xyz_local)

    return {
        "xyz": patch_xyz_local.astype(np.float32),
        "tree_id": patch_tree_id.astype(np.int32),
        "label": (patch_tree_id != 0).astype(np.uint8),
        "center": np.array([cx, cy], dtype=np.float32),
        "geom_feats": geom_feats,
    }


def patch_plot(laz_path, radius=6.0, stride_train=8.0, stride_infer=10.0,
               jitter=1.5, max_points=50_000, seed=42):
    rng = np.random.default_rng(seed)
    data = read_laz(laz_path)
    xyz = data["xyz"]
    tree_id = data["tree_id"]

    centers = _grid_centers_xy(xyz[:, :2], radius=radius,
                                stride=stride_train, jitter=jitter, rng=rng)

    patches = []
    for c in centers:
        p = extract_patch(xyz, tree_id, c, radius,
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
        save_dict = {
            "xyz": p["xyz"],
            "tree_id": p["tree_id"],
            "label": p["label"],
            "center": p["center"],
        }
        if "geom_feats" in p:
            save_dict["geom_feats"] = p["geom_feats"]
        np.savez_compressed(fname, **save_dict)
        saved.append(fname)
    return saved