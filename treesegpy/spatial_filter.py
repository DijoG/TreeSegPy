"""Spatial post-filter for TreeSegPy predictions.

Removes classifier false positives that are structurally dangerous to the
TopTreeSegR alpha-complex mesh, while leaving benign false positives untouched.

The filter is applied in three sequential stages:

    1. Confidence gate          -- keep points with tree_prob >= threshold
    2. Connected-component size -- drop tiny XY-connected islands
    3. Vertical support         -- drop components with no near-ground point

Stage 3 mirrors the criterion already used inside TopTreeSegR's
DiscreteMorseR::get_CCMESH(), but is applied *before* mesh construction, so
discarded points never perturb the Delaunay triangulation of nearby real trees.

Coordinate convention
---------------------
`xyz` may be patch-local (as produced by patch.py) or plot-global (as read
from a predicted LAZ). Only the XY coordinates are used for connected-
component linking, and only the Z coordinate is used for vertical support, so
both coordinate frames work as long as XY is internally consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ---- Defaults (match downstream_run.R and TopTreeSegR manuscript) ----
DEFAULT_CONFIDENCE     = 0.20   # tree_prob gate
DEFAULT_RADIUS_XY      = 0.50   # metres, XY linking distance for components
DEFAULT_MIN_COMPONENT  = 500    # points, minimum component size to keep
DEFAULT_STEM_HEIGHT    = 0.50   # metres, TopTreeSegR stem-height threshold
DEFAULT_SUPPORT_MARGIN = 0.20   # metres, slack above stem_height


# ----------------------------------------------------------------------
# Stage 1: confidence gate
# ----------------------------------------------------------------------
def filter_by_confidence(tree_prob, threshold=DEFAULT_CONFIDENCE):
    """Boolean mask: True where tree_prob >= threshold."""
    tree_prob = np.asarray(tree_prob, dtype=np.float32)
    return tree_prob >= float(threshold)


# ----------------------------------------------------------------------
# Stage 2 helpers: union-find over XY neighbour pairs
# ----------------------------------------------------------------------
class _UnionFind:
    """Tiny union-find with path compression. Used for connected components."""

    __slots__ = ("parent", "rank")

    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        # Iterative path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def connected_components_xy(xyz, radius=DEFAULT_RADIUS_XY):
    """Label XY-connected components via grid bucketing.

    Two points are linked if their XY distance is <= radius. Points are
    binned into a regular grid of cell size `radius`; two points can only be
    neighbours if they fall in the same cell or in one of the 8 surrounding
    cells. Union-find is run over cell pairs, then each point inherits its
    cell's component.

    Complexity: O(n) for binning + O(c) for cell-pair unions, where c is the
    number of occupied cells. Independent of point density.

    Grid-bucket approximation: two points just under `radius` apart in
    neighbouring cells are linked; two points just over `radius` apart in
    the same cell are also linked. The latter is a slight over-link, bounded
    by sqrt(2) * radius. This is acceptable for the filter's purpose
    (removing disconnected islands), and is the standard trade-off for
    avoiding O(n log n) neighbour queries.

    Returns int64 array of component labels (0-based, contiguous).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    xy = xyz[:, :2].astype(np.float64)
    r = float(radius)

    # Bin to grid
    origin = xy.min(axis=0)
    cell = np.floor((xy - origin) / r).astype(np.int64)
    cx, cy = cell[:, 0], cell[:, 1]

    # Map each unique (cx, cy) cell to a dense id
    _, cell_ids = np.unique(cell, axis=0, return_inverse=True)
    n_cells = int(cell_ids.max()) + 1

    # For each occupied cell, its (cx, cy) coordinate
    cell_coords = np.zeros((n_cells, 2), dtype=np.int64)
    cell_coords[cell_ids] = cell

    # Union occupied cells with their 4 "forward" neighbours to avoid
    # duplicate work: (+1,0), (0,+1), (+1,+1), (-1,+1)
    coord_to_id = {tuple(c): i for i, c in enumerate(cell_coords.tolist())}
    offsets = [(1, 0), (0, 1), (1, 1), (-1, 1)]

    uf = _UnionFind(n_cells)
    for i, (x, y) in enumerate(cell_coords.tolist()):
        for dx, dy in offsets:
            j = coord_to_id.get((x + dx, y + dy))
            if j is not None:
                uf.union(i, j)

    # Relabel cells to contiguous component ids
    cell_roots = np.array([uf.find(i) for i in range(n_cells)], dtype=np.int64)
    _, cell_comp = np.unique(cell_roots, return_inverse=True)

    # Each point inherits its cell's component
    return cell_comp[cell_ids].astype(np.int64)


# ----------------------------------------------------------------------
# Stage 2: size filter
# ----------------------------------------------------------------------
def filter_by_component_size(comp_labels, min_points=DEFAULT_MIN_COMPONENT):
    """Boolean mask: True for points in components with >= min_points members."""
    comp_labels = np.asarray(comp_labels, dtype=np.int64)
    n = len(comp_labels)
    if n == 0:
        return np.zeros(0, dtype=bool)

    counts = np.bincount(comp_labels)
    keep = counts >= int(min_points)
    return keep[comp_labels]


# ----------------------------------------------------------------------
# Stage 3: vertical support
# ----------------------------------------------------------------------
def filter_by_vertical_support(
    xyz,
    comp_labels,
    stem_height=DEFAULT_STEM_HEIGHT,
    support_margin=DEFAULT_SUPPORT_MARGIN,
):
    """Boolean mask: True for points in components whose min(z) is below
    stem_height + support_margin.

    Assumes `xyz` and `comp_labels` refer to the *same* set of points, i.e.
    comp_labels was computed on the full xyz (not on a masked subset). If you
    want to apply this stage after the size filter, recompute comp_labels on
    the surviving points first.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    comp_labels = np.asarray(comp_labels, dtype=np.int64)
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=bool)

    z = xyz[:, 2].astype(np.float64)

    # Per-component minimum z, vectorised with np.minimum.at
    n_comp = int(comp_labels.max()) + 1
    comp_min_z = np.full(n_comp, np.inf, dtype=np.float64)
    np.minimum.at(comp_min_z, comp_labels, z)

    threshold = float(stem_height) + float(support_margin)
    keep_comp = comp_min_z <= threshold
    return keep_comp[comp_labels]


# ----------------------------------------------------------------------
# Top-level: spatial_filter
# ----------------------------------------------------------------------
def spatial_filter(
    xyz,
    tree_prob,
    threshold=DEFAULT_CONFIDENCE,
    radius=DEFAULT_RADIUS_XY,
    min_points=DEFAULT_MIN_COMPONENT,
    stem_height=DEFAULT_STEM_HEIGHT,
    support_margin=DEFAULT_SUPPORT_MARGIN,
    return_diagnostics=False,
):
    """Apply the three-stage spatial filter.

    Parameters
    ----------
    xyz : (N, 3) float array
    tree_prob : (N,) float array
    threshold, radius, min_points, stem_height, support_margin : see defaults

    Returns
    -------
    keep : (N,) bool array
        True for points that survive all three stages.

    diagnostics : dict (only if return_diagnostics=True)
        Per-stage attrition counts and per-component summary statistics.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    tree_prob = np.asarray(tree_prob, dtype=np.float32)

    if len(xyz) != len(tree_prob):
        raise ValueError(
            f"xyz and tree_prob length mismatch: {len(xyz)} vs {len(tree_prob)}"
        )

    n_total = len(xyz)
    keep = np.zeros(n_total, dtype=bool)

    # Stage 1: confidence gate
    mask_conf = filter_by_confidence(tree_prob, threshold=threshold)
    if not mask_conf.any():
        if return_diagnostics:
            return keep, _diagnostics(n_total, mask_conf, None, None, None, None)
        return keep

    xyz_c = xyz[mask_conf]
    n_conf = len(xyz_c)

    # Stage 2: connected components on confidence-passing points
    comp_labels = connected_components_xy(xyz_c, radius=radius)
    mask_size = filter_by_component_size(comp_labels, min_points=min_points)
    n_size = int(mask_size.sum())

    if n_size == 0:
        if return_diagnostics:
            return keep, _diagnostics(
                n_total, mask_conf, comp_labels, mask_size, None, None
            )
        return keep

    # Stage 3: vertical support, computed on size-passing points.
    # Recompute components on the size-passing subset so that removing a
    # component does not silently change the min-z of a surviving one.
    xyz_s = xyz_c[mask_size]
    comp_labels_s = connected_components_xy(xyz_s, radius=radius)
    mask_support = filter_by_vertical_support(
        xyz_s, comp_labels_s,
        stem_height=stem_height,
        support_margin=support_margin,
    )
    n_support = int(mask_support.sum())

    # Map back to original indexing
    idx_conf = np.flatnonzero(mask_conf)
    idx_size = idx_conf[mask_size]
    idx_final = idx_size[mask_support]

    keep[idx_final] = True

    if return_diagnostics:
        diag = _diagnostics(
            n_total, mask_conf, comp_labels, mask_size,
            comp_labels_s, mask_support,
        )
        diag["final_count"] = int(keep.sum())
        return keep, diag
    return keep


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def _diagnostics(n_total, mask_conf, comp_labels, mask_size,
                 comp_labels_s, mask_support):
    """Build a diagnostics dict for logging / debugging."""
    d = {
        "n_input":           int(n_total),
        "n_after_conf":      int(mask_conf.sum()),
        "removed_conf":      int(n_total - mask_conf.sum()),
        "threshold_used":    None,  # filled by caller if needed
    }

    if comp_labels is not None:
        counts = np.bincount(comp_labels)
        d["n_components_pre_size"] = int(len(counts))
        d["comp_size_min"]         = int(counts.min())
        d["comp_size_median"]      = float(np.median(counts))
        d["comp_size_max"]         = int(counts.max())

    if mask_size is not None:
        d["n_after_size"] = int(mask_size.sum())
        d["removed_size"] = int(mask_conf.sum() - mask_size.sum())

    if comp_labels_s is not None and mask_support is not None:
        d["n_components_post_size"] = int(comp_labels_s.max()) + 1 if len(comp_labels_s) else 0
        d["n_after_support"]        = int(mask_support.sum())
        d["removed_support"]        = int(mask_size.sum() - mask_support.sum())

    return d