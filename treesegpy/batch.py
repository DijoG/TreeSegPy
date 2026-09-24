"""
Batch construction for TreeSegPy.

Builds a KPConv-compatible Batch object from a list of .npz patches.
Replaces the S3DIS PointCloudDataset.segmentation_inputs pipeline with a
minimal, self-contained equivalent.

The Batch object mirrors S3DISCustomBatch: attributes are lists of tensors
(one per layer) for points, neighbors, pools, upsamples, lengths, plus
tensors for features, labels, and a few unused bookkeeping fields.
"""
from pathlib import Path
import sys
import numpy as np
import torch

# ---- Make the KPConv repo importable BEFORE importing from it ----
KPConv_ROOT = Path(__file__).resolve().parents[1] / "src" / "KPConv-PyTorch"
if str(KPConv_ROOT) not in sys.path:
    sys.path.insert(0, str(KPConv_ROOT))

from datasets.common import PointCloudDataset, batch_neighbors, batch_grid_subsampling

class TreeScanBatch:
    """Minimal batch container compatible with KPFCNN.forward."""

    def __init__(self):
        self.points = []
        self.neighbors = []
        self.pools = []
        self.upsamples = []
        self.lengths = []
        self.features = None
        self.labels = None
        # Bookkeeping fields: unused during forward, set to dummies
        self.scales = torch.ones(1, 3)
        self.rots = torch.eye(3).unsqueeze(0)
        self.cloud_inds = torch.zeros(1, dtype=torch.int32)
        self.center_inds = torch.zeros(1, dtype=torch.int32)
        self.input_inds = torch.zeros(1, dtype=torch.int64)

    def pin_memory(self):
        self.points = [t.pin_memory() for t in self.points]
        self.neighbors = [t.pin_memory() for t in self.neighbors]
        self.pools = [t.pin_memory() for t in self.pools]
        self.upsamples = [t.pin_memory() for t in self.upsamples]
        self.lengths = [t.pin_memory() for t in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def to(self, device):
        self.points = [t.to(device) for t in self.points]
        self.neighbors = [t.to(device) for t in self.neighbors]
        self.pools = [t.to(device) for t in self.pools]
        self.upsamples = [t.to(device) for t in self.upsamples]
        self.lengths = [t.to(device) for t in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        return self


from datasets.common import PointCloudDataset

class _SegHelper(PointCloudDataset):
    def __init__(self, config):
        super().__init__("helper")
        self.config = config
        self.neighborhood_limits = list(config.max_neighbors_per_layer)
        
def _build_layer_pyramid(points, features, labels, lengths, config):
    helper = _SegHelper(config)
    li = helper.segmentation_inputs(points, features, labels, lengths)

    batch = TreeScanBatch()
    L = (len(li) - 2) // 5
    ind = 0
    batch.points = [torch.from_numpy(a) for a in li[ind:ind+L]]; ind += L
    batch.neighbors = [torch.from_numpy(a) for a in li[ind:ind+L]]; ind += L
    batch.pools = [torch.from_numpy(a) for a in li[ind:ind+L]]; ind += L
    batch.upsamples = [torch.from_numpy(a) for a in li[ind:ind+L]]; ind += L
    batch.lengths = [torch.from_numpy(a) for a in li[ind:ind+L]]; ind += L
    batch.features = torch.from_numpy(li[ind]); ind += 1
    batch.labels = torch.from_numpy(li[ind]).long()
    return batch

def make_batch(patches, config):
    """
    Build a KPConv-compatible batch from a list of patch dicts.

    Each patch dict must contain:
        xyz     : (Ni, 3) float32, patch-centered coordinates
        tree_id : (Ni,)   int32,   0 = non-tree, >0 = tree
    """
    xyz_list = []
    feat_list = []
    lab_list = []
    lengths = []

    for p in patches:
        xyz = p["xyz"].astype(np.float32)
        tree_id = p["tree_id"].astype(np.int32)

        # Feature vector: [bias, z, geometric features]
                # Feature vector:
        #   [bias, z, geom_multiscale(24), intensity,
        #    intensity_norm, h_below_canopy]
        ones = np.ones((len(xyz), 1), dtype=np.float32)
        z = xyz[:, 2:3].astype(np.float32)

        cols = [ones, z]
        if "geom_feats" in p:
            cols.append(p["geom_feats"].astype(np.float32))
        if "intensity" in p:
            cols.append(p["intensity"].reshape(-1, 1).astype(np.float32))
        if "intensity_norm" in p:
            cols.append(p["intensity_norm"].reshape(-1, 1).astype(np.float32))
        if "h_below_canopy" in p:
            cols.append(p["h_below_canopy"].reshape(-1, 1).astype(np.float32))

        feats = np.hstack(cols).astype(np.float32)

        # Binary label: 0 = non-tree, 1 = tree
        labels = (tree_id != 0).astype(np.int32)

        xyz_list.append(xyz)
        feat_list.append(feats)
        lab_list.append(labels)
        lengths.append(len(xyz))

    stacked_points = np.concatenate(xyz_list, axis=0)
    stacked_features = np.concatenate(feat_list, axis=0)
    stacked_labels = np.concatenate(lab_list, axis=0)
    stack_lengths = np.array(lengths, dtype=np.int32)

    return _build_layer_pyramid(
        stacked_points, stacked_features, stacked_labels,
        stack_lengths, config,
    )