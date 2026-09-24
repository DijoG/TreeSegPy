"""PyTorch Dataset for TreeSegPy patches."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .batch import make_batch


class PatchDataset(Dataset):
    """
    Dataset that returns batches of patches, already built into a TreeScanBatch.

    Each __getitem__ call returns one .pt file's worth of data:
    a fully constructed TreeScanBatch with features, labels, and the layer pyramid.
    Since Batch objects cannot go through PyTorch's default collate, the
    DataLoader must use collate_fn=lambda x: x[0].
    """

    def __init__(self, files, config):
        self.files = list(files)
        self.config = config

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        batch_files = self.files[idx]
        patches = []
        for f in batch_files:
            d = np.load(f)
            patch = {"xyz": d["xyz"], "tree_id": d["tree_id"]}
            for key in ["geom_feats", "intensity", "intensity_norm", "h_below_canopy"]:
                if key in d.files:
                    patch[key] = d[key]
            patches.append(patch)
        batch = make_batch(patches, self.config)
        return batch


def collate_single(batch_list):
    """DataLoader collate_fn — returns the single pre-built batch."""
    return batch_list[0]


def make_file_batches(files, batch_num, shuffle=True, seed=None):
    """
    Split the list of patch files into groups of `batch_num`.
    Returns a list of lists of file paths.
    """
    files = list(files)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
    groups = []
    for i in range(0, len(files), batch_num):
        groups.append(files[i:i + batch_num])
    return groups