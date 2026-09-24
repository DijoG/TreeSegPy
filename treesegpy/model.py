"""Thin wrapper around KPConv's KPFCNN for tree/non-tree classification."""
from pathlib import Path
import sys
import torch
import torch.nn as nn

# Make the KPConv repo importable
KPConv_ROOT = Path(__file__).resolve().parents[1] / "src" / "KPConv-PyTorch"
if str(KPConv_ROOT) not in sys.path:
    sys.path.insert(0, str(KPConv_ROOT))

from models.architectures import KPFCNN

class TreeSegModel(nn.Module):
    """
    Wrapper around KPFCNN configured for binary tree/non-tree segmentation.

    Forward signature matches KPFCNN: (batch, config) -> (N, 2) logits.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = KPFCNN(
            config,
            lbl_values=[0, 1],
            ign_lbls=[],
        )

    def forward(self, batch):
        return self.net(batch, self.config)

    def loss(self, outputs, labels):
        return self.net.loss(outputs, labels)

    def accuracy(self, outputs, labels):
        return self.net.accuracy(outputs, labels)