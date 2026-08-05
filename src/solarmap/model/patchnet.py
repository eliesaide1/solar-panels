"""Small fully-convolutional CNN for panel patches in high-resolution imagery.

At ~6 cm/px the cue that separates a panel from a roof is not colour -- panel
and background luminance differ by under 10 levels at Jbeil -- but the regular
grid of module edges. That is a spatial pattern, so it needs a learned filter
rather than a threshold.

The network is deliberately tiny (~90k parameters) so it trains on CPU in
minutes, and fully convolutional so inference runs once over a whole tile and
returns a dense probability map, instead of re-running a classifier at every
sliding-window position.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 64 px at 6.18 cm/px is ~4 m of ground: several module rows, enough to show
# the repeating grid that distinguishes an array.
PATCH = 64
STRIDE = 8          # three 2x pools -> one output cell per 8 input pixels


def _block(cin: int, cout: int, pool: bool) -> list[nn.Module]:
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return layers


class PatchNet(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            *_block(3, w, False),
            *_block(w, w, True),           # /2
            *_block(w, 2 * w, False),
            *_block(2 * w, 2 * w, True),   # /4
            *_block(2 * w, 4 * w, False),
            *_block(4 * w, 4 * w, True),   # /8
            *_block(4 * w, 4 * w, False),
        )
        # 1x1 head keeps the network convolutional: the same weights that
        # classify a 64x64 patch produce a dense map on a full tile.
        self.head = nn.Conv2d(4 * w, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Single logit per patch, for training on fixed-size crops."""
        return self.forward(x).mean(dim=(2, 3)).squeeze(1)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalise(arr):
    """HWC uint8 RGB -> CHW float tensor."""
    import numpy as np

    x = arr.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1))
