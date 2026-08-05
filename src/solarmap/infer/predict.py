"""Sliding-window inference over a georeferenced tile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..model.net import build_model


class SolarDetector:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)

        # encoder_weights=None: the checkpoint supplies every weight, so
        # downloading ImageNet weights here would be wasted work.
        self.model = build_model(ckpt["encoder"], encoder_weights=None)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        self.mean = np.array(ckpt["mean"], dtype=np.float32)
        self.std = np.array(ckpt["std"], dtype=np.float32)
        self.default_tile = int(ckpt.get("tile_size", 512))

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        tile_size: int | None = None,
        stride: int | None = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Return a float probability map, same H x W as ``image`` (RGB uint8).

        Windows overlap and are averaged. A single-pass tiling leaves visible
        seams exactly where a panel happens to straddle a window edge, which
        then fragments into two polygons downstream.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")

        tile = tile_size or self.default_tile
        stride = stride or tile
        h, w = image.shape[:2]

        # Reflect-pad so windows tile the image exactly and edge pixels get
        # real context instead of black.
        pad_h = max(0, _ceil_to(h, tile, stride) - h)
        pad_w = max(0, _ceil_to(w, tile, stride) - w)
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        ph, pw = padded.shape[:2]

        norm = ((padded.astype(np.float32) / 255.0) - self.mean) / self.std
        norm = np.transpose(norm, (2, 0, 1))

        acc = np.zeros((ph, pw), dtype=np.float32)
        cnt = np.zeros((ph, pw), dtype=np.float32)

        coords = [(y, x)
                  for y in range(0, ph - tile + 1, stride)
                  for x in range(0, pw - tile + 1, stride)]

        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            batch = np.stack([norm[:, y:y + tile, x:x + tile] for y, x in chunk])
            logits = self.model(torch.from_numpy(batch).to(self.device))
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            for (y, x), p in zip(chunk, probs):
                acc[y:y + tile, x:x + tile] += p
                cnt[y:y + tile, x:x + tile] += 1.0

        np.maximum(cnt, 1e-6, out=cnt)
        return (acc / cnt)[:h, :w]


def _ceil_to(size: int, tile: int, stride: int) -> int:
    """Smallest length >= size that an integer number of windows covers."""
    if size <= tile:
        return tile
    steps = -(-(size - tile) // stride)
    return tile + steps * stride
