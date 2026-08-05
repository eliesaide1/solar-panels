"""Detection by dense CNN scoring plus connected components.

The patch CNN is fully convolutional, so one forward pass over a whole tile
yields a probability map at 1/8 resolution -- far cheaper than re-running a
classifier at every sliding-window position.

Panels are then found as connected regions of that map. This is the piece the
colour-threshold detector could not do at high resolution: at ~6 cm/px a panel
surface is smooth and only ~8 luminance levels darker than its surroundings,
so the discriminating signal is the repeating grid of module edges, which the
network learns and a threshold cannot express.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..model.patchnet import STRIDE, PatchNet, normalise


class CnnDetector:
    """Same predict_boxes contract as the other detectors."""

    def __init__(self, model_path: str | Path, conf: float = 0.5,
                 min_area_m2: float = 6.0, device: str | None = None):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.net = PatchNet()
        self.net.load_state_dict(ckpt["state_dict"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net.to(self.device).eval()
        self.conf = conf
        self.min_area_m2 = min_area_m2
        self.gsd = float(ckpt.get("gsd", 0.0618))

    @torch.no_grad()
    def score_map(self, rgb: np.ndarray, chunk: int = 1024) -> np.ndarray:
        """Dense panel probability at 1/STRIDE resolution."""
        H, W = rgb.shape[:2]
        out = np.zeros(((H + STRIDE - 1) // STRIDE, (W + STRIDE - 1) // STRIDE), np.float32)

        # Process in overlapping chunks: a 2048px tile at full resolution can
        # exceed memory, and the overlap avoids seams at chunk edges.
        pad = 32
        for y0 in range(0, H, chunk):
            for x0 in range(0, W, chunk):
                y1, x1 = min(H, y0 + chunk), min(W, x0 + chunk)
                ya, xa = max(0, y0 - pad), max(0, x0 - pad)
                yb, xb = min(H, y1 + pad), min(W, x1 + pad)
                sub = rgb[ya:yb, xa:xb]
                t = normalise(sub).unsqueeze(0).to(self.device)
                p = torch.sigmoid(self.net(t))[0, 0].cpu().numpy()
                # Map the padded result back onto the unpadded region.
                oy, ox = (y0 - ya) // STRIDE, (x0 - xa) // STRIDE
                h = (y1 - y0 + STRIDE - 1) // STRIDE
                w = (x1 - x0 + STRIDE - 1) // STRIDE
                out[y0 // STRIDE:y0 // STRIDE + h, x0 // STRIDE:x0 // STRIDE + w] = \
                    p[oy:oy + h, ox:ox + w]
        return out

    def predict_boxes(self, image: np.ndarray, gsd_m: float | None = None,
                      **_ignored) -> list[tuple[float, float, float, float, float]]:
        gsd = float(gsd_m or self.gsd)
        prob = self.score_map(image)
        mask = (prob >= self.conf).astype(np.uint8)

        # Close gaps between module rows so an array is one region. The kernel
        # is sized in metres of ground, not pixels of the score map.
        k = max(3, int(round(1.5 / (gsd * STRIDE))))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        cell_area = (gsd * STRIDE) ** 2
        out = []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a * cell_area < self.min_area_m2:
                continue
            score = float(prob[labels == i].mean())
            out.append((float(x * STRIDE), float(y * STRIDE),
                        float((x + w) * STRIDE), float((y + h) * STRIDE), score))
        return out
