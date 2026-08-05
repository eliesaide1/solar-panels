"""Detection by classical proposals plus a learned box classifier.

Two stages:

1. Colour/texture thresholding proposes candidate rectangles. This has high
   recall (~76% of verified panels at Jbeil) because PV modules have a
   distinctive drab, densely-textured signature, but poor precision -- it also
   fires on concrete roofs, greenhouse glazing and tarmac.
2. A random forest, trained on human-verified accept/reject decisions, scores
   each candidate and discards the false positives.

Splitting localisation from classification is why this beats an end-to-end
detector on a small dataset: a few hundred boxes is far too little to learn
localisation, but plenty to learn "is this rectangle a panel?".

Measured on held-out tiles at Jbeil: precision 0.61, recall 0.57, versus 0.16 /
0.10 for a YOLO detector fine-tuned on the same labels.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np


# The texture window is defined in metres, not pixels. A fixed 7 px window
# measures a different physical thing at every resolution -- 1.7 m on 25 cm
# imagery but 0.43 m on 6 cm imagery -- so thresholds tuned at one scale are
# meaningless at another.
TEXTURE_WINDOW_M = 1.75


def texture_window_px(gsd_m: float) -> int:
    k = int(round(TEXTURE_WINDOW_M / max(gsd_m, 1e-6)))
    k = max(3, min(k, 61))
    return k if k % 2 else k + 1      # cv2.blur wants an odd kernel


def texture_map(bgr: np.ndarray, gsd_m: float = 0.247) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = texture_window_px(gsd_m)
    mean = cv2.blur(g, (k, k))
    return np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mean * mean, 0))


def propose_rich(bgr: np.ndarray, gsd_m: float,
                 min_area_m2: float = 4.0, max_area_m2: float = 8000.0) -> list[dict]:
    """Candidate boxes with thresholds relaxed for maximum recall.

    Measured on the Jbeil labels: these settings put a candidate on 100% of
    verified panels (versus ~96% for the tighter `propose`), at the cost of
    roughly twice as many candidates. That trade is right when a trained
    classifier does the discarding -- a panel with no candidate can never be
    recovered, but a spurious candidate merely has to be rejected.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    tex = texture_map(bgr, gsd_m)

    mask = ((v > 50) & (s < 60) & (tex > 10)).astype(np.uint8) * 255
    ck = texture_window_px(gsd_m * 0.78)
    ok = texture_window_px(gsd_m * 2.30)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))

    px_area = gsd_m * gsd_m
    out = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (min_area_m2 <= a * px_area <= max_area_m2):
            continue
        if a / float(w * h) < 0.15:
            continue
        if max(w, h) / max(min(w, h), 1) > 15:
            continue
        out.append({"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h),
                    "score": float(tex[labels == i].mean())})
    return out


def _contour_of(component_mask, ox: int, oy: int, simplify_px: float):
    """Outline of a component as a polygon, in full-image pixel coordinates.

    The connected-component mask already describes the panel's true shape; the
    bounding box discards it. Tracing the contour keeps the L-shapes, angled
    rows and irregular edges that real arrays have -- and makes the reported
    area a measurement rather than an upper bound.
    """
    cnts, _ = cv2.findContours(component_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if simplify_px > 0:
        c = cv2.approxPolyDP(c, simplify_px, True)
    if len(c) < 3:
        return None
    return [(int(pt[0][0]) + ox, int(pt[0][1]) + oy) for pt in c]


def propose(bgr: np.ndarray, gsd_m: float,
            min_area_m2: float = 8.0, max_area_m2: float = 8000.0) -> list[dict]:
    """Candidate boxes from colour and texture. Deliberately over-generates."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    tex = texture_map(bgr, gsd_m)

    mask = ((v > 60) & (s < 32) & (tex > 16)).astype(np.uint8) * 255
    ck = texture_window_px(gsd_m * 0.78)     # ~2.2 m closing
    ok = texture_window_px(gsd_m * 1.40)     # ~1.25 m opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))

    px_area = gsd_m * gsd_m
    out = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (min_area_m2 <= a * px_area <= max_area_m2):
            continue
        if a / float(w * h) < 0.25:
            continue
        if max(w, h) / max(min(w, h), 1) > 12:
            continue
        out.append({"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h),
                    "score": float(tex[labels == i].mean())})
    return out


def box_features(bgr: np.ndarray, tex: np.ndarray, b: dict, gsd: float):
    """Must stay identical to scripts/train_filter.py, or the model misreads its input."""
    H, W = bgr.shape[:2]
    x1, y1 = max(0, b["x1"]), max(0, b["y1"])
    x2, y2 = min(W, b["x2"]), min(H, b["y2"])
    if x2 <= x1 or y2 <= y1:
        return None

    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    t = tex[y1:y2, x1:x2]

    pad = max(6, int(round(3.0 / max(gsd, 1e-6))))   # ~3 m context ring
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(W, x2 + pad), min(H, y2 + pad)
    outer = cv2.cvtColor(bgr[oy1:oy2, ox1:ox2], cv2.COLOR_BGR2HSV).astype(np.float32)

    w_m, h_m = (x2 - x1) * gsd, (y2 - y1) * gsd
    long_s, short_s = max(w_m, h_m), max(min(w_m, h_m), 1e-3)

    return [
        hsv[:, :, 0].mean(), hsv[:, :, 0].std(),
        hsv[:, :, 1].mean(), hsv[:, :, 1].std(),
        hsv[:, :, 2].mean(), hsv[:, :, 2].std(),
        t.mean(), t.std(), np.percentile(t, 90),
        hsv[:, :, 1].mean() - outer[:, :, 1].mean(),
        hsv[:, :, 2].mean() - outer[:, :, 2].mean(),
        w_m * h_m, long_s, long_s / short_s,
        float(b.get("score", 0.0)),
    ]


class CvFilterDetector:
    """Drop-in replacement for YoloDetector, same predict_boxes contract."""

    def __init__(self, model_path: str | Path, conf: float | None = None):
        with open(model_path, "rb") as fh:
            blob = pickle.load(fh)
        self.model = blob["model"]
        # The stored threshold is the cross-validated best-F1 point.
        self.conf = blob["threshold"] if conf is None else conf
        self.gsd = float(blob.get("gsd", 0.25))
        # Models trained on rich proposals must be fed rich proposals; mixing
        # the two shifts the feature distribution the classifier learned.
        self.proposer = propose_rich if blob.get("proposals") == "rich" else propose

    def predict_boxes(self, image: np.ndarray, gsd_m: float | None = None,
                      **_ignored) -> list[tuple[float, float, float, float, float]]:
        gsd = float(gsd_m or self.gsd)
        # OpenCV works in BGR; callers hand us RGB.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cands = self.proposer(bgr, gsd)
        if not cands:
            return []

        tex = texture_map(bgr, gsd)
        feats, keep = [], []
        for b in cands:
            f = box_features(bgr, tex, b, gsd)
            if f is not None:
                feats.append(f); keep.append(b)
        if not feats:
            return []

        probs = self.model.predict_proba(np.array(feats, np.float32))[:, 1]
        return [
            (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]), float(p))
            for b, p in zip(keep, probs) if p >= self.conf
        ]

    def predict_polygons(self, image: np.ndarray, gsd_m: float | None = None):
        """Like predict_boxes, but returns the traced outline of each panel.

        Yields (points, score) with points as full-image pixel coordinates.
        """
        gsd = float(gsd_m or self.gsd)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        tex = texture_map(bgr, gsd)
        mask = ((v > 60) & (s < 32) & (tex > 16)).astype(np.uint8) * 255
        ck = texture_window_px(gsd * 0.78)
        ok = texture_window_px(gsd * 1.40)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))

        px_area = gsd * gsd
        # 0.4 m: enough to drop pixel staircasing, not enough to round corners.
        simplify_px = max(1.0, 0.4 / gsd)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cands, comps = [], []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if not (8.0 <= a * px_area <= 8000.0):
                continue
            if a / float(w * h) < 0.25 or max(w, h) / max(min(w, h), 1) > 12:
                continue
            cands.append({"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h),
                          "score": float(tex[labels == i].mean())})
            comps.append((labels[y:y + h, x:x + w] == i, int(x), int(y)))
        if not cands:
            return []

        feats, keep = [], []
        for b, cm in zip(cands, comps):
            f = box_features(bgr, tex, b, gsd)
            if f is not None:
                feats.append(f); keep.append(cm)
        if not feats:
            return []

        probs = self.model.predict_proba(np.array(feats, np.float32))[:, 1]
        out = []
        for (cm, ox, oy), p in zip(keep, probs):
            if p < self.conf:
                continue
            pts = _contour_of(cm, ox, oy, simplify_px)
            if pts:
                out.append((pts, float(p)))
        return out
