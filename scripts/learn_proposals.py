"""Learn the local panel appearance from verified boxes, then re-propose.

Hand-tuned colour thresholds proved too brittle for this imagery: panels at
Notre Dame des Secours span saturation 29-42 and can be brighter *or* darker
than their surroundings, so no fixed cut separates them from tarmac and pale
roofs. Instead this fits a small pixel classifier on patches drawn from boxes a
human (or reviewer) has already accepted, and applies it to every tile.

Positives come from accepted boxes; negatives from rejected boxes plus random
background, which is what teaches it to ignore the concrete roofs that the
threshold version kept firing on.

    python scripts/learn_proposals.py --capture jbeil-nds
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from solarmap.config import Config

WIN = 16  # patch side in pixels: ~4 m at 25 cm/px, a few PV modules across


def features(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel feature stack: colour, local texture and local contrast."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def blur(a, k):
        return cv2.blur(a, (k, k))

    mean7 = blur(gray, 7)
    std7 = np.sqrt(np.maximum(blur(gray * gray, 7) - mean7 * mean7, 0))
    mean31 = blur(gray, 31)
    std31 = np.sqrt(np.maximum(blur(gray * gray, 31) - mean31 * mean31, 0))

    # Directional gradients: panel rows produce strong, oriented edges.
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    return np.stack([
        hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
        std7, std31,
        gray - mean31,          # local contrast: darker or brighter than context
        blur(gx, 7), blur(gy, 7),
        np.abs(blur(gx, 7) - blur(gy, 7)),   # directional anisotropy
    ], axis=-1)


def sample(feat, boxes, label, rng, per_box=220):
    out = []
    h, w = feat.shape[:2]
    for b in boxes:
        x1, y1 = max(0, b["x1"]), max(0, b["y1"])
        x2, y2 = min(w, b["x2"]), min(h, b["y2"])
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        xs = rng.integers(x1, x2, per_box)
        ys = rng.integers(y1, y2, per_box)
        out.append(feat[ys, xs])
    if not out:
        return np.zeros((0, feat.shape[2]), np.float32), np.zeros((0,), np.int8)
    X = np.concatenate(out)
    return X, np.full(len(X), label, np.int8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--min-area", type=float, default=12.0, help="m2")
    ap.add_argument("--max-area", type=float, default=8000.0, help="m2")
    ap.add_argument("--prob", type=float, default=0.55, help="pixel probability cut")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels = json.loads((cap / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    # Per-tile gsd is authoritative: an "@2x" source returns twice the
    # pixels over the same ground, and older manifests stored the
    # zoom-only figure, which understates it by 2x. cvfilter sizes its
    # kernels in metres, so a wrong gsd silently distorts every proposal.
    tiles = manifest["tiles"]
    gsd = float((tiles[0].get("gsd_m") if tiles else None) or manifest["gsd_m"])
    rng = np.random.default_rng(0)

    # ---- build the training set from reviewed boxes ----
    Xs, ys = [], []
    n_pos_boxes = n_neg_boxes = 0
    for tid, boxes in labels["tiles"].items():
        pos = [b for b in boxes if b.get("verified") is True]
        neg = [b for b in boxes if b.get("verified") is False]
        if not pos and not neg:
            continue
        img = cv2.imread(str(cap / "tiles" / f"{tid}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        feat = features(img)

        if pos:
            X, y = sample(feat, pos, 1, rng)
            Xs.append(X); ys.append(y); n_pos_boxes += len(pos)
        if neg:
            X, y = sample(feat, neg, 0, rng, per_box=60)
            Xs.append(X); ys.append(y); n_neg_boxes += len(neg)

        # Random background from this tile, avoiding any reviewed box.
        occupied = np.zeros(img.shape[:2], bool)
        for b in boxes:
            occupied[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = True
        free = np.argwhere(~occupied)
        if len(free):
            pick = free[rng.integers(0, len(free), 1500)]
            Xs.append(feat[pick[:, 0], pick[:, 1]])
            ys.append(np.zeros(len(pick), np.int8))

    if not Xs:
        raise SystemExit("No reviewed boxes found. Accept/reject some first.")
    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"training on {len(X):,} pixels from {n_pos_boxes} accepted "
          f"and {n_neg_boxes} rejected boxes ({y.sum():,} positive)")

    clf = RandomForestClassifier(
        n_estimators=120, max_depth=14, min_samples_leaf=8,
        class_weight="balanced", n_jobs=-1, random_state=0,
    )
    clf.fit(X, y)
    print("out-of-the-box train accuracy: %.3f" % clf.score(X, y))

    # ---- apply to every tile ----
    px_area = gsd * gsd
    proposals: dict[str, list] = {}
    total = 0
    for t in manifest["tiles"]:
        tid = t["tile_id"]
        img = cv2.imread(str(cap / "tiles" / f"{tid}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            proposals[tid] = []
            continue
        feat = features(img)
        h, w = feat.shape[:2]
        prob = clf.predict_proba(feat.reshape(-1, feat.shape[2]))[:, 1].reshape(h, w)

        mask = (prob >= args.prob).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        boxes = []
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y0, bw, bh, a = stats[i]
            area = a * px_area
            if not (args.min_area <= area <= args.max_area):
                continue
            if a / float(bw * bh) < 0.25:
                continue
            if max(bw, bh) / max(min(bw, bh), 1) > 12:
                continue
            boxes.append({
                "x1": int(x), "y1": int(y0), "x2": int(x + bw), "y2": int(y0 + bh),
                "score": round(float(prob[lab == i].mean()), 3),
                "verified": None,
            })
        proposals[tid] = boxes
        total += len(boxes)

    out = cap / "labels_learned.json"
    out.write_text(json.dumps(
        {"capture": args.capture, "gsd_m": gsd, "source": "learned", "tiles": proposals},
        indent=1), encoding="utf-8")
    print(f"{total} learned proposals -> {out.name}")


if __name__ == "__main__":
    main()
