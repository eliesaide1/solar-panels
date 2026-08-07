"""List the verified arrays the classifier is wrongly rejecting.

Recall is lost in two different ways and only one is worth your time:

  * the proposal stage never generated a candidate -- no amount of labelling
    helps, because at inference there is nothing for the classifier to score;
  * a candidate exists but the classifier scores it below threshold -- this is
    exactly what more positive examples fix.

This finds the second kind, so a labelling pass can be aimed at them instead of
re-confirming arrays the model already gets right.

    python scripts/find_missed.py --capture jbeil-nds
"""

import argparse
import json
import pickle

import _bootstrap  # noqa: F401

import cv2
import numpy as np

from solarmap.config import Config
from solarmap.infer.cvfilter import box_features, propose, texture_map


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--model", default="models/box_filter.pkl")
    ap.add_argument("--crops", action="store_true",
                    help="also write a contact sheet of the missed arrays")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels = json.loads((cap / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))

    with open(cfg.path("checkpoints").parent / args.model, "rb") as fh:
        blob = pickle.load(fh)
    clf, thr = blob["model"], blob["threshold"]

    rows, crops = [], []
    for t in manifest["tiles"]:
        tid = t["tile_id"]
        gsd = float(t["gsd_m"])
        panels = [b for b in labels["tiles"].get(tid, []) if b.get("verified") is True]
        if not panels:
            continue

        img = cv2.imread(str(cap / "tiles" / f"{tid}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        props = propose(img, gsd)
        tex = texture_map(img, gsd)

        feats, keep = [], []
        for b in props:
            f = box_features(img, tex, b, gsd)
            if f is not None:
                feats.append(f)
                keep.append(b)
        probs = clf.predict_proba(np.array(feats, np.float32))[:, 1] if feats else []

        for gt in panels:
            best_p, best_b = None, None
            for b, p in zip(keep, probs):
                ix = max(0, min(gt["x2"], b["x2"]) - max(gt["x1"], b["x1"]))
                iy = max(0, min(gt["y2"], b["y2"]) - max(gt["y1"], b["y1"]))
                if ix * iy <= 0:
                    continue
                ov = ix * iy / max(1, (gt["x2"] - gt["x1"]) * (gt["y2"] - gt["y1"]))
                if ov > 0.2 and (best_p is None or p > best_p):
                    best_p, best_b = p, b
            # Only the near-misses: a candidate exists, it just scored too low.
            if best_p is not None and best_p < thr:
                area = ((gt["x2"] - gt["x1"]) * gsd) * ((gt["y2"] - gt["y1"]) * gsd)
                rows.append((best_p, tid, gt, area))
                if args.crops:
                    pad = 40
                    y1, y2 = max(0, gt["y1"] - pad), min(img.shape[0], gt["y2"] + pad)
                    x1, x2 = max(0, gt["x1"] - pad), min(img.shape[1], gt["x2"] + pad)
                    c = img[y1:y2, x1:x2].copy()
                    if c.size:
                        cv2.rectangle(c, (gt["x1"] - x1, gt["y1"] - y1),
                                      (gt["x2"] - x1, gt["y2"] - y1), (0, 255, 255), 2)
                        crops.append((cv2.resize(c, (220, 220)), tid, best_p))

    rows.sort(key=lambda r: -r[0])
    print(f"{len(rows)} verified arrays are proposed but rejected by the classifier\n")
    print(f"{'score':>6} {'thr':>6} {'area m2':>9}  tile / position")
    by_tile: dict[str, int] = {}
    for p, tid, gt, area in rows:
        print(f"{p:6.3f} {thr:6.2f} {area:9.0f}  {tid} @ ({gt['x1']},{gt['y1']})")
        by_tile[tid] = by_tile.get(tid, 0) + 1

    print("\nconcentrated in:")
    for tid, n in sorted(by_tile.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {tid}: {n}")

    if crops:
        cols = 6
        pad_n = (-len(crops)) % cols
        cells = [c[0] for c in crops] + [np.zeros((220, 220, 3), np.uint8)] * pad_n
        sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
        out = cfg.path("outputs") / "missed_arrays.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), sheet)
        print(f"\ncontact sheet -> {out}")


if __name__ == "__main__":
    main()
