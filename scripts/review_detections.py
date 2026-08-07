"""Feed a capture's detections back into the labelling UI for correction.

After a detection run you can see exactly what the model got wrong. This turns
those detections into reviewable boxes so the mistakes become training data:
mark the false positives red, draw the missed arrays yourself, retrain.

Detections that already match a verified label are skipped -- there is nothing
to learn from re-confirming what the model already gets right, and reviewing
them would waste your time. Only genuinely new boxes are added.

    python scripts/review_detections.py --capture jbeil-nds
    python scripts/label_ui                       # then review in the browser
"""

import argparse
import json
import shutil

import _bootstrap  # noqa: F401

from solarmap.config import Config


def overlaps(a, b, thresh=0.3) -> bool:
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    if inter <= 0:
        return False
    ua = ((a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
          + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter)
    return ua > 0 and inter / ua >= thresh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels_path = cap / "labels.json"

    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    dets = json.loads((cap / "detections.geojson").read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    # Keep a copy: this rewrites hand-made labels and a mistake would be costly.
    backup = cap / "labels.backup.json"
    if not backup.exists():
        shutil.copyfile(labels_path, backup)
        print(f"backed up existing labels -> {backup.name}")

    added = skipped = 0
    for t in manifest["tiles"]:
        tid = t["tile_id"]
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]
        existing = labels["tiles"].setdefault(tid, [])

        for f in dets["features"]:
            ring = f["geometry"]["coordinates"][0]
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            if not (w <= (min(xs) + max(xs)) / 2 <= e):
                continue
            if not (s <= (min(ys) + max(ys)) / 2 <= n):
                continue

            box = {
                "x1": max(0, round((min(xs) - w) / (e - w) * W)),
                "x2": min(W, round((max(xs) - w) / (e - w) * W)),
                "y1": max(0, round((n - max(ys)) / (n - s) * H)),
                "y2": min(H, round((n - min(ys)) / (n - s) * H)),
                "score": round(f["properties"].get("confidence", 0.0), 3),
                "verified": None,
            }
            if box["x2"] - box["x1"] < 3 or box["y2"] - box["y1"] < 3:
                continue

            # Already-verified matches teach nothing; only new boxes are worth
            # a human's attention.
            if any(overlaps(box, b) for b in existing):
                skipped += 1
                continue
            existing.append(box)
            added += 1

    labels_path.write_text(json.dumps(labels, indent=1), encoding="utf-8")

    total = sum(len(v) for v in labels["tiles"].values())
    unreviewed = sum(1 for v in labels["tiles"].values() for b in v
                     if b.get("verified") is None)
    print(f"added {added} detections for review ({skipped} already covered)")
    print(f"labels.json now: {total} boxes, {unreviewed} awaiting review")
    print(f"\nReview at: http://127.0.0.1:8000/label.html?capture={args.capture}")
    print("Mark false positives red, draw any arrays the model missed, then:")
    print(f"  python scripts/train_filter.py --capture {args.capture}")


if __name__ == "__main__":
    main()
