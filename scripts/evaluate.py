"""Score a capture's detections against its human-verified labels.

Reports two complementary measures, because they answer different questions:

  precision  of the area we flagged, how much is really panel?
             -> "when it points at something, is it right?"
  coverage   of the real panel area, how much did we flag?
             -> "how much did it find?"

Both are computed on pixel area rather than box counts. Box counting is
misleading here: one detection covering three adjacent arrays scores as a false
positive plus two misses, even though every panel was correctly located.

    python scripts/evaluate.py --capture jbeil-nds
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np

from solarmap.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--per-tile", action="store_true", help="also list each tile")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture

    for name in ("manifest.json", "labels.json", "detections.geojson"):
        if not (cap / name).is_file():
            raise SystemExit(f"Missing {name} in {cap}")

    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((cap / "labels.json").read_text(encoding="utf-8"))
    dets = json.loads((cap / "detections.geojson").read_text(encoding="utf-8"))

    gsd = float(manifest["tiles"][0]["gsd_m"])
    rows = []
    tot_gt = tot_hit = tot_det = 0

    for t in manifest["tiles"]:
        tid = t["tile_id"]
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]

        gt = np.zeros((H, W), bool)
        for b in labels["tiles"].get(tid, []):
            if b.get("verified") is True:
                gt[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = True

        dt = np.zeros((H, W), bool)
        for f in dets["features"]:
            xs = [c[0] for c in f["geometry"]["coordinates"][0]]
            ys = [c[1] for c in f["geometry"]["coordinates"][0]]
            # Assign each detection to the tile containing its centre.
            if not (w <= (min(xs) + max(xs)) / 2 <= e):
                continue
            if not (s <= (min(ys) + max(ys)) / 2 <= n):
                continue
            # Fill the actual polygon. Using its bounding box would credit
            # the detector with area it never claimed, once outlines replaced
            # rectangles.
            ring = np.array(
                [[(lon - w) / (e - w) * W, (n - lat) / (n - s) * H]
                 for lon, lat in f["geometry"]["coordinates"][0]],
                np.int32,
            )
            layer = np.zeros((H, W), np.uint8)
            cv2.fillPoly(layer, [ring], 1)
            dt |= layer.astype(bool)

        hit = int((gt & dt).sum())
        if gt.any():
            rows.append((hit / gt.sum(), tid, gt.sum() * gsd * gsd))
        tot_gt += int(gt.sum()); tot_hit += hit; tot_det += int(dt.sum())

    if not tot_gt:
        raise SystemExit("No verified labels in this capture -- nothing to score against.")

    px = gsd * gsd
    print(f"capture       : {args.capture}")
    print(f"model         : {dets['properties'].get('checkpoint')} "
          f"@ threshold {dets['properties'].get('threshold')}")
    print(f"tiles labelled: {len(rows)}")
    print(f"labelled panel area : {tot_gt*px:11,.0f} m2")
    print(f"detected area       : {tot_det*px:11,.0f} m2")
    print(f"correctly detected  : {tot_hit*px:11,.0f} m2")
    print()
    print(f"  PRECISION  {tot_hit/max(tot_det,1):6.1%}   of what it flagged is really panel")
    print(f"  COVERAGE   {tot_hit/tot_gt:6.1%}   of real panel area was found")

    if args.per_tile:
        print("\nper tile (worst first):")
        for c, tid, a in sorted(rows):
            print(f"  {c:6.1%}  {tid:22} {a:9,.0f} m2 labelled")


if __name__ == "__main__":
    main()
