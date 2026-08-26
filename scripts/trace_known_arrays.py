"""Trace the outline of arrays the detector missed, inside their known boxes.

A survey layer falls back to the labelled rectangle wherever detection failed,
and a rectangle over-estimates area -- at Jbeil, 141 of 233 arrays, so the
capacity total is an upper bound rather than a measurement.

Those arrays do not need finding again; they need measuring. Inside a box a
human has verified, the question is no longer "is this a panel?" but "what
shape is it?", and that is a much easier question. The model can be run far
below its detection threshold there, because a false positive inside a
confirmed array is not possible -- the usual reason for a high threshold does
not apply.

Anything still blank at the low threshold keeps its box, flagged as such. The
alternative to this is digitising 141 polygons by hand in QGIS.

    python scripts/trace_known_arrays.py --capture jbeil-mb-104 \
        --checkpoint solar_unet_ms.pt
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np
from PIL import Image

from solarmap.config import Config

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels_clean.json")
    ap.add_argument("--survey", default="survey.geojson",
                    help="survey layer to upgrade in place")
    ap.add_argument("--checkpoint", default="solar_unet_ms.pt")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="deliberately low: inside a verified array a weak "
                         "response is signal, not noise")
    ap.add_argument("--min-frac", type=float, default=0.15,
                    help="reject a trace covering less of the box than this -- "
                         "a speck is worse than the rectangle it replaces")
    ap.add_argument("--out", default="survey_traced.geojson")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    survey = json.loads((cap / args.survey).read_text(encoding="utf-8"))
    kw_per_m2 = float(cfg["capacity"]["kw_per_m2"])

    # Only the features that fell back to a box need work.
    boxed = [f for f in survey["features"]
             if f["properties"].get("source") == "hand-labelled"]
    keep = [f for f in survey["features"]
            if f["properties"].get("source") != "hand-labelled"]
    print(f"{len(boxed)} arrays to trace, {len(keep)} already traced")
    if not boxed:
        return

    from solarmap.infer.predict import SolarDetector
    det = SolarDetector(cfg.path("checkpoints") / args.checkpoint)
    ic = cfg["inference"]

    # Which tiles actually contain work, so nothing else is inferred.
    want = {}
    for f in boxed:
        ring = f["geometry"]["coordinates"][0]
        xs = [c[0] for c in ring]; ys = [c[1] for c in ring]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        for t in man["tiles"]:
            if t["west"] <= cx <= t["east"] and t["south"] <= cy <= t["north"]:
                want.setdefault(t["tile_id"], []).append(f)
                break

    traced = failed = 0
    area_before = area_after = 0.0
    for i, (tid, feats) in enumerate(want.items(), 1):
        t = next(x for x in man["tiles"] if x["tile_id"] == tid)
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]
        gsd = float(t["gsd_m"])
        print(f"\r  tile {i}/{len(want)} {tid}", end="", flush=True)

        with Image.open(cap / t["image"]) as im:
            prob = det.predict(np.array(im.convert("RGB")),
                               tile_size=int(ic["tile_size"]),
                               stride=int(ic["stride"]))

        for f in feats:
            ring = f["geometry"]["coordinates"][0]
            xs = [(lon - w) / (e - w) * W for lon, _ in ring]
            ys = [(n - lat) / (n - s) * H for _, lat in ring]
            x1, x2 = int(max(0, min(xs))), int(min(W, max(xs)))
            y1, y2 = int(max(0, min(ys))), int(min(H, max(ys)))
            box_area = (x2 - x1) * (y2 - y1) * gsd * gsd
            area_before += box_area
            if x2 - x1 < 3 or y2 - y1 < 3:
                keep.append(f); failed += 1; area_after += box_area; continue

            sub = (prob[y1:y2, x1:x2] >= args.threshold).astype(np.uint8)
            # Close small gaps: glare and inter-module lines break a mask that
            # is really one array.
            k = max(3, int(round(0.5 / gsd)) | 1)
            sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
            n_c, lbl, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
            if n_c < 2:
                keep.append(f); failed += 1; area_after += box_area; continue
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            comp = (lbl == big).astype(np.uint8)
            if comp.sum() / max(1, sub.size) < args.min_frac:
                keep.append(f); failed += 1; area_after += box_area; continue

            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, max(1.0, 0.4 / gsd), True)
            if len(c) < 3:
                keep.append(f); failed += 1; area_after += box_area; continue

            pts = [(w + (x1 + int(p[0][0])) / W * (e - w),
                    n - (y1 + int(p[0][1])) / H * (n - s)) for p in c]
            pts.append(pts[0])
            area = float(comp.sum()) * gsd * gsd
            area_after += area
            traced += 1
            keep.append({"type": "Feature",
                         "geometry": {"type": "Polygon", "coordinates": [pts]},
                         "properties": {"area_m2": round(area, 1),
                                        "capacity_kw": round(area * kw_per_m2, 2),
                                        "confidence": None,
                                        "source": "hand-labelled, traced",
                                        "area_basis": "traced inside verified box"}})

    total = sum(float(f["properties"].get("area_m2") or 0) for f in keep)
    out = dict(survey)
    out["features"] = keep
    out["properties"] = {**survey["properties"],
                         "detections": len(keep),
                         "total_area_m2": round(total, 1),
                         "total_capacity_kw": round(total * kw_per_m2, 2),
                         "traced_in_box": traced,
                         "still_boxed": failed}
    (cap / args.out).write_text(json.dumps(out), encoding="utf-8")

    print(f"\n\n{traced} traced, {failed} kept their box (no response above "
          f"{args.threshold})")
    print(f"  those arrays: {area_before:,.0f} m2 as boxes -> {area_after:,.0f} m2 traced")
    print(f"  survey total: {total:,.0f} m2  ~{total*kw_per_m2:,.0f} kW")
    print(f"-> {cap / args.out}")


if __name__ == "__main__":
    main()
