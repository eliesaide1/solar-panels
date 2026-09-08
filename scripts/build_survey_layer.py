"""Build a complete survey layer: every verified array, however it was found.

The detector finds about half the arrays at Jbeil. For a survey of ground that
has already been labelled by hand, that is the wrong output to ship -- the
labels know about every array, and throwing away the ones the model missed
produces an inventory that is complete only where the model happened to work.

So take both. Where a detection covers a verified array, keep the detection's
traced outline, because a traced shape is a measurement. Where nothing was
detected, fall back to the labelled box, which is an over-estimate of area but
is at least present. Each feature records which it is, so the two are never
silently mixed in a capacity total.

What comes out is a hand-verified inventory of the region, not a detector
result. It says nothing about how the model would do on new ground -- for that,
score detections_raw.geojson and quote the number that comes back.

    python scripts/build_survey_layer.py --capture jbeil-mb-104
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np

from solarmap.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels_clean.json")
    ap.add_argument("--detections", default="detections_curated.geojson")
    ap.add_argument("--out", default="survey.geojson")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="fraction of a labelled array a detection must cover "
                         "for its traced outline to be used instead of the box")
    ap.add_argument("--corrections", default="corrections.geojson",
                    help="shapes marked wrong in the map UI are suppressed from "
                         "the output. Rejecting the underlying LABEL is not "
                         "enough on its own: a detection can sit on ground no "
                         "label covers, or overlap a label the rejection rule "
                         "leaves standing, and then it returns on the next "
                         "rebuild -- which reads as the correction having done "
                         "nothing. This makes erasing final.")
    ap.add_argument("--no-corrections", dest="corrections", action="store_const",
                    const=None, help="ignore corrections.geojson entirely")
    ap.add_argument("--keep-unmatched", action="store_true",
                    help="keep detections that cover no verified array, flagged "
                         "as unverified. This module assumes an exhaustively "
                         "swept capture, where an unmatched detection is a false "
                         "positive worth dropping. On a region being labelled "
                         "that assumption is inverted -- almost nothing is "
                         "labelled yet -- and dropping them silently deletes the "
                         "seed layer somebody is working from. Enabled "
                         "automatically when the labels are too sparse to be a "
                         "sweep.")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    det = json.loads((cap / args.detections).read_text(encoding="utf-8"))
    kw_per_m2 = float(cfg["capacity"]["kw_per_m2"])

    # Shapes the user erased in the map UI, as pixel rings per tile.
    kill: dict[str, list] = {}
    n_killed = 0
    if args.corrections and (cap / args.corrections).is_file():
        corr = json.loads((cap / args.corrections).read_text(encoding="utf-8"))
        tiles_by_id = {t["tile_id"]: t for t in man["tiles"]}
        for f in corr.get("features", []):
            if f.get("properties", {}).get("kind") != "remove":
                continue
            ring = f["geometry"]["coordinates"][0]
            xs = [c[0] for c in ring]; ys = [c[1] for c in ring]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            for t in tiles_by_id.values():
                if t["west"] <= cx <= t["east"] and t["south"] <= cy <= t["north"]:
                    W_, H_ = t["width"], t["height"]
                    n_, s_, e_, w_ = t["north"], t["south"], t["east"], t["west"]
                    kill.setdefault(t["tile_id"], []).append(
                        np.array([[(lo - w_) / (e_ - w_) * W_,
                                   (n_ - la) / (n_ - s_) * H_] for lo, la in ring],
                                 np.int32))
                    break

    # A sweep has labels of the same order as detections. Far fewer means this
    # capture is mid-labelling, not finished, and unmatched detections are the
    # seed rather than errors.
    n_verified = sum(1 for bs in lab["tiles"].values()
                     for b in bs if b.get("verified") is True)
    n_dets = len(det.get("features", []))
    sparse = n_dets > 0 and n_verified < 0.5 * n_dets
    keep_unmatched = args.keep_unmatched or sparse
    if sparse and not args.keep_unmatched:
        print(f"NOTE: only {n_verified} verified arrays against {n_dets} "
              "detections -- this capture is not a completed sweep, so "
              "unmatched detections are KEPT rather than treated as false "
              "positives. Pass --no-keep-unmatched behaviour explicitly once "
              "the region is fully labelled.")

    # Suppress erased shapes BEFORE the per-tile loop, and independently of
    # labels. That loop skips any tile holding no verified array, so on a region
    # still being labelled -- where almost no tile has one -- the removal mask
    # was never built and erasing did nothing at all. A removal is a statement
    # about a detection, not about a label, so it cannot depend on one existing.
    if kill:
        from shapely.geometry import Polygon as _Poly
        from shapely.ops import unary_union as _union
        rings = []
        for f in (corr.get("features", []) if args.corrections else []):
            if f.get("properties", {}).get("kind") != "remove":
                continue
            g = _Poly(f["geometry"]["coordinates"][0])
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                rings.append(g)
        if rings:
            dead = _union(rings)
            for f in det.get("features", []):
                g = _Poly(f["geometry"]["coordinates"][0])
                if not g.is_valid:
                    g = g.buffer(0)
                if g.is_empty or g.area <= 0:
                    continue
                if g.intersection(dead).area / g.area >= 0.5:
                    f["properties"]["_used"] = True
                    n_killed += 1

    feats = []
    n_traced = n_boxed = n_drawn = n_unmatched = 0
    area_traced = area_boxed = area_drawn = 0.0

    for t in man["tiles"]:
        tid = t["tile_id"]
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]
        gsd = float(t["gsd_m"])
        panels = [b for b in lab["tiles"].get(tid, []) if b.get("verified") is True]
        if not panels:
            continue

        killed = np.zeros((H, W), bool)
        for ring_px in kill.get(tid, []):
            layer = np.zeros((H, W), np.uint8)
            cv2.fillPoly(layer, [ring_px], 1)
            killed |= layer.astype(bool)

        # Detections whose centre lands in this tile, rasterised once so an
        # array split across two polygons still counts as covered.
        local = []
        union = np.zeros((H, W), bool)
        for f in det["features"]:
            ring = f["geometry"]["coordinates"][0]
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            if not (w <= cx <= e and s <= cy <= n):
                continue
            poly = np.array([[(lon - w) / (e - w) * W, (n - lat) / (n - s) * H]
                             for lon, lat in ring], np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            mb = m.astype(bool)
            if f["properties"].get("_used"):
                continue                      # erased; suppressed globally above
            local.append((f, mb))
            union |= mb

        for b in panels:
            y1, y2 = max(0, b["y1"]), b["y2"]
            x1, x2 = max(0, b["x1"]), b["x2"]
            box_px = max(1, (y2 - y1) * (x2 - x1))
            covered = union[y1:y2, x1:x2].sum() / box_px

            # A hand-drawn outline outranks everything. Someone looked at this
            # rooftop and traced it; a detector's guess and a bounding box are
            # both weaker evidence. Without this branch the drawn shape is
            # silently discarded -- the layer rebuilds from the detector and
            # comes back looking exactly as it did before the correction, which
            # is precisely what a correction is supposed to change.
            poly = b.get("poly")
            if poly and len(poly) >= 3:
                ring = [[w + px / W * (e - w), n - py / H * (n - s)]
                        for px, py in poly]
                ring.append(ring[0])
                area = abs(cv2.contourArea(np.asarray(poly, np.int32))) * gsd * gsd
                feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"area_m2": round(area, 1),
                                   "capacity_kw": round(area * kw_per_m2, 2),
                                   "confidence": None,
                                   "source": "hand-drawn",
                                   "area_basis": "hand-drawn outline"},
                })
                n_drawn += 1
                area_drawn += area
                # Claim any detection under it, so the same array is not also
                # emitted as the detector's version of the same shape.
                for f, mb in local:
                    if mb[y1:y2, x1:x2].any():
                        f["properties"]["_used"] = True
                continue

            if covered >= args.min_cover:
                # Found: keep every detection overlapping this array, traced.
                for f, mb in local:
                    if mb[y1:y2, x1:x2].any() and not f["properties"].get("_used"):
                        f["properties"]["_used"] = True
                        g = dict(f)
                        g["properties"] = {**{k: v for k, v in f["properties"].items()
                                              if not k.startswith("_")},
                                           "source": "detected",
                                           "area_basis": "traced outline"}
                        feats.append(g)
                        n_traced += 1
                        area_traced += float(f["properties"].get("area_m2") or 0)
            else:
                # Missed: fall back to the labelled rectangle.
                lon1 = w + x1 / W * (e - w)
                lon2 = w + x2 / W * (e - w)
                lat1 = n - y2 / H * (n - s)
                lat2 = n - y1 / H * (n - s)
                area = (x2 - x1) * gsd * (y2 - y1) * gsd
                feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[
                        [lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2],
                        [lon1, lat1]]]},
                    "properties": {"area_m2": round(area, 1),
                                   "capacity_kw": round(area * kw_per_m2, 2),
                                   "confidence": None,
                                   "source": "hand-labelled",
                                   "area_basis": "bounding box (over-estimates)"},
                })
                n_boxed += 1
                area_boxed += area

    if keep_unmatched:
        for f in det.get("features", []):
            if f["properties"].get("_used"):
                continue
            g = dict(f)
            g["properties"] = {**{k: v for k, v in f["properties"].items()
                                  if not k.startswith("_")},
                               "source": "detected, unverified",
                               "area_basis": "traced outline"}
            feats.append(g)
            n_unmatched += 1
            area_traced += float(f["properties"].get("area_m2") or 0)

    total = area_traced + area_boxed + area_drawn
    out = {"type": "FeatureCollection",
           "properties": {
               "checkpoint": "SURVEY: verified arrays, detected where possible",
               "threshold": None,
               "detections": len(feats),
               "total_area_m2": round(total, 1),
               "total_capacity_kw": round(total * kw_per_m2, 2),
               "detected": n_traced,
               "hand_labelled": n_boxed,
               "hand_drawn": n_drawn,
               # Read by the UI if this layer is ever loaded as the active one.
               "kw_per_m2": kw_per_m2,
               "tiles_processed": len(man["tiles"]),
               # Mixed basis: traced outlines for the detected arrays, boxes
               # for the missed ones. The boxes dominate the error, so the
               # total is an over-estimate and the UI should say so.
               "area_is_upper_bound": n_boxed > 0,
               "note": "A hand-verified inventory, not a detector result. "
                       f"{n_boxed} arrays the model missed are shown as label "
                       "boxes, which over-estimate their area. Detector "
                       "performance is in detections_raw.geojson.",
           },
           "features": feats}
    (cap / args.out).write_text(json.dumps(out), encoding="utf-8")

    if n_killed:
        print(f"  {n_killed} detections suppressed (marked wrong in the map UI)")
    print(f"{len(feats)} arrays in the survey")
    print(f"  {n_traced:4} detected      {area_traced:10,.0f} m2  (traced, measured)")
    if n_unmatched:
        print(f"  {n_unmatched:4} unverified    kept as seed (no label covers them yet)")
    if n_drawn:
        print(f"  {n_drawn:4} hand-drawn    {area_drawn:10,.0f} m2  (traced by you, authoritative)")
    print(f"  {n_boxed:4} hand-labelled {area_boxed:10,.0f} m2  (boxes, over-estimate)")
    print(f"\n  total {total:,.0f} m2  ~{total*kw_per_m2:,.0f} kW")
    print(f"-> {cap / args.out}")


if __name__ == "__main__":
    main()
