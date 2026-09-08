"""Trim detection outlines back to the ground a human verified.

A traced outline follows the model's response, and the model does not stop at
the edge of an array. Where two array blocks sit a few metres apart it bridges
the gap and returns one shape covering both plus the bare ground between them:
at Jbeil, 13% of the traced area (2,170 m2, ~412 kW) lies outside every
labelled array, and two shapes exceed the largest array anyone verified.

This intersects each detection with the union of verified label boxes, so the
parts crossing unlabelled ground are cut away and a shape spanning a gap splits
into the pieces that are actually panel.

Two things make this safe here, and both stop being true elsewhere:

  * The boxes are GENEROUS, not tight. Measured on this capture, tracing inside
    a box yields a shape 21% smaller than the box, so a box contains its array
    with room to spare and clipping to it rarely cuts real panel. Were the
    boxes drawn tight, this would shave every array.

  * Every tile was swept. Unlabelled ground means a human looked and found
    nothing, rather than nobody having checked.

Like curate_detections.py, this produces a survey PRODUCT and not a
measurement. Areas afterwards cannot exceed what was labelled, so they must
never be quoted as detector accuracy -- the uncurated, unclipped layer is the
one that answers "how well does this model work?".

    python scripts/clip_to_labels.py --capture jbeil-mb-104 \
        --labels labels_reviewed.json --detections detections_curated.geojson
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform

from solarmap.config import Config
from solarmap.geo import WGS84, utm_crs_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels_reviewed.json")
    ap.add_argument("--detections", default="detections_curated.geojson")
    ap.add_argument("--out", default="detections_clipped.geojson")
    ap.add_argument("--fill-gaps-m", type=float, default=4.0,
                    help="close gaps up to this wide between neighbouring label "
                         "boxes before clipping. Boxes do not tile an array: a "
                         "large installation is labelled as dozens of separate "
                         "rectangles with a metre or two of inter-row spacing "
                         "between them, and that spacing is not inside any box. "
                         "Without closing it, clipping treats an array's own "
                         "internal spacing as open ground and cuts the shape "
                         "along every box edge -- turning one traced "
                         "installation into a patchwork of rectangles. The gap "
                         "between separate array BLOCKS is far wider than this, "
                         "so it still gets cut.")
    ap.add_argument("--max-gap-m2", type=float, default=20.0,
                    help="clip a shape if the largest CONTIGUOUS patch of "
                         "non-labelled ground inside it exceeds this. This is "
                         "the criterion that matters: a fraction cannot tell "
                         "the difference between a shape whose edges spill a "
                         "metre onto roof all round (harmless, and clipping it "
                         "only adds straight cuts) and one that bridges a gap "
                         "between two array blocks and swallows the bare ground "
                         "between them. The 2,547 m2 shape at Jbeil was 78%% on "
                         "panel and so survived a 75%% fraction test, while "
                         "covering one large patch of open ground.")
    ap.add_argument("--only-below", type=float, default=0.60,
                    help="clip ONLY shapes with less than this fraction of "
                         "their area on a labelled array; leave the rest "
                         "untouched. Clipping everything was a mistake: a box "
                         "is generous in AREA but still square in SHAPE, so "
                         "every outline crossing a box edge came back with a "
                         "straight cut, and a large array traced as one shape "
                         "returned as a patchwork of rectangles. A well-behaved "
                         "outline needs no clipping; only the shapes that "
                         "bridge gaps between arrays do, and those are exactly "
                         "the ones with a low on-panel fraction. Set to 1.0 to "
                         "clip everything (not advised).")
    ap.add_argument("--min-area-m2", type=float, default=2.0,
                    help="drop fragments smaller than this. Clipping a bridged "
                         "shape leaves slivers where the outline clipped a box "
                         "corner; they are noise, not arrays.")
    ap.add_argument("--simplify-m", type=float, default=0.25)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    det = json.loads((cap / args.detections).read_text(encoding="utf-8"))
    kw_per_m2 = float(cfg["capacity"]["kw_per_m2"])

    n_verified = sum(1 for bs in lab["tiles"].values()
                     for b in bs if b.get("verified") is True)
    if not n_verified:
        raise SystemExit(f"No verified arrays in {args.labels} -- refusing to "
                         "clip, this would erase every detection.")

    tiles = {t["tile_id"]: t for t in man["tiles"]}
    lat_c = (man["tiles"][0]["north"] + man["tiles"][0]["south"]) / 2.0
    lon_c = (man["tiles"][0]["east"] + man["tiles"][0]["west"]) / 2.0
    to_utm = Transformer.from_crs(WGS84, utm_crs_for(lat_c, lon_c), always_xy=True).transform

    out_feats: list[dict] = []
    area_before = area_after = 0.0
    n_in = n_clipped = n_split = n_dropped = n_left = 0

    for f in det["features"]:
        ring = f["geometry"]["coordinates"][0]
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        t = next((t for t in tiles.values()
                  if t["west"] <= cx <= t["east"] and t["south"] <= cy <= t["north"]), None)
        if t is None:
            out_feats.append(f)
            continue
        n_in += 1

        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]
        gsd = float(t["gsd_m"])

        poly = np.array([[(lon - w) / (e - w) * W, (n - lat) / (n - s) * H]
                         for lon, lat in ring], np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [poly], 1)

        gt = np.zeros((H, W), np.uint8)
        for b in lab["tiles"].get(t["tile_id"], []):
            if b.get("verified") is True:
                gt[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = 1
        if args.fill_gaps_m > 0:
            k = max(3, int(round(args.fill_gaps_m / gsd)) | 1)
            gt = cv2.morphologyEx(gt, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

        before_px = int(m.sum())
        clipped = (m & gt).astype(np.uint8)
        after_px = int(clipped.sum())
        area_before += before_px * gsd * gsd

        # Leave well-behaved shapes exactly as traced. Only the ones spilling
        # substantially off labelled ground are worth the straight edges that
        # clipping imposes.
        on_panel = after_px / max(before_px, 1)
        spill = (m & ~gt.astype(bool)).astype(np.uint8)
        n_c, _, stats, _ = cv2.connectedComponentsWithStats(spill, connectivity=8)
        biggest_gap = 0.0
        if n_c > 1:
            biggest_gap = float(stats[1:, cv2.CC_STAT_AREA].max()) * gsd * gsd
        bridges_gap = biggest_gap >= args.max_gap_m2

        if after_px == before_px or (on_panel >= args.only_below and not bridges_gap):
            out_feats.append(f)
            area_after += before_px * gsd * gsd
            n_left += 1
            continue
        n_clipped += 1
        if after_px == 0:
            n_dropped += 1
            continue

        # RETR_CCOMP, not RETR_EXTERNAL. The ground this clipping exists to
        # remove is usually an interior HOLE -- a shape bridging two array
        # blocks encloses the bare ground between them rather than touching it
        # from outside. RETR_EXTERNAL returns only outer boundaries, so the
        # hole was cut out of the mask and then filled straight back in by the
        # contour trace: the area barely moved and the map looked unchanged.
        cnts, hier = cv2.findContours(clipped, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
        kept_here = 0
        if hier is None:
            hier = np.empty((1, 0, 4), int)

        def to_ring(contour):
            c = cv2.approxPolyDP(contour, max(1.0, 0.2 / gsd), True)
            if len(c) < 3:
                return None
            pts = [(w + float(p[0][0]) / W * (e - w),
                    n - float(p[0][1]) / H * (n - s)) for p in c]
            pts.append(pts[0])
            return pts

        for idx, c in enumerate(cnts):
            if hier[0][idx][3] != -1:
                continue                      # a hole; attached to its parent
            shell = to_ring(c)
            if shell is None:
                continue
            holes = []
            child = hier[0][idx][2]
            while child != -1:
                ring = to_ring(cnts[child])
                if ring is not None:
                    holes.append(ring)
                child = hier[0][child][0]
            g = Polygon(shell, holes)
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                continue
            # Area in UTM, never in degrees -- a square degree is not an area.
            a = shapely_transform(to_utm, g).area
            if a < args.min_area_m2:
                continue
            if args.simplify_m > 0:
                g = g.simplify(args.simplify_m / 111_320.0, preserve_topology=True)
                if g.is_empty:
                    continue
            kept_here += 1
            area_after += a
            props = {k: v for k, v in f["properties"].items()}
            props["area_m2"] = round(a, 1)
            props["capacity_kw"] = round(a * kw_per_m2, 2)
            props["area_basis"] = "traced outline, clipped to verified labels"
            out_feats.append({"type": "Feature",
                              "geometry": mapping(g),
                              "properties": props})
        if kept_here > 1:
            n_split += 1
        if kept_here == 0:
            n_dropped += 1

    total = sum(float(f["properties"].get("area_m2") or 0) for f in out_feats)
    out = {"type": "FeatureCollection",
           "properties": {**det.get("properties", {}),
                          "detections": len(out_feats),
                          "total_area_m2": round(total, 1),
                          "total_capacity_kw": round(total * kw_per_m2, 2),
                          "clipped_to": args.labels,
                          "clipping": "detections trimmed to the union of verified "
                                      "label boxes. Area here cannot exceed what was "
                                      "labelled and is NOT a detector measurement.",
                          "area_is_upper_bound": False},
           "features": out_feats}
    (cap / args.out).write_text(json.dumps(out), encoding="utf-8")

    print(f"{n_in} detections examined")
    print(f"  {n_left} left exactly as traced (>= {args.only_below:.0%} on panel)")
    print(f"  {n_clipped} trimmed, {n_split} split into pieces, {n_dropped} lost entirely")
    print(f"  {area_before:,.0f} m2 -> {area_after:,.0f} m2 "
          f"({area_before - area_after:,.0f} m2 removed, "
          f"{100 * (area_before - area_after) / max(area_before, 1):.0f}%)")
    print(f"  {len(out_feats)} features  {total:,.0f} m2  ~{total * kw_per_m2:,.0f} kW")
    print(f"-> {cap / args.out}")


if __name__ == "__main__":
    main()
