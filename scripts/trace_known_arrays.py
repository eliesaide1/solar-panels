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


def trace_by_darkness(rgb, x1, y1, x2, y2, gsd, min_frac, solidity, contrast):
    """Segment the array inside a verified box without the model.

    Needed because the model is not merely under-confident on the arrays it
    misses, it is silent: median peak response inside them is 0.013, against
    0.982 on the ones it finds. No threshold recovers a shape from that, so
    where the model says nothing the only alternatives are a rectangle, a
    classical trace, or digitising by hand.

    Inside a box a human has verified, this is a far easier problem than
    detection -- which is why cv+filter failing to *find* arrays at 6 cm says
    nothing about its ability to *outline* one that is known to be there.
    Panels are darker than the roof they sit on, and most of the box is already
    panel, so Otsu on the inverted grey channel separates the two without a
    tuned threshold. The morphology closes inter-module lines and glare, the
    same way the model path does.
    """
    sub = rgb[y1:y2, x1:x2]
    if sub.size == 0 or sub.shape[0] < 3 or sub.shape[1] < 3:
        return None
    grey = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    # THRESH_BINARY_INV: foreground is the DARK side of the split.
    _, bw = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    k = max(3, int(round(0.5 / gsd)) | 1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n_c, lbl, stats, _ = cv2.connectedComponentsWithStats(
        (bw > 0).astype(np.uint8), connectivity=8)
    if n_c < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lbl == big).astype(np.uint8)

    # A speck is worse than the rectangle it replaces, and so is a trace that
    # swallows the whole box -- the latter means Otsu split roof from shadow
    # rather than panel from roof, and has measured nothing.
    frac = comp.sum() / max(1, comp.size)
    if frac < min_frac or frac > 0.98:
        return None

    # Otsu always returns a split, even on a bare roof: it has no notion of
    # whether the darker class is a panel. Inspected over 57 boxes, about a
    # third of what it produced outlined shadow, roof clutter or open ground,
    # and those come out looking exactly like measurements. Two cheap tests
    # reject them, and both target what the bad traces actually looked like
    # rather than panel appearance in general.
    #
    # This is not the shape-filtering the README records as a dead end. That
    # tried to tell a real array from a false positive, where the features
    # genuinely do not separate. This only asks whether a shape is a plausible
    # trace of an array already verified to be inside this box.
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    if hull_area <= 0:
        return None
    # Solidity kills the sawtooth traces, where the threshold followed the
    # module rows and the outline zigzagged along the striping.
    if cv2.contourArea(c) / hull_area < solidity:
        return None

    # An array is materially darker than the roof around it. Shadow-on-shadow
    # and roof-on-roof splits are not, because Otsu was dividing one surface
    # rather than two.
    inside = grey[comp > 0]
    outside = grey[comp == 0]
    if inside.size == 0 or outside.size == 0:
        return None
    if float(outside.mean()) - float(inside.mean()) < contrast:
        return None
    return comp


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
    ap.add_argument("--margin", type=float, default=0.0,
                    help="search this much beyond the box on each side, as a "
                         "fraction of its size. Default 0 confines the trace to "
                         "the box a human verified. Both arguments for going "
                         "wider were tried and neither survived. (1) 'boxes are "
                         "drawn tight': measured, they are not -- confining the "
                         "trace shrinks these 116 arrays by 21%%, which is the "
                         "over-estimate this script exists to remove. Growing "
                         "the window inflates instead: +3%% at 0.2, +29%% at "
                         "0.4, +68%% at 0.8, where the median trace (61 m2) "
                         "exceeds the median of the arrays the detector traced "
                         "unaided (50 m2) despite this population being the "
                         "smaller ones it missed. (2) 'boxes are clipped at "
                         "tile edges': 17%% are, but a margin cannot recover "
                         "them -- a box at x1=0 grows to max(0, -mx) = 0, and "
                         "the array continues into a neighbouring tile this "
                         "pass never reads. It is labelled separately there. So "
                         "the margin only ever grows into interior sides that "
                         "were never clipped, which is precisely the "
                         "inflation.")
    ap.add_argument("--max-growth", type=float, default=6.0,
                    help="reject a trace larger than this multiple of its box. "
                         "Letting the shape leave the box is what makes it "
                         "correctly sized, and also what lets it escape along a "
                         "roofline into a neighbouring structure -- one trace "
                         "reached 4,296 m2 against a largest real array of "
                         "1,707. Past this the rectangle is the safer answer. "
                         "The search window itself caps growth at about "
                         "(1 + 2*margin)^2, so a limit near that value "
                         "constrains nothing.")
    ap.add_argument("--max-area-m2", type=float, default=0.0,
                    help="reject a trace bigger than this. 0 derives it from "
                         "the largest array actually verified in this capture, "
                         "which is the honest ceiling: a trace exceeding every "
                         "real array has escaped into a roof or a greenhouse. "
                         "A ratio cap alone cannot catch this -- a large box "
                         "growing within its allowance still produces an "
                         "impossible shape.")
    ap.add_argument("--min-frac", type=float, default=0.15,
                    help="reject a trace covering less of the box than this -- "
                         "a speck is worse than the rectangle it replaces")
    ap.add_argument("--no-fallback", dest="fallback", action="store_false",
                    help="leave a rectangle where the model is silent, instead "
                         "of falling back to a classical trace. The model is "
                         "not quiet on these boxes, it is silent -- median peak "
                         "0.013 -- so without the fallback they can only ever "
                         "stay rectangles.")
    ap.add_argument("--cv-solidity", type=float, default=0.75,
                    help="minimum area/convex-hull for a darkness trace. Otsu "
                         "following the module rows produces a sawtooth outline "
                         "whose solidity collapses; a real array is close to "
                         "convex.")
    ap.add_argument("--cv-contrast", type=float, default=12.0,
                    help="minimum grey-level gap between the traced region and "
                         "the rest of the box, 0-255. Otsu returns a split even "
                         "on bare roof, where it is dividing one surface rather "
                         "than separating panel from roof; those splits have "
                         "almost no contrast across them.")
    ap.add_argument("--out", default="survey_traced.geojson")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    survey = json.loads((cap / args.survey).read_text(encoding="utf-8"))
    kw_per_m2 = float(cfg["capacity"]["kw_per_m2"])

    if args.max_area_m2 <= 0:
        biggest = 0.0
        for t in man["tiles"]:
            g = float(t["gsd_m"])
            for b in lab["tiles"].get(t["tile_id"], []):
                if b.get("verified") is True:
                    biggest = max(biggest, (b["x2"] - b["x1"]) * g
                                  * (b["y2"] - b["y1"]) * g)
        args.max_area_m2 = biggest
        print(f"area ceiling {biggest:,.0f} m2 (largest verified array here)")

    # Only the features that fell back to a box need work.
    boxed = [f for f in survey["features"]
             if f["properties"].get("source") == "hand-labelled"]
    keep = [f for f in survey["features"]
            if f["properties"].get("source") != "hand-labelled"]
    print(f"{len(boxed)} arrays to trace, {len(keep)} already traced")
    if not boxed:
        # Write the survey through unchanged rather than returning. Callers
        # chain build_survey_layer -> trace_known_arrays -> publish, and a
        # missing output makes that chain publish nothing at all: on a capture
        # with no fallback rectangles the map simply never updated, which reads
        # as the whole Apply having done nothing.
        (cap / args.out).write_text(json.dumps(survey), encoding="utf-8")
        print(f"nothing to trace; passed through -> {cap / args.out}")
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

    traced = failed = by_cv = 0
    area_before = area_after = 0.0
    for i, (tid, feats) in enumerate(want.items(), 1):
        t = next(x for x in man["tiles"] if x["tile_id"] == tid)
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]
        gsd = float(t["gsd_m"])
        print(f"\r  tile {i}/{len(want)} {tid}", end="", flush=True)

        with Image.open(cap / t["image"]) as im:
            rgb = np.array(im.convert("RGB"))
        prob = det.predict(rgb, tile_size=int(ic["tile_size"]),
                           stride=int(ic["stride"]))

        def model_trace(x1, y1, x2, y2, box_area):
            """The model's outline for this box, or None if it has no opinion."""
            # Search a window around the box, not the box itself, so the shape
            # can reach the array's real extent. The component is then chosen
            # by overlap with the box, which keeps it anchored to the array a
            # human actually verified rather than drifting to a neighbour.
            mx = int(round((x2 - x1) * args.margin))
            my = int(round((y2 - y1) * args.margin))
            wx1, wx2 = max(0, x1 - mx), min(W, x2 + mx)
            wy1, wy2 = max(0, y1 - my), min(H, y2 + my)

            sub = (prob[wy1:wy2, wx1:wx2] >= args.threshold).astype(np.uint8)
            # Close small gaps: glare and inter-module lines break a mask that
            # is really one array.
            k = max(3, int(round(0.5 / gsd)) | 1)
            sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
            n_c, lbl, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
            if n_c < 2:
                return None

            inner = np.zeros(sub.shape, bool)
            inner[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1] = True
            best, best_hit = 0, 0
            for c_i in range(1, n_c):
                hit = int(((lbl == c_i) & inner).sum())
                if hit > best_hit:
                    best_hit, best = hit, c_i
            if not best:
                return None
            comp = (lbl == best).astype(np.uint8)
            if best_hit / max(1, (x2 - x1) * (y2 - y1)) < args.min_frac:
                return None

            traced_area = comp.sum() * gsd * gsd
            if (traced_area > args.max_growth * box_area
                    or traced_area > args.max_area_m2):
                return None
            return comp, wx1, wy1

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

            got = model_trace(x1, y1, x2, y2, box_area)
            basis = "traced inside verified box"
            if got is None and args.fallback:
                comp = trace_by_darkness(rgb, x1, y1, x2, y2, gsd, args.min_frac,
                                         args.cv_solidity, args.cv_contrast)
                if comp is not None:
                    got = (comp, x1, y1)
                    basis = "traced by darkness inside verified box"

            if got is None:
                keep.append(f); failed += 1; area_after += box_area; continue
            comp, ox, oy = got

            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, max(1.0, 0.4 / gsd), True)
            if len(c) < 3:
                keep.append(f); failed += 1; area_after += box_area; continue

            pts = [(w + (ox + int(p[0][0])) / W * (e - w),
                    n - (oy + int(p[0][1])) / H * (n - s)) for p in c]
            pts.append(pts[0])
            area = float(comp.sum()) * gsd * gsd
            area_after += area
            traced += 1
            if basis.startswith("traced by darkness"):
                by_cv += 1
            keep.append({"type": "Feature",
                         "geometry": {"type": "Polygon", "coordinates": [pts]},
                         "properties": {"area_m2": round(area, 1),
                                        "capacity_kw": round(area * kw_per_m2, 2),
                                        "confidence": None,
                                        "source": "hand-labelled, traced",
                                        "area_basis": basis}})

    total = sum(float(f["properties"].get("area_m2") or 0) for f in keep)
    out = dict(survey)
    out["features"] = keep
    out["properties"] = {**survey["properties"],
                         "detections": len(keep),
                         "total_area_m2": round(total, 1),
                         "total_capacity_kw": round(total * kw_per_m2, 2),
                         "traced_in_box": traced,
                         "traced_by_model": traced - by_cv,
                         "traced_by_darkness": by_cv,
                         "still_boxed": failed}
    (cap / args.out).write_text(json.dumps(out), encoding="utf-8")

    print(f"\n\n{traced} traced ({traced - by_cv} by the model above "
          f"{args.threshold}, {by_cv} by darkness where it was silent), "
          f"{failed} kept their box")
    print(f"  those arrays: {area_before:,.0f} m2 as boxes -> {area_after:,.0f} m2 traced")
    print(f"  survey total: {total:,.0f} m2  ~{total*kw_per_m2:,.0f} kW")
    print(f"-> {cap / args.out}")


if __name__ == "__main__":
    main()
