"""Score detections per ARRAY, not per pixel -- the number the project targets.

`evaluate.py` answers "how much panel area did we flag?". That is the right
question for a capacity estimate and the wrong one for "did we find this
installation?". A detector that traces 40% of every array scores 40% coverage
while having located every single one.

This scores array-level recall, which is what the README's "N% of arrays
located" claims and what no committed script previously produced.

Matching rules, both deliberately explicit because the headline number moves
with them:

  an ARRAY is located   if >= --min-cover of its labelled area is covered by
                        the union of detections. Using the union, rather than
                        any single detection, is what stops a large array
                        traced as three adjacent polygons from scoring as a
                        miss.

  a DETECTION is a hit  if >= --min-on-panel of its own area lies on the union
                        of labelled arrays. Scoring against the union likewise
                        stops one polygon spanning three neighbouring arrays
                        from being counted as a false positive.

Both directions use the union, so merges and splits are penalised only when
they are genuinely wrong, never for disagreeing with the labeller about where
one array stops and the next begins.

    python scripts/score_arrays.py --capture jbeil-mb
    python scripts/score_arrays.py --capture jbeil-mb --labels labels_transferred.json
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
    ap.add_argument("--labels", default="labels.json",
                    help="label file to score against (default labels.json)")
    ap.add_argument("--detections", default="detections.geojson")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="fraction of a labelled array that must be detected "
                         "for it to count as located (default 0.5)")
    ap.add_argument("--min-on-panel", type=float, default=0.5,
                    help="fraction of a detection that must lie on labelled "
                         "panel for it to count as correct (default 0.5)")
    ap.add_argument("--misses", action="store_true",
                    help="list the arrays that were not located, largest first")
    ap.add_argument("--labelled-tiles-only", action="store_true",
                    help="score only tiles that contain a verified array. This "
                         "is the old behaviour and it FLATTERS precision: a "
                         "tile swept and found empty is ground truth saying "
                         "'nothing here', so every detection on it is a false "
                         "positive, and skipping the tile hides them. At Jbeil "
                         "that was 15 detections on greenhouses across 34 "
                         "tiles, worth 8 points of precision. Use only when the "
                         "capture was NOT exhaustively swept, where an empty "
                         "tile really does mean 'not looked at'.")
    ap.add_argument("--held-out-from", metavar="DATASET",
                    help="score ONLY the tiles held out of this training set, "
                         "named under data/datasets. Without this, a fine-tuned "
                         "checkpoint is scored on ground it trained on and "
                         "reads several points high -- at Jbeil, 65.6%% on the "
                         "36 trained tiles against 51.1%% on the 11 held out. "
                         "The split is recovered from the crop filenames, which "
                         "prepare_masks.py writes as {tile_id}_{ox}_{oy}, rather "
                         "than re-derived from the RNG: re-deriving disagrees "
                         "the moment --seed or --include-empty-tiles differs, "
                         "which is how the four shipped models ended up on "
                         "three different splits.")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture

    only_tiles = None
    if args.held_out_from:
        val_dir = cfg.path("datasets") / args.held_out_from / "val" / "images"
        if not val_dir.is_dir():
            raise SystemExit(f"No val split at {val_dir}")
        only_tiles = {p.name.rsplit("_", 2)[0] for p in val_dir.glob("*.png")}
        if not only_tiles:
            raise SystemExit(f"No crops under {val_dir}")

    for name in ("manifest.json", args.labels, args.detections):
        if not (cap / name).is_file():
            raise SystemExit(f"Missing {name} in {cap}")

    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    dets = json.loads((cap / args.detections).read_text(encoding="utf-8"))

    n_arrays = n_located = 0
    n_dets = n_hits = 0
    gt_area = det_area = hit_area = 0.0
    misses: list[tuple[float, str, dict, float]] = []
    labelled_tiles = 0

    for t in manifest["tiles"]:
        tid = t["tile_id"]
        if only_tiles is not None and tid not in only_tiles:
            continue
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]
        gsd = float(t["gsd_m"])
        px = gsd * gsd

        panels = [b for b in labels["tiles"].get(tid, []) if b.get("verified") is True]
        if not panels and args.labelled_tiles_only:
            continue
        labelled_tiles += 1

        # Union of every detection whose centre falls in this tile, rasterised
        # once. Per-detection masks would double-count overlaps.
        det_union = np.zeros((H, W), bool)
        tile_dets: list[np.ndarray] = []
        for f in dets["features"]:
            ring_ll = f["geometry"]["coordinates"][0]
            xs = [c[0] for c in ring_ll]
            ys = [c[1] for c in ring_ll]
            if not (w <= (min(xs) + max(xs)) / 2 <= e):
                continue
            if not (s <= (min(ys) + max(ys)) / 2 <= n):
                continue
            ring = np.array(
                [[(lon - w) / (e - w) * W, (n - lat) / (n - s) * H]
                 for lon, lat in ring_ll], np.int32)
            layer = np.zeros((H, W), np.uint8)
            cv2.fillPoly(layer, [ring], 1)
            m = layer.astype(bool)
            if m.any():
                tile_dets.append(m)
                det_union |= m

        gt_union = np.zeros((H, W), bool)
        for b in panels:
            gt_union[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = True

        # --- recall: is each labelled array covered by the detection union? ---
        for b in panels:
            y1, y2 = max(0, b["y1"]), b["y2"]
            x1, x2 = max(0, b["x1"]), b["x2"]
            box = det_union[y1:y2, x1:x2]
            area_px = max(1, (y2 - y1) * (x2 - x1))
            frac = box.sum() / area_px
            n_arrays += 1
            gt_area += area_px * px
            if frac >= args.min_cover:
                n_located += 1
            else:
                misses.append((area_px * px, tid, b, frac))

        # --- precision: does each detection sit on labelled panel? ---
        for m in tile_dets:
            a = int(m.sum())
            if not a:
                continue
            n_dets += 1
            on = int((m & gt_union).sum())
            det_area += a * px
            hit_area += on * px
            if on / a >= args.min_on_panel:
                n_hits += 1

    if not n_arrays:
        raise SystemExit(f"No verified labels in {cap / args.labels}")

    recall = n_located / n_arrays
    precision = n_hits / n_dets if n_dets else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"capture        : {args.capture}")
    print(f"labels         : {args.labels}")
    print(f"model          : {dets['properties'].get('checkpoint')} "
          f"@ threshold {dets['properties'].get('threshold')}")
    if only_tiles is not None:
        print(f"HELD OUT ONLY  : {len(only_tiles)} tiles never trained on "
              f"(--held-out-from {args.held_out_from})")
    print(f"tiles scored   : {labelled_tiles}"
          + ("  (only those holding an array -- flatters precision)"
             if args.labelled_tiles_only else "  (all, empty ones count as background)"))
    print(f"match rule     : array located at >= {args.min_cover:.0%} covered, "
          f"detection correct at >= {args.min_on_panel:.0%} on panel")
    print()
    print(f"  ARRAY RECALL     {recall:6.1%}   {n_located} of {n_arrays} labelled arrays located")
    print(f"  DETECTION PREC.  {precision:6.1%}   {n_hits} of {n_dets} detections on panel")
    print(f"  F1               {f1:6.3f}")
    print()
    print(f"  (area view: {hit_area:,.0f} of {det_area:,.0f} m2 flagged is panel; "
          f"{gt_area:,.0f} m2 labelled)")

    if args.misses and misses:
        misses.sort(key=lambda r: -r[0])
        print(f"\n{len(misses)} arrays not located, largest first:")
        print(f"{'area m2':>9} {'covered':>8}  tile / position")
        for area, tid, b, frac in misses:
            print(f"{area:9.0f} {frac:8.1%}  {tid} @ ({b['x1']},{b['y1']})")


if __name__ == "__main__":
    main()
