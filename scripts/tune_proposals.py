"""Re-tune the classical proposal stage for a given imagery resolution.

The thresholds in `cvfilter.propose` were measured on Esri 24.7 cm/px, where a
PV array is a uniform drab blob. At 6 cm the module grid, the inter-module
gaps and the glare are all resolved, so the same thresholds either flood
(everything merges into one component per rooftop) or fragment. Measured on
jbeil-mb: `propose` covers 43% of arrays, `propose_rich` covers 95% but as
merged blobs of which only 14 in 859 are panel-dominated.

Neither is usable, so the thresholds have to be re-derived per resolution.
This sweeps them against verified labels and reports the Pareto front.

Three numbers matter, in this order:

  cover     fraction of labelled arrays with >= 50% of their area inside the
            union of proposals. This is the HARD CEILING on array recall --
            no downstream classifier can recover an array with no candidate.
  good      fraction of proposals that are >= 50% panel. Low means the mask is
            flooding and merging arrays into their rooftops.
  count     proposals per tile. A labelling round has to be reviewable.

    python scripts/tune_proposals.py --capture jbeil-mb
    python scripts/tune_proposals.py --capture jbeil-mb --max-tiles 12   # quick
"""

import argparse
import itertools
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np

from solarmap.config import Config
from solarmap.infer.cvfilter import texture_map, texture_window_px


def components(mask, gsd, min_area_m2, max_area_m2, fill, aspect):
    px_area = gsd * gsd
    out = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (min_area_m2 <= a * px_area <= max_area_m2):
            continue
        if a / float(w * h) < fill:
            continue
        if max(w, h) / max(min(w, h), 1) > aspect:
            continue
        out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="only sweep this many labelled tiles (0 = all)")
    ap.add_argument("--min-cover", type=float, default=0.5)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((cap / args.labels).read_text(encoding="utf-8"))

    # v_min, s_max, tex_min, close_m, open_m
    GRID = list(itertools.product(
        (40, 60, 80),          # brightness floor
        (32, 45, 60),          # saturation ceiling
        (10, 20, 30, 40),      # texture floor -- the module grid raises this at 6 cm
        (1.2,),                # closing, metres
        (0.8, 1.4, 2.2),       # opening, metres
    ))
    MIN_AREA, MAX_AREA, FILL, ASPECT = 6.0, 8000.0, 0.25, 12

    stats = {k: {"cover_hit": 0, "n_prop": 0, "good": 0} for k in GRID}
    n_gt = 0
    tiles_done = 0

    for t in man["tiles"]:
        tid = t["tile_id"]
        gsd = float(t["gsd_m"])
        panels = [b for b in lab["tiles"].get(tid, []) if b.get("verified") is True]
        if not panels:
            continue
        if args.max_tiles and tiles_done >= args.max_tiles:
            break
        img = cv2.imread(str(cap / "tiles" / f"{tid}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        tiles_done += 1
        n_gt += len(panels)

        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        tex = texture_map(img, gsd)

        gt_union = np.zeros((H, W), bool)
        for b in panels:
            gt_union[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = True

        for key in GRID:
            v_min, s_max, tex_min, close_m, open_m = key
            mask = ((v > v_min) & (s < s_max) & (tex > tex_min)).astype(np.uint8) * 255
            ck = texture_window_px(gsd * (1.75 / close_m))
            ok = texture_window_px(gsd * (1.75 / open_m))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))

            boxes = components(mask, gsd, MIN_AREA, MAX_AREA, FILL, ASPECT)
            st = stats[key]
            st["n_prop"] += len(boxes)

            union = np.zeros((H, W), bool)
            for x1, y1, x2, y2 in boxes:
                union[y1:y2, x1:x2] = True
            for b in panels:
                y1, y2 = max(0, b["y1"]), b["y2"]
                x1, x2 = max(0, b["x1"]), b["x2"]
                a = max(1, (y2 - y1) * (x2 - x1))
                if union[y1:y2, x1:x2].sum() / a >= args.min_cover:
                    st["cover_hit"] += 1
            for x1, y1, x2, y2 in boxes:
                a = max(1, (y2 - y1) * (x2 - x1))
                if gt_union[y1:y2, x1:x2].sum() / a >= 0.5:
                    st["good"] += 1

        print(f"\r  swept tile {tiles_done} ({tid})", end="", flush=True)

    print(f"\n\n{tiles_done} tiles, {n_gt} labelled arrays, {len(GRID)} configs\n")
    rows = []
    for key, st in stats.items():
        cover = st["cover_hit"] / max(n_gt, 1)
        good = st["good"] / max(st["n_prop"], 1)
        rows.append((cover, good, st["n_prop"], key))

    # Pareto front on (cover, good): keep configs nothing else beats on both.
    front = [r for r in rows
             if not any(o[0] >= r[0] and o[1] >= r[1] and o[:2] != r[:2] for o in rows)]

    print(f"{'cover':>7} {'good':>7} {'prop/tile':>10}  v_min s_max tex_min close_m open_m")
    print("  -- Pareto front (nothing beats these on both cover and good) --")
    for cover, good, n, key in sorted(front, key=lambda r: -r[0]):
        print(f"{cover:7.1%} {good:7.1%} {n/max(tiles_done,1):10.1f}  "
              f"{key[0]:5} {key[1]:5} {key[2]:7} {key[3]:7} {key[4]:6}")

    print("\n  -- best 12 by cover --")
    for cover, good, n, key in sorted(rows, key=lambda r: (-r[0], -r[1]))[:12]:
        print(f"{cover:7.1%} {good:7.1%} {n/max(tiles_done,1):10.1f}  "
              f"{key[0]:5} {key[1]:5} {key[2]:7} {key[3]:7} {key[4]:6}")


if __name__ == "__main__":
    main()
