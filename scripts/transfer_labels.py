"""Carry verified labels from one capture to another of the same ground.

Labels are stored as pixel boxes inside a tile, but every tile records its
geographic bounds -- so a box can be converted to lat/lon and back into pixels
on a different capture of the same area. That lets labelling done once at a
coarse resolution be reused verbatim on sharper imagery, instead of asking the
human to draw everything again.

    python scripts/transfer_labels.py --from jbeil-nds --to jbeil-mb
"""

import argparse
import json

import _bootstrap  # noqa: F401

from solarmap.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="also carry boxes nobody has judged yet, so they can "
                         "be reviewed on the destination imagery. Useful when "
                         "the destination is sharper: a panel that is 7 px "
                         "across is guesswork, at 27 px it is obvious.")
    ap.add_argument("--include-rejected", action="store_true",
                    help="also carry boxes that were rejected, re-marked as "
                         "unreviewed so they are decided again. A rejection "
                         "made at 24.7 cm/px is a coin flip -- the same array "
                         "at 6 cm is 27 px across and obvious. Carrying the "
                         "rejection verdict itself would just re-import the "
                         "coin flip, so only the box travels, not the verdict.")
    ap.add_argument("--labels", default="labels.json",
                    help="source label file (default labels.json)")
    ap.add_argument("--out", default="labels.json",
                    help="destination label file (default labels.json)")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    src = cfg.path("captures") / args.src
    dst = cfg.path("captures") / args.dst

    src_lab = json.loads((src / args.labels).read_text(encoding="utf-8"))
    src_man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dst_man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    src_tiles = {t["tile_id"]: t for t in src_man["tiles"]}

    # Panels -> geographic rectangles, each keeping its review state so a
    # transfer never silently promotes a proposal into a confirmed label.
    geo_boxes = []
    for tid, boxes in src_lab["tiles"].items():
        t = src_tiles.get(tid)
        if not t:
            continue
        W, H = t["width"], t["height"]
        for b in boxes:
            state = b.get("verified")
            if state is False:
                if not args.include_rejected:
                    continue
                # The box travels, the verdict does not -- see --include-rejected.
                state = None
            elif state is None and not args.include_unreviewed:
                continue
            geo_boxes.append((state,
                t["west"] + b["x1"] / W * (t["east"] - t["west"]),
                t["north"] - b["y2"] / H * (t["north"] - t["south"]),
                t["west"] + b["x2"] / W * (t["east"] - t["west"]),
                t["north"] - b["y1"] / H * (t["north"] - t["south"]),
            ))
    n_conf = sum(1 for g in geo_boxes if g[0] is True)
    n_open = len(geo_boxes) - n_conf
    print(f"{n_conf} verified panels read from {args.src}"
          + (f", plus {n_open} to review" if n_open else ""))

    out: dict[str, list] = {}
    placed = 0
    for t in dst_man["tiles"]:
        W, H = t["width"], t["height"]
        rows = []
        for state, lon1, lat1, lon2, lat2 in geo_boxes:
            # Overlap test, then clip to the tile. Dropping edge-straddling
            # panels outright discarded three quarters of the labels.
            if lon2 <= t["west"] or lon1 >= t["east"]:
                continue
            if lat2 <= t["south"] or lat1 >= t["north"]:
                continue
            x1 = max(0, round((lon1 - t["west"]) / (t["east"] - t["west"]) * W))
            x2 = min(W, round((lon2 - t["west"]) / (t["east"] - t["west"]) * W))
            y1 = max(0, round((t["north"] - lat2) / (t["north"] - t["south"]) * H))
            y2 = min(H, round((t["north"] - lat1) / (t["north"] - t["south"]) * H))
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            rows.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "score": 0.0, "verified": state})
        out[t["tile_id"]] = rows
        placed += len(rows)

    (dst / args.out).write_text(
        json.dumps({"capture": args.dst, "gsd_m": dst_man["tiles"][0]["gsd_m"],
                    "source": f"transferred from {args.src}", "tiles": out}, indent=1),
        encoding="utf-8")
    print(f"{placed} panels placed on {sum(1 for v in out.values() if v)} "
          f"of {len(out)} tiles -> {args.out}")
    if placed < len(geo_boxes):
        print(f"({len(geo_boxes) - placed} fell outside the target AOI or across "
              "a tile edge)")


if __name__ == "__main__":
    main()
