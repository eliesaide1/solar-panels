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
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    src = cfg.path("captures") / args.src
    dst = cfg.path("captures") / args.dst

    src_lab = json.loads((src / "labels.json").read_text(encoding="utf-8"))
    src_man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dst_man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    src_tiles = {t["tile_id"]: t for t in src_man["tiles"]}

    # Verified panels -> geographic rectangles.
    geo_boxes = []
    for tid, boxes in src_lab["tiles"].items():
        t = src_tiles.get(tid)
        if not t:
            continue
        W, H = t["width"], t["height"]
        for b in boxes:
            if b.get("verified") is not True:
                continue
            geo_boxes.append((
                t["west"] + b["x1"] / W * (t["east"] - t["west"]),
                t["north"] - b["y2"] / H * (t["north"] - t["south"]),
                t["west"] + b["x2"] / W * (t["east"] - t["west"]),
                t["north"] - b["y1"] / H * (t["north"] - t["south"]),
            ))
    print(f"{len(geo_boxes)} verified panels read from {args.src}")

    out: dict[str, list] = {}
    placed = 0
    for t in dst_man["tiles"]:
        W, H = t["width"], t["height"]
        rows = []
        for lon1, lat1, lon2, lat2 in geo_boxes:
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
                         "score": 0.0, "verified": True})
        out[t["tile_id"]] = rows
        placed += len(rows)

    (dst / "labels_transferred.json").write_text(
        json.dumps({"capture": args.dst, "gsd_m": dst_man["tiles"][0]["gsd_m"],
                    "source": f"transferred from {args.src}", "tiles": out}, indent=1),
        encoding="utf-8")
    print(f"{placed} panels placed on {sum(1 for v in out.values() if v)} "
          f"of {len(out)} tiles -> labels_transferred.json")
    if placed < len(geo_boxes):
        print(f"({len(geo_boxes) - placed} fell outside the target AOI or across "
              "a tile edge)")


if __name__ == "__main__":
    main()
