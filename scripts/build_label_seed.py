"""Build a clean labelling seed on sharp imagery, for an exhaustive sweep.

Why this exists rather than a plain `transfer_labels.py` run: every label the
project has descends from proposals generated on 24.7 cm/px Esri imagery, where
a 30 m2 array is 31x16 px. Three separate problems follow from that, and the
seed has to handle each differently.

  1. CONFIRMED arrays (verified true at 25 cm) are real -- a human saw panel
     there. Their extents are trustworthy because they were drawn tight around
     the array. These carry over as accepted, and the sweep only has to nudge
     the boxes to the sharper edges now visible.

  2. REJECTED candidates (verified false at 25 cm) are not trustworthy. At 7 px
     across, "panel or water tank?" is close to a coin flip, and there are 974
     of them. The box carries over, the verdict does not -- they arrive
     unreviewed so the decision is made again at 27 px.

  3. Labels drawn on the DESTINATION imagery by an earlier pass are a mixed
     bag. On jbeil-mb they came from the classical proposer, which floods at
     6 cm: 89 of its 218 boxes are merged regions swallowing several arrays
     plus the rooftop between them (mean 236 m2 against 115 m2 for the tight
     Esri boxes). Promoting those to ground truth would mean scoring array
     recall against blobs. So a destination box that already contains a
     confirmed array is dropped as redundant, and only one that contains none
     survives -- as unreviewed, because it may be an array the coarse pass
     never saw, which is exactly what an exhaustive sweep must not miss.

What comes out is one labels.json where green means "known array, adjust the
edges", and unreviewed means "decide this at full resolution". Everything else
in the AOI is the sweep's job to draw.

    python scripts/build_label_seed.py --from jbeil-nds --to jbeil-mb
    python scripts/build_label_seed.py --from jbeil-nds --to jbeil-mb --dry-run
"""

import argparse
import json

import _bootstrap  # noqa: F401

from solarmap.config import Config


def to_geo(labels: dict, tiles: dict, want) -> list[tuple]:
    """Boxes matching `want(state)` as (state, lon1, lat1, lon2, lat2)."""
    out = []
    for tid, boxes in labels.get("tiles", {}).items():
        t = tiles.get(tid)
        if not t:
            continue
        W, H = t["width"], t["height"]
        for b in boxes:
            state = b.get("verified")
            if not want(state):
                continue
            out.append((
                state,
                t["west"] + b["x1"] / W * (t["east"] - t["west"]),
                t["north"] - b["y2"] / H * (t["north"] - t["south"]),
                t["west"] + b["x2"] / W * (t["east"] - t["west"]),
                t["north"] - b["y1"] / H * (t["north"] - t["south"]),
            ))
    return out


def area(g) -> float:
    return max(0.0, g[3] - g[1]) * max(0.0, g[4] - g[2])


def inter(a, b) -> float:
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    return ix * iy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--src-labels", default="labels.json")
    ap.add_argument("--out", default="labels.json")
    ap.add_argument("--no-rejected", action="store_true",
                    help="leave the rejected candidates out of the seed")
    ap.add_argument("--contains-frac", type=float, default=0.5,
                    help="a confirmed array counts as being inside a "
                         "destination box when this much of it is (default 0.5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the composition without writing anything")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    src = cfg.path("captures") / args.src
    dst = cfg.path("captures") / args.dst

    src_man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dst_man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    src_lab = json.loads((src / args.src_labels).read_text(encoding="utf-8"))
    src_tiles = {t["tile_id"]: t for t in src_man["tiles"]}
    dst_tiles = {t["tile_id"]: t for t in dst_man["tiles"]}

    confirmed = to_geo(src_lab, src_tiles, lambda s: s is True)
    rejected = ([] if args.no_rejected
                else to_geo(src_lab, src_tiles, lambda s: s is False))
    print(f"source {args.src}: {len(confirmed)} confirmed, {len(rejected)} rejected")

    # Existing destination labels, if any -- see point 3 in the module docstring.
    dst_existing = []
    dst_path = dst / "labels.json"
    if dst_path.exists():
        dst_lab = json.loads(dst_path.read_text(encoding="utf-8"))
        dst_existing = to_geo(dst_lab, dst_tiles, lambda s: s is not False)
        print(f"destination {args.dst}: {len(dst_existing)} existing boxes")

    kept_existing, dropped_redundant = [], 0
    for g in dst_existing:
        a = area(g)
        holds_known = any(
            a > 0 and inter(g, c) / max(area(c), 1e-18) >= args.contains_frac
            for c in confirmed)
        if holds_known:
            dropped_redundant += 1
        else:
            kept_existing.append(g)

    # A rejection sitting on an array we already confirmed is not a question
    # worth re-asking; it just adds clutter to the sweep.
    kept_rejected, dropped_on_known = [], 0
    for g in rejected:
        a = area(g)
        if a > 0 and any(inter(g, c) / a >= 0.5 for c in confirmed):
            dropped_on_known += 1
        else:
            kept_rejected.append(g)

    print(f"  dropped {dropped_redundant} destination boxes that already "
          f"contain a confirmed array (merged proposals)")
    print(f"  dropped {dropped_on_known} rejections lying on a confirmed array")

    geo = ([(True, *g[1:]) for g in confirmed]
           + [(None, *g[1:]) for g in kept_rejected]
           + [(None, *g[1:]) for g in kept_existing])

    out: dict[str, list] = {}
    placed = n_acc = n_open = 0
    for t in dst_man["tiles"]:
        W, H = t["width"], t["height"]
        rows = []
        for state, lon1, lat1, lon2, lat2 in geo:
            # Overlap then clip. Dropping edge-straddling panels outright
            # discarded three quarters of the labels.
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
            n_acc += state is True
            n_open += state is None
        out[t["tile_id"]] = rows
        placed += len(rows)

    tiles_with = sum(1 for v in out.values() if v)
    print(f"\nseed: {placed} boxes on {tiles_with} of {len(out)} tiles")
    print(f"  {n_acc} accepted  (known arrays -- adjust edges)")
    print(f"  {n_open} unreviewed (decide at 6 cm)")
    print(f"  ~{placed / max(len(out), 1):.0f} boxes per tile")
    print(f"\n{len(out) - tiles_with} tiles carry no seed at all and must still "
          "be swept -- an array there has never been labelled at any resolution.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    target = dst / args.out
    if target.exists():
        backup = dst / (target.stem + ".pre-seed.json")
        if backup.exists():
            raise SystemExit(
                f"{backup.name} already exists -- refusing to overwrite a second "
                f"time and lose the original. Move it aside first.")
        backup.write_bytes(target.read_bytes())
        print(f"\nexisting {target.name} backed up to {backup.name}")

    target.write_text(
        json.dumps({"capture": args.dst,
                    "gsd_m": dst_man["tiles"][0]["gsd_m"],
                    "source": f"seed from {args.src} ({args.src_labels})",
                    "tiles": out}, indent=1),
        encoding="utf-8")
    print(f"wrote {target}")
    print(f"\nreview with:  python scripts/serve.py   ->  /label.html")


if __name__ == "__main__":
    main()
