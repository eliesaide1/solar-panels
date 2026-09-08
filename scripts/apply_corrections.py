"""Fold map corrections into the labels, so they train the model.

The map UI writes ``corrections.geojson``: polygons a human drew over arrays
the layer got wrong. Nothing reads that file automatically, because a
correction is a claim about the ground and the ground truth is the labels --
merging them is a decision, not a side effect.

Two kinds:

  add     a real array the detector missed, or traced badly. Becomes a verified
          label. The drawn outline is kept as ``poly``, alongside the bounding
          box every existing script already understands, so prepare_masks.py
          can rasterise the true shape instead of a rectangle. Its own docstring
          notes that filling boxes teaches the model "slightly generous
          footprints"; a drawn polygon is the fix.

  remove  the detection here is not an array. Any verified label whose centre
          falls inside the polygon is flipped to rejected -- NOT deleted, so
          the decision stays visible and reversible in the label editor, and so
          a later transfer cannot quietly resurrect it.

The output is written to a NEW label file by default. Overwriting the labels a
published number was scored against would silently move that number.

    python scripts/apply_corrections.py --capture jbeil-mb-104 \
        --labels labels_reviewed.json --out labels_corrected.json
"""

import argparse
import json

import _bootstrap  # noqa: F401

from solarmap.config import Config


def point_in_ring(x: float, y: float, ring: list) -> bool:
    """Ray casting. Small enough to inline; avoids a shapely round-trip per box."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x < xin:
                inside = not inside
    return inside


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels_reviewed.json",
                    help="label file to start from")
    ap.add_argument("--corrections", default="corrections.geojson")
    ap.add_argument("--out", default="labels_corrected.json",
                    help="where the merged labels go. Defaults to a new file: "
                         "overwriting the labels a published score was measured "
                         "against moves that score without saying so.")
    ap.add_argument("--in-place", action="store_true",
                    help="write back over --labels instead. The previous "
                         "contents are copied to <name>.bak first.")
    ap.add_argument("--max-removals", type=int, default=3,
                    help="refuse a single 'mark wrong' polygon that would "
                         "reject more than this many verified arrays. A "
                         "removal is drawn to kill one bad detection, but it "
                         "rejects every label whose centre lands inside it -- "
                         "one loosely drawn rectangle took out 11 real arrays "
                         "including a 1,585 m2 installation, which would have "
                         "gone silently into the training masks as background. "
                         "Raise it deliberately, or redraw the polygon tighter.")
    ap.add_argument("--force", action="store_true",
                    help="apply removals that exceed --max-removals anyway")
    ap.add_argument("--min-overlap", type=float, default=0.30,
                    help="fraction of a labelled array that an erased shape "
                         "must cover for that label to be rejected. Kept low "
                         "because the erased shape is the DETECTION's outline, "
                         "which is routinely smaller than the box drawn round "
                         "the array.")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((cap / args.labels).read_text(encoding="utf-8"))

    cpath = cap / args.corrections
    if not cpath.is_file():
        raise SystemExit(f"No {args.corrections} in {cap}. Draw some in the map "
                         "UI first (section 4, Correct).")
    corr = json.loads(cpath.read_text(encoding="utf-8"))
    feats = corr.get("features", [])
    if not feats:
        raise SystemExit("No corrections to apply.")

    tiles = man["tiles"]
    n_add = n_rej = n_outside = n_refused = 0
    refused: list[tuple[str, int, float]] = []

    # Undo any previous application first, so this script is idempotent.
    # Without it, running twice appends every drawn array a second time -- and
    # the map UI's Apply button runs it on a file that may already contain the
    # last run's output, so "apply twice" is the normal case, not the odd one.
    # Corrections.geojson is the source of truth for UI-made edits; the labels
    # are rebuilt from it each time.
    n_undone = n_restored = 0
    for tid, rows in list(lab.get("tiles", {}).items()):
        kept = []
        for b in rows:
            if b.get("poly") or b.get("corrected") == "drawn in the map UI":
                # Keyed on the drawn outline itself, not on the `corrected`
                # tag: a label that was drawn and later rejected has had that
                # tag overwritten, and would then survive as a stale entry to
                # be re-rejected on every run.
                n_undone += 1
                continue                     # re-added below if still present
            if b.get("corrected") == "marked not-a-panel in the map UI":
                # A rejection whose correction has since been erased must come
                # back, or deleting a correction would be a one-way door.
                b["verified"] = True
                b.pop("corrected", None)
                n_restored += 1
            kept.append(b)
        lab["tiles"][tid] = kept

    # Removals first, additions second -- and never the reverse.
    #
    # Reshaping an outline in the map UI emits "remove the old shape" plus "add
    # the new one", and the two overlap by construction. Processed in file
    # order, the add created the label and the removal immediately rejected it,
    # so every reshaped or redrawn array silently cancelled itself and Apply
    # looked like it had done nothing. Ordering the passes makes a removal act
    # only on ground that was already labelled.
    feats = sorted(feats, key=lambda f: f.get("properties", {}).get("kind") != "remove")

    for f in feats:
        ring_ll = f["geometry"]["coordinates"][0]
        kind = f.get("properties", {}).get("kind", "add")
        xs = [c[0] for c in ring_ll]
        ys = [c[1] for c in ring_ll]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

        host = next((t for t in tiles
                     if t["west"] <= cx <= t["east"] and t["south"] <= cy <= t["north"]),
                    None)
        if host is None:
            n_outside += 1
            continue

        tid = host["tile_id"]
        W, H = host["width"], host["height"]
        n, s, e, w = host["north"], host["south"], host["east"], host["west"]
        px = [(lon - w) / (e - w) * W for lon, _ in ring_ll]
        py = [(n - lat) / (n - s) * H for _, lat in ring_ll]
        rows = lab.setdefault("tiles", {}).setdefault(tid, [])

        if kind == "remove":
            ring_px = list(zip(px, py))
            # Match on OVERLAP, not on the label's centre. A detection is often
            # smaller than the box drawn round it, or offset within it, so the
            # box centre falls outside the erased shape and a centre test flips
            # nothing -- measured on this capture, 12 of 40 small shapes could
            # not be erased at all for exactly that reason.
            rx1, rx2 = int(min(px)), int(max(px))
            ry1, ry2 = int(min(py)), int(max(py))
            hits = []
            for b in rows:
                if b.get("verified") is not True:
                    continue
                if b.get("poly") or b.get("corrected") == "drawn in the map UI":
                    # Somebody traced this by hand. A removal is a statement
                    # about the detector's output, not about a human's tracing.
                    continue
                ox = max(0, min(b["x2"], rx2) - max(b["x1"], rx1))
                oy = max(0, min(b["y2"], ry2) - max(b["y1"], ry1))
                inter = ox * oy
                box = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
                if (inter / box >= args.min_overlap
                        or point_in_ring((b["x1"] + b["x2"]) / 2.0,
                                         (b["y1"] + b["y2"]) / 2.0, ring_px)):
                    hits.append(b)
            if len(hits) > args.max_removals and not args.force:
                gsd = float(host.get("gsd_m", 0.1))
                areas = sorted(((b["x2"] - b["x1"]) * gsd * (b["y2"] - b["y1"]) * gsd)
                               for b in hits)
                refused.append((tid, len(hits), areas[-1]))
                n_refused += 1
                continue
            for b in hits:
                b["verified"] = False
                b["corrected"] = "marked not-a-panel in the map UI"
                n_rej += 1
            continue

        x1, x2 = int(round(min(px))), int(round(max(px)))
        y1, y2 = int(round(min(py))), int(round(max(py)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            n_outside += 1
            continue
        rows.append({
            "x1": max(0, x1), "y1": max(0, y1),
            "x2": min(W, x2), "y2": min(H, y2),
            "score": 1.0,
            "verified": True,
            # The drawn shape, in this tile's pixels. Everything that only
            # knows about boxes keeps working; prepare_masks.py uses this to
            # rasterise the real outline.
            "poly": [[int(round(a)), int(round(b))] for a, b in zip(px, py)],
            "corrected": "drawn in the map UI",
        })
        n_add += 1

    out_path = cap / (args.labels if args.in_place else args.out)
    if args.in_place:
        bak = cap / (args.labels + ".bak")
        bak.write_text(json.dumps(json.loads((cap / args.labels).read_text(encoding="utf-8"))),
                       encoding="utf-8")
        print(f"previous labels copied to {bak.name}")
    out_path.write_text(json.dumps(lab, indent=1), encoding="utf-8")

    verified = sum(1 for bs in lab["tiles"].values()
                   for b in bs if b.get("verified") is True)
    polys = sum(1 for bs in lab["tiles"].values() for b in bs if b.get("poly"))
    if n_undone or n_restored:
        print(f"(reapplied cleanly: dropped {n_undone} previously drawn, "
              f"restored {n_restored} previously rejected)")
    print(f"{len(feats)} corrections applied")
    print(f"  {n_add} arrays added (with drawn outlines)")
    print(f"  {n_rej} existing labels flipped to rejected")
    if n_outside:
        print(f"  {n_outside} skipped (outside the capture, or too small)")
    if refused:
        print(f"\n  {n_refused} removal polygon(s) REFUSED -- each would reject "
              f"more than {args.max_removals} verified arrays:")
        for tid, cnt, biggest in refused:
            print(f"    {cnt:3} arrays on {tid} (largest {biggest:,.0f} m2)")
        print("    Redraw them tighter around the single detection you meant,")
        print("    or pass --force if you really mean to reject that many.")
    print(f"\n{verified} verified arrays, {polys} of them with a traced outline")
    print(f"-> {out_path}")
    print("\nRebuild the training set to use them:")
    print(f"  python scripts/prepare_masks.py --capture {args.capture} "
          f"--labels {out_path.name} --name <dataset>")


if __name__ == "__main__":
    main()
