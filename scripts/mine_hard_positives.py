"""Add the arrays the model cannot see to a training set, weighted to matter.

The mirror of ``mine_hard_negatives.py``, and it exists because of a measurement
that rules out the obvious approach. At Jbeil the detector misses 57 verified
arrays outright -- its peak response inside them is 0.013, against 0.982 on the
ones it finds -- and **42 of those 57 were already in its training set**. They
were seen, with correct masks, and the model learned to ignore them anyway.

Adding them again changes nothing, because the problem is not that they are
absent from training but that they are drowned in it. A 30 m2 array at 6 cm is
roughly 8,000 pixels in a 512x512 crop: 3% of it. Across 806 crops the whole
population of missed arrays is a fraction of a percent of the loss, so the
cheapest thing the optimiser can do is call them background and spend its
capacity elsewhere. Nothing in the gradient makes that a bad trade.

So this cuts crops centred on each missed array, several per array with jitter,
which puts the array near the middle of the frame at several offsets instead of
clipped into a corner of one sliding window. Oversampling is the only lever
here that changes what the loss actually weighs.

The mask is built from the verified labels, exactly as the negative miner does
-- never forced, so neighbouring arrays in the same crop stay labelled.

**It refuses to mine from validation tiles.** Those crops would land in the
train split, and the val split is the only estimate of generalisation this
project has. At Jbeil that estimate is already thin: the four shipped models
were trained on different 75% samples of the same 81 tiles, and only 2 tiles
are held out by all of them. Contaminating what is left would make the result
of this experiment unreadable, which is the one outcome worth avoiding.

    python scripts/mine_hard_positives.py --detections jbeil-mb-104 \
        --det-file detections_raw.geojson --labels labels_clean.json \
        --imagery jbeil-mb --into jbeil06_pos
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np
from PIL import Image

from solarmap.config import Config

Image.MAX_IMAGE_PIXELS = None


def resolve_capture(cfg: Config, name: str):
    """Find a capture under data/captures, data/archive, or at a literal path.

    The sharpest imagery for a region is often the capture that has been
    archived -- detection moved to a resampled copy, and the original was set
    aside rather than deleted. Refusing to look there would push mining onto
    coarser pixels than the model was fine-tuned on.
    """
    from pathlib import Path

    direct = Path(name)
    if (direct / "manifest.json").is_file():
        return direct
    captures = cfg.path("captures")
    for root in (captures, captures.parent / "archive"):
        cand = root / name
        if (cand / "manifest.json").is_file():
            return cand
    raise SystemExit(
        f"No capture {name!r} with a manifest.json under {captures}, "
        f"{captures.parent / 'archive'}, or as a path."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True,
                    help="capture the detections were produced on")
    ap.add_argument("--det-file", default="detections.geojson",
                    help="use the UNCURATED layer. A curated one has had its "
                         "false positives deleted, which does not change which "
                         "arrays were missed, but nothing else in this project "
                         "should encourage reaching for the curated file.")
    ap.add_argument("--labels", default="labels_clean.json",
                    help="verified labels on the DETECTION capture")
    ap.add_argument("--imagery", required=True,
                    help="capture to cut the crops from (usually the sharpest)")
    ap.add_argument("--imagery-labels", default="labels.json",
                    help="verified labels on the IMAGERY capture")
    ap.add_argument("--into", required=True, help="dataset under data/datasets")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=8,
                    help="jittered crops per missed array. This is the whole "
                         "mechanism: it decides what share of the loss these "
                         "arrays carry. 6 matched the negative miner; 8 is the "
                         "default here because a missed array is rarer than a "
                         "false positive and starts from a lower base. Raise it "
                         "if the model still ignores them, but watch precision "
                         "-- oversampling a class far past its true frequency "
                         "buys recall with false positives, and this project "
                         "has already measured that trade going badly.")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="an array counts as found, and is skipped, when this "
                         "much of it is covered by the detection union. Matches "
                         "score_arrays.py so 'missed' means the same thing in "
                         "both places.")
    ap.add_argument("--jitter", type=float, default=0.25,
                    help="crop centre jitter, as a fraction of the crop size. "
                         "Smaller than the negative miner's third: the array "
                         "must stay in frame, where a fooling texture only had "
                         "to stay nearby.")
    ap.add_argument("--allow-val-tiles", action="store_true",
                    help="mine validation tiles too. This contaminates the only "
                         "held-out set with training data and makes the "
                         "experiment unfalsifiable. There is no good reason to "
                         "pass this.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    src = resolve_capture(cfg, args.detections)
    img_cap = resolve_capture(cfg, args.imagery)
    root = cfg.path("datasets") / args.into
    if not (root / "train" / "images").is_dir():
        raise SystemExit(f"{root} has no train/images -- build it with "
                         f"prepare_masks.py first, then add positives here.")

    src_man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    src_lab = json.loads((src / args.labels).read_text(encoding="utf-8"))
    dets = json.loads((src / args.det_file).read_text(encoding="utf-8"))
    img_man = json.loads((img_cap / "manifest.json").read_text(encoding="utf-8"))
    img_lab = json.loads((img_cap / args.imagery_labels).read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    # --- which tiles are held out, read off the dataset rather than re-derived ---
    # prepare_masks.py names every crop "{tile_id}_{ox}_{oy}", and its split is
    # by tile, so the val tile set is recoverable from the filenames. Deriving
    # it from the RNG instead would silently disagree the moment --seed or
    # --include-empty-tiles differs, which is exactly how the four shipped
    # models ended up with three different splits.
    val_tiles = {p.name.rsplit("_", 2)[0]
                 for p in (root / "val" / "images").glob("*.png")}
    if args.allow_val_tiles:
        print("WARNING: mining validation tiles. The held-out score after this "
              "measures memorisation, not generalisation.")
        val_tiles = set()
    elif not val_tiles:
        print(f"WARNING: no val split found under {root / 'val' / 'images'}; "
              "cannot tell which tiles are held out, so nothing is excluded.")
    else:
        print(f"{len(val_tiles)} validation tiles will not be mined")

    # --- which verified arrays did the detector miss ---
    missed = []            # (lon, lat) of each missed array's centre
    skipped_val = 0
    for t in src_man["tiles"]:
        tid = t["tile_id"]
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]
        panels = [b for b in src_lab["tiles"].get(tid, [])
                  if b.get("verified") is True]
        if not panels:
            continue

        union = np.zeros((H, W), bool)
        for f in dets["features"]:
            ring = f["geometry"]["coordinates"][0]
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            if not (w <= (min(xs) + max(xs)) / 2 <= e):
                continue
            if not (s <= (min(ys) + max(ys)) / 2 <= n):
                continue
            poly = np.array([[(lon - w) / (e - w) * W, (n - lat) / (n - s) * H]
                             for lon, lat in ring], np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            union |= m.astype(bool)

        for b in panels:
            y1, y2 = max(0, b["y1"]), b["y2"]
            x1, x2 = max(0, b["x1"]), b["x2"]
            area = max(1, (y2 - y1) * (x2 - x1))
            if union[y1:y2, x1:x2].sum() / area >= args.min_cover:
                continue                      # the model found this one
            if tid in val_tiles:
                skipped_val += 1
                continue
            cx = w + (x1 + x2) / 2.0 / W * (e - w)
            cy = n - (y1 + y2) / 2.0 / H * (n - s)
            missed.append((cx, cy))

    n_verified = sum(1 for bs in src_lab["tiles"].values()
                     for b in bs if b.get("verified") is True)
    print(f"{len(missed)} missed arrays to mine (of {n_verified} verified)"
          + (f"; {skipped_val} more skipped on validation tiles" if skipped_val else ""))
    if not missed:
        return

    # --- cut jittered crops around each, from the imagery capture ---
    S = args.crop
    jit = max(1, int(S * args.jitter))
    added = 0
    cache: dict[str, tuple] = {}
    for cx, cy in missed:
        host = next((t for t in img_man["tiles"]
                     if t["west"] <= cx <= t["east"] and t["south"] <= cy <= t["north"]),
                    None)
        if host is None:
            continue
        tid = host["tile_id"]
        if tid not in cache:
            with Image.open(img_cap / host["image"]) as im:
                im = im.convert("RGB")
                W, H = im.size
                mask = np.zeros((H, W), np.uint8)
                for b in img_lab["tiles"].get(tid, []):
                    if b.get("verified") is True:
                        mask[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = 255
                cache[tid] = (im.copy(), mask)
        im, mask = cache[tid]
        W, H = im.size
        px = int((cx - host["west"]) / (host["east"] - host["west"]) * W)
        py = int((host["north"] - cy) / (host["north"] - host["south"]) * H)

        for _ in range(args.repeats):
            jx = int(rng.integers(-jit, jit + 1))
            jy = int(rng.integers(-jit, jit + 1))
            ox = int(np.clip(px - S // 2 + jx, 0, max(0, W - S)))
            oy = int(np.clip(py - S // 2 + jy, 0, max(0, H - S)))
            sub = mask[oy:oy + S, ox:ox + S]
            if sub.shape != (S, S):
                pad = np.zeros((S, S), np.uint8)
                pad[:sub.shape[0], :sub.shape[1]] = sub
                sub = pad
            if not sub.any():
                # The jitter pushed the array out of frame, or a clip against
                # the tile edge did. An all-background crop labelled as a mined
                # positive is the opposite of the point.
                continue
            name = f"hardpos_{tid}_{ox}_{oy}"
            im.crop((ox, oy, ox + S, oy + S)).save(
                root / "train" / "images" / f"{name}.png")
            Image.fromarray(sub).save(root / "train" / "masks" / f"{name}.png")
            added += 1

    total = len(list((root / "train" / "images").glob("*")))
    mined = len(list((root / "train" / "images").glob("hardpos_*")))
    # Crops are named by offset, so two jitters landing on the same pixel write
    # the same file. That is the right behaviour -- an identical crop stored
    # twice is not a second example -- but it means `added` counts saves, not
    # distinct crops, and only the latter changes the loss.
    share = 100.0 * mined / max(total, 1)
    print(f"{added} crops written, {mined} distinct "
          f"({added - mined} were duplicate offsets) -> {root}")
    print(f"train split is now {total} crops, {share:.0f}% of them mined")

    if share > 30.0:
        print(f"\nWARNING: mined crops are {share:.0f}% of the train split, far "
              "above these arrays' true frequency. That is the regime where "
              "oversampling buys recall with false positives. Consider "
              f"--repeats {max(1, args.repeats // 2)}, and compare both against "
              "the val tiles before believing either.")

    print("\nFine-tune from the existing checkpoint, then score the VAL tiles "
          "only. A gain measured on tiles that were trained on is not a gain.")


if __name__ == "__main__":
    main()
