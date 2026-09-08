"""Build a segmentation dataset from verified box labels.

Boxes become filled rectangles in the mask. That is an approximation -- a box
over a tilted or L-shaped array includes some roof -- so the model learns
slightly generous footprints. They are still far tighter than the bounding
boxes the current detector emits, which is the point: segmentation turns area
from an upper bound into a measurement.

Only fully-reviewed tiles are used. A tile with unreviewed boxes may hold an
unlabelled panel, and in segmentation that pixel is explicitly taught to be
background.

    python scripts/prepare_masks.py --capture jbeil-nds --name jbeil_seg
"""

import argparse
import json
import shutil

import _bootstrap  # noqa: F401

import cv2
import numpy as np
from PIL import Image

from solarmap.config import Config

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", default="labels.json",
                    help="label file to build masks from. On a resampled "
                         "capture this is usually labels_clean.json -- its "
                         "own labels.json is whatever the capture shipped with.")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--neg-frac", type=float, default=0.25,
                    help="share of panel-free crops to keep, as negatives")
    ap.add_argument("--include-empty-tiles", action="store_true",
                    help="also take crops from tiles that were reviewed and "
                         "hold no panel at all. Without this the model never "
                         "sees the parts of a region that contain only "
                         "look-alikes: at Jbeil the greenhouse hillsides were "
                         "34 of 81 tiles and were absent from training, so "
                         "polytunnels were detected as arrays. Only safe on an "
                         "exhaustively swept capture -- an unvisited tile is "
                         "indistinguishable from an empty one, and teaching a "
                         "real array as background is worse than omitting it.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    usable = []
    for t in manifest["tiles"]:
        boxes = labels["tiles"].get(t["tile_id"], [])
        if boxes and any(b.get("verified") is None for b in boxes):
            continue
        panels = [b for b in boxes if b.get("verified") is True]
        if panels or args.include_empty_tiles:
            usable.append((t["tile_id"], panels))

    rng.shuffle(usable)
    n_val = max(1, int(len(usable) * args.val_frac))
    val_ids = {tid for tid, _ in usable[:n_val]}
    n_empty = sum(1 for _, p in usable if not p)
    print(f"{len(usable)} tiles ({len(usable)-n_empty} with panels, "
          f"{n_empty} pure background) -> {len(val_ids)} held out")

    root = cfg.path("datasets") / args.name
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            d = root / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts = {"train": [0, 0], "val": [0, 0]}
    S, step = args.crop, max(1, int(args.crop * (1 - args.overlap)))

    for tid, panels in usable:
        split = "val" if tid in val_ids else "train"
        with Image.open(cap / "tiles" / f"{tid}.jpg") as im:
            im = im.convert("RGB")
            W, H = im.size
            mask = np.zeros((H, W), np.uint8)
            for b in panels:
                # A hand-drawn outline, where one exists, is the real footprint.
                # Filling the bounding box instead is the approximation this
                # module's docstring warns about -- a box over a tilted or
                # L-shaped array includes roof, and the model learns to trace
                # generous footprints. apply_corrections.py writes `poly` from
                # polygons drawn in the map UI.
                poly = b.get("poly")
                if poly and len(poly) >= 3:
                    cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 255)
                else:
                    mask[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = 255

            for oy in range(0, max(1, H - S + 1), step):
                for ox in range(0, max(1, W - S + 1), step):
                    sub_mask = mask[oy:oy + S, ox:ox + S]
                    # PIL.crop pads past the edge; the numpy slice does not.
                    # Pad the mask to match, or albumentations rejects the pair.
                    if sub_mask.shape != (S, S):
                        padded = np.zeros((S, S), np.uint8)
                        padded[:sub_mask.shape[0], :sub_mask.shape[1]] = sub_mask
                        sub_mask = padded
                    has = sub_mask.any()
                    # Keep only a fraction of empty crops, or the model sees
                    # almost nothing but background and predicts all-zero.
                    if not has and rng.random() > args.neg_frac:
                        continue
                    name = f"{tid}_{ox}_{oy}"
                    im.crop((ox, oy, ox + S, oy + S)).save(
                        root / split / "images" / f"{name}.png")
                    Image.fromarray(sub_mask).save(root / split / "masks" / f"{name}.png")
                    counts[split][0] += 1
                    counts[split][1] += int(has)

    for split, (n, pos) in counts.items():
        print(f"{split}: {n} crops ({pos} containing panels)")
    print(f"-> {root}")


if __name__ == "__main__":
    main()
