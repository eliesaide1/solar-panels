"""Export verified labels as a YOLO detection dataset.

Only **fully reviewed** tiles are exported. A tile with unreviewed boxes may
contain panels nobody has confirmed, and in detection training every unlabelled
object is an explicit "this is background" signal -- so including such a tile
would actively teach the model to ignore real arrays.

    python scripts/export_yolo.py --capture jbeil-nds --name jbeil
"""

import argparse
import json
import random
import shutil

import _bootstrap  # noqa: F401

from PIL import Image

from solarmap.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--name", required=True, help="dataset name under data/datasets")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep-empty", action="store_true",
                    help="also export reviewed tiles with no panels, as negatives")
    ap.add_argument("--slice", type=int, default=0,
                    help="slice tiles into NxN crops (e.g. 512). 0 = whole tiles")
    ap.add_argument("--slice-overlap", type=float, default=0.5,
                    help="fraction of overlap between adjacent slices")
    ap.add_argument("--min-visible", type=float, default=0.35,
                    help="drop a box from a crop if less than this fraction of it survives")
    ap.add_argument("--neg-frac", type=float, default=0.15,
                    help="share of empty slices to keep as negatives")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels = json.loads((cap / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))

    reviewed, skipped = [], []
    for t in manifest["tiles"]:
        tid = t["tile_id"]
        boxes = labels["tiles"].get(tid, [])
        if boxes and any(b.get("verified") is None for b in boxes):
            skipped.append(tid)
            continue
        panels = [b for b in boxes if b.get("verified") is True]
        if panels or args.keep_empty:
            reviewed.append((tid, panels))

    if skipped:
        print(f"skipping {len(skipped)} tiles with unreviewed boxes: "
              f"{', '.join(skipped[:4])}{'...' if len(skipped) > 4 else ''}")
    if not reviewed:
        raise SystemExit("No fully reviewed tiles to export.")

    # Split by tile, never by box -- boxes from one tile must not straddle the
    # split, or validation leaks training pixels and the score is meaningless.
    #
    # Panels are very unevenly distributed across tiles (one campus tile holds
    # dozens, most residential tiles hold one or two), so a split by tile count
    # lands wildly off target by box count. Assign greedily instead: take the
    # densest tiles first and give each to whichever side is furthest below its
    # quota.
    rng = random.Random(args.seed)
    rng.shuffle(reviewed)
    reviewed.sort(key=lambda r: -len(r[1]))

    total_boxes = sum(len(p) for _, p in reviewed)
    target_val = total_boxes * args.val_frac
    train, val = [], []
    n_train = n_val = 0
    for tid, panels in reviewed:
        # Deficit relative to each side's share of the remaining budget.
        val_deficit = target_val - n_val
        train_deficit = (total_boxes - target_val) - n_train
        if val_deficit / max(target_val, 1) > train_deficit / max(total_boxes - target_val, 1):
            val.append((tid, panels)); n_val += len(panels)
        else:
            train.append((tid, panels)); n_train += len(panels)

    root = cfg.path("datasets") / args.name
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = root / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    def write_one(split, name, image, panels, W, H):
        image.save(root / split / "images" / f"{name}.jpg", quality=92)
        lines = []
        for x1, y1, x2, y2 in panels:
            # YOLO format: class cx cy w h, all normalised to [0, 1].
            lines.append(
                f"0 {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} "
                f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"
            )
        (root / split / "labels" / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    counts = {}
    for split, items in (("train", train), ("val", val)):
        n_img = n_box = 0
        neg_rng = random.Random(args.seed + 1)
        for tid, panels in items:
            src = cap / "tiles" / f"{tid}.jpg"
            with Image.open(src) as im:
                im = im.convert("RGB")
                W, H = im.size

                if not args.slice:
                    boxes = [(max(0, b["x1"]), max(0, b["y1"]),
                              min(W, b["x2"]), min(H, b["y2"])) for b in panels]
                    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
                    n_box += write_one(split, tid, im, boxes, W, H)
                    n_img += 1
                    continue

                S = args.slice
                step = max(1, int(S * (1 - args.slice_overlap)))
                for oy in range(0, max(1, H - S + 1), step):
                    for ox in range(0, max(1, W - S + 1), step):
                        cw, ch = min(S, W - ox), min(S, H - oy)
                        kept = []
                        for b in panels:
                            ix1, iy1 = max(b["x1"], ox), max(b["y1"], oy)
                            ix2, iy2 = min(b["x2"], ox + cw), min(b["y2"], oy + ch)
                            if ix2 <= ix1 or iy2 <= iy1:
                                continue
                            full = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                            # A sliver of a panel at a crop edge is a bad
                            # training target -- it teaches the model that a
                            # fragment is a whole array.
                            if full <= 0 or ((ix2 - ix1) * (iy2 - iy1)) / full < args.min_visible:
                                continue
                            kept.append((ix1 - ox, iy1 - oy, ix2 - ox, iy2 - oy))

                        if not kept and neg_rng.random() > args.neg_frac:
                            continue
                        crop = im.crop((ox, oy, ox + cw, oy + ch))
                        n_box += write_one(split, f"{tid}_{ox}_{oy}", crop, kept, cw, ch)
                        n_img += 1
        counts[split] = (n_img, n_box)

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 1\n"
        "names: ['solar_panel']\n",
        encoding="utf-8",
    )

    for split, (nt, nb) in counts.items():
        print(f"{split}: {nt} tiles, {nb} boxes")
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
