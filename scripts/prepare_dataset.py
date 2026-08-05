"""Turn a downloaded PV segmentation dataset into the train/val layout.

Works with any dataset that stores images and masks in two parallel folders
with matching filenames -- BDAPPV's ``google/img`` + ``google/mask`` is the
intended case:

    python scripts/prepare_dataset.py --src data/raw/bdappv/google --name bdappv

Images without a corresponding mask are dropped, since BDAPPV ships many
unannotated negatives alongside the labelled set. Use --keep-negatives to
retain them as all-zero masks, which helps suppress false positives on
skylights, dark flat roofs and pools.
"""

import argparse
import random
import shutil

import _bootstrap  # noqa: F401

import numpy as np
from PIL import Image
from tqdm import tqdm

from solarmap.config import Config

IMG_DIRS = ("img", "images", "image")
MASK_DIRS = ("mask", "masks", "label", "labels")


def find_pair(src):
    img_dir = next((src / d for d in IMG_DIRS if (src / d).is_dir()), None)
    mask_dir = next((src / d for d in MASK_DIRS if (src / d).is_dir()), None)
    if img_dir is None or mask_dir is None:
        raise SystemExit(
            f"Could not find image/mask folders under {src}. Expected one of "
            f"{IMG_DIRS} and one of {MASK_DIRS}."
        )
    return img_dir, mask_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="folder containing img/ and mask/")
    ap.add_argument("--name", required=True, help="output dataset name")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--keep-negatives", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from pathlib import Path

    src = Path(args.src)
    img_dir, mask_dir = find_pair(src)

    masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}
    images = sorted(p for p in img_dir.iterdir() if p.is_file())

    pairs = []
    negatives = []
    for img in images:
        (pairs if img.stem in masks else negatives).append(img)

    if args.keep_negatives:
        # Cap negatives at the positive count so the loss is not swamped by
        # empty tiles.
        random.Random(args.seed).shuffle(negatives)
        chosen_neg = negatives[: len(pairs)]
    else:
        chosen_neg = []

    print(f"{len(pairs)} annotated, {len(negatives)} unannotated "
          f"({len(chosen_neg)} kept as negatives)")
    if not pairs:
        raise SystemExit("No image/mask pairs found -- check --src.")

    items = [(p, masks[p.stem]) for p in pairs] + [(p, None) for p in chosen_neg]
    random.Random(args.seed).shuffle(items)
    split = int(len(items) * (1.0 - args.val_frac))

    out_root = Config.load().path("datasets") / args.name
    for sub in ("train/images", "train/masks", "val/images", "val/masks"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    for i, (img_path, mask_path) in enumerate(tqdm(items, desc="copying")):
        split_name = "train" if i < split else "val"
        dst_img = out_root / split_name / "images" / f"{img_path.stem}.png"
        dst_mask = out_root / split_name / "masks" / f"{img_path.stem}.png"

        if img_path.suffix.lower() == ".png":
            shutil.copyfile(img_path, dst_img)
        else:
            Image.open(img_path).convert("RGB").save(dst_img)

        if mask_path is None:
            with Image.open(img_path) as im:
                w, h = im.size
            Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(dst_mask)
        else:
            with Image.open(mask_path) as m:
                # Datasets encode masks variously as 0/1, 0/255 or palettes;
                # normalise everything to 0/255 single-channel.
                arr = np.array(m.convert("L"))
            Image.fromarray(((arr > 0) * 255).astype(np.uint8)).save(dst_mask)

    print(f"train={split}  val={len(items) - split}  ->  {out_root}")


if __name__ == "__main__":
    main()
