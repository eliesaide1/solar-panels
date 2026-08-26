"""Add the model's own false positives to a training set, as negatives.

Feeding whole array-free tiles into training did nothing: measured at Jbeil,
250 extra crops of greenhouse hillsides left false positives at 37 against 36
for a model that had never seen a greenhouse. Most of those crops are soil,
trees and roof the model already ignores, so almost none of them carry signal.

This takes the opposite approach. It finds the detections that landed on no
labelled array -- the exact patches that fool the model -- and cuts crops
centred on each, several per patch with jitter so the fooling texture appears
at different offsets and scales. A handful of examples of a specific mistake is
worth far more than hundreds of random background.

The mask is still built from the verified labels, never forced to zero: a false
positive often sits near a real array, and teaching that array as background
would trade one error for a worse one.

Detections and imagery may live in different captures -- detection usually runs
at the resolution the model prefers while training crops come from the sharpest
imagery available -- so boxes are carried across geographically.

    python scripts/mine_hard_negatives.py --detections jbeil-mb-104 \
        --labels labels_clean.json --imagery jbeil-mb --into jbeil06_hard
"""

import argparse
import json
import shutil

import _bootstrap  # noqa: F401

import numpy as np
from PIL import Image

from solarmap.config import Config

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True,
                    help="capture the detections were produced on")
    ap.add_argument("--det-file", default="detections.geojson")
    ap.add_argument("--labels", default="labels_clean.json",
                    help="verified labels on the DETECTION capture")
    ap.add_argument("--imagery", required=True,
                    help="capture to cut the crops from (usually the sharpest)")
    ap.add_argument("--imagery-labels", default="labels.json",
                    help="verified labels on the IMAGERY capture")
    ap.add_argument("--into", required=True, help="dataset under data/datasets")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=6,
                    help="jittered crops per false positive")
    ap.add_argument("--on-panel", type=float, default=0.5,
                    help="a detection counts as correct, and is skipped, when "
                         "this much of it lies on a labelled array")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    src = cfg.path("captures") / args.detections
    img_cap = cfg.path("captures") / args.imagery
    root = cfg.path("datasets") / args.into
    if not (root / "train" / "images").is_dir():
        raise SystemExit(f"{root} has no train/images -- build it with "
                         f"prepare_masks.py first, then add negatives here.")

    src_man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    src_lab = json.loads((src / args.labels).read_text(encoding="utf-8"))
    dets = json.loads((src / args.det_file).read_text(encoding="utf-8"))
    img_man = json.loads((img_cap / "manifest.json").read_text(encoding="utf-8"))
    img_lab = json.loads((img_cap / args.imagery_labels).read_text(encoding="utf-8"))
    img_tiles = {t["tile_id"]: t for t in img_man["tiles"]}
    rng = np.random.default_rng(args.seed)

    # --- which detections are wrong, in geographic terms ---
    wrong = []
    for t in src_man["tiles"]:
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]
        panels = [b for b in src_lab["tiles"].get(t["tile_id"], [])
                  if b.get("verified") is True]
        gt = np.zeros((H, W), bool)
        for b in panels:
            gt[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = True

        for f in dets["features"]:
            ring = f["geometry"]["coordinates"][0]
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            if not (w <= cx <= e and s <= cy <= n):
                continue
            import cv2
            poly = np.array([[(lon - w) / (e - w) * W, (n - lat) / (n - s) * H]
                             for lon, lat in ring], np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            mb = m.astype(bool)
            a = int(mb.sum())
            if a < 4:
                continue
            if (mb & gt).sum() / a >= args.on_panel:
                continue                      # the model got this one right
            wrong.append((cx, cy))

    print(f"{len(wrong)} false positives to mine "
          f"(of {len(dets['features'])} detections)")
    if not wrong:
        return

    # --- cut jittered crops around each, from the imagery capture ---
    S = args.crop
    added = 0
    cache: dict[str, tuple] = {}
    for cx, cy in wrong:
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

        for k in range(args.repeats):
            jx = int(rng.integers(-S // 3, S // 3 + 1))
            jy = int(rng.integers(-S // 3, S // 3 + 1))
            ox = int(np.clip(px - S // 2 + jx, 0, max(0, W - S)))
            oy = int(np.clip(py - S // 2 + jy, 0, max(0, H - S)))
            sub = mask[oy:oy + S, ox:ox + S]
            if sub.shape != (S, S):
                pad = np.zeros((S, S), np.uint8)
                pad[:sub.shape[0], :sub.shape[1]] = sub
                sub = pad
            name = f"hardneg_{tid}_{ox}_{oy}"
            im.crop((ox, oy, ox + S, oy + S)).save(
                root / "train" / "images" / f"{name}.png")
            Image.fromarray(sub).save(root / "train" / "masks" / f"{name}.png")
            added += 1

    total = len(list((root / "train" / "images").glob("*")))
    print(f"added {added} hard-negative crops -> {root}")
    print(f"train split is now {total} crops")


if __name__ == "__main__":
    main()
