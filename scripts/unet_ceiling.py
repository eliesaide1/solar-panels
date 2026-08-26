"""What is the U-Net's array-recall ceiling, and can it seed a labelling round?

The classical proposal stage cannot work at 6-10 cm (see tune_proposals.py: no
threshold setting gives both coverage and shape). So the question becomes
whether a segmentation model can, and at what operating point.

Probability maps are expensive on CPU and independent of the threshold, so
they are computed once, cached to data/outputs/probcache/<capture>/, and then
swept offline. Re-running with a different sweep costs nothing.

Reports, for each threshold and closing radius:

  cover   fraction of labelled arrays with >= --min-cover of their area above
          threshold. This is the ceiling on array recall at that operating
          point -- and, if it is high enough, the recall of a labelling round
          seeded from these masks.
  onpanel fraction of the flagged area that is really panel. Tells you how
          much a human would have to reject.

    python scripts/unet_ceiling.py --capture jbeil-mb-104
    python scripts/unet_ceiling.py --capture jbeil-mb-104 --reuse-cache
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np
from PIL import Image

from solarmap.config import Config

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--checkpoint", default="solar_unet.pt")
    ap.add_argument("--min-cover", type=float, default=0.5)
    ap.add_argument("--reuse-cache", action="store_true",
                    help="skip inference where a cached probability map exists")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    cache = cfg.path("outputs") / "probcache" / args.capture
    cache.mkdir(parents=True, exist_ok=True)

    todo = [t for t in man["tiles"]
            if any(b.get("verified") is True for b in lab["tiles"].get(t["tile_id"], []))]
    print(f"{len(todo)} labelled tiles in {args.capture}")

    detector = None
    ic = cfg["inference"]
    for i, t in enumerate(todo, 1):
        tid = t["tile_id"]
        out = cache / f"{tid}.npy"
        if out.exists() and args.reuse_cache:
            continue
        if detector is None:
            from solarmap.infer.predict import SolarDetector
            detector = SolarDetector(cfg.path("checkpoints") / args.checkpoint)
        with Image.open(cap / t["image"]) as im:
            image = np.array(im.convert("RGB"))
        prob = detector.predict(image, tile_size=int(ic["tile_size"]),
                                stride=int(ic["stride"]))
        np.save(out, prob.astype(np.float16))
        print(f"\r  inferred {i}/{len(todo)} {tid}", end="", flush=True)
    print()

    THRESHOLDS = (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01)
    CLOSE_M = (0.0, 1.0, 2.5)

    n_gt = 0
    acc = {(th, cm): {"hit": 0, "flag": 0, "on": 0} for th in THRESHOLDS for cm in CLOSE_M}

    for t in todo:
        tid = t["tile_id"]
        gsd = float(t["gsd_m"])
        f = cache / f"{tid}.npy"
        if not f.exists():
            continue
        prob = np.load(f).astype(np.float32)
        H, W = prob.shape
        panels = [b for b in lab["tiles"].get(tid, []) if b.get("verified") is True]
        n_gt += len(panels)

        gt_union = np.zeros((H, W), bool)
        for b in panels:
            gt_union[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = True

        for th in THRESHOLDS:
            base = (prob >= th).astype(np.uint8)
            for cm in CLOSE_M:
                if cm > 0:
                    k = max(3, int(round(cm / gsd)) | 1)
                    m = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
                else:
                    m = base
                mb = m.astype(bool)
                a = acc[(th, cm)]
                a["flag"] += int(mb.sum())
                a["on"] += int((mb & gt_union).sum())
                for b in panels:
                    y1, y2 = max(0, b["y1"]), b["y2"]
                    x1, x2 = max(0, b["x1"]), b["x2"]
                    ar = max(1, (y2 - y1) * (x2 - x1))
                    if mb[y1:y2, x1:x2].sum() / ar >= args.min_cover:
                        a["hit"] += 1

    print(f"\n{n_gt} labelled arrays, array counted as covered at "
          f">= {args.min_cover:.0%} of its area above threshold\n")
    print(f"{'thresh':>7} {'close_m':>8} {'cover':>8} {'onpanel':>8}  {'flagged m2':>12}")
    gsd0 = float(todo[0]["gsd_m"]) if todo else 0.1
    px = gsd0 * gsd0
    for th in THRESHOLDS:
        for cm in CLOSE_M:
            a = acc[(th, cm)]
            cover = a["hit"] / max(n_gt, 1)
            onp = a["on"] / max(a["flag"], 1)
            print(f"{th:7.2f} {cm:8.1f} {cover:8.1%} {onp:8.1%}  {a['flag']*px:12,.0f}")


if __name__ == "__main__":
    main()
