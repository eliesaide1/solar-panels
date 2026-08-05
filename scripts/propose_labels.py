"""Propose candidate solar-panel boxes for a capture, for human verification.

These are PROPOSALS, not labels. PV modules in overhead imagery share a
distinctive signature -- dark, weakly saturated, and covered in a regular
high-frequency cell grid -- which classical image processing can find without
any trained model. That is enough to seed a labelling pass, but it will also
fire on dark textured roofs, shade cloth and car parks.

Nothing here should be trained on until a human has reviewed it in the
labelling UI. Fine-tuning on unverified boxes teaches the model whatever
mistakes this script makes.

    python scripts/propose_labels.py --capture jbeil-nds
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import cv2
import numpy as np

from solarmap.config import Config


def propose(bgr: np.ndarray, gsd_m: float, min_area_m2: float, max_area_m2: float):
    """Return [(x1, y1, x2, y2, score), ...] candidate boxes in pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    # Thresholds measured from confirmed arrays at Notre Dame des Secours,
    # Jbeil (Esri z19, 24.7 cm/px):
    #
    #   region          brightness  saturation  texture
    #   panel rows         112          28        21.2
    #   panel rooftop      171          25        24.1
    #   panel carport      206          23        17.1
    #   trees               73          78        11.5
    #   bare ground        153          81        10.9
    #   tarmac             180          44        15.0
    #   building roof      129          33        18.1
    #
    # Brightness spans 112-206 across panels, so it discriminates nothing and
    # is deliberately not used -- only shadow is excluded. Saturation and
    # texture are what actually separate panels from everything else.

    not_shadow = v > 60

    # Panels are grey-blue; vegetation, soil and tile roofs are not.
    drab = s < 32

    # The cell grid gives panels a high-frequency texture that smooth surfaces
    # such as tarmac and bare roofs lack.
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (7, 7))
    sq = cv2.blur(g * g, (7, 7))
    local_std = np.sqrt(np.maximum(sq - mean * mean, 0))
    textured = local_std > 16

    mask = (not_shadow & drab & textured).astype(np.uint8) * 255

    # Close the gaps between rows so an array becomes one blob, then drop
    # speckle from isolated dark pixels.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    px_area = gsd_m * gsd_m
    out = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area_px = stats[i]
        area_m2 = area_px * px_area
        if not (min_area_m2 <= area_m2 <= max_area_m2):
            continue
        # Panels form a mostly-filled rectangle, but rooftop arrays are
        # sparser than ground-mounted ones -- gaps between rows measured
        # 0.32-0.34 fill on confirmed Jbeil roof arrays, so a 0.35 cut
        # discarded real panels. 0.25 keeps them without admitting shadow,
        # which is far straggilier.
        if area_px / float(w * h) < 0.25:
            continue
        # Reject extreme slivers.
        if max(w, h) / max(min(w, h), 1) > 12:
            continue
        score = float(local_std[labels == i].mean())
        out.append((int(x), int(y), int(x + w), int(y + h), score))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--min-area", type=float, default=8.0, help="m2")
    ap.add_argument("--max-area", type=float, default=6000.0, help="m2")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap_dir = cfg.path("captures") / args.capture
    manifest = json.loads((cap_dir / "manifest.json").read_text(encoding="utf-8"))
    gsd = float(manifest["gsd_m"])

    proposals: dict[str, list] = {}
    total = 0
    for t in manifest["tiles"]:
        img = cv2.imread(str(cap_dir / t["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        boxes = propose(img, gsd, args.min_area, args.max_area)
        proposals[t["tile_id"]] = [
            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
             "score": round(b[4], 2), "verified": None}
            for b in boxes
        ]
        total += len(boxes)

    out = cap_dir / "labels.json"
    out.write_text(
        json.dumps(
            {"capture": args.capture, "gsd_m": gsd, "source": "proposals",
             "tiles": proposals},
            indent=1,
        ),
        encoding="utf-8",
    )
    tiles_with = sum(1 for v in proposals.values() if v)
    print(f"{total} proposals across {tiles_with}/{len(proposals)} tiles -> {out}")
    print("\nThese are UNVERIFIED. Review them before training:")
    print(f"  python scripts/label_ui.py --capture {args.capture}")


if __name__ == "__main__":
    main()
