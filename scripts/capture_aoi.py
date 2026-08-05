"""Capture an AOI from Google Earth Pro.

    python scripts/capture_aoi.py --bbox 37.770,-122.430,37.780,-122.415 --name mission
"""

import argparse

import _bootstrap  # noqa: F401

from solarmap.capture.capture import CaptureSession
from solarmap.capture.grid import BBox
from solarmap.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bbox", required=True, help="south,west,north,east in WGS84 degrees")
    ap.add_argument("--name", required=True, help="capture name (output subdirectory)")
    ap.add_argument("--alt", type=float, help="override eye altitude in metres")
    ap.add_argument("--config", help="path to config.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.alt:
        cfg.raw["capture"]["eye_altitude_m"] = args.alt

    aoi = BBox.parse(args.bbox)

    def progress(i: int, total: int, label: str) -> None:
        print(f"\r[{i}/{total}] {label}", end="", flush=True)

    print("Connecting to Google Earth Pro...")
    out = CaptureSession(cfg).run(aoi, args.name, progress=progress)
    print(f"\nCapture written to {out}")


if __name__ == "__main__":
    main()
