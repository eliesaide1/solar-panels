"""Grab one frame from Google Earth Pro and report what it is worth.

Run this before your first real capture. It tells you the actual ground sample
distance your window and altitude produce, so you can check panels will be
resolvable, and it writes a sample frame you can inspect for leftover UI
furniture that the crop settings need to remove.

    python scripts/calibrate.py --at 37.7749,-122.4194 --alt 150
"""

import argparse

import _bootstrap  # noqa: F401

from solarmap.capture.capture import CaptureSession
from solarmap.config import Config
from solarmap.geo import meters_per_pixel

# Common residential module: roughly 1.7 m x 1.0 m.
MODULE_LONG_SIDE_M = 1.7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", required=True, help="lat,lon to sample")
    ap.add_argument("--alt", type=float, help="eye altitude in metres")
    ap.add_argument("--config")
    args = ap.parse_args()

    lat, lon = (float(v) for v in args.at.split(","))
    cfg = Config.load(args.config)
    if args.alt:
        cfg.raw["capture"]["eye_altitude_m"] = args.alt
    alt = cfg["capture"]["eye_altitude_m"]

    session = CaptureSession(cfg)
    out = cfg.path("outputs") / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"calib_{int(alt)}m.jpg"

    print("Connecting to Google Earth Pro...")
    session.earth.connect()
    session.earth.hide_layers(list(cfg["capture"]["hide_layers"]))
    session.earth.goto(lat, lon, alt)
    _, (north, south, east, west), (w, h) = session._grab(path)

    gsd = meters_per_pixel(north, south, h)
    footprint_m = (north - south) * 111_320.0
    module_px = MODULE_LONG_SIDE_M / gsd

    print(f"\nimage        {w} x {h} px  ->  {path}")
    print(f"eye altitude {alt:.0f} m")
    print(f"footprint    {footprint_m:.0f} m tall")
    print(f"GSD          {gsd*100:.1f} cm/px")
    print(f"one module   ~{module_px:.1f} px on its long side")

    if module_px < 6:
        print("\nTOO COARSE. A single module spans under ~6 px; segmentation will "
              "only find large arrays. Lower --alt or enlarge the Earth window.")
    elif module_px > 40:
        print("\nVery fine detail, but each tile covers little ground -- a wide AOI "
              "will take a long time. Consider raising --alt.")
    else:
        print("\nGood working range for panel segmentation.")

    print("\nNow open the image and check for Google Earth UI furniture (compass, "
          "status bar, attribution). Adjust capture.crop in config.yaml until it "
          "is gone -- the extents are shrunk to match automatically.")


if __name__ == "__main__":
    main()
