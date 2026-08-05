"""Capture an AOI from an XYZ tile service.

    python scripts/capture_tiles.py --bbox 37.770,-122.430,37.780,-122.415 \
        --name mission --source esri

Zoom is chosen from --gsd (target ground resolution) unless you pass --zoom.
"""

import argparse

import _bootstrap  # noqa: F401

from solarmap.capture.grid import BBox
from solarmap.capture.tiles import TileCapture, source_from_config
from solarmap.capture.webmercator import ground_resolution, zoom_for_resolution
from solarmap.config import Config

MODULE_LONG_SIDE_M = 1.7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bbox", required=True, help="south,west,north,east in WGS84 degrees")
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", default="esri", help="key under tile_sources in config.yaml")
    ap.add_argument("--zoom", type=int, help="explicit zoom level")
    ap.add_argument("--gsd", type=float, default=0.15, help="target ground resolution, m/px")
    ap.add_argument("--block", type=int, default=4, help="tiles per side per stitched image")
    ap.add_argument("--dry-run", action="store_true", help="report the plan and stop")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    aoi = BBox.parse(args.bbox)
    source = source_from_config(cfg, args.source)

    lat_c, _ = aoi.center
    zoom = args.zoom or zoom_for_resolution(lat_c, args.gsd, source.max_zoom)
    zoom = min(zoom, source.max_zoom)
    gsd = ground_resolution(lat_c, zoom)

    print(f"source     {source.name}  (max zoom {source.max_zoom})")
    print(f"zoom       {zoom}")
    print(f"resolution {gsd*100:.1f} cm/px  ->  a module spans ~{MODULE_LONG_SIDE_M/gsd:.1f} px")
    if MODULE_LONG_SIDE_M / gsd < 6:
        print("  WARNING: too coarse for individual modules; only large arrays "
              "will be found. Try a source with a higher max zoom.")

    if args.dry_run:
        return

    def progress(i: int, total: int, label: str) -> None:
        print(f"\r[{i}/{total}] {label}", end="", flush=True)

    out = TileCapture(cfg, source).run(
        aoi, args.name, zoom=zoom, block=args.block, progress=progress
    )
    print(f"\nCapture written to {out}")


if __name__ == "__main__":
    main()
