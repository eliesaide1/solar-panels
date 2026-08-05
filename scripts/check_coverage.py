"""Probe which zoom levels a tile source actually has imagery for at a location.

Coverage is wildly uneven by region. A source's advertised max zoom is the best
case, not a guarantee -- services return a "no data" placeholder with HTTP 200,
so the only reliable test is to fetch a tile and look at it.

    python scripts/check_coverage.py --at 33.5439,35.5844
    python scripts/check_coverage.py --at 33.5439,35.5844 --source esri,naip
"""

import argparse
from io import BytesIO

import _bootstrap  # noqa: F401

import requests
from PIL import Image

from solarmap.capture.tiles import looks_like_no_data, source_from_config
from solarmap.capture.webmercator import ground_resolution, tile_of
from solarmap.config import Config

MODULE_LONG_SIDE_M = 1.7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--at", required=True, help="lat,lon")
    ap.add_argument("--source", default="", help="comma-separated; default: all configured")
    ap.add_argument("--min-zoom", type=int, default=15)
    ap.add_argument("--config")
    args = ap.parse_args()

    lat, lon = (float(v) for v in args.at.split(","))
    cfg = Config.load(args.config)
    names = [s.strip() for s in args.source.split(",") if s.strip()] or list(
        cfg.get("tile_sources", {})
    )

    session = requests.Session()
    session.headers["User-Agent"] = cfg.get("tiles", {}).get(
        "user_agent", "SolarMap/0.1 (research)"
    )

    print(f"Probing {lat:.5f}, {lon:.5f}\n")
    best: dict[str, int] = {}

    for name in names:
        try:
            source = source_from_config(cfg, name)
        except SystemExit as exc:
            print(f"{name}: skipped -- {exc}\n")
            continue

        print(f"{name} (advertised max zoom {source.max_zoom})")
        for z in range(args.min_zoom, source.max_zoom + 1):
            x, y = tile_of(lon, lat, z)
            try:
                r = session.get(source.format(x, y, z), timeout=20)
                if r.status_code != 200:
                    verdict, ok = f"HTTP {r.status_code}", False
                else:
                    img = Image.open(BytesIO(r.content))
                    ok = not looks_like_no_data(img)
                    verdict = "imagery" if ok else "NO DATA (placeholder)"
            except Exception as exc:
                verdict, ok = f"error: {type(exc).__name__}", False

            gsd = ground_resolution(lat, z)
            px = MODULE_LONG_SIDE_M / gsd
            print(f"   z{z:<3} {gsd*100:6.1f} cm/px  module ~{px:5.1f} px   {verdict}")
            if ok:
                best[name] = z
        print()

    print("=" * 58)
    if not best:
        print("No source has imagery here. This location may be unmapped at these")
        print("zooms; try --min-zoom 12 to find what does exist.")
        return

    for name, z in sorted(best.items(), key=lambda kv: -kv[1]):
        gsd = ground_resolution(lat, z)
        px = MODULE_LONG_SIDE_M / gsd
        note = ("good for panel segmentation" if px >= 10 else
                "workable, arrays only" if px >= 6 else
                "TOO COARSE for individual modules")
        print(f"{name:8} best z{z}  {gsd*100:.1f} cm/px  module ~{px:.1f} px  -- {note}")

    top = max(best.items(), key=lambda kv: kv[1])
    print(f"\nUse: --source {top[0]} --zoom {top[1]}")


if __name__ == "__main__":
    main()
