"""Capture imagery from an XYZ tile service.

The drop-in alternative to the Google Earth Pro backend. It writes the same
``manifest.json``, so everything downstream is unchanged -- but it is exact
rather than approximate (tile bounds are computed, not measured from a GUI),
parallel, resumable, and does not require any desktop software.

Blocks of tiles are stitched into larger images so the segmentation model sees
whole rooftops instead of arrays chopped at 256 px boundaries.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from PIL import Image

from ..config import Config
from ..geo import write_world_file_proj
from .grid import BBox
from .webmercator import (
    TILE_PX,
    block_bounds_merc,
    ground_resolution,
    merc_bounds_to_wgs84,
    tile_of,
    zoom_for_resolution,
)

ProgressFn = Callable[[int, int, str], None]


class TileFetchError(RuntimeError):
    pass


def looks_like_no_data(img: Image.Image) -> bool:
    """True if a tile is a service's "no imagery here" placeholder.

    Tile services return these with HTTP 200, not 404, so they cannot be
    detected from the status code. Esri's is a flat light-grey field reading
    "Map data not yet available". Left undetected they enter the pipeline as
    valid imagery, and a survey silently reports zero panels over an area it
    never actually saw.
    """
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if g.size == 0:
        return True
    # Real aerial imagery is textured; placeholders are near-uniform.
    return bool(g.std() < 12.0 and 150.0 < g.mean() < 240.0)


@dataclass
class Source:
    name: str
    url: str
    max_zoom: int = 19
    attribution: str = ""
    token: str = ""

    def format(self, x: int, y: int, z: int) -> str:
        return (
            self.url.replace("{x}", str(x))
            .replace("{y}", str(y))
            .replace("{z}", str(z))
            .replace("{token}", self.token)
        )


class TileCapture:
    def __init__(self, cfg: Config, source: Source, out_dir: Path | None = None):
        self.cfg = cfg
        self.tc = cfg.get("tiles", {})
        self.source = source
        self.out_root = Path(out_dir) if out_dir else cfg.path("captures")

        self.session = requests.Session()
        # Tile services reject or throttle unidentified clients.
        self.session.headers["User-Agent"] = self.tc.get(
            "user_agent", "SolarMap/0.1 (solar panel mapping research)"
        )

    # -- fetching ----------------------------------------------------------

    def _fetch_tile(self, x: int, y: int, z: int) -> Image.Image:
        url = self.source.format(x, y, z)
        last: Exception | None = None
        for attempt in range(int(self.tc.get("retries", 3))):
            try:
                r = self.session.get(url, timeout=float(self.tc.get("timeout", 20)))
                if r.status_code == 200:
                    return Image.open(BytesIO(r.content)).convert("RGB")
                if r.status_code in (404, 204):
                    # Genuinely no imagery here; a blank tile is the honest
                    # answer and lets the rest of the block proceed.
                    return Image.new("RGB", (TILE_PX, TILE_PX), (0, 0, 0))
                last = TileFetchError(f"HTTP {r.status_code} for {url}")
            except Exception as exc:
                last = exc
            time.sleep(0.4 * (2 ** attempt))  # back off before retrying
        raise TileFetchError(f"Failed after retries: {url}") from last

    def _fetch_block(self, x0: int, y0: int, nx: int, ny: int, z: int) -> Image.Image:
        coords = [(x0 + dx, y0 + dy) for dy in range(ny) for dx in range(nx)]
        workers = int(self.tc.get("workers", 8))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            images = list(pool.map(lambda c: self._fetch_tile(c[0], c[1], z), coords))

        # Tile pixel size is whatever the service actually returns, not an
        # assumed 256. Mapbox's "@2x" URLs serve 512 px tiles covering the same
        # ground, so hard-coding 256 would paste them at half spacing and
        # mangle both the mosaic and its georeferencing.
        px = max((im.width for im in images), default=TILE_PX)

        canvas = Image.new("RGB", (nx * px, ny * px))
        for (x, y), im in zip(coords, images):
            if im.width != px or im.height != px:
                im = im.resize((px, px), Image.LANCZOS)
            canvas.paste(im, ((x - x0) * px, (y - y0) * px))
        return canvas

    # -- public ------------------------------------------------------------

    def run(
        self,
        aoi: BBox,
        name: str,
        zoom: int | None = None,
        target_gsd_m: float | None = None,
        block: int = 4,
        progress: ProgressFn | None = None,
    ) -> Path:
        lat_c, _ = aoi.center

        if zoom is None:
            zoom = zoom_for_resolution(lat_c, target_gsd_m or 0.15, self.source.max_zoom)
        if zoom > self.source.max_zoom:
            print(
                f"WARNING: requested zoom {zoom} exceeds {self.source.name}'s "
                f"max of {self.source.max_zoom}; clamping. Ground resolution "
                "will be coarser than requested."
            )
            zoom = self.source.max_zoom

        gsd = ground_resolution(lat_c, zoom)

        x_min, y_min = tile_of(aoi.west, aoi.north, zoom)   # NW corner
        x_max, y_max = tile_of(aoi.east, aoi.south, zoom)   # SE corner

        blocks = [
            (bx, by)
            for by in range(y_min, y_max + 1, block)
            for bx in range(x_min, x_max + 1, block)
        ]

        out_dir = self.out_root / name
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        failures: list[dict] = []
        no_data: list[str] = []

        for i, (bx, by) in enumerate(blocks, start=1):
            nx = min(block, x_max - bx + 1)
            ny = min(block, y_max - by + 1)
            tile_id = f"z{zoom}x{bx}y{by}"
            if progress:
                progress(i, len(blocks), tile_id)

            path = tiles_dir / f"{tile_id}.jpg"
            if path.exists():
                # Resume: an interrupted capture should not re-download.
                with Image.open(path) as existing:
                    if looks_like_no_data(existing):
                        no_data.append(tile_id)
                        continue
            else:
                try:
                    img = self._fetch_block(bx, by, nx, ny, zoom)
                except TileFetchError as exc:
                    failures.append({"tile_id": tile_id, "error": str(exc)})
                    continue

                if looks_like_no_data(img):
                    # Keep it out of the manifest so nothing downstream mistakes
                    # a grey placeholder for a rooftop with no panels on it.
                    no_data.append(tile_id)
                    continue
                img.save(path, quality=int(self.tc.get("jpeg_quality", 92)))

            xmin, ymin, xmax, ymax = block_bounds_merc(bx, by, nx, ny, zoom)
            north, south, east, west = merc_bounds_to_wgs84(xmin, ymin, xmax, ymax)
            # Read the real dimensions off disk: a service returning larger
            # tiles gives finer ground resolution over identical bounds.
            with Image.open(path) as _im:
                w, h = _im.size

            write_world_file_proj(path, w, h, xmin, ymin, xmax, ymax, "EPSG:3857")

            records.append(
                {
                    "tile_id": tile_id,
                    "image": f"tiles/{path.name}",
                    "lat": (north + south) / 2.0,
                    "lon": (east + west) / 2.0,
                    # WGS84 bounds drive the map display...
                    "north": north, "south": south, "east": east, "west": west,
                    # ...while the projected bounds drive georeferencing, where
                    # the tile grid is exactly square and nothing is stretched.
                    "crs": "EPSG:3857",
                    "bounds_proj": [xmin, ymin, xmax, ymax],
                    "width": w, "height": h,
                    # Ground resolution follows the actual pixel count, not the
                    # zoom level alone.
                    "gsd_m": gsd * (TILE_PX * nx) / max(w, 1),
                }
            )

        manifest = {
            "name": name,
            "aoi": asdict(aoi),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": f"XYZ tiles: {self.source.name}",
            "attribution": self.source.attribution,
            "zoom": zoom,
            # The tiles know the true resolution: a source serving "@2x" tiles
            # returns twice the pixels over the same ground, so the zoom-only
            # figure would understate it by 2x. Report what is on disk.
            "gsd_m": records[0]["gsd_m"] if records else gsd,
            "block_tiles": block,
            "tile_count": len(records),
            "failed": failures,
            "no_data": no_data,
            "blocks_requested": len(blocks),
            "coverage": round(len(records) / len(blocks), 4) if blocks else 0.0,
            "tiles": records,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if no_data:
            pct = 100.0 * len(no_data) / len(blocks)
            print(
                f"\nWARNING: {len(no_data)}/{len(blocks)} blocks ({pct:.0f}%) have no "
                f"imagery at zoom {zoom} from {self.source.name}. They were dropped, "
                "not saved as blank tiles."
            )
            if len(records) == 0:
                print(
                    "  This area has NO coverage at this zoom. Try a lower --zoom, "
                    "or a source with better coverage here (mapbox reaches z22)."
                )
        return out_dir


def source_from_config(cfg: Config, name: str) -> Source:
    sources = cfg.get("tile_sources", {})
    if name not in sources:
        raise SystemExit(
            f"Unknown tile source {name!r}. Configured: {', '.join(sources) or '(none)'}"
        )
    spec = dict(sources[name])
    token = spec.pop("token", "") or ""
    if "{token}" in spec.get("url", "") and not token:
        raise SystemExit(
            f"Tile source {name!r} needs an access token. Add `token:` under "
            f"tile_sources.{name} in config.yaml."
        )
    return Source(name=name, token=token, **spec)
