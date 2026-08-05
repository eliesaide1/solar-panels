"""Capture a georeferenced imagery mosaic of an AOI from Google Earth Pro."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from ..config import Config
from ..geo import crop_extents, meters_per_pixel, write_world_file
from .earth_com import GoogleEarth, StreamingTimeout
from .grid import BBox, CameraPoint, build_grid


@dataclass
class TileRecord:
    tile_id: str
    image: str
    lat: float
    lon: float
    north: float
    south: float
    east: float
    west: float
    width: int
    height: int
    gsd_m: float


ProgressFn = Callable[[int, int, str], None]


class CaptureSession:
    def __init__(self, cfg: Config, out_dir: Path | None = None):
        self.cfg = cfg
        self.cap = cfg["capture"]
        self.out_root = Path(out_dir) if out_dir else cfg.path("captures")
        self.earth = GoogleEarth(
            stream_timeout=float(self.cap["stream_timeout_seconds"]),
            settle=float(self.cap["settle_seconds"]),
        )

    # -- internals ---------------------------------------------------------

    def _grab(self, path: Path) -> tuple[Path, tuple[float, float, float, float], tuple[int, int]]:
        """Screenshot the current view, apply the crop, return path/extents/size."""
        self.earth.wait_for_imagery()
        ext = self.earth.view_extents()
        self.earth.screenshot(path, int(self.cap["jpeg_quality"]))

        crop = self.cap["crop"]
        north, south, east, west = crop_extents(
            ext.north, ext.south, ext.east, ext.west,
            crop["left"], crop["right"], crop["top"], crop["bottom"],
        )

        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            box = (
                round(w * crop["left"]),
                round(h * crop["top"]),
                round(w * (1.0 - crop["right"])),
                round(h * (1.0 - crop["bottom"])),
            )
            im = im.crop(box)
            size = im.size
            im.save(path, quality=int(self.cap["jpeg_quality"]))

        write_world_file(path, size[0], size[1], north, south, east, west)
        return path, (north, south, east, west), size

    def _probe_footprint(self, aoi: BBox, scratch: Path) -> tuple[float, float]:
        """Measure one view's ground footprint so the grid can be spaced."""
        lat, lon = aoi.center
        self.earth.goto(lat, lon, float(self.cap["eye_altitude_m"]))
        _, (north, south, east, west), _ = self._grab(scratch)
        return (north - south, east - west)

    # -- public ------------------------------------------------------------

    def run(
        self,
        aoi: BBox,
        name: str,
        progress: ProgressFn | None = None,
    ) -> Path:
        """Capture ``aoi`` into ``<captures>/<name>/`` and return that directory."""
        out_dir = self.out_root / name
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        self.earth.connect()
        hidden = self.earth.hide_layers(list(self.cap["hide_layers"]))

        # Probe first: the grid spacing depends on the measured footprint.
        probe_path = tiles_dir / "_probe.jpg"
        footprint = self._probe_footprint(aoi, probe_path)
        probe_path.unlink(missing_ok=True)
        probe_path.with_suffix(".jgw").unlink(missing_ok=True)
        probe_path.with_suffix(".prj").unlink(missing_ok=True)

        points: list[CameraPoint] = build_grid(
            aoi, footprint, float(self.cap["overlap"])
        )

        records: list[TileRecord] = []
        failures: list[dict] = []
        total = len(points)

        for i, pt in enumerate(points, start=1):
            if progress:
                progress(i, total, pt.tile_id)
            path = tiles_dir / f"{pt.tile_id}.jpg"
            try:
                self.earth.goto(pt.lat, pt.lon, float(self.cap["eye_altitude_m"]))
                _, (north, south, east, west), (w, h) = self._grab(path)
            except StreamingTimeout as exc:
                # One slow tile should not abandon a multi-hour capture; record
                # it so the run can be topped up later.
                failures.append({"tile_id": pt.tile_id, "error": str(exc)})
                path.unlink(missing_ok=True)
                continue

            records.append(
                TileRecord(
                    tile_id=pt.tile_id,
                    image=f"tiles/{path.name}",
                    lat=pt.lat,
                    lon=pt.lon,
                    north=north, south=south, east=east, west=west,
                    width=w, height=h,
                    gsd_m=meters_per_pixel(north, south, h),
                )
            )

        manifest = {
            "name": name,
            "aoi": asdict(aoi),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "Google Earth Pro",
            "attribution": "Imagery (c) Google, Maxar Technologies, Airbus",
            "eye_altitude_m": self.cap["eye_altitude_m"],
            "overlap": self.cap["overlap"],
            "view_footprint_deg": {"height": footprint[0], "width": footprint[1]},
            "hidden_layers": hidden,
            "tile_count": len(records),
            "failed": failures,
            "tiles": [asdict(r) for r in records],
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return out_dir
