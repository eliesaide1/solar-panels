"""Georeferencing helpers shared by capture and inference."""

from __future__ import annotations

import math
from pathlib import Path

from pyproj import CRS, Transformer

WGS84 = "EPSG:4326"


def write_world_file(
    image_path: str | Path,
    width: int,
    height: int,
    north: float,
    south: float,
    east: float,
    west: float,
) -> Path:
    """Write an ESRI world file (.jgw) for a north-up image.

    World file lines, in order: x pixel size, y rotation, x rotation,
    y pixel size, x of the *centre* of the top-left pixel, y of same.
    """
    image_path = Path(image_path)
    px_w = (east - west) / width
    px_h = -(north - south) / height  # negative: rows run north -> south

    lines = [
        px_w,
        0.0,
        0.0,
        px_h,
        west + px_w / 2.0,
        north + px_h / 2.0,
    ]
    wld = image_path.with_suffix(".jgw")
    wld.write_text("\n".join(f"{v:.12f}" for v in lines) + "\n", encoding="utf-8")

    # rasterio reads the CRS from a sidecar .prj, not the world file.
    image_path.with_suffix(".prj").write_text(CRS.from_user_input(WGS84).to_wkt())
    return wld


def write_world_file_proj(
    image_path: str | Path,
    width: int,
    height: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crs: str,
) -> Path:
    """World file for an image georeferenced in a projected CRS.

    Same layout as :func:`write_world_file`, but the units are the CRS's --
    metres for Web Mercator -- rather than degrees.
    """
    image_path = Path(image_path)
    px_w = (xmax - xmin) / width
    px_h = -(ymax - ymin) / height

    lines = [px_w, 0.0, 0.0, px_h, xmin + px_w / 2.0, ymax + px_h / 2.0]
    wld = image_path.with_suffix(".jgw")
    wld.write_text("\n".join(f"{v:.12f}" for v in lines) + "\n", encoding="utf-8")
    image_path.with_suffix(".prj").write_text(CRS.from_user_input(crs).to_wkt())
    return wld


def crop_extents(
    north: float,
    south: float,
    east: float,
    west: float,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> tuple[float, float, float, float]:
    """Shrink geographic extents to match a fractionally cropped image.

    Cropping pixels off an image without shrinking its extents by the same
    fraction silently shifts every downstream coordinate, so these two
    operations must always move together.
    """
    for name, v in (("left", left), ("right", right), ("top", top), ("bottom", bottom)):
        if not 0.0 <= v < 1.0:
            raise ValueError(f"crop.{name} must be in [0, 1), got {v}")
    if left + right >= 1.0 or top + bottom >= 1.0:
        raise ValueError("Crop fractions consume the entire image.")

    dlat = north - south
    dlon = east - west
    return (
        north - dlat * top,
        south + dlat * bottom,
        east - dlon * right,
        west + dlon * left,
    )


def utm_crs_for(lat: float, lon: float) -> CRS:
    """The UTM zone containing lat/lon.

    Areas must be computed in a projected CRS. Measuring polygon area in
    degrees produces a number in units of square-degrees, which is not a
    physical area and varies with latitude.
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def transformer_to_utm(lat: float, lon: float) -> Transformer:
    return Transformer.from_crs(WGS84, utm_crs_for(lat, lon), always_xy=True)


def meters_per_pixel(north: float, south: float, height_px: int) -> float:
    """Approximate ground sample distance, from the north-south extent."""
    return (north - south) * 111_320.0 / height_px
