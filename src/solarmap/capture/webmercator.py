"""Web Mercator (EPSG:3857) tile arithmetic.

Slippy-map tiles are square in Mercator metres, not in degrees -- a tile's
north-south span in latitude shrinks as you move toward the poles. Doing the
georeferencing in Mercator and converting to lat/lon only at the edges keeps
that exact, instead of introducing a latitude-dependent stretch.
"""

from __future__ import annotations

import math

R = 6378137.0                    # WGS84 semi-major axis, the Web Mercator sphere
ORIGIN = math.pi * R             # 20037508.342789244 -- half the world, in metres
TILE_PX = 256


def lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    return (
        R * math.radians(lon),
        R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)),
    )


def merc_to_lonlat(x: float, y: float) -> tuple[float, float]:
    return (
        math.degrees(x / R),
        math.degrees(2.0 * math.atan(math.exp(y / R)) - math.pi / 2.0),
    )


def tile_of(lon: float, lat: float, z: int) -> tuple[int, int]:
    """Tile column/row containing lon/lat at zoom ``z``."""
    n = 2 ** z
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return _clamp(x, n), _clamp(y, n)


def tile_bounds_merc(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of a tile, in Mercator metres."""
    size = 2.0 * ORIGIN / (2 ** z)
    xmin = -ORIGIN + x * size
    ymax = ORIGIN - y * size
    return xmin, ymax - size, xmin + size, ymax


def block_bounds_merc(x0: int, y0: int, nx: int, ny: int, z: int):
    """Mercator bounds of an ``nx`` x ``ny`` block of tiles starting at x0/y0."""
    left, _, _, top = tile_bounds_merc(x0, y0, z)
    _, bottom, right, _ = tile_bounds_merc(x0 + nx - 1, y0 + ny - 1, z)
    return left, bottom, right, top


def merc_bounds_to_wgs84(xmin, ymin, xmax, ymax):
    """Convert Mercator bounds to (north, south, east, west) degrees."""
    west, south = merc_to_lonlat(xmin, ymin)
    east, north = merc_to_lonlat(xmax, ymax)
    return north, south, east, west


def ground_resolution(lat: float, z: int) -> float:
    """True ground metres per pixel at ``lat``.

    Mercator inflates distances by 1/cos(lat); undoing that gives the real
    ground sample distance, which is what decides whether panels are
    resolvable.
    """
    return (2.0 * ORIGIN / (TILE_PX * 2 ** z)) * math.cos(math.radians(lat))


def zoom_for_resolution(lat: float, target_m: float, max_zoom: int = 23) -> int:
    """Smallest zoom whose ground resolution is at least as fine as target_m."""
    for z in range(max_zoom + 1):
        if ground_resolution(lat, z) <= target_m:
            return z
    return max_zoom


def _clamp(v: int, n: int) -> int:
    return max(0, min(v, n - 1))
