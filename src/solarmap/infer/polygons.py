"""Georeference traced panel outlines into GeoJSON features.

The box path reports a bounding rectangle, whose area overstates any array that
is angled, L-shaped or ragged. A traced outline follows the panel edge, so the
area it reports is a measurement rather than an upper bound -- which is what
makes a capacity figure quotable.
"""

from __future__ import annotations

from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform

from ..geo import WGS84, utm_crs_for


def polygons_to_features(
    polys,
    width: int,
    height: int,
    north: float,
    south: float,
    east: float,
    west: float,
    kw_per_m2: float = 0.19,
    min_area_m2: float = 2.0,
    max_frame_fraction: float = 0.6,
    simplify_m: float = 0.3,
    crs: str = WGS84,
    bounds_proj: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Convert [(pixel_points, score), ...] into WGS84 GeoJSON features."""
    if not polys:
        return []

    if bounds_proj is not None:
        xmin, ymin, xmax, ymax = bounds_proj
        source_crs = crs
    else:
        xmin, ymin, xmax, ymax = west, south, east, north
        source_crs = WGS84

    sx = (xmax - xmin) / width
    sy = (ymax - ymin) / height

    lat_c, lon_c = (north + south) / 2.0, (east + west) / 2.0
    utm = utm_crs_for(lat_c, lon_c)
    to_utm = Transformer.from_crs(source_crs, utm, always_xy=True).transform
    to_wgs = Transformer.from_crs(utm, WGS84, always_xy=True).transform

    frame_px = float(width) * float(height)
    out: list[dict] = []

    for pts, score in polys:
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Same whole-frame guard as the box path: a detection spanning the tile
        # is the detector shrugging, not a find.
        if frame_px > 0 and ((max(xs) - min(xs)) * (max(ys) - min(ys))) / frame_px > max_frame_fraction:
            continue

        # Pixel -> source CRS. Rows run north to south, hence the y flip.
        ring = [(xmin + px * sx, ymax - py * sy) for px, py in pts]
        geom = Polygon(ring)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue

        geom_utm = shapely_transform(to_utm, geom)
        if simplify_m > 0:
            geom_utm = geom_utm.simplify(simplify_m, preserve_topology=True)
        area_m2 = float(geom_utm.area)
        if area_m2 < min_area_m2 or geom_utm.is_empty:
            continue

        out.append({
            "type": "Feature",
            "geometry": mapping(shapely_transform(to_wgs, geom_utm)),
            "properties": {
                "area_m2": round(area_m2, 2),
                "capacity_kw": round(area_m2 * kw_per_m2, 3),
                "confidence": round(score, 4),
                # Traced outline: area is measured, not a bounding-box bound.
                "geometry_source": "outline",
            },
        })
    return out
