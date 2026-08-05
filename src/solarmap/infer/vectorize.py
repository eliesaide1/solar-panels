"""Turn probability maps into GeoJSON polygons with area and capacity."""

from __future__ import annotations

import numpy as np
import rasterio.features
from pyproj import Transformer
from rasterio.transform import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform, unary_union

from ..geo import WGS84, utm_crs_for


def mask_to_polygons(
    prob: np.ndarray,
    north: float,
    south: float,
    east: float,
    west: float,
    threshold: float = 0.5,
    min_area_m2: float = 2.0,
    simplify_m: float = 0.25,
    kw_per_m2: float = 0.19,
    crs: str = WGS84,
    bounds_proj: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Polygonise ``prob`` and return GeoJSON features in WGS84.

    ``north``/``south``/``east``/``west`` are always WGS84 degrees and set the
    UTM zone. If the image is actually georeferenced in a projected CRS --
    Web Mercator tiles are -- pass ``crs`` and ``bounds_proj``
    (xmin, ymin, xmax, ymax) so the pixel grid maps exactly, with no
    latitude-dependent stretch.

    Each feature carries ``area_m2``, ``capacity_kw`` and the mean model
    confidence over its footprint.
    """
    h, w = prob.shape
    binary = (prob >= threshold).astype(np.uint8)
    if not binary.any():
        return []

    if bounds_proj is not None:
        transform = from_bounds(*bounds_proj, w, h)
        source_crs = crs
    else:
        transform = from_bounds(west, south, east, north, w, h)
        source_crs = WGS84

    lat_c, lon_c = (north + south) / 2.0, (east + west) / 2.0
    utm = utm_crs_for(lat_c, lon_c)
    to_utm = Transformer.from_crs(source_crs, utm, always_xy=True).transform
    to_wgs = Transformer.from_crs(utm, WGS84, always_xy=True).transform

    features: list[dict] = []
    for geom_dict, value in rasterio.features.shapes(
        binary, mask=binary.astype(bool), transform=transform
    ):
        if value != 1:
            continue
        geom_utm = shapely_transform(to_utm, shape(geom_dict))
        if geom_utm.area < min_area_m2:
            continue
        if simplify_m > 0:
            # Simplify in metres, where the tolerance is meaningful; the same
            # number in degrees would mean different distances by latitude.
            geom_utm = geom_utm.simplify(simplify_m, preserve_topology=True)
        if geom_utm.is_empty:
            continue

        area_m2 = float(geom_utm.area)
        geom_wgs = shapely_transform(to_wgs, geom_utm)

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom_wgs),
                "properties": {
                    "area_m2": round(area_m2, 2),
                    "capacity_kw": round(area_m2 * kw_per_m2, 3),
                    "confidence": round(float(_mean_conf(prob, geom_dict, transform)), 4),
                },
            }
        )
    return features


def _mean_conf(prob: np.ndarray, geom_dict: dict, transform) -> float:
    m = rasterio.features.geometry_mask(
        [geom_dict], out_shape=prob.shape, transform=transform, invert=True
    )
    return prob[m].mean() if m.any() else 0.0


def dedupe_across_tiles(features: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """Merge detections duplicated in the overlap between adjacent tiles.

    Tiles are captured with deliberate overlap, so an array near a boundary is
    detected twice. Left alone, that double-counts installed capacity.
    """
    if not features:
        return []

    lons = [c[0] for f in features for c in _exterior(f)]
    lats = [c[1] for f in features for c in _exterior(f)]
    utm = utm_crs_for(sum(lats) / len(lats), sum(lons) / len(lons))
    to_utm = Transformer.from_crs(WGS84, utm, always_xy=True).transform
    to_wgs = Transformer.from_crs(utm, WGS84, always_xy=True).transform

    geoms = [shapely_transform(to_utm, shape(f["geometry"])) for f in features]
    confs = [f["properties"].get("confidence", 0.0) for f in features]

    n = len(geoms)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Only genuine cross-tile duplicates may merge. Two detections inside the
    # same tile are distinct arrays -- merging them fuses neighbouring rooftop
    # arrays into one sprawling blob whose bounding polygon is mostly roof.
    # Measured cost of getting this wrong: area precision fell from 94% to 21%.
    tiles = [f["properties"].get("tile_id") for f in features]

    # O(n^2) is acceptable here: features are per-AOI, in the thousands at
    # most. Swap in an STRtree if an AOI ever gets large enough to matter.
    for i in range(n):
        for j in range(i + 1, n):
            if tiles[i] is not None and tiles[i] == tiles[j]:
                continue
            if not geoms[i].intersects(geoms[j]):
                continue
            inter = geoms[i].intersection(geoms[j]).area
            if inter <= 0:
                continue
            un = geoms[i].area + geoms[j].area - inter
            if un > 0 and inter / un >= iou_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[dict] = []
    for members in groups.values():
        geom = unary_union([geoms[i] for i in members])
        area_m2 = float(geom.area)
        merged.append(
            {
                "type": "Feature",
                "geometry": mapping(shapely_transform(to_wgs, geom)),
                "properties": {
                    "area_m2": round(area_m2, 2),
                    "capacity_kw": round(
                        area_m2 * _kw_per_m2_of(features, members), 3
                    ),
                    "confidence": round(
                        sum(confs[i] for i in members) / len(members), 4
                    ),
                    "merged_from": len(members),
                },
            }
        )
    return merged


def _kw_per_m2_of(features: list[dict], members: list[int]) -> float:
    """Recover the kW/m2 factor used upstream, so merging keeps it consistent."""
    for i in members:
        props = features[i]["properties"]
        if props.get("area_m2"):
            return props.get("capacity_kw", 0.0) / props["area_m2"]
    return 0.19


def _exterior(feature: dict) -> list:
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        return geom["coordinates"][0]
    return [c for poly in geom["coordinates"] for c in poly[0]]
