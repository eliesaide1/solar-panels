"""YOLO detection backend.

The U-Net path produces per-pixel masks; YOLO produces axis-aligned boxes. Both
end up as georeferenced GeoJSON polygons, but the area they imply is not
equally trustworthy:

A box circumscribes the array. For a roof-parallel rectangular array the box
area is close to the true footprint. For an array set at an angle to north, the
box can overstate area by up to ~40%, and for an L-shaped or scattered layout it
overstates badly. Areas from this backend are therefore upper bounds, and every
feature is tagged ``geometry_source: "bbox"`` so downstream consumers can tell
the difference rather than silently treating them as measured footprints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import box as shapely_box, mapping
from shapely.ops import transform as shapely_transform

from ..geo import WGS84, utm_crs_for


class YoloDetector:
    """Wraps an Ultralytics YOLO detection model."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1024,
        device: str | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The YOLO backend needs Ultralytics. Run: pip install ultralytics"
            ) from exc

        self.model = YOLO(str(weights))
        if self.model.task != "detect":
            raise ValueError(
                f"Expected a detection model, got task={self.model.task!r}. "
                "Segmentation checkpoints go through the U-Net path instead."
            )
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.names = self.model.names

    def predict_boxes(
        self,
        image: np.ndarray,
        slice_size: int = 0,
        overlap: float = 0.3,
    ) -> list[tuple[float, float, float, float, float]]:
        """Return [(x1, y1, x2, y2, confidence), ...] in pixel coordinates.

        With ``slice_size`` set, the image is scanned in overlapping windows and
        the results merged. Rooftop arrays occupy a tiny fraction of a full
        tile; running the detector on smaller windows makes each array much
        larger relative to the frame, which is what small-object detection
        needs. It must match how the model was trained -- slicing at inference
        while training on whole tiles (or vice versa) hurts rather than helps.
        """
        if not slice_size or slice_size >= max(image.shape[:2]):
            return self._raw(image)

        h, w = image.shape[:2]
        step = max(1, int(slice_size * (1 - overlap)))
        out: list[tuple[float, float, float, float, float]] = []
        for oy in range(0, max(1, h - slice_size + 1), step):
            for ox in range(0, max(1, w - slice_size + 1), step):
                win = image[oy:oy + slice_size, ox:ox + slice_size]
                for x1, y1, x2, y2, c in self._raw(win):
                    out.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, c))
        return _nms(out, self.iou)

    def _raw(self, image: np.ndarray):
        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        out = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        return out


def _nms(boxes, iou_threshold: float):
    """Greedy non-maximum suppression over boxes merged from overlapping slices.

    Without this, an array straddling two windows is reported twice and its
    capacity double-counted.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for cand in boxes:
        cx1, cy1, cx2, cy2, _ = cand
        drop = False
        for kx1, ky1, kx2, ky2, _ in keep:
            ix = max(0.0, min(cx2, kx2) - max(cx1, kx1))
            iy = max(0.0, min(cy2, ky2) - max(cy1, ky1))
            inter = ix * iy
            if inter <= 0:
                continue
            union = (cx2 - cx1) * (cy2 - cy1) + (kx2 - kx1) * (ky2 - ky1) - inter
            if union > 0 and inter / union >= iou_threshold:
                drop = True
                break
        if not drop:
            keep.append(cand)
    return keep


def boxes_to_features(
    boxes: list[tuple[float, float, float, float, float]],
    width: int,
    height: int,
    north: float,
    south: float,
    east: float,
    west: float,
    kw_per_m2: float = 0.19,
    min_area_m2: float = 2.0,
    max_frame_fraction: float = 0.6,
    crs: str = WGS84,
    bounds_proj: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Georeference pixel boxes into WGS84 GeoJSON features with area and kW.

    ``max_frame_fraction`` rejects boxes covering more than that share of the
    tile. This model's characteristic false positive is a near-whole-frame box:
    a scale sweep over imagery with no panels produced boxes spanning ~92% of
    the input at every window size tested, from 32 m to 255 m of ground. A
    genuine array -- even a utility-scale one -- occupies a modest fraction of
    a tile, so anything filling the frame is the detector shrugging, not a
    find. Set to 1.0 to disable.
    """
    if not boxes:
        return []

    if bounds_proj is not None:
        xmin, ymin, xmax, ymax = bounds_proj
        source_crs = crs
    else:
        xmin, ymin, xmax, ymax = west, south, east, north
        source_crs = WGS84

    # Pixel -> source-CRS scale. Rows run north to south, hence the flip on y.
    sx = (xmax - xmin) / width
    sy = (ymax - ymin) / height

    lat_c, lon_c = (north + south) / 2.0, (east + west) / 2.0
    utm = utm_crs_for(lat_c, lon_c)
    to_utm = Transformer.from_crs(source_crs, utm, always_xy=True).transform
    to_wgs = Transformer.from_crs(utm, WGS84, always_xy=True).transform

    frame_px = float(width) * float(height)

    features: list[dict] = []
    for x1, y1, x2, y2, conf in boxes:
        if frame_px > 0 and ((x2 - x1) * (y2 - y1)) / frame_px > max_frame_fraction:
            continue
        gx1 = xmin + x1 * sx
        gx2 = xmin + x2 * sx
        gy1 = ymax - y2 * sy      # bottom edge
        gy2 = ymax - y1 * sy      # top edge

        geom_utm = shapely_transform(to_utm, shapely_box(gx1, gy1, gx2, gy2))
        area_m2 = float(geom_utm.area)
        if area_m2 < min_area_m2:
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(shapely_transform(to_wgs, geom_utm)),
                "properties": {
                    "area_m2": round(area_m2, 2),
                    "capacity_kw": round(area_m2 * kw_per_m2, 3),
                    "confidence": round(conf, 4),
                    # Flags that this area is a bounding-box upper bound, not a
                    # segmented footprint.
                    "geometry_source": "bbox",
                },
            }
        )
    return features
