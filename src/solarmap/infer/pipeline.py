"""Run detection across every tile of a capture and emit one GeoJSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ..config import Config
from .predict import SolarDetector
from .vectorize import dedupe_across_tiles, mask_to_polygons

Image.MAX_IMAGE_PIXELS = None
ProgressFn = Callable[[int, int, str], None]


def is_yolo_checkpoint(path: str | Path) -> bool:
    """True if this is an Ultralytics checkpoint rather than a SolarMap U-Net.

    Ultralytics checkpoints pickle their own classes, so they carry a
    ``train_args`` block that a plain state_dict save never has.
    """
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        # Unpickling failed for want of `ultralytics` -- itself the giveaway.
        if "ultralytics" in str(exc):
            return True
        raise
    return isinstance(ckpt, dict) and "train_args" in ckpt


def detect_capture(
    cfg: Config,
    capture_dir: str | Path,
    checkpoint: str | Path,
    progress: ProgressFn | None = None,
) -> Path:
    """Detect panels across a capture directory; write ``detections.geojson``.

    Dispatches on the checkpoint type: Ultralytics YOLO weights go through the
    box detector, anything else through the U-Net segmenter.
    """
    capture_dir = Path(capture_dir)
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {capture_dir}. Run a capture first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ic = cfg["inference"]
    kw_per_m2 = float(cfg["capacity"]["kw_per_m2"])

    # A .pkl is the classical-proposal + box-classifier detector; .pt is a
    # neural checkpoint (YOLO or U-Net).
    use_cv = str(checkpoint).endswith(".pkl")
    use_cnn = Path(checkpoint).name.startswith("patchnet")
    use_yolo = (not use_cv) and (not use_cnn) and is_yolo_checkpoint(checkpoint)

    outline_mode = bool(cfg.get("yolo", {}).get("outline", True))

    if use_cnn:
        from .cnndetect import CnnDetector
        from .yolo import boxes_to_features

        yc = cfg.get("yolo", {})
        detector = CnnDetector(checkpoint, conf=float(yc.get("cnn_conf", 0.5)),
                               min_area_m2=float(ic["min_area_m2"]))
        model_kind = "cnn"
    elif use_cv:
        from .cvfilter import CvFilterDetector
        from .yolo import boxes_to_features

        yc = cfg.get("yolo", {})
        c = yc.get("cv_conf")
        detector = CvFilterDetector(checkpoint, conf=float(c) if c is not None else None)
        model_kind = "cv+filter"
    elif use_yolo:
        from .yolo import YoloDetector, boxes_to_features

        yc = cfg.get("yolo", {})
        detector = YoloDetector(
            checkpoint,
            conf=float(yc.get("conf", 0.25)),
            iou=float(yc.get("iou", 0.45)),
            imgsz=int(yc.get("imgsz", 1024)),
        )
        model_kind = "yolo"
    else:
        detector = SolarDetector(checkpoint)
        model_kind = "unet"

    raw: list[dict] = []
    tiles = manifest["tiles"]
    for i, tile in enumerate(tiles, start=1):
        if progress:
            progress(i, len(tiles), tile["tile_id"])

        img_path = capture_dir / tile["image"]
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            image = np.array(im.convert("RGB"))

        bounds_proj = tile.get("bounds_proj")
        # Tile-service captures are georeferenced in Web Mercator; Earth Pro
        # captures carry no bounds_proj and stay in lat/lon.
        geo = dict(
            north=tile["north"], south=tile["south"],
            east=tile["east"], west=tile["west"],
            crs=tile.get("crs", "EPSG:4326"),
            bounds_proj=tuple(bounds_proj) if bounds_proj else None,
        )

        if use_cv and outline_mode:
            from .polygons import polygons_to_features
            polys = detector.predict_polygons(image, gsd_m=tile.get("gsd_m"))
            feats = polygons_to_features(
                polys,
                width=image.shape[1], height=image.shape[0],
                kw_per_m2=kw_per_m2,
                min_area_m2=float(ic["min_area_m2"]),
                max_frame_fraction=float(cfg.get("yolo", {}).get("max_frame_fraction", 0.6)),
                simplify_m=float(ic.get("simplify_m", 0.3)),
                **geo,
            )
        elif use_cv or use_cnn:
            boxes = detector.predict_boxes(image, gsd_m=tile.get("gsd_m"))
            feats = boxes_to_features(
                boxes,
                width=image.shape[1], height=image.shape[0],
                kw_per_m2=kw_per_m2,
                min_area_m2=float(ic["min_area_m2"]),
                max_frame_fraction=float(cfg.get("yolo", {}).get("max_frame_fraction", 0.6)),
                **geo,
            )
        elif use_yolo:
            boxes = detector.predict_boxes(
                image,
                slice_size=int(cfg.get("yolo", {}).get("slice", 0)),
                overlap=float(cfg.get("yolo", {}).get("slice_overlap", 0.3)),
            )
            feats = boxes_to_features(
                boxes,
                width=image.shape[1], height=image.shape[0],
                kw_per_m2=kw_per_m2,
                min_area_m2=float(ic["min_area_m2"]),
                max_frame_fraction=float(cfg.get("yolo", {}).get("max_frame_fraction", 0.6)),
                **geo,
            )
        else:
            prob = detector.predict(
                image, tile_size=int(ic["tile_size"]), stride=int(ic["stride"])
            )
            feats = mask_to_polygons(
                prob,
                threshold=float(ic["threshold"]),
                min_area_m2=float(ic["min_area_m2"]),
                simplify_m=float(ic["simplify_m"]),
                kw_per_m2=kw_per_m2,
                **geo,
            )
        for f in feats:
            f["properties"]["tile_id"] = tile["tile_id"]
        raw.extend(feats)

    # Cross-tile dedupe rebuilds geometry and drops traced outlines back to
    # their convex-ish union. Tile captures are seamless, so with outline mode
    # there is nothing to dedupe and the shapes are kept exactly as traced.
    merged = raw if (use_cv and outline_mode) else dedupe_across_tiles(raw)

    total_area = sum(f["properties"]["area_m2"] for f in merged)
    collection = {
        "type": "FeatureCollection",
        "features": merged,
        "properties": {
            "capture": manifest.get("name"),
            "aoi": manifest.get("aoi"),
            "source": manifest.get("source"),
            "attribution": manifest.get("attribution"),
            "tiles_processed": len(tiles),
            "detections_raw": len(raw),
            "detections": len(merged),
            "total_area_m2": round(total_area, 2),
            "total_capacity_kw": round(total_area * kw_per_m2, 2),
            "kw_per_m2": kw_per_m2,
            "model": model_kind,
            "checkpoint": Path(checkpoint).name,
            "threshold": (
                getattr(detector, "conf", None) if (use_yolo or use_cv or use_cnn)
                else float(ic["threshold"])
            ),
            # Box-derived areas are upper bounds, not measured footprints.
            "area_is_upper_bound": (use_yolo or use_cnn) or (use_cv and not outline_mode),
        },
    }

    out = capture_dir / "detections.geojson"
    out.write_text(json.dumps(collection), encoding="utf-8")
    return out
