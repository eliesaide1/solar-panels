"""Union detections from several captures of the same ground into one layer.

Only worth doing with a scale-robust checkpoint. A model trained at one gsd
collapses off it -- measured on the old checkpoint, recall went 0.6% at 6.2 cm,
23.2% at 10.4 cm and 5.6% at 20 cm -- so unioning its output across resolutions
just adds noise. A model trained with wide scale augmentation holds its accuracy
across the range, which makes the resolutions genuinely independent looks at the
same roof: an array missed at 10 cm may be caught at 8 or 12.

Overlapping polygons are dissolved rather than concatenated. Keeping duplicates
would inflate the detection count and make precision look worse than it is,
because the same array found three times would score as two false positives.

    python scripts/merge_detections.py --into jbeil-mb-104 --out det_union.geojson \
        --from jbeil-mb-081:det_ms_070.geojson jbeil-mb-104:det_ms_070.geojson \
               jbeil-mb-125:det_ms_070.geojson
"""

import argparse
import json

import _bootstrap  # noqa: F401

from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from solarmap.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", required=True,
                    help="capture whose directory receives the merged file")
    ap.add_argument("--sources", nargs="+", required=True, metavar="CAPTURE:FILE",
                    help="detection files to union, as capture:filename")
    ap.add_argument("--out", default="det_union.geojson")
    ap.add_argument("--min-area-m2", type=float, default=2.0)
    ap.add_argument("--min-sources", type=int, default=1,
                    help="how many input layers must cover a shape for it to "
                         "survive. 1 unions them; 2 keeps only what two models "
                         "independently agree on, which trades recall for "
                         "precision and is usually what a demo wants.")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    dst = cfg.path("captures") / args.into

    polys, n_in, layers = [], 0, []
    kw_per_m2 = float(cfg["capacity"]["kw_per_m2"])
    for spec in args.sources:
        cap_name, _, fname = spec.partition(":")
        path = cfg.path("captures") / cap_name / (fname or "detections.geojson")
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        gj = json.loads(path.read_text(encoding="utf-8"))
        feats = gj.get("features", [])
        n_in += len(feats)
        print(f"  {cap_name}/{path.name}: {len(feats)} detections")
        layer = []
        for f in feats:
            g = shape(f["geometry"])
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                polys.append(g)
                layer.append(g)
        layers.append(unary_union(layer) if layer else None)

    if not polys:
        raise SystemExit("Nothing to merge.")

    merged = unary_union(polys)
    parts = list(getattr(merged, "geoms", [merged]))
    print(f"\n{n_in} detections -> {len(parts)} after dissolving overlaps")

    if args.min_sources > 1:
        kept = []
        for g in parts:
            votes = sum(1 for lay in layers if lay is not None and g.intersects(lay))
            if votes >= args.min_sources:
                kept.append(g)
        print(f"{len(kept)} survive agreement by >= {args.min_sources} of "
              f"{len(layers)} layers")
        parts = kept

    # Areas are only meaningful in metres, so measure in the capture's own
    # projected CRS rather than in degrees. The manifest carries the projected
    # bounds; fall back to a latitude-corrected approximation without them.
    man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    t0 = man["tiles"][0]
    if t0.get("bounds_proj"):
        import math
        lat = math.radians((t0["north"] + t0["south"]) / 2.0)
        # Web Mercator metres are inflated by 1/cos(lat); undo that.
        mx = (t0["bounds_proj"][2] - t0["bounds_proj"][0]) / (t0["east"] - t0["west"])
        m_per_deg_x = mx * math.cos(lat)
        m_per_deg_y = m_per_deg_x
    else:
        import math
        lat = math.radians((t0["north"] + t0["south"]) / 2.0)
        m_per_deg_y = 111_132.0
        m_per_deg_x = 111_320.0 * math.cos(lat)

    feats, total_area = [], 0.0
    for g in parts:
        if not isinstance(g, Polygon):
            continue
        area = g.area * m_per_deg_x * m_per_deg_y
        if area < args.min_area_m2:
            continue
        total_area += area
        feats.append({"type": "Feature", "geometry": mapping(g),
                      "properties": {"area_m2": round(area, 1),
                                     "capacity_kw": round(area * kw_per_m2, 2),
                                     "confidence": None}})

    out = {"type": "FeatureCollection",
           "properties": {"checkpoint": "union:" + ",".join(args.sources),
                          "threshold": None,
                          "detections": len(feats),
                          "total_area_m2": round(total_area, 1),
                          "total_capacity_kw": round(total_area * kw_per_m2, 2)},
           "features": feats}
    (dst / args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"{len(feats)} arrays  {total_area:,.0f} m2  "
          f"~{total_area*kw_per_m2:,.0f} kW\n{dst / args.out}")


if __name__ == "__main__":
    main()
