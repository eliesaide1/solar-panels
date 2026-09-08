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

from pyproj import Transformer
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as shapely_transform, unary_union

from solarmap.config import Config
from solarmap.geo import WGS84, utm_crs_for


def _votes_for(layer, g, min_overlap: float) -> bool:
    """True if ``layer`` covers enough of dissolved shape ``g`` to count as agreeing.

    Measured in degrees, which is fine: this is a ratio of two areas in the
    same units over ground far smaller than one degree, so the projection
    cancels. Only absolute areas need UTM.
    """
    if layer is None or not layer.intersects(g):
        return False
    if g.area <= 0:
        return False
    return layer.intersection(g).area / g.area >= min_overlap


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
    ap.add_argument("--min-overlap", type=float, default=0.0,
                    help="fraction of a dissolved shape a layer must cover to "
                         "count as one of its --min-sources votes. 0 is bare "
                         "topological contact, so a shape that merely grazes a "
                         "layer scores a full vote -- unanimity reachable "
                         "without any model having agreed about the array. That "
                         "is unsound, and it is also, measured, the best "
                         "operating point at Jbeil: tightening it costs F1 "
                         "monotonically (0%%: 55.2/83.1/0.663; 10%%: "
                         "53.6/84.8/0.657; 25%%: 51.6/84.9/0.642; 40%%: "
                         "45.8/91.0/0.609). The default therefore stays at the "
                         "published behaviour rather than quietly moving it. "
                         "Raise it when precision is worth more than recall, or "
                         "re-derive it on a region where grazing votes turn out "
                         "to cost something. It cannot go much past 0.5 without "
                         "failing the case the ensemble exists for: two models "
                         "finding complementary halves of one array.")
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
            votes = sum(1 for lay in layers if _votes_for(lay, g, args.min_overlap))
            if votes >= args.min_sources:
                kept.append(g)
        print(f"{len(kept)} survive agreement by >= {args.min_sources} of "
              f"{len(layers)} layers (overlap >= {args.min_overlap:.0%})")
        parts = kept

    # Areas are only meaningful in metres, so project to UTM and measure there
    # -- the same thing vectorize.py does, so a merged layer and a single-model
    # layer report areas on the same basis.
    #
    # This previously scaled degrees by a per-axis factor and set the latitude
    # factor equal to the longitude one, which had already been multiplied by
    # cos(lat) for meridian convergence. Latitude degrees do not converge, so
    # every merged area came out a factor of cos(lat) low -- 17% at Jbeil, and
    # worse further from the equator. Capacity totals inherited the error.
    man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    t0 = man["tiles"][0]
    to_utm = Transformer.from_crs(
        WGS84,
        utm_crs_for((t0["north"] + t0["south"]) / 2.0, (t0["east"] + t0["west"]) / 2.0),
        always_xy=True,
    ).transform

    feats, total_area = [], 0.0
    for g in parts:
        if not isinstance(g, Polygon):
            continue
        area = shapely_transform(to_utm, g).area
        if area < args.min_area_m2:
            continue
        total_area += area
        feats.append({"type": "Feature", "geometry": mapping(g),
                      "properties": {"area_m2": round(area, 1),
                                     "capacity_kw": round(area * kw_per_m2, 2),
                                     # A mean confidence over several models is
                                     # not a probability of anything; the UI
                                     # renders this as "n/a" rather than 0%.
                                     "confidence": None}})

    out = {"type": "FeatureCollection",
           "properties": {"checkpoint": "union:" + ",".join(args.sources),
                          "threshold": None,
                          "detections": len(feats),
                          "total_area_m2": round(total_area, 1),
                          "total_capacity_kw": round(total_area * kw_per_m2, 2),
                          # The UI reads these three. Without them it rendered
                          # "Capacity assumes undefined kW/m2" over a capacity
                          # derived from exactly that number, and "0 tiles".
                          "kw_per_m2": kw_per_m2,
                          "tiles_processed": len(man["tiles"]),
                          # Every input traced its outlines, so the union does
                          # too: these are measurements, not box upper bounds.
                          "area_is_upper_bound": False,
                          "min_sources": args.min_sources,
                          "min_overlap": args.min_overlap},
           "features": feats}
    (dst / args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"{len(feats)} arrays  {total_area:,.0f} m2  "
          f"~{total_area*kw_per_m2:,.0f} kW\n{dst / args.out}")


if __name__ == "__main__":
    main()
