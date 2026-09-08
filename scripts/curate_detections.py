"""Remove known-wrong detections from a capture, using its verified labels.

This produces a survey PRODUCT, not a measurement. It deletes detections that
land on no verified array, so precision on the result is 100% by construction
and must never be quoted as detector accuracy -- at Jbeil that is 83.1%, and the
uncurated layer is kept alongside so the honest number stays reachable.

Curating a delivered map is normal GIS practice, and it is legitimate here for
one reason only: the whole capture was swept by hand, so "no verified array"
means a human looked and found nothing, rather than nobody having checked. On an
unswept capture this would silently delete real detections.

It also does not generalise. A new region gets the raw output, greenhouses and
all, until somebody labels it -- which is exactly why the honest figure is the
one that belongs in a report.

    python scripts/curate_detections.py --capture jbeil-mb-104 --labels labels_clean.json
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np

from solarmap.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels_clean.json")
    ap.add_argument("--detections", default="detections.geojson")
    ap.add_argument("--out", default="detections_curated.geojson")
    ap.add_argument("--keep-raw", default="detections_raw.geojson",
                    help="where the uncurated layer is preserved")
    ap.add_argument("--min-on-panel", type=float, default=0.5,
                    help="fraction of a detection that must lie on a verified "
                         "array for it to survive (default 0.5, matching "
                         "score_arrays.py so the two agree on what is correct)")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    man = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    gt = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    det = json.loads((cap / args.detections).read_text(encoding="utf-8"))

    n_verified = sum(1 for bs in gt["tiles"].values()
                     for b in bs if b.get("verified") is True)
    if not n_verified:
        raise SystemExit(f"No verified arrays in {args.labels} -- refusing to "
                         "curate, this would delete every detection.")

    keep, dropped = [], 0
    for t in man["tiles"]:
        tid = t["tile_id"]
        W, H = t["width"], t["height"]
        n, s, e, w = t["north"], t["south"], t["east"], t["west"]

        panel = np.zeros((H, W), bool)
        for b in gt["tiles"].get(tid, []):
            if b.get("verified") is True:
                panel[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = True

        for f in det["features"]:
            ring = f["geometry"]["coordinates"][0]
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            if not (w <= cx <= e and s <= cy <= n):
                continue
            poly = np.array([[(lon - w) / (e - w) * W, (n - lat) / (n - s) * H]
                             for lon, lat in ring], np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            mb = m.astype(bool)
            a = int(mb.sum())
            if a < 4:
                continue
            if (mb & panel).sum() / a >= args.min_on_panel:
                keep.append(f)
            else:
                dropped += 1

    area = sum(float(f["properties"].get("area_m2") or 0) for f in keep)
    kw = sum(float(f["properties"].get("capacity_kw") or 0) for f in keep)

    src_props = det.get("properties", {})
    out = {
        "type": "FeatureCollection",
        # Carry the source layer's properties forward rather than rebuilding a
        # subset of them. The UI reads kw_per_m2, tiles_processed and
        # area_is_upper_bound; rebuilding dropped all three, so a curated layer
        # rendered "Capacity assumes undefined kW/m2" over a capacity computed
        # from exactly that figure.
        "properties": {
            **src_props,
            "checkpoint": f"{src_props.get('checkpoint')} [CURATED]",
            "detections": len(keep),
            "total_area_m2": round(area, 1),
            "total_capacity_kw": round(kw, 2),
            "curation": "false positives removed using this capture's verified "
                        "labels. Precision here is 100% by construction and is "
                        "NOT detector accuracy; see the uncurated layer.",
            "uncurated_layer": args.keep_raw,
        },
        "features": keep,
    }

    # Always refresh the uncurated layer, so re-running on a rebuilt input
    # cannot leave the honest layer describing a detector that no longer
    # exists -- it is the one the README tells you to quote, and it used to go
    # stale silently while the curated layer beside it updated. Refuse only
    # when the input is itself curated, which is what the old check was
    # reaching for.
    if "[CURATED]" in str(src_props.get("checkpoint", "")):
        raise SystemExit(
            f"{args.detections} is already curated. Curating it again would "
            f"overwrite {args.keep_raw} with a layer that has had its false "
            "positives removed, destroying the only honest record. Point "
            "--detections at the raw merged layer instead."
        )
    (cap / args.keep_raw).write_text(json.dumps(det), encoding="utf-8")
    print(f"uncurated layer preserved -> {args.keep_raw}")
    (cap / args.out).write_text(json.dumps(out), encoding="utf-8")

    total = len(keep) + dropped
    print(f"{total} detections -> kept {len(keep)}, removed {dropped}")
    print(f"{area:,.0f} m2  ~{kw:,.0f} kW")
    print(f"-> {cap / args.out}")
    print("\nQuote the UNCURATED score in any report:")
    print(f"  python scripts/score_arrays.py --capture {args.capture} "
          f"--labels {args.labels} --detections {args.keep_raw}")


if __name__ == "__main__":
    main()
