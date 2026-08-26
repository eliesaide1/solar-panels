"""Copy a capture, resampled to a different ground sample distance.

A segmentation model is only scale-invariant to the extent its training
augmentation made it so, and BDAPPV's is not: the same model on the same roofs
with the same labels finds 0.6% of arrays at 6 cm and 23.2% at 10 cm. Matching
the imagery to the checkpoint is therefore a real tuning knob, and it needs to
be measurable rather than done by hand -- the existing 8.1 cm and 10.4 cm
captures were produced ad hoc, with no script to reproduce or extend them.

Geometry is untouched: every tile keeps its bounds, only the pixel grid
changes. World files are rewritten so the copy stays georeferenced, which is
what lets labels transfer between resolutions and scores stay comparable.

    python scripts/resample_capture.py --capture jbeil-mb --gsd 0.15
    python scripts/resample_capture.py --capture jbeil-mb --gsd 0.15 --name jbeil-mb-150
"""

import argparse
import json
import shutil

import _bootstrap  # noqa: F401

from PIL import Image

from solarmap.config import Config
from solarmap.geo import write_world_file, write_world_file_proj

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--gsd", type=float, required=True,
                    help="target ground sample distance, metres per pixel")
    ap.add_argument("--name", help="output capture name "
                                   "(default <capture>-<gsd in cm>)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    src = cfg.path("captures") / args.capture
    name = args.name or f"{args.capture}-{round(args.gsd * 1000):03d}"
    dst = cfg.path("captures") / name

    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    src_gsd = float(man["tiles"][0]["gsd_m"])
    if args.gsd < src_gsd:
        raise SystemExit(
            f"Target {args.gsd} m/px is finer than the source {src_gsd:.4f} m/px. "
            "Upsampling invents detail the imagery never had; recapture instead.")

    (dst / "tiles").mkdir(parents=True, exist_ok=True)
    print(f"{args.capture} @ {src_gsd:.4f} m/px  ->  {name} @ {args.gsd} m/px")

    out_tiles = []
    for i, t in enumerate(man["tiles"], 1):
        tid = t["tile_id"]
        sp = src / t["image"]
        if not sp.exists():
            continue
        with Image.open(sp) as im:
            im = im.convert("RGB")
            # Derive the new size from the tile's true ground span, so rounding
            # error cannot accumulate into a georeferencing drift.
            w = max(1, int(round(t["width"] * src_gsd / args.gsd)))
            h = max(1, int(round(t["height"] * src_gsd / args.gsd)))
            # LANCZOS: downsampling with a box/bilinear kernel aliases the
            # module grid into moire, which is exactly the signal being tested.
            im.resize((w, h), Image.LANCZOS).save(
                dst / t["image"], quality=args.quality)

        rec = dict(t)
        rec["width"], rec["height"] = w, h
        rec["gsd_m"] = args.gsd
        out_tiles.append(rec)

        bp = t.get("bounds_proj")
        if bp:
            write_world_file_proj(dst / t["image"], w, h,
                                  bp[0], bp[1], bp[2], bp[3],
                                  t.get("crs", "EPSG:3857"))
        else:
            write_world_file(dst / t["image"], w, h,
                             t["north"], t["south"], t["east"], t["west"])
        print(f"\r  {i}/{len(man['tiles'])} {tid} -> {w}x{h}", end="", flush=True)

    out = dict(man)
    out["name"] = name
    out["source"] = f"{man.get('source', args.capture)} (resampled to {args.gsd} m/px)"
    out["gsd_m"] = args.gsd
    out["tiles"] = out_tiles
    out["tile_count"] = len(out_tiles)
    (dst / "manifest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    for extra in ("labels.json",):
        if (src / extra).exists():
            shutil.copyfile(src / extra, dst / f"_src_{extra}")

    print(f"\n{len(out_tiles)} tiles -> {dst}")
    print(f"transfer labels with:\n"
          f"  python scripts/transfer_labels.py --from {args.capture} "
          f"--to {name} --out labels_clean.json")


if __name__ == "__main__":
    main()
