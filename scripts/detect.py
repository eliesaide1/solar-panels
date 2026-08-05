"""Run detection over a capture.

    python scripts/detect.py --capture mission
"""

import argparse
import json

import _bootstrap  # noqa: F401

from solarmap.config import Config
from solarmap.infer.pipeline import detect_capture


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True, help="capture name under data/captures")
    ap.add_argument("--checkpoint", default="solar_unet.pt")
    ap.add_argument("--threshold", type=float,
                    help="U-Net probability threshold")
    ap.add_argument("--conf", type=float,
                    help="YOLO confidence threshold (default from config.yaml)")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.threshold is not None:
        cfg.raw["inference"]["threshold"] = args.threshold
    if args.conf is not None:
        cfg.raw.setdefault("yolo", {})["conf"] = args.conf

    capture_dir = cfg.path("captures") / args.capture
    ckpt = cfg.path("checkpoints") / args.checkpoint

    def progress(i: int, total: int, label: str) -> None:
        print(f"\r[{i}/{total}] {label}", end="", flush=True)

    out = detect_capture(cfg, capture_dir, ckpt, progress=progress)
    props = json.loads(out.read_text(encoding="utf-8"))["properties"]
    print(
        f"\n{props['detections']} arrays  "
        f"{props['total_area_m2']:,.0f} m2  "
        f"~{props['total_capacity_kw']:,.1f} kW\n{out}"
    )


if __name__ == "__main__":
    main()
