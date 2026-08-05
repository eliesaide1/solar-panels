"""Train the segmentation model.

    python scripts/train.py --dataset data/datasets/bdappv
"""

import argparse

import _bootstrap  # noqa: F401

from solarmap.config import Config
from solarmap.model.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="dataset root with train/ and val/")
    ap.add_argument("--out", default="solar_unet.pt", help="checkpoint filename")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.epochs:
        cfg.raw["model"]["epochs"] = args.epochs
    if args.batch_size:
        cfg.raw["model"]["batch_size"] = args.batch_size

    train(cfg, args.dataset, args.out)


if __name__ == "__main__":
    main()
