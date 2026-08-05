"""Fine-tune the YOLO detector on locally-labelled imagery.

Starts from the existing checkpoint rather than from scratch. That model already
knows what a PV array looks like in general -- it just learned the wrong
appearance (saturated dark-blue Chinese utility farms) for this region's imagery
(desaturated grey Mediterranean rooftops, often brighter than their
surroundings). Transfer learning re-tunes that appearance model without
discarding what it knows about panel structure, which is the only reason a few
hundred local boxes can be enough.

    python scripts/finetune.py --dataset jbeil --weights unet-training/best.pt
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from solarmap.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="name under data/datasets")
    ap.add_argument("--weights", default="unet-training/best.pt",
                    help="checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.001,
                    help="fine-tuning LR; well below the 0.01 used from scratch")
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze the first N layers (0 = train everything)")
    ap.add_argument("--name", default="jbeil_finetune")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last.pt")
    ap.add_argument("--config")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = Config.load(args.config)

    if args.resume:
        # Resuming restores the optimizer state and LR schedule, so the run
        # continues its cosine decay rather than restarting at a high LR --
        # which would undo the fine-tuning already achieved.
        last = cfg.path("checkpoints") / "runs" / args.name / "weights" / "last.pt"
        if not last.is_file():
            raise SystemExit(f"Nothing to resume: {last} not found")
        print(f"resuming {args.name} from {last}")
        YOLO(str(last)).train(resume=True)
        best = last.with_name("best.pt")
        if best.is_file():
            dest = cfg.path("checkpoints") / f"solar_yolo_{args.dataset}.pt"
            dest.write_bytes(best.read_bytes())
            print(f"installed as {dest.name}")
        return
    data_yaml = cfg.path("datasets") / args.dataset / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"No dataset at {data_yaml}. Run scripts/export_yolo.py first.")

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"No checkpoint at {weights}")

    print(f"fine-tuning {weights.name} on {args.dataset}")
    print(f"  epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} lr={args.lr}")

    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        lrf=0.05,
        freeze=args.freeze or None,
        # A few dozen tiles overfits quickly; stop when val stops improving.
        patience=25,
        # Heavy augmentation to stretch a small dataset. Overhead imagery has
        # no canonical orientation, so flips and rotation are valid rather than
        # distortions; HSV jitter covers the sun-angle and haze variation that
        # broke the original model's colour assumptions.
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.5,
        degrees=180, fliplr=0.5, flipud=0.5,
        scale=0.4, translate=0.15,
        mosaic=1.0, close_mosaic=15,
        project=str(cfg.path("checkpoints") / "runs"),
        name=args.name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    best = cfg.path("checkpoints") / "runs" / args.name / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    if best.is_file():
        dest = cfg.path("checkpoints") / f"solar_yolo_{args.dataset}.pt"
        dest.write_bytes(best.read_bytes())
        print(f"installed as {dest.name} -- it will appear in the UI dropdown")


if __name__ == "__main__":
    main()
