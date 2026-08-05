"""Convert an externally-trained U-Net checkpoint into SolarMap's format.

Different training notebooks name their checkpoint keys differently. This
translates the common variants, verifies the weights actually load into the
architecture they claim, runs a forward pass, and only then writes the result
into `models/`. A checkpoint that fails any of those checks is rejected here
rather than halfway through a survey.

    python scripts/import_checkpoint.py --src "C:/Users/elie-s/Downloads/best.pth"
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from solarmap.config import Config

# SolarMap key <- any of these source keys.
ALIASES = {
    "state_dict": ("state_dict", "model_state_dict", "model", "net"),
    "encoder": ("encoder", "encoder_name", "backbone"),
    "tile_size": ("tile_size", "image_size", "img_size", "input_size"),
    "val_iou": ("val_iou", "validation_iou", "best_iou", "iou"),
    "epoch": ("epoch", "epochs", "best_epoch"),
    "threshold": ("threshold", "best_threshold"),
}

# Most segmentation training uses ImageNet statistics; smp's pretrained
# encoders expect them. Only assumed when the checkpoint records nothing.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pick(ckpt: dict, canonical: str, default=None):
    for key in ALIASES[canonical]:
        if key in ckpt:
            return ckpt[key]
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="path to the checkpoint (.pt/.pth)")
    ap.add_argument("--out", default="solar_unet.pt", help="filename to write in models/")
    ap.add_argument("--encoder", help="override the encoder name if not recorded")
    ap.add_argument("--tile-size", type=int, help="override the training tile size")
    ap.add_argument("--config")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        raise SystemExit(f"No such file: {src}")

    print(f"reading {src.name} ({src.stat().st_size/1e6:.0f} MB)")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise SystemExit(
            "This file is not a checkpoint dictionary. If it is a whole pickled "
            "model, re-save it as model.state_dict() and try again."
        )

    # A bare state_dict (tensor values at the top level) rather than a wrapper.
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        print("  looks like a bare state_dict")
        state, meta = ckpt, {}
    else:
        print(f"  keys: {', '.join(sorted(ckpt.keys()))[:160]}")
        state, meta = pick(ckpt, "state_dict"), ckpt
        if state is None:
            raise SystemExit(
                f"No weights found. Expected one of {ALIASES['state_dict']}."
            )

    encoder = args.encoder or pick(meta, "encoder")
    if not encoder:
        raise SystemExit(
            "The checkpoint does not record its encoder. Pass it explicitly, "
            "e.g. --encoder efficientnet-b3"
        )
    tile_size = args.tile_size or int(pick(meta, "tile_size", 512) or 512)

    print(f"  encoder   : {encoder}")
    print(f"  tile size : {tile_size}")
    for label, key in (("val IoU", "val_iou"), ("epoch", "epoch"), ("threshold", "threshold")):
        v = pick(meta, key)
        if v is not None:
            print(f"  {label:10}: {v}")

    # Strip DataParallel's "module." prefix if present.
    if any(k.startswith("module.") for k in state):
        print("  stripping 'module.' prefix (DataParallel checkpoint)")
        state = {k.removeprefix("module."): v for k, v in state.items()}

    print("\nverifying the weights load into a U-Net with that encoder...")
    from solarmap.model.net import build_model

    model = build_model(encoder, encoder_weights=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  missing keys   : {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")
        if len(missing) > 10:
            raise SystemExit(
                f"{len(missing)} weights are missing — this checkpoint does not "
                f"match a U-Net with encoder '{encoder}'. Check --encoder.\n"
                f"  e.g. {missing[:3]}"
            )
    else:
        print("  all weights matched exactly")

    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, tile_size, tile_size))
    if tuple(out.shape) != (1, 1, tile_size, tile_size):
        raise SystemExit(f"Unexpected output shape {tuple(out.shape)}; expected 1 channel.")
    print(f"  forward pass OK -> {tuple(out.shape)}")

    cfg = Config.load(args.config)
    dest_dir = cfg.path("checkpoints")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / args.out

    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoder": encoder,
            "tile_size": tile_size,
            # Recorded explicitly so inference normalises exactly as training did.
            "mean": tuple(meta.get("mean", IMAGENET_MEAN)),
            "std": tuple(meta.get("std", IMAGENET_STD)),
            "val_iou": float(pick(meta, "val_iou", 0.0) or 0.0),
            "epoch": int(pick(meta, "epoch", 0) or 0),
            "imported_from": src.name,
        },
        dest,
    )
    print(f"\nwrote {dest}  ({dest.stat().st_size/1e6:.0f} MB)")

    thr = pick(meta, "threshold")
    if thr is not None and abs(float(thr) - float(cfg["inference"]["threshold"])) > 1e-6:
        print(
            f"\nNOTE: this model was tuned at threshold {thr}, but config.yaml "
            f"has inference.threshold: {cfg['inference']['threshold']}. "
            "Consider matching them."
        )
    print("Restart the server and it will appear in the checkpoint dropdown.")


if __name__ == "__main__":
    main()
