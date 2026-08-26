"""Fine-tune a U-Net checkpoint on local labels. Runs on CPU in about an hour.

The notebook does this on a GPU as stage B, but stage B is the cheap half --
806 crops for 20 epochs is minutes on a T4 and roughly 100 minutes on eight CPU
threads. That makes it worth having locally: fine-tuning experiments can be run
here without spending GPU quota, and against the same checkpoint the notebook
would start from.

Selection is on ARRAY F1, not validation IoU. IoU rewards tracing an array the
model already finds more tightly, which is not the goal -- finding arrays it
currently misses is. The matching rule matches scripts/score_arrays.py so the
number here predicts the one on a real capture instead of merely correlating.

    python scripts/finetune_unet.py --dataset jbeil06_seg --weights solar_unet.pt
"""

import argparse
import time

import _bootstrap  # noqa: F401

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from solarmap.config import Config

TILE = 256
# Same range the notebook uses: scaling by f then cropping TILE px gives an
# effective gsd of source/f, so [0.5, 2.0] spans roughly 5-20 cm.
SCALE_LO, SCALE_HI = 0.5, 2.0


def build_tf(train, mean, std):
    if train:
        stages = [
            A.RandomScale(scale_limit=(SCALE_LO - 1.0, SCALE_HI - 1.0), p=1.0),
            A.PadIfNeeded(min_height=TILE, min_width=TILE,
                          border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            # Scaling up to f=2 means the window sees a quarter of the ground,
            # so a plain random crop loses the array: measured 49.7% survival
            # against 70.3% in the raw crops. A mask-aware crop restores it to
            # 72.7%; the random half still supplies hard negatives.
            A.OneOf([
                A.CropNonEmptyMaskIfExists(height=TILE, width=TILE, p=1.0),
                A.RandomCrop(height=TILE, width=TILE, p=1.0),
            ], p=1.0),
            A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomRotate90(p=1.),
            A.RandomBrightnessContrast(.25, .25, p=.7),
            A.HueSaturationValue(10, 20, 12, p=.4),
        ]
    else:
        stages = [A.PadIfNeeded(min_height=TILE, min_width=TILE,
                                border_mode=cv2.BORDER_REFLECT_101, p=1.0),
                  A.CenterCrop(height=TILE, width=TILE, p=1.0)]
    return A.Compose(stages + [A.Normalize(mean=mean, std=std), ToTensorV2()])


class SegDS(Dataset):
    def __init__(self, root, split, mean, std):
        self.imgs = sorted((root / split / "images").glob("*"))
        self.mdir = root / split / "masks"
        self.tf = build_tf(split == "train", mean, std)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(self.mdir / p.name), cv2.IMREAD_GRAYSCALE)
        m = np.zeros(img.shape[:2], np.uint8) if m is None else m
        o = self.tf(image=img, mask=(m > 127).astype(np.float32))
        return o["image"], o["mask"].unsqueeze(0)


class DiceBCE(nn.Module):
    """Panels are a small fraction of any tile, so BCE alone scores well by
    predicting roof everywhere. Dice makes it pay for missing positives."""

    def __init__(self, w=.5):
        super().__init__()
        self.w = w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, t):
        p = torch.sigmoid(logits)
        num = 2 * (p * t).sum((1, 2, 3)) + 1
        den = p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1
        return self.w * self.bce(logits, t) + (1 - self.w) * (1 - (num / den).mean())


@torch.no_grad()
def array_scores(model, loader, thr, min_cover=.5, min_on=.5):
    """Arrays located and detections correct -- same rule as score_arrays.py."""
    located = total = tp = fp = 0
    model.eval()
    for x, y in loader:
        pr = (torch.sigmoid(model(x)) > thr).numpy()[:, 0]
        gt = y.numpy()[:, 0] > .5
        for p, g in zip(pr, gt):
            n, lab = cv2.connectedComponents(g.astype(np.uint8))
            for i in range(1, n):
                m = lab == i
                if m.sum() < 4:
                    continue
                total += 1
                located += bool((m & p).sum() / m.sum() >= min_cover)
            n2, lab2 = cv2.connectedComponents(p.astype(np.uint8))
            for i in range(1, n2):
                m = lab2 == i
                if m.sum() < 4:
                    continue
                if (m & g).sum() / m.sum() >= min_on:
                    tp += 1
                else:
                    fp += 1
    # Plain floats: numpy scalars in a checkpoint make torch.load fail under
    # the weights_only=True default introduced in PyTorch 2.6.
    rec = float(located / max(total, 1))
    prec = float(tp / max(tp + fp, 1))
    return rec, prec, float(2 * prec * rec / max(prec + rec, 1e-9)), int(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="name under data/datasets")
    ap.add_argument("--weights", default="solar_unet.pt",
                    help="checkpoint under models/ to start from")
    ap.add_argument("--out", default="solar_unet_ft.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="a tenth of the pretraining LR; higher overwrites what "
                         "the base checkpoint knows instead of adapting it")
    ap.add_argument("--config")
    args = ap.parse_args()

    import segmentation_models_pytorch as smp

    cfg = Config.load(args.config)
    root = cfg.path("datasets") / args.dataset
    ckpt_path = cfg.path("checkpoints") / args.weights
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = tuple(float(v) for v in ckpt["mean"])
    std = tuple(float(v) for v in ckpt["std"])
    encoder = ckpt.get("encoder", "resnet34")

    tr = DataLoader(SegDS(root, "train", mean, std), args.batch,
                    shuffle=True, drop_last=True)
    va = DataLoader(SegDS(root, "val", mean, std), args.batch)
    print(f"start from {args.weights} ({encoder})")
    print(f"{len(tr.dataset)} train crops, {len(va.dataset)} val crops, "
          f"{torch.get_num_threads()} threads")

    model = smp.Unet(encoder, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ckpt["state_dict"])
    crit = DiceBCE(.5)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    r0, p0, f0, n0 = array_scores(model, va, .5)
    print(f"before      recall={r0:.3f} precision={p0:.3f} F1={f0:.3f} "
          f"over {n0} val arrays\n")

    best = -1.0
    out_path = cfg.path("checkpoints") / args.out
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for x, y in tr:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()

        r, p, f1, _ = array_scores(model, va, .5)
        flag = ""
        if f1 > best:
            best = f1
            torch.save({"state_dict": model.state_dict(), "encoder": encoder,
                        "tile_size": 512, "train_tile": TILE,
                        "scale_range": [SCALE_LO, SCALE_HI],
                        "mean": mean, "std": std, "val_score": float(f1),
                        "epoch": int(ep), "stage": f"finetuned:{args.dataset}"},
                       out_path)
            flag = "  <- saved"
        print(f"ep {ep:3d}  loss={run/max(len(tr),1):.4f}  recall={r:.3f} "
              f"precision={p:.3f} F1={f1:.3f}  {time.time()-t0:.0f}s{flag}",
              flush=True)

    print(f"\nbest array F1 {best:.3f} -> {out_path}")
    print("Validate on a real capture, not these crops:")
    print(f"  python scripts/detect.py --capture jbeil-mb-104 --checkpoint {args.out}")
    print("  python scripts/score_arrays.py --capture jbeil-mb-104 "
          "--labels labels_clean.json")


if __name__ == "__main__":
    main()
