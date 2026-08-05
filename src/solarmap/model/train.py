"""Training loop."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from .dataset import SolarSegDataset
from .net import DiceBCELoss, build_model, iou_score, preprocessing_params


def train(cfg: Config, dataset_dir: str | Path, out_name: str = "solar_unet.pt") -> Path:
    mc = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(
            "WARNING: no CUDA device found. Training on CPU will take many hours; "
            "consider running this on Colab or a GPU box and copying the "
            "checkpoint back."
        )

    params = preprocessing_params(mc["encoder"], mc["encoder_weights"])
    mean, std = tuple(params["mean"]), tuple(params["std"])

    train_ds = SolarSegDataset(dataset_dir, "train", mc["tile_size"], mean, std)
    val_ds = SolarSegDataset(dataset_dir, "val", mc["tile_size"], mean, std)
    print(f"train={len(train_ds)} tiles  val={len(val_ds)} tiles  device={device}")

    # num_workers=0: Windows spawns rather than forks, and the extra workers
    # cost more in process startup than they save on this dataset size.
    train_dl = DataLoader(train_ds, batch_size=mc["batch_size"], shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=mc["batch_size"], shuffle=False, num_workers=0)

    model = build_model(mc["encoder"], mc["encoder_weights"]).to(device)
    criterion = DiceBCELoss(bce_weight=mc["bce_weight"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(mc["lr"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=mc["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    ckpt_dir = cfg.path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / out_name
    best_iou = -1.0

    for epoch in range(1, int(mc["epochs"]) + 1):
        model.train()
        running = 0.0
        for images, masks in tqdm(train_dl, desc=f"epoch {epoch}/{mc['epochs']}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item()
        scheduler.step()

        model.eval()
        ious, val_loss = [], 0.0
        with torch.no_grad():
            for images, masks in val_dl:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item()
                ious.append(iou_score(logits, masks))
        mean_iou = sum(ious) / max(len(ious), 1)

        print(
            f"epoch {epoch:3d}  train_loss={running / max(len(train_dl), 1):.4f}  "
            f"val_loss={val_loss / max(len(val_dl), 1):.4f}  val_IoU={mean_iou:.4f}"
        )

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder": mc["encoder"],
                    "tile_size": mc["tile_size"],
                    "mean": mean,
                    "std": std,
                    "val_iou": mean_iou,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path.name} (val_IoU={mean_iou:.4f})")

    print(f"done. best val_IoU={best_iou:.4f} -> {ckpt_path}")
    return ckpt_path
