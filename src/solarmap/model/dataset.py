"""Training dataset.

Expected on-disk layout (this is what ``scripts/prepare_bdappv.py`` produces):

    data/datasets/<name>/
        train/images/*.png   train/masks/*.png
        val/images/*.png     val/masks/*.png

Masks are single-channel, non-zero where a panel is present. Image and mask
filenames must match.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset


class SolarSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile_size: int = 512,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool | None = None,
    ):
        self.root = Path(root) / split
        self.images = sorted((self.root / "images").glob("*"))
        self.masks_dir = self.root / "masks"
        if not self.images:
            raise FileNotFoundError(f"No images under {self.root / 'images'}")

        augment = (split == "train") if augment is None else augment
        self.transform = _build_transform(tile_size, mean, std, augment)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        mask_path = _match_mask(self.masks_dir, img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unreadable mask for {img_path.name}")
        mask = (mask > 127).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)


def _match_mask(masks_dir: Path, img_path: Path) -> Path:
    exact = masks_dir / img_path.name
    if exact.exists():
        return exact
    for candidate in masks_dir.glob(img_path.stem + ".*"):
        return candidate
    raise FileNotFoundError(f"No mask found for {img_path.name} in {masks_dir}")


def _build_transform(tile_size, mean, std, augment: bool):
    from albumentations.pytorch import ToTensorV2

    if augment:
        stages = [
            A.RandomResizedCrop(
                size=(tile_size, tile_size), scale=(0.7, 1.0), ratio=(0.9, 1.11)
            ),
            # Overhead imagery has no canonical orientation, so the full
            # dihedral group is valid augmentation rather than a distortion.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            # Google Earth mosaics stitch captures from different dates,
            # sensors and sun angles; this is the dominant domain shift.
            A.RandomBrightnessContrast(0.25, 0.25, p=0.7),
            A.HueSaturationValue(10, 20, 12, p=0.4),
            A.GaussNoise(p=0.2),
            A.MotionBlur(blur_limit=3, p=0.15),
        ]
    else:
        stages = [A.Resize(tile_size, tile_size)]

    return A.Compose(stages + [A.Normalize(mean=mean, std=std), ToTensorV2()])
