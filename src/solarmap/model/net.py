"""Segmentation network and loss."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(encoder: str = "resnet34", encoder_weights: str | None = "imagenet") -> nn.Module:
    """U-Net with an ImageNet-pretrained encoder, one output channel.

    Solar arrays are small, high-contrast, strongly rectilinear objects. U-Net's
    skip connections preserve the fine edges that a plain encoder-decoder
    blurs away at 0.1-0.3 m/px, which is what the polygonisation step needs.
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )


def preprocessing_params(encoder: str, encoder_weights: str | None = "imagenet") -> dict:
    """Normalisation stats the pretrained encoder expects."""
    return smp.encoders.get_preprocessing_params(encoder, encoder_weights)


class DiceBCELoss(nn.Module):
    """BCE + Dice.

    Panels cover a small fraction of any tile, so the classes are heavily
    imbalanced. BCE alone lets the model score well by predicting "roof
    everywhere"; the Dice term makes it pay for missing the positive class.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * target).sum(dim=(1, 2, 3)) + self.smooth
        den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth
        dice = 1.0 - (num / den).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter / union).item()) if union > 0 else 1.0
