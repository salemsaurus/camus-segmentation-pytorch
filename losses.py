"""Loss functions for CAMUS segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn


def _multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute multiclass Dice loss from logits and integer masks."""
    if logits.ndim != 4:
        raise ValueError("Expected logits shape [B, C, H, W]")
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    probs = torch.softmax(logits, dim=1)
    targets = targets.long()

    batch_size = logits.size(0)
    losses = []
    for cls in range(num_classes):
        if ignore_background and cls == 0:
            continue
        pred_cls = probs[:, cls]
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice = (2.0 * intersection + eps) / (union + eps)
        losses.append(1.0 - dice)

    if not losses:
        raise ValueError("No classes were selected for Dice loss")

    loss = torch.stack(losses, dim=0).mean(dim=0)
    return loss.mean()


def segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute segmentation loss for CAMUS segmentation.

    For multiclass (num_classes > 1):
        - Uses multiclass Dice loss on classes 1..C-1
        - Masks are class indices: {0=bg, 1=LV, 2=myocardium, 3=atrium}
        - Logits shape: [B, C, H, W]
        - Masks shape: [B, H, W] with dtype torch.long

    For binary (num_classes == 1):
        - Uses BCEWithLogitsLoss
        - Logits shape: [B, 1, H, W]
        - Masks shape: [B, H, W] or [B, 1, H, W], float32
    """
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        return criterion(logits, masks.float())
    return _multiclass_dice_loss(logits, masks, num_classes=num_classes)
