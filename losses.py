"""Loss functions for CAMUS segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn


def segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute segmentation loss for multiclass CAMUS segmentation.

    For multiclass (num_classes > 1):
        - Uses CrossEntropyLoss (raw logits input)
        - Masks are class indices: {0=bg, 1=LV, 2=myocardium, 3=atrium}
        - Logits shape: [B, C, H, W]
        - Masks shape: [B, H, W] with dtype torch.long

    For binary (num_classes == 1):
        - Uses BCEWithLogitsLoss
        - Logits shape: [B, 1, H, W]
        - Masks shape: [B, H, W] or [B, 1, H, W], float32
    """
    if num_classes == 1:
        # Binary segmentation
        criterion = nn.BCEWithLogitsLoss()
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        return criterion(logits, masks.float())
    else:
        # Multiclass segmentation
        criterion = nn.CrossEntropyLoss(reduction="mean")
        # CrossEntropyLoss expects: logits [B, C, H, W], targets [B, H, W] with long dtype
        return criterion(logits, masks.long())
