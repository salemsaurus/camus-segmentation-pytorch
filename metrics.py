"""Segmentation and boundary metrics for CAMUS evaluation."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _get_class_masks(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract class-specific binary masks.

    For multiclass:
        pred: [B, C, H, W] logits or [B, H, W] class labels
        target: [B, H, W] or [B, 1, H, W] class indices
    For binary:
        pred: [B, 1, H, W] logits or [B, H, W] binary labels
        target: [B, H, W] or [B, 1, H, W]
    """
    if num_classes == 1:
        if pred.ndim == 4:
            pred_m = (torch.sigmoid(pred) > 0.5).squeeze(1)
        elif pred.ndim == 3:
            pred_m = (pred > 0).long()
        else:
            raise ValueError("Binary predictions must be either logits [B,1,H,W] or labels [B,H,W]")

        if target.ndim == 4:
            target = target.squeeze(1)
        tgt_m = (target > 0).long().cpu().numpy().astype(np.int32)
        pred_m = pred_m.cpu().numpy().astype(np.int32)
    else:
        if pred.ndim == 4:
            pred_m = torch.argmax(pred, dim=1)
        elif pred.ndim == 3:
            pred_m = pred.long()
        else:
            raise ValueError("Multiclass predictions must be either logits [B,C,H,W] or labels [B,H,W]")
        pred_m = pred_m.cpu().numpy().astype(np.int32)
        tgt_m = target.cpu().numpy().astype(np.int32)
    return pred_m, tgt_m


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    Compute Dice score. For multiclass, compute per-class then average.

    CAMUS multiclass:
        - Class 1: LV cavity
        - Class 2: myocardium
        - Class 3: left atrium
        Returns: mean Dice over classes 1-3
    """
    pred_m, tgt_m = _get_class_masks(pred, target, num_classes)

    if num_classes == 1:
        # Binary case
        scores = []
        for p, t in zip(pred_m, tgt_m):
            inter = np.logical_and(p, t).sum()
            denom = p.sum() + t.sum()
            scores.append((2.0 * inter + 1e-6) / (denom + 1e-6))
        return float(np.mean(scores))
    else:
        # Multiclass: compute per-class Dice
        class_scores = {cls: [] for cls in range(1, num_classes)}
        for cls in range(1, num_classes):
            for p, t in zip(pred_m, tgt_m):
                p_cls = (p == cls).astype(np.int32)
                t_cls = (t == cls).astype(np.int32)
                inter = np.logical_and(p_cls, t_cls).sum()
                denom = p_cls.sum() + t_cls.sum()
                class_scores[cls].append((2.0 * inter + 1e-6) / (denom + 1e-6))

        # Return mean across all samples and classes
        all_scores = []
        for cls_list in class_scores.values():
            all_scores.extend(cls_list)
        return float(np.mean(all_scores)) if all_scores else 0.0


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    Compute IoU (Jaccard) score. For multiclass, compute per-class then average.
    """
    pred_m, tgt_m = _get_class_masks(pred, target, num_classes)

    if num_classes == 1:
        # Binary case
        scores = []
        for p, t in zip(pred_m, tgt_m):
            inter = np.logical_and(p, t).sum()
            union = np.logical_or(p, t).sum()
            scores.append((inter + 1e-6) / (union + 1e-6))
        return float(np.mean(scores))
    else:
        # Multiclass: compute per-class IoU
        class_scores = {cls: [] for cls in range(1, num_classes)}
        for cls in range(1, num_classes):
            for p, t in zip(pred_m, tgt_m):
                p_cls = (p == cls).astype(np.int32)
                t_cls = (t == cls).astype(np.int32)
                inter = np.logical_and(p_cls, t_cls).sum()
                union = np.logical_or(p_cls, t_cls).sum()
                class_scores[cls].append((inter + 1e-6) / (union + 1e-6))

        # Return mean across all samples and classes
        all_scores = []
        for cls_list in class_scores.values():
            all_scores.extend(cls_list)
        return float(np.mean(all_scores)) if all_scores else 0.0


def _surface_points(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty((0, 2), dtype=np.float32)
    eroded = binary_erosion(mask)
    surface = np.logical_and(mask, np.logical_not(eroded))
    ys, xs = np.where(surface)
    return np.stack([ys, xs], axis=1).astype(np.float32)


def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    Compute Hausdorff distance. For multiclass, compute per-class then average.
    """
    pred_m, tgt_m = _get_class_masks(pred, target, num_classes)

    if num_classes == 1:
        # Binary case: compute on foreground
        values = []
        for p, t in zip(pred_m, tgt_m):
            ps, ts = _surface_points(p), _surface_points(t)
            if len(ps) == 0 or len(ts) == 0:
                values.append(np.nan)
                continue
            dt_t = distance_transform_edt(~t)
            dt_p = distance_transform_edt(~p)
            d1 = dt_t[ps[:, 0].astype(int), ps[:, 1].astype(int)]
            d2 = dt_p[ts[:, 0].astype(int), ts[:, 1].astype(int)]
            values.append(max(d1.max(), d2.max()))
        return float(np.nanmean(values))
    else:
        # Multiclass: compute per-class Hausdorff distance
        class_values = {cls: [] for cls in range(1, num_classes)}
        for cls in range(1, num_classes):
            for p, t in zip(pred_m, tgt_m):
                p_cls = (p == cls).astype(np.int32)
                t_cls = (t == cls).astype(np.int32)
                ps, ts = _surface_points(p_cls), _surface_points(t_cls)
                if len(ps) == 0 or len(ts) == 0:
                    class_values[cls].append(np.nan)
                    continue
                dt_t = distance_transform_edt(~t_cls)
                dt_p = distance_transform_edt(~p_cls)
                d1 = dt_t[ps[:, 0].astype(int), ps[:, 1].astype(int)]
                d2 = dt_p[ts[:, 0].astype(int), ts[:, 1].astype(int)]
                class_values[cls].append(max(d1.max(), d2.max()))

        all_values = []
        for cls_list in class_values.values():
            all_values.extend(cls_list)
        return float(np.nanmean(all_values)) if all_values else 0.0


def mean_surface_distance(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    Compute mean surface distance. For multiclass, compute per-class then average.
    """
    pred_m, tgt_m = _get_class_masks(pred, target, num_classes)

    if num_classes == 1:
        # Binary case
        values = []
        for p, t in zip(pred_m, tgt_m):
            ps, ts = _surface_points(p), _surface_points(t)
            if len(ps) == 0 or len(ts) == 0:
                values.append(np.nan)
                continue
            dt_t = distance_transform_edt(~t)
            dt_p = distance_transform_edt(~p)
            d1 = dt_t[ps[:, 0].astype(int), ps[:, 1].astype(int)]
            d2 = dt_p[ts[:, 0].astype(int), ts[:, 1].astype(int)]
            values.append(0.5 * (d1.mean() + d2.mean()))
        return float(np.nanmean(values))
    else:
        # Multiclass: compute per-class MSD
        class_values = {cls: [] for cls in range(1, num_classes)}
        for cls in range(1, num_classes):
            for p, t in zip(pred_m, tgt_m):
                p_cls = (p == cls).astype(np.int32)
                t_cls = (t == cls).astype(np.int32)
                ps, ts = _surface_points(p_cls), _surface_points(t_cls)
                if len(ps) == 0 or len(ts) == 0:
                    class_values[cls].append(np.nan)
                    continue
                dt_t = distance_transform_edt(~t_cls)
                dt_p = distance_transform_edt(~p_cls)
                d1 = dt_t[ps[:, 0].astype(int), ps[:, 1].astype(int)]
                d2 = dt_p[ts[:, 0].astype(int), ts[:, 1].astype(int)]
                class_values[cls].append(0.5 * (d1.mean() + d2.mean()))

        all_values = []
        for cls_list in class_values.values():
            all_values.extend(cls_list)
        return float(np.nanmean(all_values)) if all_values else 0.0



def compute_batch_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    return {
        "dice": dice_score(pred, target, num_classes),
        "iou": iou_score(pred, target, num_classes),
        "hausdorff": hausdorff_distance(pred, target, num_classes),
        "msd": mean_surface_distance(pred, target, num_classes),
    }
