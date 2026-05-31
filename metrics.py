"""Segmentation and boundary metrics for CAMUS evaluation."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

CLASS_NAMES = {
    1: "lv",
    2: "myocardium",
    3: "atrium",
}


def _get_metric_class_names(num_classes: int) -> Dict[int, str]:
    """Return label names for metric computation based on mode."""
    if num_classes == 1:
        return {1: "foreground"}
    return {
        cls: CLASS_NAMES.get(cls, f"class_{cls}")
        for cls in range(1, num_classes)
    }


def _safe_mean(values: list[float]) -> float:
    """Return the mean of finite values or nan if there are none."""
    valid = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(valid)) if valid else float("nan")


def _get_class_masks(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert predictions and targets to label tensors for metric computation."""
    if num_classes == 1:
        if pred.ndim == 4:
            pred_labels = (torch.sigmoid(pred) > 0.5).squeeze(1).long()
        elif pred.ndim == 3:
            pred_labels = pred.long()
        else:
            raise ValueError("Binary predictions must be logits [B,1,H,W] or labels [B,H,W]")

        if target.ndim == 4:
            target = target.squeeze(1)
        target_labels = (target > 0).long()
    else:
        if pred.ndim == 4:
            pred_labels = torch.argmax(pred, dim=1).long()
        elif pred.ndim == 3:
            pred_labels = pred.long()
        else:
            raise ValueError("Multiclass predictions must be logits [B,C,H,W] or labels [B,H,W]")

        if target.ndim == 4:
            target = target.squeeze(1)
        target_labels = target.long()

    return pred_labels.cpu().numpy().astype(np.int32), target_labels.cpu().numpy().astype(np.int32)


def _surface_points(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty((0, 2), dtype=np.float32)
    eroded = binary_erosion(mask)
    surface = np.logical_and(mask, np.logical_not(eroded))
    ys, xs = np.where(surface)
    return np.stack([ys, xs], axis=1).astype(np.float32)


def _has_class(target: np.ndarray, cls: int) -> bool:
    return np.any(target == cls)


def compute_classwise_dice(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """Compute classwise Dice scores for CAMUS classes or binary foreground."""
    pred_m, tgt_m = _get_class_masks(pred, target, num_classes)
    results: Dict[str, float] = {}
    for cls, name in _get_metric_class_names(num_classes).items():
        scores: list[float] = []
        for p, t in zip(pred_m, tgt_m):
            t_cls = (t == cls)
            if not t_cls.any():
                continue
            p_cls = (p == cls)
            inter = np.logical_and(p_cls, t_cls).sum()
            denom = p_cls.sum() + t_cls.sum()
            scores.append((2.0 * inter + 1e-6) / (denom + 1e-6))
        results[name] = _safe_mean(scores)
    return results


def compute_classwise_iou(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """Compute classwise IoU scores for CAMUS classes or binary foreground."""
    pred_m, tgt_m = _get_class_masks(pred, target, num_classes)
    results: Dict[str, float] = {}
    for cls, name in _get_metric_class_names(num_classes).items():
        scores: list[float] = []
        for p, t in zip(pred_m, tgt_m):
            t_cls = (t == cls)
            if not t_cls.any():
                continue
            p_cls = (p == cls)
            inter = np.logical_and(p_cls, t_cls).sum()
            union = np.logical_or(p_cls, t_cls).sum()
            scores.append((inter + 1e-6) / (union + 1e-6))
        results[name] = _safe_mean(scores)
    return results


def _compute_pairwise_surface_distance(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    ps = _surface_points(pred_mask)
    ts = _surface_points(target_mask)
    if len(ps) == 0 or len(ts) == 0:
        return float("nan")
    dt_t = distance_transform_edt(~target_mask)
    dt_p = distance_transform_edt(~pred_mask)
    d1 = dt_t[ps[:, 0].astype(int), ps[:, 1].astype(int)]
    d2 = dt_p[ts[:, 0].astype(int), ts[:, 1].astype(int)]
    return float(max(d1.max(), d2.max()))


def _compute_pairwise_mean_surface_distance(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    ps = _surface_points(pred_mask)
    ts = _surface_points(target_mask)
    if len(ps) == 0 or len(ts) == 0:
        return float("nan")
    dt_t = distance_transform_edt(~target_mask)
    dt_p = distance_transform_edt(~pred_mask)
    d1 = dt_t[ps[:, 0].astype(int), ps[:, 1].astype(int)]
    d2 = dt_p[ts[:, 0].astype(int), ts[:, 1].astype(int)]
    return float(0.5 * (d1.mean() + d2.mean()))


def compute_classwise_hausdorff(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """Compute classwise Hausdorff distances for CAMUS classes or binary foreground."""
    pred_m, tgt_m = _get_class_masks(pred, target, num_classes)
    results: Dict[str, float] = {}
    for cls, name in _get_metric_class_names(num_classes).items():
        distances: list[float] = []
        for p, t in zip(pred_m, tgt_m):
            t_cls = (t == cls)
            if not t_cls.any():
                continue
            p_cls = (p == cls)
            distances.append(_compute_pairwise_surface_distance(p_cls, t_cls))
        results[name] = _safe_mean(distances)
    return results


def compute_classwise_msd(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """Compute classwise mean surface distances for CAMUS classes or binary foreground."""
    pred_m, tgt_m = _get_class_masks(pred, target, num_classes)
    results: Dict[str, float] = {}
    for cls, name in _get_metric_class_names(num_classes).items():
        distances: list[float] = []
        for p, t in zip(pred_m, tgt_m):
            t_cls = (t == cls)
            if not t_cls.any():
                continue
            p_cls = (p == cls)
            distances.append(_compute_pairwise_mean_surface_distance(p_cls, t_cls))
        results[name] = _safe_mean(distances)
    return results


def compute_confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> np.ndarray:
    """Compute a confusion matrix with ground truth rows and prediction columns."""
    if target.ndim == 4:
        target = target.squeeze(1)

    pred_labels, target_labels = _get_class_masks(pred, target, num_classes)
    if num_classes == 1:
        n_classes = 2
    else:
        n_classes = num_classes

    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t_flat, p_flat in zip(target_labels.ravel(), pred_labels.ravel()):
        if num_classes == 1:
            t_index = int(t_flat > 0)
            p_index = int(p_flat > 0)
        else:
            t_index = int(t_flat)
            p_index = int(p_flat)
        if 0 <= t_index < n_classes and 0 <= p_index < n_classes:
            matrix[t_index, p_index] += 1
    return matrix


def _add_classwise_to_metrics(
    metrics: Dict[str, float],
    prefix: str,
    class_metrics: Dict[str, float],
) -> None:
    for name, value in class_metrics.items():
        metrics[f"{prefix}_{name}"] = value


def compute_batch_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """Compute batch-level metrics including classwise and macro averages."""
    dice_by_class = compute_classwise_dice(pred, target, num_classes)
    iou_by_class = compute_classwise_iou(pred, target, num_classes)
    hausdorff_by_class = compute_classwise_hausdorff(pred, target, num_classes)
    msd_by_class = compute_classwise_msd(pred, target, num_classes)

    metrics: Dict[str, float] = {
        "dice": _safe_mean(list(dice_by_class.values())) if dice_by_class else float("nan"),
        "iou": _safe_mean(list(iou_by_class.values())) if iou_by_class else float("nan"),
        "hausdorff": _safe_mean(list(hausdorff_by_class.values())) if hausdorff_by_class else float("nan"),
        "msd": _safe_mean(list(msd_by_class.values())) if msd_by_class else float("nan"),
    }

    _add_classwise_to_metrics(metrics, "dice", dice_by_class)
    _add_classwise_to_metrics(metrics, "iou", iou_by_class)
    _add_classwise_to_metrics(metrics, "hausdorff", hausdorff_by_class)
    _add_classwise_to_metrics(metrics, "msd", msd_by_class)

    return metrics
