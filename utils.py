"""Utilities: patient discovery, post-processing, plotting, checkpoints, visualization."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_fill_holes, label

CAMUS_CLASS_ORDER = (1, 2, 3)
CAMUS_CLASS_COLOR = {
    1: (1.0, 0.0, 0.0),  # LV cavity
    2: (0.0, 1.0, 0.0),  # myocardium
    3: (0.0, 0.0, 1.0),  # left atrium
}

# Configurable post-processing: whether to apply largest_connected_component per class
# LV and LA benefit from largest component filtering (reduces noise)
# Myocardium is ring-shaped and may be damaged by aggressive component filtering
POSTPROCESS_CLASSES: Dict[int, bool] = {
    1: True,   # LV: apply largest_connected_component
    2: False,  # Myocardium: skip (ring-shaped structure)
    3: True,   # LA: apply largest_connected_component
}


def list_patients(nifti_root: Path) -> List[str]:
    """
    List all patient directories under CAMUS NIfTI root.
    
    Args:
        nifti_root: Path to database_nifti directory
    
    Returns:
        Sorted list of patient IDs (e.g., ['patient0001', 'patient0002', ...])
    
    Raises:
        RuntimeError: If no patient directories found
    """
    patients = sorted(p.name for p in nifti_root.glob("patient*") if p.is_dir())
    if not patients:
        raise RuntimeError(f"No patients found under {nifti_root}")
    return patients


def load_patient_list(path: Path) -> List[str]:
    """
    Load official CAMUS split file (one patient ID per line).
    
    Args:
        path: Path to split file (e.g., subgroup_training.txt)
    
    Returns:
        List of patient IDs
    
    Raises:
        FileNotFoundError: If split file not found
        RuntimeError: If split file is empty
    """
    if not path.exists():
        raise FileNotFoundError(f"Official CAMUS split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        patients = [line.strip() for line in f if line.strip()]
    if not patients:
        raise RuntimeError(f"Official CAMUS split file is empty: {path}")
    return patients


def load_official_camus_splits(
    split_root: Path,
    strict_counts: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load official CAMUS dataset splits (patient-level).
    
    Expected file structure:
        split_root/
            subgroup_training.txt      (400 patients)
            subgroup_validation.txt    (50 patients)
            subgroup_testing.txt       (50 patients)
    
    Args:
        split_root: Path to directory containing split files
        strict_counts: If True, enforce expected split sizes (400/50/50)
    
    Returns:
        Tuple of (train_patients, val_patients, test_patients)
    
    Raises:
        FileNotFoundError: If any split file not found
        ValueError: If splits overlap or (if strict_counts) counts don't match
    """
    train = load_patient_list(split_root / "subgroup_training.txt")
    validation = load_patient_list(split_root / "subgroup_validation.txt")
    test = load_patient_list(split_root / "subgroup_testing.txt")

    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise ValueError("Official CAMUS split files contain overlapping patients")

    if strict_counts:
        expected_counts = {"train": 400, "validation": 50, "test": 50}
        actual_counts = {"train": len(train), "validation": len(validation), "test": len(test)}
        if actual_counts != expected_counts:
            raise ValueError(
                "Official CAMUS split counts are invalid: "
                f"train={actual_counts['train']}, val={actual_counts['validation']}, "
                f"test={actual_counts['test']}"
            )

    return train, validation, test


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep largest connected component (paper post-processing).
    
    Args:
        mask: Binary 2D or 3D mask array.
    
    Returns:
        Mask with only the largest connected component.
    """
    if not mask.any():
        return mask
    labeled, n = label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return labeled == keep


def remove_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes in binary mask.
    
    Args:
        mask: Binary 2D or 3D mask array.
    
    Returns:
        Mask with holes filled.
    """
    return binary_fill_holes(mask)


def postprocess_prediction(
    pred_mask: np.ndarray,
    *,
    keep_largest: bool = True,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Safe minimal post-processing for CAMUS multiclass segmentation.

    This function preserves class exclusivity without artificial anatomical constraints.
    It removes the dilation constraint that was artificially limiting myocardium to
    the LV neighborhood. Instead, it:
    
    1. Applies per-class post-processing (LV and LA use largest_connected_component,
       myocardium skips it to preserve ring structure)
    2. Removes holes per class
    3. Enforces exclusivity: myocardium excludes LV, atrium excludes LV and myocardium
    
    Args:
        pred_mask: Shape (H, W) or (C, H, W) where C=4 (background + 3 classes)
        keep_largest: Whether to apply largest_connected_component per class
        fill_holes: Whether to fill holes per class
    
    Returns:
        Postprocessed multiclass mask with same shape as input.
        
    Note:
        The dilation constraint (myocardium_mask = myocardium_mask & lv_dilation)
        has been removed to avoid artificially constraining predictions and hiding
        model errors. Evaluation should reflect faithful model output.
    """
    if pred_mask.size == 0:
        return pred_mask

    if pred_mask.ndim == 3:
        return np.stack(
            [postprocess_prediction(mask, keep_largest=keep_largest, fill_holes=fill_holes) for mask in pred_mask],
            axis=0,
        )

    if pred_mask.ndim != 2:
        raise ValueError("postprocess_prediction expects a 2D or 3D mask array")

    output_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    pred_mask = pred_mask.astype(np.int32, copy=False)

    lv_mask = (pred_mask == 1)
    myocardium_mask = (pred_mask == 2)
    atrium_mask = (pred_mask == 3)

    # Apply configurable per-class post-processing
    if keep_largest:
        if POSTPROCESS_CLASSES.get(1, True):
            lv_mask = largest_connected_component(lv_mask)
        if POSTPROCESS_CLASSES.get(2, False):
            myocardium_mask = largest_connected_component(myocardium_mask)
        if POSTPROCESS_CLASSES.get(3, True):
            atrium_mask = largest_connected_component(atrium_mask)

    if fill_holes:
        lv_mask = remove_holes(lv_mask)
        myocardium_mask = remove_holes(myocardium_mask)
        atrium_mask = remove_holes(atrium_mask)

    # Enforce exclusivity without anatomical constraints
    # Myocardium must not overlap with LV
    myocardium_mask = myocardium_mask & ~lv_mask
    # Atrium must not overlap with LV or myocardium
    atrium_mask = atrium_mask & ~(lv_mask | myocardium_mask)

    output_mask[lv_mask] = 1
    output_mask[myocardium_mask] = 2
    output_mask[atrium_mask] = 3

    return output_mask


def save_loss_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    """Save training/validation loss curves.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' keys
        out_path: Path to save PNG figure
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_dice_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    """Save training/validation Dice curves.
    
    Args:
        history: Dictionary with 'train_dice' and 'val_dice' keys
        out_path: Path to save PNG figure
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_dice"], label="train_dice")
    plt.plot(history["val_dice"], label="val_dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    """
    Save segmentation overlay for CAMUS multiclass masks.

    Uses an explicit CAMUS class mapping and alpha blending so predictions
    remain visible without silently overwriting ground truth.
    
    Args:
        image: Grayscale 2D ultrasound image (H, W)
        gt: Ground truth multiclass mask (H, W) with classes {0, 1, 2, 3}
        pred: Predicted multiclass mask (H, W) with classes {0, 1, 2, 3}
        out_path: Path to save PNG figure
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    rgb = np.stack([img, img, img], axis=-1)
    alpha = 0.5

    gt = gt.astype(np.int32, copy=False)
    pred = pred.astype(np.int32, copy=False)

    for cls, color in CAMUS_CLASS_COLOR.items():
        gt_mask = gt == cls
        pred_mask = pred == cls

        if gt_mask.any():
            rgb[gt_mask] = np.array(color, dtype=np.float32)

        if pred_mask.any():
            rgb[pred_mask] = np.clip(
                rgb[pred_mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha,
                0.0,
                1.0,
            )

    plt.imsave(out_path, np.clip(rgb, 0, 1))


def save_contour_overlay(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save contour overlay visualization for CAMUS multiclass masks.
    
    Ground truth contours are shown with solid lines.
    Prediction contours are shown with dashed lines.
    
    Args:
        image: Grayscale 2D ultrasound image (H, W)
        gt: Ground truth multiclass mask (H, W) with classes {0, 1, 2, 3}
        pred: Predicted multiclass mask (H, W) with classes {0, 1, 2, 3}
        out_path: Path to save PNG figure
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    # Normalize image to [0, 1]
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    ax.imshow(img, cmap='gray')
    
    gt = gt.astype(np.int32, copy=False)
    pred = pred.astype(np.int32, copy=False)
    
    # Plot ground truth contours (solid lines)
    for cls in CAMUS_CLASS_ORDER:
        gt_mask = gt == cls
        color = CAMUS_CLASS_COLOR[cls]
        if gt_mask.any():
            ax.contour(gt_mask, levels=[0.5], colors=[color], linewidths=2, linestyles='solid')
    
    # Plot prediction contours (dashed lines)
    for cls in CAMUS_CLASS_ORDER:
        pred_mask = pred == cls
        color = CAMUS_CLASS_COLOR[cls]
        if pred_mask.any():
            ax.contour(pred_mask, levels=[0.5], colors=[color], linewidths=2, linestyles='dashed')
    
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def save_comparison_figure(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save three-panel comparison figure: ultrasound | GT segmentation | prediction.
    
    Args:
        image: Grayscale 2D ultrasound image (H, W)
        gt: Ground truth multiclass mask (H, W) with classes {0, 1, 2, 3}
        pred: Predicted multiclass mask (H, W) with classes {0, 1, 2, 3}
        out_path: Path to save PNG figure
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normalize image
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Create colored segmentation overlays
    def create_segmentation_rgb(mask: np.ndarray) -> np.ndarray:
        """Convert multiclass mask to RGB using CAMUS colors."""
        h, w = mask.shape
        rgb = np.ones((h, w, 3), dtype=np.float32)
        mask = mask.astype(np.int32, copy=False)
        
        for cls, color in CAMUS_CLASS_COLOR.items():
            mask_cls = mask == cls
            for ch in range(3):
                rgb[mask_cls, ch] = color[ch]
        
        return rgb
    
    # Panel 1: Ultrasound image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ultrasound Image')
    axes[0].axis('off')
    
    # Panel 2: Ground truth
    gt_rgb = create_segmentation_rgb(gt)
    axes[1].imshow(gt_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Panel 3: Prediction
    pred_rgb = create_segmentation_rgb(pred)
    axes[2].imshow(pred_rgb)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=CAMUS_CLASS_COLOR[1], label='LV Cavity'),
        mpatches.Patch(color=CAMUS_CLASS_COLOR[2], label='Myocardium'),
        mpatches.Patch(color=CAMUS_CLASS_COLOR[3], label='Left Atrium'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def save_error_map(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save error map highlighting false positives and false negatives.
    
    False negatives (FN) shown in red: GT has class but pred doesn't
    False positives (FP) shown in blue: pred has class but GT doesn't
    Correct predictions shown in green: GT and pred agree
    
    Args:
        image: Grayscale 2D ultrasound image (H, W)
        gt: Ground truth multiclass mask (H, W) with classes {0, 1, 2, 3}
        pred: Predicted multiclass mask (H, W) with classes {0, 1, 2, 3}
        out_path: Path to save PNG figure
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w = image.shape
    error_map = np.ones((h, w, 3), dtype=np.float32)
    
    # Normalize image to grayscale background
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    for ch in range(3):
        error_map[:, :, ch] = img * 0.5 + 0.5  # dim background
    
    gt = gt.astype(np.int32, copy=False)
    pred = pred.astype(np.int32, copy=False)
    
    disagreement = (gt != pred)
    
    # False negatives: GT has class, pred doesn't (or differs)
    fn_mask = (gt > 0) & (pred != gt)
    error_map[fn_mask] = [1.0, 0.0, 0.0]  # Red
    
    # False positives: Pred has class, GT doesn't
    fp_mask = (gt == 0) & (pred > 0)
    error_map[fp_mask] = [0.0, 0.0, 1.0]  # Blue
    
    # Correct: GT and pred agree
    correct_mask = (gt == pred)
    error_map[correct_mask] = [0.0, 1.0, 0.0]  # Green
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.clip(error_map, 0, 1))
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=[1.0, 0.0, 0.0], label='False Negative (GT only)'),
        mpatches.Patch(color=[0.0, 0.0, 1.0], label='False Positive (Pred only)'),
        mpatches.Patch(color=[0.0, 1.0, 0.0], label='Correct'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')
    ax.set_axis_off()
    
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def save_batch_visualizations(
    images: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save comprehensive visualizations for a batch of samples.
    
    For each sample (i), saves:
    - overlay.png: Direct mask overlay
    - contour_overlay.png: Contour comparison (GT solid, Pred dashed)
    - comparison.png: Three-panel figure
    - error_map.png: FN/FP/correct highlighting
    
    Args:
        images: Batch of ultrasound images (N, H, W)
        gt_masks: Batch of GT multiclass masks (N, H, W)
        pred_masks: Batch of pred multiclass masks (N, H, W)
        output_dir: Directory to save visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = images.shape[0]
    for i in range(n_samples):
        sample_dir = output_dir / f"sample_{i:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        image = images[i]
        gt = gt_masks[i]
        pred = pred_masks[i]
        
        save_overlay(image, gt, pred, sample_dir / "overlay.png")
        save_contour_overlay(image, gt, pred, sample_dir / "contour_overlay.png")
        save_comparison_figure(image, gt, pred, sample_dir / "comparison.png")
        save_error_map(image, gt, pred, sample_dir / "error_map.png")


def save_json(path: Path, payload: Dict) -> None:
    """Save dictionary to JSON file.
    
    Args:
        path: Path to save JSON file
        payload: Dictionary to serialize
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_training_history(history: Dict[str, List[float]], out_json: Path) -> None:
    """
    Save training metrics history to JSON for later analysis.
    
    Stores train/val metrics across epochs without requiring re-training.
    Useful for offline analysis, plotting, and model selection.
    
    Args:
        history: Dictionary with metric lists, e.g.:
            {
                'train_loss': [0.5, 0.4, ...],
                'val_loss': [0.6, 0.5, ...],
                'train_dice': [0.7, 0.75, ...],
                'val_dice': [0.65, 0.7, ...],
            }
        out_json: Path to save JSON file
    """
    save_json(out_json, history)


def split_patients_validation(
    train_patients: Sequence[str], val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    """Patient-level validation split inside a training fold.
    
    Args:
        train_patients: List of training patient IDs
        val_fraction: Fraction of patients to use for validation (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_patients, val_patients)
    """
    if not train_patients:
        raise ValueError("train_patients must contain at least one patient")

    rng = np.random.default_rng(seed)
    patients = list(train_patients)
    rng.shuffle(patients)

    n_val = max(1, int(round(len(patients) * val_fraction)))
    n_val = min(n_val, len(patients) - 1)

    val_patients = patients[:n_val]
    tr_patients = patients[n_val:]
    return tr_patients, val_patients


def seed_everything(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Synchronizes random number generators across:
    - Python's random module
    - NumPy
    - PyTorch (CPU and GPU)
    - CuDNN
    
    Also enforces deterministic algorithms and disables benchmarking
    which may introduce randomness in CUDA kernels.
    
    Args:
        seed: Integer seed value (0-2^32)
    
    Note:
        Some operations may still be non-deterministic depending on
        PyTorch version and GPU capabilities. Set PYTHONHASHSEED before
        script execution for full reproducibility.
    """
    # Set PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Standard library
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms and disable benchmarking
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.12+
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # Older PyTorch versions may not expose this API.
        pass
