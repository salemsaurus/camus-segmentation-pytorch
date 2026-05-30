"""Utilities: patient discovery, post-processing, plotting, checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_fill_holes, label

CAMUS_CLASS_ORDER = (1, 2, 3)
CAMUS_CLASS_COLOR = {
    1: (1.0, 0.0, 0.0),  # LV cavity
    2: (0.0, 1.0, 0.0),  # myocardium
    3: (0.0, 0.0, 1.0),  # left atrium
}


def list_patients(nifti_root: Path) -> List[str]:
    patients = sorted(p.name for p in nifti_root.glob("patient*") if p.is_dir())
    if not patients:
        raise RuntimeError(f"No patients found under {nifti_root}")
    return patients


def load_patient_list(path: Path) -> List[str]:
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
) -> tuple[List[str], List[str], List[str]]:
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
    """Keep largest connected component (paper post-processing)."""
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
    """Fill internal holes in binary mask."""
    return binary_fill_holes(mask)


def postprocess_prediction(
    pred_mask: np.ndarray,
    *,
    keep_largest: bool = True,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Safe minimal post-processing for CAMUS multiclass segmentation.

    This function preserves class exclusivity and avoids aggressive cross-class
    morphology. It processes the LV cavity first, then constrains myocardium to
    a neighborhood around the LV, and finally applies atrium processing without
    overwriting LV or myocardium.
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

    if keep_largest:
        lv_mask = largest_connected_component(lv_mask)
        atrium_mask = largest_connected_component(atrium_mask)

    if fill_holes:
        lv_mask = remove_holes(lv_mask)
        myocardium_mask = remove_holes(myocardium_mask)
        atrium_mask = remove_holes(atrium_mask)

    if lv_mask.any():
        lv_dilation = binary_dilation(lv_mask, structure=np.ones((3, 3)), iterations=1)
        myocardium_mask = myocardium_mask & lv_dilation
    else:
        myocardium_mask = myocardium_mask

    myocardium_mask = myocardium_mask & ~lv_mask
    atrium_mask = atrium_mask & ~(lv_mask | myocardium_mask)

    output_mask[lv_mask] = 1
    output_mask[myocardium_mask] = 2
    output_mask[atrium_mask] = 3

    return output_mask


def save_loss_curves(history: dict, out_path: Path) -> None:
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


def save_dice_curves(history: dict, out_path: Path) -> None:
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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def split_patients_validation(
    train_patients: Sequence[str], val_fraction: float, seed: int
) -> tuple[List[str], List[str]]:
    """Patient-level validation split inside a training fold."""
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
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # Older PyTorch versions may not expose this API.
        pass
