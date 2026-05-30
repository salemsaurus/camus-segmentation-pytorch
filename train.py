"""
Faithful CAMUS U-Net1 training with true multiclass segmentation.

Multiclass structure (4 classes):
    - Class 0: Background
    - Class 1: LV cavity
    - Class 2: Myocardium
    - Class 3: Left atrium

Patient-level splitting is mandatory: never split by frame/view/phase.
Uses official CAMUS dataset splits (400 train, 50 val, 50 test).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import CamusPatientDataset
from losses import segmentation_loss
from metrics import compute_batch_metrics
from model import CamusUnet1
from utils import (
    list_patients,
    load_official_camus_splits,
    postprocess_prediction,
    save_dice_curves,
    save_json,
    save_loss_curves,
    save_overlay,
    seed_everything,
)


def run_epoch(
    model: CamusUnet1,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    optimizer=None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    postprocess: bool = False,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    metric_sums = {"dice": 0.0, "iou": 0.0, "hausdorff": 0.0, "msd": 0.0}
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = segmentation_loss(logits, masks, num_classes=num_classes)

        if train_mode:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            if postprocess and num_classes == 1:
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
            else:
                pred = logits

            batch_metrics = compute_batch_metrics(pred, masks, num_classes)
            total_loss += loss.item()
            for k in metric_sums:
                metric_sums[k] += batch_metrics[k]
            n_batches += 1

    out = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
    out["loss"] = total_loss / max(n_batches, 1)
    return out


def train_one_fold(
    fold: int,
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    fold_dir = args.output
    fold_dir.mkdir(parents=True, exist_ok=True)

    tr_patients, va_patients = train_patients, val_patients

    train_ds = CamusPatientDataset(
        tr_patients, args.nifti_root, image_size=(args.height, args.width), augment=True, seed=args.seed
    )
    val_ds = CamusPatientDataset(
        va_patients, args.nifti_root, image_size=(args.height, args.width), augment=False, seed=args.seed
    )
    test_ds = CamusPatientDataset(
        test_patients, args.nifti_root, image_size=(args.height, args.width), augment=False, seed=args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CamusUnet1(
        num_classes=args.num_classes,
        bilinear=True,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"patients train/val/test = {len(tr_patients)}/{len(va_patients)}/{len(test_patients)}")
    print(f"samples train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"parameters = {num_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val = float("inf")
    best_path = fold_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, device, args.num_classes, optimizer=optimizer, scaler=scaler, use_amp=use_amp
        )
        va = run_epoch(model, val_loader, device, args.num_classes, postprocess=args.postprocess_eval)

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_dice"].append(tr["dice"])
        history["val_dice"].append(va["dice"])

        scheduler.step(va["loss"])

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss={tr['loss']:.4f} dice={tr['dice']:.4f} | "
            f"val loss={va['loss']:.4f} dice={va['dice']:.4f}"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": "camus_unet1",
                    "num_classes": args.num_classes,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va["loss"],
                    "val_dice": va["dice"],
                    "image_size": (args.height, args.width),
                },
                best_path,
            )

    save_loss_curves(history, fold_dir / "loss_curve.png")
    save_dice_curves(history, fold_dir / "dice_curve.png")

    # Test fold evaluation with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = run_epoch(
        model, test_loader, device, args.num_classes, postprocess=args.postprocess_eval
    )

    # Save a few overlays from test set
    overlay_dir = fold_dir / "overlays"
    model.eval()
    saved = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            for i in range(images.size(0)):
                if saved >= args.max_overlays:
                    break
                img = images[i, 0].cpu().numpy()
                gt = masks[i].cpu().numpy()
                if args.num_classes == 1:
                    pred = (torch.sigmoid(logits[i : i + 1]) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                else:
                    pred = torch.argmax(logits[i : i + 1], dim=1).squeeze().cpu().numpy().astype(np.uint8)
                if args.postprocess_eval:
                    pred = postprocess_prediction(pred)
                save_overlay(img, gt.astype(np.uint8), pred, overlay_dir / f"sample_{saved:03d}.png")
                saved += 1
            if saved >= args.max_overlays:
                break

    save_json(fold_dir / "test_metrics.json", test_metrics)
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CAMUS U-Net1 faithful training using official 400/50/50 patient-level splits")
    parser.add_argument("--nifti-root", type=Path, default=Path("database_nifti"))
    parser.add_argument("--split-root", type=Path, default=Path("database_split"), help="Folder containing official CAMUS split files")
    parser.add_argument("--output", type=Path, default=Path("runs/training"))
    parser.add_argument("--num-classes", type=int, default=4, help="1=binary, >1=multi-class")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--postprocess-eval", action="store_true", default=True)
    parser.add_argument("--max-overlays", type=int, default=8)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    official_train, official_val, official_test = load_official_camus_splits(args.split_root)
    patients = list_patients(args.nifti_root)
    missing = set(official_train + official_val + official_test) - set(patients)
    if missing:
        raise RuntimeError(
            "The official split references patient directories missing from the NIfTI root: "
            f"{sorted(missing)}"
        )

    metrics = train_one_fold(
        fold=0,
        train_patients=official_train,
        val_patients=official_val,
        test_patients=official_test,
        args=args,
        device=device,
    )

    save_json(args.output / "test_metrics.json", metrics)
    print("Training complete. Results saved to:", args.output)


if __name__ == "__main__":
    main()
