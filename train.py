"""
Faithful CAMUS U-Net1 training with true multiclass segmentation.

Multiclass structure (4 classes):
    - Class 0: Background
    - Class 1: LV cavity
    - Class 2: Myocardium
    - Class 3: Left atrium

Patient-level splitting is mandatory: never split by frame/view/phase.
Uses official CAMUS dataset splits (400 train, 50 val, 50 test).

Default hyperparameters are set to the original U-Net1 paper style:
    - batch size = 32
    - learning rate = 1e-3
    - weight decay = 0.0
    - dropout = 0.0
    - no normalization
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

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


def seed_worker(worker_id: int) -> None:
    import random
    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def prepare_predictions(
    logits: torch.Tensor,
    num_classes: int,
    postprocess_eval: bool,
) -> torch.Tensor:
    """Convert model output to class labels and optionally apply post-processing."""
    if num_classes == 1:
        preds = (torch.sigmoid(logits) > 0.5).squeeze(1).long()
    else:
        preds = torch.argmax(logits, dim=1).long()

    if not postprocess_eval:
        return preds

    predictions = preds.cpu().numpy()
    processed = [postprocess_prediction(mask) for mask in predictions]
    return torch.from_numpy(np.stack(processed, axis=0)).to(logits.device).long()


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
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    # Aggregation containers: support arbitrary metric keys returned by compute_batch_metrics
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    total_samples = 0

    for images, masks in loader:
        batch_size = images.size(0)
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
                if max_grad_norm is not None and max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        with torch.no_grad():
            pred = prepare_predictions(logits, num_classes, postprocess_eval=postprocess)
            batch_metrics = compute_batch_metrics(pred, masks, num_classes)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        for k, value in batch_metrics.items():
            # Initialize containers if first time seeing this metric key
            if k not in metric_sums:
                metric_sums[k] = 0.0
                metric_counts[k] = 0
            # Only aggregate finite values; keep a per-metric valid-sample count
            try:
                v = float(value)
            except Exception:
                continue
            if np.isfinite(v):
                metric_sums[k] += v * batch_size
                metric_counts[k] += batch_size
            # If value is not finite (nan/inf), skip adding but ensure key exists

    if total_samples == 0:
        raise RuntimeError("No samples were processed during epoch")

    # Compute averages per metric using their valid-sample counts
    out: Dict[str, float] = {}
    for k, s in metric_sums.items():
        cnt = metric_counts.get(k, 0)
        out[k] = float(s / cnt) if cnt > 0 else float("nan")
    out["loss"] = total_loss / total_samples
    return out


def train_one_fold(
    fold: int,
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    fold_dir = args.output / f"fold_{fold}"
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

    def make_generator(seed: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed + 1),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed + 2),
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
    best_val_dice = float("-inf")
    best_val_loss = float("inf")
    best_path = fold_dir / "best_model.pt"

    # Track training time
    epoch_times = []
    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        tr = run_epoch(
            model, train_loader, device, args.num_classes, optimizer=optimizer, scaler=scaler, use_amp=use_amp
        )
        va = run_epoch(model, val_loader, device, args.num_classes, postprocess=args.postprocess_eval)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_dice"].append(tr["dice"])
        history["val_dice"].append(va["dice"])

        scheduler.step(va["loss"])

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss={tr['loss']:.4f} dice={tr['dice']:.4f} | "
            f"val loss={va['loss']:.4f} dice={va['dice']:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        is_new_best = (
            va["dice"] > best_val_dice
            or (va["dice"] == best_val_dice and va["loss"] < best_val_loss)
        )
        if is_new_best:
            best_val_dice = va["dice"]
            best_val_loss = va["loss"]
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

    training_total_time = time.time() - training_start_time

    save_loss_curves(history, fold_dir / "loss_curve.png")
    save_dice_curves(history, fold_dir / "dice_curve.png")

    # Test fold evaluation with best checkpoint
    test_start_time = time.time()
    
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        args.num_classes,
        postprocess=args.postprocess_eval,
    )
    
    test_time = time.time() - test_start_time

    # Save a few overlays from test set
    overlay_dir = fold_dir / "overlays"
    model.eval()
    saved = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = prepare_predictions(logits, args.num_classes, postprocess_eval=args.postprocess_eval)
            for i in range(images.size(0)):
                if saved >= args.max_overlays:
                    break
                img = images[i, 0].cpu().numpy()
                gt = masks[i].cpu().numpy()
                pred = preds[i].cpu().numpy().astype(np.uint8)
                save_overlay(img, gt.astype(np.uint8), pred, overlay_dir / f"sample_{saved:03d}.png")
                saved += 1
            if saved >= args.max_overlays:
                break

    save_json(fold_dir / "test_metrics.json", test_metrics)
    
    # Compute and save timing statistics
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0.0
    min_epoch_time = np.min(epoch_times) if epoch_times else 0.0
    max_epoch_time = np.max(epoch_times) if epoch_times else 0.0
    
    timing_info = {
        "total_training_time_seconds": training_total_time,
        "total_training_time_minutes": training_total_time / 60.0,
        "total_training_time_hours": training_total_time / 3600.0,
        "average_epoch_time_seconds": avg_epoch_time,
        "min_epoch_time_seconds": min_epoch_time,
        "max_epoch_time_seconds": max_epoch_time,
        "test_evaluation_time_seconds": test_time,
        "num_epochs": args.epochs,
    }
    
    save_json(fold_dir / "training_time.json", timing_info)
    
    # Print timing summary
    print("\n" + "=" * 70)
    print("TRAINING TIME SUMMARY")
    print("=" * 70)
    print(f"Total training time:  {training_total_time:.1f}s ({training_total_time/60:.1f}m / {training_total_time/3600:.2f}h)")
    print(f"Average epoch time:   {avg_epoch_time:.2f}s")
    print(f"Min epoch time:       {min_epoch_time:.2f}s")
    print(f"Max epoch time:       {max_epoch_time:.2f}s")
    print(f"Test evaluation time: {test_time:.2f}s")
    print(f"Total time (incl. test): {training_total_time + test_time:.1f}s ({(training_total_time + test_time)/60:.1f}m)")
    print("=" * 70 + "\n")
    
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CAMUS U-Net1 faithful training using official 400/50/50 patient-level splits")
    parser.add_argument("--nifti-root", type=Path, default=Path("database_nifti"))
    parser.add_argument("--split-root", type=Path, default=Path("database_split"), help="Folder containing official CAMUS split files")
    parser.add_argument("--output", type=Path, default=Path("runs/training"))
    parser.add_argument("--num-classes", type=int, default=4, help="1=binary, >1=multi-class")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (paper default = 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (paper default = 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay regularization (default = 0.0 for paper-style training)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers (default = 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max norm for gradient clipping. Set 0 to disable.")
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

    print("Training complete. Results saved to:", args.output)


if __name__ == "__main__":
    main()
