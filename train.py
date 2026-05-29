import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from camus_unet.camus_unet1 import CamusUnet1

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError(
        "SimpleITK is required to read .nii.gz CAMUS files. Install it with: pip install SimpleITK"
    ) from exc


def read_split_file(split_path: Path) -> List[str]:
    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_nifti(path: Path) -> torch.Tensor:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)  # [D, H, W] or [H, W]
    tensor = torch.from_numpy(array).float()
    if tensor.ndim == 3:
        tensor = tensor[0]
    return tensor


class CamusSegmentationDataset(Dataset):
    def __init__(
        self,
        patients: List[str],
        nifti_root: Path,
        image_size: Tuple[int, int],
        phases: Tuple[str, ...] = ("ED", "ES"),
        views: Tuple[str, ...] = ("2CH", "4CH"),
    ) -> None:
        self.samples = []
        self.nifti_root = nifti_root
        self.image_size = image_size

        for patient in patients:
            patient_dir = nifti_root / patient
            for view in views:
                for phase in phases:
                    image_name = f"{patient}_{view}_{phase}.nii.gz"
                    mask_name = f"{patient}_{view}_{phase}_gt.nii.gz"
                    image_path = patient_dir / image_name
                    mask_path = patient_dir / mask_name
                    if image_path.exists() and mask_path.exists():
                        self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No training samples found under {nifti_root}. "
                "Check your split files and dataset paths."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]
        image = load_nifti(image_path)
        mask = load_nifti(mask_path).long()

        image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

        image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.image_size, mode="nearest")

        image = image.squeeze(0)  # [1, H, W]
        mask = mask.squeeze(0).squeeze(0).long()  # [H, W]

        # Per-sample normalization keeps intensity scale stable across patients.
        img_min = image.min()
        img_max = image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)

        return image, mask


def make_model() -> nn.Module:
    return CamusUnet1()


def multiclass_dice_score(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> float:
    pred = torch.argmax(logits, dim=1)
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        denom = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
        dice_scores.append(dice.mean().item())
    return sum(dice_scores) / len(dice_scores)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    return 1.0 - dice.mean()


def combined_ce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, ce_weight: float = 1.0, dice_weight: float = 1.0
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    d_loss = dice_loss(logits, target)
    return ce_weight * ce + dice_weight * d_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = combined_ce_dice_loss(logits, masks)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_dice += multiclass_dice_score(logits.detach(), masks)

    n_batches = len(loader)
    return total_loss / n_batches, total_dice / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train CAMUS U-Net segmentation model.")
    parser.add_argument("--nifti-root", type=Path, default=Path("database_nifti"))
    parser.add_argument("--split-root", type=Path, default=Path("database_split"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    train_patients = read_split_file(args.split_root / "subgroup_training.txt")
    val_patients = read_split_file(args.split_root / "subgroup_validation.txt")

    image_size = (args.height, args.width)
    train_ds = CamusSegmentationDataset(train_patients, args.nifti_root, image_size=image_size)
    val_ds = CamusSegmentationDataset(val_patients, args.nifti_root, image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    args.output.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt = args.output / "unet1_best.pt"

    print(f"Device: {device}")
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, device, optimizer=None)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": "unet1",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "image_size": image_size,
                },
                best_ckpt,
            )
            print(f"Saved new best checkpoint: {best_ckpt}")

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
