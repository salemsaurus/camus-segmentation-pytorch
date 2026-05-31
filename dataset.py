"""
CAMUS dataset loader with strict patient-level sample grouping.

Why patient-level splitting matters
---------------------------------
Each CAMUS patient has correlated views/phases (2CH/4CH, ED/ES) from the same
cardiac cycle and acquisition. Each patient contributes four related frames:
2CH_ED, 2CH_ES, 4CH_ED, 4CH_ES. Splitting at the frame level would put
near-duplicate anatomy and the same acoustic appearance into different splits,
causing data leakage and inflating evaluation metrics.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("Install SimpleITK: pip install SimpleITK") from exc


VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")


def load_nifti(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    array = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    return array


class CamusPatientDataset(Dataset):
    """
    Multiclass CAMUS dataset loader with patient-level sample grouping.

    CAMUS multiclass structure:
        - Class 0: Background (no cardiac structure)
        - Class 1: LV cavity (left ventricle blood pool)
        - Class 2: Myocardium (LV wall)
        - Class 3: Left atrium (LA)

    Patient-level splitting is mandatory to prevent data leakage:
        - Each patient has 4 related frames: 2CH_ED, 2CH_ES, 4CH_ED, 4CH_ES
        - All frames from one patient stay in the same train/val/test split
        - Never split by frame/view/phase

    Each item is one frame + multiclass mask pair from one patient.
    """

    def __init__(
        self,
        patients: Sequence[str],
        nifti_root: Path,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        seed: int = 42,
        num_classes: int = 4,
    ) -> None:
        self.patients = list(patients)
        self.nifti_root = nifti_root
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)
        self.num_classes = num_classes
        self.samples: List[Tuple[Path, Path]] = []

        for patient in self.patients:
            patient_dir = nifti_root / patient
            for view in VIEWS:
                for phase in PHASES:
                    image_path = patient_dir / f"{patient}_{view}_{phase}.nii.gz"
                    mask_path = patient_dir / f"{patient}_{view}_{phase}_gt.nii.gz"
                    if image_path.exists() and mask_path.exists():
                        self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No samples for {len(self.patients)} patients under {nifti_root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        mn, mx = image.min(), image.max()
        return (image - mn) / (mx - mn + 1e-8)

    def _rotate(self, tensor: torch.Tensor, angle_deg: float, mode: str) -> torch.Tensor:
        angle = torch.tensor(angle_deg * np.pi / 180.0, dtype=tensor.dtype, device=tensor.device)
        c, s = torch.cos(angle), torch.sin(angle)
        theta = torch.stack(
            [
                torch.stack([c, -s, torch.zeros((), dtype=tensor.dtype, device=tensor.device)]),
                torch.stack([s, c, torch.zeros((), dtype=tensor.dtype, device=tensor.device)]),
            ]
        ).unsqueeze(0)
        grid = F.affine_grid(
            theta,
            tensor.unsqueeze(0).shape,
            align_corners=False,
        )
        return F.grid_sample(
            tensor.unsqueeze(0),
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

    def _scale(self, tensor: torch.Tensor, scale: float, mode: str) -> torch.Tensor:
        h, w = tensor.shape[-2:]
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

        kwargs = {"mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        scaled = F.interpolate(tensor.unsqueeze(0), size=(nh, nw), **kwargs).squeeze(0)

        if nh > h:
            top = (nh - h) // 2
            scaled = scaled[..., top:top + h, :]
        elif nh < h:
            pad_top = (h - nh) // 2
            pad_bottom = h - nh - pad_top
            scaled = F.pad(scaled, (0, 0, pad_top, pad_bottom))

        if nw > w:
            left = (nw - w) // 2
            scaled = scaled[..., :, left:left + w]
        elif nw < w:
            pad_left = (w - nw) // 2
            pad_right = w - nw - pad_left
            scaled = F.pad(scaled, (pad_left, pad_right, 0, 0))

        return scaled

    def _augment(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Realistic ultrasound augmentations only.
        if self.rng.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        if self.rng.random() < 0.3:
            angle = self.rng.uniform(-10, 10)
            image = self._rotate(image, angle, mode="bilinear")
            mask = self._rotate(mask.unsqueeze(0).float(), angle, mode="nearest").squeeze(0).long()

        if self.rng.random() < 0.3:
            scale = self.rng.uniform(0.9, 1.1)
            image = self._scale(image, scale, mode="bilinear")
            mask = self._scale(mask.unsqueeze(0).float(), scale, mode="nearest").squeeze(0).long()

        if self.rng.random() < 0.5:
            factor = self.rng.uniform(0.85, 1.15)
            bias = self.rng.uniform(-0.05, 0.05)
            image = torch.clamp(image * factor + bias, 0.0, 1.0)

        if self.rng.random() < 0.3:
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image, mask

    def _prepare_mask_labels(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.long()
        if self.num_classes == 1:
            return (mask > 0).long()
        if mask.numel() and (mask.min() < 0 or mask.max() >= self.num_classes):
            raise ValueError(
                f"Mask labels must be in [0, {self.num_classes - 1}], "
                f"got min={int(mask.min())}, max={int(mask.max())}"
            )
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = torch.from_numpy(load_nifti(image_path)).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(load_nifti(mask_path)).long().unsqueeze(0).unsqueeze(0).float()

        image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.image_size, mode="nearest").squeeze(0).squeeze(0).long()
        mask = self._prepare_mask_labels(mask)

        image = self._normalize(image.squeeze(0))
        if self.augment:
            image, mask = self._augment(image, mask)
            mask = self._prepare_mask_labels(mask)

        return image, mask
