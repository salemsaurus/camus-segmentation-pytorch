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
    ) -> None:
        self.patients = list(patients)
        self.nifti_root = nifti_root
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)
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

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Realistic ultrasound augmentations only.
        if self.rng.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        if self.rng.random() < 0.3:
            angle = self.rng.uniform(-10, 10)
            image = self._rotate(image, angle)
            mask = self._rotate(mask.unsqueeze(0).float(), angle).squeeze(0).long()

        if self.rng.random() < 0.3:
            scale = self.rng.uniform(0.9, 1.1)
            image = self._scale(image, scale)
            mask = self._scale(mask.unsqueeze(0).float(), scale).squeeze(0).long()

        if self.rng.random() < 0.5:
            factor = self.rng.uniform(0.85, 1.15)
            bias = self.rng.uniform(-0.05, 0.05)
            image = torch.clamp(image * factor + bias, 0.0, 1.0)

        if self.rng.random() < 0.3:
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image, mask
    def _rotate(self, tensor: torch.Tensor, angle_deg: float) -> torch.Tensor:
        angle = torch.tensor(angle_deg * np.pi / 180.0)
        c, s = torch.cos(angle), torch.sin(angle)
        grid = F.affine_grid(
            torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=tensor.dtype).unsqueeze(0),
            tensor.unsqueeze(0).shape,
            align_corners=False,
        )
        return F.grid_sample(
            tensor.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(0)

    def _scale(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        h, w = tensor.shape[-2:]
        nh, nw = int(h * scale), int(w * scale)
        scaled = F.interpolate(tensor.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)
        return F.interpolate(scaled, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = torch.from_numpy(load_nifti(image_path)).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(load_nifti(mask_path)).long().unsqueeze(0).unsqueeze(0).float()

        image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.image_size, mode="nearest").squeeze(0).squeeze(0).long()

        image = self._normalize(image.squeeze(0))
        if self.augment:
            image, mask = self._augment(image, mask)

        return image, mask
