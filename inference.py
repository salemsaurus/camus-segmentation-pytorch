"""Run inference with a trained CAMUS U-Net1 checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import load_nifti
from model import CamusUnet1
from utils import postprocess_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="CAMUS U-Net1 inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--postprocess", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = int(ckpt.get("num_classes", 4))

    model = CamusUnet1(num_classes=num_classes, bilinear=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image = torch.from_numpy(load_nifti(args.image)).float().unsqueeze(0).unsqueeze(0)
    image = torch.nn.functional.interpolate(
        image, size=(args.height, args.width), mode="bilinear", align_corners=False
    )
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = image.to(device)

    with torch.no_grad():
        logits = model(image)
        if num_classes == 1:
            pred = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        else:
            pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    if args.postprocess:
        pred = postprocess_prediction(pred)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, pred)
    print(f"Saved prediction mask to {args.output}")


if __name__ == "__main__":
    main()
