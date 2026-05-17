import argparse
import os
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from camus_unet.camus_unet2 import CamusUnet2
from camus_unet.camus_unet1 import CamusUnet1


class SegmentationDataset(Dataset):
    """Simple dataset that expects folders with images and masks.

    Structure:
      dataset_root/
        images/*.png
        masks/*.png    (integer class labels 0..C-1)
    """

    def __init__(self, root, image_size=(256, 256)):
        self.images = sorted(glob(os.path.join(root, "images", "*.png")))
        self.masks = sorted(glob(os.path.join(root, "masks", "*.png")))
        assert len(self.images) == len(self.masks), "images and masks count mismatch"
        self.image_size = image_size
        self.img_tf = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        m = Image.open(self.masks[idx])
        img = self.img_tf(img)
        m = m.resize(self.image_size, resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(m)).long()
        return img, mask


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        opt.zero_grad()
        out = model(imgs)
        loss = loss_fn(out, masks)
        loss.backward()
        opt.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)
            loss = loss_fn(out, masks)
            running += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == masks).sum().item()
            total += masks.numel()
    return running / len(loader.dataset), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to dataset root (train/ val subfolders)')
    parser.add_argument('--model', choices=['unet1', 'unet2'], default='unet2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', default='checkpoint.pth')
    parser.add_argument('--img-size', type=int, nargs=2, default=(256, 256))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = SegmentationDataset(os.path.join(args.data, 'train'), image_size=tuple(args.img_size))
    val_ds = SegmentationDataset(os.path.join(args.data, 'val'), image_size=tuple(args.img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    if args.model == 'unet2':
        model = CamusUnet2()
    else:
        model = CamusUnet1()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, args.save)


if __name__ == '__main__':
    main()
