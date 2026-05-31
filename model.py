"""
Faithful CAMUS U-Net 1 (Leclerc et al., IEEE TMI 2019).

Architecture constraints:
- Conv2d -> ReLU only (no normalization layers)
- Bilinear upsampling + 1x1 channel projection in decoder (paper default for U-Net 1)
- Lightweight channel progression: 1 -> 32 -> 32 -> 64 -> 128 -> 128 -> 128
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv3x3 -> ReLU -> Conv3x3 -> ReLU (no normalization)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """Bilinear upsample x2 then 1x1 conv for channel reduction."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class CamusUnet1(nn.Module):
    """
    Original-style compact U-Net 1 used in the CAMUS challenge paper.

    Supports binary (1 channel + sigmoid) or multi-class (C channels + softmax).
    """

    def __init__(
        self,
        num_classes: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if not bilinear:
            raise ValueError("Faithful CAMUS U-Net1 reproduction requires bilinear upsampling.")
        self.num_classes = num_classes
        self.binary = num_classes == 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: 1 -> 32 -> 32 -> 64 -> 128 -> 128 -> 128
        self.enc1 = double_conv(1, 32)
        self.enc2 = double_conv(32, 32)
        self.enc3 = double_conv(32, 64)
        self.enc4 = double_conv(64, 128)
        self.enc5 = double_conv(128, 128)
        self.enc6 = double_conv(128, 128)

        # Decoder upsampling (bilinear + 1x1), matching original CAMUS U-Net1 wiring
        self.up1 = UpBlock(128, 128)  # bottleneck -> match enc5 resolution
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 32)
        self.up5 = UpBlock(32, 32)

        self.dec1 = double_conv(256, 128)  # 128 skip + 128 up
        self.dec2 = double_conv(192, 64)   # 128 skip + 64 up
        self.dec3 = double_conv(96, 32)    # 64 skip + 32 up
        self.dec4 = double_conv(64, 32)    # 32 skip + 32 up
        self.dec5 = double_conv(64, 16)    # 32 skip + 32 up -> 16 (paper-style head)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    @staticmethod
    def _match_spatial(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            return F.interpolate(
                upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        return upsampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))

        u1 = self._match_spatial(self.up1(e6), e5)
        d1 = self.dec1(torch.cat([e5, u1], dim=1))

        u2 = self._match_spatial(self.up2(d1), e4)
        d2 = self.dec2(torch.cat([e4, u2], dim=1))

        u3 = self._match_spatial(self.up3(d2), e3)
        d3 = self.dec3(torch.cat([e3, u3], dim=1))

        u4 = self._match_spatial(self.up4(d3), e2)
        d4 = self.dec4(torch.cat([e2, u4], dim=1))

        u5 = self._match_spatial(self.up5(d4), e1)
        d5 = self.dropout(self.dec5(torch.cat([e1, u5], dim=1)))

        logits = self.out(d5)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if self.binary:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)
