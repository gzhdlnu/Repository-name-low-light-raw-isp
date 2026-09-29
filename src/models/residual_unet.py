# =============================================================================
# Project: RAW ISP | Task: Lightweight Residual RAW U-Net
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase4-Model-v0.1.0
# =============================================================================
"""A compact U-Net for four-channel packed Bayer RAW restoration."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.norm2(self.conv2(self.conv1(inputs)))
        return self.activation(inputs + residual)


class DecoderStage(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.fusion = ConvNormAct(input_channels + skip_channels, skip_channels)
        self.refinement = ResidualBlock(skip_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refinement(self.fusion(torch.cat([upsampled, skip], dim=1)))


class RawDenoisingUNet(nn.Module):
    """Predict a residual correction for normalized R,G1,G2,B RAW tensors.

    Input and output layouts are ``B x 4 x H x W``. Spatial dimensions must be
    divisible by four because the network has two stride-2 encoder stages.
    The output is intentionally not clipped during training so saturated
    gradients remain available to the loss function.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 4:
            raise ValueError("base_channels must be >= 8 and divisible by 4")
        self.stem = ConvNormAct(4, base_channels)
        self.encoder1 = ResidualBlock(base_channels)
        self.down1 = ConvNormAct(base_channels, base_channels * 2, stride=2)
        self.encoder2 = ResidualBlock(base_channels * 2)
        self.down2 = ConvNormAct(base_channels * 2, base_channels * 4, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )
        self.decoder2 = DecoderStage(base_channels * 4, base_channels * 2)
        self.decoder1 = DecoderStage(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("input must have shape B x 4 x H x W")
        if inputs.shape[-2] % 4 or inputs.shape[-1] % 4:
            raise ValueError("input height and width must be divisible by four")
        skip1 = self.encoder1(self.stem(inputs))
        skip2 = self.encoder2(self.down1(skip1))
        features = self.bottleneck(self.down2(skip2))
        features = self.decoder2(features, skip2)
        features = self.decoder1(features, skip1)
        return inputs + self.output(features)
