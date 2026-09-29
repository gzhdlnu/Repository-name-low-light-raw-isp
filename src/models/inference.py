# =============================================================================
# Project: RAW ISP | Task: Seam-Resistant Tiled RAW Inference
# Timestamp: 2026-09-22 | Author: Guozh | Version: Phase4-Infer-v0.1.0
# =============================================================================
"""Memory-bounded tiled inference for packed RAW restoration models."""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def tiled_predict(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    tile_size: int = 512,
    halo: int = 32,
    use_amp: bool = True,
    tile_batch_size: int = 1,
) -> torch.Tensor:
    """Predict one full packed-RAW frame using non-overlapping halo tiles.

    The model sees ``halo`` neighboring pixels around every output tile. Only
    the central region is stitched into the result, avoiding visible seams
    without averaging independent predictions.
    """
    if inputs.ndim != 4 or inputs.shape[0] != 1 or inputs.shape[1] != 4:
        raise ValueError("inputs must have shape 1 x 4 x H x W")
    if tile_size <= 0 or tile_size % 4:
        raise ValueError("tile_size must be positive and divisible by four")
    if halo < 0 or halo % 4:
        raise ValueError("halo must be nonnegative and divisible by four")
    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be positive")

    height, width = inputs.shape[-2:]
    padded_height = ((height + tile_size - 1) // tile_size) * tile_size
    padded_width = ((width + tile_size - 1) // tile_size) * tile_size
    core = F.pad(inputs, (0, padded_width - width, 0, padded_height - height), mode="reflect")
    haloed = F.pad(core, (halo, halo, halo, halo), mode="reflect") if halo else core
    prediction = torch.empty_like(core)
    was_training = model.training
    model.eval()
    try:
        locations = [
            (top, left)
            for top in range(0, padded_height, tile_size)
            for left in range(0, padded_width, tile_size)
        ]
        for start in range(0, len(locations), tile_batch_size):
            batch_locations = locations[start : start + tile_batch_size]
            tiles = torch.cat(
                [
                    haloed[
                        :,
                        :,
                        top : top + tile_size + 2 * halo,
                        left : left + tile_size + 2 * halo,
                    ]
                    for top, left in batch_locations
                ],
                dim=0,
            )
            with torch.autocast(
                device_type=inputs.device.type,
                dtype=torch.float16,
                enabled=use_amp and inputs.device.type == "cuda",
            ):
                restored_batch = model(tiles)
            if halo:
                restored_batch = restored_batch[:, :, halo:-halo, halo:-halo]
            for batch_index, (top, left) in enumerate(batch_locations):
                prediction[:, :, top : top + tile_size, left : left + tile_size] = restored_batch[
                    batch_index : batch_index + 1
                ]
    finally:
        model.train(was_training)
    return prediction[:, :, :height, :width]
