# =============================================================================
# Project: RAW ISP | Task: RAW Restoration Losses
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase4-Loss-v0.1.0
# =============================================================================
"""Loss functions for learning-based RAW restoration."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Return the robust differentiable L1 loss used for image restoration."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    difference = prediction - target
    return torch.sqrt(difference * difference + epsilon * epsilon).mean()


def raw_edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare first-order gradients of packed RAW without cross-color mixing."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError("prediction and target must have shape B x 4 x H x W")
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))


def raw_chroma_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Preserve RAW color-opponent differences while respecting Bayer planes."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError("prediction and target must have shape B x 4 x H x W")
    pred_green = 0.5 * (prediction[:, 1:2] + prediction[:, 2:3])
    target_green = 0.5 * (target[:, 1:2] + target[:, 2:3])
    pred_opponents = torch.cat([prediction[:, 0:1] - pred_green, prediction[:, 3:4] - pred_green], dim=1)
    target_opponents = torch.cat([target[:, 0:1] - target_green, target[:, 3:4] - target_green], dim=1)
    return F.l1_loss(pred_opponents, target_opponents)
