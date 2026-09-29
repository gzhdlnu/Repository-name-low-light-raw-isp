# =============================================================================
# Project: RAW ISP | Task: Global Tone Mapping
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase2-Tone-v0.1.0
# =============================================================================
"""Tone-mapping operators for scene-linear RGB."""

from __future__ import annotations

import numpy as np


def reinhard_tone_map(rgb: np.ndarray, white_point: float = 4.0) -> np.ndarray:
    """Compress scene-linear RGB while preserving channel ratios.

    Args:
        rgb: Scene-linear RGB array. Values may exceed 1 after exposure gain.
        white_point: Scene-linear value mapped close to display white. Must be
            positive. Larger values retain more highlight contrast.

    Returns:
        Float32 RGB in [0, 1], still linear and ready for an sRGB transfer curve.
    """
    if white_point <= 0:
        raise ValueError("white_point must be positive")
    x = np.maximum(np.asarray(rgb, dtype=np.float32), 0.0)
    mapped = x * (1.0 + x / (white_point * white_point)) / (1.0 + x)
    return np.clip(mapped, 0.0, 1.0)
