# =============================================================================
# Project: RAW ISP | Task: Normalized RAW Input
# Timestamp: 2026-09-23 | Author: Guozh | Version: Phase7-RawIO-v0.1.0
# =============================================================================
"""Decode camera RAW files using the same normalization as model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy

from .baseline import pack_bayer
from .color import camera_rgb_to_srgb_matrix


@dataclass(frozen=True)
class NormalizedRawFrame:
    """Normalized packed RAW and metadata needed by the downstream ISP."""

    packed: np.ndarray
    pattern: str
    black_levels: tuple[float, float, float, float]
    white_level: float
    white_balance_gains: tuple[float, float, float, float]
    color_matrix: np.ndarray


def load_normalized_raw(path: str | Path) -> NormalizedRawFrame:
    """Load RAW as H/2 x W/2 x 4 float32 in R,G1,G2,B order and [0, 1]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with rawpy.imread(str(path)) as image:
        raw = image.raw_image_visible.astype(np.float32)
        white = float(image.white_level)
        black_rgbg = np.asarray(image.black_level_per_channel[:4], dtype=np.float32)
        pattern = "".join("RGBG"[int(value)] for value in image.raw_pattern.flat)
        wb_rgbg = np.asarray(image.camera_whitebalance[:4], dtype=np.float32)
        color_matrix = camera_rgb_to_srgb_matrix(image.rgb_xyz_matrix)

    packed = pack_bayer(raw, pattern)
    # rawpy metadata order is R,G,B,G; packed order is R,G1,G2,B.
    black = black_rgbg[[0, 1, 3, 2]]
    denominator = white - black
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError(f"invalid black/white level metadata: {path}")
    normalized = np.clip(
        (packed - black.reshape(1, 1, 4)) / denominator.reshape(1, 1, 4),
        0.0,
        1.0,
    ).astype(np.float32)

    gains = wb_rgbg[[0, 1, 3, 2]]
    green_reference = max(float(gains[1]), float(gains[2]), 1e-6)
    gains = gains / green_reference
    if np.any(~np.isfinite(gains)) or np.any(gains <= 0):
        raise ValueError(f"invalid camera white-balance metadata: {path}")

    return NormalizedRawFrame(
        packed=normalized,
        pattern=pattern,
        black_levels=tuple(float(value) for value in black),
        white_level=white,
        white_balance_gains=tuple(float(value) for value in gains),
        color_matrix=color_matrix,
    )
