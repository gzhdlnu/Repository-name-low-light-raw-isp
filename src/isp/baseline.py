# =============================================================================
# Project: RAW ISP | Task: Traditional ISP Baseline
# Timestamp: 2026-09-18 | Author: Guozh | Version: Phase1-Baseline-v0.1.0
# =============================================================================
"""A minimal, explicit RAW-to-sRGB ISP baseline."""

from __future__ import annotations

import cv2
import numpy as np


def _check_pattern(pattern: str) -> str:
    pattern = pattern.upper()
    if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    return pattern


def normalize_raw(
    raw: np.ndarray,
    black_level: float,
    white_level: float,
    clip: bool = True,
) -> np.ndarray:
    """Remove sensor offset and scale RAW values; optionally retain negatives."""
    if white_level <= black_level:
        raise ValueError("white_level must be greater than black_level")
    x = np.asarray(raw, dtype=np.float32)
    normalized = (x - black_level) / (white_level - black_level)
    return np.clip(normalized, 0.0, 1.0) if clip else normalized


def pack_bayer(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Pack a 2D Bayer image into H/2 x W/2 x 4 in R,G1,G2,B order."""
    pattern = _check_pattern(pattern)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] % 2 or raw.shape[1] % 2:
        raise ValueError("RAW must be a 2D array with even height and width")
    # Extract physical locations first, then map them to semantic channels.
    locations = {
        "RGGB": ("R", "G1", "G2", "B"),
        "BGGR": ("B", "G1", "G2", "R"),
        "GRBG": ("G1", "R", "B", "G2"),
        "GBRG": ("G1", "B", "R", "G2"),
    }[pattern]
    # For non-RGGB patterns, the names above describe positions; remap values
    # so the returned channels are always R,G1,G2,B.
    position_values = [raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]]
    position_names = {
        "RGGB": ("R", "G1", "G2", "B"),
        "BGGR": ("B", "G1", "G2", "R"),
        "GRBG": ("G1", "R", "B", "G2"),
        "GBRG": ("G1", "B", "R", "G2"),
    }[pattern]
    by_name = dict(zip(position_names, position_values))
    return np.stack([by_name[name] for name in ("R", "G1", "G2", "B")], axis=-1)


def white_balance(packed: np.ndarray, gains: tuple[float, float, float, float]) -> np.ndarray:
    """Apply per-plane gains while retaining values above display white."""
    x = np.asarray(packed, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError("packed Bayer input must have shape H x W x 4")
    return np.maximum(x * np.asarray(gains, dtype=np.float32), 0.0)


def _unpack_to_mosaic(packed: np.ndarray, pattern: str) -> np.ndarray:
    pattern = _check_pattern(pattern)
    h, w, _ = packed.shape
    mosaic = np.empty((h * 2, w * 2), dtype=np.float32)
    positions = {
        "RGGB": ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)),
        "BGGR": ((0, 0, 3), (0, 1, 2), (1, 0, 1), (1, 1, 0)),
        "GRBG": ((0, 0, 1), (0, 1, 0), (1, 0, 3), (1, 1, 2)),
        "GBRG": ((0, 0, 2), (0, 1, 3), (1, 0, 0), (1, 1, 1)),
    }[pattern]
    for row, col, channel in positions:
        mosaic[row::2, col::2] = packed[..., channel]
    return mosaic


def demosaic_bilinear(packed: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Bilinear demosaic while preserving floating-point scene range."""
    pattern = _check_pattern(pattern)
    mosaic = _unpack_to_mosaic(np.asarray(packed, dtype=np.float32), pattern)
    codes = {
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
    }
    # OpenCV accepts uint8/uint16 only. Normalize temporarily because Bayer
    # interpolation is linear, then restore the original scene-linear scale.
    scale = max(float(mosaic.max()), 1.0)
    mosaic_u16 = np.round(np.clip(mosaic / scale, 0.0, 1.0) * 65535.0).astype(np.uint16)
    rgb_u16 = cv2.cvtColor(mosaic_u16, codes[pattern])
    return rgb_u16.astype(np.float32) * (scale / 65535.0)


def apply_color_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix and retain positive scene-linear highlight range."""
    rgb = np.asarray(rgb, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or matrix.shape != (3, 3):
        raise ValueError("Expected RGB H x W x 3 and a 3 x 3 matrix")
    return np.maximum(np.einsum("...c,dc->...d", rgb, matrix), 0.0)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to the standard piecewise sRGB transfer curve."""
    x = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
