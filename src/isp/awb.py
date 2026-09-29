# =============================================================================
# Project: RAW ISP | Task: Automatic White Balance Estimation
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase2-AWB-v0.1.0
# =============================================================================
"""Explainable AWB estimators for packed linear Bayer RAW."""

from __future__ import annotations

import numpy as np


CHANNEL_NAMES = ("R", "G1", "G2", "B")


def _validate(packed: np.ndarray) -> np.ndarray:
    x = np.asarray(packed, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError("packed Bayer input must have shape H x W x 4")
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("packed Bayer input must be finite and non-negative")
    return x


def _valid_mask(x: np.ndarray, low: float, high: float) -> np.ndarray:
    green = 0.5 * (x[..., 1] + x[..., 2])
    brightness = (x[..., 0] + green + x[..., 3]) / 3.0
    return (brightness > low) & (brightness < high)


def _normalize_gains(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float32)
    green = max(float(0.5 * (values[1] + values[2])), 1e-8)
    # values are measured channel responses; AWB gains compensate their
    # inverse response while using green as the unity reference.
    gains = green / np.maximum(values, 1e-8)
    return tuple(float(value) for value in gains)


def _masked_channel_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(np.count_nonzero(mask)) < 32:
        raise ValueError("AWB valid pixel count is too small")
    return np.asarray([np.mean(x[..., channel][mask]) for channel in range(4)], dtype=np.float32)


def gray_world_gains(
    packed: np.ndarray,
    low: float = 0.01,
    high: float = 0.98,
) -> tuple[float, float, float, float]:
    """Estimate gains by equalizing robustly masked channel means."""
    x = _validate(packed)
    return _normalize_gains(_masked_channel_mean(x, _valid_mask(x, low, high)))


def shades_of_gray_gains(
    packed: np.ndarray,
    p: float = 6.0,
    low: float = 0.01,
    high: float = 0.98,
) -> tuple[float, float, float, float]:
    """Estimate gains from Minkowski channel statistics."""
    if p <= 0:
        raise ValueError("p must be positive")
    x = _validate(packed)
    mask = _valid_mask(x, low, high)
    if int(np.count_nonzero(mask)) < 32:
        raise ValueError("AWB valid pixel count is too small")
    values = np.asarray(
        [np.mean(np.power(np.maximum(x[..., channel][mask], 0.0), p)) ** (1.0 / p) for channel in range(4)],
        dtype=np.float32,
    )
    return _normalize_gains(values)


def gray_edge_gains(
    packed: np.ndarray,
    low: float = 0.01,
    high: float = 0.98,
) -> tuple[float, float, float, float]:
    """Estimate gains from channel gradient energy, suppressing flat-color bias."""
    x = _validate(packed)
    mask = _valid_mask(x, low, high)
    if int(np.count_nonzero(mask)) < 32:
        raise ValueError("AWB valid pixel count is too small")
    values = []
    for channel in range(4):
        plane = x[..., channel]
        dx = np.diff(plane, axis=1, prepend=plane[:, :1])
        dy = np.diff(plane, axis=0, prepend=plane[:1, :])
        energy = np.sqrt(dx * dx + dy * dy)
        selected = energy[mask]
        values.append(float(np.mean(selected)) if selected.size else 0.0)
    values = np.asarray(values, dtype=np.float32)
    if np.max(values) <= 1e-8:
        return gray_world_gains(x, low=low, high=high)
    return _normalize_gains(values)


def apply_white_balance_gains(
    packed: np.ndarray,
    gains: tuple[float, float, float, float],
) -> np.ndarray:
    """Apply R/G1/G2/B gains while retaining scene-linear headroom."""
    x = _validate(packed)
    gains_array = np.asarray(gains, dtype=np.float32)
    if gains_array.shape != (4,) or not np.isfinite(gains_array).all() or np.any(gains_array <= 0):
        raise ValueError("gains must contain four finite positive values")
    return x * gains_array.reshape(1, 1, 4)
