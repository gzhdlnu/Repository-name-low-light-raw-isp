# =============================================================================
# Project: RAW ISP | Task: Bayer Lens Shading Correction
# Timestamp: 2026-09-28 | Author: Guozh | Version: Phase1-LSC-v0.1.0
# =============================================================================
"""Lens-shading gain-map application for packed Bayer RAW data."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LscConfig:
    """Safety range for multiplicative R/G1/G2/B correction gains."""

    minimum_gain: float = 0.5
    maximum_gain: float = 4.0

    def validate(self) -> None:
        if not np.isfinite(self.minimum_gain) or self.minimum_gain <= 0:
            raise ValueError("minimum_gain must be finite and positive")
        if not np.isfinite(self.maximum_gain) or self.maximum_gain < self.minimum_gain:
            raise ValueError("maximum_gain must be finite and >= minimum_gain")


@dataclass(frozen=True)
class LscResult:
    corrected: np.ndarray
    full_resolution_gain_map: np.ndarray


def resize_lsc_gain_map(gain_map: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a coarse R/G1/G2/B gain grid to packed RAW size."""
    gains = np.asarray(gain_map, dtype=np.float32)
    if gains.ndim != 3 or gains.shape[-1] != 4:
        raise ValueError("LSC gain map must have shape H x W x 4 in R/G1/G2/B order")
    output_height, output_width = output_shape
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_shape must contain positive height and width")
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("LSC gain map must contain finite positive gains")
    if gains.shape[:2] == output_shape:
        return gains.copy()
    # Calibration grid nodes represent the image boundary coordinates. Use an
    # explicit align-corners map instead of cv2.resize's half-pixel convention.
    map_x, map_y = np.meshgrid(
        np.linspace(0.0, gains.shape[1] - 1, output_width, dtype=np.float32),
        np.linspace(0.0, gains.shape[0] - 1, output_height, dtype=np.float32),
    )
    resized = np.empty((output_height, output_width, 4), dtype=np.float32)
    for channel in range(4):
        resized[..., channel] = cv2.remap(
            gains[..., channel],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return resized


def apply_lens_shading_correction_packed(
    packed: np.ndarray,
    gain_map: np.ndarray,
    config: LscConfig | None = None,
) -> LscResult:
    """Apply a calibrated gain map without clipping scene-linear highlights."""
    config = config or LscConfig()
    config.validate()
    source = np.asarray(packed, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != 4:
        raise ValueError("packed Bayer input must have shape H x W x 4")
    if not np.isfinite(source).all():
        raise ValueError("packed Bayer input contains NaN or Inf")

    full_gains = resize_lsc_gain_map(gain_map, source.shape[:2])
    observed_min = float(full_gains.min())
    observed_max = float(full_gains.max())
    if observed_min < config.minimum_gain or observed_max > config.maximum_gain:
        raise ValueError(
            f"LSC gain range [{observed_min:.6g}, {observed_max:.6g}] exceeds "
            f"configured [{config.minimum_gain:.6g}, {config.maximum_gain:.6g}]"
        )
    return LscResult(
        corrected=(source * full_gains).astype(np.float32),
        full_resolution_gain_map=full_gains,
    )


def identity_lsc_gain_map() -> np.ndarray:
    """Return the smallest valid four-channel identity calibration grid."""
    return np.ones((2, 2, 4), dtype=np.float32)
