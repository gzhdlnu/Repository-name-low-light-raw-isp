# =============================================================================
# Project: RAW ISP | Task: Noise-Aware RAW Denoising
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase3-Denoise-v0.1.0
# =============================================================================
"""Traditional denoisers for packed Bayer data in scene-linear RAW space."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np


CHANNEL_NAMES = ("R", "G1", "G2", "B")


def _validate_packed(packed: np.ndarray) -> np.ndarray:
    x = np.asarray(packed, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError("packed Bayer input must have shape H x W x 4")
    if not np.isfinite(x).all():
        raise ValueError("packed Bayer input contains NaN or Inf")
    return x


def gaussian_denoise_packed(packed: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply the same fixed Gaussian blur independently to each Bayer plane.

    This is intentionally signal-independent and serves as a conventional
    smoothing baseline. No values are mixed between R, G1, G2 and B planes.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x = _validate_packed(packed)
    filtered = cv2.GaussianBlur(
        x,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.maximum(filtered, 0.0).astype(np.float32, copy=False)


def parse_noise_models(
    channels: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Read per-channel ``variance = a * signal + b`` model parameters."""
    slopes: list[float] = []
    intercepts: list[float] = []
    for name in CHANNEL_NAMES:
        if name not in channels:
            raise ValueError(f"noise report is missing channel {name}")
        model = channels[name].get("variance_model")
        if not isinstance(model, Mapping):
            raise ValueError(f"noise report channel {name} has no variance_model")
        slope = float(model["slope_a"])
        intercept = float(model["intercept_b"])
        if not np.isfinite(slope) or not np.isfinite(intercept) or slope < 0 or intercept < 0:
            raise ValueError(f"invalid nonnegative noise model for channel {name}")
        slopes.append(slope)
        intercepts.append(intercept)
    return np.asarray(slopes, dtype=np.float32), np.asarray(intercepts, dtype=np.float32)


def noise_aware_wiener_packed(
    packed: np.ndarray,
    slope_a: Sequence[float] | np.ndarray,
    intercept_b: Sequence[float] | np.ndarray,
    window_size: int = 7,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply a heteroscedastic local Wiener filter per Bayer plane.

    The input is exposure-corrected linear RAW. For each pixel, the expected
    noise variance is ``strength * (a * local_mean + b)``. Local structure is
    retained when measured neighborhood variance exceeds expected noise;
    otherwise the estimate moves toward the local mean.

    Args:
        packed: Float32 H x W x 4 Bayer planes in R,G1,G2,B order.
        slope_a: Four signal-dependent variance coefficients.
        intercept_b: Four signal-independent variance floors.
        window_size: Odd local-statistics window size in packed-pixel units.
        strength: Multiplier for the estimated variance model.
    """
    x = _validate_packed(packed)
    slopes = np.asarray(slope_a, dtype=np.float32)
    intercepts = np.asarray(intercept_b, dtype=np.float32)
    if slopes.shape != (4,) or intercepts.shape != (4,):
        raise ValueError("slope_a and intercept_b must each contain four values")
    if np.any(slopes < 0) or np.any(intercepts < 0):
        raise ValueError("noise model coefficients must be nonnegative")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if strength <= 0:
        raise ValueError("strength must be positive")

    kernel = (window_size, window_size)
    output = np.empty_like(x)
    for channel in range(4):
        plane = x[..., channel]
        local_mean = cv2.boxFilter(
            plane, cv2.CV_32F, kernel, normalize=True, borderType=cv2.BORDER_REFLECT_101
        )
        local_second_moment = cv2.boxFilter(
            plane * plane,
            cv2.CV_32F,
            kernel,
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )
        local_variance = np.maximum(local_second_moment - local_mean * local_mean, 0.0)
        expected_noise = strength * (
            slopes[channel] * np.maximum(local_mean, 0.0) + intercepts[channel]
        )
        detail_gain = np.maximum(local_variance - expected_noise, 0.0) / np.maximum(
            local_variance, 1e-12
        )
        output[..., channel] = local_mean + detail_gain * (plane - local_mean)

    return np.maximum(output, 0.0).astype(np.float32, copy=False)
