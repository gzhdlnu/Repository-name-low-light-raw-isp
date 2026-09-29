# =============================================================================
# Project: RAW ISP | Task: Bayer Defective Pixel Correction
# Timestamp: 2026-09-28 | Author: Guozh | Version: Phase1-DPC-v0.1.0
# =============================================================================
"""Conservative defective-pixel correction for packed Bayer RAW data."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DpcConfig:
    """Thresholds for normalized linear RAW in the [0, 1] sensor range."""

    absolute_threshold: float = 0.002
    relative_threshold: float = 0.05
    mad_scale: float = 6.0

    def validate(self) -> None:
        if self.absolute_threshold < 0:
            raise ValueError("absolute_threshold must be non-negative")
        if self.relative_threshold < 0:
            raise ValueError("relative_threshold must be non-negative")
        if self.mad_scale < 0:
            raise ValueError("mad_scale must be non-negative")


@dataclass(frozen=True)
class DpcResult:
    corrected: np.ndarray
    defect_mask: np.ndarray
    hot_mask: np.ndarray
    dead_mask: np.ndarray

    @property
    def corrected_count(self) -> int:
        return int(np.count_nonzero(self.defect_mask))


def _shifted_neighbors(plane: np.ndarray) -> dict[str, np.ndarray]:
    padded = np.pad(plane, 1, mode="reflect")
    height, width = plane.shape
    return {
        "nw": padded[0:height, 0:width],
        "n": padded[0:height, 1 : width + 1],
        "ne": padded[0:height, 2 : width + 2],
        "w": padded[1 : height + 1, 0:width],
        "e": padded[1 : height + 1, 2 : width + 2],
        "sw": padded[2 : height + 2, 0:width],
        "s": padded[2 : height + 2, 1 : width + 1],
        "se": padded[2 : height + 2, 2 : width + 2],
    }


def _directional_replacement(neighbors: dict[str, np.ndarray]) -> np.ndarray:
    pairs = (
        (neighbors["w"], neighbors["e"]),
        (neighbors["n"], neighbors["s"]),
        (neighbors["nw"], neighbors["se"]),
        (neighbors["ne"], neighbors["sw"]),
    )
    best_gradient = np.full(pairs[0][0].shape, np.inf, dtype=np.float32)
    replacement = np.zeros(pairs[0][0].shape, dtype=np.float32)
    for first, second in pairs:
        gradient = np.abs(first - second)
        candidate = 0.5 * (first + second)
        select = gradient < best_gradient
        best_gradient[select] = gradient[select]
        replacement[select] = candidate[select]
    return replacement


def _correct_plane(plane: np.ndarray, config: DpcConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neighbors = _shifted_neighbors(plane)
    neighbor_values = tuple(neighbors.values())
    neighbor_min = np.minimum.reduce(neighbor_values)
    neighbor_max = np.maximum.reduce(neighbor_values)

    # A 3x3 median is robust to one impulsive center pixel. The second median
    # estimates local residual noise without assuming a globally fixed sigma.
    local_median = cv2.medianBlur(plane, 3)
    local_mad = cv2.medianBlur(np.abs(plane - local_median), 3)
    threshold = (
        config.absolute_threshold
        + config.relative_threshold * np.maximum(local_median, 0.0)
        + config.mad_scale * 1.4826 * local_mad
    )
    residual = np.abs(plane - local_median)

    hot = (plane > neighbor_max) & (residual > threshold)
    dead = (plane < neighbor_min) & (residual > threshold)
    defect = hot | dead
    defect[[0, -1], :] = False
    defect[:, [0, -1]] = False
    hot &= defect
    dead &= defect

    replacement = _directional_replacement(neighbors)
    corrected = plane.copy()
    corrected[defect] = replacement[defect]
    return corrected, defect, hot, dead


def correct_defective_pixels_packed(
    packed: np.ndarray,
    config: DpcConfig | None = None,
) -> DpcResult:
    """Detect and repair isolated hot/dead pixels independently per CFA plane.

    Packed input must use R/G1/G2/B channel order. Operating independently on
    each plane ensures that replacement values always come from the same CFA
    color instead of mixing red, green and blue sensor samples.
    """
    config = config or DpcConfig()
    config.validate()
    source = np.asarray(packed, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != 4:
        raise ValueError("packed Bayer input must have shape H x W x 4")
    if source.shape[0] < 3 or source.shape[1] < 3:
        raise ValueError("packed Bayer input must be at least 3 x 3")
    if not np.isfinite(source).all():
        raise ValueError("packed Bayer input contains NaN or Inf")

    corrected = np.empty_like(source)
    defect_mask = np.zeros(source.shape, dtype=bool)
    hot_mask = np.zeros(source.shape, dtype=bool)
    dead_mask = np.zeros(source.shape, dtype=bool)
    for channel in range(4):
        channel_result = _correct_plane(source[..., channel], config)
        corrected[..., channel] = channel_result[0]
        defect_mask[..., channel] = channel_result[1]
        hot_mask[..., channel] = channel_result[2]
        dead_mask[..., channel] = channel_result[3]

    return DpcResult(
        corrected=corrected,
        defect_mask=defect_mask,
        hot_mask=hot_mask,
        dead_mask=dead_mask,
    )
