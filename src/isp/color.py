# =============================================================================
# Project: RAW ISP | Task: Camera Color Correction
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase2-Color-v0.2.0
# =============================================================================
"""Color-space utilities for camera-linear RGB data."""

from __future__ import annotations

import numpy as np


# IEC 61966-2-1 linear sRGB (D65) -> CIE XYZ matrix.
_SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float32,
)


def camera_rgb_to_srgb_matrix(rgb_xyz_matrix: np.ndarray) -> np.ndarray:
    """Derive a camera-RGB-to-linear-sRGB matrix from LibRaw calibration.

    LibRaw exposes ``cam_xyz`` as camera-channel responses to XYZ. Following
    dcraw/LibRaw practice, it is composed with sRGB-to-XYZ, row-normalized so
    neutral sRGB maps to equal camera channels, and then inverted.
    """
    xyz_to_camera = np.asarray(rgb_xyz_matrix, dtype=np.float32)[:3, :3]
    if xyz_to_camera.shape != (3, 3) or not np.isfinite(xyz_to_camera).all():
        raise ValueError("rgb_xyz_matrix must contain a finite 3 x 3 camera matrix")
    camera_from_srgb = xyz_to_camera @ _SRGB_TO_XYZ
    row_sums = camera_from_srgb.sum(axis=1, keepdims=True)
    if np.any(np.abs(row_sums) < 1e-8):
        raise ValueError("camera calibration matrix has a zero row response")
    camera_from_srgb /= row_sums
    return np.linalg.inv(camera_from_srgb).astype(np.float32)
