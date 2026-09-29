# =============================================================================
# Project: RAW ISP | Task: YCbCr Output Backend
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase7-YUV-v0.1.0
# =============================================================================
"""Small, explicit BT.709 RGB/YCbCr and NV12 conversion utilities."""

from __future__ import annotations

import numpy as np


def rgb_to_ycbcr(rgb: np.ndarray, *, limited_range: bool = False) -> np.ndarray:
    """Convert float RGB in [0, 1] to float YCbCr in full or video range."""
    x = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    if x.shape[-1] != 3:
        raise ValueError("RGB input must have a final dimension of 3")
    matrix = np.array([[0.2126, 0.7152, 0.0722], [-0.114572, -0.385428, 0.5], [0.5, -0.454153, -0.045847]], np.float32)
    ycbcr = x @ matrix.T
    ycbcr[..., 1:] += 0.5
    if limited_range:
        ycbcr[..., 0] = 16.0 / 255.0 + ycbcr[..., 0] * (219.0 / 255.0)
        ycbcr[..., 1:] = 128.0 / 255.0 + (ycbcr[..., 1:] - 0.5) * (224.0 / 255.0)
    return np.clip(ycbcr, 0.0, 1.0)


def ycbcr_to_rgb(ycbcr: np.ndarray, *, limited_range: bool = False) -> np.ndarray:
    """Convert float YCbCr in [0, 1] back to float RGB in [0, 1]."""
    x = np.asarray(ycbcr, np.float32)
    if x.shape[-1] != 3:
        raise ValueError("YCbCr input must have a final dimension of 3")
    if limited_range:
        y = (x[..., 0] - 16.0 / 255.0) * (255.0 / 219.0)
        cb = (x[..., 1] - 128.0 / 255.0) * (255.0 / 224.0)
        cr = (x[..., 2] - 128.0 / 255.0) * (255.0 / 224.0)
    else:
        y, cb, cr = x[..., 0], x[..., 1] - 0.5, x[..., 2] - 0.5
    rgb = np.stack((y + 1.5748 * cr, y - 0.187324 * cb - 0.468124 * cr, y + 1.8556 * cb), axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def ycbcr_to_nv12(ycbcr: np.ndarray, *, limited_range: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Convert even-sized YCbCr to uint8 NV12 planes (Y, interleaved UV)."""
    x = np.clip(rgb_to_ycbcr(ycbcr, limited_range=limited_range), 0.0, 1.0)
    h, w, _ = x.shape
    if h % 2 or w % 2:
        raise ValueError("NV12 requires even height and width")
    y = np.rint(x[..., 0] * 255.0).astype(np.uint8)
    chroma = x[..., 1:].reshape(h // 2, 2, w // 2, 2, 2).mean(axis=(1, 3))
    uv = np.rint(chroma * 255.0).astype(np.uint8).reshape(h // 2, w)
    return y, uv
