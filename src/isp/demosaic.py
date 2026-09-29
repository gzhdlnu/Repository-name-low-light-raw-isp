# =============================================================================
# Project: RAW ISP | Task: Bayer Demosaic Algorithms
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase2-Demosaic-v0.1.0
# =============================================================================
"""Demosaic backends with a shared floating-point packed-RAW contract."""

from __future__ import annotations

import cv2
import numpy as np

from .baseline import _check_pattern, _unpack_to_mosaic


_CODES = {
    "bilinear": {
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
    },
    "vng": {
        "RGGB": cv2.COLOR_BayerRG2RGB_VNG,
        "BGGR": cv2.COLOR_BayerBG2RGB_VNG,
        "GRBG": cv2.COLOR_BayerGR2RGB_VNG,
        "GBRG": cv2.COLOR_BayerGB2RGB_VNG,
    },
    "edge_aware": {
        "RGGB": cv2.COLOR_BayerRG2RGB_EA,
        "BGGR": cv2.COLOR_BayerBG2RGB_EA,
        "GRBG": cv2.COLOR_BayerGB2RGB_EA,
        "GBRG": cv2.COLOR_BayerGR2RGB_EA,
    },
}


def demosaic_opencv(
    packed: np.ndarray,
    pattern: str = "RGGB",
    method: str = "bilinear",
) -> np.ndarray:
    """Demosaic packed R/G1/G2/B data using an OpenCV Bayer backend."""
    pattern = _check_pattern(pattern)
    method = method.lower()
    if method not in _CODES:
        raise ValueError(f"unsupported demosaic method: {method}")
    mosaic = _unpack_to_mosaic(np.asarray(packed, dtype=np.float32), pattern)
    if mosaic.ndim != 2 or not np.isfinite(mosaic).all():
        raise ValueError("packed Bayer input must be finite")
    scale = max(float(mosaic.max()), 1.0)
    if method == "bilinear":
        mosaic_input = np.round(np.clip(mosaic / scale, 0.0, 1.0) * 65535.0).astype(np.uint16)
        output_scale = 65535.0
    else:
        # OpenCV's VNG backend is CV_8U-only. Keep the conversion explicit so
        # the report can distinguish algorithm behavior from quantization cost.
        mosaic_input = np.round(np.clip(mosaic / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
        output_scale = 255.0
    rgb = cv2.cvtColor(mosaic_input, _CODES[method][pattern])
    return rgb.astype(np.float32) * (scale / output_scale)
