# =============================================================================
# Project: RAW ISP | Task: YUV Backend Filters
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase7-YUVFilter-v0.1.0
# =============================================================================
"""Conservative YUV-domain filters for backend ablation studies."""

from __future__ import annotations

import cv2
import numpy as np


def apply_yuv_filters(rgb: np.ndarray, *, ynr_sigma: float = 0.0, cnr_sigma: float = 0.0, sharpen_amount: float = 0.0) -> np.ndarray:
    """Apply optional luma denoise, chroma denoise and guarded luma sharpen."""
    x = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    yuv = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
    if ynr_sigma > 0:
        blurred = cv2.GaussianBlur(yuv[..., 0], (0, 0), ynr_sigma)
        yuv[..., 0] = 0.65 * blurred + 0.35 * yuv[..., 0]
    if cnr_sigma > 0:
        for channel in (1, 2):
            yuv[..., channel] = cv2.GaussianBlur(yuv[..., channel], (0, 0), cnr_sigma)
    if sharpen_amount > 0:
        y = yuv[..., 0]
        low = cv2.GaussianBlur(y, (0, 0), 1.0)
        yuv[..., 0] = np.clip(y + sharpen_amount * (y - low), 0.0, 1.0)
    return np.clip(cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)
