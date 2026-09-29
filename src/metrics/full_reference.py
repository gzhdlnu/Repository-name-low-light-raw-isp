# =============================================================================
# Project: RAW ISP | Task: Full-Reference Image Quality Metrics
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase3-Metrics-v0.3.0
# =============================================================================
"""Metrics for aligned display-referred RGB image pairs."""

from __future__ import annotations

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import deltaE_ciede2000, rgb2lab


def compare_display_rgb(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Calculate structural and perceptual-color metrics for aligned RGB."""
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and reference must have matching H x W x 3 shapes")
    # Subsample the color calculation to limit memory on 12 MP RAW frames.
    pred_lab = rgb2lab(prediction[::4, ::4].astype(np.float32) / 255.0)
    ref_lab = rgb2lab(reference[::4, ::4].astype(np.float32) / 255.0)
    delta_e = deltaE_ciede2000(ref_lab, pred_lab)
    return {
        "psnr_db": float(peak_signal_noise_ratio(reference, prediction, data_range=255)),
        "ssim": float(structural_similarity(reference, prediction, channel_axis=2, data_range=255)),
        "delta_e2000_mean": float(np.mean(delta_e)),
        "delta_e2000_p50": float(np.percentile(delta_e, 50)),
        "delta_e2000_p95": float(np.percentile(delta_e, 95)),
        "gradient_similarity": gradient_magnitude_similarity(prediction, reference),
    }


def gradient_magnitude_similarity(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Measure display-luma edge agreement; higher is better, with 1 ideal.

    This complements PSNR/SSIM when evaluating denoisers: excessive smoothing
    and residual high-frequency noise both reduce gradient agreement.
    """
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and reference must have matching H x W x 3 shapes")

    def gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        dx = cv2.Scharr(luma, cv2.CV_32F, 1, 0)
        dy = cv2.Scharr(luma, cv2.CV_32F, 0, 1)
        return cv2.magnitude(dx, dy)

    pred_gradient = gradient_magnitude(prediction)
    ref_gradient = gradient_magnitude(reference)
    stabilizer = 0.0026
    similarity = (2.0 * pred_gradient * ref_gradient + stabilizer) / (
        pred_gradient * pred_gradient + ref_gradient * ref_gradient + stabilizer
    )
    return float(np.mean(similarity))


def display_luma_summary(rgb: np.ndarray) -> dict[str, float]:
    """Summarize display-encoded Rec.709 luma and near-white pixels."""
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    return {
        "luma_p01": float(np.percentile(luma, 1)),
        "luma_p50": float(np.percentile(luma, 50)),
        "luma_p99": float(np.percentile(luma, 99)),
        "near_white_ratio": float(np.mean(np.max(x, axis=-1) >= 0.99)),
        "near_black_ratio": float(np.mean(np.max(x, axis=-1) <= 0.01)),
    }
