# =============================================================================
# Project: RAW ISP | Task: RAW Residual Noise Analysis
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase3-Noise-v0.2.0
# =============================================================================
"""Robust RAW residual statistics for aligned short/long SID pairs."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares, nnls


def _robust_sigma(values: np.ndarray) -> float:
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def fit_nonnegative_variance_model(
    signal: np.ndarray,
    variance: np.ndarray,
) -> dict[str, object]:
    """Robustly fit ``variance = a * signal + b`` with ``a,b >= 0``.

    Each observation receives equal weight so a scene with a large flat area
    cannot dominate the cross-scene calibration merely through pixel count.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    y = np.asarray(variance, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or np.ptp(x) <= 0:
        raise ValueError("at least three valid observations across multiple signal levels are required")

    design = np.column_stack([x, np.ones_like(x)])
    initial = nnls(design, y)[0]
    initial_residual = design @ initial - y
    residual_scale = max(float(np.median(np.abs(initial_residual))), 1e-8)
    fitted = least_squares(
        lambda parameters: (design @ parameters - y) / residual_scale,
        x0=initial,
        bounds=(0.0, np.inf),
        loss="soft_l1",
    ).x
    prediction = design @ fitted
    residual = prediction - y
    total_variation = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / max(total_variation, 1e-16)
    return {
        "variance_model": {"slope_a": float(fitted[0]), "intercept_b": float(fitted[1])},
        "fit_quality": {
            "observations": int(x.size),
            "r_squared": r_squared,
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
        },
    }


def estimate_residual_noise_profile(
    short_packed: np.ndarray,
    long_packed: np.ndarray,
    exposure_gain: float,
    bin_edges: np.ndarray,
    flat_fraction: float = 0.4,
    minimum_samples: int = 500,
) -> dict[str, dict[str, object]]:
    """Estimate variance(signal) from flat regions of an aligned RAW pair.

    This measures gain-normalized short-vs-long residuals, not a laboratory
    photon-transfer curve. Scene motion, misalignment and long-frame noise can
    still contribute to the residual.
    """
    if short_packed.shape != long_packed.shape or short_packed.shape[-1] != 4:
        raise ValueError("short and long inputs must have matching H x W x 4 shapes")
    if not 0 < flat_fraction <= 1:
        raise ValueError("flat_fraction must be in (0, 1]")

    channel_names = ("R", "G1", "G2", "B")
    result: dict[str, dict[str, object]] = {}
    for channel_index, channel_name in enumerate(channel_names):
        reference = long_packed[..., channel_index].astype(np.float32)
        corrected_short = short_packed[..., channel_index].astype(np.float32) * exposure_gain
        residual = corrected_short - reference

        grad_x = cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        valid_signal = (reference >= bin_edges[0]) & (reference < bin_edges[-1])
        threshold = float(np.quantile(gradient[valid_signal], flat_fraction))
        flat = gradient <= threshold

        bins = []
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            mask = flat & (reference >= lower) & (reference < upper)
            values = residual[mask]
            if values.size < minimum_samples:
                continue
            sigma = _robust_sigma(values)
            signal = float(np.median(reference[mask]))
            bins.append(
                {
                    "signal": signal,
                    "bias": float(np.median(values)),
                    "sigma": sigma,
                    "variance": sigma * sigma,
                    "snr_db": float(20.0 * np.log10(max(signal, 1e-8) / max(sigma, 1e-8))),
                    "samples": int(values.size),
                }
            )

        if len(bins) >= 2:
            design = np.column_stack(
                [np.array([item["signal"] for item in bins]), np.ones(len(bins), dtype=np.float64)]
            )
            variance = np.array([item["variance"] for item in bins], dtype=np.float64)
            slope, intercept = nnls(design, variance)[0]
        else:
            slope, intercept = float("nan"), float("nan")
        result[channel_name] = {
            "flat_gradient_threshold": threshold,
            "variance_model": {"slope_a": float(slope), "intercept_b": float(intercept)},
            "bins": bins,
        }
    return result
