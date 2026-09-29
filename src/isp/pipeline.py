# =============================================================================
# Project: RAW ISP | Task: Reusable Traditional ISP Pipeline
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase3-Pipeline-v0.2.0
# =============================================================================
"""Reusable Sony RAW processing pipeline for Phase 2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy

from .baseline import apply_color_matrix, demosaic_bilinear, linear_to_srgb, normalize_raw, pack_bayer, white_balance
from .color import camera_rgb_to_srgb_matrix
from .tone import reinhard_tone_map


@dataclass(frozen=True)
class ProcessedRaw:
    display_rgb: np.ndarray
    pattern: str
    black_level: float
    white_level: float
    white_balance_gains: tuple[float, float, float, float]
    exposure_gain: float
    pre_tone_over_white_ratio: float


def render_packed_raw(
    packed: np.ndarray,
    pattern: str,
    white_balance_gains: tuple[float, float, float, float],
    exposure_gain: float = 1.0,
    tone_map: str = "reinhard",
    white_point: float = 4.0,
    color_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Render packed linear Bayer data through the common downstream ISP.

    Keeping this stage shared ensures denoising alternatives are compared with
    identical white balance, demosaic, color and tone-mapping operations.
    """
    if exposure_gain <= 0:
        raise ValueError("exposure_gain must be positive")
    balanced = white_balance(packed, white_balance_gains)
    scene_rgb = np.maximum(demosaic_bilinear(balanced, pattern) * exposure_gain, 0.0)
    if color_matrix is not None:
        scene_rgb = apply_color_matrix(scene_rgb, color_matrix)

    over_white = float(np.mean(np.max(scene_rgb, axis=-1) > 1.0))
    if tone_map == "reinhard":
        display_linear = reinhard_tone_map(scene_rgb, white_point=white_point)
    elif tone_map == "clip":
        display_linear = np.clip(scene_rgb, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported tone map: {tone_map}")
    display_rgb = np.clip(linear_to_srgb(display_linear) * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return display_rgb, over_white


def process_sony_raw(
    raw_path: str | Path,
    exposure_gain: float,
    tone_map: str = "reinhard",
    white_point: float = 4.0,
    use_color_matrix: bool = False,
) -> ProcessedRaw:
    """Process one Sony ARW into uint8 display RGB.

    The working data stays scene-linear until tone mapping. Values above 1 are
    retained so highlight compression happens before display clipping.
    """
    if exposure_gain <= 0:
        raise ValueError("exposure_gain must be positive")
    raw_path = Path(raw_path)
    with rawpy.imread(str(raw_path)) as image:
        raw = image.raw_image_visible.astype(np.float32)
        black = float(min(image.black_level_per_channel))
        white = float(image.white_level)
        pattern = "".join("RGBG"[int(v)] for v in image.raw_pattern.flat)
        wb_rgbg = np.asarray(image.camera_whitebalance[:4], dtype=np.float32)
        color_matrix = camera_rgb_to_srgb_matrix(image.rgb_xyz_matrix)

    # rawpy metadata order: R,G,B,G. Packed Bayer order: R,G1,G2,B.
    wb = np.array([wb_rgbg[0], wb_rgbg[1], wb_rgbg[3], wb_rgbg[2]], dtype=np.float32)
    wb /= max(float(wb[1]), float(wb[2]), 1e-6)

    linear = normalize_raw(raw, black, white)
    display_rgb, over_white = render_packed_raw(
        pack_bayer(linear, pattern),
        pattern,
        tuple(float(v) for v in wb),
        exposure_gain=exposure_gain,
        tone_map=tone_map,
        white_point=white_point,
        color_matrix=color_matrix if use_color_matrix else None,
    )
    return ProcessedRaw(
        display_rgb=display_rgb,
        pattern=pattern,
        black_level=black,
        white_level=white,
        white_balance_gains=tuple(float(v) for v in wb),
        exposure_gain=float(exposure_gain),
        pre_tone_over_white_ratio=over_white,
    )
