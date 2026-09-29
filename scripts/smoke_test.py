# =============================================================================
# Project: RAW ISP | Task: Traditional ISP Baseline Smoke Test
# Timestamp: 2026-09-18 | Author: Guozh | Version: Phase1-Baseline-v0.1.0
# =============================================================================
"""CPU smoke test for the traditional ISP baseline."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isp import (  # noqa: E402
    apply_color_matrix,
    demosaic_bilinear,
    linear_to_srgb,
    normalize_raw,
    pack_bayer,
    white_balance,
)


def main() -> None:
    h, w = 128, 160
    yy, xx = np.mgrid[:h, :w]
    raw = 512 + 12000 * (0.15 + 0.7 * xx / w + 0.15 * yy / h)
    linear = normalize_raw(raw, black_level=512, white_level=16383)
    packed = pack_bayer(linear, "RGGB")
    balanced = white_balance(packed, (2.0, 1.0, 1.0, 1.6))
    rgb = demosaic_bilinear(balanced, "RGGB")
    rgb = apply_color_matrix(rgb, np.eye(3, dtype=np.float32))
    srgb = linear_to_srgb(rgb)
    assert packed.shape == (h // 2, w // 2, 4)
    assert rgb.shape == (h, w, 3)
    assert np.isfinite(srgb).all() and 0.0 <= srgb.min() <= srgb.max() <= 1.0
    print(f"packed={packed.shape}, rgb={rgb.shape}, range=[{srgb.min():.4f}, {srgb.max():.4f}]")
    print("ISP baseline smoke test: PASS")


if __name__ == "__main__":
    main()
