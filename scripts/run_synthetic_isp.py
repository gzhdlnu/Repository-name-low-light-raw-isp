# =============================================================================
# Project: RAW ISP | Task: Synthetic Bayer ISP Validation
# Timestamp: 2026-09-18 | Author: Guozh | Version: Phase1-Baseline-v0.1.0
# =============================================================================
"""Generate a synthetic Bayer capture and save the baseline ISP output."""

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isp import apply_color_matrix, demosaic_bilinear, linear_to_srgb, normalize_raw, pack_bayer, white_balance  # noqa: E402


def main() -> None:
    height, width = 512, 768
    yy, xx = np.mgrid[:height, :width]
    # A smooth exposure gradient plus a bright patch approximates a simple
    # scene with both shadow and highlight regions.
    scene = 0.03 + 0.55 * (xx / width) + 0.18 * (yy / height)
    scene += 0.35 * (((xx - width * 0.72) ** 2 + (yy - height * 0.35) ** 2) < (height * 0.14) ** 2)
    raw = 512.0 + np.clip(scene, 0.0, 1.0) * (16383.0 - 512.0)
    raw += np.random.default_rng(7).normal(0.0, 18.0, raw.shape)

    linear = normalize_raw(raw, 512.0, 16383.0)
    packed = pack_bayer(linear, "RGGB")
    packed = white_balance(packed, (2.0, 1.0, 1.0, 1.55))
    rgb = demosaic_bilinear(packed, "RGGB")
    rgb = apply_color_matrix(rgb, np.eye(3, dtype=np.float32))
    srgb = (linear_to_srgb(rgb) * 255.0).round().astype(np.uint8)

    output = ROOT / "outputs" / "phase1_raw_baseline" / "synthetic_isp.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV writes BGR, while the ISP output is RGB.
    if not cv2.imwrite(str(output), cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {output}")
    print(f"wrote {output} ({width}x{height})")


if __name__ == "__main__":
    main()
