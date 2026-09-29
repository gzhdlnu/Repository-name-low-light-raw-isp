#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 7 YCbCr/NV12 Audit
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase7-YUV-v0.1.0
# =============================================================================
"""Validate RGB -> BT.709 YCbCr -> RGB and NV12 output contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from isp.yuv import rgb_to_ycbcr, ycbcr_to_nv12, ycbcr_to_rgb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 YCbCr/NV12输出验收")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase7_deployment/yuv/phase7_yuv_report.json")
    args = parser.parse_args()
    rng = np.random.default_rng(20260929)
    rgb = rng.random((128, 192, 3), dtype=np.float32)
    results = {}
    for limited in (False, True):
        yuv = rgb_to_ycbcr(rgb, limited_range=limited)
        restored = ycbcr_to_rgb(yuv, limited_range=limited)
        y, uv = ycbcr_to_nv12(rgb, limited_range=limited)
        key = "limited" if limited else "full"
        results[key] = {
            "round_trip_max_abs_error": float(np.max(np.abs(rgb - restored))),
            "round_trip_mean_abs_error": float(np.mean(np.abs(rgb - restored))),
            "ycbcr_range": [float(yuv.min()), float(yuv.max())],
            "y_shape": list(y.shape),
            "uv_shape": list(uv.shape),
            "y_dtype": str(y.dtype),
            "uv_dtype": str(uv.dtype),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"version": "Phase7-YUV-v0.1.0", "results": results}, ensure_ascii=False, indent=2) + "\n")
    print("Phase 7 YCbCr/NV12 输出验收")
    for key, result in results.items():
        print(f"{key}: round-trip max/mean={result['round_trip_max_abs_error']:.7f}/{result['round_trip_mean_abs_error']:.7f}, Y={result['y_shape']}, UV={result['uv_shape']}")
    print(f"JSON报告: {args.output.resolve()}")
    if any(r["round_trip_max_abs_error"] > 2.0 / 255.0 for r in results.values()):
        raise SystemExit("YCbCr round-trip误差超过验收阈值")
    print("Phase 7 YCbCr/NV12输出验收: PASS")


if __name__ == "__main__":
    main()
